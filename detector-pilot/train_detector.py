"""Train the Detector pilot: DeBERTa-v3-base on true origin (0 human, 1 AI), fixed budget, no checkpoint selection.

Follows plan.json. Reuses the Scorer-1H model and tokenizer code so the only differences are the labels,
the data and the budget. Run with .venv-score/bin/python; writes detector-seed-<seed>/.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scorer' / 'deberta'))  # frozen model and tokenizer code
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import build

read = lambda p: json.loads(Path(p).read_text())
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def logits(model, **inputs):
    cls = model.encoder(**inputs).last_hidden_state[:, 0]
    return model.head(model.pooler(cls)).squeeze(-1)


def score(model, tokenizer, texts, device, max_length, batch_size=32):
    """Probability of AI origin for each text, in order."""
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length, padding=True, return_tensors='pt')
            out.extend(torch.sigmoid(logits(model, **{k: v.to(device) for k, v in batch.items()})).cpu().tolist())
    return np.array(out, dtype=np.float64)


def auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, help='Pipeline check only: tiny subset, written to this directory')
    args = parser.parse_args()
    plan = read(HERE / 'plan.json')
    settings = plan['training']
    seed = settings['seed']
    manifest = read(HERE / 'dataset/manifest.json')
    for name, expected in manifest['file_sha256'].items():   # the dataset must be exactly what the builder wrote
        assert sha_file(HERE / 'dataset' / name) == expected, f'Changed dataset file: {name}'
    output = args.smoke or HERE / f'detector-seed-{seed}'
    assert not output.exists(), 'Use a fresh run directory'
    output.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    assert torch.backends.mps.is_available(), 'MPS required'
    device = torch.device('mps')
    train, valid = read(HERE / 'dataset/train.json'), read(HERE / 'dataset/validation.json')
    epochs = settings['epochs']
    if args.smoke:
        train, valid, epochs = train[:96], valid[:64], 1
    model, tokenizer = build('deberta')   # gradient checkpointing on, as in the plan
    model.to(device)
    maxlen = settings['max_length']
    encoded = [{'inputs': tokenizer(r['text'], truncation=True, max_length=maxlen), 'label': float(r['label'])} for r in train]

    def collate(rows):
        values = tokenizer.pad([r['inputs'] for r in rows], return_tensors='pt')
        return {k: v.to(device) for k, v in values.items()}, torch.tensor([r['label'] for r in rows], device=device, dtype=torch.float32)

    loader = DataLoader(encoded, batch_size=settings['batch_size'], shuffle=True, drop_last=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    updates = epochs * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'], eps=settings['adam_eps'])
    warmup = round(settings['warmup_fraction'] * updates)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / warmup if s < warmup else max(0., (updates - s) / (updates - warmup)))
    config = {'seed': seed, 'encoder': plan['encoder'], 'architecture': plan['architecture'], 'max_length': maxlen, 'training': settings,
              'optimizer_updates': updates, 'warmup_updates': warmup, 'train_texts': len(train), 'validation_texts': len(valid),
              'dataset_manifest_sha256': sha_file(HERE / 'dataset/manifest.json'), 'plan_sha256': sha_file(HERE / 'plan.json')}
    (output / 'config.json').write_text(json.dumps(config, indent=1))
    tokenizer.save_pretrained(output / 'tokenizer')
    gold = np.array([r['label'] for r in valid])
    step = 0
    start = time.monotonic()

    def record(row):
        with (output / 'loss-history.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    for epoch in range(epochs):
        model.train()
        total = seen = 0
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(model, **values), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'])
            optimizer.step(); scheduler.step(); step += 1
            assert np.isfinite(loss.item()), f'Non-finite loss at step {step}'
            total += loss.item() * len(labels); seen += len(labels)
            if step % 25 == 0:
                record({'step': step, 'of': updates, 'epoch': epoch + seen / len(encoded), 'train_bce': total / seen, 'elapsed_seconds': time.monotonic() - start})
        pred = score(model, tokenizer, [r['text'] for r in valid], device, maxlen)
        record({'step': step, 'epoch': epoch + 1, 'train_bce_epoch': total / seen, 'validation_auc': auc(gold, pred),
                'validation_bce': float(-np.mean(gold * np.log(pred + 1e-9) + (1 - gold) * np.log(1 - pred + 1e-9))), 'diagnostic_only': True})
    assert step == updates
    torch.save({'model': model.state_dict(), 'step': step, 'config': config, 'training_complete': True}, output / 'model.pt')
    with (output / 'validation-scores.jsonl').open('w') as handle:
        for row, p in zip(valid, pred):
            handle.write(json.dumps({'text_sha256': row['text_sha256'], 'family': row['family'], 'label': row['label'], 'type': row['type'], 'score': float(p)}) + '\n')
    summary = {'seed': seed, 'completed_updates': step, 'elapsed_seconds': time.monotonic() - start, 'final_validation_auc': auc(gold, pred),
               'model_sha256': sha_file(output / 'model.pt'), 'validation_used_for_selection': False, 'production_changed': False}
    (output / 'training-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
