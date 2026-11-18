"""Train one fresh encoder arm (all weights) for the frozen fixed budget; no checkpoint selection."""
import argparse
import json
import random
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from common import P, read, write, sha, build, splits, predict, metrics, ROW_METADATA

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--encoder', choices=['bert', 'deberta'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke', type=Path, help='Pipeline check only: tiny subset, written to this directory')
    args = parser.parse_args()
    plan = read(P / 'plan.json'); settings = plan['training']; assert args.seed in plan['seeds']
    for path, expected in plan['protected_files'].items():
        if not path.endswith('humanizer-master.json'):
            assert sha(path) == expected, f'Changed protected input: {path}'
    output = args.smoke or P / f'{args.encoder}-seed-{args.seed}'; assert not output.exists(), 'Use a fresh run directory'; output.mkdir()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(4)
    assert torch.backends.mps.is_available(), 'MPS required'; device = torch.device('mps')
    bundle = splits(); train, valid = bundle.splits['train'], bundle.splits['validation']
    if args.smoke:
        train, valid = train[:80], valid[:64]; settings = {**settings, 'epochs': 2, 'optimizer_updates': 20}
    model, tokenizer = build(args.encoder); model.to(device)
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    maxlen = settings['max_length']
    encoded = [{'inputs': tokenizer(r['text'], truncation=True, max_length=maxlen), 'label': r['score_pct'] / 100} for r in train]
    def collate(rows):
        values = tokenizer.pad([r['inputs'] for r in rows], return_tensors='pt')
        return {k: v.to(device) for k, v in values.items()}, torch.tensor([r['label'] for r in rows], device=device, dtype=torch.float32)
    loader = DataLoader(encoded, batch_size=settings['batch_size'], shuffle=True, drop_last=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    updates = settings['optimizer_updates']; assert updates == settings['epochs'] * len(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'], eps=settings['adam_eps'])
    warmup = round(settings['warmup_fraction'] * updates)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / warmup if s < warmup else max(0., (updates - s) / (updates - warmup)))
    config = {'encoder': args.encoder, 'seed': args.seed, 'model': plan['encoders'][args.encoder]['model'],
              'revision': plan['encoders'][args.encoder]['revision'], 'architecture': plan['architectures'][args.encoder],
              'max_length': maxlen, 'training': settings, 'warmup_updates': warmup, 'device': str(device),
              'trainable_parameters': sum(p.numel() for p in model.parameters()), 'dataset_provenance': bundle.metadata}
    write(output / 'config.json', config); tokenizer.save_pretrained(output / 'tokenizer')
    gold = np.array([r['score_pct'] for r in valid]); step = 0; start = time.monotonic()
    def record(row):
        with (output / 'loss-history.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    for epoch in range(settings['epochs']):
        model.train(); total = 0.; seen = 0
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.mse_loss(model(**values), labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'])
            optimizer.step(); scheduler.step(); step += 1; total += loss.item() * len(labels); seen += len(labels)
            assert np.isfinite(loss.item()), f'Non-finite loss at step {step}'
            if step % 50 == 0:
                record({'step': step, 'epoch': epoch + seen / len(train), 'lr': scheduler.get_last_lr()[0], 'train_mse': total / seen,
                        'elapsed_seconds': time.monotonic() - start})
        pred = predict(model, tokenizer, valid, device, maxlen)
        record({'step': step, 'epoch': epoch + 1, 'train_mse_epoch': total / seen, 'validation_mse': float(np.mean(((pred - gold) / 100)**2)),
                **{'validation_' + k: v for k, v in metrics(pred, gold).items()}, 'diagnostic_only': True})
    assert step == updates
    torch.save({'model': model.state_dict(), 'step': step, 'config': config, 'training_complete': True,
                'epoch_complete': True, 'optimizer_saved': False}, output / 'model.pt')
    provenance = {r['text_sha256']: r for r in read(ROW_METADATA)['validation']}
    with (output / 'validation-predictions.jsonl').open('w') as f:
        for r, p in zip(valid, pred):
            m = provenance[r['text_sha256']]
            f.write(json.dumps({'text_sha256': r['text_sha256'], 'group_id': r['group_id'], 'humanization_source': m['humanization_source'],
                                'text_role': m['text_role'], 'target_score_pct': r['score_pct'], 'predicted_score_pct': float(p),
                                'error_pp': float(p - r['score_pct'])}) + '\n')
    summary = {'encoder': args.encoder, 'seed': args.seed, 'completed_updates': step, 'example_presentations': step * settings['batch_size'],
               'elapsed_seconds': time.monotonic() - start, 'final_validation': metrics(pred, gold), 'model_sha256': sha(output / 'model.pt'),
               'validation_used_for_selection': False, 'test_evaluated': False, 'production_changed': False}
    write(output / 'training-summary.json', summary); print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
