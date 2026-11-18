"""Score Model 2 predictions against the joint holdout's official Detector A labels.

The holdout (datasets/joint-holdout-20261106) is the final test for Model 2: use it once per finished
candidate, after model selection, never for tuning. Results are reported per origin as well as pooled,
because the AI-written half is shorter and a model could separate the halves by length alone.

Either pass predictions made elsewhere:
  python evaluate_joint_holdout.py --predictions preds.jsonl
    (one JSON object per line: holdout_id or text_sha256, plus predicted_score_pct)
or let this script run a repo BERT checkpoint (train_score_bert.py / predict_score_bert.py format):
  .venv-score/bin/python evaluate_joint_holdout.py --bert-checkpoint path/to/checkpoint.pt --write-predictions out.jsonl
Output files hold IDs, hashes and numbers only, never holdout text.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

HOLDOUT = Path(os.environ.get('DATA_ROOT', 'data')).resolve() / 'datasets/joint-holdout-20261106'  # holdout data is not in this repo


def load_holdout():
    """Holdout rows with their Detector A labels; refuses if the frozen file or the score file has drifted."""
    manifest = json.loads((HOLDOUT / 'manifest.json').read_text())
    raw = (HOLDOUT / 'holdout.json').read_text()
    if hashlib.sha256(raw.encode()).hexdigest() != manifest['holdout_sha256']:
        raise SystemExit('holdout.json no longer matches its manifest')
    scores = json.loads((HOLDOUT / 'detector-a-scores.json').read_text())
    if scores['holdout_sha256'] != manifest['holdout_sha256']:
        raise SystemExit('detector-a-scores.json was made for a different holdout version')
    labels = {r['holdout_id']: r for r in scores['rows']}
    rows = []
    for r in json.loads(raw):
        s = labels[r['holdout_id']]
        if s['text_sha256'] != r['text_sha256'] or s['detector_a']['status'] != 'scored':
            raise SystemExit(f"{r['holdout_id']}: label does not match its text")
        rows.append({**r, 'label_pct': s['detector_a']['score_pct']})
    return rows


def auc(pos, neg):
    """Probability a 100%-labelled text is ranked above a 0%-labelled one; ties count half."""
    if not len(pos) or not len(neg):
        return None
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + .5 * (pos == neg).mean())


def summarize(pred, gold):
    pred, gold = np.asarray(pred, float), np.asarray(gold, float); e = pred - gold
    band = lambda m: float(abs(e[m]).mean()) if m.any() else None
    return {'n': int(len(gold)), 'mae_pp': float(abs(e).mean()), 'rmse_pp': float(np.sqrt((e**2).mean())),
            'mean_label_pct': float(gold.mean()), 'mean_predicted_pct': float(pred.mean()),
            'auc_0_vs_100': auc(pred[gold == 100], pred[gold == 0]),
            'side_agreement': float(((pred >= 50) == (gold >= 50)).mean()),
            'confident_misses': int((((gold == 100) & (pred < 20)) | ((gold == 0) & (pred > 80))).sum()),
            'hundred_mae_pp': band(gold == 100), 'zero_mae_pp': band(gold == 0),
            'intermediate_mae_pp': band((gold > 0) & (gold < 100))}


def evaluate(predictions, rows=None):
    """predictions maps holdout_id or text_sha256 to a 0-100 score; every holdout row must be covered."""
    rows = rows or load_holdout()
    pick = lambda r: predictions.get(r['holdout_id'], predictions.get(r['text_sha256']))
    missing = [r['holdout_id'] for r in rows if pick(r) is None]
    if missing:
        raise SystemExit(f'No prediction for {len(missing)} holdout rows, e.g. {missing[:3]}')
    out = {}
    for origin in sorted({r['origin'] for r in rows}) + ['all']:
        sel = [r for r in rows if origin in ('all', r['origin'])]
        out[origin] = summarize([pick(r) for r in sel], [r['label_pct'] for r in sel])
    return out


def bert_predictions(checkpoint, rows, batch_size=32):
    import torch
    from transformers import AutoTokenizer
    from train_score_bert import ScoreBert
    state = torch.load(checkpoint, map_location='cpu', weights_only=False); c = state['config']
    model = ScoreBert(c['model']); model.load_state_dict(state['model']); model.eval()
    tokenizer_dir = Path(checkpoint).parent / 'tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.is_dir() else c['model'])
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model.to(device); out = {}
    with torch.inference_mode():
        for batch in itertools.batched(rows, batch_size):
            x = tokenizer([r['text'] for r in batch], truncation=True, max_length=c['max_length'], padding=True, return_tensors='pt')
            scores = (model(**{k: v.to(device) for k, v in x.items()}) * 100).cpu().tolist()
            out.update({r['holdout_id']: float(s) for r, s in zip(batch, scores)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--predictions', type=Path); src.add_argument('--bert-checkpoint', type=Path)
    ap.add_argument('--write-predictions', type=Path, help='Save the predictions (IDs, hashes, scores only)')
    a = ap.parse_args(); rows = load_holdout()
    if a.predictions:
        preds = {}
        for line in a.predictions.read_text().splitlines():
            if line.strip():
                r = json.loads(line); preds[r.get('holdout_id') or r['text_sha256']] = float(r['predicted_score_pct'])
    else:
        preds = bert_predictions(a.bert_checkpoint, rows)
    if a.write_predictions:
        a.write_predictions.write_text(''.join(json.dumps({'holdout_id': r['holdout_id'], 'text_sha256': r['text_sha256'],
                                                           'predicted_score_pct': preds.get(r['holdout_id'], preds.get(r['text_sha256']))}) + '\n' for r in rows))
    print(json.dumps(evaluate(preds, rows), indent=1))


if __name__ == '__main__':
    main()
