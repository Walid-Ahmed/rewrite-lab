"""Score the existing checkpoint, then compare every finished fresh-encoder run against it on the same validation families."""
import argparse
import csv
import json
import numpy as np
import torch
from common import P, H, read, write, sha, splits, predict, metrics, gate, clustered_contrast, load_existing, ROW_METADATA, ENCODERS

def existing_predictions(plan, valid):
    path = P / 'existing-validation-predictions.jsonl'
    if path.exists():
        return np.array([json.loads(x)['predicted_score_pct'] for x in path.read_text().splitlines()])
    assert sha(plan['existing_checkpoint']) == plan['existing_checkpoint_sha256']
    model, tokenizer = load_existing(); torch.set_num_threads(4)
    pred = predict(model, tokenizer, valid, torch.device('cpu'), 512)
    cached = [json.loads(x) for x in (H / 'model2-retrain-balanced-20261029/model-expanded/validation-step-000800.jsonl').read_text().splitlines()]
    assert [c['text_sha256'] for c in cached] == [r['text_sha256'] for r in valid]
    drift = float(np.max(abs(pred - np.array([c['predicted_score_pct'] for c in cached]))))
    assert drift < .05, f'Existing checkpoint does not reproduce its cached predictions ({drift:.4f} pp)'
    provenance = {r['text_sha256']: r for r in read(ROW_METADATA)['validation']}
    with path.open('w') as f:
        for r, p in zip(valid, pred):
            m = provenance[r['text_sha256']]
            f.write(json.dumps({'text_sha256': r['text_sha256'], 'group_id': r['group_id'], 'humanization_source': m['humanization_source'],
                                'text_role': m['text_role'], 'target_score_pct': r['score_pct'], 'predicted_score_pct': float(p),
                                'error_pp': float(p - r['score_pct'])}) + '\n')
    write(P / 'existing-reproduction.json', {'max_abs_difference_vs_cached_pp': drift, 'device': 'cpu'})
    return pred

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--existing-only', action='store_true'); args = parser.parse_args()
    plan = read(P / 'plan.json'); bundle = splits(); valid = bundle.splits['validation']
    gold = np.array([r['score_pct'] for r in valid]); groups = [r['group_id'] for r in valid]
    existing = existing_predictions(plan, valid); base = metrics(existing, gold)
    if args.existing_only:
        print(json.dumps(base)); return
    boot = plan['bootstrap']; runs = {}; preds = {}
    for name in plan['run_order']:
        encoder, seed = name.rsplit('-', 1); folder = P / f'{encoder}-seed-{seed}'
        if not (folder / 'training-summary.json').exists():
            continue
        rows = [json.loads(x) for x in (folder / 'validation-predictions.jsonl').read_text().splitlines()]
        assert [r['text_sha256'] for r in rows] == [r['text_sha256'] for r in valid]
        p = np.array([r['predicted_score_pct'] for r in rows]); preds[name] = p
        runs[name] = {'encoder': encoder, 'seed': int(seed), 'metrics': metrics(p, gold), 'gate_vs_existing': gate(metrics(p, gold), base),
                      'contrast_vs_existing': clustered_contrast(p, existing, gold, groups, boot['seed'], boot['replicates']),
                      'elapsed_seconds': read(folder / 'training-summary.json')['elapsed_seconds']}
    encoders = {}
    for encoder in ENCODERS:
        names = [n for n in runs if runs[n]['encoder'] == encoder]
        if not names:
            continue
        mean = {k: float(np.mean([runs[n]['metrics'][k] for n in names])) for k in base}
        spread = {k: float(np.std([runs[n]['metrics'][k] for n in names], ddof=1)) if len(names) > 1 else None for k in base}
        mean_gate = gate(mean, base)
        complete = len(names) == len(plan['seeds'])
        encoders[encoder] = {'seeds_completed': len(names), 'seed_mean_metrics': mean, 'seed_sd_metrics': spread,
                             'gate_seed_mean': mean_gate, 'every_seed_passed': all(runs[n]['gate_vs_existing']['passed'] for n in names),
                             'accepted': complete and mean_gate['passed'] and all(runs[n]['gate_vs_existing']['passed'] for n in names),
                             'decision_final': complete}
    head_to_head = {}
    for seed in plan['seeds']:
        a, b = f'deberta-{seed}', f'bert-{seed}'
        if a in preds and b in preds:
            head_to_head[f'deberta_minus_bert_seed_{seed}'] = clustered_contrast(preds[a], preds[b], gold, groups, boot['seed'], boot['replicates'])
    summary = {'validation_texts': len(valid), 'validation_families': len(set(groups)), 'existing_checkpoint': plan['existing_checkpoint'],
               'existing_metrics': base, 'runs': runs, 'encoders': encoders, 'deberta_vs_bert': head_to_head,
               'acceptance': plan['acceptance'], 'test_evaluated': False, 'production_changed': False}
    write(P / 'summary.json', summary)
    with (P / 'comparison.csv').open('w', newline='') as f:
        w = csv.writer(f); keys = list(base)
        w.writerow(['model', *keys, 'gate_passed'])
        w.writerow(['existing', *[f'{base[k]:.3f}' for k in keys], ''])
        for n, r in runs.items():
            w.writerow([n, *[f"{r['metrics'][k]:.3f}" for k in keys], r['gate_vs_existing']['passed']])
        for e, r in encoders.items():
            w.writerow([f'{e}-seed-mean', *[f"{r['seed_mean_metrics'][k]:.3f}" for k in keys], r['accepted']])
    print(json.dumps({'existing': base, **{e: {'mean': r['seed_mean_metrics'], 'accepted': r['accepted'], 'seeds': r['seeds_completed']}
                                           for e, r in encoders.items()}}, indent=1))

if __name__ == '__main__':
    main()
