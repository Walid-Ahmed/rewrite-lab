"""Evaluate the Detector pilot in the order fixed by plan.json.

    .venv-score/bin/python evaluate.py            # steps 1-4: validation threshold, detector-test, humanized set, arXiv
    .venv-score/bin/python evaluate.py --holdout  # step 5: joint holdout, once, after results.json exists

The threshold comes from validation human texts only and is never changed afterwards.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DATA_ROOT', HERE.parent / 'data')).resolve()  # private project data, not in this repo
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from train_detector import auc, build, read, score, sha_file

HOLDOUT = ROOT / 'datasets/joint-holdout-20261106'


def load(run):
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
    assert saved['training_complete']
    model, tokenizer = build('deberta')
    model.load_state_dict(saved['model'])
    return model.to('mps'), tokenizer, saved['config']['max_length']


def rate(flags):
    flags = list(flags)
    return {'flagged': int(sum(flags)), 'of': len(flags), 'rate': round(float(np.mean(flags)), 4) if flags else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true')
    args = parser.parse_args()
    plan = read(HERE / 'plan.json')
    run = HERE / f'detector-seed-{plan["training"]["seed"]}'
    results_path = HERE / 'results.json'
    model, tokenizer, maxlen = load(run)
    scores_of = lambda rows: score(model, tokenizer, [r['text'] for r in rows], torch.device('mps'), maxlen)

    if args.holdout:
        assert results_path.exists(), 'Run steps 1-4 first'
        out = HERE / 'results-holdout.json'
        assert not out.exists(), 'The joint holdout has already been scored once for this model'
        t = read(results_path)['threshold']['value']
        rows = read(HOLDOUT / 'holdout.json')
        jd = {r['holdout_id']: r['detector_a']['score_pct'] for r in read(HOLDOUT / 'detector-a-scores.json')['rows']}
        s = scores_of(rows)
        y = np.array([0 if r['origin'].startswith('human') else 1 for r in rows])
        j = np.array([jd[r['holdout_id']] for r in rows], dtype=float)
        result = {'model_sha256': sha_file(run / 'model.pt'), 'threshold': t,
                  'detector': {'auc': round(auc(y, s), 4), 'human': rate(s[y == 0] > t), 'ai': rate(s[y == 1] > t)},
                  'detector_a_score_above_0': {'auc': round(auc(y, j), 4), 'human': rate(j[y == 0] > 0), 'ai': rate(j[y == 1] > 0)},
                  'ai_missed_by_detector_a_caught_by_detector': int(((j == 0) & (y == 1) & (s > t)).sum()),
                  'by_domain_human_flagged': {d: int(sum(1 for r, v in zip(rows, s) if r['domain'] == d and r['origin'].startswith('human') and v > t))
                                              for d in sorted({r['domain'] for r in rows})}}
        out.write_text(json.dumps(result, indent=1))
        with (HERE / 'scores-holdout.jsonl').open('w') as handle:
            for r, v in zip(rows, s):
                handle.write(json.dumps({'holdout_id': r['holdout_id'], 'origin': r['origin'], 'score': float(v)}) + '\n')
        print(json.dumps(result, indent=1))
        return

    assert not results_path.exists(), 'results.json exists; steps 1-4 were already run'
    # 1. Validation: threshold from human texts only.
    valid = [json.loads(line) for line in (run / 'validation-scores.jsonl').read_text().splitlines()]
    vs = np.array([v['score'] for v in valid]); vy = np.array([v['label'] for v in valid])
    human = np.sort(vs[vy == 0])[::-1]
    allowed = int(0.05 * len(human))
    t = float(human[allowed])            # at most `allowed` validation human texts score above t
    by_type = collections.defaultdict(list)
    for v in valid:
        by_type[v['type']].append(v['score'] > t)
    results = {'model_sha256': sha_file(run / 'model.pt'),
               'threshold': {'value': t, 'rule': plan['threshold'], 'validation_human_texts': int(len(human)), 'allowed_above': allowed},
               'validation': {'auc': round(auc(vy, vs), 4), 'flagged_by_type': {k: rate(v) for k, v in sorted(by_type.items())}}}

    # 2. Held-out Wikipedia articles.
    test = read(HERE / 'dataset/detector-test.json')
    ts = scores_of(test); ty = np.array([r['label'] for r in test])
    by_type = collections.defaultdict(list); by_gen = collections.defaultdict(list)
    for r, v in zip(test, ts):
        by_type[r['type']].append(v > t)
        if r.get('generator'):
            by_gen[r['type'] + ':' + r['generator']].append(v > t)
    results['detector_test'] = {'auc': round(auc(ty, ts), 4), 'flagged_by_type': {k: rate(v) for k, v in sorted(by_type.items())},
                                'flagged_by_generator': {k: rate(v) for k, v in sorted(by_gen.items())}}

    # 3. Humanized set, against Detector A's own flags.
    humanized = read(HERE / 'dataset/eval-humanized.json')
    hs = scores_of(humanized)
    table = {}
    for kind in ('ai_original', 'humanizer_b_rewrite', 'rewriter_greedy', 'rewriter_best_of_4_pick'):
        idx = [i for i, h in enumerate(humanized) if h['kind'] == kind]
        jd = [humanized[i]['detector_a_score_pct'] for i in idx]
        table[kind] = {'detector': rate(hs[i] > t for i in idx),
                       'detector_a_score_above_0': rate(x > 0 for x in jd) if all(x is not None for x in jd) else None}
    results['humanized_set'] = table

    # 4. Unseen human source.
    arxiv = read(HERE / 'arxiv-human.json')
    xs = scores_of(arxiv)
    by_cat = collections.defaultdict(list)
    for r, v in zip(arxiv, xs):
        by_cat[r['category']].append(v > t)
    results['arxiv_human'] = {'all': rate(xs > t), 'by_category': {k: rate(v) for k, v in sorted(by_cat.items())},
                              'median_score': round(float(np.median(xs)), 4)}

    results_path.write_text(json.dumps(results, indent=1))
    for name, rows, values in (('detector-test', test, ts), ('humanized', humanized, hs), ('arxiv', arxiv, xs)):
        with (HERE / f'scores-{name}.jsonl').open('w') as handle:
            for r, v in zip(rows, values):
                handle.write(json.dumps({'text_sha256': r.get('text_sha256'), 'kind': r.get('type') or r.get('kind') or r.get('category'), 'score': float(v)}) + '\n')
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
