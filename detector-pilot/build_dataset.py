"""Build the Detector pilot dataset from true origin, following plan.json (written before any result).

Reads the master, the Wikipedia human paragraphs, the matched AI paragraphs and their Rewriter versions, and writes
dataset/{train,validation,detector-test}.json, dataset/eval-humanized.json and dataset/manifest.json.
Refuses to overwrite an existing dataset folder. Run with the repo's system python3 (standard library only).
"""
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DATA_ROOT', HERE.parent / 'data')).resolve()  # private project data, not in this repo
OUT = HERE / 'dataset'
HOLDOUT = ROOT / 'datasets/joint-holdout-20261106'
EXPORT = ROOT / 'datasets/model2-rewriter-outputs-20261116'
BENCH = ROOT / 'dataset-runs/rewriter-vs-humanizer-b-20261111'
MIN_WORDS = 40
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: ' '.join(re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", t.lower()))   # same as the holdout's verify.py
read = lambda p: json.loads(Path(p).read_text())
jsonl = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
bucket = lambda key, n: int(sha(key)[:8], 16) % n


def master_type(record):
    """True-origin type of a master source, or None when there is no origin evidence."""
    p = record.get('provenance') or {}
    if p.get('source_type') == 'project_gutenberg_public_domain':
        return 'human_gutenberg'
    if p.get('source_type') in ('ai_written_chapter_user_declared', 'ai_written_book_user_declared'):
        return 'ai_raw_chapter'
    if p.get('source_origin') == 'agent_authored_textbook':
        return 'ai_raw_agent'
    if p.get('source_origin') in ('unadapted_base_qwen3', 'unadapted_base_qwen3_academic_register'):
        return 'ai_rewrite_base_qwen3'
    if p.get('source_origin') == 'trained_qwen3_humanizer':
        return 'ai_humanized_rewriter'
    if p.get('collection_kind') == 'detector_a_only_import':
        return 'ai_raw_qwen3'
    return None


def main():
    assert not OUT.exists(), 'dataset/ exists; move it aside to rebuild'
    plan = read(HERE / 'plan.json')
    caps = plan['splits']['train_caps_by_type_lowest_sha256_first']
    groups = read(EXPORT / 'split-groups.json')
    pairs = read(BENCH / 'pairs.json')
    picks = read(BENCH / 'picks.json')
    eval_groups = {p['group_id'] for p in pairs}
    eval_texts = {norm(t) for p in pairs for t in (p['original'], p['humanizer_b'], p['rewriter'])}
    eval_texts |= {norm(c['text']) for p in picks for c in p['candidates'] if isinstance(c, dict) and c.get('text')}
    eval_texts |= {norm(p['picked_text']) for p in picks}
    frozen_test = {r['text_sha256'] for r in read(EXPORT / 'model2-detector-a/test.json')}
    held = {norm(r['text']) for r in read(HOLDOUT / 'holdout.json')}
    dropped = collections.Counter()
    rows = []

    def add(text, label, kind, family, split, **extra):
        text = text.strip()
        if len(text.split()) < MIN_WORDS:
            dropped['under_40_words'] += 1
            return
        key = norm(text)
        assert key not in held, 'joint-holdout text reached the dataset'
        if key in eval_texts:
            dropped['humanized_eval_text'] += 1
            return
        rows.append({'text': text, 'text_sha256': sha(text), 'label': label, 'type': kind, 'family': family, 'split': split, **extra})

    # Master: sources, then their Humanizer B outputs.
    for record in read(ROOT / 'humanizer-master.json'):
        kind = master_type(record)
        if kind is None:
            dropped['master_source_without_origin_evidence'] += 1
            continue
        group = groups[record['source_id']]
        if group in eval_groups:
            dropped['humanized_eval_family_record'] += 1
            continue
        split = 'validation' if bucket(group, 10) == 1 else 'train'
        if record['source_id'] in frozen_test:
            dropped['model2_frozen_test_text'] += 1
        else:
            add(record['source']['text'], 0 if kind == 'human_gutenberg' else 1, kind, group, split, source_id=record['source_id'])
        out_kind = 'ai_humanized_humanizer_b_of_human' if kind == 'human_gutenberg' else 'ai_humanized_humanizer_b'
        slots = [slot for views in record['outputs'].values() for slot in views.values()] + list(record.get('unassigned_outputs') or [])
        for slot in slots:
            if slot.get('text') and slot.get('humanization_source') == 'Humanizer B' and slot['text_sha256'] not in frozen_test:
                add(slot['text'], 1, out_kind, group, split, source_id=record['source_id'])

    # Wikipedia: human paragraphs, matched AI paragraphs, Rewriter versions; one family per article.
    def wiki_split(pageid):
        b = bucket(str(pageid), 20)
        return 'detector-test' if b <= 1 else 'validation' if b <= 4 else 'train'

    for area, items in read(HERE / 'wiki-human.json').items():
        for item in items:
            for text in item.get('paragraphs') or []:
                add(text, 0, 'human_wiki', f'wiki-{item["pageid"]}', wiki_split(item['pageid']), title=item['title'], area=area, revid=item['revid'])
    for row in jsonl(HERE / 'ai-matched.jsonl'):
        add(row['text'], 1, 'ai_raw_wiki', f'wiki-{row["pageid"]}', wiki_split(row['pageid']), title=row['title'], area=row['area'], generator=row['generator'])
    for row in jsonl(HERE / 'ai-matched-rewritten.jsonl'):
        if row['finish_reason'] == 'stop':
            add(row['text'], 1, 'ai_humanized_rewriter_wiki', f'wiki-{row["pageid"]}', wiki_split(row['pageid']), title=row['title'], area=row['area'],
                generator=row['source_generator'])
        else:
            dropped['rewriter_did_not_stop'] += 1

    # Normalized duplicates: keep one; drop a text entirely if it carries both labels or sits in more than one split.
    by_text = collections.defaultdict(list)
    for row in rows:
        by_text[norm(row['text'])].append(row)
    unique = []
    for same in by_text.values():
        if len({r['label'] for r in same}) > 1:
            dropped['both_labels'] += len(same)
        elif len({r['split'] for r in same}) > 1:
            dropped['same_text_in_two_splits'] += len(same)
        else:
            unique.append(min(same, key=lambda r: r['text_sha256']))
            dropped['duplicate'] += len(same) - 1

    # A family lives in exactly one split.
    family_split = {}
    for row in unique:
        assert family_split.setdefault(row['family'], row['split']) == row['split'], row['family']

    splits = collections.defaultdict(list)
    for row in sorted(unique, key=lambda r: r['text_sha256']):
        splits[row['split']].append(row)
    kept, per_type = [], collections.Counter()
    for row in splits['train']:
        cap = caps.get('human' if row['label'] == 0 else row['type'], 'all')
        if cap == 'all' or per_type[row['type']] < cap:
            per_type[row['type']] += 1
            kept.append(row)
        else:
            dropped['over_train_cap:' + row['type']] += 1
    splits['train'] = kept

    OUT.mkdir()
    counts = {}
    for name, items in splits.items():
        (OUT / f'{name}.json').write_text(json.dumps(items, ensure_ascii=False, indent=1))
        counts[name] = {'texts': len(items), 'families': len({r['family'] for r in items}),
                        'human': sum(r['label'] == 0 for r in items), 'ai': sum(r['label'] == 1 for r in items),
                        'by_type': dict(sorted(collections.Counter(r['type'] for r in items).items()))}

    # Humanized evaluation set: the 30 benchmark families with Detector A's own scores.
    analysis = {a['source_id']: a for a in read(BENCH / 'analysis.json')}
    humanized = []
    for p in pairs:
        a = analysis[p['source_id']]
        for kind, text, score in (('ai_original', p['original'], a['original_jd']), ('humanizer_b_rewrite', p['humanizer_b'], a['humanizer_b_jd']),
                                  ('rewriter_greedy', p['rewriter'], a['rewriter_jd'])):
            humanized.append({'source_id': p['source_id'], 'kind': kind, 'text': text, 'text_sha256': sha(text), 'label': 1, 'detector_a_score_pct': score})
    for p in picks:
        if p['picked_k'] != 0:
            humanized.append({'source_id': p['source_id'], 'kind': 'rewriter_best_of_4_pick', 'text': p['picked_text'], 'text_sha256': sha(p['picked_text']),
                              'label': 1, 'detector_a_score_pct': None})
    (OUT / 'eval-humanized.json').write_text(json.dumps(humanized, ensure_ascii=False, indent=1))

    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.glob('*.json'))}
    manifest = {'built_from_plan_sha256': hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest(), 'counts': counts,
                'dropped': dict(sorted(dropped.items())), 'humanized_eval': dict(collections.Counter(h['kind'] for h in humanized)),
                'file_sha256': files, 'joint_holdout_sha256': read(HOLDOUT / 'manifest.json')['holdout_sha256']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    check = subprocess.run([sys.executable, str(HOLDOUT / 'verify.py'), str(OUT)], capture_output=True, text=True)
    print('holdout verify:', check.stdout.strip())
    assert check.returncode == 0, 'joint-holdout overlap'


if __name__ == '__main__':
    main()
