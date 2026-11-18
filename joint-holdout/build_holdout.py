"""Freeze the joint holdout: select texts, screen them against every known project text, write files."""
import collections as C
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(os.environ.get('DATA_ROOT', 'data')).resolve()  # private project data, not in this repo
H = ROOT / 'model-runs'
S = ROOT / 'work'  # candidate files from wiki_fetch.py and the AI generator
OUT = ROOT / 'datasets/joint-holdout-20261106'
VERSION = 'joint-holdout-20261106'
digest = lambda t: hashlib.sha256(t.encode()).hexdigest()
words = lambda t: re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", t.lower())
norm = lambda t: ' '.join(words(t))

DOMAINS = ['survey_methods', 'operations', 'labor_economics', 'public_economics', 'agriculture',
           'public_health', 'law_governance', 'history', 'psychology_education', 'computing']
CLAUDE_N = {'survey_methods': 8, 'operations': 7, 'labor_economics': 8, 'public_economics': 7, 'agriculture': 8,
            'public_health': 7, 'law_governance': 8, 'history': 7, 'psychology_education': 8, 'computing': 7}
PROMPT = ('One self-contained academic paragraph of about 70-130 words on the named topic, with at least one '
          'qualification or condition the rewrite must keep.')


# Fixed before looking at any score: drop paragraphs that lean on earlier text, open with a quotation, or carry markup residue.
NOT_STANDALONE = re.compile(r'^(?:["“]|(?:Secondly|Another|Alternatively|Nevertheless|Despite|Shortly|Congruent|Although recent|Sometimes|Historically,)\b)|Template:')


def corpus():
    """Every text the project could have trained or evaluated on."""
    texts = {}
    def add(t, where):
        if isinstance(t, str) and t.strip():
            texts.setdefault(digest(t), (t, where))
    for r in json.load(open(ROOT / 'humanizer-master.json')):
        add(r['source']['text'], 'master:source')
        for views in r['outputs'].values():
            for o in views.values():
                add(o.get('text'), 'master:output')
        for o in r['unassigned_outputs']:
            add(o.get('text'), 'master:unassigned')
    for b in json.load(open(H / 'validation-experiment-20261026/benchmark.json')):
        add(b['text'], '36-benchmark')
    for g in ('baseline-generations.json', 'candidate-generations.json'):
        for r in json.load(open(H / 'humanizer-filtered-experiment-20261101' / g))['rows']:
            add(r['after'], 'benchmark-output')
    for f in ROOT.glob('datasets/*/model2-detector-a/*.json'):
        if f.name in ('train.json', 'validation.json', 'test.json'):
            for r in json.load(open(f)):
                add(r.get('text'), 'export:model2')
    for f in (H / 'model2-retrain-balanced-20261029/dataset-expanded/model2-detector-a').glob('*.json'):
        if f.name in ('train.json', 'validation.json', 'test.json'):
            for r in json.load(open(f)):
                add(r.get('text'), 'export:model2-oct29')
    for f in ROOT.glob('datasets/*/model1-humanizer/*.json'):
        if f.name in ('train.json', 'validation.json', 'test.json'):
            for r in json.load(open(f)):
                add(r.get('before'), 'export:model1')
                add(r.get('after'), 'export:model1')
    return texts


class Screen:
    def __init__(self, texts):
        self.items = [(t, w) for t, w in texts.values()]
        self.norm = {norm(t) for t, _ in self.items}
        self.tok = [words(t) for t, _ in self.items]
        self.five = C.defaultdict(list)
        self.ten = set()
        for i, w in enumerate(self.tok):
            for j in range(len(w) - 4):
                self.five[' '.join(w[j:j + 5])].append(i)
            for j in range(len(w) - 9):
                self.ten.add(' '.join(w[j:j + 10]))
        self.df = C.Counter(x for w in self.tok for x in set(w))
        self.index = C.defaultdict(list)
        for i, w in enumerate(self.tok):
            for x in set(w):
                if self.df[x] < 400:
                    self.index[x].append(i)

    def check(self, text):
        w = words(text)
        result = {'exact_normalized': norm(text) in self.norm, 'shared_10_word_passage': False,
                  'max_word_sequence_similarity': 0.0, 'closest_corpus_location': None}
        result['shared_10_word_passage'] = any(' '.join(w[j:j + 10]) in self.ten for j in range(len(w) - 9))
        hits = C.Counter()
        for j in range(len(w) - 4):
            hits.update(set(self.five.get(' '.join(w[j:j + 5]), ())))
        for x in set(w):
            hits.update({i: 1 / (1 + self.df[x]) * 50 for i in self.index.get(x, ())})
        for i, _ in hits.most_common(25):
            sim = SequenceMatcher(None, w, self.tok[i], autojunk=False).ratio()
            if sim > result['max_word_sequence_similarity']:
                result['max_word_sequence_similarity'] = round(sim, 4)
                result['closest_corpus_location'] = self.items[i][1]
        result['passed'] = not result['exact_normalized'] and not result['shared_10_word_passage'] and result['max_word_sequence_similarity'] < 0.65
        return result


def main():
    wiki = json.load(open(S / 'wiki_candidates.json'))
    claude = json.load(open(S / 'claude_paragraphs.json'))
    screen = Screen(corpus())
    print('corpus texts', len(screen.items), flush=True)
    rows, rejected = [], []
    for dom in DOMAINS:
        for topic, text in claude[dom][:CLAUDE_N[dom]]:
            check = screen.check(text)
            row = {'domain': dom, 'topic': topic, 'origin': 'ai_generated', 'generator': 'claude-opus-5-5', 'generation_prompt': PROMPT,
                   'license': 'generated for this project', 'text': text, 'overlap_screen': check}
            if check['passed']:
                rows.append(row)
            else:
                rejected.append({'domain': dom, 'topic': topic, 'check': check})
        need = 15 - CLAUDE_N[dom]
        got = 0
        pages = set()
        for page in wiki[dom]:
            if got == need:
                break
            if page['pageid'] in pages:
                continue
            pages.add(page['pageid'])
            for option in page['options']:
                if NOT_STANDALONE.search(option):
                    continue
                check = screen.check(option)
                if check['passed']:
                    rows.append({'domain': dom, 'topic': page['title'], 'origin': 'human_likely_pre_2022', 'generator': None,
                                 'source': 'Wikipedia', 'page_title': page['title'], 'pageid': page['pageid'], 'revision_id': page['revid'],
                                 'revision_timestamp': page['rev_timestamp'], 'url': page['url'], 'license': 'CC BY-SA 3.0; contributors via page history',
                                 'text': option, 'overlap_screen': check})
                    got += 1
                    break
                rejected.append({'domain': dom, 'topic': page['title'], 'check': check})
        if got != need:
            sys.exit(f'{dom}: only {got}/{need} Wikipedia paragraphs available')
    for i, r in enumerate(sorted(rows, key=lambda x: digest(x['text'])), 1):
        r.update(holdout_id=f'jh-{i:03d}', family_id=digest('family|' + r['topic'])[:24], text_sha256=digest(r['text']), word_count=len(r['text'].split()),
                 detector_a={'status': 'pending', 'score_pct': None}, split='joint_holdout', training_eligible=False)
    rows.sort(key=lambda x: x['holdout_id'])
    if len(rows) != 150 or len({r['text_sha256'] for r in rows}) != 150 or len({r['family_id'] for r in rows}) != 150:
        sys.exit(f'bad counts {len(rows)}')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'holdout.json').write_text(json.dumps(rows, indent=1, ensure_ascii=False) + '\n')
    (OUT / 'overlap-rejected.json').write_text(json.dumps(rejected, indent=1) + '\n')
    wc = [r['word_count'] for r in rows]
    manifest = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(), 'purpose': 'Joint final test for Model 1 and Model 2. Keep out of training and out of checkpoint selection for both.',
                'counts': {'total': 150, 'by_origin': dict(C.Counter(r['origin'] for r in rows)), 'by_domain': dict(C.Counter(r['domain'] for r in rows))},
                'word_count': {'min': min(wc), 'max': max(wc), 'mean': round(sum(wc) / 150, 1)},
                'independence': {'one_paragraph_per_topic': True, 'corpus_texts_screened': len(screen.items),
                                 'screen': 'Rejected on exact normalized match, any shared 10-word passage, or word-sequence similarity >= 0.65 among the 25 best-matching corpus texts.',
                                 'max_similarity_found': max(r['overlap_screen']['max_word_sequence_similarity'] for r in rows), 'candidates_rejected': len(rejected)},
                'not_in_master': True, 'holdout_sha256': digest((OUT / 'holdout.json').read_text()), 'text_sha256_list': sorted(r['text_sha256'] for r in rows)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: manifest[k] for k in ('counts', 'word_count', 'independence', 'holdout_sha256')}, indent=1))
    print('rejected', [(x['topic'], x['check']['max_word_sequence_similarity'], x['check']['shared_10_word_passage']) for x in rejected])


if __name__ == '__main__':
    main()
