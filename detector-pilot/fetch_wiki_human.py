"""Fetch human-written paragraphs for the Detector pilot from Wikipedia revisions saved on or before 2021-12-31.

Same method as datasets/joint-holdout-20261106/wiki_fetch.py, on different articles: titles come from the
Level-4 vital-article lists, every page that was fetched for the joint holdout is excluded, and the paragraph
filter is looser (real input is not always tidy). Resumable: rerun to continue. One request at a time.

Output: wiki-human.json beside this file, {area: [{title, pageid, revid, rev_timestamp, url, paragraphs}]}.
Text is CC BY-SA 3.0; each row keeps its page and revision for attribution.
"""
import hashlib
import html
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'wiki-human.json'
TITLES = HERE / 'wiki-titles.json'
HOLDOUT = Path(os.environ.get('DATA_ROOT', HERE.parent / 'data')).resolve() / 'datasets' / 'joint-holdout-20261106'
API = 'https://en.wikipedia.org/w/api.php?'
UA = {'User-Agent': 'research-corpus-fetch/1.0 (polite single-threaded)'}  # generic on purpose: no project or personal details
AREAS = ['People', 'History', 'Geography', 'Arts', 'Philosophy and religion', 'Everyday life',
         'Society and social sciences', 'Biology and health sciences', 'Physical sciences', 'Technology', 'Mathematics']
PER_AREA = 48          # articles asked per area
PER_ARTICLE = 6        # paragraphs kept per article
PAUSE = 4              # seconds between requests
SEED = 20261118


def get(**params):
    params.update(format='json', formatversion='2')
    last = RuntimeError('no attempt made')
    for attempt in range(5):
        try:
            request = urllib.request.Request(API + urllib.parse.urlencode(params), headers=UA)
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last = error
            wait = int(error.headers.get('Retry-After') or 0) or 60 * (attempt + 1)
            print('backoff', error.code, wait, flush=True)
            time.sleep(wait)
        except Exception as error:  # network hiccup: wait and retry
            last = error
            time.sleep(20 * (attempt + 1))
    raise last


class Paragraphs(HTMLParser):
    """Collect the text of <p> elements, skipping references, tables, styles and formulas."""
    SKIP = ('sup', 'style', 'script', 'math', 'table')

    def __init__(self):
        super().__init__()
        self.current = None
        self.out = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP or (tag == 'span' and 'mwe-math' in (dict(attrs).get('class') or '')):
            self.skip += 1
        if tag == 'p' and self.current is None and not self.skip:
            self.current = []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        if tag == 'p' and self.current is not None:
            self.out.append(''.join(self.current))
            self.current = None

    def handle_data(self, data):
        if self.current is not None and not self.skip:
            self.current.append(data)


def usable(text):
    words = len(text.split())
    if not 40 <= words <= 200:
        return False
    if not re.search(r'[.!?]["”)]?$', text):
        return False
    if len(re.findall(r'[.!?](?:\s|$)', text)) < 2:
        return False
    if re.search(r'displaystyle|\{|\}|\\|\[|\]|citation needed|ISBN|\bhttp|:$', text):
        return False
    if text[0].islower():
        return False
    if sum(ord(c) > 0x24F and c not in '–—‘’“”…' for c in text) > 2:
        return False
    return True


def excluded_titles():
    """Every page asked for or used by the joint holdout, so no article is shared with it."""
    names = set()
    for rows in json.loads((HOLDOUT / 'wiki_candidates.json').read_text()).values():
        for row in rows:
            names.update({row.get('asked'), row.get('title')})
    for row in json.loads((HOLDOUT / 'holdout.json').read_text()):
        names.update({row.get('topic'), row.get('page_title')})
    return {n.lower() for n in names if n}


def choose_titles():
    if TITLES.exists():
        return json.loads(TITLES.read_text())
    banned = excluded_titles()
    rng = random.Random(SEED)
    chosen, seen = {}, set()
    for area in AREAS:
        page = 'Wikipedia:Vital articles/Level 4/' + area
        links = get(action='parse', page=page, prop='links')['parse']['links']
        titles = sorted({l['title'] for l in links if l['ns'] == 0 and l.get('exists', True)})
        titles = [t for t in titles if t.lower() not in banned and t not in seen]
        rng.shuffle(titles)
        assert len(titles) >= PER_AREA, f'{page}: only {len(titles)} titles'
        chosen[area] = titles[:PER_AREA]
        seen.update(chosen[area])
        print('titles', area, len(titles), '->', len(chosen[area]), flush=True)
        time.sleep(PAUSE)
    TITLES.write_text(json.dumps(chosen, ensure_ascii=False, indent=1))
    return chosen


def main():
    chosen = choose_titles()
    banned = excluded_titles()
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    for area, titles in chosen.items():
        done = {row['asked'] for row in result.get(area, [])}
        for title in titles:
            if title in done:
                continue
            row = {'asked': title, 'paragraphs': []}
            query = get(action='query', prop='revisions', titles=title, rvlimit=1, rvstart='2021-12-31T23:59:59Z',
                        rvdir='older', rvprop='ids|timestamp', redirects=1)
            page = query['query']['pages'][0]
            time.sleep(PAUSE)
            if 'revisions' in page and page['title'].lower() not in banned:
                revision = page['revisions'][0]
                parsed = get(action='parse', oldid=revision['revid'], prop='text', disableeditsection=1, disablelimitreport=1)
                parser = Paragraphs()
                parser.feed(parsed['parse']['text'])
                texts = [re.sub(r'\s+', ' ', html.unescape(p)).strip() for p in parser.out]
                keep = sorted({t for t in texts if usable(t)}, key=lambda t: hashlib.sha256(t.encode()).hexdigest())
                row.update(title=page['title'], pageid=page['pageid'], revid=revision['revid'],
                           rev_timestamp=revision['timestamp'], eligible=len(keep),
                           url=f"https://en.wikipedia.org/w/index.php?oldid={revision['revid']}",
                           paragraphs=keep[:PER_ARTICLE])
                time.sleep(PAUSE)
            result.setdefault(area, []).append(row)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
            print(area, title, row.get('rev_timestamp', 'none')[:10], len(row['paragraphs']), flush=True)
    total = sum(len(r['paragraphs']) for rows in result.values() for r in rows)
    print('DONE articles', sum(len(v) for v in result.values()), 'paragraphs', total, flush=True)


if __name__ == '__main__':
    main()
