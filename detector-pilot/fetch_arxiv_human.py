"""Fetch about 200 arXiv abstracts submitted before 2022 as an unseen human-written test source.

Test only: these never enter training or validation. One request per category, a few seconds apart.
Output: arxiv-human.json beside this file. Abstracts are the authors' text; each row keeps its arXiv id for attribution.
"""
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'arxiv-human.json'
CATEGORIES = ['econ.GN', 'cs.SE', 'cs.CY', 'q-bio.PE', 'physics.soc-ph', 'stat.AP', 'physics.ed-ph', 'cs.HC']
PER_CATEGORY = 40
ATOM = '{http://www.w3.org/2005/Atom}'
UA = {'User-Agent': 'research-corpus-fetch/1.0 (polite single-threaded)'}  # generic on purpose


def main():
    rows = []
    for category in CATEGORIES:
        query = f'cat:{category} AND submittedDate:[201801010000 TO 202112312359]'
        url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode(
            {'search_query': query, 'start': 0, 'max_results': PER_CATEGORY, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as response:
            feed = ET.fromstring(response.read())
        kept = 0
        for entry in feed.iter(ATOM + 'entry'):
            text = re.sub(r'\s+', ' ', entry.findtext(ATOM + 'summary') or '').strip()
            published = entry.findtext(ATOM + 'published') or ''
            words = len(text.split())
            # Same shape limits as the training paragraphs; drop abstracts with LaTeX or links.
            if not 40 <= words <= 200 or published[:4] > '2021' or re.search(r'\$|\\|\{|\}|http', text):
                continue
            rows.append({'arxiv_id': entry.findtext(ATOM + 'id'), 'category': category, 'published': published, 'text': text, 'words': words})
            kept += 1
        print(category, kept, flush=True)
        time.sleep(4)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print('DONE', len(rows))


if __name__ == '__main__':
    main()
