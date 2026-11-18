"""Check that a dataset file or export folder shares no text with the joint holdout.

Usage: python verify.py /path/to/export-or-file.json [more paths]
Exits 1 and lists the rows if any exact or whitespace/case-normalized match is found.
"""
import hashlib, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
norm = lambda t: ' '.join(re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", t.lower()))
rows = json.loads((HERE / 'holdout.json').read_text())
manifest = json.loads((HERE / 'manifest.json').read_text())
if hashlib.sha256((HERE / 'holdout.json').read_text().encode()).hexdigest() != manifest['holdout_sha256']:
    sys.exit('holdout.json no longer matches its manifest')
held = {norm(r['text']): r['holdout_id'] for r in rows}

def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)

found = checked = 0
for arg in sys.argv[1:]:
    path = Path(arg)
    for file in ([path] if path.is_file() else sorted(path.rglob('*.json*'))):
        lines = file.read_text().splitlines() if file.suffix == '.jsonl' else None
        values = [json.loads(line) for line in lines if line.strip()] if lines is not None else json.loads(file.read_text())
        checked += 1
        for text in strings(values):
            if len(text) > 200 and norm(text) in held:
                found += 1
                print(f'{file}: matches {held[norm(text)]}')
print(json.dumps({'files_checked': checked, 'matches': found}))
sys.exit(1 if found else 0)
