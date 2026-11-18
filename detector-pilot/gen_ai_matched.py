"""Write AI paragraphs on the same Wikipedia topics as wiki-human.json, so topic and register do not give origin away.

For each fetched article, up to three AI paragraphs are written, one per local generator, each aimed at the word
count of one of that article's human paragraphs (so length does not give origin away either). The human text is
never shown to the generator. Everything runs on this Mac: two Ollama models and the unadapted Qwen3 base in MLX.

Usage (resumable; processes whatever wiki-human.json holds so far):
    .venv-score/bin/python gen_ai_matched.py --generator llama3
    .venv-score/bin/python gen_ai_matched.py --generator qwen2.5:14b
    .venv-mlx/bin/python   gen_ai_matched.py --generator qwen3-1.7b
    .venv-mlx/bin/python   gen_ai_matched.py --rewrite     # live Rewriter on one AI paragraph per article

Output: ai-matched.jsonl and ai-matched-rewritten.jsonl beside this file.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DATA_ROOT', HERE.parent / 'data')).resolve()  # private project data, not in this repo
GENERATORS = ['llama3', 'qwen2.5:14b', 'qwen3-1.7b']   # slot i of article k uses GENERATORS[(k + i) % 3]
ASPECTS = ['its definition and basic characteristics', 'its history and origins', 'how it works or how it came about',
           'its significance and wider influence', 'the main debates or criticisms around it',
           'notable examples or episodes', 'its relationship to neighbouring subjects', 'how understanding of it has changed over time']
ADAPTER = ROOT / 'dataset-runs/two-models-2epochs-20261024T175519Z/humanizer-candidate-iteration-600'
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def articles():
    """Fetched articles with at least one human paragraph, in a stable order, with a stable index."""
    data = json.loads((HERE / 'wiki-human.json').read_text())
    rows = [dict(row, area=area) for area, items in data.items() for row in items if row.get('paragraphs')]
    rows.sort(key=lambda r: r['pageid'])
    for row in rows:
        row['index'] = int(sha(str(row['pageid']))[:8], 16)   # stable even while the fetch is still adding articles
    return rows


def jobs(generator):
    for row in articles():
        for slot, human in enumerate(row['paragraphs'][:3]):
            if GENERATORS[(row['index'] + slot) % 3] != generator:
                continue
            aspect = ASPECTS[(row['index'] // 3 + slot * 3) % len(ASPECTS)]
            words = len(human.split())
            prompt = (f'Write one paragraph of about {words} words for an encyclopedia article about "{row["title"]}". '
                      f'Focus on {aspect}. Use plain prose in a neutral, factual tone. '
                      'Output only the paragraph itself, with no heading, list, note or commentary.')
            yield {'key': f'{row["pageid"]}:{slot}', 'pageid': row['pageid'], 'title': row['title'], 'area': row['area'],
                   'slot': slot, 'generator': generator, 'aspect': aspect, 'target_words': words, 'prompt': prompt}


def clean(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    return re.sub(r'\s+', ' ', text).strip().strip('"').strip()


def load_done(path):
    return {json.loads(line)['key']: json.loads(line) for line in path.read_text().splitlines() if line.strip()} if path.exists() else {}


def ollama(model, prompt, seed):
    body = json.dumps({'model': model, 'prompt': prompt, 'stream': False, 'keep_alive': '10m',
                       'options': {'temperature': 0.7, 'top_p': 0.95, 'seed': seed, 'num_predict': 420}}).encode()
    request = urllib.request.Request('http://localhost:11434/api/generate', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)['response']


def generate(generator):
    out = HERE / 'ai-matched.jsonl'
    done = load_done(out)
    pending = [j for j in jobs(generator) if j['key'] not in done]
    if not pending:
        return 0
    if generator == 'qwen3-1.7b':
        from mlx_lm import load, generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
        model, tokenizer = load('mlx-community/Qwen3-1.7B-4bit')
    for job in pending:
        seed = int(sha(job['key'])[:6], 16)
        if generator == 'qwen3-1.7b':
            mx.random.seed(seed)
            chat = tokenizer.apply_chat_template([{'role': 'user', 'content': job['prompt']}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            raw = mlx_generate(model, tokenizer, prompt=chat, max_tokens=420, sampler=make_sampler(temp=0.7, top_p=0.95), verbose=False)
        else:
            raw = ollama(generator, job['prompt'], seed)
        text = clean(raw)
        row = dict(job, seed=seed, text=text, words=len(text.split()), text_sha256=sha(text))
        with out.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(generator, job['title'], job['target_words'], '->', row['words'], flush=True)
    return len(pending)


def rewrite():
    """Live Rewriter (checkpoint 600, app default prompt, temperature 0) on one AI paragraph per article."""
    sys.path.insert(0, str(HERE.parent / 'rewriter'))
    from mlx_lm import load
    from humanizer_prompts import inference_system_prompt
    from ai_content_eval import generate_rewrite
    source = load_done(HERE / 'ai-matched.jsonl')
    out = HERE / 'ai-matched-rewritten.jsonl'
    done = load_done(out)
    pending = []
    for row in articles():
        slot = row['index'] % min(3, len(row['paragraphs']))
        key = f'{row["pageid"]}:{slot}'
        if key in source and key not in done and source[key]['words'] >= 40:
            pending.append(source[key])
    if not pending:
        return 0
    system = inference_system_prompt(ADAPTER, 'academic', 'Humanized')
    model, tokenizer = load('mlx-community/Qwen3-1.7B-4bit', adapter_path=str(ADAPTER))
    for item in pending:
        text, finish = generate_rewrite(model, tokenizer, item['text'], 768, system)
        text = clean(text)
        row = {'key': item['key'], 'pageid': item['pageid'], 'title': item['title'], 'area': item['area'], 'slot': item['slot'],
               'source_generator': item['generator'], 'source_text_sha256': item['text_sha256'], 'humanization_source': 'Rewriter',
               'finish_reason': finish, 'text': text, 'words': len(text.split()), 'text_sha256': sha(text)}
        with out.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('rewriter', item['title'], item['words'], '->', row['words'], finish, flush=True)
    return len(pending)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generator', choices=GENERATORS)
    parser.add_argument('--rewrite', action='store_true')
    args = parser.parse_args()
    count = rewrite() if args.rewrite else generate(args.generator)
    print(json.dumps({'written': count}))
