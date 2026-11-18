"""Export the paragraph pairs as MLX chat JSONL; no ML dependencies needed."""
import argparse
import json
from pathlib import Path
from train_qwen3_humanizer import SYSTEM, load_pairs, split_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='humanized-paragraphs.json')
    parser.add_argument('--output', default='mlx-data')
    parser.add_argument('--reviewed-only', action='store_true')
    args = parser.parse_args()
    rows = load_pairs(args.data, args.reviewed_only)
    train, valid = split_pairs(rows, 3407)
    # Preserve the validation split; hold out another complete batch for final testing.
    groups = {}
    for row in train:
        key = (row.get('document', 'unknown'), row.get('batch', 'unknown'))
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError('Need at least three source groups for separate train/validation/test splits.')
    candidates = [key for key in groups if len(groups[key]) >= 20] or list(groups)
    test_key = min(candidates, key=lambda key: (len(groups[key]), str(key)))
    test = groups[test_key]
    train = [row for key, group in groups.items() if key != test_key for row in group]
    splits = [('train', train), ('valid', valid), ('test', test)]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in splits:
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in split:
                record = {'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': row['before']},
                    {'role': 'assistant', 'content': row['after']},
                ]}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    manifest = {name: [{'document': r.get('document'), 'paragraph_index': r.get('paragraph_index'),
                         'batch': r.get('batch')} for r in split]
                for name, split in splits}
    (out / 'split-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Exported {len(train)} training and {len(valid)} validation and {len(test)} test examples to {out}.')


if __name__ == '__main__':
    main()
