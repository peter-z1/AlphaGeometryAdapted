"""Deduplicate and split converted AlphaGeometry LM-pair JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = Path('outputs/synthetic_data/lm_dataset')


def iter_rows(paths: Iterable[Path]):
    for path in paths:
        with path.open('r', encoding='utf-8') as src:
            for line_no, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line), path, line_no
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: invalid JSONL: {exc}') from exc


def row_text(row: dict[str, object], text_field: str) -> str:
    value = row.get(text_field)
    if value:
        return str(value)
    prompt = str(row['prompt'])
    target = str(row['target'])
    return prompt + ' ' + target


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.blake2b(
        f'{seed}\0{key}'.encode('utf-8'), digest_size=8
    ).digest()
    value = int.from_bytes(digest, byteorder='big', signed=False)
    return value / float(1 << 64)


def choose_split(key: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    value = stable_fraction(key, seed)
    if value < test_ratio:
        return 'test'
    if value < test_ratio + val_ratio:
        return 'val'
    return 'train'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Deduplicate and split AlphaGeometry LM-pair JSONL data.'
    )
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--text_field', default='text')
    parser.add_argument(
        '--dedupe_key',
        choices=['text', 'prompt', 'target', 'full_problem', 'none'],
        default='text',
    )
    parser.add_argument(
        '--split_key',
        choices=['text', 'prompt', 'full_problem', 'full_setup', 'diagram_id', 'id'],
        default='text',
    )
    parser.add_argument('--val_ratio', type=float, default=0.01)
    parser.add_argument('--test_ratio', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_rows', type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.val_ratio < 0 or args.test_ratio < 0:
        raise ValueError('split ratios must be non-negative')
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError('val_ratio + test_ratio must be less than 1')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_files = {
        split: (args.out_dir / f'{split}.jsonl').open('w', encoding='utf-8')
        for split in ['train', 'val', 'test']
    }
    text_files = {
        split: (args.out_dir / f'{split}.txt').open('w', encoding='utf-8')
        for split in ['train', 'val', 'test']
    }

    stats = {
        'read': 0,
        'written': 0,
        'duplicates': 0,
        'splits': {'train': 0, 'val': 0, 'test': 0},
    }
    seen: set[str] = set()

    try:
        for row, _path, _line_no in iter_rows(args.inputs):
            stats['read'] += 1
            text = row_text(row, args.text_field)
            dedupe_value = ''
            if args.dedupe_key != 'none':
                dedupe_value = text if args.dedupe_key == 'text' else str(row.get(args.dedupe_key, ''))
                if dedupe_value in seen:
                    stats['duplicates'] += 1
                    continue
                seen.add(dedupe_value)

            if args.split_key == 'text':
                split_value = text
            else:
                split_value = str(row.get(args.split_key, dedupe_value or text))
            split = choose_split(split_value, args.seed, args.val_ratio, args.test_ratio)

            row = dict(row)
            row['text'] = text
            jsonl_files[split].write(json.dumps(row, sort_keys=True) + '\n')
            text_files[split].write(text + '\n')
            stats['splits'][split] += 1
            stats['written'] += 1

            if args.max_rows and stats['written'] >= args.max_rows:
                break
    finally:
        for handle in list(jsonl_files.values()) + list(text_files.values()):
            handle.close()

    summary_path = args.out_dir / 'summary.json'
    summary_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'out_dir': str(args.out_dir), **stats}, indent=2, sort_keys=True))
    return 0 if stats['written'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
