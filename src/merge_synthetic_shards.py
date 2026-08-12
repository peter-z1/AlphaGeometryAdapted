"""Merge auxiliary-example shards, dedupe by canonical shape, report diversity.

Workers write independent JSONL shards.  This tool merges them into one file
while removing true duplicates: two rows count as duplicates when their
(visible_problem, target_auxiliary) pair is identical *after canonically
renaming points* in order of first appearance, so name-shuffled copies of the
same configuration collapse.

It also writes a stats JSON with the distributions that matter for corpus
diversity: aux construction types, goal predicates, aux clause counts and the
aux_step_fraction histogram.  Use it to spot a sampler skew early, before
spending CPU-weeks.

Example:
    python src/merge_synthetic_shards.py \
        'outputs/synthetic_data/raw_aux_cpu_v1/part_*.jsonl' \
        --out outputs/synthetic_data/aux_merged_v1.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path


POINT_TOKEN = re.compile(r'^(?:[a-z]|[a-z]\d+|p\d+)$')
TOKEN = re.compile(r'[A-Za-z_]\w*|\S')


def canonical_shape(visible_problem: str, target_auxiliary: str) -> str:
    """Rename point tokens in order of first appearance: a b x -> P0 P1 P2."""
    text = visible_problem + ' | ' + target_auxiliary
    mapping: dict[str, str] = {}
    out: list[str] = []
    for tok in TOKEN.findall(text):
        if POINT_TOKEN.fullmatch(tok):
            if tok not in mapping:
                mapping[tok] = f'P{len(mapping)}'
            out.append(mapping[tok])
        else:
            out.append(tok)
    return ' '.join(out)


def aux_construction_names(target_auxiliary: str) -> list[str]:
    names = []
    for clause in target_auxiliary.split(';'):
        rhs = clause.split('=', 1)[1] if '=' in clause else clause
        for part in rhs.split(','):
            part = part.strip()
            if part:
                names.append(part.split()[0])
    return names


def expand_inputs(specs: list[str]) -> list[str]:
    """Expand globs and @manifest files without shell argument-size limits."""
    expanded_specs: list[str] = []
    for spec in specs:
        if spec.startswith('@'):
            manifest = Path(spec[1:])
            with manifest.open(encoding='utf-8') as handle:
                expanded_specs.extend(
                    line.strip()
                    for line in handle
                    if line.strip() and not line.lstrip().startswith('#')
                )
        else:
            expanded_specs.append(spec)

    files: list[str] = []
    for pattern in expanded_specs:
        matched = sorted(glob.glob(pattern, recursive=True))
        if not matched and Path(pattern).exists():
            matched = [pattern]
        files.extend(matched)
    return files


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('inputs', nargs='+', help='Shard files or globs.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--min_aux_step_fraction',
        type=float,
        default=0.0,
        help='Drop rows whose hidden point touches a smaller fraction of proof '
             'steps (aux-centrality filter). Rows without the field are kept.',
    )
    parser.add_argument(
        '--dedupe',
        choices=['canonical', 'exact', 'none'],
        default='canonical',
        help='canonical: dedupe modulo point renaming (default); exact: dedupe '
             'on the raw (visible, aux) pair; none: concatenate only.',
    )
    args = parser.parse_args(argv)

    files = expand_inputs(args.inputs)
    if not files:
        print('no input shards matched', file=sys.stderr)
        return 1

    seen: set[str] = set()
    stats = {
        'rows_read': 0,
        'rows_bad_json': 0,
        'rows_dropped_low_aux_fraction': 0,
        'rows_duplicate': 0,
        'rows_written': 0,
    }
    aux_types = collections.Counter()
    goal_preds = collections.Counter()
    aux_clause_counts = collections.Counter()
    aux_fraction_hist = collections.Counter()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as out_f:
        for path in files:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats['rows_read'] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats['rows_bad_json'] += 1
                        continue

                    frac = row.get('aux_step_fraction')
                    if (
                        frac is not None
                        and frac < args.min_aux_step_fraction
                    ):
                        stats['rows_dropped_low_aux_fraction'] += 1
                        continue

                    if args.dedupe != 'none':
                        if args.dedupe == 'canonical':
                            key = canonical_shape(
                                row['visible_problem'], row['target_auxiliary']
                            )
                        else:
                            key = (
                                row['visible_problem']
                                + '\x00'
                                + row['target_auxiliary']
                            )
                        if key in seen:
                            stats['rows_duplicate'] += 1
                            continue
                        seen.add(key)

                    out_f.write(json.dumps(row, sort_keys=True) + '\n')
                    stats['rows_written'] += 1
                    for name in aux_construction_names(row['target_auxiliary']):
                        aux_types[name] += 1
                    goal_preds[row['goal'].split()[0]] += 1
                    aux_clause_counts[
                        len([c for c in row['target_auxiliary'].split(';') if c.strip()])
                    ] += 1
                    if frac is not None:
                        aux_fraction_hist[round(frac, 1)] += 1

    stats['shards'] = len(files)
    stats['aux_construction_types'] = dict(aux_types.most_common())
    stats['goal_predicates'] = dict(goal_preds.most_common())
    stats['aux_clause_counts'] = {
        str(k): v for k, v in sorted(aux_clause_counts.items())
    }
    stats['aux_step_fraction_hist'] = {
        str(k): v for k, v in sorted(aux_fraction_hist.items())
    }

    stats_path = args.out.with_suffix(args.out.suffix + '.stats.json')
    with stats_path.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(
        f"read {stats['rows_read']} rows from {len(files)} shards -> "
        f"wrote {stats['rows_written']} unique to {args.out}"
    )
    print(f'stats -> {stats_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
