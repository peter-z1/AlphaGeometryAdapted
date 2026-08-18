"""Select an executable, construction-balanced auxiliary training corpus.

Generation and strict filtering create valid rows, but their accepted
distribution reflects very different constructor survival rates.  This final
selection stage keeps the complete strict archive untouched and writes a
deterministic training view that round-robins construction families and goal
predicates.  Production selects *actions*, rather than multi-action strict
rows, so the measured distribution is the distribution the model actually
sees.  Absolute and fractional caps prevent a small corpus from silently
bypassing overly large nominal limits.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import problem as pr
from merge_synthetic_shards import canonical_shape
from synthetic_data_to_lm import (
    convert_row_to_action_pairs,
    validate_target_for_current_inference,
    within_limits,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEFS = ROOT / 'data' / 'defs.txt'


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(value) for value in sorted(glob.glob(pattern, recursive=True))]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        for path in matches:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return files


def construction_names(target_auxiliary: str) -> list[str]:
    names: list[str] = []
    for text in target_auxiliary.split(';'):
        text = text.strip()
        if not text:
            continue
        clause = pr.Clause.from_txt(text)
        names.extend(construction.name for construction in clause.constructions)
    return names


def row_construction_names(row: dict[str, object]) -> list[str]:
    """Return the named source family, falling back to inverse action syntax.

    Sequential conversion may express one ``centroid`` action as midpoint and
    line-intersection clauses.  Those inverse clauses are executable details,
    not the family label that generation and balancing should measure.
    """
    source = row.get('source_constructions')
    if isinstance(source, list) and source:
        return sorted({str(name) for name in source})
    recorded = row.get('target_construction_names')
    if isinstance(recorded, list) and recorded:
        return sorted({str(name) for name in recorded})
    return construction_names(str(row['target_auxiliary']))


def construction_family(row: dict[str, object]) -> str:
    names = row_construction_names(row)
    if not names:
        raise ValueError('target auxiliary has no construction')
    return '+'.join(sorted(names))


def construction_type_counts(
    rows: Iterable[dict[str, object]],
) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        counts.update(set(row_construction_names(row)))
    return counts


def balance_family(
    row: dict[str, object], prevalence: collections.Counter[str]
) -> str:
    """Assign a multi-action row to its rarest constituent construction."""
    names = set(row_construction_names(row))
    if not names:
        raise ValueError('target auxiliary has no construction')
    return min(names, key=lambda name: (prevalence[name], name))


def goal_predicate(row: dict[str, object]) -> str:
    goal = str(row['goal']).strip()
    if not goal:
        raise ValueError('row has no goal')
    return goal.split()[0]


def stable_priority(row: dict[str, object], seed: int) -> str:
    shape = canonical_shape(
        str(row['visible_problem']), str(row['target_auxiliary'])
    )
    return hashlib.sha256(f'{seed}\0{shape}'.encode('utf-8')).hexdigest()


def focus_match(row: dict[str, object]) -> bool:
    """Whether a soft-focus request became part of the hidden action.

    Soft focus is a preference only.  A valid row is never rejected merely
    because another construction became proof-relevant, but successful focus
    rows are consumed first inside the same constructor/goal cell.
    """
    retained = row.get('strict_focus_retained')
    if isinstance(retained, bool):
        return retained
    focus = row.get('focus_construction')
    return bool(
        isinstance(focus, str)
        and focus
        and focus in row_construction_names(row)
    )


def distribution_metrics(counter: collections.Counter[str]) -> dict[str, object]:
    total = sum(counter.values())
    if total == 0:
        return {
            'counts': {},
            'coverage': 0,
            'effective_families': 0.0,
            'top_four_fraction': 0.0,
        }
    probabilities = [count / total for count in counter.values() if count]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return {
        'counts': dict(counter.most_common()),
        'coverage': len(counter),
        'effective_families': round(math.exp(entropy), 3),
        'top_four_fraction': round(
            sum(count for _name, count in counter.most_common(4)) / total, 4
        ),
    }


def select_balanced_rows(
    rows: list[dict[str, object]],
    seed: int,
    target_rows: int = 0,
    per_family_cap: int = 0,
    per_goal_cap: int = 0,
    max_family_fraction: float = 0.0,
    max_goal_fraction: float = 0.0,
) -> list[dict[str, object]]:
    """Round-robin families and goals under absolute/fractional caps.

    Fractional caps are computed against the final selected size.  Because
    that size is not known in advance, selection starts from the requested
    size and monotonically tightens it until the selected size and quotas form
    a fixed point.  No row is duplicated to fill a quota.
    """
    if not rows:
        return []
    for name, value in (
        ('max_family_fraction', max_family_fraction),
        ('max_goal_fraction', max_goal_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')

    prevalence = construction_type_counts(rows)
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        family = balance_family(row, prevalence)
        goal = goal_predicate(row)
        grouped.setdefault(family, {}).setdefault(goal, []).append(row)

    for goals in grouped.values():
        for values in goals.values():
            values.sort(key=lambda row: (not focus_match(row), stable_priority(row, seed)))

    families = sorted(grouped)

    def capped(absolute: int, fraction: float, total: int) -> int:
        values = []
        if absolute > 0:
            values.append(absolute)
        if fraction > 0:
            # Ceil is the only sensible integer interpretation for small
            # corpora: with three available families, a 0.25 cap otherwise
            # permits zero rows from every family.  The realized fraction can
            # exceed the requested value by at most one row / selected size.
            values.append(max(1, math.ceil(fraction * total)))
        return min(values) if values else 0

    def select_once(maximum: int) -> list[dict[str, object]]:
        family_cap = capped(per_family_cap, max_family_fraction, maximum)
        goal_cap = capped(per_goal_cap, max_goal_fraction, maximum)
        positions: dict[tuple[str, str], int] = collections.defaultdict(int)
        family_goal_cursor: dict[str, int] = collections.defaultdict(int)
        family_counts: collections.Counter[str] = collections.Counter()
        goal_counts: collections.Counter[str] = collections.Counter()
        selected: list[dict[str, object]] = []

        while len(selected) < maximum:
            made_progress = False
            for family in families:
                if len(selected) >= maximum:
                    break
                if family_cap > 0 and family_counts[family] >= family_cap:
                    continue

                goals = sorted(grouped[family])
                if not goals:
                    continue
                start = family_goal_cursor[family] % len(goals)
                chosen: tuple[str, dict[str, object]] | None = None
                for offset in range(len(goals)):
                    goal = goals[(start + offset) % len(goals)]
                    if goal_cap > 0 and goal_counts[goal] >= goal_cap:
                        continue
                    key = (family, goal)
                    index = positions[key]
                    values = grouped[family][goal]
                    if index < len(values):
                        chosen = goal, values[index]
                        positions[key] += 1
                        family_goal_cursor[family] = start + offset + 1
                        break
                if chosen is None:
                    continue

                goal, row = chosen
                selected.append(row)
                family_counts[family] += 1
                goal_counts[goal] += 1
                made_progress = True

            if not made_progress:
                break
        return selected

    desired = min(target_rows, len(rows)) if target_rows > 0 else len(rows)
    while True:
        selected = select_once(desired)
        if len(selected) == desired or not selected:
            return selected
        desired = len(selected)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('inputs', nargs='+', help='Strict JSONL files or globs.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stats_out', type=Path)
    parser.add_argument('--rejected_out', type=Path)
    parser.add_argument('--defs_file', type=Path, default=DEFAULT_DEFS)
    parser.add_argument('--seed', type=int, default=20260813)
    parser.add_argument('--target_rows', type=int, default=0)
    parser.add_argument('--per_family_cap', type=int, default=0)
    parser.add_argument('--per_goal_cap', type=int, default=0)
    parser.add_argument(
        '--max_family_fraction',
        type=float,
        default=0.0,
        help='Maximum final fraction assigned to one constructor; 0 disables.',
    )
    parser.add_argument(
        '--max_goal_fraction',
        type=float,
        default=0.0,
        help='Maximum final fraction with one goal predicate; 0 disables.',
    )
    parser.add_argument(
        '--selection_unit',
        choices=['strict', 'action'],
        default='strict',
        help='Balance strict rows or executable autoregressive actions. Production uses action.',
    )
    parser.add_argument(
        '--allow_multi_target',
        action='store_true',
        help='Keep labels with multiple generated point groups.',
    )
    parser.add_argument(
        '--max_target_predicates',
        type=int,
        default=2,
        help='Maximum constrained predicates in a label; -1 disables the cap.',
    )
    parser.add_argument(
        '--allow_nonexecutable',
        action='store_true',
        help='Skip conversion validation. Intended only for strict-data archives.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_rows < 0 or args.per_family_cap < 0 or args.per_goal_cap < 0:
        raise ValueError('row targets and caps must be nonnegative')
    if not 0.0 <= args.max_family_fraction <= 1.0:
        raise ValueError('--max_family_fraction must be in [0, 1]')
    if not 0.0 <= args.max_goal_fraction <= 1.0:
        raise ValueError('--max_goal_fraction must be in [0, 1]')
    if args.selection_unit == 'action' and args.allow_nonexecutable:
        raise ValueError('action selection requires executable validation')
    files = expand_inputs(args.inputs)
    if not files:
        print('no input files matched', file=sys.stderr)
        return 1

    definitions = pr.Definition.from_txt_file(str(args.defs_file), to_dict=True)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_strict: set[str] = set()
    seen_units: set[str] = set()
    counters: collections.Counter[str] = collections.Counter()

    for path in files:
        opener = gzip.open if path.suffix == '.gz' else Path.open
        open_kwargs = (
            {'mode': 'rt', 'encoding': 'utf-8'}
            if path.suffix == '.gz'
            else {'mode': 'r', 'encoding': 'utf-8'}
        )
        with opener(path, **open_kwargs) as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                counters['rows_read'] += 1
                try:
                    row = json.loads(line)
                    family = construction_family(row)
                    shape = canonical_shape(
                        str(row['visible_problem']), str(row['target_auxiliary'])
                    )
                    if shape in seen_strict:
                        counters['rows_duplicate'] += 1
                        continue
                    converted: list[tuple[dict[str, object], dict[str, int]]] = []
                    if not args.allow_nonexecutable:
                        converted = convert_row_to_action_pairs(row, definitions)
                        for pair, metrics in converted:
                            executable, reason = within_limits(
                                metrics,
                                args.allow_multi_target,
                                args.max_target_predicates,
                            )
                            if not executable:
                                raise ValueError(f'nonexecutable_{reason}')
                            executable, reason = validate_target_for_current_inference(
                                str(pair['target']), str(pair['visible_problem'])
                            )
                            if not executable:
                                raise ValueError(f'nonexecutable_{reason}')
                    seen_strict.add(shape)
                    if args.selection_unit == 'strict':
                        row['target_construction_names'] = construction_names(
                            str(row['target_auxiliary'])
                        )
                        row['target_construction_signature'] = family
                        units = [row]
                    else:
                        units = []
                        for pair, _metrics in converted:
                            for key in (
                                'focus_construction',
                                'strict_focus_retained',
                                'aux_step_fraction',
                                'strict_num_aux_clauses',
                            ):
                                if key in row:
                                    pair[key] = row[key]
                            pair['strict_source_id'] = row.get('id')
                            pair['target_construction_names'] = row_construction_names(
                                pair
                            )
                            pair['target_construction_signature'] = (
                                construction_family(pair)
                            )
                            units.append(pair)
                        counters['actions_executable'] += len(units)

                    for unit in units:
                        unit_shape = canonical_shape(
                            str(unit['visible_problem']), str(unit['target_auxiliary'])
                        )
                        if unit_shape in seen_units:
                            counters['units_duplicate'] += 1
                            continue
                        seen_units.add(unit_shape)
                        accepted.append(unit)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    counters['rows_rejected'] += 1
                    reason = str(exc)
                    counters[f'rejected_{reason}'] += 1
                    rejected.append({
                        'source_file': str(path),
                        'source_line': line_no,
                        'reason': reason,
                    })

    before_families = collections.Counter(construction_family(row) for row in accepted)
    before_types = construction_type_counts(accepted)
    before_goals = collections.Counter(goal_predicate(row) for row in accepted)
    selected = select_balanced_rows(
        accepted,
        args.seed,
        args.target_rows,
        args.per_family_cap,
        args.per_goal_cap,
        args.max_family_fraction,
        args.max_goal_fraction,
    )
    after_families = collections.Counter(construction_family(row) for row in selected)
    after_types = construction_type_counts(selected)
    after_goals = collections.Counter(goal_predicate(row) for row in selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + '\n')

    if args.rejected_out:
        args.rejected_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rejected_out.open('w', encoding='utf-8') as handle:
            for row in rejected:
                handle.write(json.dumps(row, sort_keys=True) + '\n')

    stats = {
        'input_files': len(files),
        **dict(sorted(counters.items())),
        'rows_executable_unique': len(accepted),
        'rows_selected': len(selected),
        'selection': {
            'seed': args.seed,
            'target_rows': args.target_rows,
            'per_family_cap': args.per_family_cap,
            'per_goal_cap': args.per_goal_cap,
            'max_family_fraction': args.max_family_fraction,
            'max_goal_fraction': args.max_goal_fraction,
            'selection_unit': args.selection_unit,
            'allow_multi_target': args.allow_multi_target,
            'max_target_predicates': args.max_target_predicates,
        },
        'before': {
            'construction_types': distribution_metrics(before_types),
            'construction_families': distribution_metrics(before_families),
            'goal_predicates': distribution_metrics(before_goals),
        },
        'focus': {
            'available_matching': sum(focus_match(row) for row in accepted),
            'selected_matching': sum(focus_match(row) for row in selected),
        },
        'after': {
            'construction_types': distribution_metrics(after_types),
            'construction_families': distribution_metrics(after_families),
            'goal_predicates': distribution_metrics(after_goals),
        },
        'out': str(args.out),
    }
    stats_path = args.stats_out or args.out.with_suffix(args.out.suffix + '.stats.json')
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if selected else 1


if __name__ == '__main__':
    raise SystemExit(main())
