"""Convert synthetic auxiliary-construction JSONL to LM training pairs.

The generator in ``generate_synthetic_data.py`` emits proof-audited examples in
the constructive AlphaGeometry DSL.  The language model, however, is trained to
predict constrained auxiliary strings such as:

    e : C a c e 02 C b d e 03 ;

This script converts each JSONL row into a prompt/target pair compatible with
``lm_inference.py`` and ``alphageometry.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Iterable

import pretty as pt
import problem as pr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEFS = ROOT / 'data' / 'defs.txt'
DEFAULT_FEATURE_TOKEN = '{F1} x00'


def normalize_basic(
    name: str, args: list[str]
) -> tuple[str, list[str]]:
    """Match Problem.setup_str_from_problem's basic-predicate normalization."""
    if name in ['s_angle', 'aconst']:
        x, y, z, v = args
        name = 'aconst'
        v = int(v)

        if v < 0:
            v = -v
            x, z = z, x

        m, n = pr.simplify(int(v), 180)
        args = [y, z, y, x, f'{m}pi/{n}']

    return name, args


def dep_to_lm_txt(dep: tuple[str, ...]) -> str:
    dep_str = pt.pretty(dep)
    if dep[0] == 'aconst':
        m, n = map(int, dep[-1].split('pi/'))
        mn = f'{m}. pi / {n}.'
        dep_str = ' '.join(dep_str.split()[:-1] + [mn])
    return dep_str


def construction_args(
    clause: pr.Clause,
    construction: pr.Construction,
    definitions: dict[str, pr.Definition],
) -> list[str]:
    cdef = definitions[construction.name]
    args = list(construction.args)
    if len(args) != len(cdef.construction.args):
        if len(args) + len(clause.points) != len(cdef.construction.args):
            correct_form = ' '.join(
                cdef.points + ['=', construction.name] + cdef.args
            )
            raise ValueError('Argument mismatch. ' + correct_form)
        args = list(clause.points) + args
    return args


def clauses_to_lm_segments(
    clauses: Iterable[pr.Clause],
    definitions: dict[str, pr.Definition],
    start_ref: int = 0,
) -> tuple[list[str], int, list[int]]:
    """Format clauses as LM setup/auxiliary segments.

    Returns the formatted segments, the next available reference number, and the
    number of basic predicates in each segment.
    """
    ref = start_ref
    segments = []
    predicate_counts = []

    for clause in clauses:
        group = {}
        p2deps = defaultdict(list)
        for construction in clause.constructions:
            cdef = definitions[construction.name]
            args = construction_args(clause, construction, definitions)
            mapping = dict(zip(cdef.construction.args, args))

            for points, basics in cdef.basics:
                points = tuple([mapping[x] for x in points])
                for p in points:
                    group[p] = points

                for basic in basics:
                    basic_args = [mapping[a] for a in basic.args]
                    name, basic_args = normalize_basic(basic.name, basic_args)
                    p2deps[points].append(pr.hashed_txt(name, basic_args))

        for k, v in p2deps.items():
            p2deps[k] = pr.sort_deps(v)

        points = list(clause.points)
        while points:
            p = points[0]
            if p not in group:
                raise ValueError(f'Cannot find LM point group for `{p}` in `{clause.txt()}`')

            gr = group[p]
            points = [x for x in points if x not in gr]

            deps_str = []
            for dep in p2deps[gr]:
                deps_str.append(dep_to_lm_txt(dep) + ' {:02}'.format(ref))
                ref += 1

            segments.append((' '.join(gr) + ' : ' + ' '.join(deps_str)).strip())
            predicate_counts.append(len(p2deps[gr]))

    return segments, ref, predicate_counts


def problem_to_prompt(
    problem: pr.Problem,
    definitions: dict[str, pr.Definition],
    feature_token: str = DEFAULT_FEATURE_TOKEN,
) -> tuple[str, int]:
    if problem.goal is None:
        raise ValueError('visible_problem must include a goal')

    segments, next_ref, _ = clauses_to_lm_segments(problem.clauses, definitions)
    prompt = '{S} ' + ' ; '.join([s.strip() for s in segments])
    prompt += ' ? ' + pt.pretty([problem.goal.name] + problem.goal.args)
    if feature_token:
        prompt += ' ' + feature_token
    return prompt, next_ref


def parse_clause_list(txt: str) -> list[pr.Clause]:
    return [pr.Clause.from_txt(c.strip()) for c in txt.split('; ') if c.strip()]


def auxiliary_to_target(
    target_auxiliary: str,
    definitions: dict[str, pr.Definition],
    start_ref: int,
) -> tuple[str, dict[str, int]]:
    clauses = parse_clause_list(target_auxiliary)
    segments, _next_ref, predicate_counts = clauses_to_lm_segments(
        clauses, definitions, start_ref=start_ref
    )
    target = ' ; '.join(segments) + ' ;'
    metrics = {
        'target_clauses': len(clauses),
        'target_groups': len(segments),
        'target_points': sum(len(c.points) for c in clauses),
        'target_predicates': sum(predicate_counts),
    }
    return target, metrics


def convert_row(
    row: dict[str, object],
    definitions: dict[str, pr.Definition],
    feature_token: str = DEFAULT_FEATURE_TOKEN,
) -> tuple[dict[str, object], dict[str, int]]:
    visible_problem = str(row['visible_problem'])
    target_auxiliary = str(row['target_auxiliary'])

    problem = pr.Problem.from_txt(visible_problem, translate=False)
    prompt, next_ref = problem_to_prompt(problem, definitions, feature_token)
    target, metrics = auxiliary_to_target(target_auxiliary, definitions, next_ref)

    pair = {
        'id': row.get('id'),
        'seed': row.get('seed'),
        'diagram_id': row.get('diagram_id'),
        'prompt': prompt,
        'target': target,
        'text': prompt + ' ' + target,
        'visible_problem': visible_problem,
        'target_auxiliary': target_auxiliary,
        'full_problem': row.get('full_problem'),
        'full_setup': row.get('full_setup'),
        'goal': row.get('goal'),
    }
    return pair, metrics


def within_limits(
    metrics: dict[str, int],
    allow_multi_target: bool,
    max_target_predicates: int,
) -> tuple[bool, str]:
    if not allow_multi_target and metrics['target_groups'] != 1:
        return False, 'multi_target'
    if max_target_predicates >= 0 and metrics['target_predicates'] > max_target_predicates:
        return False, 'too_many_predicates'
    return True, ''


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert synthetic auxiliary JSONL to LM prompt/target JSONL.'
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--defs_file', type=Path, default=DEFAULT_DEFS)
    parser.add_argument('--feature_token', default=DEFAULT_FEATURE_TOKEN)
    parser.add_argument(
        '--allow_multi_target',
        action='store_true',
        help='Keep targets that require more than one generated point/group.',
    )
    parser.add_argument(
        '--max_target_predicates',
        type=int,
        default=2,
        help='Skip targets with more constrained predicates. Use -1 for no limit.',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    definitions = pr.Definition.from_txt_file(str(args.defs_file), to_dict=True)

    stats = {
        'read': 0,
        'written': 0,
        'multi_target': 0,
        'too_many_predicates': 0,
        'errors': 0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open('r', encoding='utf-8') as src, args.out.open(
        'w', encoding='utf-8'
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            stats['read'] += 1
            try:
                row = json.loads(line)
                pair, metrics = convert_row(row, definitions, args.feature_token)
                ok, reason = within_limits(
                    metrics, args.allow_multi_target, args.max_target_predicates
                )
                if not ok:
                    stats[reason] += 1
                    continue
            except Exception as exc:  # pylint: disable=broad-exception-caught
                stats['errors'] += 1
                print(f'warning: skipped row {stats["read"]}: {exc}', file=sys.stderr)
                continue

            dst.write(json.dumps(pair, sort_keys=True) + '\n')
            stats['written'] += 1

    print(
        'converted {written}/{read} rows to {out} '
        '(multi_target={multi_target}, too_many_predicates={too_many_predicates}, '
        'errors={errors})'.format(out=args.out, **stats)
    )
    return 0 if stats['written'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
