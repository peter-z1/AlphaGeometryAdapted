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
    if name == 'rconst':
        a, b, c, d, num, den = args
        num, den = pr.simplify(int(num), int(den))
        args = [a, b, c, d, f'{num}/{den}']

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
                    # Definition basics may contain integer literals.  In
                    # particular, triangle12 emits ``rconst ... 1 2``; those
                    # ratio terms are values, not formal point variables.
                    basic_args = [
                        arg if pr.isint(arg) else mapping[arg]
                        for arg in basic.args
                    ]
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
    return [pr.Clause.from_txt(c.strip()) for c in txt.split(';') if c.strip()]


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


def validate_target_for_current_inference(
    target: str,
    visible_problem: str,
) -> tuple[bool, str]:
    """Statically validate the one-new-point action interface used in search.

    The LM serializer can represent multi-point groups and predicates that the
    current AlphaGeometry search adapter cannot turn back into constructive
    clauses.  Training on such strings creates unreachable labels.  This
    check mirrors the adapter's lexical/arity translation without performing
    another numerical graph build.
    """
    visible = pr.Problem.from_txt(visible_problem, translate=False)
    existing = {point for clause in visible.clauses for point in clause.points}
    segments = [segment.strip() for segment in target.split(';') if segment.strip()]
    if not segments:
        return False, 'empty_target'

    for segment in segments:
        clause_txt, reason = constrained_segment_to_constructive(
            segment, existing
        )
        if clause_txt is None:
            return False, reason
        clause = pr.Clause.from_txt(clause_txt)
        existing.update(clause.points)
    return True, ''


def parse_target_segment(segment: str) -> tuple[list[str], list[list[str]]]:
    """Parse one LM action segment without interpreting reference numbers."""
    parts = segment.strip().removesuffix(';').strip().split(' : ')
    if len(parts) != 2:
        raise ValueError('invalid_segment')
    head, premise_text = parts
    point_names = head.split()

    predicates: list[list[str]] = [[]]
    for token in premise_text.split():
        if token.isdigit():
            if predicates[-1]:
                predicates.append([])
        else:
            predicates[-1].append(token)
    predicates = [predicate for predicate in predicates if predicate]
    return point_names, predicates


def format_target_segment(
    point_names: list[str],
    predicates: list[list[str]],
    start_ref: int,
) -> tuple[str, int]:
    """Format a parsed action with fresh monotonically increasing references."""
    formatted = []
    ref = start_ref
    for predicate in predicates:
        formatted.append(' '.join(predicate) + f' {ref:02}')
        ref += 1
    return ' '.join(point_names) + ' : ' + ' '.join(formatted), ref


def dependency_order_segments(
    segments: list[tuple[str, int, list[str]]],
    existing: set[str],
) -> list[tuple[str, int, list[str]]]:
    """Topologically order one-point groups when their clauses permit it.

    Some named multi-output constructions list dependent feet before their
    center.  The constrained groups themselves contain enough information to
    put the center first.  Genuinely coupled multi-point heads are deliberately
    not decomposed here.
    """
    pending = list(segments)
    ordered: list[tuple[str, int, list[str]]] = []
    known = set(existing)
    while pending:
        selected = None
        for index, (segment, _count, _families) in enumerate(pending):
            point_names, predicates = parse_target_segment(segment)
            if len(point_names) != 1:
                continue
            point = point_names[0]
            dependencies = {
                arg
                for predicate in predicates
                for arg in predicate[1:]
                if arg != point
            }
            if dependencies <= known:
                selected = index
                break
        if selected is None:
            # Preserve the original remainder so validation reports the first
            # unsupported/cyclic group and later groups are never emitted.
            ordered.extend(pending)
            break
        item = pending.pop(selected)
        ordered.append(item)
        point_names, _predicates = parse_target_segment(item[0])
        known.update(point_names)
    return ordered


def constrained_segment_to_constructive(
    segment: str,
    existing: set[str],
) -> tuple[str | None, str]:
    """Translate one executable constrained action to a constructive clause."""
    import alphageometry  # pylint: disable=import-outside-toplevel

    try:
        point_names, predicates = parse_target_segment(segment)
    except ValueError as exc:
        return None, str(exc)
    if len(point_names) != 1 or len(point_names[0]) != 1:
        return None, 'multi_point_head'
    point = point_names[0]
    if point in existing:
        return None, 'reused_point'
    if not predicates or len(predicates) > 2:
        return None, 'predicate_count'

    translated = []
    repeated_output_aline = False
    for predicate in predicates:
        name, *args = predicate
        try:
            mapped = pt.map_symbol(name)
        except Exception:  # pylint: disable=broad-exception-caught
            return None, 'unknown_predicate'
        if point not in args:
            return None, 'output_not_in_predicate'
        if not alphageometry.check_valid_args(mapped, args):
            return None, 'invalid_predicate'
        if any(arg != point and arg not in existing for arg in args):
            return None, 'hidden_prerequisite'
        try:
            construction, construction_args_ = (
                alphageometry.translate_constrained_to_constructive(
                    point, mapped, args
                )
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None, 'translation_error'
        if construction == 'on_aline' and construction_args_.count(point) > 1:
            repeated_output_aline = True
            continue
        translated.append(construction + ' ' + ' '.join(construction_args_))

    if repeated_output_aline and not any(
        construction.startswith('on_bline ') for construction in translated
    ):
        return None, 'repeated_output_in_on_aline'

    clause_txt = point + ' = ' + ', '.join(translated)
    try:
        pr.Clause.from_txt(clause_txt)
    except Exception:  # pylint: disable=broad-exception-caught
        return None, 'constructive_parse_error'
    return clause_txt, ''


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
        'source_constructions': sorted({
            construction.name
            for clause in parse_clause_list(target_auxiliary)
            for construction in clause.constructions
        }),
    }
    return pair, metrics


def problem_txt(problem: pr.Problem) -> str:
    if problem.goal is None:
        raise ValueError('problem has no goal')
    return (
        '; '.join(clause.txt() for clause in problem.clauses)
        + ' ? '
        + problem.goal.txt()
    )


def convert_row_to_action_pairs(
    row: dict[str, object],
    definitions: dict[str, pr.Definition],
    feature_token: str = DEFAULT_FEATURE_TOKEN,
) -> list[tuple[dict[str, object], dict[str, int]]]:
    """Split a rabbit into sequential one-point autoregressive actions.

    The constrained serializer sometimes exposes a multi-output named
    construction as several independent one-point groups.  Centroids and
    nine-point configurations are examples: their midpoint-like points can be
    introduced one at a time, and later groups only mention earlier outputs.
    Such groups are safe to train as successive inference actions even though
    they originated in one constructive clause.  A genuinely coupled group
    with a multi-point head (for example, both trisection points at once)
    remains unsupported and is left for validation to reject.
    """
    visible = pr.Problem.from_txt(str(row['visible_problem']), translate=False)
    if visible.goal is None:
        raise ValueError('visible_problem must include a goal')
    clauses = parse_clause_list(str(row['target_auxiliary']))
    if not clauses:
        raise ValueError('target_auxiliary contains no clauses')

    prompt, first_action_ref = problem_to_prompt(
        visible, definitions, feature_token
    )
    serialized: list[tuple[str, int, list[str]]] = []
    ordering_existing = {
        point for clause in visible.clauses for point in clause.points
    }
    for clause in clauses:
        segments, _next_ref, predicate_counts = clauses_to_lm_segments(
            [clause], definitions, start_ref=0
        )
        source_constructions = sorted({
            construction.name for construction in clause.constructions
        })
        clause_segments = list(zip(
            segments,
            predicate_counts,
            [source_constructions] * len(segments),
        ))
        serialized.extend(
            dependency_order_segments(clause_segments, ordering_existing)
        )
        ordering_existing.update(clause.points)

    # Reordering changes the presentation order, so assign references only
    # after the dependency order is final.
    next_ref = first_action_ref
    renumbered = []
    for segment, predicate_count, source_constructions in serialized:
        point_names, predicates = parse_target_segment(segment)
        segment, next_ref = format_target_segment(
            point_names, predicates, next_ref
        )
        renumbered.append((segment, predicate_count, source_constructions))
    serialized = renumbered

    pairs: list[tuple[dict[str, object], dict[str, int]]] = []
    current_clauses = list(visible.clauses)
    existing = {point for clause in current_clauses for point in clause.points}
    current_prompt = prompt
    action_count = len(serialized)
    for action_index, (segment, predicate_count, source_constructions) in enumerate(
        serialized
    ):
        current = pr.Problem(url='', clauses=list(current_clauses), goal=visible.goal)
        target = segment + ' ;'
        point_names, _ = parse_target_segment(segment)
        metrics = {
            'target_clauses': 1,
            'target_groups': 1,
            'target_points': len(point_names),
            'target_predicates': predicate_count,
        }
        current_visible = problem_txt(current)
        constructive_clause, _reason = constrained_segment_to_constructive(
            segment, existing
        )
        pair = {
            'id': f'{row.get("id")}-action-{action_index}',
            'source_id': row.get('id'),
            'seed': row.get('seed'),
            'diagram_id': row.get('diagram_id'),
            'action_index': action_index,
            'action_count': action_count,
            'prompt': current_prompt,
            'target': target,
            'text': current_prompt + ' ' + target,
            'visible_problem': current_visible,
            'target_auxiliary': (
                constructive_clause
                if constructive_clause is not None
                else str(row['target_auxiliary'])
            ),
            'source_target_auxiliary': row.get('target_auxiliary'),
            'source_constructions': source_constructions,
            'full_problem': row.get('full_problem'),
            'full_setup': row.get('full_setup'),
            'goal': row.get('goal'),
        }
        pairs.append((pair, metrics))
        if constructive_clause is None:
            # Later groups cannot be reached if this action cannot be applied.
            break
        clause = pr.Clause.from_txt(constructive_clause)
        current_clauses.append(clause)
        existing.update(clause.points)
        # This is exactly how run_alphageometry extends the LM context after a
        # successful action; rebuilding from generic inverse constructions can
        # otherwise introduce redundant basic predicates.
        current_prompt += ' ' + target + ' x00'
    return pairs


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
    parser.add_argument(
        '--split_actions',
        action='store_true',
        help='Write one autoregressive row per sequential one-point group, '
             'including groups from a multi-output constructive clause.',
    )
    parser.add_argument(
        '--stats_out',
        type=Path,
        help='Optional JSON report of acceptance and rejection by source '
             'construction family.',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    definitions = pr.Definition.from_txt_file(str(args.defs_file), to_dict=True)

    stats = {
        'read': 0,
        'written': 0,
        'actions_seen': 0,
        'actions_rejected': 0,
        'actions_discarded_with_row': 0,
        'rows_complete': 0,
        'rows_partial': 0,
        'rows_rejected': 0,
        'unreachable_actions': 0,
        'multi_target': 0,
        'too_many_predicates': 0,
        'errors': 0,
    }
    family_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            'seen': 0,
            'written': 0,
            'rejected': 0,
            'rejection_reasons': defaultdict(int),
        }
    )

    def pair_families(pair: dict[str, object]) -> list[str]:
        families = pair.get('source_constructions', [])
        if not isinstance(families, list) or not families:
            return ['unknown']
        return [str(family) for family in families]

    def record(pair: dict[str, object], outcome: str, reason: str = '') -> None:
        for family in pair_families(pair):
            family_stats[family][outcome] = int(
                family_stats[family][outcome]
            ) + 1
            if reason:
                reasons = family_stats[family]['rejection_reasons']
                assert isinstance(reasons, defaultdict)
                reasons[reason] += 1

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
                converted = (
                    convert_row_to_action_pairs(row, definitions, args.feature_token)
                    if args.split_actions
                    else [convert_row(row, definitions, args.feature_token)]
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                stats['errors'] += 1
                print(f'warning: skipped row {stats["read"]}: {exc}', file=sys.stderr)
                continue

            row_pairs: list[dict[str, object]] = []
            rejected_reason = ''
            for pair, metrics in converted:
                stats['actions_seen'] += 1
                record(pair, 'seen')
                ok, reason = within_limits(
                    metrics, args.allow_multi_target, args.max_target_predicates
                )
                if not ok:
                    rejected_reason = reason
                else:
                    executable, reason = validate_target_for_current_inference(
                        str(pair['target']), str(pair['visible_problem'])
                    )
                    if not executable:
                        rejected_reason = reason
                if rejected_reason:
                    stats['actions_rejected'] += 1
                    record(pair, 'rejected', rejected_reason)
                    if args.split_actions:
                        remaining = max(
                            0,
                            int(pair.get('action_count', len(converted)))
                            - int(pair.get('action_index', 0)) - 1,
                        )
                        stats['unreachable_actions'] += remaining
                    break
                row_pairs.append(pair)
            if rejected_reason:
                stats[rejected_reason] = stats.get(rejected_reason, 0) + 1
                if row_pairs:
                    stats['rows_partial'] += 1
                    stats['actions_discarded_with_row'] += len(row_pairs)
                    for pair in row_pairs:
                        record(pair, 'rejected', 'incomplete_action_sequence')
                # A strict auxiliary example is useful only if inference can
                # reproduce its entire action chain.  Do not train on an easy
                # prefix whose required suffix is unreachable.
                row_pairs = []
                stats['rows_rejected'] += 1
            else:
                stats['rows_complete'] += 1
            for pair in row_pairs:
                dst.write(json.dumps(pair, sort_keys=True) + '\n')
                stats['written'] += 1
                record(pair, 'written')

    report = {
        **stats,
        'input': str(args.input),
        'out': str(args.out),
        'split_actions': args.split_actions,
        'by_source_construction': {
            family: {
                **values,
                'rejection_reasons': dict(values['rejection_reasons']),
            }
            for family, values in sorted(family_stats.items())
        },
    }
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    print(
        'converted {written} actions from {read} rows to {out} '
        '(complete={rows_complete}, partial={rows_partial}, '
        'rejected={rows_rejected}, action_rejections={actions_rejected}, '
        'errors={errors})'.format(out=args.out, **stats)
    )
    return 0 if stats['written'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
