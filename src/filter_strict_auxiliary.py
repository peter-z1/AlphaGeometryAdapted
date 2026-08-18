"""Keep only synthetic rows whose auxiliary is required by DD+AR.

The fast synthetic generator accepts a row when a traced proof uses a point
outside the visible goal's dependency closure.  That does not rule out a
second proof that DD+AR can find without the hidden point.  This module does
the expensive post-generation check:

1. solve the visible problem with DD+AR;
2. reject it if the visible problem is already solved;
3. restore the target auxiliary and solve again;
4. keep the row only when the restored problem is solved;
5. optionally remove superfluous target clauses.

It accepts file names or quoted globs and can partition input files across
independent workers with ``--shard_index`` and ``--num_shards``.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass, field
import gc
import glob
import hashlib
import itertools
import json
import logging
from pathlib import Path
import random
import signal
import sys
import time
from typing import Iterable

import numpy as np

import ddar
import graph as gh
import problem as pr
from generate_synthetic_data import proof_log_to_txt, traceback_goal


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEFS = ROOT / 'data' / 'defs.txt'
DEFAULT_RULES = ROOT / 'data' / 'rules.txt'


class SolveWallTimeout(BaseException):
    """Hard wall-clock timeout that broad ``except Exception`` cannot swallow."""


def _alarm_handler(signum, frame):  # pylint: disable=unused-argument
    raise SolveWallTimeout()


@dataclass
class SolveResult:
    outcome: str
    elapsed_seconds: float
    levels: int = 0
    level_seconds: float = 0.0
    error: str | None = None
    graph: gh.Graph | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            'outcome': self.outcome,
            'elapsed_seconds': round(self.elapsed_seconds, 3),
            'levels': self.levels,
            'level_seconds': round(self.level_seconds, 3),
        }
        if self.error:
            result['error'] = self.error
        return result


@dataclass
class RepeatedSolveResult:
    outcome: str
    attempts: list[SolveResult]
    graph: gh.Graph | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, object]:
        return {
            'outcome': self.outcome,
            'attempts': [attempt.to_json() for attempt in self.attempts],
        }


@dataclass
class FilterResult:
    category: str
    row: dict[str, object] | None
    details: dict[str, object]


def parse_auxiliary_clauses(text: str) -> list[pr.Clause]:
    clauses = [
        pr.Clause.from_txt(part.strip())
        for part in text.split(';')
        if part.strip()
    ]
    if not clauses:
        raise ValueError('target_auxiliary contains no clauses')
    return clauses


def auxiliary_construction_names(clauses: list[pr.Clause]) -> list[str]:
    """Return the minimized target's constructor names in clause order."""
    return [
        construction.name
        for clause in clauses
        for construction in clause.constructions
    ]


def stable_attempt_seed(row_id: object, attempt: int) -> int:
    digest = hashlib.blake2b(
        f'{row_id}:{attempt}'.encode('utf-8'), digest_size=4
    ).digest()
    return int.from_bytes(digest, 'big')


def solve_problem(
    clauses: list[pr.Clause],
    goal: pr.Construction,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    ddar_timeout: int,
    wall_timeout: int,
    seed: int,
) -> SolveResult:
    """Build and solve one problem while distinguishing failure from timeout."""
    random.seed(seed)
    np.random.seed(seed)
    problem = pr.Problem(url='', clauses=list(clauses), goal=goal)
    started = time.monotonic()
    use_alarm = wall_timeout > 0 and hasattr(signal, 'SIGALRM')
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(wall_timeout)

    try:
        graph, _ = gh.Graph.build_problem(problem, definitions, verbose=False)
        graph, level_times, _status, _branches, _added = ddar.solve(
            graph,
            rules,
            problem,
            max_level=max_level,
            timeout=ddar_timeout,
        )
        solved = graph.check(goal.name, graph.names2nodes(goal.args))
        elapsed = time.monotonic() - started
        level_seconds = sum(level_times)
        if solved:
            return SolveResult(
                'solved', elapsed, len(level_times), level_seconds, graph=graph
            )
        if len(level_times) >= max_level:
            return SolveResult('level_limit', elapsed, len(level_times), level_seconds)
        if level_times and level_times[-1] > ddar_timeout:
            return SolveResult('ddar_timeout', elapsed, len(level_times), level_seconds)
        return SolveResult('unsolved', elapsed, len(level_times), level_seconds)
    except SolveWallTimeout:
        return SolveResult('wall_timeout', time.monotonic() - started)
    except (Exception, SystemExit) as exc:  # pylint: disable=broad-exception-caught
        return SolveResult(
            'error',
            time.monotonic() - started,
            error=f'{type(exc).__name__}: {exc}',
        )
    finally:
        if use_alarm:
            signal.alarm(0)


def solve_repeated(
    clauses: list[pr.Clause],
    goal: pr.Construction,
    row_id: object,
    rebuild_attempts: int,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    ddar_timeout: int,
    wall_timeout: int,
    visible_phase: bool,
) -> RepeatedSolveResult:
    """Repeat numerical rebuilds and conservatively combine their outcomes.

    A visible problem is considered solved if any rebuild solves it.  It is
    considered unsolved only if every rebuild cleanly saturates without the
    goal.  A restored auxiliary is considered solved only if every rebuild
    solves it.  This avoids accepting numerical/build instability as evidence
    that an auxiliary is necessary.
    """
    attempts = []
    solved_graph = None
    for attempt in range(rebuild_attempts):
        result = solve_problem(
            clauses,
            goal,
            definitions,
            rules,
            max_level,
            ddar_timeout,
            wall_timeout,
            stable_attempt_seed(row_id, attempt),
        )
        attempts.append(result)
        if result.graph is not None and solved_graph is None:
            solved_graph = result.graph

    outcomes = [attempt.outcome for attempt in attempts]
    if visible_phase:
        if 'solved' in outcomes:
            outcome = 'solved'
        elif all(value == 'unsolved' for value in outcomes):
            outcome = 'unsolved'
        elif any('timeout' in value or value == 'level_limit' for value in outcomes):
            outcome = 'timeout'
        else:
            outcome = 'error'
    else:
        if all(value == 'solved' for value in outcomes):
            outcome = 'solved'
        elif any('timeout' in value or value == 'level_limit' for value in outcomes):
            outcome = 'timeout'
        elif any(value == 'error' for value in outcomes):
            outcome = 'error'
        else:
            outcome = 'unsolved'

    return RepeatedSolveResult(outcome, attempts, solved_graph)


def minimize_auxiliary(
    visible_clauses: list[pr.Clause],
    auxiliary_clauses: list[pr.Clause],
    goal: pr.Construction,
    row_id: object,
    mode: str,
    max_exact_clauses: int,
    solve_subset,
) -> tuple[list[pr.Clause], str, RepeatedSolveResult]:
    """Find a sufficient subset, exactly for small targets and greedily otherwise."""
    full_result = solve_subset(auxiliary_clauses)
    if full_result.outcome != 'solved':
        return auxiliary_clauses, 'none', full_result
    if mode == 'none' or len(auxiliary_clauses) == 1:
        return auxiliary_clauses, 'none', full_result

    if mode == 'exact' and len(auxiliary_clauses) <= max_exact_clauses:
        for size in range(1, len(auxiliary_clauses)):
            for indices in itertools.combinations(range(len(auxiliary_clauses)), size):
                subset = [auxiliary_clauses[index] for index in indices]
                result = solve_subset(subset)
                if result.outcome == 'solved':
                    return subset, 'exact', result
        return auxiliary_clauses, 'exact', full_result

    current = list(auxiliary_clauses)
    current_result = full_result
    changed = True
    while changed and len(current) > 1:
        changed = False
        for index in range(len(current)):
            subset = current[:index] + current[index + 1:]
            result = solve_subset(subset)
            if result.outcome == 'solved':
                current = subset
                current_result = result
                changed = True
                break
    return current, 'greedy', current_result


def filter_row(
    row: dict[str, object],
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    ddar_timeout: int,
    wall_timeout: int,
    rebuild_attempts: int,
    minimize: str,
    max_exact_clauses: int,
) -> FilterResult:
    row_id = row.get('id', '')
    details: dict[str, object] = {'id': row_id}
    try:
        visible = pr.Problem.from_txt(str(row['visible_problem']), translate=False)
        if visible.goal is None:
            raise ValueError('visible_problem has no goal')
        auxiliaries = parse_auxiliary_clauses(str(row['target_auxiliary']))
    except (KeyError, ValueError, TypeError) as exc:
        details['error'] = f'{type(exc).__name__}: {exc}'
        return FilterResult('parse_error', None, details)

    visible_result = solve_repeated(
        visible.clauses,
        visible.goal,
        row_id,
        rebuild_attempts,
        definitions,
        rules,
        max_level,
        ddar_timeout,
        wall_timeout,
        visible_phase=True,
    )
    details['visible'] = visible_result.to_json()
    if visible_result.outcome == 'solved':
        return FilterResult('visible_solved', None, details)
    if visible_result.outcome == 'timeout':
        return FilterResult('visible_timeout', None, details)
    if visible_result.outcome != 'unsolved':
        return FilterResult('visible_error', None, details)

    def solve_subset(subset: list[pr.Clause]) -> RepeatedSolveResult:
        return solve_repeated(
            visible.clauses + subset,
            visible.goal,
            row_id,
            rebuild_attempts,
            definitions,
            rules,
            max_level,
            ddar_timeout,
            wall_timeout,
            visible_phase=False,
        )

    selected, minimization, auxiliary_result = minimize_auxiliary(
        visible.clauses,
        auxiliaries,
        visible.goal,
        row_id,
        minimize,
        max_exact_clauses,
        solve_subset,
    )
    details['with_auxiliary'] = auxiliary_result.to_json()
    if auxiliary_result.outcome == 'timeout':
        return FilterResult('auxiliary_timeout', None, details)
    if auxiliary_result.outcome == 'error':
        return FilterResult('auxiliary_error', None, details)
    if auxiliary_result.outcome != 'solved':
        return FilterResult('restored_failed', None, details)

    strict_row = dict(row)
    original_target = str(row['target_auxiliary'])
    target = '; '.join(clause.txt() for clause in selected)
    strict_row['original_target_auxiliary'] = original_target
    strict_row['target_auxiliary'] = target
    strict_row['strict_auxiliary'] = True
    strict_row['strict_minimization'] = minimization
    strict_row['strict_num_original_aux_clauses'] = len(auxiliaries)
    strict_row['strict_num_aux_clauses'] = len(selected)
    target_names = auxiliary_construction_names(selected)
    strict_row['target_construction_names'] = target_names
    strict_row['target_construction_signature'] = '+'.join(sorted(target_names))
    focus = strict_row.get('focus_construction')
    strict_row['strict_focus_retained'] = (
        focus in target_names if isinstance(focus, str) and focus else None
    )
    strict_row['strict_visible_solve'] = visible_result.to_json()
    strict_row['strict_with_auxiliary_solve'] = auxiliary_result.to_json()
    strict_row['strict_problem'] = (
        '; '.join(clause.txt() for clause in visible.clauses + selected)
        + ' ? '
        + visible.goal.txt()
    )

    if auxiliary_result.graph is not None:
        traced = traceback_goal(auxiliary_result.graph, visible.goal)
        if traced is not None:
            setup, aux_setup, proof_log, _setup_points = traced
            strict_row['strict_setup_dependencies'] = [
                ' '.join([dep.name] + [getattr(arg, 'name', str(arg)) for arg in dep.args])
                for dep in setup
            ]
            strict_row['strict_auxiliary_dependencies'] = [
                ' '.join([dep.name] + [getattr(arg, 'name', str(arg)) for arg in dep.args])
                for dep in aux_setup
            ]
            strict_row['strict_proof_steps'] = proof_log_to_txt(proof_log)

    return FilterResult('aux_required', strict_row, details)


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(path) for path in sorted(glob.glob(pattern))]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        for path in matches:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('inputs', nargs='+', help='Input JSONL files or quoted globs.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rejected_out', type=Path)
    parser.add_argument('--stats_out', type=Path)
    parser.add_argument('--defs_file', type=Path, default=DEFAULT_DEFS)
    parser.add_argument('--rules_file', type=Path, default=DEFAULT_RULES)
    parser.add_argument('--max_level', type=int, default=1000)
    parser.add_argument('--ddar_timeout', type=int, default=10)
    parser.add_argument(
        '--wall_timeout',
        type=int,
        default=180,
        help='Hard wall-clock limit in seconds for one DD+AR solve; 0 disables.',
    )
    parser.add_argument(
        '--rebuild_attempts',
        type=int,
        default=1,
        help='Numerical rebuilds per solve. Values above one are more conservative.',
    )
    parser.add_argument('--minimize', choices=['exact', 'greedy', 'none'], default='exact')
    parser.add_argument(
        '--max_exact_clauses',
        type=int,
        default=4,
        help='Use exhaustive minimum-cardinality subset search up to this size.',
    )
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--max_rows', type=int, default=0)
    parser.add_argument('--flush_every', type=int, default=1)
    parser.add_argument(
        '--gc_every',
        type=int,
        default=100,
        help='Run cyclic garbage collection after this many rows; 0 disables.',
    )
    parser.add_argument('--log_every', type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rebuild_attempts < 1:
        raise ValueError('--rebuild_attempts must be at least 1')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('require 0 <= --shard_index < --num_shards')

    files = expand_inputs(args.inputs)
    if not files:
        print('no input files matched', file=sys.stderr)
        return 1
    assigned = files[args.shard_index::args.num_shards]
    if not assigned:
        print('no input files assigned to this shard', file=sys.stderr)
        return 1

    definitions = pr.Definition.from_txt_file(str(args.defs_file), to_dict=True)
    rules = pr.Theorem.from_txt_file(str(args.rules_file), to_dict=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.rejected_out:
        args.rejected_out.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.stats_out or args.out.with_suffix(args.out.suffix + '.stats.json')

    counts: collections.Counter[str] = collections.Counter()
    started = time.monotonic()
    read = 0
    kept = 0
    rejected_handle = (
        args.rejected_out.open('w', encoding='utf-8') if args.rejected_out else None
    )
    try:
        with args.out.open('w', encoding='utf-8') as out_handle:
            for path in assigned:
                with path.open('r', encoding='utf-8') as source:
                    for line_no, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        read += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            result = FilterResult(
                                'bad_json',
                                None,
                                {'file': str(path), 'line': line_no, 'error': str(exc)},
                            )
                        else:
                            result = filter_row(
                                row,
                                definitions,
                                rules,
                                args.max_level,
                                args.ddar_timeout,
                                args.wall_timeout,
                                args.rebuild_attempts,
                                args.minimize,
                                args.max_exact_clauses,
                            )

                        counts[result.category] += 1
                        if result.row is not None:
                            out_handle.write(json.dumps(result.row, sort_keys=True) + '\n')
                            kept += 1
                            for name in set(result.row['target_construction_names']):
                                counts[f'kept_aux_type_{name}'] += 1
                            signature = result.row['target_construction_signature']
                            counts[f'kept_aux_signature_{signature}'] += 1
                            if result.row.get('strict_focus_retained') is True:
                                counts['kept_focus_retained'] += 1
                            elif result.row.get('strict_focus_retained') is False:
                                counts['kept_focus_removed_by_minimization'] += 1
                            if args.flush_every > 0 and kept % args.flush_every == 0:
                                out_handle.flush()
                        elif rejected_handle is not None:
                            rejected_handle.write(
                                json.dumps(
                                    {
                                        'category': result.category,
                                        'source_file': str(path),
                                        'source_line': line_no,
                                        **result.details,
                                    },
                                    sort_keys=True,
                                )
                                + '\n'
                            )

                        if args.log_every > 0 and read % args.log_every == 0:
                            print(
                                json.dumps(
                                    {
                                        'read': read,
                                        'kept': kept,
                                        'counts': dict(counts),
                                        'elapsed_seconds': round(time.monotonic() - started, 1),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                        if args.gc_every > 0 and read % args.gc_every == 0:
                            gc.collect()
                        if args.max_rows and read >= args.max_rows:
                            break
                if args.max_rows and read >= args.max_rows:
                    break
            out_handle.flush()
    finally:
        if rejected_handle is not None:
            rejected_handle.close()

    stats = {
        'input_files_matched': len(files),
        'input_files_assigned': len(assigned),
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'rows_read': read,
        'rows_kept': kept,
        'counts': dict(sorted(counts.items())),
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'solver': {
            'max_level': args.max_level,
            'ddar_timeout': args.ddar_timeout,
            'wall_timeout': args.wall_timeout,
            'rebuild_attempts': args.rebuild_attempts,
            'minimize': args.minimize,
            'max_exact_clauses': args.max_exact_clauses,
        },
        'out': str(args.out),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    logging.disable(logging.ERROR)
    raise SystemExit(main())
