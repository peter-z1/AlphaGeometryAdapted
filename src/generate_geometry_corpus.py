"""Generate general-proof and auxiliary-candidate corpora from shared closures.

Each accepted diagram is built once and DD+AR is run until saturation.  The
resulting closure is then mined into two durable streams:

* theorem/proof rows for language-model pretraining;
* dependency-difference rows for later strict auxiliary-point pruning.

Every selected conclusion must also hold in the diagram's numerical
realization.  Invalid numerical realizations and non-saturated closures are
reported in the audit JSON and never emitted as training data.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

import problem as pr
from generate_pretraining_data import (
    balanced_candidate_goals,
    is_useful_goal,
    make_pretraining_example,
)
from generate_synthetic_data import (
    GOAL_PREDICATES,
    PRETRAIN_GOAL_PREDICATES,
    AttemptTimeout,
    ClosureResult,
    attempt_seed_for,
    choose_full_problem,
    enumerate_candidate_goals,
    goal_holds_numerically,
    load_defs_rules,
    make_example,
    run_ddar,
    traceback_goal,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / 'outputs' / 'synthetic_data' / 'geometry_corpus_v2'


def _alarm_handler(signum, frame):  # pylint: disable=unused-argument
    raise AttemptTimeout()


def mine_diagram_rows(
    diagram_id: str,
    attempt_seed: int,
    attempt: int,
    full_problem: pr.Problem,
    closure: ClosureResult,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    rng: random.Random,
    construction_set: str,
    max_candidates: int,
    max_pretraining_per_diagram: int,
    max_auxiliary_per_diagram: int,
    max_per_predicate: int,
    min_proof_steps: int,
    allow_trivial_goals: bool,
    text_mode: str,
    emit_pretraining: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], collections.Counter]:
    """Mine both row types without recomputing the diagram closure."""
    counts: collections.Counter[str] = collections.Counter()
    pretraining_rows: list[dict[str, object]] = []
    auxiliary_rows: list[dict[str, object]] = []

    raw_candidates = enumerate_candidate_goals(
        closure.graph,
        closure.added,
        rng,
        max_candidates,
        goal_predicates=PRETRAIN_GOAL_PREDICATES,
    )
    candidates = balanced_candidate_goals(raw_candidates, rng, max_per_predicate)
    for candidate_index, goal in enumerate(candidates):
        counts['candidates_considered'] += 1
        if not goal_holds_numerically(closure.graph, goal):
            counts['candidates_rejected_numerical'] += 1
            continue

        traced = traceback_goal(closure.graph, goal)
        if traced is None:
            counts['candidates_rejected_traceback'] += 1
            continue
        setup, aux_setup, log, setup_points = traced
        if len(log) < min_proof_steps:
            counts['candidates_rejected_short_proof'] += 1
            continue
        counts['candidates_verified'] += 1

        if (
            emit_pretraining
            and (max_pretraining_per_diagram <= 0
             or len(pretraining_rows) < max_pretraining_per_diagram)
            and (allow_trivial_goals or is_useful_goal(goal))
        ):
            row = make_pretraining_example(
                f'{diagram_id}-proof-{candidate_index}',
                attempt_seed,
                attempt,
                candidate_index,
                full_problem,
                goal,
                setup,
                aux_setup,
                log,
                text_mode,
                construction_set,
                closure,
            )
            row['diagram_id'] = diagram_id
            pretraining_rows.append(row)
            counts['pretraining_emitted'] += 1

        if (
            goal.name in GOAL_PREDICATES
            and (max_auxiliary_per_diagram <= 0
                 or len(auxiliary_rows) < max_auxiliary_per_diagram)
        ):
            row = make_example(
                f'{diagram_id}-aux-{candidate_index}',
                attempt_seed,
                full_problem,
                goal,
                setup,
                aux_setup,
                log,
                setup_points,
                definitions,
                rules,
                require_unsolved_without_aux=False,
                max_level=1000,
                timeout=10,
            )
            if row is not None:
                row['source'] = 'dependency_difference_candidate'
                row['diagram_id'] = diagram_id
                row['numerically_verified'] = True
                row['closure'] = closure.to_json()
                auxiliary_rows.append(row)
                counts['auxiliary_candidates_emitted'] += 1

        pretraining_full = (
            not emit_pretraining
            or (
                max_pretraining_per_diagram > 0
                and len(pretraining_rows) >= max_pretraining_per_diagram
            )
        )
        auxiliary_full = (
            max_auxiliary_per_diagram > 0
            and len(auxiliary_rows) >= max_auxiliary_per_diagram
        )
        if pretraining_full and auxiliary_full:
            break

    return pretraining_rows, auxiliary_rows, counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--num_diagrams', type=int, default=10)
    parser.add_argument('--max_attempts', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min_steps', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--per_step_attempts', type=int, default=25)
    parser.add_argument('--max_level', type=int, default=1000)
    parser.add_argument('--ddar_timeout', type=int, default=10)
    parser.add_argument('--attempt_time_budget', type=int, default=180)
    parser.add_argument('--max_runtime_seconds', type=int, default=0)
    parser.add_argument('--max_candidates', type=int, default=500)
    parser.add_argument('--max_pretraining_per_diagram', type=int, default=96)
    parser.add_argument('--max_auxiliary_per_diagram', type=int, default=32)
    parser.add_argument('--max_per_predicate', type=int, default=24)
    parser.add_argument('--min_proof_steps', type=int, default=1)
    parser.add_argument('--allow_trivial_goals', action='store_true')
    parser.add_argument(
        '--skip_pretraining',
        action='store_true',
        help='Mine only auxiliary candidates; do not materialize theorem/proof rows.',
    )
    parser.add_argument(
        '--gzip_pretraining',
        action='store_true',
        help='Write pretraining rows directly to pretraining.jsonl.gz.',
    )
    parser.add_argument('--curated_rate', type=float, default=0)
    parser.add_argument(
        '--construction_set',
        choices=['conservative', 'expanded', 'all'],
        default='expanded',
    )
    parser.add_argument(
        '--text_mode',
        choices=['proof', 'construction_proof'],
        default='construction_proof',
    )
    parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_diagrams < 1 or args.max_attempts < 1:
        raise ValueError('--num_diagrams and --max_attempts must be positive')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pretraining_path = args.out_dir / (
        'pretraining.jsonl.gz' if args.gzip_pretraining else 'pretraining.jsonl'
    )
    auxiliary_path = args.out_dir / 'auxiliary_candidates.jsonl'
    audit_path = args.out_dir / 'audit.json'

    definitions, rules = load_defs_rules()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    counts: collections.Counter[str] = collections.Counter()
    closure_statuses: collections.Counter[str] = collections.Counter()
    started = time.monotonic()
    diagrams_written = 0
    use_alarm = args.attempt_time_budget > 0 and hasattr(signal, 'SIGALRM')
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)

    pretraining_open = gzip.open if args.gzip_pretraining else Path.open
    pretraining_kwargs = (
        {'mode': 'wt', 'encoding': 'utf-8'}
        if args.gzip_pretraining
        else {'mode': 'w', 'encoding': 'utf-8'}
    )
    with (
        pretraining_open(pretraining_path, **pretraining_kwargs) as pretraining_handle,
        auxiliary_path.open('w', encoding='utf-8') as auxiliary_handle,
    ):
        for attempt in range(args.max_attempts):
            if diagrams_written >= args.num_diagrams:
                break
            if (
                args.max_runtime_seconds > 0
                and time.monotonic() - started >= args.max_runtime_seconds
            ):
                counts['stopped_runtime_limit'] += 1
                break

            counts['diagrams_sampled'] += 1
            attempt_seed = attempt_seed_for(args.seed, attempt)
            rng.seed(attempt_seed)
            np.random.seed(attempt_seed)
            if use_alarm:
                signal.alarm(args.attempt_time_budget)
            try:
                full_problem = choose_full_problem(
                    definitions,
                    rng,
                    args.min_steps,
                    args.max_steps,
                    args.per_step_attempts,
                    args.curated_rate,
                    args.construction_set,
                )
                closure = run_ddar(
                    full_problem,
                    definitions,
                    rules,
                    args.max_level,
                    args.ddar_timeout,
                )
                closure_statuses[closure.status] += 1
                if not closure.saturated:
                    counts['diagrams_rejected_not_saturated'] += 1
                    continue

                diagram_id = f'{args.seed}-{attempt}'
                pretraining_rows, auxiliary_rows, diagram_counts = mine_diagram_rows(
                    diagram_id,
                    attempt_seed,
                    attempt,
                    full_problem,
                    closure,
                    definitions,
                    rules,
                    rng,
                    args.construction_set,
                    args.max_candidates,
                    args.max_pretraining_per_diagram,
                    args.max_auxiliary_per_diagram,
                    args.max_per_predicate,
                    args.min_proof_steps,
                    args.allow_trivial_goals,
                    args.text_mode,
                    not args.skip_pretraining,
                )
                counts.update(diagram_counts)
                for row in pretraining_rows:
                    pretraining_handle.write(json.dumps(row, sort_keys=True) + '\n')
                for row in auxiliary_rows:
                    auxiliary_handle.write(json.dumps(row, sort_keys=True) + '\n')
                pretraining_handle.flush()
                auxiliary_handle.flush()
                diagrams_written += 1
                counts['diagrams_emitted'] += 1
            except AttemptTimeout:
                counts['diagrams_rejected_attempt_timeout'] += 1
            except (Exception, SystemExit) as exc:  # pylint: disable=broad-exception-caught
                counts[f'diagrams_rejected_{type(exc).__name__}'] += 1
            finally:
                if use_alarm:
                    signal.alarm(0)

    audit = {
        'config': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'counts': dict(sorted(counts.items())),
        'closure_statuses': dict(sorted(closure_statuses.items())),
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'outputs': {
            'pretraining': str(pretraining_path),
            'auxiliary_candidates': str(auxiliary_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if diagrams_written else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
