"""Generate full theorem/proof examples for AlphaGeometry-style pretraining.

This is the "track 2" data path: unlike generate_synthetic_data.py, it does
not require a proof-critical hidden auxiliary point.  It samples a diagram,
runs DD+AR, traces true conclusions, and writes proof-imitation rows that can
be used before auxiliary-construction fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import problem as pr
from generate_synthetic_data import (
    PRETRAIN_GOAL_PREDICATES,
    ClosureResult,
    attempt_seed_for,
    choose_full_problem,
    construction_names_for_set,
    dep_to_txt,
    deps_to_txt,
    enumerate_candidate_goals,
    goal_holds_numerically,
    load_defs_rules,
    problem_txt,
    proof_log_to_txt,
    run_ddar,
    traceback_goal,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs' / 'synthetic_data' / 'pretrain_examples.jsonl'


def spaced_formal_text(text: str) -> str:
    for symbol in ['=', ',', ';', '?']:
        text = text.replace(symbol, f' {symbol} ')
    return ' '.join(text.split())


def deps_text(deps: Iterable[pr.Dependency]) -> str:
    return ' , '.join(dep_to_txt(dep) for dep in deps)


def proof_steps_text(
    log: list[tuple[list[pr.Dependency], list[pr.Dependency]]]
) -> str:
    steps = []
    for prems, cons in log:
        steps.append(f'{deps_text(prems)} => {deps_text(cons)}')
    return ' ; '.join(steps) + ' ;'


def unordered_pair(args: list[str], offset: int) -> frozenset[str]:
    return frozenset(args[offset:offset + 2])


def is_useful_goal(goal: pr.Construction) -> bool:
    """Drop the most obvious one-line/identity goals from pretraining data."""
    args = list(goal.args)
    if goal.name == 'para' and len(args) == 4:
        # Parallel lines sharing a point usually reduce to same-line facts.
        return not unordered_pair(args, 0).intersection(unordered_pair(args, 2))
    if goal.name in {'cong', 'perp'} and len(args) == 4:
        return unordered_pair(args, 0) != unordered_pair(args, 2)
    return True


def balanced_candidate_goals(
    candidates: list[pr.Construction],
    rng: random.Random,
    max_per_predicate: int,
) -> Iterable[pr.Construction]:
    """Yield candidates in a round-robin order across goal predicates.

    Raw DD+AR closures can be dominated by one predicate family, often congruent
    segments or collinearities.  A round-robin pass keeps each sampled diagram
    from spending its whole example budget on whichever predicate happens to be
    most numerous.
    """

    buckets: dict[str, list[pr.Construction]] = {}
    for goal in candidates:
        buckets.setdefault(goal.name, []).append(goal)

    predicate_names = list(buckets)
    rng.shuffle(predicate_names)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    emitted_by_predicate = {name: 0 for name in predicate_names}
    while True:
        emitted = False
        for name in predicate_names:
            if (
                max_per_predicate > 0
                and emitted_by_predicate[name] >= max_per_predicate
            ):
                continue
            bucket = buckets[name]
            if not bucket:
                continue
            emitted_by_predicate[name] += 1
            emitted = True
            yield bucket.pop()
        if not emitted:
            break


def make_pretraining_example(
    example_id: str,
    seed: int,
    attempt: int,
    candidate_index: int,
    full_problem: pr.Problem,
    goal: pr.Construction,
    setup: list[pr.Dependency],
    aux_setup: list[pr.Dependency],
    log: list[tuple[list[pr.Dependency], list[pr.Dependency]]],
    text_mode: str,
    construction_set: str,
    closure: ClosureResult,
) -> dict[str, object]:
    theorem_premises = setup + aux_setup
    premise_text = deps_text(theorem_premises)
    goal_text = dep_to_txt(goal)
    proof_prompt = f'{{S}} {premise_text} ? {goal_text} {{P}}'
    target = proof_steps_text(log) + ' {QED}'
    construction_statement = spaced_formal_text(problem_txt(full_problem.clauses, goal))
    construction_setup = spaced_formal_text(problem_txt(full_problem.clauses))

    if text_mode == 'construction_proof':
        prompt = f'{{C}} {construction_statement} {proof_prompt}'
    else:
        prompt = proof_prompt
    text = f'{prompt} {target}'

    return {
        'id': example_id,
        'seed': seed,
        'attempt': attempt,
        'candidate_index': candidate_index,
        'source': 'full_theorem_proof',
        'construction_set': construction_set,
        'full_problem': problem_txt(full_problem.clauses, goal),
        'full_setup': problem_txt(full_problem.clauses),
        'goal': goal.txt(),
        'construction_statement': construction_statement,
        'construction_setup': construction_setup,
        'proof_prompt': proof_prompt,
        'prompt': prompt,
        'target': target,
        'text': text,
        'text_mode': text_mode,
        'setup_dependencies': deps_to_txt(setup),
        'auxiliary_dependencies': deps_to_txt(aux_setup),
        'proof_steps': proof_log_to_txt(log),
        'num_setup_dependencies': len(setup),
        'num_auxiliary_dependencies': len(aux_setup),
        'num_proof_steps': len(log),
        'has_dependency_difference': bool(aux_setup),
        'numerically_verified': True,
        'closure': closure.to_json(),
    }


def iter_pretraining_examples(
    num_examples: int,
    seed: int,
    max_attempts: int,
    min_steps: int,
    max_steps: int,
    per_step_attempts: int,
    max_level: int,
    timeout: int,
    max_candidates: int,
    curated_rate: float,
    max_examples_per_problem: int,
    max_examples_per_predicate_per_problem: int,
    min_proof_steps: int,
    allow_trivial_goals: bool,
    dedupe_within_shard: bool,
    text_mode: str,
    construction_set: str,
    max_runtime_seconds: int,
    require_saturation: bool = False,
    goal_predicates: set[str] | None = None,
) -> Iterable[dict[str, object]]:
    goal_predicates = goal_predicates or PRETRAIN_GOAL_PREDICATES
    definitions, rules = load_defs_rules()
    construction_names = construction_names_for_set(definitions, construction_set)
    logging.info(
        'using construction_set=%s with %d constructions: %s',
        construction_set,
        len(construction_names),
        ' '.join(construction_names),
    )
    logging.info(
        'using %d goal predicates: %s',
        len(goal_predicates),
        ' '.join(sorted(goal_predicates)),
    )
    rng = random.Random(seed)
    np.random.seed(seed)
    num_found = 0
    start_time = time.monotonic()
    seen_text: set[str] = set()

    for attempt in range(max_attempts):
        if num_found >= num_examples:
            break
        if max_runtime_seconds and time.monotonic() - start_time >= max_runtime_seconds:
            logging.info(
                'stopping after %.1f seconds with %d examples',
                time.monotonic() - start_time,
                num_found,
            )
            break

        attempt_seed = attempt_seed_for(seed, attempt)
        rng.seed(attempt_seed)
        np.random.seed(attempt_seed)

        try:
            full_problem = choose_full_problem(
                definitions,
                rng,
                min_steps,
                max_steps,
                per_step_attempts,
                curated_rate,
                construction_set,
            )
            closure = run_ddar(full_problem, definitions, rules, max_level, timeout)
        except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
            continue

        if require_saturation and not closure.saturated:
            continue
        g, added = closure.graph, closure.added

        emitted_for_problem = 0
        raw_candidates = enumerate_candidate_goals(
            g, added, rng, max_candidates, goal_predicates=goal_predicates
        )
        candidates = balanced_candidate_goals(
            raw_candidates, rng, max_examples_per_predicate_per_problem
        )
        for candidate_index, goal in enumerate(candidates):
            if num_found >= num_examples:
                break
            if max_runtime_seconds and time.monotonic() - start_time >= max_runtime_seconds:
                break
            if (
                max_examples_per_problem > 0
                and emitted_for_problem >= max_examples_per_problem
            ):
                break
            if not allow_trivial_goals and not is_useful_goal(goal):
                continue
            if not goal_holds_numerically(g, goal):
                continue

            traced = traceback_goal(g, goal)
            if traced is None:
                continue
            setup, aux_setup, log, _setup_points = traced
            if len(log) < min_proof_steps:
                continue

            example = make_pretraining_example(
                f'{seed}-{attempt}-{candidate_index}-{num_found}',
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
            text_key = str(example['text'])
            if dedupe_within_shard and text_key in seen_text:
                continue
            seen_text.add(text_key)
            num_found += 1
            emitted_for_problem += 1
            yield example


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate full theorem/proof JSONL rows for LM pretraining.'
    )
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_attempts', type=int, default=1000)
    parser.add_argument('--min_steps', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=7)
    parser.add_argument('--per_step_attempts', type=int, default=25)
    parser.add_argument('--max_level', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--max_candidates', type=int, default=250)
    parser.add_argument('--curated_rate', type=float, default=0)
    parser.add_argument('--max_examples_per_problem', type=int, default=32)
    parser.add_argument(
        '--max_examples_per_predicate_per_problem',
        type=int,
        default=8,
        help='Per-diagram cap for each goal predicate; 0 disables this cap.',
    )
    parser.add_argument('--min_proof_steps', type=int, default=1)
    parser.add_argument('--allow_trivial_goals', action='store_true')
    parser.add_argument(
        '--no_dedupe_within_shard',
        action='store_true',
        help='Allow duplicate text rows inside one generated shard.',
    )
    parser.add_argument(
        '--text_mode',
        choices=['proof', 'construction_proof'],
        default='proof',
        help='Use proof-only text, or prepend the construction-form problem statement.',
    )
    parser.add_argument(
        '--construction_set',
        choices=['conservative', 'expanded', 'all'],
        default='conservative',
    )
    parser.add_argument(
        '--goal_predicates',
        default='',
        help=(
            'Comma-separated predicate names eligible as a theorem goal. '
            'Empty (default) uses PRETRAIN_GOAL_PREDICATES from '
            'generate_synthetic_data.py, which is wider than the 6-predicate '
            'GOAL_PREDICATES set used by the aux-construction fine-tuning '
            'generator.'
        ),
    )
    parser.add_argument(
        '--max_runtime_seconds',
        type=int,
        default=0,
        help='Stop cleanly after this many seconds; 0 means no runtime stop.',
    )
    parser.add_argument(
        '--require_saturation',
        action='store_true',
        help='Discard diagrams that stop at a DD+AR level limit or timeout.',
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--flush_every', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    goal_predicates = None
    if args.goal_predicates.strip():
        goal_predicates = {
            name.strip() for name in args.goal_predicates.split(',') if name.strip()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open('w', encoding='utf-8') as handle:
        for example in iter_pretraining_examples(
            num_examples=args.num_examples,
            seed=args.seed,
            max_attempts=args.max_attempts,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            per_step_attempts=args.per_step_attempts,
            max_level=args.max_level,
            timeout=args.timeout,
            max_candidates=args.max_candidates,
            curated_rate=args.curated_rate,
            max_examples_per_problem=args.max_examples_per_problem,
            max_examples_per_predicate_per_problem=(
                args.max_examples_per_predicate_per_problem
            ),
            min_proof_steps=args.min_proof_steps,
            allow_trivial_goals=args.allow_trivial_goals,
            dedupe_within_shard=not args.no_dedupe_within_shard,
            text_mode=args.text_mode,
            construction_set=args.construction_set,
            max_runtime_seconds=args.max_runtime_seconds,
            require_saturation=args.require_saturation,
            goal_predicates=goal_predicates,
        ):
            handle.write(json.dumps(example, sort_keys=True) + '\n')
            written += 1
            if args.flush_every > 0 and written % args.flush_every == 0:
                handle.flush()
            if args.log_every > 0 and written % args.log_every == 0:
                logging.info('wrote %d theorem/proof examples', written)

    logging.info('finished: wrote %d examples to %s', written, args.out)
    return 0 if written else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
