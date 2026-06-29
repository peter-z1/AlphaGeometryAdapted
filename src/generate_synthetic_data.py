"""Generate AlphaGeometry-style synthetic auxiliary-construction data.

This is not an attempt to reproduce DeepMind's private training set bit for
bit.  It rebuilds the same kind of pipeline from the released solver:

1. sample a construction sequence;
2. build the full geometry graph;
3. run DD+AR;
4. mine true conclusions whose traceback uses a proof-only point;
5. hide the construction of that point and write a training pair.

The output is JSONL so that long independent runs can later be concatenated.
For large runs, prefer many CPU workers with different seed ranges.  The data
generation itself is mostly symbolic/numerical geometry, not GPU work; GPUs
become relevant for training the language model on the generated examples.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

import ddar
import graph as gh
import problem as pr
import trace_back


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs' / 'synthetic_data' / 'auxiliary_examples.jsonl'

POINT_NAMES = list('abcdefghijklmnopqrstuvwxyz') + [f'p{i}' for i in range(1000)]

# A deliberately conservative first library.  These definitions all have point
# arguments only, and the sampler still checks their formal dependencies before
# adding them to the current graph.
RANDOM_CONSTRUCTIONS = [
    'angle_bisector',
    'angle_mirror',
    'circle',
    'circumcenter',
    'eqangle2',
    'eqdistance',
    'excenter',
    'foot',
    'incenter',
    'intersection_ll',
    'intersection_lp',
    'intersection_lt',
    'intersection_tt',
    'midpoint',
    'mirror',
    'nsquare',
    'on_circle',
    'on_circum',
    'on_line',
    'on_pline',
    'on_tline',
    'orthocenter',
    'psquare',
    'reflect',
]

# A tiny set of known-good full diagrams.  These are useful as "smoke seeds":
# they verify that the mining stage can discover an auxiliary construction even
# before the random sampler finds one.
CURATED_FULL_PROBLEMS = [
    (
        'a b c = triangle a b c; '
        'd = on_tline d b a c, on_tline d c a b; '
        'e = on_line e a c, on_line e b d'
    ),
]

GOAL_PREDICATES = {'coll', 'cong', 'cyclic', 'midp', 'para', 'perp'}


def load_defs_rules() -> tuple[dict[str, pr.Definition], dict[str, pr.Theorem]]:
    defs = pr.Definition.from_txt_file(str(ROOT / 'data' / 'defs.txt'), to_dict=True)
    rules = pr.Theorem.from_txt_file(str(ROOT / 'data' / 'rules.txt'), to_dict=True)
    return defs, rules


def node_name(x: object) -> str:
    return getattr(x, 'name', str(x))


def dep_to_txt(dep: pr.Dependency | pr.Construction) -> str:
    return ' '.join([dep.name] + [node_name(a) for a in dep.args])


def deps_to_txt(deps: Iterable[pr.Dependency]) -> list[str]:
    return [dep_to_txt(d) for d in deps]


def proof_log_to_txt(
    log: list[tuple[list[pr.Dependency], list[pr.Dependency]]]
) -> list[dict[str, list[str]]]:
    return [
        {'premises': deps_to_txt(prems), 'conclusions': deps_to_txt(cons)}
        for prems, cons in log
    ]


def make_problem(clauses: list[pr.Clause], goal: pr.Construction | None = None) -> pr.Problem:
    return pr.Problem(url='', clauses=list(clauses), goal=goal)


def problem_txt(clauses: list[pr.Clause], goal: pr.Construction | None = None) -> str:
    setup = '; '.join(c.txt() for c in clauses)
    if goal is None:
        return setup
    return setup + ' ? ' + goal.txt()


def base_triangle() -> pr.Clause:
    return pr.Clause(
        ['a', 'b', 'c'],
        [pr.Construction('triangle', ['a', 'b', 'c'])],
    )


def construction_clause(
    name: str,
    cdef: pr.Definition,
    existing: list[str],
    next_name_index: int,
    rng: random.Random,
) -> tuple[pr.Clause, list[str], int] | None:
    """Sample one clause for a definition, using the full argument form."""
    output_vars = list(cdef.points)
    input_vars = list(cdef.args)

    if not output_vars or len(input_vars) > len(existing):
        return None

    outputs = POINT_NAMES[next_name_index: next_name_index + len(output_vars)]
    if len(outputs) < len(output_vars):
        return None

    # Use distinct existing points for the formal input variables.  This avoids
    # many immediate diff/ncoll failures while still letting the graph reject
    # geometrically impossible choices.
    inputs = rng.sample(existing, len(input_vars))
    mapping = dict(zip(output_vars, outputs))
    mapping.update(zip(input_vars, inputs))
    args = [mapping[v] for v in cdef.construction.args]
    clause = pr.Clause(outputs, [pr.Construction(name, args)])
    return clause, outputs, next_name_index + len(outputs)


def dependencies_hold(
    g: gh.Graph, clause: pr.Clause, definitions: dict[str, pr.Definition]
) -> bool:
    """Check formal construction dependencies without invoking build retries."""
    for construction in clause.constructions:
        cdef = definitions[construction.name]
        if len(cdef.construction.args) != len(construction.args):
            return False

        mapping = dict(zip(cdef.construction.args, construction.args))
        for dep in cdef.deps.constructions:
            try:
                args = g.names2points([mapping[a] for a in dep.args])
            except (KeyError, ValueError):
                return False
            try:
                if not g.check(dep.name, args):
                    return False
            except Exception:  # pylint: disable=broad-exception-caught
                return False
    return True


def add_clause_safely(
    g: gh.Graph,
    clause: pr.Clause,
    plevel: int,
    definitions: dict[str, pr.Definition],
) -> tuple[bool, list[pr.Dependency], int]:
    if not dependencies_hold(g, clause, definitions):
        return False, [], plevel

    try:
        added, plevel = g.add_clause(clause, plevel, definitions, verbose=False)
    except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
        return False, [], plevel

    for dep in added:
        g.add_algebra(dep, level=0)
    return True, added, plevel


def sample_random_problem(
    definitions: dict[str, pr.Definition],
    rng: random.Random,
    min_steps: int,
    max_steps: int,
    per_step_attempts: int,
) -> pr.Problem:
    clauses = [base_triangle()]
    problem = make_problem(clauses)
    g, added = gh.Graph.build_problem(problem, definitions, verbose=False)
    for dep in added:
        g.add_algebra(dep, level=0)

    plevel = g.plevel
    existing = ['a', 'b', 'c']
    next_name_index = 3
    target_steps = rng.randint(min_steps, max_steps)

    for _ in range(target_steps):
        for _attempt in range(per_step_attempts):
            name = rng.choice(RANDOM_CONSTRUCTIONS)
            if name not in definitions:
                continue

            sampled = construction_clause(
                name, definitions[name], existing, next_name_index, rng
            )
            if sampled is None:
                continue

            clause, new_points, candidate_next_index = sampled
            ok, _added, candidate_plevel = add_clause_safely(
                g, clause, plevel, definitions
            )
            if not ok:
                continue

            clauses.append(clause)
            existing.extend(new_points)
            next_name_index = candidate_next_index
            plevel = candidate_plevel
            break

    return make_problem(clauses)


def choose_full_problem(
    definitions: dict[str, pr.Definition],
    rng: random.Random,
    min_steps: int,
    max_steps: int,
    per_step_attempts: int,
    curated_rate: float,
) -> pr.Problem:
    if rng.random() < curated_rate:
        return pr.Problem.from_txt(rng.choice(CURATED_FULL_PROBLEMS), translate=False)
    return sample_random_problem(
        definitions, rng, min_steps, max_steps, per_step_attempts
    )


def run_ddar(
    problem: pr.Problem,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    timeout: int,
) -> tuple[gh.Graph, list[pr.Dependency]]:
    g, added = gh.Graph.build_problem(problem, definitions, verbose=False)
    controller = pr.Problem(url='', clauses=problem.clauses, goal=None)
    g, _level_times, _status, _branches, inferred = ddar.solve(
        g, rules, controller, max_level=max_level, timeout=timeout
    )
    return g, added + inferred


def as_goal(name: str, args: Iterable[object]) -> pr.Construction:
    return pr.Construction(name, [node_name(a) for a in args])


def enumerate_candidate_goals(
    g: gh.Graph,
    added: list[pr.Dependency],
    rng: random.Random,
    max_candidates: int,
) -> list[pr.Construction]:
    candidates: list[pr.Construction] = []
    seen: set[tuple[str, ...]] = set()

    def maybe_add(name: str, args: Iterable[object]) -> None:
        if name not in GOAL_PREDICATES:
            return
        goal = as_goal(name, args)
        key = pr.hashed_txt(goal.name, goal.args)
        if key in seen:
            return
        seen.add(key)
        candidates.append(goal)

    for dep in added:
        if all(hasattr(arg, 'name') for arg in dep.args):
            maybe_add(dep.name, dep.args)

    points = sorted(g.all_points(), key=lambda p: p.name)
    for pts in itertools.combinations(points, 3):
        try:
            if g.check('coll', list(pts)):
                maybe_add('coll', pts)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    for pts in itertools.combinations(points, 4):
        try:
            if g.check('cyclic', list(pts)):
                maybe_add('cyclic', pts)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    segments = list(itertools.combinations(points, 2))
    for s1, s2 in itertools.combinations(segments, 2):
        args = list(s1 + s2)
        for name in ['cong', 'para', 'perp']:
            try:
                if g.check(name, args):
                    maybe_add(name, args)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    for m in points:
        for a, b in segments:
            if m in {a, b}:
                continue
            args = [m, a, b]
            try:
                if g.check('midp', args):
                    maybe_add('midp', args)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    rng.shuffle(candidates)
    return candidates[:max_candidates]


def traceback_goal(
    g: gh.Graph, goal: pr.Construction
) -> tuple[
    list[pr.Dependency],
    list[pr.Dependency],
    list[tuple[list[pr.Dependency], list[pr.Dependency]]],
    set[object],
] | None:
    try:
        args = g.names2nodes(goal.args)
        query = pr.Dependency(goal.name, args, None, None)
        return trace_back.get_logs(query, g, merge_trivials=False)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def hidden_point_names(
    aux_setup: list[pr.Dependency], setup_points: set[object]
) -> set[str]:
    setup = {node_name(p) for p in setup_points}
    hidden = set()
    for dep in aux_setup:
        for arg in dep.args:
            if hasattr(arg, 'name') and arg.name not in setup:
                hidden.add(arg.name)
    return hidden


def clause_mentions_hidden(clause: pr.Clause, hidden: set[str]) -> bool:
    if any(p in hidden for p in clause.points):
        return True
    for construction in clause.constructions:
        if any(arg in hidden for arg in construction.args):
            return True
    return False


def split_visible_aux_and_dropped(
    clauses: list[pr.Clause], hidden: set[str]
) -> tuple[list[pr.Clause], list[pr.Clause], list[pr.Clause], set[str]]:
    """Remove hidden-dependent clauses, but target only traceback hidden points."""
    direct_hidden = set(hidden)
    hidden_for_visibility = set(hidden)
    visible: list[pr.Clause] = []
    removed: list[pr.Clause] = []
    changed = True

    while changed:
        changed = False
        visible.clear()
        removed.clear()
        for clause in clauses:
            if clause_mentions_hidden(clause, hidden_for_visibility):
                removed.append(clause)
                for p in clause.points:
                    if p not in hidden_for_visibility:
                        hidden_for_visibility.add(p)
                        changed = True
            else:
                visible.append(clause)

    target_auxiliary = [
        clause for clause in removed if any(p in direct_hidden for p in clause.points)
    ]
    dropped = [clause for clause in removed if clause not in target_auxiliary]
    return visible, target_auxiliary, dropped, hidden_for_visibility


def solved_without_auxiliary(
    visible_clauses: list[pr.Clause],
    goal: pr.Construction,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    timeout: int,
) -> bool:
    p = make_problem(visible_clauses, goal)
    try:
        g, _added = gh.Graph.build_problem(p, definitions, verbose=False)
        g, _times, _status, _branches, _all_added = ddar.solve(
            g, rules, p, max_level=max_level, timeout=timeout
        )
        return g.check(goal.name, g.names2nodes(goal.args))
    except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
        return False


def solved_with_auxiliary(
    visible_clauses: list[pr.Clause],
    aux_clauses: list[pr.Clause],
    goal: pr.Construction,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    timeout: int,
) -> bool:
    p = make_problem(visible_clauses + aux_clauses, goal)
    try:
        g, _added = gh.Graph.build_problem(p, definitions, verbose=False)
        g, _times, _status, _branches, _all_added = ddar.solve(
            g, rules, p, max_level=max_level, timeout=timeout
        )
        return g.check(goal.name, g.names2nodes(goal.args))
    except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
        return False


def make_example(
    example_id: str,
    seed: int,
    full_problem: pr.Problem,
    goal: pr.Construction,
    setup: list[pr.Dependency],
    aux_setup: list[pr.Dependency],
    log: list[tuple[list[pr.Dependency], list[pr.Dependency]]],
    setup_points: set[object],
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    require_unsolved_without_aux: bool,
    max_level: int,
    timeout: int,
) -> dict[str, object] | None:
    hidden = hidden_point_names(aux_setup, setup_points)
    if not hidden:
        return None

    visible_clauses, aux_clauses, dropped_clauses, removed_points = (
        split_visible_aux_and_dropped(full_problem.clauses, hidden)
    )
    if not aux_clauses:
        return None

    if any(arg in removed_points for arg in goal.args):
        return None

    if not solved_with_auxiliary(
        visible_clauses, aux_clauses, goal, definitions, rules, max_level, timeout
    ):
        return None

    if require_unsolved_without_aux and solved_without_auxiliary(
        visible_clauses, goal, definitions, rules, max_level, timeout
    ):
        return None

    return {
        'id': example_id,
        'seed': seed,
        'visible_problem': problem_txt(visible_clauses, goal),
        'target_auxiliary': '; '.join(c.txt() for c in aux_clauses),
        'removed_dependent_clauses': [c.txt() for c in dropped_clauses],
        'full_problem': problem_txt(full_problem.clauses, goal),
        'full_setup': problem_txt(full_problem.clauses),
        'goal': goal.txt(),
        'hidden_points': sorted(hidden),
        'setup_dependencies': deps_to_txt(setup),
        'auxiliary_dependencies': deps_to_txt(aux_setup),
        'proof_steps': proof_log_to_txt(log),
    }


def generate_examples(
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
    require_unsolved_without_aux: bool,
) -> list[dict[str, object]]:
    definitions, rules = load_defs_rules()
    rng = random.Random(seed)
    np.random.seed(seed)
    examples = []

    for attempt in range(max_attempts):
        if len(examples) >= num_examples:
            break

        attempt_seed = seed + attempt
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
            )
            g, added = run_ddar(full_problem, definitions, rules, max_level, timeout)
        except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
            continue

        candidates = enumerate_candidate_goals(g, added, rng, max_candidates)
        for goal in candidates:
            traced = traceback_goal(g, goal)
            if traced is None:
                continue
            setup, aux_setup, log, setup_points = traced
            example = make_example(
                f'{seed}-{attempt}-{len(examples)}',
                attempt_seed,
                full_problem,
                goal,
                setup,
                aux_setup,
                log,
                setup_points,
                definitions,
                rules,
                require_unsolved_without_aux,
                max_level,
                timeout,
            )
            if example is None:
                continue
            examples.append(example)
            break

    return examples


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate AlphaGeometry-style auxiliary-construction JSONL data.'
    )
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_attempts', type=int, default=200)
    parser.add_argument('--min_steps', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=7)
    parser.add_argument('--per_step_attempts', type=int, default=25)
    parser.add_argument('--max_level', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--max_candidates', type=int, default=250)
    parser.add_argument('--curated_rate', type=float, default=0.15)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--allow_solved_without_aux',
        action='store_true',
        help='Keep examples even if DD+AR can solve the visible problem without the auxiliary.',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.disable(logging.ERROR)
    examples = generate_examples(
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
        require_unsolved_without_aux=not args.allow_solved_without_aux,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, sort_keys=True) + '\n')

    print(f'wrote {len(examples)} examples to {args.out}')
    if len(examples) < args.num_examples:
        print(
            f'warning: requested {args.num_examples}, found {len(examples)}; '
            'increase --max_attempts or --curated_rate for a denser smoke run.'
        )
    return 0 if examples else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
