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
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import logging
import random
import signal
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import geometry as gm
import numericals as nm

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

ROOT_CONSTRUCTIONS = {
    'eq_quadrangle',
    'eq_trapezoid',
    'eqdia_quadrangle',
    'free',
    'ieq_triangle',
    'iso_triangle',
    'isquare',
    'pentagon',
    'quadrangle',
    'r_trapezoid',
    'r_triangle',
    'rectangle',
    'risos',
    'segment',
    'trapezoid',
    'triangle',
    'triangle12',
}


def construction_names_for_set(
    definitions: dict[str, pr.Definition], construction_set: str
) -> list[str]:
    if construction_set == 'conservative':
        return [name for name in RANDOM_CONSTRUCTIONS if name in definitions]
    if construction_set in {'expanded', 'all'}:
        return sorted(
            name
            for name, cdef in definitions.items()
            if name not in ROOT_CONSTRUCTIONS and cdef.points and cdef.args
        )
    raise ValueError(f'unknown construction set: {construction_set}')


def root_construction_names_for_set(
    definitions: dict[str, pr.Definition], construction_set: str
) -> list[str]:
    if construction_set == 'all':
        return sorted(name for name in ROOT_CONSTRUCTIONS if name in definitions)
    return ['triangle']

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

# `enumerate_candidate_goals` only special-cases GOAL_PREDICATES with dedicated
# graph scans (line/circle/length-class enumeration); predicates outside that
# set are still picked up opportunistically from `added` (facts DD+AR actually
# derived), just without the extra combinatorial expansion. `data/rules.txt`
# also concludes eqangle(6)/eqratio(3/6)/simtri(2)/contri(2) rules, all of
# which DD+AR already derives as intermediate proof steps -- widening the goal
# set to include them gives full theorem/proof pretraining data (see
# generate_pretraining_data.py) many more distinct (construction, goal)
# skeletons instead of collapsing everything onto these 6 predicates. This
# stays separate from GOAL_PREDICATES because the aux-construction fine-tuning
# path (this module) hides the goal's defining point and needs
# `translate_constrained_to_constructive` (src/alphageometry.py) to invert the
# chosen goal back into a construction; that function does not yet know how to
# invert eqratio/simtri/contri.
PRETRAIN_GOAL_PREDICATES = GOAL_PREDICATES | {
    'contri',
    'contri2',
    'eqangle',
    'eqangle6',
    'eqratio',
    'eqratio3',
    'eqratio6',
    'simtri',
    'simtri2',
}


def attempt_seed_for(seed: int, attempt: int) -> int:
    """Decorrelated per-attempt seed.

    The old `seed + attempt` scheme made parallel workers with consecutive base
    seeds walk the same attempt-seed sequence shifted by one, so almost every
    diagram was generated by several workers at once.  Hashing the pair gives
    every (worker, attempt) an independent stream while staying reproducible.
    """
    digest = hashlib.blake2b(
        f'{seed}:{attempt}'.encode(), digest_size=4
    ).digest()
    return int.from_bytes(digest, 'big')


class AttemptTimeout(BaseException):
    """Raised by SIGALRM when one attempt exceeds its wall-clock budget.

    Inherits BaseException so the broad `except Exception` guards inside the
    mining loop cannot swallow it.
    """


def _alarm_handler(signum, frame):  # pylint: disable=unused-argument
    raise AttemptTimeout()


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


def root_construction_clause(
    name: str, cdef: pr.Definition
) -> tuple[pr.Clause, list[str], int] | None:
    outputs = POINT_NAMES[:len(cdef.points)]
    if len(outputs) < len(cdef.points):
        return None
    mapping = dict(zip(cdef.points, outputs))
    args = [mapping.get(arg, arg) for arg in cdef.construction.args]
    return pr.Clause(outputs, [pr.Construction(name, args)]), outputs, len(outputs)


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
    construction_set: str = 'conservative',
) -> pr.Problem:
    root_names = root_construction_names_for_set(definitions, construction_set)
    root_name = rng.choice(root_names)
    root_clause = root_construction_clause(root_name, definitions[root_name])
    if root_clause is None:
        clauses = [base_triangle()]
        existing = ['a', 'b', 'c']
        next_name_index = 3
    else:
        clause, existing, next_name_index = root_clause
        clauses = [clause]

    problem = make_problem(clauses)
    g, added = gh.Graph.build_problem(problem, definitions, verbose=False)
    for dep in added:
        g.add_algebra(dep, level=0)

    plevel = g.plevel
    target_steps = rng.randint(min_steps, max_steps)
    construction_names = construction_names_for_set(definitions, construction_set)

    for _ in range(target_steps):
        for _attempt in range(per_step_attempts):
            name = rng.choice(construction_names)
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
    construction_set: str = 'conservative',
) -> pr.Problem:
    if rng.random() < curated_rate:
        return pr.Problem.from_txt(rng.choice(CURATED_FULL_PROBLEMS), translate=False)
    return sample_random_problem(
        definitions, rng, min_steps, max_steps, per_step_attempts, construction_set
    )


@dataclass
class ClosureResult:
    """One DD+AR diagram closure plus an auditable termination reason."""

    graph: gh.Graph = field(repr=False)
    added: list[pr.Dependency] = field(repr=False)
    status: str
    saturated: bool
    levels: int
    level_seconds: float

    def to_json(self) -> dict[str, object]:
        return {
            'status': self.status,
            'saturated': self.saturated,
            'levels': self.levels,
            'level_seconds': round(self.level_seconds, 3),
        }


def run_ddar(
    problem: pr.Problem,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    timeout: int,
) -> ClosureResult:
    """Build one diagram and run DD+AR, preserving how exploration stopped."""
    g, added = gh.Graph.build_problem(problem, definitions, verbose=False)
    controller = pr.Problem(url='', clauses=problem.clauses, goal=None)
    g, level_times, _solver_status, _branches, inferred = ddar.solve(
        g, rules, controller, max_level=max_level, timeout=timeout
    )
    hit_level_limit = len(level_times) >= max_level
    hit_timeout = bool(level_times and level_times[-1] > timeout)
    if hit_timeout:
        status = 'ddar_timeout'
    elif hit_level_limit:
        status = 'level_limit'
    else:
        status = 'saturated'
    return ClosureResult(
        graph=g,
        added=added + inferred,
        status=status,
        saturated=status == 'saturated',
        levels=len(level_times),
        level_seconds=sum(level_times),
    )


def as_goal(name: str, args: Iterable[object]) -> pr.Construction:
    return pr.Construction(name, [node_name(a) for a in args])


def goal_holds_numerically(g: gh.Graph, goal: pr.Construction) -> bool:
    """Check a mined symbolic conclusion against the diagram realization.

    Unsupported numerical predicates are rejected.  This guard is deliberately
    independent of ``g.check``: the latter only says that DD+AR inserted the
    symbolic fact and cannot catch an inconsistent or degenerate derivation.
    """
    try:
        args = [g.get(arg, lambda: int(arg)) for arg in goal.args]
        result = nm.check(goal.name, args)
        return result is not None and bool(result)
    except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
        return False


def orphan_point_count(goal_args: list[str], g: gh.Graph) -> int:
    """Number of diagram points not in the transitive rely_on closure of the goal args.

    A goal can only require an auxiliary construction if at least one point exists
    outside its dependency closure — that orphan is what a proof might route through.
    Scoring candidates by this count lets us prioritise the ones with any chance of
    yielding an aux-requiring proof, and drop those with score 0 outright.
    """
    try:
        goal_pts = set(g.names2nodes(goal_args))
    except (KeyError, ValueError, AttributeError):
        return 0
    closure = set(goal_pts)
    stack = list(goal_pts)
    while stack:
        p = stack.pop()
        for q in getattr(p, 'rely_on', None) or []:
            if q not in closure:
                closure.add(q)
                stack.append(q)
    return sum(1 for p in g.all_points() if p not in closure)


def _capped_combinations(
    items: list, r: int, cap: int, rng: random.Random
) -> Iterable[tuple]:
    combos = list(itertools.combinations(items, r))
    if len(combos) > cap:
        combos = rng.sample(combos, cap)
    return combos


def enumerate_candidate_goals(
    g: gh.Graph,
    added: list[pr.Dependency],
    rng: random.Random,
    max_candidates: int,
    rank_by_orphan: bool = False,
    diagnostics: dict[str, int] | None = None,
    goal_predicates: set[str] | None = None,
) -> list[pr.Construction]:
    """Enumerate true facts of the closure as candidate goals.

    Facts are read off the graph's own structures (lines, circles, merged
    length/direction value classes) plus the dependency list from DD+AR,
    instead of re-checking all O(n^3)/O(n^4) point combinations.  Each family
    is capped so a single dense structure cannot flood the candidate list.
    """
    goal_predicates = goal_predicates or GOAL_PREDICATES
    candidates: list[pr.Construction] = []
    seen: set[tuple[str, ...]] = set()
    per_family_cap = max(max_candidates, 1) * 4

    def maybe_add(name: str, args: Iterable[object]) -> None:
        if name not in goal_predicates:
            return
        goal = as_goal(name, args)
        key = pr.hashed_txt(goal.name, goal.args)
        if key in seen:
            return
        seen.add(key)
        candidates.append(goal)

    # Facts explicitly constructed or derived by DD+AR (any predicate).
    for dep in added:
        if all(hasattr(arg, 'name') for arg in dep.args):
            maybe_add(dep.name, dep.args)

    # coll: every line node knows its points.
    if 'coll' in goal_predicates:
        for line in g.type2nodes[gm.Line]:
            pts = line.neighbors(gm.Point)
            if len(pts) >= 3:
                for combo in _capped_combinations(pts, 3, per_family_cap, rng):
                    maybe_add('coll', combo)

    # cyclic: every circle node knows its points.
    if 'cyclic' in goal_predicates:
        for circle in g.type2nodes[gm.Circle]:
            pts = circle.neighbors(gm.Point)
            if len(pts) >= 4:
                for combo in _capped_combinations(pts, 4, per_family_cap, rng):
                    maybe_add('cyclic', combo)

    # cong: segments sharing a merged length value class.
    if 'cong' in goal_predicates:
        for length in g.type2nodes[gm.Length]:
            if length.rep() is not length:
                continue
            segs = length.neighbors(gm.Segment)
            if len(segs) >= 2:
                for s1, s2 in _capped_combinations(segs, 2, per_family_cap, rng):
                    a, b = s1.points
                    c, d = s2.points
                    maybe_add('cong', [a, b, c, d])

    # Lines grouped by merged direction value class.  Every point-pair
    # representation of a line pair is a distinct goal statement with its own
    # dependency closure, so all (capped) representations are enumerated —
    # some representations need an auxiliary point where others do not.
    val2lines: dict[object, list] = {}
    for line in g.type2nodes[gm.Line]:
        if line.val is not None:
            val2lines.setdefault(line.val, []).append(line)

    def emit_line_pair_goals(name: str, l1, l2, budget: list[int]) -> None:
        pts1 = _capped_combinations(l1.neighbors(gm.Point), 2, 21, rng)
        pts2 = _capped_combinations(l2.neighbors(gm.Point), 2, 21, rng)
        for a, b in pts1:
            for c, d in pts2:
                if {a, b} == {c, d}:
                    continue
                if budget[0] <= 0:
                    return
                budget[0] -= 1
                maybe_add(name, [a, b, c, d])

    # para: distinct lines in the same direction class.
    if 'para' in goal_predicates:
        budget = [per_family_cap]
        for lines in val2lines.values():
            if len(lines) < 2:
                continue
            for l1, l2 in itertools.combinations(lines, 2):
                if l1.rep() is l2.rep():
                    continue
                emit_line_pair_goals('para', l1, l2, budget)

    # perp: direction classes connected by an angle equal to its opposite
    # (that is how check_perp answers True).  These facts largely come from
    # the algebraic closure and never appear as explicit deps in `added`.
    if 'perp' in goal_predicates:
        budget = [per_family_cap]
        seen_dir_pairs: set[tuple[object, object]] = set()
        for ang in g.type2nodes[gm.Angle]:
            opp = getattr(ang, 'opposite', None)
            if opp is None:
                continue
            try:
                if not g.is_equal(ang, opp):
                    continue
                d1, d2 = ang.directions
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if d1 is None or d2 is None:
                continue
            d1, d2 = d1.rep(), d2.rep()
            if (d1, d2) in seen_dir_pairs or (d2, d1) in seen_dir_pairs:
                continue
            seen_dir_pairs.add((d1, d2))
            for l1 in val2lines.get(d1, []):
                for l2 in val2lines.get(d2, []):
                    emit_line_pair_goals('perp', l1, l2, budget)

    # midp: collinear triples with the cong check, from line structures.
    if 'midp' in goal_predicates:
        emitted = 0
        for m, a, b in g.all_midps():
            maybe_add('midp', [m, a, b])
            emitted += 1
            if emitted >= per_family_cap:
                break

    rng.shuffle(candidates)
    if diagnostics is not None:
        diagnostics['candidates_enumerated_total'] = (
            diagnostics.get('candidates_enumerated_total', 0) + len(candidates)
        )
    if rank_by_orphan:
        scored = [(orphan_point_count(list(c.args), g), c) for c in candidates]
        dropped_zero = sum(1 for s, _ in scored if s == 0)
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda sc: -sc[0])
        candidates = [c for _, c in scored]
        if diagnostics is not None:
            diagnostics['candidates_dropped_zero_orphan'] = (
                diagnostics.get('candidates_dropped_zero_orphan', 0) + dropped_zero
            )
    kept = candidates[:max_candidates]
    if diagnostics is not None:
        diagnostics['candidates_kept'] = (
            diagnostics.get('candidates_kept', 0) + len(kept)
        )
    return kept


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


def _bump(diagnostics: dict[str, int] | None, key: str) -> None:
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + 1


def aux_step_fraction(
    log: list[tuple[list[pr.Dependency], list[pr.Dependency]]],
    hidden: set[str],
) -> float:
    """Fraction of proof steps whose premises/conclusions touch a hidden point.

    A cheap centrality signal: 1.0 means the auxiliary point is used in every
    deduction, values near 0 mean it appears in only one or two steps.  Kept in
    the JSONL row so aux-central examples can be selected without re-tracing.
    """
    if not log:
        return 0.0
    touching = 0
    for prems, cons in log:
        for dep in list(prems) + list(cons):
            if any(node_name(arg) in hidden for arg in dep.args):
                touching += 1
                break
    return touching / len(log)


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
    diagnostics: dict[str, int] | None = None,
    verify_with_aux_rate: float = 0.0,
    rng: random.Random | None = None,
) -> dict[str, object] | None:
    """Turn one traced goal into a training row.

    Acceptance follows the AlphaGeometry definition: the example is valid iff
    the traceback of a true conclusion routes through a point outside the
    goal's dependency closure.  Re-solving the problem to verify is only done
    as sampled QA (verify_with_aux_rate) or when the caller explicitly asks
    for aux-essential examples (require_unsolved_without_aux) — both re-solves
    cost a full DD+AR run each and dominated generation time before.
    """
    hidden = hidden_point_names(aux_setup, setup_points)
    if not hidden:
        _bump(diagnostics, 'rejected_hidden_empty')
        return None

    visible_clauses, aux_clauses, dropped_clauses, removed_points = (
        split_visible_aux_and_dropped(full_problem.clauses, hidden)
    )
    if not aux_clauses:
        _bump(diagnostics, 'rejected_no_aux_clauses')
        return None

    if any(arg in removed_points for arg in goal.args):
        _bump(diagnostics, 'rejected_goal_in_removed_points')
        return None

    verified_with_aux = None
    if verify_with_aux_rate > 0 and rng is not None:
        if rng.random() < verify_with_aux_rate:
            verified_with_aux = solved_with_auxiliary(
                visible_clauses, aux_clauses, goal, definitions, rules,
                max_level, timeout,
            )
            _bump(diagnostics, 'qa_verified_with_aux_checked')
            if not verified_with_aux:
                _bump(diagnostics, 'rejected_solved_with_aux_failed')
                return None

    if require_unsolved_without_aux and solved_without_auxiliary(
        visible_clauses, goal, definitions, rules, max_level, timeout
    ):
        _bump(diagnostics, 'rejected_solved_without_aux')
        return None

    if diagnostics is not None:
        for clause in aux_clauses:
            for construction in clause.constructions:
                _bump(diagnostics, f'auxtype_{construction.name}')
        _bump(diagnostics, f'goalpred_{goal.name}')

    return {
        'aux_step_fraction': round(aux_step_fraction(log, hidden), 3),
        'num_aux_deps': len(aux_setup),
        'verified_with_aux': verified_with_aux,
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
    max_examples_per_problem: int = 1,
    construction_set: str = 'conservative',
    rank_by_orphan: bool = False,
    diagnostics: dict[str, int] | None = None,
    verify_with_aux_rate: float = 0.0,
    attempt_time_budget: int = 0,
) -> list[dict[str, object]]:
    return list(
        iter_examples(
            num_examples=num_examples,
            seed=seed,
            max_attempts=max_attempts,
            min_steps=min_steps,
            max_steps=max_steps,
            per_step_attempts=per_step_attempts,
            max_level=max_level,
            timeout=timeout,
            max_candidates=max_candidates,
            curated_rate=curated_rate,
            require_unsolved_without_aux=require_unsolved_without_aux,
            max_examples_per_problem=max_examples_per_problem,
            construction_set=construction_set,
            rank_by_orphan=rank_by_orphan,
            diagnostics=diagnostics,
            verify_with_aux_rate=verify_with_aux_rate,
            attempt_time_budget=attempt_time_budget,
        )
    )


def _mine_attempt(
    out: list[dict[str, object]],
    attempt: int,
    attempt_seed: int,
    seed: int,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    rng: random.Random,
    seen_examples: set[tuple[object, object]],
    num_found: int,
    num_examples: int,
    min_steps: int,
    max_steps: int,
    per_step_attempts: int,
    max_level: int,
    timeout: int,
    max_candidates: int,
    curated_rate: float,
    require_unsolved_without_aux: bool,
    max_examples_per_problem: int,
    construction_set: str,
    rank_by_orphan: bool,
    diagnostics: dict[str, int] | None,
    verify_with_aux_rate: float,
) -> None:
    """Sample one diagram, saturate it, and mine accepted examples into `out`."""
    _bump(diagnostics, 'diagrams_sampled')
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
        _bump(diagnostics, 'diagrams_ddar_failed')
        return
    _bump(diagnostics, 'diagrams_ddar_ok')
    _bump(diagnostics, f'diagrams_closure_{closure.status}')
    g, added = closure.graph, closure.added

    candidates = enumerate_candidate_goals(
        g, added, rng, max_candidates,
        rank_by_orphan=rank_by_orphan,
        diagnostics=diagnostics,
    )
    for goal in candidates:
        if num_found + len(out) >= num_examples:
            break
        if (
            max_examples_per_problem > 0
            and len(out) >= max_examples_per_problem
        ):
            break

        traced = traceback_goal(g, goal)
        if traced is None:
            _bump(diagnostics, 'traceback_failed')
            continue
        _bump(diagnostics, 'traceback_ok')
        setup, aux_setup, log, setup_points = traced
        example = make_example(
            f'{seed}-{attempt}-{num_found + len(out)}',
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
            diagnostics=diagnostics,
            verify_with_aux_rate=verify_with_aux_rate,
            rng=rng,
        )
        if example is None:
            continue
        key = (example['visible_problem'], example['target_auxiliary'])
        if key in seen_examples:
            _bump(diagnostics, 'rejected_duplicate')
            continue
        seen_examples.add(key)
        _bump(diagnostics, 'accepted')
        out.append(example)


def iter_examples(
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
    max_examples_per_problem: int = 1,
    construction_set: str = 'conservative',
    rank_by_orphan: bool = False,
    diagnostics: dict[str, int] | None = None,
    verify_with_aux_rate: float = 0.0,
    attempt_time_budget: int = 0,
) -> Iterable[dict[str, object]]:
    definitions, rules = load_defs_rules()
    rng = random.Random(seed)
    np.random.seed(seed)
    num_found = 0
    seen_examples: set[tuple[object, object]] = set()

    use_alarm = attempt_time_budget > 0 and hasattr(signal, 'SIGALRM')
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)

    for attempt in range(max_attempts):
        if num_found >= num_examples:
            break

        attempt_seed = attempt_seed_for(seed, attempt)
        rng.seed(attempt_seed)
        np.random.seed(attempt_seed)

        found_this_attempt: list[dict[str, object]] = []
        if use_alarm:
            signal.alarm(attempt_time_budget)
        try:
            _mine_attempt(
                found_this_attempt,
                attempt,
                attempt_seed,
                seed,
                definitions,
                rules,
                rng,
                seen_examples,
                num_found,
                num_examples,
                min_steps,
                max_steps,
                per_step_attempts,
                max_level,
                timeout,
                max_candidates,
                curated_rate,
                require_unsolved_without_aux,
                max_examples_per_problem,
                construction_set,
                rank_by_orphan,
                diagnostics,
                verify_with_aux_rate,
            )
        except AttemptTimeout:
            # Heavy-tail diagram: keep whatever was mined before the alarm.
            _bump(diagnostics, 'diagrams_time_budget_exceeded')
        finally:
            if use_alarm:
                signal.alarm(0)

        for example in found_this_attempt:
            num_found += 1
            yield example


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate AlphaGeometry-style auxiliary-construction JSONL data.'
    )
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_attempts', type=int, default=200)
    parser.add_argument(
        '--min_steps', type=int, default=8,
        help='Deeper diagrams (8-12 steps) yield far more auxiliary-bearing '
             'proofs than shallow ones; at 3-7 steps almost every traceback '
             'stays inside the goal closure.',
    )
    parser.add_argument('--max_steps', type=int, default=12)
    parser.add_argument('--per_step_attempts', type=int, default=25)
    parser.add_argument('--max_level', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=10)
    parser.add_argument('--max_candidates', type=int, default=250)
    parser.add_argument(
        '--max_examples_per_problem',
        type=int,
        default=16,
        help='Maximum accepted goals to emit per sampled full problem. The '
             'DD+AR closure is the expensive step, so amortising it over many '
             'goals is the main yield lever. Use 0 for no cap.',
    )
    parser.add_argument('--curated_rate', type=float, default=0.15)
    parser.add_argument(
        '--construction_set',
        choices=['conservative', 'expanded', 'all'],
        default='conservative',
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--flush_every',
        type=int,
        default=1,
        help='Flush the JSONL output after this many accepted examples.',
    )
    parser.add_argument(
        '--require_unsolved_without_aux',
        action='store_true',
        help='Only keep examples whose visible problem DD+AR cannot solve without '
             'the auxiliary. Costs one extra full DD+AR solve per candidate; off '
             'by default (matches the AlphaGeometry paper pipeline). Prefer '
             'post-filtering on aux_step_fraction instead.',
    )
    parser.add_argument(
        '--verify_with_aux_rate',
        type=float,
        default=0.05,
        help='Fraction of accepted examples to re-verify by re-building and '
             're-solving visible+aux (sampled QA). 0 disables.',
    )
    parser.add_argument(
        '--attempt_time_budget',
        type=int,
        default=180,
        help='Hard wall-clock cap in seconds for one diagram attempt (sampling, '
             'DD+AR and mining), enforced with SIGALRM. The per-level DD timeout '
             'does not bound theorem matching inside a level, so heavy-tail '
             'diagrams can otherwise run for many minutes. 0 disables.',
    )
    parser.add_argument(
        '--rank_by_orphan',
        action='store_true',
        help='Rank candidate goals by orphan-point count and drop score-zero goals '
             '(those whose args transitively cover the whole diagram cannot need aux).',
    )
    parser.add_argument(
        '--diagnostics',
        action='store_true',
        help='Track per-stage counters (diagrams, candidates, rejections) and write '
             'them to <out>.diagnostics.json alongside the JSONL output.',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.disable(logging.ERROR)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    diagnostics: dict[str, int] | None = {} if args.diagnostics else None
    start_time = time.time()
    with args.out.open('w', encoding='utf-8') as f:
        for example in iter_examples(
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
            require_unsolved_without_aux=args.require_unsolved_without_aux,
            max_examples_per_problem=args.max_examples_per_problem,
            construction_set=args.construction_set,
            rank_by_orphan=args.rank_by_orphan,
            diagnostics=diagnostics,
            verify_with_aux_rate=args.verify_with_aux_rate,
            attempt_time_budget=args.attempt_time_budget,
        ):
            f.write(json.dumps(example, sort_keys=True) + '\n')
            num_written += 1
            if args.flush_every > 0 and num_written % args.flush_every == 0:
                f.flush()
        f.flush()

    print(f'wrote {num_written} examples to {args.out}')
    if num_written < args.num_examples:
        print(
            f'warning: requested {args.num_examples}, found {num_written}; '
            'increase --max_attempts or --curated_rate for a denser smoke run.'
        )
    if diagnostics is not None:
        diagnostics['elapsed_seconds'] = round(time.time() - start_time, 2)
        diagnostics['num_written'] = num_written
        diag_path = args.out.with_suffix(args.out.suffix + '.diagnostics.json')
        with diag_path.open('w', encoding='utf-8') as df:
            json.dump(diagnostics, df, sort_keys=True, indent=2)
        print(f'diagnostics -> {diag_path}')
        for k in sorted(diagnostics):
            print(f'  {k}: {diagnostics[k]}')
    return 0 if num_written else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
