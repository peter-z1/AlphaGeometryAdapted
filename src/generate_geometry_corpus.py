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
import copy
import gzip
import itertools
import json
import logging
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

import ddar
import geometry as gm
import problem as pr
from generate_pretraining_data import (
    balanced_candidate_goals,
    is_useful_goal,
    make_pretraining_example,
)
from generate_synthetic_data import (
    FOCUS_BRIDGE_MODES,
    FOCUS_DEPENDENCY_MODES,
    FOCUS_PLACEMENTS,
    GOAL_PREDICATES,
    PRETRAIN_GOAL_PREDICATES,
    ROOT_POLICIES,
    AttemptTimeout,
    ClosureResult,
    add_clause_safely,
    attempt_seed_for,
    choose_full_problem,
    construction_names_for_set,
    enumerate_candidate_goals,
    goal_holds_numerically,
    load_defs_rules,
    make_example,
    problem_construction_names,
    problem_root_construction,
    run_ddar,
    traceback_goal,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / 'outputs' / 'synthetic_data' / 'geometry_corpus_v2'


def auxiliary_construction_names(text: str) -> list[str]:
    """Parse the construction recipes used by an auxiliary target."""
    names: list[str] = []
    for part in text.split(';'):
        part = part.strip()
        if not part:
            continue
        clause = pr.Clause.from_txt(part)
        names.extend(construction.name for construction in clause.constructions)
    return names


def construction_signature(names: list[str]) -> str:
    """Canonical family label for one possibly-compound auxiliary action."""
    return '+'.join(sorted(names))


def select_mined_auxiliary_rows(
    rows: list[dict[str, object]],
    maximum: int,
    focus_construction: str | None,
) -> list[dict[str, object]]:
    """Prefer proof-relevant focus rows, then round-robin goal predicates.

    This is a bounded reservoir policy, not a hard focus filter.  If a focused
    diagram produces no row whose hidden action contains the requested
    constructor, its other valid candidates remain available.
    """
    if maximum <= 0 or len(rows) <= maximum:
        return rows

    selected: list[dict[str, object]] = []
    for matching_focus in (True, False):
        grouped: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
        for row in rows:
            names = row.get('target_construction_names', [])
            matches = bool(
                focus_construction is not None and focus_construction in names
            )
            if matches == matching_focus:
                grouped[str(row['goal']).split()[0]].append(row)

        goals = sorted(grouped)
        while goals and len(selected) < maximum:
            next_goals = []
            for goal in goals:
                if len(selected) >= maximum:
                    break
                values = grouped[goal]
                if values:
                    selected.append(values.pop(0))
                if values:
                    next_goals.append(goal)
            goals = next_goals
    return selected


def counterfactual_target_indices(
    problem: pr.Problem,
    focus_construction: str | None,
    maximum: int,
    rng: random.Random,
) -> list[int]:
    """Choose extension clauses to remove for closure-difference mining.

    The first clause is the root configuration, so it is not a useful hidden
    action.  A focused run considers only clauses that actually contain the
    requested constructor.  Unfocused runs sample extension clauses so that a
    large diagram does not trigger an unbounded number of DD+AR closures.
    """
    indices = [
        index
        for index, clause in enumerate(problem.clauses)
        if index > 0
        and clause.points
        and (
            focus_construction is None
            or any(
                construction.name == focus_construction
                for construction in clause.constructions
            )
        )
    ]
    rng.shuffle(indices)
    if maximum > 0:
        indices = indices[:maximum]
    return indices


def resample_focus_target_clauses(
    target_clause: pr.Clause,
    visible_clauses: list[pr.Clause],
    focus_construction: str | None,
    definitions: dict[str, pr.Definition],
    proposal_seed: int,
    maximum: int,
) -> list[pr.Clause]:
    """Enumerate deterministic, unique placements for one focus action.

    The original clause is always first.  Remaining visible point names are
    shuffled once with ``proposal_seed`` and distinct input assignments are
    then visited in ``itertools.permutations`` order.  Consequently a smaller
    cap is a prefix of a larger run with the same seed, and a cap at least as
    large as the permutation space exhausts every syntactically legal
    distinct-input placement.  Exact construction and closure checks below
    remain the semantic acceptance criterion.
    """
    proposals = [target_clause]
    if maximum <= 1 or focus_construction is None:
        return proposals
    if len(target_clause.constructions) != 1:
        return proposals
    original = target_clause.constructions[0]
    if original.name != focus_construction:
        return proposals
    cdef = definitions.get(focus_construction)
    if cdef is None or len(cdef.points) != len(target_clause.points):
        return proposals

    existing: list[str] = []
    for clause in visible_clauses:
        for point in clause.points:
            if point not in existing:
                existing.append(point)
    input_vars = list(cdef.args)
    if len(input_vars) > len(existing):
        return proposals

    ordered_existing = list(existing)
    random.Random(proposal_seed).shuffle(ordered_existing)
    seen = {target_clause.txt()}
    for inputs in itertools.permutations(ordered_existing, len(input_vars)):
        mapping = dict(zip(cdef.points, target_clause.points))
        mapping.update(zip(input_vars, inputs))
        try:
            args = [mapping[value] for value in cdef.construction.args]
        except KeyError:
            return proposals
        candidate = pr.Clause(
            list(target_clause.points),
            [pr.Construction(focus_construction, args)],
        )
        key = candidate.txt()
        if key in seen:
            continue
        seen.add(key)
        proposals.append(candidate)
        if len(proposals) >= maximum:
            break
    return proposals


def split_counterfactual_visible_clauses(
    clauses: list[pr.Clause], target_index: int
) -> tuple[list[pr.Clause], list[pr.Clause], set[str]]:
    """Remove one target clause and every construction depending on it.

    Removing descendants is important: otherwise the supposedly visible
    problem can still contain a later clause whose inputs refer to the hidden
    point.  The returned hidden set therefore contains the target outputs and
    the outputs of all transitively dependent clauses.
    """
    if not 0 <= target_index < len(clauses):
        raise IndexError(target_index)

    hidden = set(clauses[target_index].points)
    removed_indices = {target_index}
    changed = True
    while changed:
        changed = False
        for index, clause in enumerate(clauses):
            if index in removed_indices:
                continue
            mentions_hidden = any(point in hidden for point in clause.points)
            if not mentions_hidden:
                mentions_hidden = any(
                    argument in hidden
                    for construction in clause.constructions
                    for argument in construction.args
                )
            if mentions_hidden:
                removed_indices.add(index)
                before = len(hidden)
                hidden.update(clause.points)
                changed = changed or len(hidden) != before

    visible = [
        clause for index, clause in enumerate(clauses)
        if index not in removed_indices
    ]
    removed = [
        clause for index, clause in enumerate(clauses)
        if index in removed_indices
    ]
    return visible, removed, hidden


def _dependency_mentions_points(
    dependencies: list[pr.Dependency], point_names: set[str]
) -> bool:
    return any(
        getattr(argument, 'name', str(argument)) in point_names
        for dependency in dependencies
        for argument in dependency.args
    )


def _trace_mentions_points(
    traced: tuple[
        list[pr.Dependency],
        list[pr.Dependency],
        list[tuple[list[pr.Dependency], list[pr.Dependency]]],
        set[object],
    ],
    point_names: set[str],
) -> bool:
    """Whether the restored proof actually passes through the target output."""
    _setup, aux_setup, log, _setup_points = traced
    if _dependency_mentions_points(aux_setup, point_names):
        return True
    return any(
        _dependency_mentions_points(premises + conclusions, point_names)
        for premises, conclusions in log
    )


def _closure_proves(
    closure: ClosureResult, goal: pr.Construction
) -> bool | None:
    """Return whether a saturated closure proves ``goal``, or None on lookup error."""
    try:
        return bool(
            closure.graph.check(
                goal.name, closure.graph.names2nodes(goal.args)
            )
        )
    except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
        return None


def extend_saturated_visible_closure(
    visible_closure: ClosureResult,
    visible_clauses: list[pr.Clause],
    target_clause: pr.Clause,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    max_level: int,
    timeout: int,
    numeric_rng_state: tuple | None = None,
) -> tuple[pr.Problem, ClosureResult] | None:
    """Add one target to a saturated visible graph and continue DD+AR.

    The returned problem is deliberately canonical: all visible clauses keep
    their order and the hidden target is last.  Restoring NumPy's post-visible
    RNG state makes its numerical realization identical to a fresh build of
    that same canonical problem, while the deep copy avoids rebuilding and
    re-saturating the visible prefix for every proposal.
    """
    restored_problem = pr.Problem(
        url='', clauses=list(visible_clauses) + [target_clause], goal=None
    )
    graph = copy.deepcopy(visible_closure.graph)
    if numeric_rng_state is not None:
        np.random.set_state(numeric_rng_state)
    ok, construction_added, plevel = add_clause_safely(
        graph, target_clause, graph.plevel, definitions
    )
    if not ok:
        return None
    graph.plevel = plevel
    graph, level_times, _status, _branches, inferred = ddar.solve(
        graph,
        rules,
        restored_problem,
        max_level=max_level,
        timeout=timeout,
    )
    hit_level_limit = len(level_times) >= max_level
    hit_timeout = bool(level_times and level_times[-1] > timeout)
    if hit_timeout:
        status = 'ddar_timeout'
    elif hit_level_limit:
        status = 'level_limit'
    else:
        status = 'saturated'
    closure = ClosureResult(
        graph=graph,
        added=list(visible_closure.added) + construction_added + inferred,
        status=status,
        saturated=status == 'saturated',
        levels=len(level_times),
        level_seconds=sum(level_times),
    )
    return restored_problem, closure


def bridge_potential_score(
    visible_closure: ClosureResult,
    target_clause: pr.Clause,
    definitions: dict[str, pr.Definition],
    numeric_rng_state: tuple | None = None,
) -> int | None:
    """Cheap graph-channel score before an exact DD+AR trial.

    A proposal scores when a newly constructed point attaches to geometric
    objects already incident with visible points.  Extra credit is given for
    two visible incidences and for sharing a direction/length/etc. value with
    a visible object.  This is only a ranking heuristic; exact saturation and
    closure-difference checks remain the acceptance test.
    """
    graph = copy.deepcopy(visible_closure.graph)
    if numeric_rng_state is not None:
        np.random.set_state(numeric_rng_state)
    ok, _added, plevel = add_clause_safely(
        graph, target_clause, graph.plevel, definitions
    )
    if not ok:
        return None
    graph.plevel = plevel
    visible_names = {point.name for point in visible_closure.graph.all_points()}
    channels: set[tuple[str, tuple[str, ...]]] = set()
    value_channels: set[tuple[str, tuple[str, ...]]] = set()
    for point_name in target_clause.points:
        point = graph.get(point_name, lambda: None)
        if point is None:
            continue
        for relation in point.neighbors(None):
            if isinstance(relation, gm.Point):
                continue
            old_points = tuple(sorted({
                p.name for p in relation.neighbors(gm.Point)
                if p.name in visible_names
            }))
            if old_points:
                channels.add((type(relation).__name__, old_points))
            value = getattr(relation, 'val', None)
            if value is None:
                continue
            linked_old: set[str] = set()
            for peer in value.neighbors(type(relation)):
                linked_old.update(
                    p.name for p in peer.neighbors(gm.Point)
                    if p.name in visible_names
                )
            if linked_old:
                value_channels.add(
                    (type(value).__name__, tuple(sorted(linked_old)))
                )
    score = sum(1 + min(2, len(points)) for _kind, points in channels)
    score += 2 * len(value_channels)
    score += max(0, len(channels) + len(value_channels) - 1)
    return score


def rank_counterfactual_proposals(
    proposals: list[pr.Clause],
    visible_closure: ClosureResult,
    definitions: dict[str, pr.Definition],
    numeric_rng_state: tuple | None,
    maximum: int,
) -> list[tuple[int, pr.Clause, int]]:
    """Rank valid proposals deterministically by cheap bridge potential."""
    ranked: list[tuple[int, str, int, pr.Clause]] = []
    for index, clause in enumerate(proposals):
        score = bridge_potential_score(
            visible_closure, clause, definitions, numeric_rng_state
        )
        if score is not None:
            ranked.append((-score, clause.txt(), index, clause))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    if maximum > 0:
        ranked = ranked[:maximum]
    return [(index, clause, -negative) for negative, _text, index, clause in ranked]


def mine_counterfactual_auxiliary_rows(
    diagram_id: str,
    attempt_seed: int,
    full_problem: pr.Problem,
    full_closure: ClosureResult | None,
    definitions: dict[str, pr.Definition],
    rules: dict[str, pr.Theorem],
    rng: random.Random,
    focus_construction: str | None,
    max_targets: int,
    focus_trials: int,
    max_candidates: int,
    max_per_predicate: int,
    min_proof_steps: int,
    max_level: int,
    timeout: int,
    counts: collections.Counter,
    proposal_pool: int = 0,
    exact_trials: int = 0,
    auxiliary_goal_predicates: set[str] | None = None,
) -> list[dict[str, object]]:
    """Mine goals in restored closure minus a target-hidden visible closure.

    Unlike traceback-only mining, this performs the counterfactual before a
    row is emitted.  For each selected action we remove it and all of its
    construction descendants, saturate that visible diagram, then restore
    *only that one target clause* and saturate again.  Candidate goals and
    their tracebacks come from this minimal restored closure, not from the
    original full diagram (which also contains the removed descendants).
    The later strict filter remains the authoritative rebuild/minimization QA
    stage.
    """
    auxiliary_goal_predicates = set(
        auxiliary_goal_predicates or GOAL_PREDICATES
    )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    target_indices = counterfactual_target_indices(
        full_problem, focus_construction, max_targets, rng
    )
    counts['counterfactual_targets_available'] += len(target_indices)
    if focus_construction is not None and not target_indices:
        counts['counterfactual_rejected_no_focus_target'] += 1

    proposal_pool = proposal_pool or focus_trials
    exact_trials = exact_trials or focus_trials
    target_specs: list[tuple[int, int, pr.Clause, int]] = []
    visible_cache: dict[
        int,
        tuple[
            list[pr.Clause],
            list[pr.Clause],
            set[str],
            ClosureResult,
            int,
            tuple,
        ],
    ] = {}
    for target_index in target_indices:
        counts['counterfactual_targets_considered'] += 1
        visible_clauses, removed_clauses, removed_points = (
            split_counterfactual_visible_clauses(
                full_problem.clauses, target_index
            )
        )
        counts['counterfactual_dependent_clauses_removed'] += (
            len(removed_clauses) - 1
        )
        rebuild_seed = attempt_seed_for(attempt_seed, target_index)
        proposals = resample_focus_target_clauses(
            full_problem.clauses[target_index],
            visible_clauses,
            focus_construction,
            definitions,
            rebuild_seed,
            proposal_pool,
        )
        counts['counterfactual_proposals_generated'] += len(proposals)

        counts['counterfactual_visible_closures_attempted'] += 1
        try:
            # All proposals share this one visible closure.  Restored trials
            # reuse its numerical seed, so every shared point has the same
            # realization on both sides of the counterfactual.
            np.random.seed(rebuild_seed)
            visible_closure = run_ddar(
                pr.Problem(url='', clauses=visible_clauses, goal=None),
                definitions,
                rules,
                max_level,
                timeout,
            )
            numeric_rng_state = np.random.get_state()
        except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
            counts['counterfactual_visible_closure_error'] += 1
            counts['counterfactual_focus_trials_skipped_visible_failure'] += (
                len(proposals)
            )
            continue
        counts[
            f'counterfactual_visible_status_{visible_closure.status}'
        ] += 1
        if not visible_closure.saturated:
            counts['counterfactual_rejected_visible_not_saturated'] += 1
            counts['counterfactual_focus_trials_skipped_visible_failure'] += (
                len(proposals)
            )
            continue
        counts['counterfactual_visible_closures_saturated'] += 1
        visible_cache[target_index] = (
            visible_clauses,
            removed_clauses,
            removed_points,
            visible_closure,
            rebuild_seed,
            numeric_rng_state,
        )
        ranked = rank_counterfactual_proposals(
            proposals,
            visible_closure,
            definitions,
            numeric_rng_state,
            exact_trials,
        )
        counts['counterfactual_focus_trials_proposed'] += len(ranked)
        counts['counterfactual_bridge_positive'] += sum(
            score > 0 for _trial, _clause, score in ranked
        )
        target_specs.extend(
            (target_index, trial_index, clause, score)
            for trial_index, clause, score in ranked
        )

    for target_index, target_trial, target_clause, bridge_score in target_specs:
        counts['counterfactual_focus_trials_considered'] += 1
        target_points = set(target_clause.points)
        target_names = {
            construction.name for construction in target_clause.constructions
        }
        (
            visible_clauses,
            removed_clauses,
            removed_points,
            visible_closure,
            rebuild_seed,
            numeric_rng_state,
        ) = visible_cache[target_index]
        try:
            # Exact trials deliberately use a canonical fresh build.  A graph
            # extended after visible saturation is logically equivalent, but
            # it has fewer redundant relation objects/dependencies; those are
            # part of candidate enumeration and traceback semantics today.
            restored_problem = pr.Problem(
                url='', clauses=list(visible_clauses) + [target_clause], goal=None
            )
            np.random.seed(rebuild_seed)
            restored_closure = run_ddar(
                restored_problem,
                definitions,
                rules,
                max_level,
                timeout,
            )
        except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
            counts['counterfactual_restored_closure_error'] += 1
            continue
        counts[
            f'counterfactual_restored_status_{restored_closure.status}'
        ] += 1
        if not restored_closure.saturated:
            counts['counterfactual_rejected_restored_not_saturated'] += 1
            continue
        counts['counterfactual_restored_closures_saturated'] += 1

        raw_candidates = enumerate_candidate_goals(
            restored_closure.graph,
            restored_closure.added,
            rng,
            max_candidates,
            goal_predicates=auxiliary_goal_predicates,
        )
        candidates = balanced_candidate_goals(
            raw_candidates, rng, max_per_predicate
        )
        for candidate_index, goal in enumerate(candidates):
            counts['counterfactual_candidates_considered'] += 1
            if any(str(argument) in removed_points for argument in goal.args):
                counts['counterfactual_rejected_goal_mentions_removed'] += 1
                continue
            if not goal_holds_numerically(restored_closure.graph, goal):
                counts['counterfactual_rejected_numerical'] += 1
                continue

            visible_proves = _closure_proves(visible_closure, goal)
            if visible_proves is None:
                counts['counterfactual_rejected_visible_lookup_error'] += 1
                continue
            if visible_proves:
                counts['counterfactual_rejected_visible_solved'] += 1
                continue
            counts['counterfactual_closure_difference_goals'] += 1

            traced = traceback_goal(restored_closure.graph, goal)
            if traced is None:
                counts['counterfactual_rejected_traceback'] += 1
                continue
            _setup, _aux_setup, log, _setup_points = traced
            if len(log) < min_proof_steps:
                counts['counterfactual_rejected_short_proof'] += 1
                continue
            if not _trace_mentions_points(traced, target_points):
                counts['counterfactual_rejected_trace_omits_target'] += 1
                continue
            counts['counterfactual_traces_using_target'] += 1

            setup, aux_setup, log, setup_points = traced
            row = make_example(
                f'{diagram_id}-counterfactual-{target_index}-'
                f'{target_trial}-{candidate_index}',
                attempt_seed,
                restored_problem,
                goal,
                setup,
                aux_setup,
                log,
                setup_points,
                definitions,
                rules,
                require_unsolved_without_aux=False,
                max_level=max_level,
                timeout=timeout,
                diagnostics=counts,
            )
            if row is None:
                counts['counterfactual_rejected_row_conversion'] += 1
                continue

            row_hidden = set(row.get('hidden_points', []))
            if not target_points.issubset(row_hidden):
                counts['counterfactual_rejected_target_not_hidden'] += 1
                continue
            names = auxiliary_construction_names(str(row['target_auxiliary']))
            if not target_names.issubset(set(names)):
                counts['counterfactual_rejected_target_not_in_action'] += 1
                continue
            if str(row['target_auxiliary']) != target_clause.txt():
                counts['counterfactual_rejected_nonminimal_target'] += 1
                continue
            if focus_construction is not None and focus_construction not in names:
                counts['counterfactual_rejected_focus_not_in_action'] += 1
                continue

            signature = construction_signature(names)
            key = (
                str(row['visible_problem']),
                str(row['target_auxiliary']),
                str(row['goal']),
            )
            if key in seen:
                counts['counterfactual_rejected_duplicate'] += 1
                continue
            seen.add(key)
            row['target_construction_names'] = names
            row['target_construction_signature'] = signature
            row['focus_construction'] = focus_construction
            row['source'] = 'counterfactual_closure_difference_candidate'
            row['auxiliary_mining_mode'] = 'counterfactual'
            row['diagram_id'] = diagram_id
            row['numerically_verified'] = True
            row['closure'] = restored_closure.to_json()
            if full_closure is not None:
                row['source_diagram_closure'] = full_closure.to_json()
            row['counterfactual_verified'] = True
            row['counterfactual_target_clause'] = target_clause.txt()
            row['counterfactual_target_index'] = target_index
            row['counterfactual_target_trial'] = target_trial
            row['counterfactual_bridge_score'] = bridge_score
            row['counterfactual_rebuild_seed'] = rebuild_seed
            row['counterfactual_proposal_seed'] = rebuild_seed
            row['counterfactual_removed_clause_count'] = len(removed_clauses)
            row['counterfactual_visible_closure'] = visible_closure.to_json()
            row['counterfactual_restored_closure'] = restored_closure.to_json()
            rows.append(row)
            counts['counterfactual_candidates_mined'] += 1
            if focus_construction is not None:
                counts['auxiliary_candidates_mined_matching_focus'] += 1

    return rows


def _alarm_handler(signum, frame):  # pylint: disable=unused-argument
    raise AttemptTimeout()


def mine_diagram_rows(
    diagram_id: str,
    attempt_seed: int,
    attempt: int,
    full_problem: pr.Problem,
    closure: ClosureResult | None,
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
    focus_construction: str | None = None,
    require_focus_in_auxiliary: bool = False,
    focus_scan_candidates: int = 160,
    auxiliary_mining_mode: str = 'traceback',
    counterfactual_max_targets: int = 4,
    counterfactual_focus_trials: int = 1,
    max_level: int = 1000,
    ddar_timeout: int = 10,
    counterfactual_proposal_pool: int = 0,
    counterfactual_exact_trials: int = 0,
    auxiliary_goal_predicates: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], collections.Counter]:
    """Mine theorem rows and either traceback or closure-difference aux rows."""
    if auxiliary_mining_mode not in {'traceback', 'counterfactual'}:
        raise ValueError(f'unknown auxiliary mining mode: {auxiliary_mining_mode}')
    auxiliary_goal_predicates = set(
        auxiliary_goal_predicates or GOAL_PREDICATES
    )
    counts: collections.Counter[str] = collections.Counter()
    pretraining_rows: list[dict[str, object]] = []
    auxiliary_pool: list[dict[str, object]] = []

    if emit_pretraining or auxiliary_mining_mode == 'traceback':
        if closure is None:
            raise ValueError('source closure is required for this mining mode')
        candidate_goal_predicates = (
            PRETRAIN_GOAL_PREDICATES
            if emit_pretraining
            else auxiliary_goal_predicates
        )
        raw_candidates = enumerate_candidate_goals(
            closure.graph,
            closure.added,
            rng,
            max_candidates,
            goal_predicates=candidate_goal_predicates,
        )
        candidates = balanced_candidate_goals(
            raw_candidates, rng, max_per_predicate
        )
    else:
        # Counterfactual candidates are enumerated from each target-only
        # restored closure below.  Scanning the original full closure cannot
        # contribute an auxiliary row and is pure overhead in auxiliary-only
        # production jobs.
        candidates = []
        counts['counterfactual_skipped_source_closure_scan'] += 1
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
            row['root_construction'] = problem_root_construction(full_problem)
            pretraining_rows.append(row)
            counts['pretraining_emitted'] += 1

        auxiliary_full = (
            max_auxiliary_per_diagram > 0
            and len(auxiliary_pool) >= max_auxiliary_per_diagram
        )
        focus_scan_active = bool(
            focus_construction is not None
            and counts['candidates_considered'] <= focus_scan_candidates
        )
        if (
            auxiliary_mining_mode == 'traceback'
            and goal.name in auxiliary_goal_predicates
            and (not auxiliary_full or focus_scan_active)
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
                max_level=max_level,
                timeout=ddar_timeout,
                diagnostics=counts,
            )
            if row is not None:
                names = auxiliary_construction_names(str(row['target_auxiliary']))
                signature = construction_signature(names)
                row['target_construction_names'] = names
                row['target_construction_signature'] = signature
                row['focus_construction'] = focus_construction
                row['auxiliary_mining_mode'] = 'traceback'
                row['counterfactual_verified'] = False
                if focus_construction is not None and focus_construction in names:
                    counts['auxiliary_candidates_mined_matching_focus'] += 1
                elif require_focus_in_auxiliary:
                    counts['auxiliary_candidates_rejected_focus_mismatch'] += 1
                    continue
                row['source'] = 'dependency_difference_candidate'
                row['diagram_id'] = diagram_id
                row['numerically_verified'] = True
                row['closure'] = closure.to_json()
                auxiliary_pool.append(row)
                counts['auxiliary_candidates_mined'] += 1

        pretraining_full = (
            not emit_pretraining
            or (
                max_pretraining_per_diagram > 0
                and len(pretraining_rows) >= max_pretraining_per_diagram
            )
        )
        auxiliary_full = (
            max_auxiliary_per_diagram > 0
            and len(auxiliary_pool) >= max_auxiliary_per_diagram
        )
        focus_scan_complete = (
            focus_construction is None
            or counts['candidates_considered'] >= focus_scan_candidates
        )
        if (
            auxiliary_mining_mode == 'traceback'
            and pretraining_full
            and auxiliary_full
            and focus_scan_complete
        ):
            break
        if auxiliary_mining_mode == 'counterfactual' and pretraining_full:
            break

    if auxiliary_mining_mode == 'counterfactual':
        auxiliary_pool.extend(
            mine_counterfactual_auxiliary_rows(
                diagram_id,
                attempt_seed,
                full_problem,
                closure,
                definitions,
                rules,
                rng,
                focus_construction,
                counterfactual_max_targets,
                counterfactual_focus_trials,
                max_candidates,
                max_per_predicate,
                min_proof_steps,
                max_level,
                ddar_timeout,
                counts,
                counterfactual_proposal_pool,
                counterfactual_exact_trials,
                auxiliary_goal_predicates,
            )
        )
        counts['auxiliary_candidates_mined'] += len(auxiliary_pool)

    auxiliary_rows = select_mined_auxiliary_rows(
        auxiliary_pool, max_auxiliary_per_diagram, focus_construction
    )
    for row in auxiliary_rows:
        names = list(row['target_construction_names'])
        signature = str(row['target_construction_signature'])
        for name in set(names):
            counts[f'auxiliary_candidate_type_{name}'] += 1
        counts[f'auxiliary_candidate_signature_{signature}'] += 1
        if focus_construction is not None and focus_construction in names:
            counts['auxiliary_candidates_matching_focus'] += 1
        counts['auxiliary_candidates_emitted'] += 1
    counts['auxiliary_candidates_dropped_by_diagram_cap'] += (
        len(auxiliary_pool) - len(auxiliary_rows)
    )
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
    parser.add_argument(
        '--auxiliary_goal_predicates',
        default='',
        help='Comma-separated auxiliary goal predicates. Empty uses the '
             'default auxiliary goal set.',
    )
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
        '--focus_constructions',
        default='',
        help='Comma-separated constructor families to target round-robin. Each '
             'accepted targeted diagram is guaranteed to contain its requested family.',
    )
    parser.add_argument(
        '--focus_attempts',
        type=int,
        default=100,
        help='Argument proposals reserved for realizing a requested family.',
    )
    parser.add_argument(
        '--focus_dependency_rate',
        type=float,
        default=0.75,
        help='After the focus is built, probability that a new construction '
             'uses the focus point or one of its construction descendants.',
    )
    parser.add_argument(
        '--focus_placement',
        choices=FOCUS_PLACEMENTS,
        default='eager',
        help='eager preserves legacy insertion; last builds the visible '
             'scaffold first and reserves the final construction slot for '
             'the requested auxiliary family.',
    )
    parser.add_argument(
        '--focus_dependency_mode',
        choices=FOCUS_DEPENDENCY_MODES,
        default='mixed',
        help='mixed preserves legacy sampling after the focus is built; '
             'independent prevents later scaffold constructions from using '
             'the focus point or its descendants.',
    )
    parser.add_argument(
        '--focus_bridge_mode',
        choices=FOCUS_BRIDGE_MODES,
        default='none',
        help='none preserves the general sampler; empirical_v6 biases focused '
             'actions toward bridge patterns observed in genuine v6 rows.',
    )
    parser.add_argument(
        '--focus_scan_candidates',
        type=int,
        default=160,
        help='When focused, scan at least this many candidate goals before '
             'applying the per-diagram auxiliary cap.',
    )
    parser.add_argument(
        '--require_focus_in_auxiliary',
        action='store_true',
        help='Emit an auxiliary candidate only when its hidden target uses the '
             'diagram\'s requested focus constructor.',
    )
    parser.add_argument(
        '--auxiliary_mining_mode',
        choices=['traceback', 'counterfactual'],
        default='traceback',
        help='traceback preserves the original fast candidate miner; '
             'counterfactual retains only goals absent after the selected '
             'auxiliary action and its descendants are hidden.',
    )
    parser.add_argument(
        '--counterfactual_max_targets',
        type=int,
        default=4,
        help='Maximum target clauses whose visible DD+AR closure is computed '
             'per diagram in counterfactual mode; 0 means all. Focused runs '
             'consider only clauses containing the requested constructor.',
    )
    parser.add_argument(
        '--counterfactual_focus_trials',
        type=int,
        default=1,
        help='Maximum unique argument placements tested per focused target. '
             'The original is first; alternatives follow a deterministic '
             'distinct-input permutation order seeded by diagram and target.',
    )
    parser.add_argument(
        '--counterfactual_proposal_pool',
        type=int,
        default=0,
        help='Cheap placements scored per target; 0 inherits the legacy '
             '--counterfactual_focus_trials value.',
    )
    parser.add_argument(
        '--counterfactual_exact_trials',
        type=int,
        default=0,
        help='Top-ranked placements fully saturated per target; 0 inherits '
             'the legacy --counterfactual_focus_trials value.',
    )
    parser.add_argument(
        '--construction_set',
        choices=['conservative', 'expanded', 'all'],
        default='expanded',
    )
    parser.add_argument(
        '--root_policy',
        choices=ROOT_POLICIES,
        default='triangle',
        help='Initial-configuration sampler, independent of --construction_set. '
             'Use diversified for the audited weighted root mixture.',
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
    if args.focus_scan_candidates < 1:
        raise ValueError('--focus_scan_candidates must be positive')
    if args.counterfactual_max_targets < 0:
        raise ValueError('--counterfactual_max_targets must be nonnegative')
    if args.counterfactual_focus_trials < 1:
        raise ValueError('--counterfactual_focus_trials must be positive')
    if args.counterfactual_proposal_pool < 0:
        raise ValueError('--counterfactual_proposal_pool must be nonnegative')
    if args.counterfactual_exact_trials < 0:
        raise ValueError('--counterfactual_exact_trials must be nonnegative')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pretraining_path = args.out_dir / (
        'pretraining.jsonl.gz' if args.gzip_pretraining else 'pretraining.jsonl'
    )
    auxiliary_path = args.out_dir / 'auxiliary_candidates.jsonl'
    audit_path = args.out_dir / 'audit.json'

    definitions, rules = load_defs_rules()
    auxiliary_goal_predicates = {
        value.strip()
        for value in args.auxiliary_goal_predicates.split(',')
        if value.strip()
    } or set(GOAL_PREDICATES)
    unknown_goals = sorted(auxiliary_goal_predicates - GOAL_PREDICATES)
    if unknown_goals:
        raise ValueError(
            'unsupported --auxiliary_goal_predicates: '
            + ', '.join(unknown_goals)
        )
    focus_constructions = [
        value.strip()
        for value in args.focus_constructions.split(',')
        if value.strip()
    ]
    eligible = set(construction_names_for_set(definitions, args.construction_set))
    unknown_focus = sorted(set(focus_constructions) - eligible)
    if unknown_focus:
        raise ValueError(
            'ineligible --focus_constructions: ' + ', '.join(unknown_focus)
        )
    if args.require_focus_in_auxiliary and not focus_constructions:
        raise ValueError('--require_focus_in_auxiliary needs --focus_constructions')
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
                focus_construction = (
                    focus_constructions[(args.seed + attempt) % len(focus_constructions)]
                    if focus_constructions
                    else None
                )
                if focus_construction is not None:
                    counts[f'focus_requested_{focus_construction}'] += 1
                full_problem = choose_full_problem(
                    definitions,
                    rng,
                    args.min_steps,
                    args.max_steps,
                    args.per_step_attempts,
                    args.curated_rate,
                    args.construction_set,
                    focus_construction,
                    args.focus_attempts,
                    args.focus_dependency_rate,
                    counts,
                    args.root_policy,
                    args.focus_placement,
                    args.focus_dependency_mode,
                    args.focus_bridge_mode,
                )
                if focus_construction is not None:
                    if focus_construction not in problem_construction_names(full_problem):
                        raise AssertionError('focused diagram omitted requested constructor')
                    counts[f'focus_diagram_built_{focus_construction}'] += 1
                needs_source_closure = not (
                    args.auxiliary_mining_mode == 'counterfactual'
                    and args.skip_pretraining
                )
                closure = None
                if needs_source_closure:
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
                else:
                    counts['source_closure_skipped_counterfactual_aux_only'] += 1

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
                    focus_construction,
                    args.require_focus_in_auxiliary,
                    args.focus_scan_candidates,
                    args.auxiliary_mining_mode,
                    args.counterfactual_max_targets,
                    args.counterfactual_focus_trials,
                    args.max_level,
                    args.ddar_timeout,
                    args.counterfactual_proposal_pool,
                    args.counterfactual_exact_trials,
                    auxiliary_goal_predicates,
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
