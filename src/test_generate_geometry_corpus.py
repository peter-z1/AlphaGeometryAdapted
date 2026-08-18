"""Tests for shared-closure geometry corpus generation."""

from __future__ import annotations

import collections
import json
import random
import tempfile
import unittest
from unittest import mock

import numpy as np

import problem as pr
from generate_geometry_corpus import (
    bridge_potential_score,
    counterfactual_target_indices,
    extend_saturated_visible_closure,
    main,
    mine_diagram_rows,
    parse_args,
    rank_counterfactual_proposals,
    resample_focus_target_clauses,
    select_mined_auxiliary_rows,
    split_counterfactual_visible_clauses,
)
from generate_synthetic_data import (
    CURATED_FULL_PROBLEMS,
    GOAL_PREDICATES,
    enumerate_candidate_goals,
    goal_holds_numerically,
    load_defs_rules,
    run_ddar,
)


class GenerateGeometryCorpusTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.definitions, cls.rules = load_defs_rules()

    def test_closure_reports_saturation(self):
        np.random.seed(7)
        problem = pr.Problem.from_txt(CURATED_FULL_PROBLEMS[0], translate=False)
        closure = run_ddar(problem, self.definitions, self.rules, 1000, 10)

        self.assertTrue(closure.saturated)
        self.assertEqual(closure.status, 'saturated')
        self.assertGreater(closure.levels, 0)
        self.assertEqual(closure.to_json()['saturated'], True)

    def test_focus_dependency_mode_cli_and_main_pass_through(self):
        self.assertEqual(parse_args([]).focus_dependency_mode, 'mixed')
        self.assertEqual(parse_args([]).focus_bridge_mode, 'none')
        self.assertEqual(parse_args([]).auxiliary_goal_predicates, '')
        self.assertEqual(
            parse_args([
                '--focus_dependency_mode', 'independent'
            ]).focus_dependency_mode,
            'independent',
        )
        self.assertEqual(
            parse_args([
                '--focus_bridge_mode', 'empirical_v6'
            ]).focus_bridge_mode,
            'empirical_v6',
        )
        self.assertEqual(
            parse_args([
                '--auxiliary_goal_predicates', 'eqangle,eqratio'
            ]).auxiliary_goal_predicates,
            'eqangle,eqratio',
        )
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; d = on_line d a b',
            translate=False,
        )
        with tempfile.TemporaryDirectory() as out_dir, mock.patch(
            'generate_geometry_corpus.load_defs_rules',
            return_value=(self.definitions, self.rules),
        ), mock.patch(
            'generate_geometry_corpus.choose_full_problem',
            return_value=problem,
        ) as choose_mock, mock.patch(
            'generate_geometry_corpus.mine_diagram_rows',
            return_value=([], [], collections.Counter()),
        ):
            result = main([
                '--num_diagrams', '1',
                '--max_attempts', '1',
                '--attempt_time_budget', '0',
                '--skip_pretraining',
                '--auxiliary_mining_mode', 'counterfactual',
                '--focus_constructions', 'on_line',
                '--focus_dependency_mode', 'independent',
                '--focus_bridge_mode', 'empirical_v6',
                '--out_dir', out_dir,
            ])
            with open(  # pylint: disable=unspecified-encoding
                f'{out_dir}/audit.json', encoding='utf-8'
            ) as audit_handle:
                audit = json.load(audit_handle)

        self.assertEqual(result, 0)
        self.assertEqual(choose_mock.call_args.args[-2], 'independent')
        self.assertEqual(choose_mock.call_args.args[-1], 'empirical_v6')
        self.assertEqual(
            audit['config']['focus_dependency_mode'], 'independent'
        )
        self.assertEqual(
            audit['config']['focus_bridge_mode'], 'empirical_v6'
        )

    def test_numerical_validation_rejects_false_symbolic_shape(self):
        np.random.seed(11)
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; d = mirror d a b', translate=False
        )
        closure = run_ddar(problem, self.definitions, self.rules, 1000, 10)

        self.assertTrue(
            goal_holds_numerically(
                closure.graph, pr.Construction('coll', ['a', 'b', 'd'])
            )
        )
        self.assertFalse(
            goal_holds_numerically(
                closure.graph, pr.Construction('perp', ['b', 'a', 'b', 'd'])
            )
        )

    def test_auxiliary_goal_policy_enumerates_eqangle_and_eqratio(self):
        self.assertTrue({'eqangle', 'eqratio'} <= GOAL_PREDICATES)

        np.random.seed(7)
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; '
            'h = orthocenter h a b c; '
            'g1 g2 g3 g = centroid g1 g2 g3 g a b c; '
            'o = circle o a b c',
            translate=False,
        )
        closure = run_ddar(
            problem, self.definitions, self.rules, 1000, 10
        )

        for predicate in ('eqangle', 'eqratio'):
            with self.subTest(predicate=predicate):
                goals = enumerate_candidate_goals(
                    closure.graph,
                    [],
                    random.Random(7),
                    max_candidates=20,
                    goal_predicates={predicate},
                )
                self.assertTrue(goals)
                self.assertTrue(all(goal.name == predicate for goal in goals))
                self.assertTrue(
                    all(
                        goal_holds_numerically(closure.graph, goal)
                        for goal in goals
                    )
                )

    def test_one_closure_emits_both_row_types(self):
        seed = 19
        np.random.seed(seed)
        rng = random.Random(seed)
        problem = pr.Problem.from_txt(CURATED_FULL_PROBLEMS[0], translate=False)
        closure = run_ddar(problem, self.definitions, self.rules, 1000, 10)

        proof_rows, auxiliary_rows, counts = mine_diagram_rows(
            'curated-test',
            seed,
            0,
            problem,
            closure,
            self.definitions,
            self.rules,
            rng,
            'expanded',
            max_candidates=100,
            max_pretraining_per_diagram=100,
            max_auxiliary_per_diagram=100,
            max_per_predicate=100,
            min_proof_steps=1,
            allow_trivial_goals=True,
            text_mode='construction_proof',
        )

        self.assertTrue(proof_rows)
        self.assertTrue(auxiliary_rows)
        self.assertGreater(counts['candidates_verified'], 0)
        self.assertTrue(all(row['diagram_id'] == 'curated-test' for row in proof_rows))
        self.assertTrue(
            all(row['diagram_id'] == 'curated-test' for row in auxiliary_rows)
        )
        self.assertTrue(all(row['numerically_verified'] for row in proof_rows))
        self.assertTrue(all(row['numerically_verified'] for row in auxiliary_rows))
        self.assertTrue(all(row['closure']['saturated'] for row in proof_rows))

    def test_auxiliary_only_mode_skips_pretraining_rows(self):
        seed = 23
        np.random.seed(seed)
        rng = random.Random(seed)
        problem = pr.Problem.from_txt(CURATED_FULL_PROBLEMS[0], translate=False)
        closure = run_ddar(problem, self.definitions, self.rules, 1000, 10)

        proof_rows, auxiliary_rows, counts = mine_diagram_rows(
            'curated-aux-only',
            seed,
            0,
            problem,
            closure,
            self.definitions,
            self.rules,
            rng,
            'expanded',
            max_candidates=100,
            max_pretraining_per_diagram=100,
            max_auxiliary_per_diagram=100,
            max_per_predicate=100,
            min_proof_steps=1,
            allow_trivial_goals=True,
            text_mode='construction_proof',
            emit_pretraining=False,
        )

        self.assertEqual(proof_rows, [])
        self.assertTrue(auxiliary_rows)
        self.assertEqual(counts['pretraining_emitted'], 0)

    def test_mining_reservoir_prefers_focus_without_hard_filtering(self):
        rows = [
            {
                'id': 'ordinary-perp',
                'goal': 'perp a b c d',
                'target_construction_names': ['reflect'],
            },
            {
                'id': 'focused-perp',
                'goal': 'perp a b c d',
                'target_construction_names': ['midpoint'],
            },
            {
                'id': 'ordinary-para',
                'goal': 'para a b c d',
                'target_construction_names': ['foot'],
            },
        ]

        selected = select_mined_auxiliary_rows(rows, 2, 'midpoint')

        self.assertEqual(selected[0]['id'], 'focused-perp')
        self.assertEqual(len(selected), 2)

    def test_counterfactual_split_removes_construction_descendants(self):
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; '
            'd = midpoint d a b; '
            'e = on_line e d c; '
            'f = midpoint f a c',
            translate=False,
        )

        visible, removed, hidden = split_counterfactual_visible_clauses(
            problem.clauses, 1
        )

        self.assertEqual([clause.points for clause in visible], [
            ['a', 'b', 'c'], ['f']
        ])
        self.assertEqual([clause.points for clause in removed], [['d'], ['e']])
        self.assertEqual(hidden, {'d', 'e'})

    def test_focused_counterfactual_targets_only_requested_constructor(self):
        problem = pr.Problem.from_txt(
            CURATED_FULL_PROBLEMS[0], translate=False
        )

        indices = counterfactual_target_indices(
            problem, 'on_line', 4, random.Random(3)
        )

        self.assertEqual(indices, [2])

    def test_focus_trials_exhaust_legal_permutations_deterministically(self):
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; d = on_line d a b',
            translate=False,
        )
        proposals = resample_focus_target_clauses(
            problem.clauses[1],
            problem.clauses[:1],
            'on_line',
            self.definitions,
            proposal_seed=13,
            maximum=99,
        )

        self.assertEqual(proposals[0].txt(), problem.clauses[1].txt())
        self.assertEqual(len(proposals), 6)
        self.assertTrue(all(clause.points == ['d'] for clause in proposals))
        self.assertTrue(all(
            clause.constructions[0].name == 'on_line'
            for clause in proposals
        ))
        self.assertEqual(len({clause.txt() for clause in proposals}), 6)
        self.assertEqual(
            {
                tuple(clause.constructions[0].args[1:])
                for clause in proposals
            },
            {
                ('a', 'b'), ('a', 'c'), ('b', 'a'),
                ('b', 'c'), ('c', 'a'), ('c', 'b'),
            },
        )

        repeated = resample_focus_target_clauses(
            problem.clauses[1], problem.clauses[:1], 'on_line',
            self.definitions, proposal_seed=13, maximum=99,
        )
        capped = resample_focus_target_clauses(
            problem.clauses[1], problem.clauses[:1], 'on_line',
            self.definitions, proposal_seed=13, maximum=4,
        )
        self.assertEqual(
            [clause.txt() for clause in repeated],
            [clause.txt() for clause in proposals],
        )
        self.assertEqual(
            [clause.txt() for clause in capped],
            [clause.txt() for clause in proposals[:4]],
        )

    def test_counterfactual_visible_closure_runs_once_per_target(self):
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; d = on_line d a b',
            translate=False,
        )

        with mock.patch(
            'generate_geometry_corpus.run_ddar', wraps=run_ddar
        ) as run_mock:
            _proof_rows, _auxiliary_rows, counts = mine_diagram_rows(
                'visible-cache',
                19,
                0,
                problem,
                None,
                self.definitions,
                self.rules,
                random.Random(19),
                'expanded',
                max_candidates=50,
                max_pretraining_per_diagram=0,
                max_auxiliary_per_diagram=50,
                max_per_predicate=50,
                min_proof_steps=1,
                allow_trivial_goals=True,
                text_mode='construction_proof',
                emit_pretraining=False,
                focus_construction='on_line',
                auxiliary_mining_mode='counterfactual',
                counterfactual_max_targets=1,
                counterfactual_focus_trials=5,
            )

        solved_clause_counts = [
            len(call.args[0].clauses) for call in run_mock.call_args_list
        ]
        self.assertEqual(solved_clause_counts.count(1), 1)
        self.assertEqual(solved_clause_counts.count(2), 5)
        self.assertEqual(counts['counterfactual_targets_considered'], 1)
        self.assertEqual(counts['counterfactual_visible_closures_attempted'], 1)
        self.assertEqual(counts['counterfactual_focus_trials_proposed'], 5)
        self.assertEqual(counts['counterfactual_focus_trials_considered'], 5)

    def test_separate_proposal_and_exact_caps(self):
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; d = on_line d a b',
            translate=False,
        )
        with mock.patch(
            'generate_geometry_corpus.run_ddar', wraps=run_ddar
        ) as run_mock:
            _proof, _auxiliary, counts = mine_diagram_rows(
                'rank-caps', 19, 0, problem, None,
                self.definitions, self.rules, random.Random(19), 'expanded',
                max_candidates=50,
                max_pretraining_per_diagram=0,
                max_auxiliary_per_diagram=50,
                max_per_predicate=50,
                min_proof_steps=1,
                allow_trivial_goals=True,
                text_mode='construction_proof',
                emit_pretraining=False,
                focus_construction='on_line',
                auxiliary_mining_mode='counterfactual',
                counterfactual_max_targets=1,
                counterfactual_focus_trials=1,
                counterfactual_proposal_pool=6,
                counterfactual_exact_trials=2,
            )

        solved_clause_counts = [
            len(call.args[0].clauses) for call in run_mock.call_args_list
        ]
        self.assertEqual(solved_clause_counts.count(1), 1)
        self.assertEqual(solved_clause_counts.count(2), 2)
        self.assertEqual(counts['counterfactual_proposals_generated'], 6)
        self.assertEqual(counts['counterfactual_focus_trials_proposed'], 2)
        self.assertEqual(counts['counterfactual_focus_trials_considered'], 2)

    def test_exact_restore_places_an_earlier_target_last(self):
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; d = on_line d a b; '
            'e = midpoint e a c',
            translate=False,
        )
        with mock.patch(
            'generate_geometry_corpus.run_ddar', wraps=run_ddar
        ) as run_mock:
            mine_diagram_rows(
                'target-last', 29, 0, problem, None,
                self.definitions, self.rules, random.Random(29), 'expanded',
                max_candidates=20,
                max_pretraining_per_diagram=0,
                max_auxiliary_per_diagram=20,
                max_per_predicate=20,
                min_proof_steps=1,
                allow_trivial_goals=True,
                text_mode='construction_proof',
                emit_pretraining=False,
                focus_construction='on_line',
                auxiliary_mining_mode='counterfactual',
                counterfactual_max_targets=1,
            )

        restored = [
            call.args[0] for call in run_mock.call_args_list
            if len(call.args[0].clauses) == 3
        ]
        self.assertEqual(len(restored), 1)
        self.assertEqual(
            [clause.txt() for clause in restored[0].clauses],
            [problem.clauses[0].txt(), problem.clauses[2].txt(),
             problem.clauses[1].txt()],
        )

    def test_incremental_extension_matches_curated_fresh_known_fact(self):
        full = pr.Problem.from_txt(CURATED_FULL_PROBLEMS[0], translate=False)
        visible = full.clauses[:2]
        target = full.clauses[2]
        seed = 101
        np.random.seed(seed)
        visible_closure = run_ddar(
            pr.Problem(url='', clauses=visible, goal=None),
            self.definitions, self.rules, 1000, 10,
        )
        numeric_state = np.random.get_state()
        restored_problem, incremental = extend_saturated_visible_closure(
            visible_closure, visible, target,
            self.definitions, self.rules, 1000, 10, numeric_state,
        )
        np.random.seed(seed)
        fresh = run_ddar(
            pr.Problem(url='', clauses=visible + [target], goal=None),
            self.definitions, self.rules, 1000, 10,
        )

        self.assertEqual(
            [clause.txt() for clause in restored_problem.clauses],
            [clause.txt() for clause in visible] + [target.txt()],
        )
        self.assertTrue(incremental.saturated)
        self.assertTrue(fresh.saturated)
        goal = pr.Construction('perp', ['e', 'b', 'a', 'c'])
        for closure in (incremental, fresh):
            self.assertTrue(closure.graph.check(
                goal.name, closure.graph.names2nodes(goal.args)
            ))
        for name in ['a', 'b', 'c', 'd', 'e']:
            inc_num = incremental.graph.get(name, lambda: None).num
            fresh_num = fresh.graph.get(name, lambda: None).num
            self.assertAlmostEqual(inc_num.x, fresh_num.x)
            self.assertAlmostEqual(inc_num.y, fresh_num.y)

    def test_bridge_ranking_is_deterministic_and_prefers_known_bridge(self):
        visible_problem = pr.Problem.from_txt(
            'a b c = triangle a b c; '
            'd = on_tline d b a c, on_tline d c a b',
            translate=False,
        )
        np.random.seed(5)
        visible = run_ddar(
            visible_problem, self.definitions, self.rules, 1000, 10
        )
        numeric_state = np.random.get_state()
        bridge = pr.Clause.from_txt('e = on_line e b d')
        unrelated = pr.Clause.from_txt('e = on_line e b c')

        self.assertGreater(
            bridge_potential_score(
                visible, bridge, self.definitions, numeric_state
            ),
            bridge_potential_score(
                visible, unrelated, self.definitions, numeric_state
            ),
        )
        first = rank_counterfactual_proposals(
            [unrelated, bridge], visible, self.definitions, numeric_state, 2
        )
        second = rank_counterfactual_proposals(
            [unrelated, bridge], visible, self.definitions, numeric_state, 2
        )
        self.assertEqual(
            [(index, clause.txt(), score) for index, clause, score in first],
            [(index, clause.txt(), score) for index, clause, score in second],
        )
        self.assertEqual(first[0][1].txt(), bridge.txt())

    def test_counterfactual_miner_emits_closure_difference(self):
        seed = 19
        np.random.seed(seed)
        rng = random.Random(seed)
        problem = pr.Problem.from_txt(
            CURATED_FULL_PROBLEMS[0], translate=False
        )
        closure = run_ddar(problem, self.definitions, self.rules, 1000, 10)

        proof_rows, auxiliary_rows, counts = mine_diagram_rows(
            'curated-counterfactual',
            seed,
            0,
            problem,
            closure,
            self.definitions,
            self.rules,
            rng,
            'expanded',
            max_candidates=100,
            max_pretraining_per_diagram=100,
            max_auxiliary_per_diagram=100,
            max_per_predicate=100,
            min_proof_steps=1,
            allow_trivial_goals=True,
            text_mode='construction_proof',
            emit_pretraining=False,
            focus_construction='on_line',
            auxiliary_mining_mode='counterfactual',
            counterfactual_max_targets=4,
        )

        self.assertEqual(proof_rows, [])
        self.assertTrue(auxiliary_rows)
        self.assertGreater(counts['counterfactual_closure_difference_goals'], 0)
        self.assertTrue(
            all(row['counterfactual_verified'] for row in auxiliary_rows)
        )
        self.assertTrue(
            all(
                row['source'] == 'counterfactual_closure_difference_candidate'
                for row in auxiliary_rows
            )
        )
        self.assertTrue(
            all(
                'on_line' in row['target_construction_names']
                for row in auxiliary_rows
            )
        )

        for row in auxiliary_rows:
            visible = pr.Problem.from_txt(
                str(row['visible_problem']), translate=False
            )
            visible_closure = run_ddar(
                visible, self.definitions, self.rules, 1000, 10
            )
            self.assertTrue(visible_closure.saturated)
            self.assertFalse(
                visible_closure.graph.check(
                    visible.goal.name,
                    visible_closure.graph.names2nodes(visible.goal.args),
                )
            )

    def test_counterfactual_does_not_credit_target_for_descendant_goal(self):
        """A goal unlocked only by a removed descendant is not target-enabled."""
        seed = 19
        np.random.seed(seed)
        problem = pr.Problem.from_txt(
            'a b c = triangle a b c; '
            'd = on_tline d b a c, on_tline d c a b; '
            'x = on_line x a c; '
            'e = intersection_ll e a x b d',
            translate=False,
        )
        full_closure = run_ddar(
            problem, self.definitions, self.rules, 1000, 10
        )
        target_only = pr.Problem(
            url='', clauses=problem.clauses[:3], goal=None
        )
        target_only_closure = run_ddar(
            target_only, self.definitions, self.rules, 1000, 10
        )
        descendant_goal = pr.Construction('perp', ['d', 'a', 'b', 'c'])
        self.assertTrue(
            full_closure.graph.check(
                descendant_goal.name,
                full_closure.graph.names2nodes(descendant_goal.args),
            )
        )
        self.assertFalse(
            target_only_closure.graph.check(
                descendant_goal.name,
                target_only_closure.graph.names2nodes(descendant_goal.args),
            )
        )

        np.random.seed(seed)
        _proof_rows, auxiliary_rows, counts = mine_diagram_rows(
            'descendant-regression',
            seed,
            0,
            problem,
            full_closure,
            self.definitions,
            self.rules,
            random.Random(seed),
            'expanded',
            max_candidates=100,
            max_pretraining_per_diagram=0,
            max_auxiliary_per_diagram=100,
            max_per_predicate=100,
            min_proof_steps=1,
            allow_trivial_goals=True,
            text_mode='construction_proof',
            emit_pretraining=False,
            focus_construction='on_line',
            auxiliary_mining_mode='counterfactual',
            counterfactual_max_targets=4,
        )

        self.assertEqual(auxiliary_rows, [])
        self.assertEqual(counts['counterfactual_dependent_clauses_removed'], 1)
        self.assertEqual(counts['counterfactual_restored_closures_saturated'], 1)


if __name__ == '__main__':
    unittest.main()
