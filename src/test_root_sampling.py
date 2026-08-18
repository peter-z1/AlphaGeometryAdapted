"""Tests for independent, weighted root-configuration sampling."""

from __future__ import annotations

import random
import unittest

import numpy as np

from generate_geometry_corpus import parse_args as parse_corpus_args
from generate_synthetic_data import (
    DIVERSIFIED_ROOT_WEIGHTS,
    choose_root_construction,
    load_defs_rules,
    parse_args,
    problem_root_construction,
    sample_random_problem,
)


class RootSamplingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.definitions, _rules = load_defs_rules()

    def test_diversified_weights_match_audited_mixture(self):
        self.assertEqual(
            DIVERSIFIED_ROOT_WEIGHTS,
            {
                'triangle': 20,
                'quadrangle': 15,
                'pentagon': 10,
                'segment': 5,
                'iso_triangle': 5,
                'r_triangle': 5,
                'risos': 5,
                'ieq_triangle': 5,
                'triangle12': 5,
                'eq_quadrangle': 4,
                'eq_trapezoid': 4,
                'eqdia_quadrangle': 4,
                'rectangle': 4,
                'r_trapezoid': 3,
                'isquare': 3,
                'trapezoid': 3,
            },
        )
        self.assertEqual(sum(DIVERSIFIED_ROOT_WEIGHTS.values()), 100)
        self.assertNotIn('free', DIVERSIFIED_ROOT_WEIGHTS)

    def test_weighted_choice_is_deterministic(self):
        first = [
            choose_root_construction(
                self.definitions, 'diversified', random.Random(seed)
            )
            for seed in range(100)
        ]
        second = [
            choose_root_construction(
                self.definitions, 'diversified', random.Random(seed)
            )
            for seed in range(100)
        ]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 8)

    def test_default_root_is_triangle_even_for_all_extension_set(self):
        diagnostics: dict[str, int] = {}
        problem = sample_random_problem(
            self.definitions,
            random.Random(29),
            min_steps=0,
            max_steps=0,
            per_step_attempts=1,
            construction_set='all',
            diagnostics=diagnostics,
        )

        root = problem_root_construction(problem)
        self.assertEqual(root, 'triangle')
        self.assertEqual(diagnostics['root_policy_triangle'], 1)
        self.assertEqual(diagnostics['root_selected_triangle'], 1)

    def test_diversified_root_is_independent_of_extension_set(self):
        roots = []
        for construction_set in ('conservative', 'expanded', 'all'):
            problem = sample_random_problem(
                self.definitions,
                random.Random(53),
                min_steps=0,
                max_steps=0,
                per_step_attempts=1,
                construction_set=construction_set,
                root_policy='diversified',
            )
            roots.append(problem.clauses[0].constructions[0].name)

        self.assertEqual(len(set(roots)), 1)
        self.assertNotEqual(roots[0], 'free')

    def test_cli_defaults_to_triangle_and_accepts_diversified(self):
        self.assertEqual(parse_args([]).root_policy, 'triangle')
        self.assertEqual(
            parse_args(['--root_policy', 'diversified']).root_policy,
            'diversified',
        )
        self.assertEqual(parse_corpus_args([]).root_policy, 'triangle')
        self.assertEqual(
            parse_corpus_args(['--root_policy', 'diversified']).root_policy,
            'diversified',
        )

    def test_last_focus_placement_builds_scaffold_before_target(self):
        diagnostics: dict[str, int] = {}
        problem = sample_random_problem(
            self.definitions,
            random.Random(71),
            min_steps=5,
            max_steps=5,
            per_step_attempts=30,
            construction_set='expanded',
            focus_construction='midpoint',
            focus_attempts=100,
            diagnostics=diagnostics,
            root_policy='diversified',
            focus_placement='last',
        )

        clause_names = [
            [construction.name for construction in clause.constructions]
            for clause in problem.clauses
        ]
        self.assertIn('midpoint', clause_names[-1])
        self.assertTrue(all('midpoint' not in names for names in clause_names[:-1]))
        self.assertGreater(len(problem.clauses), 2)
        self.assertEqual(diagnostics['focus_placement_last'], 1)

    def test_independent_focus_mode_builds_only_independent_later_clauses(self):
        diagnostics: dict[str, int] = {}
        post_focus_clauses = 0

        for seed in range(24):
            problem = sample_random_problem(
                self.definitions,
                random.Random(seed),
                min_steps=7,
                max_steps=7,
                per_step_attempts=30,
                construction_set='expanded',
                focus_construction='midpoint',
                focus_attempts=100,
                diagnostics=diagnostics,
                focus_placement='eager',
                focus_dependency_mode='independent',
            )

            focus_index = next(
                index
                for index, clause in enumerate(problem.clauses)
                if any(
                    construction.name == 'midpoint'
                    for construction in clause.constructions
                )
            )
            focus_descendants = set(problem.clauses[focus_index].points)
            for clause in problem.clauses[focus_index + 1:]:
                referenced_points = {
                    argument
                    for construction in clause.constructions
                    for argument in construction.args
                }
                depends_on_focus = bool(referenced_points & focus_descendants)
                if depends_on_focus:
                    focus_descendants.update(clause.points)
                self.assertFalse(
                    depends_on_focus,
                    f'{clause.txt()} depends on the focus in {problem.txt()}',
                )
                post_focus_clauses += 1

        self.assertGreater(post_focus_clauses, 48)
        self.assertEqual(diagnostics['focus_dependent_commits'], 0)
        self.assertEqual(diagnostics['focus_dependency_violations'], 0)
        self.assertEqual(diagnostics['focus_dependent_commit_midpoint'], 0)
        self.assertEqual(diagnostics['focus_dependency_violation_midpoint'], 0)
        self.assertGreater(
            diagnostics.get('focus_independent_postfocus_commits', 0), 48
        )

    def test_empirical_bridge_is_visible_and_independent(self):
        # Graph construction uses NumPy for its numerical realization while
        # clause selection uses the explicit Python RNG below. Seed both so
        # bridge feasibility is stable across fresh test processes.
        np.random.seed(0)
        diagnostics: dict[str, int] = {}
        problem = sample_random_problem(
            self.definitions,
            random.Random(0),
            min_steps=7,
            max_steps=7,
            per_step_attempts=30,
            construction_set='expanded',
            focus_construction='foot',
            focus_attempts=100,
            focus_dependency_rate=0,
            diagnostics=diagnostics,
            focus_placement='eager',
            focus_dependency_mode='independent',
            focus_bridge_mode='empirical_v6',
        )

        names = [
            [construction.name for construction in clause.constructions]
            for clause in problem.clauses
        ]
        focus_index = next(i for i, values in enumerate(names) if 'foot' in values)
        self.assertIn('orthocenter', names[focus_index + 1])
        hidden = set(problem.clauses[focus_index].points)
        bridge_arguments = {
            argument
            for construction in problem.clauses[focus_index + 1].constructions
            for argument in construction.args
        }
        self.assertTrue(hidden.isdisjoint(bridge_arguments))
        self.assertEqual(
            diagnostics['focus_bridge_requested_foot_to_orthocenter'], 1
        )
        self.assertEqual(
            diagnostics['focus_bridge_committed_foot_to_orthocenter'], 1
        )
        self.assertEqual(diagnostics['focus_dependent_commits'], 0)
        self.assertEqual(diagnostics['focus_dependency_violations'], 0)


if __name__ == '__main__':
    unittest.main()
