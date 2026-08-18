"""Tests for targeted generation and diversity-aware auxiliary selection."""

from __future__ import annotations

import collections
import math
import random
import unittest

from generate_synthetic_data import (
    load_defs_rules,
    problem_construction_names,
    sample_random_problem,
)
from select_diverse_auxiliary import (
    construction_family,
    goal_predicate,
    select_balanced_rows,
)


def _row(index: int, construction: str, goal: str = 'perp') -> dict[str, object]:
    return {
        'id': str(index),
        'visible_problem': f'a b c = triangle a b c ? {goal} a b b c',
        'target_auxiliary': f'd = {construction} d a b',
        'goal': f'{goal} a b b c',
    }


class SelectDiverseAuxiliaryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.definitions, _rules = load_defs_rules()

    def test_focused_sampler_commits_requested_construction(self):
        diagnostics: dict[str, int] = {}
        problem = sample_random_problem(
            self.definitions,
            random.Random(17),
            min_steps=3,
            max_steps=3,
            per_step_attempts=10,
            construction_set='expanded',
            focus_construction='midpoint',
            focus_attempts=20,
            diagnostics=diagnostics,
        )

        self.assertIn('midpoint', problem_construction_names(problem))
        self.assertEqual(diagnostics['focus_committed_midpoint'], 1)

    def test_selection_round_robins_families_before_dominant_tail(self):
        rows = (
            [_row(i, 'reflect') for i in range(10)]
            + [_row(100 + i, 'midpoint') for i in range(2)]
            + [_row(200 + i, 'foot') for i in range(2)]
        )
        selected = select_balanced_rows(rows, seed=3, target_rows=6)
        counts = collections.Counter(construction_family(row) for row in selected)

        self.assertEqual(counts, {'foot': 2, 'midpoint': 2, 'reflect': 2})

    def test_selection_round_robins_goals_within_family(self):
        rows = (
            [_row(i, 'reflect', 'perp') for i in range(4)]
            + [_row(100 + i, 'reflect', 'para') for i in range(4)]
        )
        selected = select_balanced_rows(rows, seed=9, target_rows=4)
        counts = collections.Counter(goal_predicate(row) for row in selected)

        self.assertEqual(counts, {'para': 2, 'perp': 2})

    def test_family_cap_is_enforced(self):
        rows = [_row(i, 'reflect') for i in range(10)]
        selected = select_balanced_rows(rows, seed=1, per_family_cap=3)
        self.assertEqual(len(selected), 3)

    def test_fractional_family_cap_scales_with_final_corpus(self):
        rows = (
            [_row(i, 'reflect') for i in range(20)]
            + [_row(100 + i, 'midpoint') for i in range(10)]
            + [_row(200 + i, 'foot') for i in range(10)]
        )
        selected = select_balanced_rows(
            rows, seed=5, max_family_fraction=0.40
        )
        counts = collections.Counter(construction_family(row) for row in selected)

        self.assertLess(len(selected), len(rows))
        self.assertLessEqual(max(counts.values()), math.ceil(0.40 * len(selected)))

    def test_fractional_goal_cap_limits_perpendicular_tail(self):
        rows = (
            [_row(i, 'reflect', 'perp') for i in range(20)]
            + [_row(100 + i, 'midpoint', 'para') for i in range(5)]
            + [_row(200 + i, 'foot', 'coll') for i in range(5)]
        )
        selected = select_balanced_rows(rows, seed=7, max_goal_fraction=0.50)
        counts = collections.Counter(goal_predicate(row) for row in selected)

        self.assertLessEqual(counts['perp'], math.ceil(0.50 * len(selected)))

    def test_focus_match_is_preferred_within_same_cell(self):
        rows = [_row(i, 'reflect') for i in range(3)]
        rows[2]['focus_construction'] = 'reflect'
        rows[2]['strict_focus_retained'] = True

        selected = select_balanced_rows(rows, seed=11, target_rows=1)

        self.assertEqual(selected[0]['id'], '2')

    def test_source_family_survives_inverse_action_conversion(self):
        row = _row(1, 'midpoint')
        row['source_constructions'] = ['centroid']
        row['target_construction_names'] = ['midpoint']

        self.assertEqual(construction_family(row), 'centroid')


if __name__ == '__main__':
    unittest.main()
