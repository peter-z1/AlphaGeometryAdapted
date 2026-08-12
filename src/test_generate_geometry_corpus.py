"""Tests for shared-closure geometry corpus generation."""

from __future__ import annotations

import random
import unittest

import numpy as np

import problem as pr
from generate_geometry_corpus import mine_diagram_rows
from generate_synthetic_data import (
    CURATED_FULL_PROBLEMS,
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


if __name__ == '__main__':
    unittest.main()
