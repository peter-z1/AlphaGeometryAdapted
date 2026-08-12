"""Tests for strict auxiliary filtering."""

import unittest

import problem as pr
import test_alphageometry
from filter_strict_auxiliary import filter_row


class StrictAuxiliaryFilterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.definitions = pr.Definition.from_txt_file('data/defs.txt', to_dict=True)
        cls.rules = pr.Theorem.from_txt_file('data/rules.txt', to_dict=True)

    def classify(self, row):
        return filter_row(
            row,
            self.definitions,
            self.rules,
            max_level=1000,
            ddar_timeout=10,
            wall_timeout=60,
            rebuild_attempts=1,
            minimize='exact',
            max_exact_clauses=4,
        )

    def test_required_reflection_is_kept(self):
        # This is row 10-237-296 from aux_cpu_v1_merged.jsonl.  D is given by
        # two orthocenter altitudes; reflecting D across CA lets DD+AR prove
        # the missing third altitude CD perpendicular AB (and hence BG).
        row = {
            'id': '10-237-296',
            'visible_problem': (
                'a b c = triangle a b c; '
                'd = orthocenter d b a c; '
                'g = intersection_lc g a c b ? perp c d b g'
            ),
            'target_auxiliary': 'e = reflect e d c a',
            'goal': 'perp c d b g',
        }

        result = self.classify(row)

        self.assertEqual(result.category, 'aux_required')
        self.assertIsNotNone(result.row)
        self.assertEqual(result.row['target_auxiliary'], 'e = reflect e d c a')
        self.assertTrue(result.row['strict_auxiliary'])
        self.assertEqual(
            result.row['strict_visible_solve']['outcome'], 'unsolved'
        )
        self.assertEqual(
            result.row['strict_with_auxiliary_solve']['outcome'], 'solved'
        )
        self.assertTrue(result.row['strict_proof_steps'])

    def test_visible_solution_is_rejected(self):
        row = {
            'id': 'visible-control',
            'visible_problem': (
                'a b c = triangle a b c; '
                'd = midpoint d a b ? midp d a b'
            ),
            'target_auxiliary': 'e = midpoint e b c',
            'goal': 'midp d a b',
        }

        result = self.classify(row)

        self.assertEqual(result.category, 'visible_solved')
        self.assertIsNone(result.row)

    def test_exact_minimization_removes_vacuous_clause(self):
        row = {
            'id': '18-132-117',
            'visible_problem': (
                'a b c = triangle a b c; '
                'e = orthocenter e b a c ? perp a b e c'
            ),
            'target_auxiliary': (
                'd = intersection_lc d a b c; f = reflect f a c e'
            ),
            'goal': 'perp a b e c',
        }

        result = self.classify(row)

        self.assertEqual(result.category, 'aux_required')
        self.assertEqual(
            result.row['original_target_auxiliary'], row['target_auxiliary']
        )
        self.assertEqual(
            result.row['target_auxiliary'], 'd = intersection_lc d a b c'
        )
        self.assertEqual(result.row['strict_minimization'], 'exact')
        self.assertEqual(result.row['strict_num_original_aux_clauses'], 2)
        self.assertEqual(result.row['strict_num_aux_clauses'], 1)


if __name__ == '__main__':
    test_alphageometry.run_test(StrictAuxiliaryFilterTest)
