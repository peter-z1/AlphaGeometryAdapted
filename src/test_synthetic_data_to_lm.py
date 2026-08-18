"""Unit tests for synthetic_data_to_lm.py."""

import unittest

import problem as pr
import synthetic_data_to_lm as conv
import test_alphageometry


class SyntheticDataToLmTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.defs = pr.Definition.from_txt_file('data/defs.txt', to_dict=True)

    def test_orthocenter_auxiliary_conversion(self):
        row = {
            'id': 'example',
            'seed': 10,
            'diagram_id': 'diagram-10',
            'visible_problem': (
                'a b c = triangle a b c; '
                'd = on_tline d b a c, on_tline d c a b ? perp a d b c'
            ),
            'target_auxiliary': 'e = on_line e a c, on_line e b d',
            'full_problem': (
                'a b c = triangle a b c; '
                'd = on_tline d b a c, on_tline d c a b; '
                'e = on_line e a c, on_line e b d ? perp a d b c'
            ),
            'goal': 'perp a d b c',
        }

        pair, metrics = conv.convert_row(row, SyntheticDataToLmTest.defs)

        self.assertEqual(
            pair['prompt'],
            '{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 '
            '? T a d b c {F1} x00',
        )
        self.assertEqual(pair['target'], 'e : C a c e 02 C b d e 03 ;')
        self.assertEqual(
            pair['text'],
            pair['prompt'] + ' ' + pair['target'],
        )
        self.assertEqual(metrics['target_groups'], 1)
        self.assertEqual(metrics['target_predicates'], 2)
        self.assertEqual(pair['diagram_id'], 'diagram-10')

        executable, reason = conv.validate_target_for_current_inference(
            pair['target'], row['visible_problem']
        )
        self.assertTrue(executable, reason)

    def test_triangle12_root_numeric_ratio_converts_to_executable_action(self):
        row = {
            'id': 'triangle12-action',
            'visible_problem': (
                'a b c = triangle12 a b c; '
                'e = orthocenter e a b c; '
                'f = on_dia f e b ? perp c e a b'
            ),
            'target_auxiliary': 'd = intersection_lc d a b c',
            'goal': 'perp c e a b',
        }

        pairs = conv.convert_row_to_action_pairs(
            row, SyntheticDataToLmTest.defs
        )

        self.assertEqual(len(pairs), 1)
        pair, _metrics = pairs[0]
        self.assertIn('/ a b a c 1/2 00', pair['prompt'])
        executable, reason = conv.validate_target_for_current_inference(
            pair['target'], pair['visible_problem']
        )
        self.assertTrue(executable, reason)

    def test_current_inference_rejects_multi_point_head(self):
        executable, reason = conv.validate_target_for_current_inference(
            'd e : C a b d 00 C a c e 01 ;',
            'a b c = triangle a b c ? perp a b b c',
        )
        self.assertFalse(executable)
        self.assertEqual(reason, 'multi_point_head')

    def test_multi_clause_target_splits_into_autoregressive_actions(self):
        row = {
            'id': 'two-actions',
            'seed': 4,
            'diagram_id': 'diagram-4',
            'visible_problem': (
                'a b c = triangle a b c ? perp a b b c'
            ),
            'target_auxiliary': (
                'd = midpoint d a b; e = on_line e d c'
            ),
            'goal': 'perp a b b c',
        }

        pairs = conv.convert_row_to_action_pairs(row, SyntheticDataToLmTest.defs)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0]['action_index'], 0)
        self.assertEqual(pairs[1][0]['action_index'], 1)
        self.assertIn('d = on_line d a b', pairs[1][0]['visible_problem'])
        self.assertIn(pairs[0][0]['target'], pairs[1][0]['prompt'])
        self.assertIn('on_bline', pairs[0][0]['target_auxiliary'])
        self.assertIn(
            pairs[1][0]['target_auxiliary'],
            ('e = on_line e d c', 'e = on_line e c d'),
        )

    def test_eqangle_named_constructions_round_trip_to_inference(self):
        visible = (
            'a b c d = quadrangle a b c d; e = free e '
            '? coll a b c'
        )
        cases = {
            'x = on_aline x a b c d e': 'on_aline',
            'x = eqangle3 x a b d e c': 'eqangle3',
            'x = angle_bisector x a b c': 'angle_bisector',
            'x = on_bline x a b': 'on_bline',
        }
        for auxiliary, expected in cases.items():
            with self.subTest(auxiliary=auxiliary):
                target, _metrics = conv.auxiliary_to_target(
                    auxiliary, SyntheticDataToLmTest.defs, 0
                )
                executable, reason = conv.validate_target_for_current_inference(
                    target, visible
                )
                self.assertTrue(executable, reason)
                clause, reason = conv.constrained_segment_to_constructive(
                    target, set('abcde')
                )
                self.assertEqual(reason, '')
                self.assertIn(expected, clause)

    def test_multi_output_centroid_splits_into_four_executable_actions(self):
        row = {
            'id': 'centroid-actions',
            'seed': 7,
            'diagram_id': 'diagram-7',
            'visible_problem': (
                'd e h = triangle d e h ? coll d e h'
            ),
            'target_auxiliary': 'i j k l = centroid i j k l h e d',
            'goal': 'coll d e h',
        }

        pairs = conv.convert_row_to_action_pairs(row, SyntheticDataToLmTest.defs)

        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            [pair['target'].split(' : ')[0] for pair, _metrics in pairs],
            list('ijkl'),
        )
        for pair, metrics in pairs:
            executable, reason = conv.validate_target_for_current_inference(
                pair['target'], pair['visible_problem']
            )
            self.assertTrue(executable, reason)
            self.assertEqual(metrics['target_groups'], 1)
            self.assertLessEqual(metrics['target_predicates'], 2)
            self.assertEqual(pair['source_constructions'], ['centroid'])

    def test_coupled_trisegment_remains_unsupported(self):
        row = {
            'id': 'coupled-action',
            'visible_problem': 'a b c = triangle a b c ? coll a b c',
            'target_auxiliary': 'd e = trisegment d e a b',
            'goal': 'coll a b c',
        }

        pairs = conv.convert_row_to_action_pairs(row, SyntheticDataToLmTest.defs)

        self.assertEqual(len(pairs), 1)
        executable, reason = conv.validate_target_for_current_inference(
            pairs[0][0]['target'], pairs[0][0]['visible_problem']
        )
        self.assertFalse(executable)
        self.assertEqual(reason, 'multi_point_head')

    def test_dependent_incenter_group_is_topologically_ordered(self):
        row = {
            'id': 'incenter-with-feet',
            'visible_problem': 'a b c = triangle a b c ? coll a b c',
            'target_auxiliary': 'd e f i = incenter2 d e f i a b c',
            'goal': 'coll a b c',
        }

        pairs = conv.convert_row_to_action_pairs(row, SyntheticDataToLmTest.defs)

        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            [pair['target'].split(' : ')[0] for pair, _metrics in pairs],
            ['i', 'd', 'e', 'f'],
        )
        for pair, _metrics in pairs:
            executable, reason = conv.validate_target_for_current_inference(
                pair['target'], pair['visible_problem']
            )
            self.assertTrue(executable, reason)


if __name__ == '__main__':
    test_alphageometry.run_test(SyntheticDataToLmTest)
