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


if __name__ == '__main__':
    test_alphageometry.run_test(SyntheticDataToLmTest)
