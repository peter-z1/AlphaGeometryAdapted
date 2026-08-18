# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for alphageometry.py."""

import unittest
from unittest import mock

import alphageometry

class AlphaGeometryTest(unittest.TestCase):

  def test_empty_auxiliary_completion_is_rejected(self):
    self.assertEqual(
        alphageometry.try_translate_constrained_to_construct('', None),
        'ERROR: empty construction',
    )
    self.assertEqual(
        alphageometry.try_translate_constrained_to_construct('   ', None),
        'ERROR: empty construction',
    )
    self.assertEqual(
        alphageometry.try_translate_constrained_sequence_to_construct('', None),
        'ERROR: empty construction',
    )

  def test_translate_constrained_to_constructive(self):
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'T', list('addb')
        ),
        ('on_dia', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'T', list('adbc')
        ),
        ('on_tline', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'P', list('bcda')
        ),
        ('on_pline', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bdcd')
        ),
        ('on_bline', ['d', 'c', 'b']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bdcb')
        ),
        ('on_circle', ['d', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'D', list('bacd')
        ),
        ('eqdistance', ['d', 'c', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'C', list('bad')
        ),
        ('on_line', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'C', list('bad')
        ),
        ('on_line', ['d', 'b', 'a']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'd', 'O', list('abcd')
        ),
        ('on_circum', ['d', 'a', 'b', 'c']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'x', '^', 'a b a x d e d c'.split()
        ),
        ('on_aline', ['x', 'a', 'b', 'c', 'd', 'e']),
    )
    self.assertEqual(
        alphageometry.translate_constrained_to_constructive(
            'x', '^', 'x a x b d e d f'.split()
        ),
        ('eqangle3', ['x', 'a', 'b', 'd', 'e', 'f']),
    )

  def test_eight_argument_eqangle_completion_round_trip(self):
    graph = mock.Mock()
    points = []
    for name in 'abcde':
      point = mock.Mock()
      point.name = name
      points.append(point)
    graph.all_points.return_value = points
    graph.copy.return_value = mock.Mock()

    translated = alphageometry.try_translate_constrained_to_construct(
        'x : ^ a b a x d e d c 00 ;', graph
    )

    self.assertEqual(translated, 'x = on_aline x a b c d e')

    translated = alphageometry.try_translate_constrained_to_construct(
        'x : D a x b x 00 ^ b a b x a x a b 01 ;', graph
    )
    self.assertEqual(translated, 'x = on_bline x b a')

  def test_insert_aux_to_premise(self):
    pstring = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b ? perp a d b c'  # pylint: disable=line-too-long
    auxstring = 'e = on_line e a c, on_line e b d'

    target = 'a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c'  # pylint: disable=line-too-long
    self.assertEqual(
        alphageometry.insert_aux_to_premise(pstring, auxstring), target
    )

  def test_invalid_candidate_reasoning_is_local_to_search(self):
    model = mock.Mock()
    model.beam_decode.return_value = {
        'seqs_str': ['d : C a b d 00 ;'],
        'scores': [0.0],
    }
    problem = mock.Mock()
    problem.setup_str_from_problem.return_value = '{S} a : ; b : ;'
    problem.txt.return_value = 'a b = segment a b ? cong a b a b'
    initial_graph = mock.Mock()
    candidate_graph = mock.Mock()
    stats = {}

    with mock.patch.object(
        alphageometry,
        'build_problem_with_retries',
        side_effect=[(initial_graph, []), (candidate_graph, [])],
    ), mock.patch.object(
        alphageometry,
        'run_ddar',
        side_effect=[False, ValueError('degenerate candidate')],
    ), mock.patch.object(
        alphageometry,
        'try_translate_constrained_to_construct',
        return_value='d = on_line d a b',
    ), mock.patch.object(
        alphageometry,
        'insert_aux_to_premise',
        return_value='a b = segment a b; d = on_line d a b ? cong a b a b',
    ), mock.patch.object(
        alphageometry.pr.Problem,
        'from_txt',
        return_value=mock.Mock(),
    ):
      solved = alphageometry.run_alphageometry(
          model=model,
          p=problem,
          search_depth=1,
          beam_size=1,
          out_file='',
          search_stats=stats,
      )

    self.assertFalse(solved)
    self.assertEqual(stats['candidate_reasoning_rejections'], 1)
    self.assertEqual(stats.get('accepted_candidates', 0), 0)

  def test_beam_queue(self):
    beam_queue = alphageometry.BeamQueue(max_size=2)

    beam_queue.add('a', 1)
    beam_queue.add('b', 2)
    beam_queue.add('c', 3)

    beam_queue = list(beam_queue)
    self.assertEqual(beam_queue, [(3, 'c'), (2, 'b')])

def run_test(cls):
    cls.setUpClass()
    obj = cls()
    for c in [s for s in dir(obj) if s.startswith('test_')]:
       print(f"testing {c} ...")
       m = getattr(obj, c)
       m()

if __name__ == '__main__':
    run_test(AlphaGeometryTest)
