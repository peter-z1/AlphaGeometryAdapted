"""Tests for the geometry corpus sweep diversity report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from report_geometry_corpus_sweep import build_report, render_markdown


class GeometryCorpusSweepReportTest(unittest.TestCase):

    def _write_worker(
        self,
        root: Path,
        config: str,
        worker: int,
        strict_rows: list[dict[str, object]],
        elapsed: float,
    ) -> None:
        part = root / config / f'part_{worker}'
        part.mkdir(parents=True)
        audit = {
            'config': {
                'construction_set': 'expanded',
                'min_steps': 8,
                'max_steps': 8,
                'num_diagrams': 10,
            },
            'counts': {
                'diagrams_emitted': 10,
                'auxiliary_candidates_emitted': 4,
            },
            'closure_statuses': {'saturated': 10},
            'elapsed_seconds': elapsed,
        }
        (part / 'audit.json').write_text(json.dumps(audit), encoding='utf-8')
        with (part / 'auxiliary_strict.jsonl').open('w', encoding='utf-8') as out:
            for row in strict_rows:
                out.write(json.dumps(row) + '\n')
        stats = {
            'rows_read': 4,
            'rows_kept': len(strict_rows),
            'counts': {'aux_required': len(strict_rows)},
            'elapsed_seconds': elapsed,
        }
        (part / 'auxiliary_strict.stats.json').write_text(
            json.dumps(stats), encoding='utf-8'
        )

    @staticmethod
    def _row(diagram_id: str, construction: str = 'midpoint') -> dict[str, str]:
        return {
            'diagram_id': diagram_id,
            'target_auxiliary': f'd = {construction} d a b',
            'goal': 'perp a b c d',
        }

    def test_ranks_unique_diagrams_per_cpu_hour(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_worker(
                root,
                'diverse',
                0,
                [self._row('a'), self._row('b')],
                1800,
            )
            self._write_worker(
                root,
                'repeated',
                0,
                [self._row('c'), self._row('c')],
                1800,
            )

            report = build_report(root, ['repeated', 'diverse'], 1)

            self.assertTrue(report['complete'])
            self.assertEqual(report['recommended_config'], 'diverse')
            self.assertEqual(
                report['configs']['diverse']['unique_auxiliary_diagrams'], 2
            )
            self.assertEqual(
                report['configs']['repeated']['max_diagram_row_fraction'], 1.0
            )
            self.assertIn('`diverse`', render_markdown(report))

    def test_missing_worker_marks_report_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_worker(root, 'partial', 0, [self._row('a')], 10)

            report = build_report(root, ['partial'], 2)

            self.assertFalse(report['complete'])
            self.assertEqual(
                report['configs']['partial']['workers']['missing_generation'],
                ['part_1'],
            )


if __name__ == '__main__':
    unittest.main()

