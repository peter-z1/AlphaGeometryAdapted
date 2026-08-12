"""Aggregate and rank shared-closure geometry corpus sweep configurations."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

from merge_synthetic_shards import aux_construction_names


DEFAULT_CONFIGS = ('expanded_s8', 'expanded_s10', 'expanded_s12', 'all_s10')


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _sum_counter_dicts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    total: collections.Counter[str] = collections.Counter()
    for row in rows:
        total.update({key: int(value) for key, value in row.items()})
    return dict(sorted(total.items()))


def _rate(numerator: float, denominator: float, scale: float = 1.0) -> float:
    if not denominator:
        return 0.0
    return round(scale * numerator / denominator, 4)


def _iter_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSONL') from exc


def summarize_config(
    input_root: Path,
    config_name: str,
    expected_workers: int,
) -> dict[str, Any]:
    config_root = input_root / config_name
    audit_paths = sorted(config_root.glob('part_*/audit.json'))
    strict_stats_paths = sorted(config_root.glob('part_*/auxiliary_strict.stats.json'))
    strict_paths = sorted(config_root.glob('part_*/auxiliary_strict.jsonl'))

    audits = [_read_json(path) for path in audit_paths]
    strict_stats = [_read_json(path) for path in strict_stats_paths]
    worker_names = {path.parent.name for path in audit_paths}
    strict_worker_names = {path.parent.name for path in strict_paths}
    expected_names = {f'part_{index}' for index in range(expected_workers)}

    generation_counts = _sum_counter_dicts(
        audit.get('counts', {}) for audit in audits
    )
    closure_statuses = _sum_counter_dicts(
        audit.get('closure_statuses', {}) for audit in audits
    )
    strict_counts = _sum_counter_dicts(
        stats.get('counts', {}) for stats in strict_stats
    )

    diagram_counts: collections.Counter[str] = collections.Counter()
    target_counts: collections.Counter[str] = collections.Counter()
    aux_type_counts: collections.Counter[str] = collections.Counter()
    goal_counts: collections.Counter[str] = collections.Counter()
    strict_rows = 0
    for row in _iter_jsonl(strict_paths):
        strict_rows += 1
        diagram_counts[str(row.get('diagram_id', ''))] += 1
        target = str(row.get('target_auxiliary', ''))
        target_counts[target] += 1
        aux_type_counts.update(aux_construction_names(target))
        goal = str(row.get('goal', '')).split()
        if goal:
            goal_counts[goal[0]] += 1

    diagrams_emitted = generation_counts.get('diagrams_emitted', 0)
    candidates = generation_counts.get('auxiliary_candidates_emitted', 0)
    generator_seconds = sum(float(audit.get('elapsed_seconds', 0)) for audit in audits)
    strict_seconds = sum(float(stats.get('elapsed_seconds', 0)) for stats in strict_stats)
    cpu_hours = (generator_seconds + strict_seconds) / 3600
    unique_diagrams = len([key for key in diagram_counts if key])
    max_rows = max(diagram_counts.values(), default=0)

    config_values: dict[str, Any] = {}
    if audits:
        config = audits[0].get('config', {})
        config_values = {
            'construction_set': config.get('construction_set'),
            'min_steps': config.get('min_steps'),
            'max_steps': config.get('max_steps'),
            'target_diagrams_per_worker': config.get('num_diagrams'),
        }

    complete = (
        worker_names == expected_names
        and strict_worker_names == expected_names
        and len(strict_stats_paths) == expected_workers
    )
    return {
        'config': config_values,
        'complete': complete,
        'workers': {
            'expected': expected_workers,
            'generation_audits': len(audit_paths),
            'strict_outputs': len(strict_paths),
            'strict_stats': len(strict_stats_paths),
            'missing_generation': sorted(expected_names - worker_names),
            'missing_strict': sorted(expected_names - strict_worker_names),
        },
        'cpu_hours': round(cpu_hours, 4),
        'generator_cpu_hours': round(generator_seconds / 3600, 4),
        'strict_filter_cpu_hours': round(strict_seconds / 3600, 4),
        'generation_counts': generation_counts,
        'closure_statuses': closure_statuses,
        'strict_filter_counts': strict_counts,
        'strict_rows': strict_rows,
        'unique_auxiliary_diagrams': unique_diagrams,
        'unique_target_strings': len(target_counts),
        'max_rows_from_one_diagram': max_rows,
        'max_diagram_row_fraction': _rate(max_rows, strict_rows),
        'strict_candidate_keep_fraction': _rate(strict_rows, candidates),
        'strict_rows_per_1000_diagrams': _rate(strict_rows, diagrams_emitted, 1000),
        'auxiliary_diagrams_per_1000_diagrams': _rate(
            unique_diagrams, diagrams_emitted, 1000
        ),
        'strict_rows_per_cpu_hour': _rate(strict_rows, cpu_hours),
        'auxiliary_diagrams_per_cpu_hour': _rate(unique_diagrams, cpu_hours),
        'aux_construction_types': dict(aux_type_counts.most_common()),
        'goal_predicates': dict(goal_counts.most_common()),
    }


def build_report(
    input_root: Path,
    configs: Iterable[str],
    expected_workers: int,
) -> dict[str, Any]:
    summaries = {
        name: summarize_config(input_root, name, expected_workers)
        for name in configs
    }
    ranking = sorted(
        summaries,
        key=lambda name: (
            summaries[name]['auxiliary_diagrams_per_cpu_hour'],
            summaries[name]['auxiliary_diagrams_per_1000_diagrams'],
            -summaries[name]['max_diagram_row_fraction'],
        ),
        reverse=True,
    )
    return {
        'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'input_root': str(input_root),
        'complete': all(summary['complete'] for summary in summaries.values()),
        'ranking_metric': 'auxiliary_diagrams_per_cpu_hour',
        'ranking': ranking,
        'recommended_config': ranking[0] if ranking else None,
        'configs': summaries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Geometry corpus v2 sweep report',
        '',
        f"Complete: **{report['complete']}**",
        '',
        '| Rank | Configuration | Diagrams | Strict rows | Aux diagrams | '
        'Aux diagrams / CPU-h | Aux diagrams / 1k | Max diagram share | CPU-h |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for rank, name in enumerate(report['ranking'], start=1):
        summary = report['configs'][name]
        emitted = summary['generation_counts'].get('diagrams_emitted', 0)
        lines.append(
            f"| {rank} | `{name}` | {emitted} | {summary['strict_rows']} | "
            f"{summary['unique_auxiliary_diagrams']} | "
            f"{summary['auxiliary_diagrams_per_cpu_hour']:.4f} | "
            f"{summary['auxiliary_diagrams_per_1000_diagrams']:.4f} | "
            f"{summary['max_diagram_row_fraction']:.2%} | "
            f"{summary['cpu_hours']:.2f} |"
        )
    lines.extend([
        '',
        f"Recommended configuration: **{report['recommended_config']}**",
        '',
        'The recommendation prioritizes distinct diagrams requiring an auxiliary '
        'construction per CPU-hour. Raw strict rows remain preserved; diagram caps '
        'or weights should be applied only when preparing the eventual training set.',
        '',
    ])
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--expected_workers_per_config', type=int, default=16)
    parser.add_argument(
        '--configs',
        nargs='+',
        default=list(DEFAULT_CONFIGS),
    )
    parser.add_argument('--require_complete', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_workers_per_config < 1:
        raise ValueError('--expected_workers_per_config must be positive')
    report = build_report(
        args.input_root,
        args.configs,
        args.expected_workers_per_config,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / 'report.json'
    markdown_path = args.out_dir / 'report.md'
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    markdown_path.write_text(render_markdown(report), encoding='utf-8')
    print(render_markdown(report))
    print(f'JSON report: {json_path}')
    print(f'Markdown report: {markdown_path}')
    if args.require_complete and not report['complete']:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

