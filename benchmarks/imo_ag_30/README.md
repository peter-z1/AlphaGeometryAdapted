# Canonical IMO-AG-30 evaluation

The problem definitions live in `examples/imo_ag_30.txt`. `evaluate_one.py`
runs one indexed theorem with the optional AlphaGeometryRE/ChatLLM reference
backend, writes progress immediately, and applies a per-problem timeout.

Example:

```bash
python benchmarks/imo_ag_30/evaluate_one.py \
  --problem_index 0 \
  --model_file artifacts/reference/alphageometry-lm-f32.bin \
  --model_batch_size 32 --beam_size 512 --search_depth 16 \
  --timeout_seconds 300

python benchmarks/imo_ag_30/collect_results.py
```

Run different indices in independent local processes when resources allow.
The checked-in `results.json`/`results.md` are a historical, time-limited
reference-backend run and include unfinished items. Selected reconstruction-v4
results are kept separately under `benchmarks/results/reconstruction_v4/`.

The runner disables diagram-aesthetic `too close`/`too far` filters by default
for canonical fixed-coordinate inputs. This does not remove symbolic premises
or DD+AR rules. Pass `--enforce_point_distance_checks` to restore those
heuristics.
