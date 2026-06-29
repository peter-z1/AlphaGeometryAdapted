# Synthetic Data Pipeline

This pipeline generates AlphaGeometry-style auxiliary-construction examples from
the released solver code.  It is meant to produce a compatible corpus for
experiments, not to reproduce DeepMind's private data bit-for-bit.

The generator works as follows:

1. sample a construction sequence from `data/defs.txt`;
2. build the full geometry graph;
3. run DD+AR using `data/rules.txt`;
4. mine true conclusions whose traceback uses a proof-only point;
5. remove that point's construction from the visible problem;
6. write a JSONL training pair.

Each JSONL row contains:

- `visible_problem`: theorem statement without the hidden auxiliary;
- `target_auxiliary`: construction(s) that should be inserted;
- `full_problem`: visible problem plus the hidden construction;
- `setup_dependencies`, `auxiliary_dependencies`, `proof_steps`: audit trail from traceback;
- `removed_dependent_clauses`: later random clauses dropped because they depended on the hidden point but were not themselves the target.

## Quick smoke test

Run from the repository root:

```bash
python src/generate_synthetic_data.py \
  --num_examples 2 \
  --max_attempts 20 \
  --curated_rate 0.5 \
  --seed 10 \
  --out outputs/synthetic_data/strict_smoke.jsonl
```

For a pure random smoke run:

```bash
python src/generate_synthetic_data.py \
  --num_examples 1 \
  --max_attempts 50 \
  --curated_rate 0 \
  --seed 3 \
  --out outputs/synthetic_data/random_smoke.jsonl
```

Random generation is sparse: many sampled diagrams are valid but do not yield a
useful hidden-auxiliary example.  Increase `--max_attempts` before assuming a
seed range is bad.

## Scaling on a cluster

Data generation is mostly CPU work.  GPUs become important later, when training
or running the language model.  The easiest scale-out pattern is many independent
seed shards:

```bash
python src/generate_synthetic_data.py \
  --num_examples 1000 \
  --seed 42 \
  --max_attempts 50000 \
  --curated_rate 0 \
  --out outputs/synthetic_data/part_0042.jsonl
```

For a Slurm array job, use the array index as the seed and output suffix:

```bash
python src/generate_synthetic_data.py \
  --num_examples 1000 \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --max_attempts 50000 \
  --curated_rate 0 \
  --out "outputs/synthetic_data/part_${SLURM_ARRAY_TASK_ID}.jsonl"
```

After the jobs finish, concatenate the JSONL files:

```bash
cat outputs/synthetic_data/part_*.jsonl > outputs/synthetic_data/all.jsonl
```

Use `--allow_solved_without_aux` only for debugging.  By default, the generator
rejects examples that DD+AR can already solve without the target auxiliary.
