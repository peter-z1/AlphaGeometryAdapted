# Qwen3.5-9B auxiliary-construction experiment

This folder adapts `Qwen/Qwen3.5-9B-Base` to the same formal
auxiliary-construction task as the educational 152M-parameter model. It keeps
two experimental arms separate:

1. **Auxiliary only:** fine-tune the pretrained Qwen model directly on the
   strict hidden-construction examples.
2. **Syntax then auxiliary:** first continue pretraining on synthetic
   theorem/proof strings, then fine-tune on exactly the same auxiliary split.

Their difference measures whether the geometry-language pretraining stage is
useful. The existing 152M reconstruction remains the from-scratch baseline.

## What is and is not stored here

The folder contains portable data preparation, tokenizer auditing, training,
evaluation, configuration, and tests. Lightweight Oscar scheduler recipes are
kept under `scripts/`. The repository deliberately contains no Qwen weights,
generated datasets, Hugging Face cache, or trained adapters. Those artifacts
go under `outputs/`, which is ignored by Git.

The official base model is Apache-2.0 licensed. Its language backbone has about
9B parameters. We load it with `AutoModelForCausalLM`, which excludes the
unused vision tower and exposes the text-only Qwen3.5 model.

## 1. Activate the isolated environment

Qwen3.5 support in Transformers requires Python 3.10 or newer. The repository's
existing Oscar `.venv` currently uses Python 3.9, so it must remain separate.
The Oscar setup created for this experiment is a self-contained Miniforge
environment at `.venv-qwen35` (Python 3.13, PyTorch 2.8 with CUDA 12.8). From
the `AlphaGeometryAdapted` repository root, activate it with:

```bash
. .venv-qwen35/bin/activate
```

The environment and its packages are already installed locally. The
`.venv-qwen*` pattern is ignored by Git. To reproduce the setup on another
machine, create any Python 3.10-or-newer environment and run:

```bash
pip install -r experiments/qwen3_5_9b/requirements.txt
```

On Oscar, Miniforge is a reliable source of a self-contained modern Python.
The tested Oscar Python 3.11 module had an incompatible OpenSSL library on the
login node, so it should not be used for this experiment unless that module is
repaired.

The full-run and smoke-test configurations pin the tested Hugging Face model
revision, so later model-card updates cannot silently change the experiment.

Transformers 5.3 or newer is required for the Qwen3.5 text-only class. The
default configuration uses four-bit QLoRA. On a GPU with enough memory for
BF16 LoRA, change `"quantization": "4bit"` to `"quantization": "none"` in
the three full-run configuration files.

Qwen3.5 can use optimized Gated DeltaNet kernels. First verify the base setup;
then, if the target GPU supports compiling them, optionally install:

```bash
pip install -r experiments/qwen3_5_9b/requirements-kernels.txt
```

Without those optional kernels, Transformers uses a slower and more
memory-hungry PyTorch implementation.

## 2. Prepare fixed data

```bash
python experiments/qwen3_5_9b/prepare_data.py
```

This command:

- scans the full v4 syntax training split once and selects exactly 250,000
  evenly distributed rows using seed 35;
- selects 5,000 validation rows;
- preserves the existing 14,284/790/677 auxiliary train/validation/test split;
- writes row counts and SHA-256 hashes to
  `outputs/qwen3_5_9b/data/manifest.json`.

The source release is discovered in either
`artifacts/alphageometry_educational_release_v4/` or the current workspace's
`../release_artifacts/alphageometry_educational_release_v4/`. Pass
`--syntax_source_dir` if it is stored elsewhere. Use `--syntax_rows 0
--syntax_val_rows 0` when preparing only the auxiliary-only arm.

## 3. Audit the Qwen tokenizer

Choose a cache under ignored output storage so the downloaded tokenizer and
later base weights never enter Git:

```bash
python experiments/qwen3_5_9b/audit_tokenizer.py \
  --cache_dir outputs/qwen3_5_9b/hf_cache
```

The audit records token-length percentiles, examples exceeding the 1,024-token
limit, unknown tokens, and the tokenization of `{S}`, `{F1}`, `x00`, dependency
numbers, colons, and semicolons. The observed strict auxiliary examples all fit
comfortably. Some long syntax-pretraining rows do not, so that stage keeps the
final 1,024-token proof suffix (`"overflow": "truncate_left"`). This controls
GPU memory while still teaching the construction/proof language. Auxiliary
training never truncates: do not begin it if a future audit reports over-length
auxiliary examples; increase `max_length` consistently in both auxiliary
configs first.

We retain Qwen's tokenizer. Replacing it with the reconstruction's 368-token
tokenizer would invalidate Qwen's pretrained embedding and output layers.

## 4. Validate the configurations without a GPU

```bash
python experiments/qwen3_5_9b/train.py \
  --config experiments/qwen3_5_9b/configs/auxiliary_only.json \
  --validate_only

python experiments/qwen3_5_9b/train.py \
  --config experiments/qwen3_5_9b/configs/syntax_pretrain.json \
  --validate_only
```

Validation checks the configuration and reports the fully resolved data and
output paths. It neither imports Transformers nor downloads the model.

## 5. Train the two comparison arms

Before committing GPU time, run a two-step end-to-end smoke test inside a GPU
allocation. This is the first command that downloads the base weights:

```bash
python experiments/qwen3_5_9b/train.py \
  --config experiments/qwen3_5_9b/configs/smoke_test.json
```

It uses only 64 training and 32 validation examples and writes to a disposable,
ignored `outputs/qwen3_5_9b/smoke_test_bf16/` directory.

Run the auxiliary-only control:

```bash
python experiments/qwen3_5_9b/train.py \
  --config experiments/qwen3_5_9b/configs/auxiliary_only.json
```

Run the two-stage alternative:

```bash
python experiments/qwen3_5_9b/train.py \
  --config experiments/qwen3_5_9b/configs/syntax_pretrain.json

python experiments/qwen3_5_9b/train.py \
  --config experiments/qwen3_5_9b/configs/auxiliary_after_syntax.json
```

The syntax stage uses full-sequence next-token loss. The auxiliary stages mask
the prompt labels with `-100`, the standard marker meaning “do not include this
token in the loss,” so only the hidden construction is learned. The three full
training configurations use the same Qwen base, LoRA rank 32, maximum length
1,024, effective batch size 32, data order, seed, and optimizer settings. The
two auxiliary stages differ only in whether they initialize from the base Qwen
model or the syntax adapter.

The first real run downloads the base weights. A QLoRA adapter is much smaller
than a full model checkpoint, but the shared Hugging Face cache still needs
space for the base model.

### Eqangle/eqratio coverage continuation

The optional coverage continuation generates strict examples whose goals are
limited to `eqangle` and `eqratio`, then replays them together with the locked
v4 auxiliary split. The scripts store every generated row and model artifact
under ignored `outputs/` paths:

```bash
generation_job=$(sbatch --parsable scripts/slurm_generate_eq_goals_v1.sh)
sbatch --dependency="afterok:${generation_job}" scripts/slurm_finalize_eq_goals_v1.sh

sbatch scripts/slurm_finetune_qwen_eq_goals_v1.sh
sbatch scripts/slurm_finetune_selftrained_eq_goals_v1.sh
```

The generation array uses 300 one-CPU tasks for four hours. Finalization
verifies every completed shard, merges and converts strict rows, creates a
deterministic 80/10/10 split using seed 1517, and builds replay train,
validation, and test inputs without copying them into Git. The Qwen
continuation starts from the selected syntax-then-auxiliary adapter; the 152M
continuation starts from the released reconstruction checkpoint.

## 6. Use an adapter in AlphaGeometry search

For the syntax-then-auxiliary model:

```bash
python src/alphageometry.py \
  --mode alphageometry \
  --backend huggingface \
  --model Qwen/Qwen3.5-9B-Base \
  --model_revision 68c46c4b3498877f3ef123c856ecfde50c39f404 \
  --adapter outputs/qwen3_5_9b/auxiliary_after_syntax/final_adapter \
  --hf_cache_dir outputs/qwen3_5_9b/hf_cache \
  --load_in_4bit --dtype bfloat16 --device cuda \
  --problems_file examples/imo_ag_30.txt \
  --problem_name translated_imo_2000_p6 \
  --batch_size 8 --beam_size 8 --search_depth 2
```

Replace the adapter path with
`outputs/qwen3_5_9b/auxiliary_only/final_adapter` for the control. The backend
uses the formal prompt directly, adds no chat template or thinking tokens,
generates until the first semicolon, and submits every proposal to the same
DD+AR verifier used by the 152M model.

The shared evaluator loads either backend and reports the locked v4 held-out
split, the new eqangle/eqratio held-out split, IMO-AG-30, and JGEX-33 under the
same search settings. After the coverage continuations finish, run both tasks
with:

```bash
sbatch scripts/slurm_benchmark_eq_models_v1.sh
```

Only aggregate results are committed at
`benchmarks/results/eq_goals_replay_v1/summary.json`; predictions, proofs, and
incremental summaries remain in ignored output directories.

## Comparison protocol

Keep problem files, point-distance settings, beam size, search depth, DD+AR
limits, and wall-clock timeout identical. Report separately:

- problems solved before an LM call;
- syntactically valid top-1/top-8 construction proposals;
- held-out targets matched exactly (a secondary metric because alternatives
  can be equivalent);
- problems solved after a nonempty predicted construction;
- invalid proposals, numerical rebuild failures, and timeouts;
- model size, GPU memory, and elapsed inference time.

Use the 677 strict auxiliary test examples for clean held-out measurement, then
the IMO-AG-30 and the previously identified 33 JGEX problems for end-to-end
comparison. Do not include the local Math 1810C final material in repository
results.
