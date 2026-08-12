# AlphaGeometry Educational Reconstruction

This repository combines the symbolic AlphaGeometry solver with a complete,
smaller-scale pipeline for generating synthetic geometry data, training an
auxiliary-construction language model, and evaluating the combined system.
It is intended for a course lecture and reproducible student experiments.

The project is based on
[`foldl/AlphaGeometryRE`](https://github.com/foldl/alphageometryre), which is a
re-engineering of
[`google-deepmind/alphageometry`](https://github.com/google-deepmind/alphageometry).
It is not an official Google DeepMind project and does not contain DeepMind's
private training data. See [PROVENANCE.md](PROVENANCE.md) for the exact code,
checkpoint, tokenizer, and dataset lineage.

## What is included

- The original symbolic DD+AR geometry engine and AlphaGeometry search loop.
- Random construction and theorem/proof generation.
- Strict filtering that keeps examples whose hidden auxiliary point is
  genuinely needed by DD+AR.
- Deterministic conversion, deduplication, splitting, and tokenizer training.
- A 152M-parameter PyTorch causal transformer and inference adapter.
- Formalized IMO-AG-30 benchmarks, saved proofs, and evaluation summaries.
- The complete 15,751-example v4 auxiliary fine-tuning split and a 2,000-row
  pretraining sample in Git-friendly form.
- A manifest for the full 8.6 GB educational artifact bundle.

The large checkpoints and 5.35 GB pretraining split are deliberately outside
Git. Put the released bundle at
`artifacts/alphageometry_educational_release_v4/` and verify it with:

```bash
python scripts/verify_release.py \
  artifacts/alphageometry_educational_release_v4
```

## Repository map

| Path | Purpose |
| --- | --- |
| `src/alphageometry.py` | DD+AR plus auxiliary-LM proof search |
| `src/generate_geometry_corpus.py` | Shared-closure generator for pretraining and auxiliary candidates |
| `src/filter_strict_auxiliary.py` | Rebuilds the visible problem and verifies that DD+AR needs the hidden point |
| `src/synthetic_data_to_lm.py` | Converts constructive records to LM prompt/target strings |
| `src/prepare_lm_dataset.py` | Deduplicates and creates stable train/validation/test splits |
| `src/train_geometry_tokenizer.py` | Trains the symbolic SentencePiece word tokenizer |
| `src/train_demo_lm.py` | Trains the PyTorch causal transformer on CPU or GPU |
| `src/pytorch_lm_inference.py` | Uses a trained PyTorch checkpoint during proof search |
| `data/defs.txt`, `data/rules.txt` | AlphaGeometry construction language and deduction rules |
| `data/generated/` | Small, committed teaching data and the complete v4 auxiliary split |
| `benchmarks/` | Formalizations, portable runners, diagrams, proofs, and results |
| `configs/reconstruction_v4/` | Sanitized configuration of the selected model |
| `release/manifest-v4.json` | Sizes and SHA-256 hashes for large release artifacts |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Install the optional training stack when generating data, training a
tokenizer/model, or using the PyTorch auxiliary model:

```bash
pip install -r requirements-training.txt
```

Run the unit tests:

```bash
bash run_tests.sh
```

## Solve with DD+AR

DD+AR requires no language-model checkpoint:

```bash
python src/alphageometry.py \
  --problems_file examples/examples.txt \
  --problem_name orthocenter \
  --mode ddar
```

The proof consists only of deductions from `data/rules.txt` and algebraic
reasoning.

## Solve with the reconstruction model

After placing the release bundle under `artifacts/`:

```bash
python src/alphageometry.py \
  --problems_file examples/examples.txt \
  --problem_name orthocenter \
  --mode alphageometry \
  --backend pytorch \
  --checkpoint artifacts/alphageometry_educational_release_v4/models/finetuned/checkpoint_latest.pt \
  --tokenizer artifacts/alphageometry_educational_release_v4/tokenizer/ag_word_757_pretrain_rich_v2.model \
  --device cuda \
  --batch_size 8 \
  --beam_size 8 \
  --search_depth 2
```

Use `--device cpu` on a machine without CUDA. The solver itself is CPU-based;
the GPU accelerates only neural decoding. Search branches can still be costly
because every proposed construction is rebuilt and checked by DD+AR.

The legacy AlphaGeometryRE/ChatLLM path remains available with
`--backend chatllm --model <model.bin>`. Its native library and converted
third-party model are optional and are not part of this repository. Install its
build helpers with `pip install -r requirements-chatllm.txt`, then run
`scripts/install_chatllm_native.sh` if that comparison backend is needed.

## Small end-to-end teaching run

Generate shared proof and auxiliary-candidate streams:

```bash
python src/generate_geometry_corpus.py \
  --num_diagrams 10 \
  --max_attempts 100 \
  --seed 1 \
  --construction_set expanded \
  --out_dir outputs/teaching/corpus
```

Strict auxiliary filtering is the important step: an auxiliary candidate is
not training evidence until the visible problem has been rebuilt and DD+AR has
failed without the hidden construction. The full commands and invariants are
in [SYNTHETIC_DATA_PIPELINE.md](SYNTHETIC_DATA_PIPELINE.md).

For a fast model demonstration, train on the committed corpus rather than
waiting for generation:

```bash
python src/train_geometry_tokenizer.py \
  data/generated/pretraining_sample/train.txt \
  data/generated/auxiliary_v4/train.txt \
  --model_prefix outputs/teaching/tokenizer/ag_word

python src/train_demo_lm.py \
  --train data/generated/auxiliary_v4/train.txt \
  --val data/generated/auxiliary_v4/val.txt \
  --tokenizer outputs/teaching/tokenizer/ag_word.model \
  --out_dir outputs/teaching/tiny_lm \
  --max_steps 500 \
  --batch_size 8 \
  --block_size 192 \
  --d_model 256 --n_layers 4 --n_heads 4 --d_ff 1024 \
  --device cuda --amp
```

This is a classroom overfit/smoke test, not the selected v4 model.

## Benchmarks

- `benchmarks/imo_ag_30/` contains the canonical 30-problem suite and saved
  reconstruction results.
- `benchmarks/official_alphageometry_canary/` records reference-checkpoint
  experiments used to validate the comparison setup.

Large-search results are stochastic and budget-dependent. Keep DD+AR solves,
genuine auxiliary solves, timeouts, and invalid/no-op LM proposals separate
when reporting a score.

## Reconstruction scale

The selected v4 model has 152,057,856 parameters, 12 layers, hidden size 1024,
8 heads, feed-forward size 4096, and context length 512. It was pretrained on
5,400,640 unique theorem/proof strings and fine-tuned on 15,751 unique strict
auxiliary examples.

This mirrors the two-stage AlphaGeometry idea on a much smaller scale. It is
not architecture- or data-identical to DeepMind's Meliad model, which used a
1024-token context and far larger private corpora. Detailed comparison and
hashes are in [PROVENANCE.md](PROVENANCE.md).

## Publishing the large artifacts

Do not commit `.pt` or `.bin` model files to ordinary Git history. Publish the
versioned bundle separately (for example as a GitHub Release or archival
dataset), attach `release/manifest-v4.json`, and document its permanent URL.
The `.gitignore` already excludes local artifact and output directories.

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[PROVENANCE.md](PROVENANCE.md). Upstream names and model results remain the
property of their respective authors; this educational adaptation is not
endorsed by them.
