# Project provenance

This repository is an educational AlphaGeometry-style reconstruction. It is
not an official Google DeepMind project and does not reproduce DeepMind's
private training corpus.

## Code lineage

1. Google DeepMind released `google-deepmind/alphageometry` under Apache 2.0.
2. `foldl/AlphaGeometryRE` forked DeepMind revision
   `a8a1dc70818c1253b6524d761510a6ec6df39c07` and replaced JAX inference with
   ChatLLM.cpp. AlphaGeometryRE later incorporated a small fix from
   `tpgh24/ag4masses` in commit `fec6de3`.
3. This repository forked AlphaGeometryRE at commit `04c64f5` and added the
   data generation, PyTorch training, inference, and evaluation pipeline.

Upstream projects:

- https://github.com/google-deepmind/alphageometry
- https://github.com/foldl/AlphaGeometryRE
- https://github.com/tpgh24/ag4masses

## Reference model paths

Three different model artifacts were used during development:

| Artifact | Origin | Role | SHA-256 |
| --- | --- | --- | --- |
| `alphageometry-lm-f32.bin` | AlphaGeometryRE/ChatLLM ModelScope catalog | Early compatibility comparison; not converted locally | `dc041221169daaae36dd148fb9d4cb7c3f014956807322bbc34be66b7f4c4714` |
| `checkpoint_10999999` | DeepMind's official `download.sh` | Trusted reference evaluation with JAX/Meliad | `02d6728be6269e768a485620834a627aaa11d2852f16a8a478be38aee04123cb` |
| `geometry.757.model` | DeepMind's official `download.sh` | Official model tokenizer | `a219a8cf71d57c2e71e345bf1777fedec81452086ec39446c804fb7fd07fbedb` |

The direct official evaluation used DeepMind checkout `6777cb5` and Meliad
checkout `e8af054`. Two local runtime fixes were applied: postponed annotation
evaluation in `dd.py`, and headless/degeneracy-safe numerical construction in
`numericals.py`.

Official weights and external repositories are intentionally not committed.
They should be downloaded from their upstream sources and verified by hash.

## Educational reconstruction v4

The released reconstruction is a 12-layer PyTorch causal transformer with
hidden size 1024, 8 attention heads, feed-forward size 4096, context length
512, and 152,057,856 parameters.

Pretraining used 5,400,640 unique theorem/proof strings derived from 9,452,200
generated rows. Fine-tuning used 15,751 unique strict auxiliary-construction
examples. These were produced by combining the day-one and targeted-v3 strict
corpora, canonicalizing 16,277 rows to 16,000 rows, then text-deduplicating and
splitting them into 14,284 train, 790 validation, and 677 test examples.

Six fine-tuning settings were compared. Variant 2 was selected with learning
rate `3e-5`, 2,000 steps, and validation loss `0.4571874572`.

Critical reconstruction artifacts:

| Artifact | SHA-256 |
| --- | --- |
| Base pretraining checkpoint | `dd1d87e800c1afb0aa9c1bd921e561a88dff23ca70f3ecb04634922fbe1541b2` |
| Selected fine-tuned checkpoint | `dbe698a6755d7ec27b16ac431d4d8b865f1a5431c2eede61ab8c2ef5eb4d0df6` |
| Reconstruction tokenizer | `0198213fec7569204dac59e65149182d8698d29a02fa1aa1763d2812c1b2708b` |

Despite its historical filename containing `757`, the reconstruction
tokenizer and model have an actual vocabulary size of 368.

## Scope of the reconstruction

The model follows the broad AlphaGeometry architecture and pretrain/fine-tune
workflow, but it is not a bit-for-bit reproduction. DeepMind used Meliad,
T5-style relative positions, context length 1024, and much larger private
datasets. This project uses a compact PyTorch implementation and substantially
smaller generated corpora for education and reproducible experimentation.

## Release and repository policy

The public Git tree includes source, benchmark definitions/results, the full
v4 auxiliary split, and a deterministic 2,000-row pretraining sample. The
5.35 GB full pretraining split and two 1.82 GB checkpoints are distributed as
a separate versioned bundle whose critical hashes are listed in
`release/manifest-v4.json`.

Machine-specific scheduler scripts, job logs, Python environments, downloaded
upstream repositories, converted third-party weights, failed pilot corpora,
and superseded checkpoints are not part of the educational repository. They
were development infrastructure rather than inputs to the selected v4 model.
