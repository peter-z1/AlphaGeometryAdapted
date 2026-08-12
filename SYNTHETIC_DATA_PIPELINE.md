# Synthetic Data Pipeline

This is the portable recipe used by the educational reconstruction. It avoids
machine-specific schedulers: independent generator/filter invocations can be
distributed with any multiprocessing or cluster system, then merged by the
same deterministic commands.

## Why there are two datasets

AlphaGeometry does not train only on hard auxiliary-construction problems.
The broad pipeline has two stages:

1. **Pretraining:** full synthetic theorem/proof strings teach the formal
   geometry language and deduction patterns. Most of these are solvable by
   DD+AR alone.
2. **Fine-tuning:** the problem shown to the model hides a proof-critical
   construction. The target is the missing construction in AlphaGeometry LM
   syntax.

DD+AR-solvable examples are therefore expected and useful in pretraining. They
are not valid positive examples in the strict auxiliary fine-tuning split.

## The correctness invariant

For every strict auxiliary example, the pipeline checks both states:

```text
visible premises              --DD+AR-->  not proved
visible premises + hidden aux --DD+AR-->  proved
```

Merely observing that a traced proof mentions a point is insufficient: DD+AR
may know a second proof that avoids it. `filter_strict_auxiliary.py` rebuilds
and solves both forms, and rejects the row unless the hidden construction
changes the outcome. With `--minimize exact`, it also removes superfluous
target clauses.

Numerical construction failures such as near-coincident points are sampling
failures, not logical counterexamples. The generator records/rejects them; it
must not silently treat them as successful strict examples.

## 1. Generate shared closures

The preferred generator constructs each diagram once, saturates it with
DD+AR, then mines both output streams from that closure:

```bash
python src/generate_geometry_corpus.py \
  --num_diagrams 100 \
  --max_attempts 2000 \
  --seed 1000 \
  --min_steps 8 --max_steps 10 \
  --construction_set all \
  --text_mode construction_proof \
  --gzip_pretraining \
  --out_dir outputs/corpus/shard_1000
```

Outputs are written incrementally:

- `pretraining.jsonl.gz`: numerically checked theorem/proof rows;
- `auxiliary_candidates.jsonl`: dependency-difference candidates;
- `audit.json`: configuration, yields, rejection counts, and closure status.

Only saturated closures are mined. Use a different `--seed` and output
directory for each independent worker. Preserve every `audit.json`; yield
without the rejection denominators is not interpretable.

For simple uncompressed classroom data, omit `--gzip_pretraining`.

## 2. Strictly filter auxiliary candidates

```bash
python src/filter_strict_auxiliary.py \
  'outputs/corpus/shard_*/auxiliary_candidates.jsonl' \
  --out outputs/corpus/auxiliary_strict.jsonl \
  --rejected_out outputs/corpus/auxiliary_rejected.jsonl \
  --stats_out outputs/corpus/auxiliary_strict.stats.json \
  --max_level 1000 \
  --ddar_timeout 10 \
  --wall_timeout 180 \
  --rebuild_attempts 1 \
  --minimize exact
```

The quoted glob is expanded inside the script. For portable parallel
filtering, start `N` independent commands with distinct
`--shard_index 0..N-1 --num_shards N`, then merge their kept outputs. Each
worker writes continuously, so an interrupted run retains completed records.

The most important rejection categories are:

- `visible_solved`: DD+AR did not need the hidden point;
- restored failure/timeout: necessity may hold, but sufficiency was not
  established under the budget;
- rebuild failure: the numerical construction could not be validated;
- invalid/minimization failure: the target could not be represented reliably.

Do not relabel timeouts or rebuild errors as positives.

## 3. Merge and deduplicate

When generation/filtering was sharded, merge modulo canonical point renaming:

```bash
python src/merge_synthetic_shards.py \
  'outputs/corpus/strict_shards/*.jsonl' \
  --out outputs/corpus/auxiliary_strict_merged.jsonl \
  --dedupe canonical
```

The adjacent stats file reports auxiliary construction types, goal predicates,
clause counts, and auxiliary centrality. Inspect it before training; a large
row count can still represent a narrow distribution.

Pretraining data is deduplicated later by its final `text` field. Point-renamed
copies should not cross train/validation/test boundaries.

## 4. Convert strict rows to LM examples

```bash
python src/synthetic_data_to_lm.py \
  --input outputs/corpus/auxiliary_strict_merged.jsonl \
  --out outputs/corpus/auxiliary_pairs.jsonl
```

The converter creates records of the form:

```text
prompt: {S} ... ? goal {F1} x00
target: e : C a c e 02 C b d e 03 ;
text:   <prompt> <target>
```

The target grammar is deliberately constrained. A semicolon terminates one
auxiliary construction proposal.

## 5. Deduplicate and split

Auxiliary data:

```bash
python src/prepare_lm_dataset.py \
  outputs/corpus/auxiliary_pairs.jsonl \
  --out_dir outputs/datasets/auxiliary \
  --dedupe_key text \
  --split_key full_problem \
  --val_ratio 0.05 --test_ratio 0.05 --seed 0
```

Pretraining data (repeat inputs for all uncompressed shards):

```bash
python src/prepare_lm_dataset.py \
  outputs/corpus/pretraining_shards/*.jsonl \
  --out_dir outputs/datasets/pretraining \
  --dedupe_key text \
  --split_key diagram_id \
  --val_ratio 0.01 --test_ratio 0.01 --seed 0
```

`prepare_lm_dataset.py` currently reads plain JSONL. Decompress `.jsonl.gz`
shards before this step or omit compression for small runs. Splitting by
`full_problem`/`diagram_id` is safer than splitting individual rows from the
same synthetic diagram.

## 6. Train and verify the tokenizer

```bash
python src/train_geometry_tokenizer.py \
  outputs/datasets/pretraining/train.txt \
  outputs/datasets/auxiliary/train.txt \
  --model_prefix outputs/tokenizers/ag_word \
  --vocab_size 757
```

The script reserves `{S}`, `{F1}`, punctuation, predicate symbols, and
two-digit dependency references, then runs a self-check. Never train with a
generic tokenizer that maps `:`, `;`, or reference tokens to `<unk>`.

SentencePiece can produce fewer than 757 pieces when the small corpus does not
contain enough distinct words and `--hard_vocab_limit` is off. The selected v4
tokenizer has 368 pieces despite its historical `757` filename; the checkpoint
and tokenizer are consistent with each other.

## 7. Pretrain, then fine-tune

A small classroom model:

```bash
python src/train_demo_lm.py \
  --train outputs/datasets/pretraining/train.txt \
  --val outputs/datasets/pretraining/val.txt \
  --tokenizer outputs/tokenizers/ag_word.model \
  --out_dir outputs/models/pretrain \
  --max_steps 2000 --batch_size 8 --grad_accum_steps 4 \
  --block_size 256 \
  --d_model 384 --n_layers 6 --n_heads 6 --d_ff 1536 \
  --device cuda --amp

python src/train_demo_lm.py \
  --train outputs/datasets/auxiliary/train.txt \
  --val outputs/datasets/auxiliary/val.txt \
  --tokenizer outputs/tokenizers/ag_word.model \
  --out_dir outputs/models/finetuned \
  --resume_from outputs/models/pretrain/checkpoint_latest.pt \
  --resume_weights_only \
  --max_steps 1000 --batch_size 8 --grad_accum_steps 4 \
  --block_size 256 \
  --d_model 384 --n_layers 6 --n_heads 6 --d_ff 1536 \
  --lr 3e-5 --device cuda --amp
```

The actual selected v4 settings are recorded in
`configs/reconstruction_v4/`. Use GPU for training when available. Generation,
strict DD+AR filtering, and symbolic candidate checking are CPU workloads.

## 8. Evaluate the combined system

```bash
python src/alphageometry.py \
  --mode alphageometry \
  --backend pytorch \
  --checkpoint outputs/models/finetuned/checkpoint_latest.pt \
  --tokenizer outputs/tokenizers/ag_word.model \
  --problems_file examples/imo_ag_30.txt \
  --problem_name translated_imo_2000_p6 \
  --device cuda \
  --batch_size 16 --beam_size 32 --search_depth 4
```

Report four categories rather than only a total:

1. solved before any LM call (DD+AR baseline);
2. solved after a nonempty, valid auxiliary construction (genuine LM solve);
3. unsolved within the budget;
4. invalid proposal, build error, or timeout.

This prevents a direct DD+AR solution or an empty/no-op LM proposal from being
misreported as an auxiliary-model success.

## v4 corpus used by the released prototype

- Pretraining raw rows: 9,452,200
- Unique pretraining strings: 5,400,640
- Pretraining split: 5,292,720 / 54,144 / 53,776
- Strict auxiliary rows before final text deduplication: 16,000
- Unique auxiliary examples: 15,751
- Auxiliary split: 14,284 / 790 / 677

The complete auxiliary split is committed under `data/generated/auxiliary_v4/`.
The full pretraining split and checkpoints live in the external release bundle
described by `release/manifest-v4.json`.
