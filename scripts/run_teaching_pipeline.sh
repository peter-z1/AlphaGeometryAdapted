#!/usr/bin/env bash
# Run a bounded generation-to-dataset demonstration on one machine.

set -euo pipefail

repo_root="${REPO_ROOT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
run_root="${OUT_DIR:-outputs/teaching/pipeline}"
corpus_dir="$run_root/corpus"
strict_file="$run_root/auxiliary_strict.jsonl"
pairs_file="$run_root/auxiliary_pairs.jsonl"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ag-teaching-matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$run_root"

"$python_bin" src/generate_geometry_corpus.py \
    --num_diagrams "${NUM_DIAGRAMS:-10}" \
    --max_attempts "${MAX_ATTEMPTS:-100}" \
    --seed "${SEED:-1}" \
    --min_steps "${MIN_STEPS:-8}" \
    --max_steps "${MAX_STEPS:-10}" \
    --max_runtime_seconds "${MAX_RUNTIME_SECONDS:-1800}" \
    --construction_set "${CONSTRUCTION_SET:-expanded}" \
    --text_mode construction_proof \
    --out_dir "$corpus_dir"

"$python_bin" src/filter_strict_auxiliary.py \
    "$corpus_dir/auxiliary_candidates.jsonl" \
    --out "$strict_file" \
    --rejected_out "$run_root/auxiliary_rejected.jsonl" \
    --stats_out "$run_root/auxiliary_strict.stats.json" \
    --max_level 1000 --ddar_timeout 10 --wall_timeout 180 \
    --minimize exact

"$python_bin" src/prepare_lm_dataset.py \
    "$corpus_dir/pretraining.jsonl" \
    --out_dir "$run_root/pretraining_dataset" \
    --dedupe_key text --split_key diagram_id

if [[ -s "$strict_file" ]]; then
    "$python_bin" src/synthetic_data_to_lm.py \
        --input "$strict_file" \
        --out "$pairs_file" \
        --allow_multi_target --max_target_predicates -1

    "$python_bin" src/prepare_lm_dataset.py \
        "$pairs_file" \
        --out_dir "$run_root/auxiliary_dataset" \
        --dedupe_key text --split_key diagram_id \
        --val_ratio 0.05 --test_ratio 0.05
else
    echo "No strict auxiliary row was found in this small random sample."
    echo "Increase NUM_DIAGRAMS/MAX_ATTEMPTS or use data/generated/auxiliary_v4."
fi

echo "Teaching datasets are under $run_root"
