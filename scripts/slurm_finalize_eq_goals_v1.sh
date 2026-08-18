#!/usr/bin/env bash
# Merge strictly verified rows after the eqangle/eqratio array terminates.

#SBATCH --job-name=ag-eq-goals-final
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ag-eq-goals-final-%j.out
#SBATCH --error=logs/ag-eq-goals-final-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

production_root="${PRODUCTION_ROOT:-outputs/synthetic_data/geometry_corpus_v15_eq_goals_production}"
out_root="${OUT_ROOT:-outputs/synthetic_data/geometry_corpus_v15_eq_goals_finalized}"
min_complete_chunks="${MIN_COMPLETE_CHUNKS:-300}"
mkdir -p "$out_root"

mapfile -t done_markers < <(find "$production_root" -type f -name .done | sort)
if (( ${#done_markers[@]} < min_complete_chunks )); then
    echo "only ${#done_markers[@]} complete chunks; require $min_complete_chunks" >&2
    exit 1
fi

strict_manifest="$out_root/strict_inputs.txt"
: > "$strict_manifest"
for marker in "${done_markers[@]}"; do
    chunk_dir="${marker%/.done}"
    for required in audit.json auxiliary_strict.jsonl auxiliary_strict.stats.json; do
        if [[ ! -f "$chunk_dir/$required" ]]; then
            echo "complete marker has missing $required: $chunk_dir" >&2
            exit 1
        fi
    done
    printf '%s\n' "$chunk_dir/auxiliary_strict.jsonl" >> "$strict_manifest"
done

"$ROOT/.venv/bin/python" src/merge_synthetic_shards.py \
    "@$strict_manifest" \
    --out "$out_root/auxiliary_strict.jsonl" \
    --dedupe canonical

echo "complete chunks: ${#done_markers[@]}"
echo "merged strict corpus: $out_root/auxiliary_strict.jsonl"
echo "diversity stats: $out_root/auxiliary_strict.jsonl.stats.json"

# Convert only targets executable by the current one-action inference loop,
# then recreate the exact deterministic split used by the coverage experiment.
lm_root="${LM_ROOT:-outputs/synthetic_data/geometry_corpus_v15_eq_goals_lm}"
mkdir -p "$lm_root"
"$ROOT/.venv/bin/python" src/synthetic_data_to_lm.py \
    --input "$out_root/auxiliary_strict.jsonl" \
    --out "$lm_root/converted.jsonl" \
    --stats_out "$lm_root/conversion_stats.json"
"$ROOT/.venv/bin/python" src/prepare_lm_dataset.py \
    "$lm_root/converted.jsonl" \
    --out_dir "$lm_root/split" \
    --dedupe_key text \
    --split_key diagram_id \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 1517

# Materialize the committed v4 split as plain JSONL, then append each new
# split to its matching replay partition. A zero validation/test ratio keeps
# all rows in train.jsonl inside each partition directory.
qwen_data="${QWEN_DATA_ROOT:-outputs/qwen3_5_9b/data}"
"$ROOT/.venv/bin/python" experiments/qwen3_5_9b/prepare_data.py \
    --out_dir "$qwen_data" \
    --syntax_rows 0 \
    --syntax_val_rows 0 \
    --force

for split in train val test; do
    "$ROOT/.venv/bin/python" src/prepare_lm_dataset.py \
        "$qwen_data/auxiliary/$split.jsonl" \
        "$lm_root/split/$split.jsonl" \
        --out_dir "$lm_root/replay_$split" \
        --dedupe_key text \
        --split_key diagram_id \
        --val_ratio 0 \
        --test_ratio 0 \
        --seed 1517
done

echo "coverage split: $lm_root/split"
echo "replay partitions: $lm_root/replay_{train,val,test}"
