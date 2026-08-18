#!/usr/bin/env bash
# Evaluate both eqangle/eqratio-finetuned models under one locked protocol.

#SBATCH --job-name=ag-eq-bench
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/ag-eq-bench-%A_%a.out
#SBATCH --error=logs/ag-eq-bench-%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/qwen3_5_9b/evaluations outputs/self_trained/evaluations

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

eq_test=outputs/synthetic_data/geometry_corpus_v15_eq_goals_lm/split/test.jsonl
common_args=(
  --model_batch_size 8
  --max_decode_len 128
  --beam_size 8
  --search_depth 2
  --timeout_seconds 180
  --seed 0
  --eq_test_file "$eq_test"
)

echo "job_id=${SLURM_JOB_ID:-local}"
echo "array_task=${SLURM_ARRAY_TASK_ID:-0}"
echo "started=$(date --iso-8601=seconds)"
nvidia-smi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    export MPLCONFIGDIR="$ROOT/outputs/qwen3_5_9b/matplotlib"
    mkdir -p "$MPLCONFIGDIR"
    .venv-qwen35/bin/python experiments/qwen3_5_9b/evaluate.py \
      --label eq_goals_replay_v1 \
      --backend huggingface \
      --adapter outputs/qwen3_5_9b/eq_goals_replay_v1/final_adapter \
      --out_root outputs/qwen3_5_9b/evaluations \
      "${common_args[@]}"
    ;;
  1)
    export MPLCONFIGDIR="$ROOT/outputs/self_trained/matplotlib"
    mkdir -p "$MPLCONFIGDIR"
    release_root="$ROOT/../release_artifacts/alphageometry_educational_release_v4"
    .venv/bin/python experiments/qwen3_5_9b/evaluate.py \
      --label eq_goals_replay_v1 \
      --backend pytorch \
      --checkpoint outputs/self_trained/eq_goals_replay_v1/checkpoint_latest.pt \
      --tokenizer "$release_root/tokenizer/ag_word_757_pretrain_rich_v2.model" \
      --out_root outputs/self_trained/evaluations \
      "${common_args[@]}"
    ;;
  *)
    echo "unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "finished=$(date --iso-8601=seconds)"
