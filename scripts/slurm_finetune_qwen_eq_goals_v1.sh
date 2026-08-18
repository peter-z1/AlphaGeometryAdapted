#!/usr/bin/env bash
# Continue the selected Qwen auxiliary adapter for one replay epoch.

#SBATCH --job-name=ag-qwen-eq-ft
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ag-qwen-eq-ft-%j.out
#SBATCH --error=logs/ag-qwen-eq-ft-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/qwen3_5_9b/matplotlib

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="$ROOT/outputs/qwen3_5_9b/matplotlib"

config=experiments/qwen3_5_9b/configs/eq_goals_replay_v1.json
echo "job_id=${SLURM_JOB_ID:-local}"
echo "started=$(date --iso-8601=seconds)"
nvidia-smi

.venv-qwen35/bin/python experiments/qwen3_5_9b/train.py \
  --config "$config" \
  --validate_only
.venv-qwen35/bin/python experiments/qwen3_5_9b/train.py \
  --config "$config"

echo "finished=$(date --iso-8601=seconds)"
