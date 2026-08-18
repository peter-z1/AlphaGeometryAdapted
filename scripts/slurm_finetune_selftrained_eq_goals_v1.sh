#!/usr/bin/env bash
# Continue the 152M from-scratch reconstruction for one replay epoch.

#SBATCH --job-name=ag-self-eq-ft
#SBATCH --partition=gpu-he
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ag-self-eq-ft-%j.out
#SBATCH --error=logs/ag-self-eq-ft-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs outputs/self_trained/eq_goals_replay_v1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="$ROOT/outputs/self_trained/matplotlib"
mkdir -p "$MPLCONFIGDIR"

release_root="$ROOT/../release_artifacts/alphageometry_educational_release_v4"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "started=$(date --iso-8601=seconds)"
nvidia-smi

.venv/bin/python src/train_demo_lm.py \
  --train outputs/synthetic_data/geometry_corpus_v15_eq_goals_lm/replay_train/train.txt \
  --val outputs/synthetic_data/geometry_corpus_v15_eq_goals_lm/replay_val/train.txt \
  --tokenizer "$release_root/tokenizer/ag_word_757_pretrain_rich_v2.model" \
  --out_dir outputs/self_trained/eq_goals_replay_v1 \
  --max_steps 500 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --block_size 512 \
  --d_model 1024 \
  --n_layers 12 \
  --n_heads 8 \
  --d_ff 4096 \
  --dropout 0.1 \
  --lr 0.00001 \
  --weight_decay 0.1 \
  --warmup_steps 25 \
  --clip_grad_norm 1.0 \
  --eval_every 100 \
  --save_every 100 \
  --log_every 10 \
  --max_val_batches 200 \
  --seed 1517 \
  --num_workers 2 \
  --device cuda \
  --amp \
  --resume_from "$release_root/models/finetuned/checkpoint_latest.pt" \
  --resume_weights_only

echo "finished=$(date --iso-8601=seconds)"
