#!/usr/bin/env bash
# Four-hour, 300-CPU auxiliary-goal production for eqangle and eqratio.

#SBATCH --job-name=ag-eq-goals-v1
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --cpus-per-task=1
# Observed peak RSS is about 1.3 GiB; retain more than 2x headroom while
# allowing all 300 workers through the per-user aggregate-memory QOS.
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --array=0-299
#SBATCH --output=logs/ag-eq-goals-v1-%A_%a.out
#SBATCH --error=logs/ag-eq-goals-v1-%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

task_id="${SLURM_ARRAY_TASK_ID:-0}"
out_root="${OUT_ROOT:-outputs/synthetic_data/geometry_corpus_v15_eq_goals_production}"
seed_base="${SEED_BASE:-5300000}"
start_guard_seconds="${START_GUARD_SECONDS:-12600}"

# Preserve broad construction coverage while spending candidate-goal capacity
# exclusively on the two newly enabled theorem predicates.
focus_csv="${FOCUS_CONSTRUCTIONS:-angle_bisector,centroid,circle,circumcenter,eq_triangle,eqangle3,eqdistance,foot,incenter,incenter2,intersection_cc,intersection_lc,intersection_ll,intersection_lp,intersection_lt,intersection_pp,intersection_tt,lc_tangent,midpoint,mirror,ninepoints,nsquare,on_aline,on_bline,on_circle,on_circum,on_dia,on_line,on_opline,on_pline,on_tline,orthocenter,parallelogram,psquare,reflect,shift}"
IFS=',' read -r -a focus_names <<< "$focus_csv"
focus_name="${focus_names[$((task_id % ${#focus_names[@]}))]}"

chunk=0
while (( SECONDS < start_guard_seconds )); do
    seed=$((seed_base + task_id * 10000 + chunk))
    out_dir="$out_root/part_$task_id/chunk_$chunk"
    done_marker="$out_dir/.done"
    if [[ -f "$done_marker" ]]; then
        echo "task=$task_id chunk=$chunk already complete; skipping"
        chunk=$((chunk + 1))
        continue
    fi

    mkdir -p "$out_dir"
    echo "task=$task_id chunk=$chunk seed=$seed focus=$focus_name"

    NUM_DIAGRAMS="${NUM_DIAGRAMS:-4}" \
    MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}" \
    SEED="$seed" \
    MIN_STEPS="${MIN_STEPS:-8}" \
    MAX_STEPS="${MAX_STEPS:-10}" \
    PER_STEP_ATTEMPTS="${PER_STEP_ATTEMPTS:-35}" \
    MAX_CANDIDATES="${MAX_CANDIDATES:-400}" \
    MAX_AUXILIARY_PER_DIAGRAM="${MAX_AUXILIARY_PER_DIAGRAM:-16}" \
    MAX_PER_PREDICATE="${MAX_PER_PREDICATE:-12}" \
    CONSTRUCTION_SET=expanded \
    ROOT_POLICY="${ROOT_POLICY:-diversified}" \
    FOCUS_CONSTRUCTIONS="$focus_name" \
    FOCUS_ATTEMPTS="${FOCUS_ATTEMPTS:-200}" \
    FOCUS_DEPENDENCY_RATE="${FOCUS_DEPENDENCY_RATE:-0.90}" \
    FOCUS_DEPENDENCY_MODE="${FOCUS_DEPENDENCY_MODE:-mixed}" \
    FOCUS_BRIDGE_MODE="${FOCUS_BRIDGE_MODE:-empirical_v6}" \
    FOCUS_PLACEMENT="${FOCUS_PLACEMENT:-eager}" \
    FOCUS_SCAN_CANDIDATES="${FOCUS_SCAN_CANDIDATES:-160}" \
    REQUIRE_FOCUS_IN_AUXILIARY=0 \
    AUXILIARY_GOAL_PREDICATES=eqangle,eqratio \
    AUXILIARY_MINING_MODE=traceback \
    SKIP_PRETRAINING=1 \
    STRICT_FILTER=1 \
    REBUILD_ATTEMPTS=1 \
    ATTEMPT_TIME_BUDGET="${ATTEMPT_TIME_BUDGET:-180}" \
    MAX_RUNTIME_SECONDS="${CHUNK_RUNTIME_SECONDS:-1800}" \
    STRICT_WALL_TIMEOUT="${STRICT_WALL_TIMEOUT:-60}" \
    OUT_DIR="$out_dir" \
        bash scripts/run_geometry_corpus_v2_pilot.sh

    touch "$done_marker"
    chunk=$((chunk + 1))
done

echo "task=$task_id stopped before four-hour wall limit after $chunk chunks"
