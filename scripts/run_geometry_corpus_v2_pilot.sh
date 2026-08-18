#!/usr/bin/env bash
# Bounded environment-variable wrapper around generation and strict filtering.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ag-mpl-geometry-corpus-v2}"
mkdir -p "$MPLCONFIGDIR"

out_dir="${OUT_DIR:-outputs/synthetic_data/geometry_corpus_v2_pilot}"

extra_args=()
if [[ "${SKIP_PRETRAINING:-0}" == "1" ]]; then
    extra_args+=(--skip_pretraining)
fi
if [[ -n "${FOCUS_CONSTRUCTIONS:-}" ]]; then
    extra_args+=(--focus_constructions "$FOCUS_CONSTRUCTIONS")
    extra_args+=(--focus_attempts "${FOCUS_ATTEMPTS:-100}")
    extra_args+=(--focus_dependency_rate "${FOCUS_DEPENDENCY_RATE:-0.75}")
    extra_args+=(--focus_dependency_mode "${FOCUS_DEPENDENCY_MODE:-mixed}")
    extra_args+=(--focus_bridge_mode "${FOCUS_BRIDGE_MODE:-none}")
    extra_args+=(--focus_placement "${FOCUS_PLACEMENT:-eager}")
    extra_args+=(--focus_scan_candidates "${FOCUS_SCAN_CANDIDATES:-160}")
    # Soft focus is a proposal preference, never an acceptance requirement.
    if [[ "${REQUIRE_FOCUS_IN_AUXILIARY:-0}" == "1" ]]; then
        extra_args+=(--require_focus_in_auxiliary)
    fi
fi
if [[ "${GZIP_PRETRAINING:-0}" == "1" ]]; then
    extra_args+=(--gzip_pretraining)
fi
extra_args+=(--auxiliary_mining_mode "${AUXILIARY_MINING_MODE:-traceback}")
extra_args+=(--counterfactual_max_targets "${COUNTERFACTUAL_MAX_TARGETS:-4}")
extra_args+=(--counterfactual_focus_trials "${COUNTERFACTUAL_FOCUS_TRIALS:-1}")
extra_args+=(--counterfactual_proposal_pool "${COUNTERFACTUAL_PROPOSAL_POOL:-0}")
extra_args+=(--counterfactual_exact_trials "${COUNTERFACTUAL_EXACT_TRIALS:-0}")
if [[ -n "${AUXILIARY_GOAL_PREDICATES:-}" ]]; then
    extra_args+=(--auxiliary_goal_predicates "$AUXILIARY_GOAL_PREDICATES")
fi

"$ROOT/.venv/bin/python" src/generate_geometry_corpus.py \
    --num_diagrams "${NUM_DIAGRAMS:-10}" \
    --max_attempts "${MAX_ATTEMPTS:-100}" \
    --seed "${SEED:-600000}" \
    --min_steps "${MIN_STEPS:-8}" \
    --max_steps "${MAX_STEPS:-8}" \
    --per_step_attempts "${PER_STEP_ATTEMPTS:-25}" \
    --max_level "${MAX_LEVEL:-1000}" \
    --ddar_timeout "${DDAR_TIMEOUT:-10}" \
    --attempt_time_budget "${ATTEMPT_TIME_BUDGET:-180}" \
    --max_runtime_seconds "${MAX_RUNTIME_SECONDS:-1800}" \
    --max_candidates "${MAX_CANDIDATES:-500}" \
    --max_pretraining_per_diagram "${MAX_PRETRAINING_PER_DIAGRAM:-96}" \
    --max_auxiliary_per_diagram "${MAX_AUXILIARY_PER_DIAGRAM:-32}" \
    --max_per_predicate "${MAX_PER_PREDICATE:-24}" \
    --min_proof_steps "${MIN_PROOF_STEPS:-1}" \
    --curated_rate "${CURATED_RATE:-0}" \
    --construction_set "${CONSTRUCTION_SET:-expanded}" \
    --root_policy "${ROOT_POLICY:-triangle}" \
    --text_mode "${TEXT_MODE:-construction_proof}" \
    --out_dir "$out_dir" \
    "${extra_args[@]}"

if [[ "${STRICT_FILTER:-1}" == "1" ]]; then
    "$ROOT/.venv/bin/python" src/filter_strict_auxiliary.py \
        "$out_dir/auxiliary_candidates.jsonl" \
        --out "$out_dir/auxiliary_strict.jsonl" \
        --rejected_out "$out_dir/auxiliary_rejected.jsonl" \
        --stats_out "$out_dir/auxiliary_strict.stats.json" \
        --max_level "${STRICT_MAX_LEVEL:-1000}" \
        --ddar_timeout "${STRICT_DDAR_TIMEOUT:-10}" \
        --wall_timeout "${STRICT_WALL_TIMEOUT:-180}" \
        --rebuild_attempts "${REBUILD_ATTEMPTS:-1}" \
        --minimize exact \
        --flush_every 1 \
        --log_every "${STRICT_LOG_EVERY:-100}"
fi
