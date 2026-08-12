#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ag-tests-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

"$python_bin" src/test_problem.py
"$python_bin" src/test_geometry.py
"$python_bin" src/test_graph_utils.py
"$python_bin" src/test_numericals.py
"$python_bin" src/test_graph.py
"$python_bin" src/test_dd.py
"$python_bin" src/test_ar.py
"$python_bin" src/test_ddar.py
"$python_bin" src/test_trace_back.py
"$python_bin" src/test_synthetic_data_to_lm.py
"$python_bin" src/test_filter_strict_auxiliary.py
"$python_bin" src/test_generate_geometry_corpus.py
"$python_bin" src/test_alphageometry.py
if [[ -n "${CHATLLM_MODEL:-}" ]]; then
    "$python_bin" src/test_lm_inference.py --model "$CHATLLM_MODEL"
else
    echo "Skipping optional ChatLLM integration test (set CHATLLM_MODEL to enable)."
fi
