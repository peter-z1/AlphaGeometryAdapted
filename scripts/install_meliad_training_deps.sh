#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MELIAD_DIR="${MELIAD_DIR:-external/meliad}"
if [[ ! -d "$MELIAD_DIR" ]]; then
    git clone https://github.com/google-research/meliad.git "$MELIAD_DIR"
fi

source .venv/bin/activate
python -m pip install -r "$MELIAD_DIR/requirements.txt"

if [[ "${INSTALL_JAX_CUDA:-0}" == "1" ]]; then
    python -m pip install --upgrade "jax[cuda12]"
    python -m pip install t5
fi

python -m pip install -r requirements-training.txt
