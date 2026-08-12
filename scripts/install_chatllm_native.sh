#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${CHATLLM_REF:-master}"
SRC_DIR="$(mktemp -d /tmp/chatllm.cpp-src.XXXXXX)"
BUILD_DIR="$(mktemp -d /tmp/chatllm.cpp-build.XXXXXX)"
OUT_DIR="$ROOT/src/chatllm/bindings"

python -m pip install -r "$ROOT/requirements-chatllm.txt"

echo "Cloning ChatLLM.cpp (${REF})..."
git clone --depth 1 --branch "$REF" https://github.com/foldl/chatllm.cpp.git "$SRC_DIR"

echo "Configuring ChatLLM.cpp..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "Building libchatllm..."
cmake --build "$BUILD_DIR" --target libchatllm --parallel "${JOBS:-4}"

echo "Installing shared libraries into ${OUT_DIR}..."
mapfile -d '' chatllm_libs < <(
    find "$BUILD_DIR" "$SRC_DIR" \( -type f -o -type l \) \
        -name 'libchatllm.so' -print0
)
mapfile -d '' ggml_libs < <(
    find "$BUILD_DIR" "$SRC_DIR" \( -type f -o -type l \) \
        -name 'libggml*.so*' -print0
)
if (( ${#chatllm_libs[@]} == 0 )); then
    echo "No libchatllm.so was produced" >&2
    exit 1
fi
if (( ${#ggml_libs[@]} == 0 )); then
    echo "No libggml shared libraries were produced" >&2
    exit 1
fi
cp -a "${chatllm_libs[0]}" "$OUT_DIR/"
cp -a "${ggml_libs[@]}" "$OUT_DIR/"

if command -v patchelf >/dev/null 2>&1; then
    while IFS= read -r -d '' lib; do
        patchelf --set-rpath '$ORIGIN' "$lib"
    done < <(find "$OUT_DIR" -maxdepth 1 -type f -name 'lib*.so*' -print0)
fi

echo "Installed ChatLLM native library."
