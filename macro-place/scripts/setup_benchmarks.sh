#!/usr/bin/env bash
# setup_benchmarks.sh — clone the TILOS MacroPlacement repo for benchmarks +
# evaluator bindings.  Idempotent: skips work if already present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="${REPO_ROOT}/benchmarks/external"
TILOS_DIR="${EXT_DIR}/MacroPlacement"

mkdir -p "${EXT_DIR}"

if [ -d "${TILOS_DIR}" ]; then
    echo "[setup] TILOS MacroPlacement already present at ${TILOS_DIR}"
else
    echo "[setup] Cloning TILOS MacroPlacement (this is large, ~500MB)..."
    git clone --depth 1 \
        https://github.com/TILOS-AI-Institute/MacroPlacement.git \
        "${TILOS_DIR}"
fi

echo "[setup] Benchmarks available at:"
echo "  ICCAD04: ${TILOS_DIR}/Testcases/ICCAD04"
echo "  NG45:    ${TILOS_DIR}/Testcases/NG45  (if present)"

echo ""
echo "[setup] Done."
echo "  Set TILOS_MACRO_PLACEMENT_ROOT=${TILOS_DIR} if you want to be explicit,"
echo "  but the loader will auto-find it at this default location."
