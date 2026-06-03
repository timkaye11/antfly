#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PKG_DIR="$ROOT_DIR/pkg/inference"
SRC="${CUDA_KERNEL_SRC:-$PKG_DIR/src/ops/cuda/artifacts/inference_cuda_kernels.cu}"
OUT="${CUDA_KERNEL_PTX_OUT:-$PKG_DIR/src/ops/cuda/artifacts/inference_cuda_kernels.ptx}"
NVCC_BIN="${NVCC:-nvcc}"
ARCH="${CUDA_PTX_ARCH:-compute_75}"

if ! command -v "$NVCC_BIN" >/dev/null 2>&1; then
  echo "nvcc not found; set NVCC=/path/to/nvcc or install the CUDA toolkit" >&2
  exit 2
fi

"$NVCC_BIN" \
  -ptx \
  -arch="$ARCH" \
  -O3 \
  --use_fast_math \
  -o "$OUT" \
  "$SRC"

echo "wrote $OUT"
