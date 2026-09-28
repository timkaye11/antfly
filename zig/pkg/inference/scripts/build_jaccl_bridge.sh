#!/usr/bin/env bash
set -euo pipefail

# Pinned MLX source: v0.32.2, 1f8e74e3f12f31365464a6867c6579f0e9b29d85.
# Produces an optional runtime library; ordinary Antfly builds do not need it.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source_root="${ANTFLY_JACCL_SOURCE_ROOT:-${TMPDIR:-/tmp}/antfly-mlx-jaccl-v0.32.2}"
build_root="${ANTFLY_JACCL_BUILD_ROOT:-${TMPDIR:-/tmp}/antfly-jaccl-build}"
output_root="${ANTFLY_JACCL_OUTPUT_ROOT:-${repo_root}/zig/pkg/inference/zig-out/lib}"
revision=1f8e74e3f12f31365464a6867c6579f0e9b29d85

if [[ ! -d "${source_root}/.git" ]]; then
  git clone --depth 1 --branch v0.32.2 https://github.com/ml-explore/mlx.git "${source_root}"
fi
actual="$(git -C "${source_root}" rev-parse HEAD)"
if [[ "${actual}" != "${revision}" ]]; then
  echo "JACCL source must be MLX ${revision}; found ${actual}" >&2
  exit 1
fi

cmake -S "${source_root}/mlx/distributed/jaccl/lib" -B "${build_root}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_root}" -j "${ANTFLY_JACCL_BUILD_JOBS:-4}"
mkdir -p "${output_root}"
c++ -std=c++20 -O2 -dynamiclib \
  -I "${source_root}/mlx/distributed/jaccl/lib" \
  "${repo_root}/zig/pkg/inference/src/finetune/distributed/jaccl_bridge.cpp" \
  "${build_root}/libjaccl.a" \
  -o "${output_root}/libantfly_jaccl.dylib"
echo "${output_root}/libantfly_jaccl.dylib"
