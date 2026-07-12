#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
inference_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$inference_dir/../../.." && pwd)"
nvcc="${NVCC:-/usr/local/cuda-13.2/bin/nvcc}"
zig_bin="${ZIG_BIN:-$repo_dir/.tools/zig-x86_64-linux-0.16.0/zig}"
default_binary="$inference_dir/zig-out/bin/antfly-inference"
binary="${ANTFLY_INFERENCE_BINARY:-$default_binary}"
optimize="${OPTIMIZE:-ReleaseFast}"
artifact_dir="${ARTIFACT_DIR:-/tmp/antfly-e2b-ffn-sm89}"
warmups="${WARMUPS:-20}"
measure="${MEASURE:-200}"
repeats="${REPEATS:-5}"

if [[ ! -x "$nvcc" ]]; then
  printf 'error: CUDA compiler is not executable: %s\n' "$nvcc" >&2
  exit 1
fi

if [[ -z "${ANTFLY_INFERENCE_BINARY+x}" ]]; then
  if [[ ! -x "$zig_bin" ]]; then
    printf 'error: Zig compiler is not executable: %s\n' "$zig_bin" >&2
    exit 1
  fi
  printf 'building CUDA benchmark binary (%s): %s\n' "$optimize" "$default_binary"
  (
    cd "$inference_dir"
    "$zig_bin" build -Dcuda=true -Dmetal=false -Dcuda-artifacts=sm89 -Doptimize="$optimize"
  )
else
  if [[ ! -x "$binary" ]]; then
    printf 'error: custom inference binary is not executable: %s\n' "$binary" >&2
    exit 1
  fi
  stale_input=""
  while IFS= read -r -d '' input; do
    if [[ -z "$stale_input" && "$input" -nt "$binary" ]]; then
      stale_input="$input"
    fi
  done < <(find "$inference_dir/src" "$inference_dir/build" "$inference_dir/build.zig" "$inference_dir/build.zig.zon" -type f -print0)
  if [[ -n "$stale_input" ]]; then
    printf 'error: custom inference binary is stale: %s\n' "$binary" >&2
    printf 'newer build input: %s\n' "$stale_input" >&2
    printf 'rebuild the custom binary or unset ANTFLY_INFERENCE_BINARY to let this script build one\n' >&2
    exit 1
  fi
fi

if [[ ! -x "$binary" ]]; then
  printf 'error: inference binary is not executable: %s\n' "$binary" >&2
  exit 1
fi

mkdir -p "$artifact_dir"

compile_candidate() {
  local source="$1"
  local output="$2"
  "$nvcc" -cubin -arch=sm_89 "$inference_dir/$source" -o "$artifact_dir/$output"
}

compile_candidate \
  src/ops/cuda/generated/quant_kernel_q4_0_pair_activation_q8_1_e2b_6144.cu \
  antfly_q4_0_pair_activation_q8_1_e2b_6144_mmv_v1.sm89.cubin
compile_candidate \
  src/ops/cuda/generated/quant_kernel_q4_0_pair_activation_q8_1_e2b_12288.cu \
  antfly_q4_0_pair_activation_q8_1_e2b_12288_mmv_v1.sm89.cubin
compile_candidate \
  src/ops/cuda/generated/quant_kernel_q4_0_down_q8_1_e2b_6144.cu \
  antfly_q4_0_down_q8_1_e2b_6144_mmv_v1.sm89.cubin
compile_candidate \
  src/ops/cuda/generated/quant_kernel_q4_0_down_q8_1_e2b_12288.cu \
  antfly_q4_0_down_q8_1_e2b_12288_mmv_v1.sm89.cubin

cd "$repo_dir"
exec "$binary" bench-cuda \
  --q4-0-q8-1-e2b-ffn-sm89 "$artifact_dir" \
  --warmup-iters "$warmups" \
  --measure-iters "$measure" \
  --repeat-runs "$repeats"
