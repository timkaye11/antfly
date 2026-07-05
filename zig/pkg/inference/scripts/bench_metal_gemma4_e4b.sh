#!/usr/bin/env bash
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ANTFLY_INFERENCE_GEMMA4_MODEL_NAME="${ANTFLY_INFERENCE_GEMMA4_MODEL_NAME:-ggml-org/gemma-4-E4B-it-GGUF}"
export OUT_DIR="${OUT_DIR:-/tmp/antfly-inference-gemma4-e4b-metal-$(date -u +%Y%m%d-%H%M%S)}"

macos_free_speculative_mb() {
  local stats page_size pages
  stats="$(vm_stat 2>/dev/null)" || return 1
  page_size="$(awk '/page size of/ { print $8; exit }' <<<"$stats")"
  pages="$(awk '
    /Pages free:/ { gsub(/\./, "", $3); free = $3 }
    /Pages speculative:/ { gsub(/\./, "", $3); speculative = $3 }
    END { print free + speculative }
  ' <<<"$stats")"
  if [[ -z "$page_size" || -z "$pages" ]]; then
    return 1
  fi
  echo $((pages * page_size / 1024 / 1024))
}

macos_pressure_available_mb() {
  local stats total_bytes free_pct
  stats="$(memory_pressure -Q 2>/dev/null)" || return 1
  total_bytes="$(awk '/The system has/ { print $4; exit }' <<<"$stats")"
  free_pct="$(awk '/System-wide memory free percentage:/ { gsub(/%/, "", $5); print $5; exit }' <<<"$stats")"
  if [[ -z "$total_bytes" || -z "$free_pct" ]]; then
    return 1
  fi
  echo $((total_bytes * free_pct / 100 / 1024 / 1024))
}

MIN_FREE_MB="${ANTFLY_INFERENCE_GEMMA4_E4B_MIN_FREE_MB:-1024}"
if [[ "$MIN_FREE_MB" != "0" ]]; then
  free_mb="$(macos_pressure_available_mb || macos_free_speculative_mb || true)"
  if [[ -n "$free_mb" && "$free_mb" -lt "$MIN_FREE_MB" ]]; then
    echo "Gemma4 E4B Metal bench refused: available memory ${free_mb} MiB below ${MIN_FREE_MB} MiB" >&2
    echo "set ANTFLY_INFERENCE_GEMMA4_E4B_MIN_FREE_MB=0 to force the run" >&2
    memory_pressure -Q >&2 2>/dev/null || true
    exit 2
  fi
fi

FAST_RESIDENCY="${ANTFLY_INFERENCE_GEMMA4_E4B_FAST_RESIDENCY:-1}"
if [[ "$FAST_RESIDENCY" != "0" ]]; then
  export TERMITE_METAL_DISABLE_GEMMA4_E4B_FAST_RESIDENCY="${TERMITE_METAL_DISABLE_GEMMA4_E4B_FAST_RESIDENCY:-0}"
  export ANTFLY_INFERENCE_GEMMA4_MIN_Q4_PAIR_ACT_REDUCE_OUT_F16="${ANTFLY_INFERENCE_GEMMA4_MIN_Q4_PAIR_ACT_REDUCE_OUT_F16:-0}"
  export ANTFLY_INFERENCE_GEMMA4_MIN_Q6_REDUCE_IN_F16="${ANTFLY_INFERENCE_GEMMA4_MIN_Q6_REDUCE_IN_F16:-0}"
  export ANTFLY_INFERENCE_GEMMA4_BENCH_SERVER_TOKENS="${ANTFLY_INFERENCE_GEMMA4_BENCH_SERVER_TOKENS:-16 64}"
  export ANTFLY_INFERENCE_GEMMA4_BENCH_SERVER_READY_POLLS="${ANTFLY_INFERENCE_GEMMA4_BENCH_SERVER_READY_POLLS:-3600}"
  export ANTFLY_INFERENCE_GEMMA4_MIN_SERVER_TOK_S="${ANTFLY_INFERENCE_GEMMA4_MIN_SERVER_TOK_S:-10}"
else
  export TERMITE_METAL_DISABLE_GEMMA4_E4B_FAST_RESIDENCY=1
  export TERMITE_METAL_Q8_RUNTIME_STAGING_MAX_MB="${ANTFLY_INFERENCE_GEMMA4_E4B_BASELINE_Q8_STAGING_MB:-32}"
  export ANTFLY_INFERENCE_GEMMA4_MIN_Q4_PAIR_ACT_REDUCE_OUT_F16="${ANTFLY_INFERENCE_GEMMA4_MIN_Q4_PAIR_ACT_REDUCE_OUT_F16:-1}"
  export ANTFLY_INFERENCE_GEMMA4_MIN_Q6_REDUCE_IN_F16="${ANTFLY_INFERENCE_GEMMA4_MIN_Q6_REDUCE_IN_F16:-1}"
fi

exec "$SCRIPT_DIR/bench_metal_gemma4_e2b.sh"
