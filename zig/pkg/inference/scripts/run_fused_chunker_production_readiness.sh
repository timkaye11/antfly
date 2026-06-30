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
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd -- "$script_dir/.." && pwd)"

zig_bin="${ANTFLY_ZIG_BIN:-}"
if [[ -z "$zig_bin" ]]; then
  if command -v zig >/dev/null 2>&1; then
    zig_bin="$(command -v zig)"
  else
    echo "missing Zig compiler; set ANTFLY_ZIG_BIN=/path/to/zig" >&2
    exit 1
  fi
fi

mode="${ANTFLY_FUSED_CHUNKER_READINESS_MODE:-probe}"
run_id="${ANTFLY_FUSED_CHUNKER_READINESS_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
out_root="${ANTFLY_FUSED_CHUNKER_READINESS_ROOT:-/private/tmp/zig-fused-chunker-readiness}"
out_dir="${ANTFLY_FUSED_CHUNKER_OUTPUT:-$out_root/$run_id}"
log_path="${ANTFLY_FUSED_CHUNKER_READINESS_LOG:-$out_dir/train.log}"
summary_path="$out_dir/readiness_summary.json"

train_data="${ANTFLY_FUSED_CHUNKER_TRAIN_DATA:-/Users/tim/Documents/af/gopeft/data/fused_train.jsonl}"
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-embed-base}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"

skip_batch_parity="${ANTFLY_FUSED_CHUNKER_SKIP_BATCH_PARITY:-0}"
skip_log_verify="${ANTFLY_FUSED_CHUNKER_SKIP_LOG_VERIFY:-0}"
memory_abort_rss_gb="${ANTFLY_FUSED_CHUNKER_MEMORY_ABORT_RSS_GB:-44}"
memory_warn_rss_gb="${ANTFLY_FUSED_CHUNKER_MEMORY_WARN_RSS_GB:-36}"
max_avg_step_ms="${ANTFLY_FUSED_CHUNKER_MAX_AVG_STEP_MS:-12000}"
max_peak_rss_gb="${ANTFLY_FUSED_CHUNKER_MAX_PEAK_RSS_GB:-44}"
require_encoder_neftune="${ANTFLY_FUSED_CHUNKER_REQUIRE_ENCODER_NEFTUNE:-0}"
min_probability_gap="${ANTFLY_FUSED_CHUNKER_MIN_PROBABILITY_GAP:-0}"

case "$mode" in
  probe)
    export ANTFLY_FUSED_CHUNKER_MAX_STEPS="${ANTFLY_FUSED_CHUNKER_MAX_STEPS:-200}"
    export ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS:-100}"
    export ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES="${ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES:-256}"
    min_steps="${ANTFLY_FUSED_CHUNKER_MIN_STEPS:-100}"
    min_fixed_f1="${ANTFLY_FUSED_CHUNKER_MIN_FIXED_F1:-0}"
    min_best_f1="${ANTFLY_FUSED_CHUNKER_MIN_BEST_F1:-0}"
    ;;
  full)
    export ANTFLY_FUSED_CHUNKER_MAX_STEPS="${ANTFLY_FUSED_CHUNKER_MAX_STEPS:-0}"
    export ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS:-500}"
    export ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES="${ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES:-0}"
    min_steps="${ANTFLY_FUSED_CHUNKER_MIN_STEPS:-1000}"
    min_fixed_f1="${ANTFLY_FUSED_CHUNKER_MIN_FIXED_F1:-0.766}"
    min_best_f1="${ANTFLY_FUSED_CHUNKER_MIN_BEST_F1:-0.766}"
    ;;
  *)
    echo "invalid ANTFLY_FUSED_CHUNKER_READINESS_MODE=$mode (expected probe or full)" >&2
    exit 1
    ;;
esac

mkdir -p "$out_dir"

export ANTFLY_FUSED_CHUNKER_OUTPUT="$out_dir"
export ANTFLY_FUSED_CHUNKER_MEMORY_ABORT_RSS_GB="$memory_abort_rss_gb"
export ANTFLY_FUSED_CHUNKER_MEMORY_WARN_RSS_GB="$memory_warn_rss_gb"
export ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION="${ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION:-mpsgraph_required}"
export TERMITE_MPSGRAPH_SMOKE="${TERMITE_MPSGRAPH_SMOKE:-1}"

echo "fused chunker readiness mode=$mode out_dir=$out_dir log=$log_path"
echo "gates min_steps=$min_steps min_fixed_f1=$min_fixed_f1 min_best_f1=$min_best_f1 min_probability_gap=$min_probability_gap max_avg_step_ms=$max_avg_step_ms max_peak_rss_gb=$max_peak_rss_gb require_encoder_neftune=$require_encoder_neftune"

probability_gap_args=()
if [[ "$min_probability_gap" != "skip" ]]; then
  probability_gap_args=(--min-mean-positive-probability-gap "$min_probability_gap")
fi

if [[ "$skip_batch_parity" != "1" ]]; then
  ANTFLY_FUSED_CHUNKER_PARITY_DATA="$train_data" \
    ANTFLY_FUSED_CHUNKER_MODEL_DIR="$model_dir" \
    ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN="$max_seq_len" \
    ANTFLY_FUSED_CHUNKER_MAX_CHUNKS="$max_chunks" \
    ANTFLY_FUSED_CHUNKER_BATCH_SIZE="$batch_size" \
    ANTFLY_FUSED_CHUNKER_PARITY_LIMIT="${ANTFLY_FUSED_CHUNKER_BATCH_PARITY_LIMIT:-8}" \
    "$script_dir/run_fused_chunker_go_zig_batch_parity.sh"
fi

"$script_dir/run_fused_chunker_phase20_metal.sh" >"$log_path" 2>&1

if [[ "$skip_log_verify" != "1" ]]; then
  "$script_dir/verify_fused_chunker_probe_log.sh" "$log_path"
fi

(
  cd "$pkg_root"
  "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true validate-fused-chunker-run -- \
    --out-dir "$out_dir" \
    --require-backend metal \
    --require-mpsgraph-vjp \
    --require-encoder-neftune "$require_encoder_neftune" \
    --max-vjp-fallbacks 0 \
    --min-steps "$min_steps" \
    --min-fixed-f1 "$min_fixed_f1" \
    --min-best-f1 "$min_best_f1" \
    "${probability_gap_args[@]}" \
    --max-avg-step-ms "$max_avg_step_ms" \
    --max-peak-rss-gb "$max_peak_rss_gb" \
    ${ANTFLY_FUSED_CHUNKER_REQUIRE_COMPLETE:+--require-complete}
) | tee "$summary_path"

echo "fused chunker readiness passed summary=$summary_path"
