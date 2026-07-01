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
analysis_path="${ANTFLY_FUSED_CHUNKER_VALIDATION_FAILURE_ANALYSIS:-$out_dir/validation_failure_analysis.json}"

train_data="${ANTFLY_FUSED_CHUNKER_TRAIN_DATA:-/Users/tim/Documents/af/gopeft/data/fused_train.jsonl}"
val_data="${ANTFLY_FUSED_CHUNKER_VAL_DATA:-/Users/tim/Documents/af/gopeft/data/fused_val.jsonl}"
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-embed-base}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"
val_split="${ANTFLY_FUSED_CHUNKER_VAL_SPLIT:-val}"

skip_batch_parity="${ANTFLY_FUSED_CHUNKER_SKIP_BATCH_PARITY:-0}"
skip_log_verify="${ANTFLY_FUSED_CHUNKER_SKIP_LOG_VERIFY:-0}"
memory_abort_rss_gb="${ANTFLY_FUSED_CHUNKER_MEMORY_ABORT_RSS_GB:-44}"
memory_warn_rss_gb="${ANTFLY_FUSED_CHUNKER_MEMORY_WARN_RSS_GB:-36}"
max_avg_step_ms="${ANTFLY_FUSED_CHUNKER_MAX_AVG_STEP_MS:-12000}"
max_peak_rss_gb="${ANTFLY_FUSED_CHUNKER_MAX_PEAK_RSS_GB:-44}"
require_encoder_neftune="${ANTFLY_FUSED_CHUNKER_REQUIRE_ENCODER_NEFTUNE:-1}"
min_probability_gap="${ANTFLY_FUSED_CHUNKER_MIN_PROBABILITY_GAP:-0}"
min_calibrated_f1="${ANTFLY_FUSED_CHUNKER_MIN_CALIBRATED_F1:-skip}"
min_average_precision="${ANTFLY_FUSED_CHUNKER_MIN_AVERAGE_PRECISION:-skip}"
min_max_rank_f1="${ANTFLY_FUSED_CHUNKER_MIN_MAX_RANK_F1:-skip}"
dense_device_forward="${TERMITE_METAL_DISABLE_DEVICE_DENSE_LINEAR_FORWARD:-1}"
checkpoint_eval="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_EVAL:-1}"
checkpoint_eval_results="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_EVAL_RESULTS:-$out_dir/checkpoint_eval_results.json}"
checkpoint_eval_max_examples="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_EVAL_MAX_EXAMPLES:-${ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES:-256}}"
min_checkpoint_fixed_f1="${ANTFLY_FUSED_CHUNKER_MIN_CHECKPOINT_FIXED_F1:-}"
min_checkpoint_best_f1="${ANTFLY_FUSED_CHUNKER_MIN_CHECKPOINT_BEST_F1:-}"
min_checkpoint_average_precision="${ANTFLY_FUSED_CHUNKER_MIN_CHECKPOINT_AVERAGE_PRECISION:-}"
min_checkpoint_max_rank_f1="${ANTFLY_FUSED_CHUNKER_MIN_CHECKPOINT_MAX_RANK_F1:-}"

case "$mode" in
  probe)
    export ANTFLY_FUSED_CHUNKER_MAX_STEPS="${ANTFLY_FUSED_CHUNKER_MAX_STEPS:-200}"
    export ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS:-100}"
    export ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES="${ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES:-256}"
    min_steps="${ANTFLY_FUSED_CHUNKER_MIN_STEPS:-100}"
    min_fixed_f1="${ANTFLY_FUSED_CHUNKER_MIN_FIXED_F1:-${ANTFLY_FUSED_CHUNKER_PROBE_MIN_FIXED_F1:-0.05}}"
    min_best_f1="${ANTFLY_FUSED_CHUNKER_MIN_BEST_F1:-${ANTFLY_FUSED_CHUNKER_PROBE_MIN_BEST_F1:-0.05}}"
    if [[ -z "${ANTFLY_FUSED_CHUNKER_MIN_AVERAGE_PRECISION:-}" ]]; then
      min_average_precision="${ANTFLY_FUSED_CHUNKER_PROBE_MIN_AVERAGE_PRECISION:-0.02}"
    fi
    if [[ -z "${ANTFLY_FUSED_CHUNKER_MIN_MAX_RANK_F1:-}" ]]; then
      min_max_rank_f1="${ANTFLY_FUSED_CHUNKER_PROBE_MIN_MAX_RANK_F1:-0.05}"
    fi
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

min_checkpoint_average_precision="${min_checkpoint_average_precision:-$min_average_precision}"

min_last_fixed_f1="${ANTFLY_FUSED_CHUNKER_MIN_LAST_FIXED_F1:-}"
min_last_best_f1="${ANTFLY_FUSED_CHUNKER_MIN_LAST_BEST_F1:-}"
min_last_calibrated_f1="${ANTFLY_FUSED_CHUNKER_MIN_LAST_CALIBRATED_F1:-}"
min_last_average_precision="${ANTFLY_FUSED_CHUNKER_MIN_LAST_AVERAGE_PRECISION:-}"
min_last_max_rank_f1="${ANTFLY_FUSED_CHUNKER_MIN_LAST_MAX_RANK_F1:-}"
min_last_probability_gap="${ANTFLY_FUSED_CHUNKER_MIN_LAST_MEAN_POSITIVE_PROBABILITY_GAP:-${ANTFLY_FUSED_CHUNKER_MIN_LAST_PROBABILITY_GAP:-}}"
if [[ "$mode" == "full" ]]; then
  min_last_fixed_f1="${min_last_fixed_f1:-$min_fixed_f1}"
  min_last_best_f1="${min_last_best_f1:-$min_best_f1}"
  min_last_calibrated_f1="${min_last_calibrated_f1:-$min_calibrated_f1}"
  min_last_average_precision="${min_last_average_precision:-$min_average_precision}"
  min_last_max_rank_f1="${min_last_max_rank_f1:-$min_max_rank_f1}"
  min_last_probability_gap="${min_last_probability_gap:-$min_probability_gap}"
fi

eval_every_steps="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS:-0}"
if [[ -n "${ANTFLY_FUSED_CHUNKER_MIN_VALIDATIONS:-}" ]]; then
  min_validations="$ANTFLY_FUSED_CHUNKER_MIN_VALIDATIONS"
elif [[ "$eval_every_steps" =~ ^[0-9]+$ && "$min_steps" =~ ^[0-9]+$ && "$eval_every_steps" -gt 0 && "$min_steps" -ge "$eval_every_steps" ]]; then
  min_validations=1
else
  min_validations=0
fi

mkdir -p "$out_dir"

export ANTFLY_FUSED_CHUNKER_OUTPUT="$out_dir"
export ANTFLY_FUSED_CHUNKER_MEMORY_ABORT_RSS_GB="$memory_abort_rss_gb"
export ANTFLY_FUSED_CHUNKER_MEMORY_WARN_RSS_GB="$memory_warn_rss_gb"
export ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION="${ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION:-mpsgraph_required}"
export TERMITE_MPSGRAPH_SMOKE="${TERMITE_MPSGRAPH_SMOKE:-1}"
export TERMITE_METAL_DISABLE_DEVICE_DENSE_LINEAR_FORWARD="$dense_device_forward"

echo "fused chunker readiness mode=$mode out_dir=$out_dir log=$log_path"
echo "gates min_steps=$min_steps min_validations=$min_validations min_fixed_f1=$min_fixed_f1 min_best_f1=$min_best_f1 min_calibrated_f1=$min_calibrated_f1 min_average_precision=$min_average_precision min_max_rank_f1=$min_max_rank_f1 min_probability_gap=$min_probability_gap min_last_fixed_f1=${min_last_fixed_f1:-skip} min_last_best_f1=${min_last_best_f1:-skip} min_last_calibrated_f1=${min_last_calibrated_f1:-skip} min_last_average_precision=${min_last_average_precision:-skip} min_last_max_rank_f1=${min_last_max_rank_f1:-skip} min_last_probability_gap=${min_last_probability_gap:-skip} max_avg_step_ms=$max_avg_step_ms max_peak_rss_gb=$max_peak_rss_gb require_encoder_neftune=$require_encoder_neftune"
echo "checkpoint_eval enabled=$checkpoint_eval max_examples=$checkpoint_eval_max_examples results=$checkpoint_eval_results dense_device_forward_disabled=$dense_device_forward analysis=$analysis_path"

probability_gap_args=()
if [[ "$min_probability_gap" != "skip" ]]; then
  probability_gap_args=(--min-mean-positive-probability-gap "$min_probability_gap")
fi
calibrated_args=()
if [[ "$min_calibrated_f1" != "skip" ]]; then
  calibrated_args=(--min-calibrated-f1 "$min_calibrated_f1")
fi
average_precision_args=()
if [[ "$min_average_precision" != "skip" ]]; then
  average_precision_args=(--min-average-precision "$min_average_precision")
fi
max_rank_args=()
if [[ "$min_max_rank_f1" != "skip" ]]; then
  max_rank_args=(--min-max-rank-f1 "$min_max_rank_f1")
fi
last_validation_args=()
if [[ -n "$min_last_fixed_f1" && "$min_last_fixed_f1" != "skip" ]]; then
  last_validation_args+=(--min-last-fixed-f1 "$min_last_fixed_f1")
fi
if [[ -n "$min_last_best_f1" && "$min_last_best_f1" != "skip" ]]; then
  last_validation_args+=(--min-last-best-f1 "$min_last_best_f1")
fi
if [[ -n "$min_last_calibrated_f1" && "$min_last_calibrated_f1" != "skip" ]]; then
  last_validation_args+=(--min-last-calibrated-f1 "$min_last_calibrated_f1")
fi
if [[ -n "$min_last_average_precision" && "$min_last_average_precision" != "skip" ]]; then
  last_validation_args+=(--min-last-average-precision "$min_last_average_precision")
fi
if [[ -n "$min_last_max_rank_f1" && "$min_last_max_rank_f1" != "skip" ]]; then
  last_validation_args+=(--min-last-max-rank-f1 "$min_last_max_rank_f1")
fi
if [[ -n "$min_last_probability_gap" && "$min_last_probability_gap" != "skip" ]]; then
  last_validation_args+=(--min-last-mean-positive-probability-gap "$min_last_probability_gap")
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

checkpoint_eval_args=()
if [[ "$checkpoint_eval" == "1" || "$checkpoint_eval" == "true" || "$checkpoint_eval" == "TRUE" || "$checkpoint_eval" == "yes" || "$checkpoint_eval" == "YES" ]]; then
  final_checkpoint="$out_dir/checkpoint_final.safetensors"
  if [[ ! -f "$final_checkpoint" ]]; then
    echo "missing final checkpoint for persisted eval: $final_checkpoint" >&2
    exit 1
  fi

  eval_args=(
    --data "$val_data"
    --checkpoint "$final_checkpoint"
    --model-dir "$model_dir"
    --split "$val_split"
    --backend metal
    --batch-size "$batch_size"
    --max-seq-len "$max_seq_len"
    --max-chunks "$max_chunks"
    --results-out "$checkpoint_eval_results"
    --benchmark-dataset-name "phase20_checkpoint_eval"
  )
  if [[ "$checkpoint_eval_max_examples" != "0" ]]; then
    eval_args+=(--max-examples "$checkpoint_eval_max_examples")
  fi

  (
    cd "$pkg_root"
    "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true eval-fused-chunker -- "${eval_args[@]}"
  )

  checkpoint_eval_args=(--checkpoint-eval-results "$checkpoint_eval_results")
  checkpoint_min_fixed="${min_checkpoint_fixed_f1:-$min_fixed_f1}"
  checkpoint_min_best="${min_checkpoint_best_f1:-$min_best_f1}"
  checkpoint_min_rank="${min_checkpoint_max_rank_f1:-$checkpoint_min_best}"
  if [[ "$checkpoint_min_fixed" != "skip" ]]; then
    checkpoint_eval_args+=(--min-checkpoint-fixed-f1 "$checkpoint_min_fixed")
  fi
  if [[ "$checkpoint_min_best" != "skip" ]]; then
    checkpoint_eval_args+=(--min-checkpoint-best-f1 "$checkpoint_min_best")
  fi
  if [[ "$min_checkpoint_average_precision" != "skip" ]]; then
    checkpoint_eval_args+=(--min-checkpoint-average-precision "$min_checkpoint_average_precision")
  fi
  if [[ "$checkpoint_min_rank" != "skip" ]]; then
    checkpoint_eval_args+=(--min-checkpoint-max-rank-f1 "$checkpoint_min_rank")
  fi
fi

validator_args=(
  --out-dir "$out_dir"
  --require-backend metal
  --require-mpsgraph-vjp
  --require-encoder-neftune "$require_encoder_neftune"
  --max-vjp-fallbacks 0
  --min-steps "$min_steps"
  --min-validations "$min_validations"
  --min-fixed-f1 "$min_fixed_f1"
  --min-best-f1 "$min_best_f1"
)
if ((${#average_precision_args[@]} > 0)); then
  validator_args+=("${average_precision_args[@]}")
fi
if ((${#calibrated_args[@]} > 0)); then
  validator_args+=("${calibrated_args[@]}")
fi
if ((${#max_rank_args[@]} > 0)); then
  validator_args+=("${max_rank_args[@]}")
fi
if ((${#probability_gap_args[@]} > 0)); then
  validator_args+=("${probability_gap_args[@]}")
fi
if ((${#last_validation_args[@]} > 0)); then
  validator_args+=("${last_validation_args[@]}")
fi
if ((${#checkpoint_eval_args[@]} > 0)); then
  validator_args+=("${checkpoint_eval_args[@]}")
fi
validator_args+=(
  --max-avg-step-ms "$max_avg_step_ms"
  --max-peak-rss-gb "$max_peak_rss_gb"
)
if [[ -n "${ANTFLY_FUSED_CHUNKER_REQUIRE_COMPLETE:-}" ]]; then
  validator_args+=(--require-complete)
fi

set +e
(
  cd "$pkg_root"
  "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true validate-fused-chunker-run -- "${validator_args[@]}"
) | tee "$summary_path"
validation_status="${PIPESTATUS[0]}"
set -e

if [[ "$validation_status" -ne 0 ]]; then
  echo "fused chunker readiness validation failed status=$validation_status; writing analysis=$analysis_path" >&2
  if [[ -f "$out_dir/fused_training_metrics.jsonl" ]]; then
    (
      cd "$pkg_root"
      "$zig_bin" build -Dskip-openapi=true -Dmetal=false analyze-fused-chunker-validation-failure -- \
        --out-dir "$out_dir" \
        --analysis-out "$analysis_path"
    ) || echo "warning: failed to write validation failure analysis" >&2
  else
    echo "warning: missing metrics for validation failure analysis: $out_dir/fused_training_metrics.jsonl" >&2
  fi
  exit "$validation_status"
fi

echo "fused chunker readiness passed summary=$summary_path"
