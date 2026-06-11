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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resume_from="${ANTFLY_FUSED_CHUNKER_RESUME_FROM:-}"
if [[ -z "$resume_from" ]]; then
  echo "set ANTFLY_FUSED_CHUNKER_RESUME_FROM to a stable checkpoint, e.g. checkpoint_step_4000.safetensors" >&2
  exit 2
fi

base_output="${ANTFLY_FUSED_CHUNKER_ABLATION_ROOT:-/private/tmp/zig-fused-quality-ablations}"
max_steps="${ANTFLY_FUSED_CHUNKER_ABLATION_MAX_STEPS:-4500}"
eval_every_steps="${ANTFLY_FUSED_CHUNKER_ABLATION_EVAL_EVERY_STEPS:-500}"
step_eval_max_examples="${ANTFLY_FUSED_CHUNKER_ABLATION_STEP_EVAL_MAX_EXAMPLES:-512}"
log_every="${ANTFLY_FUSED_CHUNKER_ABLATION_LOG_EVERY:-100}"

mkdir -p "$base_output"

run_case() {
  local name="$1"
  local pos_weight="$2"
  local lambda_embed="$3"
  local focus_lambda_embed="$4"
  local loss_type="$5"
  local focal_gamma="$6"
  local focal_alpha="$7"
  local output_dir="$base_output/$name"
  local log_path="$output_dir.log"

  echo "running ablation $name -> $output_dir"
  env \
    ANTFLY_FUSED_CHUNKER_OUTPUT="$output_dir" \
    ANTFLY_FUSED_CHUNKER_RESUME_FROM="$resume_from" \
    ANTFLY_FUSED_CHUNKER_MAX_STEPS="$max_steps" \
    ANTFLY_FUSED_CHUNKER_EVAL_EVERY=0 \
    ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS="$eval_every_steps" \
    ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES="$step_eval_max_examples" \
    ANTFLY_FUSED_CHUNKER_LOG_EVERY="$log_every" \
    ANTFLY_FUSED_CHUNKER_ENCODER_VJP=full \
    ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION=mpsgraph_required \
    ANTFLY_FUSED_CHUNKER_LAYERS_PER_SEGMENT=1 \
    ANTFLY_FUSED_CHUNKER_SAVE_OPTIMIZER_STATE=1 \
    ANTFLY_FUSED_CHUNKER_POS_WEIGHT="$pos_weight" \
    ANTFLY_FUSED_CHUNKER_LAMBDA_EMBED="$lambda_embed" \
    ANTFLY_FUSED_CHUNKER_BOUNDARY_FOCUS_LAMBDA_EMBED="$focus_lambda_embed" \
    ANTFLY_FUSED_CHUNKER_LOSS_TYPE="$loss_type" \
    ANTFLY_FUSED_CHUNKER_FOCAL_GAMMA="$focal_gamma" \
    ANTFLY_FUSED_CHUNKER_FOCAL_ALPHA="$focal_alpha" \
    "$script_dir/run_fused_chunker_phase20_metal.sh" >"$log_path" 2>&1
  "$script_dir/verify_fused_chunker_probe_log.sh" "$log_path"
  grep -E 'validation (step|epoch)|threshold_sweep|prob_hist|avg_step_ms' "$log_path" | tail -20
}

run_case "pos1_embed03_ce" "1.0" "0.3" "0.1" "ce" "2.0" "0.75"
run_case "pos5_embed03_ce" "5.0" "0.3" "0.1" "ce" "2.0" "0.75"
run_case "pos10_embed01_ce" "10.0" "0.1" "0.0" "ce" "2.0" "0.75"
run_case "pos5_embed00_ce" "5.0" "0.0" "0.0" "ce" "2.0" "0.75"
run_case "pos5_embed01_focal" "5.0" "0.1" "0.0" "focal" "2.0" "0.75"
