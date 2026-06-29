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

run_path="${1:-}"
if [[ -z "$run_path" ]]; then
  echo "usage: summarize_fused_chunker_readiness_run.sh <run-dir|fused_training_metrics.jsonl>" >&2
  exit 2
fi

if [[ -f "$run_path" ]]; then
  metrics_path="$run_path"
  run_dir="$(dirname -- "$metrics_path")"
else
  run_dir="$run_path"
  metrics_path="$run_dir/fused_training_metrics.jsonl"
fi
manifest_path="$run_dir/fused_training_manifest.json"

if [[ ! -f "$metrics_path" ]]; then
  echo "missing metrics file: $metrics_path" >&2
  exit 2
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "missing manifest file: $manifest_path" >&2
  exit 2
fi

jq -Rn --arg run_dir "$run_dir" --slurpfile manifest "$manifest_path" '
  def bytes_to_gib: if . == null then null else (. / 1073741824) end;
  def finite_numbers(xs): xs | map(select(type == "number"));
  [inputs | fromjson?] as $records |
  ($records | map(select(.event == "step"))) as $steps |
  ($records | map(select((.event // "") | startswith("validation_")))) as $validations |
  ($steps | last // {}) as $latest_step |
  {
    run_dir: $run_dir,
    manifest_status: ($manifest[0].status // null),
    manifest_backend: ($manifest[0].backend // null),
    manifest_total_steps: ($manifest[0].total_steps // 0),
    latest_step: ($latest_step.step // 0),
    latest_epoch: ($latest_step.epoch // null),
    latest_loss: ($latest_step.loss // null),
    latest_boundary_loss: ($latest_step.boundary_loss // null),
    latest_lr: ($latest_step.learning_rate // null),
    avg_step_ms: (
      if ($steps | length) == 0 then null
      else ((finite_numbers($steps | map(.step_wall_ms)) | add) / ($steps | length))
      end
    ),
    max_peak_rss_gib: (
      finite_numbers($steps | map(.peak_resident_bytes)) | max // ($manifest[0].peak_resident_bytes // 0) | bytes_to_gib
    ),
    max_vjp_fallbacks: (finite_numbers($steps | map(.vjp_interpreter_fallbacks)) | max // 0),
    mpsgraph_step_count: ($steps | map(select(.vjp_runtime == "mpsgraph")) | length),
    non_mpsgraph_vjp_step_count: (
      $steps | map(select((.vjp_runtime // "none") != "none" and (.vjp_runtime // "none") != "mpsgraph")) | length
    ),
    validation_count: ($validations | length),
    best_fixed_f1: (finite_numbers($validations | map(.f1)) | max // 0),
    best_threshold_f1: (finite_numbers($validations | map(.best_f1)) | max // 0),
    validations: (
      $validations |
      map({
        event,
        epoch,
        step,
        samples,
        total_samples,
        f1,
        precision,
        recall,
        tp,
        fp,
        fn,
        best_f1,
        best_threshold,
        best_tp,
        best_fp,
        best_fn,
        predicted_positive_rate,
        best_predicted_positive_rate,
        gold_positive_rate,
        mean_positive_probability_gold_positive,
        mean_positive_probability_gold_negative
      })
    )
  }
' "$metrics_path"
