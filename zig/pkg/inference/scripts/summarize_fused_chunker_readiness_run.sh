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
checkpoint_eval_path="$run_dir/checkpoint_eval_results.json"
checkpoint_eval_source="/dev/null"
if [[ -f "$checkpoint_eval_path" ]]; then
  checkpoint_eval_source="$checkpoint_eval_path"
fi

if [[ ! -f "$metrics_path" ]]; then
  echo "missing metrics file: $metrics_path" >&2
  exit 2
fi
if [[ ! -f "$manifest_path" ]]; then
  echo "missing manifest file: $manifest_path" >&2
  exit 2
fi

jq -Rn \
  --arg run_dir "$run_dir" \
  --arg checkpoint_eval_path "$checkpoint_eval_path" \
  --slurpfile manifest "$manifest_path" \
  --slurpfile checkpoint_eval "$checkpoint_eval_source" '
  def bytes_to_gib: if . == null then null else (. / 1073741824) end;
  def finite_numbers(xs): xs | map(select(type == "number"));
  def metric_max(xs): finite_numbers(xs) | max // 0;
  def metric_max_or_null(xs): finite_numbers(xs) as $nums | if ($nums | length) == 0 then null else ($nums | max) end;
  def latest_metric(xs): finite_numbers(xs) | last // null;
  def probability_gap:
    if (.mean_positive_probability_gold_positive | type) == "number" and
       (.mean_positive_probability_gold_negative | type) == "number"
    then (.mean_positive_probability_gold_positive - .mean_positive_probability_gold_negative)
    else null
    end;
  def field_or_null(k): if type == "object" and has(k) then .[k] else null end;
  def checkpoint_boundary:
    if ($checkpoint_eval | length) == 0 then null
    else ($checkpoint_eval[0].boundary_f1 // null)
    end;
  [inputs | fromjson?] as $records |
  ($records | map(select(.event == "step"))) as $steps |
  ($records | map(select((.event // "") | startswith("validation_")))) as $validations |
  ($records | map(select((.event // "") | startswith("train_validation_")))) as $train_validations |
  ($records | map(select(.event == "frozen_feature_probe")) | last // {}) as $frozen_feature_probe |
  ($steps | last // {}) as $latest_step |
  ($validations | last // {}) as $latest_validation |
  ($train_validations | last // {}) as $latest_train_validation |
  (checkpoint_boundary) as $checkpoint_boundary |
  {
    run_dir: $run_dir,
    manifest_status: ($manifest[0].status // null),
    manifest_backend: ($manifest[0].backend // null),
    manifest_encoder_neftune: ($manifest[0] | field_or_null("encoder_neftune")),
    manifest_total_steps: ($manifest[0].total_steps // 0),
    manifest_best_val_f1: ($manifest[0].best_val_f1 // null),
    manifest_recommended_boundary_threshold_available: ($manifest[0].recommended_boundary_threshold_available // false),
    manifest_recommended_boundary_threshold: ($manifest[0].recommended_boundary_threshold // null),
    manifest_recommended_boundary_threshold_f1: ($manifest[0].recommended_boundary_threshold_f1 // null),
    manifest_recommended_boundary_threshold_fixed_f1: ($manifest[0].recommended_boundary_threshold_fixed_f1 // null),
    manifest_recommended_boundary_threshold_average_precision: ($manifest[0].recommended_boundary_threshold_average_precision // null),
    manifest_recommended_boundary_threshold_max_rank_f1: ($manifest[0].recommended_boundary_threshold_max_rank_f1 // null),
    manifest_recommended_boundary_threshold_source: ($manifest[0].recommended_boundary_threshold_source // null),
    manifest_recommended_boundary_threshold_step: ($manifest[0].recommended_boundary_threshold_step // null),
    manifest_recommended_boundary_threshold_epoch: ($manifest[0].recommended_boundary_threshold_epoch // null),
    manifest_recommended_boundary_threshold_samples: ($manifest[0].recommended_boundary_threshold_samples // null),
    manifest_recommended_boundary_threshold_total_samples: ($manifest[0].recommended_boundary_threshold_total_samples // null),
    manifest_recommended_boundary_threshold_full_validation: ($manifest[0] | field_or_null("recommended_boundary_threshold_full_validation")),
    latest_step: ($latest_step.step // 0),
    latest_epoch: ($latest_step.epoch // null),
    latest_loss: ($latest_step.loss // null),
    latest_boundary_loss: ($latest_step.boundary_loss // null),
    latest_boundary_ce_loss: ($latest_step.boundary_ce_loss // null),
    latest_boundary_rank_loss: ($latest_step.boundary_rank_loss // null),
    latest_boundary_local_window_loss: ($latest_step.boundary_local_window_loss // null),
    latest_lr: ($latest_step.learning_rate // null),
    latest_boundary_head_lr: ($latest_step.boundary_head_learning_rate // null),
    avg_boundary_rank_loss: (
      finite_numbers($steps | map(.boundary_rank_loss)) as $nums |
      if ($nums | length) == 0 then null else (($nums | add) / ($nums | length)) end
    ),
    avg_boundary_local_window_loss: (
      finite_numbers($steps | map(.boundary_local_window_loss)) as $nums |
      if ($nums | length) == 0 then null else (($nums | add) / ($nums | length)) end
    ),
    max_boundary_rank_loss: (metric_max_or_null($steps | map(.boundary_rank_loss))),
    max_boundary_local_window_loss: (metric_max_or_null($steps | map(.boundary_local_window_loss))),
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
    train_validation_count: ($train_validations | length),
    latest_validation_step: ($latest_validation.step // null),
    latest_train_validation_step: ($latest_train_validation.step // null),
    latest_fixed_f1: (latest_metric($validations | map(.f1))),
    latest_best_threshold_f1: (latest_metric($validations | map(.best_f1))),
    latest_calibrated_boundary_f1: (latest_metric($validations | map(.calibrated_boundary_f1))),
    latest_calibrated_boundary_threshold: (latest_metric($validations | map(.calibrated_boundary_threshold))),
    latest_calibrated_predicted_positive_rate: (latest_metric($validations | map(.calibrated_predicted_positive_rate))),
    latest_fitted_boundary_f1: (latest_metric($validations | map(.fitted_boundary_f1))),
    latest_fitted_boundary_threshold: (latest_metric($validations | map(.fitted_boundary_threshold))),
    latest_fitted_predicted_positive_rate: (latest_metric($validations | map(.fitted_predicted_positive_rate))),
    latest_average_precision: (latest_metric($validations | map(.average_precision))),
    latest_max_rank_f1: (latest_metric($validations | map(.max_rank_f1))),
    latest_probability_gap: ($latest_validation | probability_gap),
    best_fixed_f1: (metric_max($validations | map(.f1))),
    best_threshold_f1: (metric_max($validations | map(.best_f1))),
    best_calibrated_boundary_f1: (metric_max_or_null($validations | map(.calibrated_boundary_f1))),
    best_fitted_boundary_f1: (metric_max_or_null($validations | map(.fitted_boundary_f1))),
    best_average_precision: (metric_max($validations | map(.average_precision))),
    best_max_rank_f1: (metric_max($validations | map(.max_rank_f1))),
    best_probability_gap: (
      finite_numbers($validations | map(probability_gap)) | max // null
    ),
    latest_train_fixed_f1: (latest_metric($train_validations | map(.f1))),
    latest_train_best_threshold_f1: (latest_metric($train_validations | map(.best_f1))),
    latest_train_calibrated_boundary_f1: (latest_metric($train_validations | map(.calibrated_boundary_f1))),
    latest_train_calibrated_boundary_threshold: (latest_metric($train_validations | map(.calibrated_boundary_threshold))),
    latest_train_calibrated_predicted_positive_rate: (latest_metric($train_validations | map(.calibrated_predicted_positive_rate))),
    latest_train_fitted_boundary_f1: (latest_metric($train_validations | map(.fitted_boundary_f1))),
    latest_train_fitted_boundary_threshold: (latest_metric($train_validations | map(.fitted_boundary_threshold))),
    latest_train_fitted_predicted_positive_rate: (latest_metric($train_validations | map(.fitted_predicted_positive_rate))),
    latest_train_average_precision: (latest_metric($train_validations | map(.average_precision))),
    latest_train_max_rank_f1: (latest_metric($train_validations | map(.max_rank_f1))),
    latest_train_probability_gap: ($latest_train_validation | probability_gap),
    frozen_feature_probe: (
      if ($frozen_feature_probe | has("event")) then {
        status: ($frozen_feature_probe.status // null),
        train_examples: ($frozen_feature_probe.train_examples // null),
        val_examples: ($frozen_feature_probe.val_examples // null),
        train_offset: ($frozen_feature_probe.train_offset // null),
        val_offset: ($frozen_feature_probe.val_offset // null),
        epochs: ($frozen_feature_probe.epochs // null),
        lr: ($frozen_feature_probe.lr // null),
        final_step: ($frozen_feature_probe.final_step // null),
        final_boundary_loss: ($frozen_feature_probe.final_boundary_loss // null),
        final_total_loss: ($frozen_feature_probe.final_total_loss // null),
        train_best_f1: ($frozen_feature_probe.train_best_f1 // null),
        train_fixed_f1: ($frozen_feature_probe.train_fixed_f1 // null),
        train_max_rank_f1: ($frozen_feature_probe.train_max_rank_f1 // null),
        train_average_precision: ($frozen_feature_probe.train_average_precision // null),
        train_probability_gap: ($frozen_feature_probe.train_probability_gap // null),
        val_best_f1: ($frozen_feature_probe.val_best_f1 // null),
        val_fixed_f1: ($frozen_feature_probe.val_fixed_f1 // null),
        val_max_rank_f1: ($frozen_feature_probe.val_max_rank_f1 // null),
        val_average_precision: ($frozen_feature_probe.val_average_precision // null),
        val_probability_gap: ($frozen_feature_probe.val_probability_gap // null)
      } else null end
    ),
    checkpoint_eval: (
      if $checkpoint_boundary == null then null
      else {
        path: $checkpoint_eval_path,
        fixed_threshold_f1: ($checkpoint_boundary.fixed_threshold_f1 // null),
        best_threshold_f1: ($checkpoint_boundary.best_threshold_f1 // null),
        average_precision: ($checkpoint_boundary.average_precision // null),
        precision_at_gold_count: ($checkpoint_boundary.precision_at_gold_count // null),
        max_rank_f1: ($checkpoint_boundary.max_rank_f1 // null),
        max_rank_threshold: ($checkpoint_boundary.max_rank_threshold // null)
      }
      end
    ),
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
        calibrated_threshold_available,
        calibrated_boundary_threshold,
        calibrated_boundary_f1,
        calibrated_boundary_precision,
        calibrated_boundary_recall,
        calibrated_boundary_tp,
        calibrated_boundary_fp,
        calibrated_boundary_fn,
        calibrated_predicted_positive_rate,
        fitted_threshold_available,
        fitted_boundary_threshold,
        fitted_boundary_f1,
        fitted_boundary_precision,
        fitted_boundary_recall,
        fitted_boundary_tp,
        fitted_boundary_fp,
        fitted_boundary_fn,
        fitted_predicted_positive_rate,
        predicted_positive_rate,
        best_predicted_positive_rate,
        gold_positive_rate,
        average_precision,
        precision_at_gold_count,
        max_rank_f1,
        max_rank_precision,
        max_rank_recall,
        max_rank_threshold,
        gold_positive_mean_rank,
        gold_positive_mean_rank_percentile,
        gold_positive_median_rank,
        gold_positive_median_rank_percentile,
        gold_positive_p90_rank,
        gold_positive_p90_rank_percentile,
        gold_positive_p99_rank,
        gold_positive_p99_rank_percentile,
        gold_positive_worst_rank,
        gold_positive_top_5x_count,
        gold_positive_top_5x_recall,
        gold_positive_top_10x_count,
        gold_positive_top_10x_recall,
        mean_positive_probability_gold_positive,
        mean_positive_probability_gold_negative,
        probability_gap: probability_gap
      })
    ),
    train_validations: (
      $train_validations |
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
        calibrated_threshold_available,
        calibrated_boundary_threshold,
        calibrated_boundary_f1,
        calibrated_boundary_precision,
        calibrated_boundary_recall,
        calibrated_boundary_tp,
        calibrated_boundary_fp,
        calibrated_boundary_fn,
        calibrated_predicted_positive_rate,
        fitted_threshold_available,
        fitted_boundary_threshold,
        fitted_boundary_f1,
        fitted_boundary_precision,
        fitted_boundary_recall,
        fitted_boundary_tp,
        fitted_boundary_fp,
        fitted_boundary_fn,
        fitted_predicted_positive_rate,
        predicted_positive_rate,
        best_predicted_positive_rate,
        gold_positive_rate,
        average_precision,
        precision_at_gold_count,
        max_rank_f1,
        max_rank_precision,
        max_rank_recall,
        max_rank_threshold,
        gold_positive_mean_rank,
        gold_positive_mean_rank_percentile,
        gold_positive_median_rank,
        gold_positive_median_rank_percentile,
        gold_positive_p90_rank,
        gold_positive_p90_rank_percentile,
        gold_positive_p99_rank,
        gold_positive_p99_rank_percentile,
        gold_positive_worst_rank,
        gold_positive_top_5x_count,
        gold_positive_top_5x_recall,
        gold_positive_top_10x_count,
        gold_positive_top_10x_recall,
        mean_positive_probability_gold_positive,
        mean_positive_probability_gold_negative,
        probability_gap: probability_gap
      })
    )
  }
' "$metrics_path"
