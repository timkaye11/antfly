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
pkg_root="$(cd -- "$script_dir/.." && pwd)"

zig_bin="${ANTFLY_ZIG_BIN:-}"
if [[ -z "$zig_bin" ]]; then
  if command -v zig >/dev/null 2>&1; then
    zig_bin="$(command -v zig)"
  elif [[ -x "$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig" ]]; then
    zig_bin="$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig"
  else
    echo "missing Zig compiler; set ANTFLY_ZIG_BIN=/path/to/zig" >&2
    exit 1
  fi
fi

out_dir="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_OUT:-/tmp/fused_chunker_go_zig_step_parity}"
checkpoint="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_CHECKPOINT:-/private/tmp/zig-fused-chunker-readiness/20260617-164911/checkpoint_final.safetensors}"
checkpoint_optimizer="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_OPTIMIZER:-${checkpoint%.safetensors}_optimizer.safetensors}"
step="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_STEP:-201}"
checkpoint_step_count="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_CHECKPOINT_STEP:-200}"
go_update_step="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_GO_UPDATE_STEP:-$checkpoint_step_count}"
split="${ANTFLY_FUSED_CHUNKER_PARITY_SPLIT:-train}"
default_offset=1600
case "$split" in
  val|validation)
    default_offset=0
    ;;
esac
offset="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_OFFSET:-$default_offset}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"
limit="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_LIMIT:-$batch_size}"
max_steps="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_MAX_STEPS:-$step}"
skip_batch="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_SKIP_BATCH:-0}"
skip_go_step="${ANTFLY_FUSED_CHUNKER_STEP_PARITY_SKIP_GO:-0}"
go_step_mode="${ANTFLY_FUSED_CHUNKER_GO_STEP_MODE:-skip}"
go_step_backend="${ANTFLY_FUSED_CHUNKER_GO_STEP_BACKEND:-mpsgraph}"
go_mixed_precision="${ANTFLY_FUSED_CHUNKER_GO_MIXED_PRECISION:-none}"
zig_mixed_precision="${ANTFLY_FUSED_CHUNKER_ZIG_MIXED_PRECISION:-0}"
debug_encoder_probe_layer="${ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_PROBE_LAYER:-0}"
debug_encoder_layer_inputs_only="${ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_LAYER_INPUTS_ONLY:-0}"
dump_encoder_replay_input="${ANTFLY_FUSED_CHUNKER_DUMP_ENCODER_REPLAY_INPUT:-0}"
dump_encoder_replay_upstream="${ANTFLY_FUSED_CHUNKER_DUMP_ENCODER_REPLAY_UPSTREAM:-0}"
layer_backward_decomp="${ANTFLY_FUSED_CHUNKER_LAYER_BACKWARD_DECOMP:-0}"
qkv_split_vjp="${ANTFLY_FUSED_CHUNKER_QKV_SPLIT_VJP:-0}"
zig_repeatability_repeats="${ANTFLY_FUSED_CHUNKER_ZIG_REPEATABILITY_REPEATS:-1}"
zig_profile_baseline="${ANTFLY_FUSED_CHUNKER_ZIG_PROFILE_BASELINE:-}"
zig_profile_abs_tol="${ANTFLY_FUSED_CHUNKER_ZIG_PROFILE_ABS_TOL:-1e-5}"
zig_profile_rel_tol="${ANTFLY_FUSED_CHUNKER_ZIG_PROFILE_REL_TOL:-1e-3}"
zig_profile_survey_repeats="${ANTFLY_FUSED_CHUNKER_ZIG_PROFILE_SURVEY_REPEATS:-1}"

gopeft_dir="${ANTFLY_GOPEFT_DIR:-/Users/tim/Documents/af/gopeft}"
default_data_path="$gopeft_dir/data/fused_train.jsonl"
case "$split" in
  val|validation)
    default_data_path="$gopeft_dir/data/fused_val.jsonl"
    ;;
esac
data_path="${ANTFLY_FUSED_CHUNKER_PARITY_DATA:-$default_data_path}"
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-embed-base}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
go_timeout="${ANTFLY_FUSED_CHUNKER_GO_STEP_TIMEOUT:-30m}"

batch_dir="$out_dir/batch"
zig_no_update_dir="$out_dir/zig_no_update_run"
zig_apply_update_dir="$out_dir/zig_apply_update_run"
zig_no_update_json="$out_dir/zig_step_no_update.json"
zig_apply_update_json="$out_dir/zig_step_apply_update.json"
zig_no_update_replay_input="$out_dir/zig_step_no_update_encoder_replay_input.f32"
zig_apply_update_replay_input="$out_dir/zig_step_apply_update_encoder_replay_input.f32"
zig_apply_update_replay_upstream="$out_dir/zig_step_apply_update_encoder_replay_upstream.f32"
go_step_json="$out_dir/go_step.json"
summary_json="$out_dir/parity_summary.json"
adapter_json="$out_dir/checkpoint_adapter_summary.json"
go_checkpoint="$out_dir/go_compatible_checkpoint.safetensors"
go_optimizer="$out_dir/go_compatible_optimizer.safetensors"
go_test_src="$out_dir/fused_step_parity_test.go"
go_empty_test_src="$out_dir/empty_finetune_test.go"
go_overlay_json="$out_dir/go_overlay.json"
go_stdout="$out_dir/go_step_stdout.log"
go_stderr="$out_dir/go_step_stderr.log"
zig_repeatability_summary="$out_dir/zig_repeatability_summary.json"
zig_profile_baseline_summary="$out_dir/zig_profile_baseline_summary.json"

mkdir -p "$out_dir" "$batch_dir" "$zig_no_update_dir" "$zig_apply_update_dir"

if [[ ! -f "$checkpoint" ]]; then
  echo "missing checkpoint at $checkpoint" >&2
  exit 1
fi
if [[ ! -d "$gopeft_dir" ]]; then
  echo "missing gopeft checkout at $gopeft_dir" >&2
  exit 1
fi
if [[ ! -f "$data_path" ]]; then
  echo "missing fused data at $data_path" >&2
  exit 1
fi
if [[ ! -f "$model_dir/tokenizer.json" ]]; then
  echo "missing tokenizer at $model_dir/tokenizer.json" >&2
  exit 1
fi
case "$go_step_mode" in
  skip|full) ;;
  *)
    echo "invalid ANTFLY_FUSED_CHUNKER_GO_STEP_MODE=$go_step_mode; expected skip or full" >&2
    exit 1
    ;;
esac
if ! [[ "$zig_repeatability_repeats" =~ ^[0-9]+$ ]] || (( zig_repeatability_repeats < 1 )); then
  echo "invalid ANTFLY_FUSED_CHUNKER_ZIG_REPEATABILITY_REPEATS=$zig_repeatability_repeats; expected integer >= 1" >&2
  exit 1
fi
if [[ -n "$zig_profile_baseline" && ! -f "$zig_profile_baseline" ]]; then
  echo "missing ANTFLY_FUSED_CHUNKER_ZIG_PROFILE_BASELINE at $zig_profile_baseline" >&2
  exit 1
fi
if ! [[ "$zig_profile_survey_repeats" =~ ^[0-9]+$ ]] || (( zig_profile_survey_repeats < 1 )); then
  echo "invalid ANTFLY_FUSED_CHUNKER_ZIG_PROFILE_SURVEY_REPEATS=$zig_profile_survey_repeats; expected integer >= 1" >&2
  exit 1
fi

echo "fused chunker frozen-step parity"
echo "  out_dir=$out_dir"
echo "  checkpoint=$checkpoint"
echo "  checkpoint_optimizer=$checkpoint_optimizer"
echo "  step=$step"
echo "  checkpoint_step_count=$checkpoint_step_count"
echo "  go_update_step=$go_update_step"
echo "  offset=$offset"
echo "  batch_size=$batch_size"
echo "  limit=$limit"
echo "  split=$split"
echo "  data_path=$data_path"
echo "  go_step_mode=$go_step_mode"
echo "  go_step_backend=$go_step_backend"
echo "  debug_encoder_probe_layer=$debug_encoder_probe_layer"
echo "  debug_encoder_layer_inputs_only=$debug_encoder_layer_inputs_only"
echo "  dump_encoder_replay_input=$dump_encoder_replay_input"
echo "  dump_encoder_replay_upstream=$dump_encoder_replay_upstream"
echo "  layer_backward_decomp=$layer_backward_decomp"
echo "  qkv_split_vjp=$qkv_split_vjp"
echo "  zig_repeatability_repeats=$zig_repeatability_repeats"
echo "  zig_profile_baseline=$zig_profile_baseline"
echo "  zig_profile_survey_repeats=$zig_profile_survey_repeats"

if [[ "$skip_batch" != "1" ]]; then
  ANTFLY_FUSED_CHUNKER_PARITY_OUT="$batch_dir" \
  ANTFLY_FUSED_CHUNKER_PARITY_OFFSET="$offset" \
  ANTFLY_FUSED_CHUNKER_PARITY_LIMIT="$limit" \
  ANTFLY_FUSED_CHUNKER_BATCH_SIZE="$batch_size" \
    "$script_dir/run_fused_chunker_go_zig_batch_parity.sh"
fi

convert_args=(
  --checkpoint "$checkpoint"
  --out "$go_checkpoint"
  --summary "$adapter_json"
)
if [[ -f "$checkpoint_optimizer" ]]; then
  convert_args+=(--optimizer "$checkpoint_optimizer" --optimizer-out "$go_optimizer")
fi

echo "converting Zig checkpoint to Go tensor names"
(
  cd "$pkg_root"
  "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true convert-fused-chunker-checkpoint-for-go -- "${convert_args[@]}"
)

run_zig_step() {
  local phase="$1"
  local run_dir="$2"
  local json_path="$3"
  local update_exit="$4"
  local replay_input_path="$5"
  local replay_upstream_path="$6"
  local boundary_exit=1
  local replay_input_env=""
  local replay_upstream_env=""
  if [[ "$update_exit" == "1" ]]; then
    boundary_exit=0
  fi
  case "$dump_encoder_replay_input" in
    1|true|TRUE|yes|YES)
      replay_input_env="$replay_input_path"
      ;;
  esac
  case "$dump_encoder_replay_upstream" in
    1|true|TRUE|yes|YES)
      replay_upstream_env="$replay_upstream_path"
      ;;
  esac

  mkdir -p "$run_dir"
  echo "running Zig frozen step phase=$phase"
  ANTFLY_FUSED_CHUNKER_OUTPUT="$run_dir" \
  ANTFLY_FUSED_CHUNKER_RESUME_FROM="$checkpoint" \
  ANTFLY_FUSED_CHUNKER_TRAIN_DATA="$data_path" \
  ANTFLY_FUSED_CHUNKER_SPLIT="$split" \
  ANTFLY_FUSED_CHUNKER_DEBUG_BATCH_OFFSET="$offset" \
  ANTFLY_FUSED_CHUNKER_MAX_STEPS="$max_steps" \
  ANTFLY_FUSED_CHUNKER_DETERMINISTIC=1 \
  ANTFLY_FUSED_CHUNKER_MIXED_PRECISION="$zig_mixed_precision" \
  ANTFLY_FUSED_CHUNKER_LOG_EVERY=1 \
  ANTFLY_FUSED_CHUNKER_EVAL_EVERY=0 \
  ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS=0 \
  ANTFLY_FUSED_CHUNKER_CHECKPOINT_EVERY_STEPS=0 \
  ANTFLY_FUSED_CHUNKER_SPLADE=0 \
  ANTFLY_FUSED_CHUNKER_DEBUG_BOUNDARY_STEP="$step" \
  ANTFLY_FUSED_CHUNKER_DEBUG_UPDATE_STEP="$step" \
  ANTFLY_FUSED_CHUNKER_DEBUG_STEP_JSON="$json_path" \
  ANTFLY_FUSED_CHUNKER_DEBUG_BOUNDARY_STEP_EXIT="$boundary_exit" \
  ANTFLY_FUSED_CHUNKER_DEBUG_UPDATE_STEP_EXIT="$update_exit" \
  ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_PROBE_LAYER="$debug_encoder_probe_layer" \
  ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_LAYER_INPUTS_ONLY="$debug_encoder_layer_inputs_only" \
  ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_REPLAY_INPUT="$replay_input_env" \
  ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_REPLAY_UPSTREAM="$replay_upstream_env" \
  ANTFLY_FUSED_CHUNKER_DEBUG_LAYER_BACKWARD_DECOMP="$layer_backward_decomp" \
  ANTFLY_FUSED_CHUNKER_QKV_SPLIT_VJP="$qkv_split_vjp" \
    "$script_dir/run_fused_chunker_phase20_metal.sh" > "$run_dir/train.log" 2>&1

  if [[ ! -f "$json_path" ]]; then
    echo "missing Zig parity JSON: $json_path" >&2
    echo "tail of $run_dir/train.log:" >&2
    tail -n 80 "$run_dir/train.log" >&2 || true
    exit 1
  fi
}

run_zig_step "no_update" "$zig_no_update_dir" "$zig_no_update_json" 0 "$zig_no_update_replay_input" ""
run_zig_step "apply_update" "$zig_apply_update_dir" "$zig_apply_update_json" 1 "$zig_apply_update_replay_input" "$zig_apply_update_replay_upstream"

run_zig_repeatability_gate() {
  if (( zig_repeatability_repeats <= 1 )); then
    return
  fi

  local repeat_jsons=("$zig_apply_update_json")
  local repeat_idx
  for ((repeat_idx = 2; repeat_idx <= zig_repeatability_repeats; repeat_idx++)); do
    local repeat_dir="$out_dir/zig_apply_update_repeat_${repeat_idx}_run"
    local repeat_json="$out_dir/zig_step_apply_update_repeat_${repeat_idx}.json"
    local repeat_replay_input="$out_dir/zig_step_apply_update_repeat_${repeat_idx}_encoder_replay_input.f32"
    local repeat_replay_upstream="$out_dir/zig_step_apply_update_repeat_${repeat_idx}_encoder_replay_upstream.f32"
    run_zig_step "apply_update_repeat_${repeat_idx}" "$repeat_dir" "$repeat_json" 1 "$repeat_replay_input" "$repeat_replay_upstream"
    repeat_jsons+=("$repeat_json")
  done

  python3 - "$zig_repeatability_summary" "${repeat_jsons[@]}" <<'PYEOF'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
json_paths = [Path(path) for path in sys.argv[2:]]
abs_tol = 1e-6
rel_tol = 1e-5

def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_path(obj, path):
    cur = obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True

def add_field(fields, path, mode="float"):
    key = ".".join(path)
    if key not in fields:
        fields[key] = (path, mode)

def add_stat_fields(fields, prefix, obj):
    if isinstance(obj, dict):
        for stat_name in ("mean", "rms", "max_abs", "max_abs_value", "l2", "mean_abs"):
            if stat_name in obj:
                add_field(fields, prefix + [stat_name])
        for stat_name in ("max_abs_index", "hash"):
            if stat_name in obj:
                add_field(fields, prefix + [stat_name], "exact")

def add_probe_map_stat_fields(fields, prefix, obj):
    if not isinstance(obj, dict):
        return
    for name, stats in sorted(obj.items()):
        add_stat_fields(fields, prefix + [name], stats)

def add_layer_decomp_fields(fields, prefix, obj):
    stages = (obj or {}).get("stages") if isinstance(obj, dict) else None
    if not isinstance(stages, dict):
        return
    for stage_name, stage in sorted(stages.items()):
        if not isinstance(stage, dict):
            continue
        add_stat_fields(fields, prefix + ["stages", stage_name, "stats"], stage.get("stats"))
        add_stat_fields(fields, prefix + ["stages", stage_name, "adapter_a"], stage.get("adapter_a"))
        add_stat_fields(fields, prefix + ["stages", stage_name, "adapter_b"], stage.get("adapter_b"))
        for component_name, stats in sorted((stage.get("components") or {}).items()):
            add_stat_fields(fields, prefix + ["stages", stage_name, "components", component_name], stats)

docs = [load(path) for path in json_paths]
base = docs[0]
fields = {}

for name in ("sample_indices", "input_ids", "attention_mask", "labels", "chunks"):
    add_field(fields, ["hashes", name], "exact")
for name in ("batch_size", "max_seq_len", "max_chunks", "global_step", "target_probe_layer", "trainer_step_count"):
    add_field(fields, [name], "exact")
for name in (
    "loss",
    "prob_gold_pos",
    "prob_gold_neg",
    "features_grad_norm",
    "features_grad_max_abs",
    "grad_norm_w1",
    "grad_norm_w2",
    "grad_norm_b1",
    "grad_norm_b2",
):
    add_field(fields, ["boundary", name])

add_probe_map_stat_fields(fields, ["boundary_forward_probe"], base.get("boundary_forward_probe"))
add_probe_map_stat_fields(fields, ["embedding_probe"], base.get("embedding_probe"))
add_probe_map_stat_fields(fields, ["encoder_activation_input_probe"], base.get("encoder_activation_input_probe"))
add_probe_map_stat_fields(fields, ["encoder_layer_input_probe"], base.get("encoder_layer_input_probe"))
add_probe_map_stat_fields(fields, ["encoder_layer_state_probe"], base.get("encoder_layer_state_probe"))
add_probe_map_stat_fields(fields, ["encoder_attention_internal_probe"], base.get("encoder_attention_internal_probe"))

for row_name, row in sorted((base.get("encoder_attention_row_probe") or {}).items()):
    if not isinstance(row, dict):
        continue
    for stat_name in (
        "score_mean",
        "score_rms",
        "score_min",
        "score_max",
        "prob_entropy",
        "prob_max",
        "prob_top2_gap",
        "query_rms",
        "query_max_abs",
        "key_query_rms",
        "key_query_max_abs",
        "value_query_rms",
        "value_query_max_abs",
        "output_mean",
        "output_rms",
        "output_max_abs",
    ):
        if stat_name in row:
            add_field(fields, ["encoder_attention_row_probe", row_name, stat_name])
    for stat_name in ("score_argmax", "prob_argmax", "valid_keys"):
        if stat_name in row:
            add_field(fields, ["encoder_attention_row_probe", row_name, stat_name], "exact")

for stage_name, stats in sorted(((base.get("upstream_grad_probe") or {}).get("stages") or {}).items()):
    add_stat_fields(fields, ["upstream_grad_probe", "stages", stage_name], stats)
for stage_name, stats in sorted(((base.get("upstream_grad_probe") or {}).get("upper_encoder_ladder") or {}).items()):
    add_stat_fields(fields, ["upstream_grad_probe", "upper_encoder_ladder", stage_name], stats)
for stat_name in ("upstream", "hidden_grad", "adapter_a", "adapter_b"):
    add_stat_fields(fields, ["segment_vjp_probe", stat_name], (base.get("segment_vjp_probe") or {}).get(stat_name))
add_layer_decomp_fields(fields, ["layer_backward_decomp_probe"], base.get("layer_backward_decomp_probe"))

failures = []
max_abs_diff = 0.0
max_rel_diff = 0.0
worst_field = None
compared = 0

for repeat_index, doc in enumerate(docs[1:], start=2):
    for field_name, (path, mode) in fields.items():
        expected, expected_ok = get_path(base, path)
        observed, observed_ok = get_path(doc, path)
        compared += 1
        if not expected_ok or not observed_ok:
            failures.append({
                "repeat": repeat_index,
                "field": field_name,
                "reason": "missing",
                "expected_present": expected_ok,
                "observed_present": observed_ok,
            })
            continue
        if mode == "exact" or isinstance(expected, str) or isinstance(expected, bool):
            if expected != observed:
                failures.append({
                    "repeat": repeat_index,
                    "field": field_name,
                    "reason": "exact_mismatch",
                    "expected": expected,
                    "observed": observed,
                })
            continue
        if not isinstance(expected, (int, float)) or not isinstance(observed, (int, float)):
            if expected != observed:
                failures.append({
                    "repeat": repeat_index,
                    "field": field_name,
                    "reason": "type_or_value_mismatch",
                    "expected": expected,
                    "observed": observed,
                })
            continue
        if not math.isfinite(float(expected)) or not math.isfinite(float(observed)):
            if expected != observed:
                failures.append({
                    "repeat": repeat_index,
                    "field": field_name,
                    "reason": "nonfinite_mismatch",
                    "expected": expected,
                    "observed": observed,
                })
            continue
        abs_diff = abs(float(expected) - float(observed))
        denom = max(abs(float(expected)), abs(float(observed)), 1e-12)
        rel_diff = abs_diff / denom
        if abs_diff > max_abs_diff or rel_diff > max_rel_diff:
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
            worst_field = field_name
        if abs_diff > abs_tol and rel_diff > rel_tol:
            failures.append({
                "repeat": repeat_index,
                "field": field_name,
                "reason": "float_drift",
                "expected": expected,
                "observed": observed,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            })

summary = {
    "tool": "zig_repeatability_gate",
    "schema_version": 1,
    "status": "failed" if failures else "passed",
    "repeats": len(docs),
    "paths": [str(path) for path in json_paths],
    "field_count": len(fields),
    "comparison_count": compared,
    "failure_count": len(failures),
    "max_abs_diff": max_abs_diff,
    "max_rel_diff": max_rel_diff,
    "worst_field": worst_field,
    "abs_tol": abs_tol,
    "rel_tol": rel_tol,
    "failures": failures[:64],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if failures:
    sys.exit(1)
PYEOF
}

run_zig_repeatability_gate

run_zig_profile_baseline_gate() {
  if [[ -z "$zig_profile_baseline" ]]; then
    return
  fi

  local observed_args=(
    "no_update=$zig_no_update_json"
    "apply_update=$zig_apply_update_json"
  )
  local repeat_idx
  for ((repeat_idx = 2; repeat_idx <= zig_profile_survey_repeats; repeat_idx++)); do
    local no_update_repeat_dir="$out_dir/zig_no_update_profile_repeat_${repeat_idx}_run"
    local no_update_repeat_json="$out_dir/zig_step_no_update_profile_repeat_${repeat_idx}.json"
    local no_update_repeat_replay_input="$out_dir/zig_step_no_update_profile_repeat_${repeat_idx}_encoder_replay_input.f32"
    run_zig_step "no_update_profile_repeat_${repeat_idx}" "$no_update_repeat_dir" "$no_update_repeat_json" 0 "$no_update_repeat_replay_input" ""
    observed_args+=("no_update_repeat_${repeat_idx}=$no_update_repeat_json")

    local apply_update_repeat_dir="$out_dir/zig_apply_update_profile_repeat_${repeat_idx}_run"
    local apply_update_repeat_json="$out_dir/zig_step_apply_update_profile_repeat_${repeat_idx}.json"
    local apply_update_repeat_replay_input="$out_dir/zig_step_apply_update_profile_repeat_${repeat_idx}_encoder_replay_input.f32"
    local apply_update_repeat_replay_upstream="$out_dir/zig_step_apply_update_profile_repeat_${repeat_idx}_encoder_replay_upstream.f32"
    run_zig_step "apply_update_profile_repeat_${repeat_idx}" "$apply_update_repeat_dir" "$apply_update_repeat_json" 1 "$apply_update_repeat_replay_input" "$apply_update_repeat_replay_upstream"
    observed_args+=("apply_update_repeat_${repeat_idx}=$apply_update_repeat_json")
  done

  python3 - "$zig_profile_baseline_summary" "$zig_profile_baseline" "$zig_profile_abs_tol" "$zig_profile_rel_tol" "${observed_args[@]}" <<'PYEOF'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
abs_tol = float(sys.argv[3])
rel_tol = float(sys.argv[4])
observed_paths = {}
for entry in sys.argv[5:]:
    if "=" not in entry:
        raise SystemExit(f"invalid observed profile entry {entry!r}; expected label=path")
    label, path = entry.split("=", 1)
    if not label or not path:
        raise SystemExit(f"invalid observed profile entry {entry!r}; expected label=path")
    observed_paths[label] = Path(path)

def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_path(obj, path):
    cur = obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True

def add_field(fields, path, mode="float"):
    key = ".".join(path)
    if key not in fields:
        fields[key] = (path, mode)

def add_stat_fields(fields, prefix, obj):
    if not isinstance(obj, dict):
        return
    if "hash" in obj:
        add_field(fields, prefix + ["hash"], "exact")
    for stat_name in ("mean", "rms", "max_abs", "l2", "mean_abs"):
        if stat_name in obj:
            add_field(fields, prefix + [stat_name])

def add_probe_map_stat_fields(fields, prefix, obj, target_layer=None):
    if not isinstance(obj, dict):
        return
    target_prefix = None
    if target_layer is not None:
        try:
            target_prefix = f"layer_{int(target_layer):02d}_"
        except Exception:
            target_prefix = None
    for name, stats in sorted(obj.items()):
        if target_prefix is not None and str(name).startswith("layer_") and not str(name).startswith(target_prefix):
            continue
        add_stat_fields(fields, prefix + [name], stats)

def add_attention_row_fields(fields, obj, target_layer=None):
    if not isinstance(obj, dict):
        return
    target_prefix = None
    if target_layer is not None:
        try:
            target_prefix = f"layer_{int(target_layer):02d}_"
        except Exception:
            target_prefix = None
    for row_name, row in sorted(obj.items()):
        if not isinstance(row, dict):
            continue
        if target_prefix is not None and str(row_name).startswith("layer_") and not str(row_name).startswith(target_prefix):
            continue
        for stat_name in (
            "score_mean",
            "score_rms",
            "score_min",
            "score_max",
            "prob_entropy",
            "prob_max",
            "prob_top2_gap",
            "query_rms",
            "query_max_abs",
            "key_query_rms",
            "key_query_max_abs",
            "value_query_rms",
            "value_query_max_abs",
            "output_mean",
            "output_rms",
            "output_max_abs",
        ):
            if stat_name in row:
                add_field(fields, ["encoder_attention_row_probe", row_name, stat_name])
        for stat_name in ("score_argmax", "prob_argmax", "valid_keys"):
            if stat_name in row:
                add_field(fields, ["encoder_attention_row_probe", row_name, stat_name], "exact")

def numeric_value(obj, path):
    value, ok = get_path(obj, path)
    if ok and isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None

def layer_sort_key(name):
    text = str(name)
    if text.startswith("layer_"):
        try:
            return (0, int(text.split("_", 1)[1]))
        except Exception:
            pass
    return (1, text)

def add_all_probe_map_stat_fields(fields, prefix, obj):
    if not isinstance(obj, dict):
        return
    for name, stats in sorted(obj.items(), key=lambda item: layer_sort_key(item[0])):
        add_stat_fields(fields, prefix + [name], stats)

def layer_input_profile(doc):
    out = {}
    layer_inputs = doc.get("encoder_layer_input_probe") or {}
    if not isinstance(layer_inputs, dict):
        return out
    for name, stats in sorted(layer_inputs.items(), key=lambda item: layer_sort_key(item[0])):
        if not isinstance(stats, dict):
            continue
        out[name] = {
            "mean": numeric_value(doc, ["encoder_layer_input_probe", name, "mean"]),
            "rms": numeric_value(doc, ["encoder_layer_input_probe", name, "rms"]),
            "max_abs": numeric_value(doc, ["encoder_layer_input_probe", name, "max_abs"]),
            "hash": stats.get("hash"),
        }
    return out

def layer_input_divergence_signal(baseline_doc, docs, rel_threshold):
    baseline_inputs = baseline_doc.get("encoder_layer_input_probe") or {}
    signal_by_phase = {}
    for phase, doc in sorted(docs.items()):
        observed_inputs = doc.get("encoder_layer_input_probe") or {}
        phase_signal = {
            "first_divergent_layer": None,
            "first_hash_mismatch_layer": None,
            "max_rms_rel_diff": 0.0,
            "max_rms_abs_diff": 0.0,
            "worst_layer": None,
            "rms_rel_by_layer": {},
            "hash_mismatch_by_layer": {},
        }
        layer_names = sorted(set(baseline_inputs.keys()) | set(observed_inputs.keys()), key=layer_sort_key)
        for layer_name in layer_names:
            baseline_stats = baseline_inputs.get(layer_name) or {}
            observed_stats = observed_inputs.get(layer_name) or {}
            expected_hash = baseline_stats.get("hash")
            observed_hash = observed_stats.get("hash")
            if expected_hash is not None or observed_hash is not None:
                hashes_match = expected_hash == observed_hash
                phase_signal["hash_mismatch_by_layer"][layer_name] = not hashes_match
                if phase_signal["first_hash_mismatch_layer"] is None and not hashes_match:
                    phase_signal["first_hash_mismatch_layer"] = layer_name
            expected = baseline_stats.get("rms")
            observed = observed_stats.get("rms")
            if not isinstance(expected, (int, float)) or not isinstance(observed, (int, float)):
                continue
            abs_diff = abs(float(expected) - float(observed))
            rel_diff = abs_diff / max(abs(float(expected)), abs(float(observed)), 1e-12)
            phase_signal["rms_rel_by_layer"][layer_name] = rel_diff
            if rel_diff > phase_signal["max_rms_rel_diff"]:
                phase_signal["max_rms_rel_diff"] = rel_diff
                phase_signal["max_rms_abs_diff"] = abs_diff
                phase_signal["worst_layer"] = layer_name
            if phase_signal["first_divergent_layer"] is None and rel_diff > rel_threshold:
                phase_signal["first_divergent_layer"] = layer_name
        signal_by_phase[phase] = phase_signal
    return signal_by_phase

def profile_snapshot(doc):
    target_layer = doc.get("target_probe_layer")
    target_prefix = f"layer_{int(target_layer):02d}" if isinstance(target_layer, int) else "layer_00"
    return {
        "target_probe_layer": target_layer,
        "hashes": doc.get("hashes"),
        "boundary": {
            "loss": numeric_value(doc, ["boundary", "loss"]),
            "prob_gold_pos": numeric_value(doc, ["boundary", "prob_gold_pos"]),
            "prob_gold_neg": numeric_value(doc, ["boundary", "prob_gold_neg"]),
            "features_grad_norm": numeric_value(doc, ["boundary", "features_grad_norm"]),
        },
        "forward": {
            "final_norm_input_rms": numeric_value(doc, ["boundary_forward_probe", "final_norm_input", "rms"]),
            "boundary_head_input_rms": numeric_value(doc, ["boundary_forward_probe", "boundary_head_input", "rms"]),
            "logits_rms": numeric_value(doc, ["boundary_forward_probe", "logits", "rms"]),
        },
        "target_layer": {
            "hidden_after_attn_rms": numeric_value(doc, ["encoder_layer_state_probe", f"{target_prefix}_hidden_after_attn", "rms"]),
            "hidden_after_attn_hash": (doc.get("encoder_layer_state_probe") or {}).get(f"{target_prefix}_hidden_after_attn", {}).get("hash"),
            "layer_output_rms": numeric_value(doc, ["encoder_layer_state_probe", f"{target_prefix}_layer_output", "rms"]),
            "layer_output_hash": (doc.get("encoder_layer_state_probe") or {}).get(f"{target_prefix}_layer_output", {}).get("hash"),
            "q_raw_rms": numeric_value(doc, ["encoder_attention_internal_probe", f"{target_prefix}_q_raw", "rms"]),
            "q_raw_hash": (doc.get("encoder_attention_internal_probe") or {}).get(f"{target_prefix}_q_raw", {}).get("hash"),
            "attn_token_delta_rms": numeric_value(doc, ["encoder_attention_internal_probe", f"{target_prefix}_attn_token_delta", "rms"]),
            "attn_token_delta_hash": (doc.get("encoder_attention_internal_probe") or {}).get(f"{target_prefix}_attn_token_delta", {}).get("hash"),
        },
        "layer_input_rms": {
            name: stats.get("rms")
            for name, stats in layer_input_profile(doc).items()
        },
        "layer_input_hash": {
            name: stats.get("hash")
            for name, stats in layer_input_profile(doc).items()
        },
    }

baseline = load(baseline_path)
observed_docs = {phase: load(path) for phase, path in observed_paths.items()}
fields = {}
target_layer = baseline.get("target_probe_layer")

for name in ("sample_indices", "input_ids", "attention_mask", "labels", "chunks"):
    add_field(fields, ["hashes", name], "exact")
for name in ("batch_size", "max_seq_len", "max_chunks", "global_step", "target_probe_layer"):
    add_field(fields, [name], "exact")
for name in (
    "loss",
    "prob_gold_pos",
    "prob_gold_neg",
    "features_grad_norm",
    "features_grad_max_abs",
    "grad_norm_w1",
    "grad_norm_w2",
    "grad_norm_b1",
    "grad_norm_b2",
):
    add_field(fields, ["boundary", name])

for name in ("final_norm_input", "boundary_head_input", "logits"):
    add_stat_fields(fields, ["boundary_forward_probe", name], (baseline.get("boundary_forward_probe") or {}).get(name))
add_all_probe_map_stat_fields(fields, ["encoder_layer_input_probe"], baseline.get("encoder_layer_input_probe"))
add_probe_map_stat_fields(fields, ["encoder_layer_state_probe"], baseline.get("encoder_layer_state_probe"), target_layer)
add_probe_map_stat_fields(fields, ["encoder_attention_internal_probe"], baseline.get("encoder_attention_internal_probe"), target_layer)
add_attention_row_fields(fields, baseline.get("encoder_attention_row_probe"), target_layer)

failures = []
max_abs_diff = 0.0
max_rel_diff = 0.0
worst_field = None
compared = 0

for phase, observed in observed_docs.items():
    for field_name, (path, mode) in fields.items():
        expected, expected_ok = get_path(baseline, path)
        observed_value, observed_ok = get_path(observed, path)
        compared += 1
        if not expected_ok or not observed_ok:
            failures.append({
                "phase": phase,
                "field": field_name,
                "reason": "missing",
                "expected_present": expected_ok,
                "observed_present": observed_ok,
            })
            continue
        if mode == "exact" or isinstance(expected, str) or isinstance(expected, bool):
            if expected != observed_value:
                failures.append({
                    "phase": phase,
                    "field": field_name,
                    "reason": "exact_mismatch",
                    "expected": expected,
                    "observed": observed_value,
                })
            continue
        if not isinstance(expected, (int, float)) or not isinstance(observed_value, (int, float)):
            if expected != observed_value:
                failures.append({
                    "phase": phase,
                    "field": field_name,
                    "reason": "type_or_value_mismatch",
                    "expected": expected,
                    "observed": observed_value,
                })
            continue
        if not math.isfinite(float(expected)) or not math.isfinite(float(observed_value)):
            if expected != observed_value:
                failures.append({
                    "phase": phase,
                    "field": field_name,
                    "reason": "nonfinite_mismatch",
                    "expected": expected,
                    "observed": observed_value,
                })
            continue
        abs_diff = abs(float(expected) - float(observed_value))
        denom = max(abs(float(expected)), abs(float(observed_value)), 1e-12)
        rel_diff = abs_diff / denom
        if abs_diff > max_abs_diff or rel_diff > max_rel_diff:
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)
            worst_field = f"{phase}:{field_name}"
        if abs_diff > abs_tol and rel_diff > rel_tol:
            failures.append({
                "phase": phase,
                "field": field_name,
                "reason": "profile_drift",
                "expected": expected,
                "observed": observed_value,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            })

phase_failure_counts = {phase: 0 for phase in observed_docs.keys()}
for failure in failures:
    phase = failure.get("phase")
    if phase in phase_failure_counts:
        phase_failure_counts[phase] += 1
phase_status = {
    phase: "failed" if count > 0 else "passed"
    for phase, count in sorted(phase_failure_counts.items())
}
failed_phases = [
    phase
    for phase, count in sorted(phase_failure_counts.items())
    if count > 0
]

summary = {
    "tool": "zig_profile_baseline_gate",
    "schema_version": 1,
    "status": "failed" if failures else "passed",
    "baseline_path": str(baseline_path),
    "observed_paths": {phase: str(path) for phase, path in observed_paths.items()},
    "observed_phase_count": len(observed_docs),
    "phase_status": phase_status,
    "failed_phases": failed_phases,
    "field_count": len(fields),
    "comparison_count": compared,
    "failure_count": len(failures),
    "max_abs_diff": max_abs_diff,
    "max_rel_diff": max_rel_diff,
    "worst_field": worst_field,
    "abs_tol": abs_tol,
    "rel_tol": rel_tol,
    "baseline_profile": profile_snapshot(baseline),
    "observed_profile": profile_snapshot(observed_docs.get("apply_update") or {}),
    "observed_profiles": {phase: profile_snapshot(doc) for phase, doc in observed_docs.items()},
    "layer_input_signal": layer_input_divergence_signal(baseline, observed_docs, rel_tol),
    "failures": failures[:64],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
if failures:
    sys.exit(1)
PYEOF
}

run_zig_profile_baseline_gate

write_go_test() {
  cat > "$go_test_src" <<'GOEOF'
package finetune

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gopeft/adapters/lora"

	_ "github.com/gomlx/go-coreml/mpsgraph/gomlx"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/backends/xla"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

const stepParityFNVOffset uint64 = 14695981039346656037
const stepParityFNVPrime uint64 = 1099511628211

type parityStats struct {
	Elems   int       `json:"elems"`
	L2      float64   `json:"l2"`
	MaxAbs  float64   `json:"max_abs"`
	MeanAbs float64   `json:"mean_abs"`
	Sample  []float32 `json:"sample,omitempty"`
}

type boundaryProbeStats struct {
	ValidTokens        int       `json:"valid_tokens"`
	GoldPositives      int       `json:"gold_positives"`
	PredictedPositives int       `json:"predicted_positives"`
	TP                 int       `json:"tp"`
	FP                 int       `json:"fp"`
	FN                 int       `json:"fn"`
	ProbGoldPos        float64   `json:"prob_gold_pos"`
	ProbGoldNeg        float64   `json:"prob_gold_neg"`
	CPUCELoss          float64   `json:"cpu_ce_loss"`
	Logit0Mean         float64   `json:"logit0_mean"`
	Logit1Mean         float64   `json:"logit1_mean"`
	MarginGoldPos      float64   `json:"margin_gold_pos"`
	MarginGoldNeg      float64   `json:"margin_gold_neg"`
	LogitPairSample    []float32 `json:"logit_pair_sample,omitempty"`
}

type tensorProbeStats struct {
	Elems         int       `json:"elems"`
	Mean          float64   `json:"mean"`
	RMS           float64   `json:"rms"`
	MaxAbs        float64   `json:"max_abs"`
	MaxAbsIndex   int       `json:"max_abs_index"`
	MaxAbsValue   float32   `json:"max_abs_value"`
	Hash          string    `json:"hash,omitempty"`
	Sample        []float32 `json:"sample,omitempty"`
	TopAbsIndices []int     `json:"top_abs_indices,omitempty"`
	TopAbsValues  []float32 `json:"top_abs_values,omitempty"`
}

type floatSliceProbeStats struct {
	Elems   int     `json:"elems"`
	Nonzero int     `json:"nonzero"`
	L2      float64 `json:"l2"`
	MaxAbs  float64 `json:"max_abs"`
	MeanAbs float64 `json:"mean_abs"`
}

type contrastiveStepProbe struct {
	ActiveChunks              int                  `json:"active_chunks"`
	ContrastiveLoss           float64              `json:"contrastive_loss"`
	TotalLoss                 float64              `json:"total_loss"`
	Embeddings                floatSliceProbeStats `json:"embeddings"`
	Grad                      floatSliceProbeStats `json:"grad"`
	FirstActiveIndex          *int                 `json:"first_active_index,omitempty"`
	FirstActiveDocID          *int                 `json:"first_active_doc_id,omitempty"`
	FirstActiveEmbeddingSample []float32           `json:"first_active_embedding_sample,omitempty"`
	FirstActiveGradSample     []float32            `json:"first_active_grad_sample,omitempty"`
	ActiveDocIDSample         []int                `json:"active_doc_id_sample,omitempty"`
	ActiveEmbeddingNormSample []float32            `json:"active_embedding_norm_sample,omitempty"`
	ActiveGradNormSample      []float32            `json:"active_grad_norm_sample,omitempty"`
}

type segmentVJPProfileProbe struct {
	Runtime string `json:"runtime"`
}

type segmentVJPProbe struct {
	TargetLayer        int                    `json:"target_layer"`
	SegmentStart       int                    `json:"segment_start"`
	SegmentEnd         int                    `json:"segment_end"`
	IncludeHiddenGrad  bool                   `json:"include_hidden_grad"`
	IncludeAdapterGrads bool                  `json:"include_adapter_grads"`
	Runtime            string                 `json:"runtime"`
	Profile            segmentVJPProfileProbe `json:"profile"`
	Upstream           parityStats            `json:"upstream"`
	HiddenGrad         parityStats            `json:"hidden_grad"`
	AdapterA           parityStats            `json:"adapter_a"`
	AdapterB           parityStats            `json:"adapter_b"`
	AdapterAByName     map[string]parityStats `json:"adapter_a_by_name"`
	AdapterBByName     map[string]parityStats `json:"adapter_b_by_name"`
}

type layerBackwardDecompStage struct {
	Status         string                 `json:"status"`
	Reason         string                 `json:"reason"`
	Stats          *parityStats           `json:"stats,omitempty"`
	Components     map[string]parityStats `json:"components,omitempty"`
	AdapterA       *parityStats           `json:"adapter_a,omitempty"`
	AdapterB       *parityStats           `json:"adapter_b,omitempty"`
	AdapterAByName map[string]parityStats `json:"adapter_a_by_name,omitempty"`
	AdapterBByName map[string]parityStats `json:"adapter_b_by_name,omitempty"`
}

type layerBackwardDecompProbe struct {
	Status       string                                `json:"status"`
	Reason       string                                `json:"reason,omitempty"`
	Version      int                                   `json:"version,omitempty"`
	TargetLayer  int                                   `json:"target_layer,omitempty"`
	SegmentStart int                                   `json:"segment_start,omitempty"`
	SegmentEnd   int                                   `json:"segment_end,omitempty"`
	Runtime      string                                `json:"runtime,omitempty"`
	Stages       map[string]layerBackwardDecompStage   `json:"stages,omitempty"`
}

type softmaxVJPCase struct {
	Status                 string      `json:"status"`
	Reason                 string      `json:"reason"`
	Outer                  int         `json:"outer"`
	Queries                int         `json:"queries"`
	Keys                   int         `json:"keys"`
	HasMask                bool        `json:"has_mask"`
	MaskBias               float32     `json:"mask_bias"`
	ScoresMasked           parityStats `json:"scores_masked"`
	Probs                  parityStats `json:"probs"`
	UpstreamProbsGrad      parityStats `json:"upstream_probs_grad"`
	ScoresMaskedGrad       parityStats `json:"scores_masked_grad"`
	CPUScoresMaskedGrad    parityStats `json:"cpu_scores_masked_grad"`
	CPUAbsError            parityStats `json:"cpu_abs_error"`
	ValidScoresMaskedGrad  parityStats `json:"valid_scores_masked_grad"`
	MaskedScoresMaskedGrad parityStats `json:"masked_scores_masked_grad"`
}

type softmaxVJPProbe struct {
	Status  string                    `json:"status"`
	Version int                       `json:"version"`
	Runtime string                    `json:"runtime"`
	Cases   map[string]softmaxVJPCase `json:"cases"`
}

type qkvSplitVJPCase struct {
	Status     string                 `json:"status"`
	Reason     string                 `json:"reason"`
	Batch      int                    `json:"batch"`
	SeqLen     int                    `json:"seq_len"`
	NumHeads   int                    `json:"num_heads"`
	HeadDim    int                    `json:"head_dim"`
	HiddenSize int                    `json:"hidden_size"`
	Outer      int                    `json:"outer"`
	Components map[string]parityStats `json:"components"`
}

type qkvSplitVJPProbe struct {
	Status  string                     `json:"status"`
	Version int                        `json:"version"`
	Runtime string                     `json:"runtime"`
	Cases   map[string]qkvSplitVJPCase `json:"cases"`
}

type projectionDecompositionProbe struct {
	Scale       float64          `json:"scale"`
	Rank        int              `json:"rank"`
	Rows        int              `json:"rows"`
	InDim       int              `json:"in_dim"`
	OutDim      int              `json:"out_dim"`
	HasBias     bool             `json:"has_bias"`
	Input       tensorProbeStats `json:"input"`
	Base        tensorProbeStats `json:"base"`
	LoRAA       tensorProbeStats `json:"lora_a"`
	LoRAB       tensorProbeStats `json:"lora_b"`
	Delta       tensorProbeStats `json:"delta"`
	Output      tensorProbeStats `json:"output"`
	Weight      tensorProbeStats `json:"weight"`
	Bias        tensorProbeStats `json:"bias"`
	LoRAAWeight tensorProbeStats `json:"lora_a_weight"`
	LoRABWeight tensorProbeStats `json:"lora_b_weight"`
	BaseReferenceError  tensorProbeStats `json:"base_reference_error"`
	LoRAAReferenceError tensorProbeStats `json:"lora_a_reference_error"`
	LoRABReferenceError tensorProbeStats `json:"lora_b_reference_error"`
	DeltaReferenceError tensorProbeStats `json:"delta_reference_error"`
	OutputReferenceError tensorProbeStats `json:"output_reference_error"`
}

type encoderLayerReplayProbe struct {
	Status                              string                                `json:"status"`
	Reason                              string                                `json:"reason,omitempty"`
	Path                                string                                `json:"path,omitempty"`
	TargetLayer                         int                                   `json:"target_layer,omitempty"`
	BatchSize                           int                                   `json:"batch_size,omitempty"`
	SeqLen                              int                                   `json:"seq_len,omitempty"`
	HiddenSize                          int                                   `json:"hidden_size,omitempty"`
	Elems                               int                                   `json:"elems,omitempty"`
	AttentionMaskValidLengths           []int                                 `json:"attention_mask_valid_lengths,omitempty"`
	Input                               tensorProbeStats                      `json:"input,omitempty"`
	ActualInputDiff                     tensorProbeStats                      `json:"actual_input_diff,omitempty"`
	ActualInputValidTokenDiff           tensorProbeStats                      `json:"actual_input_valid_token_diff,omitempty"`
	ActualInputPaddingTokenDiff         tensorProbeStats                      `json:"actual_input_padding_token_diff,omitempty"`
	EncoderActivationInputProbe          map[string]tensorProbeStats          `json:"encoder_activation_input_probe,omitempty"`
	EncoderAttentionInternalProbe        map[string]tensorProbeStats          `json:"encoder_attention_internal_probe,omitempty"`
	EncoderProjectionDecompositionProbe  map[string]projectionDecompositionProbe `json:"encoder_projection_decomposition_probe,omitempty"`
	EncoderLayerStateProbe               map[string]tensorProbeStats          `json:"encoder_layer_state_probe,omitempty"`
}

type encoderLayerBackwardReplayProbe struct {
	Status                   string                    `json:"status"`
	Reason                   string                    `json:"reason,omitempty"`
	InputPath                string                    `json:"input_path,omitempty"`
	UpstreamPath             string                    `json:"upstream_path,omitempty"`
	TargetLayer              int                       `json:"target_layer,omitempty"`
	BatchSize                int                       `json:"batch_size,omitempty"`
	SeqLen                   int                       `json:"seq_len,omitempty"`
	HiddenSize               int                       `json:"hidden_size,omitempty"`
	Elems                    int                       `json:"elems,omitempty"`
	Input                    parityStats               `json:"input,omitempty"`
	Upstream                 parityStats               `json:"upstream,omitempty"`
	SegmentVJPProbe          *segmentVJPProbe          `json:"segment_vjp_probe,omitempty"`
	LayerBackwardDecompProbe *layerBackwardDecompProbe `json:"layer_backward_decomp_probe,omitempty"`
}

type attentionRowProbe struct {
	Batch        int       `json:"batch"`
	Head         int       `json:"head"`
	Query        int       `json:"query"`
	ValidKeys    int       `json:"valid_keys"`
	ScoreMean   float64   `json:"score_mean"`
	ScoreRMS    float64   `json:"score_rms"`
	ScoreMin    float32   `json:"score_min"`
	ScoreMax    float32   `json:"score_max"`
	ScoreArgmax int       `json:"score_argmax"`
	ProbEntropy float64   `json:"prob_entropy"`
	ProbMax     float32   `json:"prob_max"`
	ProbArgmax  int       `json:"prob_argmax"`
	ProbTop2Gap float32   `json:"prob_top2_gap"`
	QueryRMS         float64   `json:"query_rms"`
	QueryMaxAbs      float32   `json:"query_max_abs"`
	KeyQueryRMS      float64   `json:"key_query_rms"`
	KeyQueryMaxAbs   float32   `json:"key_query_max_abs"`
	ValueQueryRMS    float64   `json:"value_query_rms"`
	ValueQueryMaxAbs float32   `json:"value_query_max_abs"`
	OutputMean       float64   `json:"output_mean"`
	OutputRMS        float64   `json:"output_rms"`
	OutputMaxAbs     float32   `json:"output_max_abs"`
	QuerySample      []float32 `json:"query_sample,omitempty"`
	KeyQuerySample   []float32 `json:"key_query_sample,omitempty"`
	ValueQuerySample []float32 `json:"value_query_sample,omitempty"`
	ScoreSample      []float32 `json:"score_sample,omitempty"`
	ProbSample       []float32 `json:"prob_sample,omitempty"`
	OutputSample     []float32 `json:"output_sample,omitempty"`
}

type boundaryForwardProbe struct {
	FinalNormInput       tensorProbeStats `json:"final_norm_input,omitempty"`
	BoundaryHeadInput    tensorProbeStats `json:"boundary_head_input"`
	Dense1PreActivation  tensorProbeStats `json:"dense1_pre_activation"`
	Dense1PostActivation tensorProbeStats `json:"dense1_post_activation"`
	Logits               tensorProbeStats `json:"logits"`
}

type upstreamGradProbe struct {
	Status             string                      `json:"status"`
	TargetLayer        int                         `json:"target_layer"`
	Stages             map[string]tensorProbeStats `json:"stages"`
	UpperEncoderLadder map[string]tensorProbeStats `json:"upper_encoder_ladder,omitempty"`
}

type embeddingProbe struct {
	WordEmbeddingWeight tensorProbeStats `json:"word_embedding_weight"`
	TokenLookup     tensorProbeStats `json:"token_lookup"`
	LayerNormOutput tensorProbeStats `json:"layer_norm_output"`
	LayerNormWeight tensorProbeStats `json:"layer_norm_weight"`
	LayerNormBias   tensorProbeStats `json:"layer_norm_bias"`
}

const embeddingRowProbeMax = 8

type boundaryCheckpointProbe struct {
	FinalNormWeight tensorProbeStats `json:"final_norm_weight"`
	FinalNormBias   tensorProbeStats `json:"final_norm_bias"`
	W1              tensorProbeStats `json:"w1"`
	B1              tensorProbeStats `json:"b1"`
	W2              tensorProbeStats `json:"w2"`
	B2              tensorProbeStats `json:"b2"`
}

type gradShapeSample struct {
	Name  string `json:"name"`
	Shape []int  `json:"shape"`
}

type stepParityBatchHashes struct {
	SampleIndices string `json:"sample_indices"`
	InputIDs      string `json:"input_ids"`
	AttentionMask string `json:"attention_mask"`
	Labels        string `json:"labels"`
	Chunks        string `json:"chunks"`
}

type parityOutput struct {
	Tool                string                 `json:"tool"`
	SchemaVersion       int                    `json:"schema_version"`
	Status              string                 `json:"status"`
	Backend             string                 `json:"backend"`
	MixedPrecision      string                 `json:"mixed_precision"`
	UseBF16             bool                   `json:"use_bf16"`
	Checkpoint          string                 `json:"checkpoint"`
	Optimizer           string                 `json:"optimizer,omitempty"`
	TargetProbeLayer    int                    `json:"target_probe_layer"`
	CheckpointStepCount int                    `json:"checkpoint_step_count"`
	UpdateStep          int                    `json:"update_step"`
	DiagnosticMode      string                 `json:"diagnostic_mode"`
	StepsPerEpoch       int                    `json:"steps_per_epoch"`
		BatchSize           int                    `json:"batch_size"`
	Offset              int                    `json:"offset"`
	Limit               int                    `json:"limit"`
	TrainableTensors    int                    `json:"trainable_tensors"`
	NoUpdate            map[string]any         `json:"no_update"`
	ApplyUpdate         map[string]any         `json:"apply_update"`
	Notes               []string               `json:"notes,omitempty"`
}

func envString(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(t *testing.T, key string, fallback int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("%s=%q is not an int: %v", key, v, err)
	}
	return n
}

func addStats(s *parityStats, values []float32) {
	for _, v := range values {
		addStatValue(s, v)
	}
}

func addStatValue(s *parityStats, v float32) {
	if len(s.Sample) < 16 {
		s.Sample = append(s.Sample, v)
	}
	x := float64(v)
	ax := math.Abs(x)
	s.Elems++
	s.L2 += x * x
	s.MeanAbs += ax
	if ax > s.MaxAbs {
		s.MaxAbs = ax
	}
}

func finishStats(s parityStats) parityStats {
	if s.Elems > 0 {
		s.L2 = math.Sqrt(s.L2)
		s.MeanAbs /= float64(s.Elems)
	}
	return s
}

func tensorStats(tensor *tensors.Tensor) parityStats {
	return parityStatsFromSlice(readFloat32Slice(tensor))
}

func parityStatsFromSlice(values []float32) parityStats {
	var s parityStats
	addStats(&s, values)
	return finishStats(s)
}

func diffStats(before, after []float32) parityStats {
	var s parityStats
	for i := range before {
		x := float64(after[i] - before[i])
		ax := math.Abs(x)
		s.Elems++
		s.L2 += x * x
		s.MeanAbs += ax
		if ax > s.MaxAbs {
			s.MaxAbs = ax
		}
	}
	return finishStats(s)
}

func diffFloat32Slices(lhs, rhs []float32) []float32 {
	n := len(lhs)
	if len(rhs) < n {
		n = len(rhs)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = lhs[i] - rhs[i]
	}
	return out
}

func layerNormInputGradReference(upstream, input, weight []float32, rows, hidden int, eps float32) []float32 {
	out := make([]float32, rows*hidden)
	if len(upstream) < rows*hidden || len(input) < rows*hidden || len(weight) < hidden || rows <= 0 || hidden <= 0 {
		return out
	}
	invHidden := float32(1.0 / float64(hidden))
	for row := 0; row < rows; row++ {
		base := row * hidden
		var mean float32
		for dim := 0; dim < hidden; dim++ {
			mean += input[base+dim]
		}
		mean *= invHidden

		var variance float32
		for dim := 0; dim < hidden; dim++ {
			centered := input[base+dim] - mean
			variance += centered * centered
		}
		variance *= invHidden
		invStd := float32(1.0 / math.Sqrt(float64(variance+eps)))

		var meanDyGamma float32
		var meanDyGammaXhat float32
		for dim := 0; dim < hidden; dim++ {
			centered := input[base+dim] - mean
			xhat := centered * invStd
			dyGamma := upstream[base+dim] * weight[dim]
			meanDyGamma += dyGamma
			meanDyGammaXhat += dyGamma * xhat
		}
		meanDyGamma *= invHidden
		meanDyGammaXhat *= invHidden

		for dim := 0; dim < hidden; dim++ {
			centered := input[base+dim] - mean
			xhat := centered * invStd
			dyGamma := upstream[base+dim] * weight[dim]
			out[base+dim] = invStd * (dyGamma - meanDyGamma - xhat*meanDyGammaXhat)
		}
	}
	return out
}

func tokenMajorAttentionIndex(b, token, head, dim, seqLen, numHeads, headDim int) int {
	return ((b*seqLen+token)*numHeads+head)*headDim + dim
}

func kernelMajorAttentionIndex(b, token, head, dim, seqLen, numHeads, headDim int) int {
	return ((b*numHeads+head)*seqLen+token)*headDim + dim
}

func attentionIndex(layout string, b, token, head, dim, seqLen, numHeads, headDim int) int {
	if layout == "kernel" {
		return kernelMajorAttentionIndex(b, token, head, dim, seqLen, numHeads, headDim)
	}
	return tokenMajorAttentionIndex(b, token, head, dim, seqLen, numHeads, headDim)
}

func sdpaReference(q, k, v []float32, attentionMask []int32, batchSize, seqLen, numHeads, headDim int, layout string, isLocal bool, localWindow int) []float32 {
	total := batchSize * seqLen * numHeads * headDim
	out := make([]float32, total)
	if len(q) != total || len(k) != total || len(v) != total || len(attentionMask) < batchSize*seqLen {
		return out
	}
	scores := make([]float32, seqLen)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	windowHalf := localWindow / 2
	for b := 0; b < batchSize; b++ {
		for head := 0; head < numHeads; head++ {
			for qi := 0; qi < seqLen; qi++ {
				best := float32(math.Inf(-1))
				for ki := 0; ki < seqLen; ki++ {
					dist := qi - ki
					if dist < 0 {
						dist = -dist
					}
					if attentionMask[b*seqLen+ki] == 0 || (isLocal && dist > windowHalf) {
						scores[ki] = 0
						continue
					}
					var score float32
					for dim := 0; dim < headDim; dim++ {
						qIdx := attentionIndex(layout, b, qi, head, dim, seqLen, numHeads, headDim)
						kIdx := attentionIndex(layout, b, ki, head, dim, seqLen, numHeads, headDim)
						score += q[qIdx] * k[kIdx]
					}
					score *= scale
					scores[ki] = score
					if score > best {
						best = score
					}
				}

				var sum float32
				for ki := 0; ki < seqLen; ki++ {
					dist := qi - ki
					if dist < 0 {
						dist = -dist
					}
					if attentionMask[b*seqLen+ki] == 0 || (isLocal && dist > windowHalf) {
						scores[ki] = 0
						continue
					}
					weight := float32(math.Exp(float64(scores[ki] - best)))
					scores[ki] = weight
					sum += weight
				}
				if sum <= 0 {
					continue
				}

				for dim := 0; dim < headDim; dim++ {
					var accum float32
					for ki := 0; ki < seqLen; ki++ {
						dist := qi - ki
						if dist < 0 {
							dist = -dist
						}
						if isLocal && dist > windowHalf {
							continue
						}
						if scores[ki] == 0 {
							continue
						}
						vIdx := attentionIndex(layout, b, ki, head, dim, seqLen, numHeads, headDim)
						accum += (scores[ki] / sum) * v[vIdx]
					}
					outIdx := attentionIndex(layout, b, qi, head, dim, seqLen, numHeads, headDim)
					out[outIdx] = accum
				}
			}
		}
	}
	return out
}

func attentionContextReference(attnProbs, v []float32, batchSize, seqLen, numHeads, headDim int, vLayout string) []float32 {
	total := batchSize * seqLen * numHeads * headDim
	out := make([]float32, total)
	if len(attnProbs) != batchSize*numHeads*seqLen*seqLen || len(v) != total {
		return out
	}
	for b := 0; b < batchSize; b++ {
		for head := 0; head < numHeads; head++ {
			for qi := 0; qi < seqLen; qi++ {
				for dim := 0; dim < headDim; dim++ {
					var accum float32
					for ki := 0; ki < seqLen; ki++ {
						probIdx := (((b*numHeads+head)*seqLen+qi)*seqLen + ki)
						vIdx := attentionIndex(vLayout, b, ki, head, dim, seqLen, numHeads, headDim)
						accum += attnProbs[probIdx] * v[vIdx]
					}
					outIdx := tokenMajorAttentionIndex(b, qi, head, dim, seqLen, numHeads, headDim)
					out[outIdx] = accum
				}
			}
		}
	}
	return out
}

func attentionRowProbeStats(q, k, v []float32, attentionMask []int32, batchSize, seqLen, numHeads, headDim int, layout string, isLocal bool, localWindow int, batchIdx, headIdx, queryIdx int) attentionRowProbe {
	total := batchSize * seqLen * numHeads * headDim
	out := attentionRowProbe{
		Batch:        batchIdx,
		Head:         headIdx,
		Query:        queryIdx,
		ScoreMin:     float32(math.Inf(1)),
		ScoreMax:     float32(math.Inf(-1)),
		QuerySample:  make([]float32, 0, 16),
		KeyQuerySample: make([]float32, 0, 16),
		ValueQuerySample: make([]float32, 0, 16),
		ScoreSample:  make([]float32, 0, 16),
		ProbSample:   make([]float32, 0, 16),
		OutputSample: make([]float32, 0, 16),
	}
	if len(q) != total || len(k) != total || len(v) != total || len(attentionMask) < batchSize*seqLen || batchIdx >= batchSize || headIdx >= numHeads || queryIdx >= seqLen {
		out.ScoreMin = 0
		out.ScoreMax = 0
		return out
	}

	scores := make([]float32, seqLen)
	probs := make([]float32, seqLen)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	windowHalf := localWindow / 2
	var querySumSq, keyQuerySumSq, valueQuerySumSq float64
	for dim := 0; dim < headDim; dim++ {
		qValue := q[attentionIndex(layout, batchIdx, queryIdx, headIdx, dim, seqLen, numHeads, headDim)]
		kValue := k[attentionIndex(layout, batchIdx, queryIdx, headIdx, dim, seqLen, numHeads, headDim)]
		vValue := v[attentionIndex(layout, batchIdx, queryIdx, headIdx, dim, seqLen, numHeads, headDim)]
		if len(out.QuerySample) < 16 {
			out.QuerySample = append(out.QuerySample, qValue)
		}
		if len(out.KeyQuerySample) < 16 {
			out.KeyQuerySample = append(out.KeyQuerySample, kValue)
		}
		if len(out.ValueQuerySample) < 16 {
			out.ValueQuerySample = append(out.ValueQuerySample, vValue)
		}
		querySumSq += float64(qValue) * float64(qValue)
		keyQuerySumSq += float64(kValue) * float64(kValue)
		valueQuerySumSq += float64(vValue) * float64(vValue)
		if ax := math.Abs(float64(qValue)); ax > float64(out.QueryMaxAbs) {
			out.QueryMaxAbs = float32(ax)
		}
		if ax := math.Abs(float64(kValue)); ax > float64(out.KeyQueryMaxAbs) {
			out.KeyQueryMaxAbs = float32(ax)
		}
		if ax := math.Abs(float64(vValue)); ax > float64(out.ValueQueryMaxAbs) {
			out.ValueQueryMaxAbs = float32(ax)
		}
	}
	if headDim > 0 {
		headDenom := float64(headDim)
		out.QueryRMS = math.Sqrt(querySumSq / headDenom)
		out.KeyQueryRMS = math.Sqrt(keyQuerySumSq / headDenom)
		out.ValueQueryRMS = math.Sqrt(valueQuerySumSq / headDenom)
	}
	var scoreSum, scoreSumSq float64
	for keyIdx := 0; keyIdx < seqLen; keyIdx++ {
		dist := queryIdx - keyIdx
		if dist < 0 {
			dist = -dist
		}
		if attentionMask[batchIdx*seqLen+keyIdx] == 0 || (isLocal && dist > windowHalf) {
			continue
		}
		var score float32
		for dim := 0; dim < headDim; dim++ {
			qIdx := attentionIndex(layout, batchIdx, queryIdx, headIdx, dim, seqLen, numHeads, headDim)
			kIdx := attentionIndex(layout, batchIdx, keyIdx, headIdx, dim, seqLen, numHeads, headDim)
			score += q[qIdx] * k[kIdx]
		}
		score *= scale
		scores[keyIdx] = score
		if len(out.ScoreSample) < 16 {
			out.ScoreSample = append(out.ScoreSample, score)
		}
		out.ValidKeys++
		scoreSum += float64(score)
		scoreSumSq += float64(score) * float64(score)
		if score < out.ScoreMin {
			out.ScoreMin = score
		}
		if score > out.ScoreMax {
			out.ScoreMax = score
			out.ScoreArgmax = keyIdx
		}
	}
	if out.ValidKeys == 0 {
		out.ScoreMin = 0
		out.ScoreMax = 0
		return out
	}
	denom := float64(out.ValidKeys)
	out.ScoreMean = scoreSum / denom
	out.ScoreRMS = math.Sqrt(scoreSumSq / denom)

	var expSum float32
	for keyIdx := 0; keyIdx < seqLen; keyIdx++ {
		dist := queryIdx - keyIdx
		if dist < 0 {
			dist = -dist
		}
		if attentionMask[batchIdx*seqLen+keyIdx] == 0 || (isLocal && dist > windowHalf) {
			continue
		}
		weight := float32(math.Exp(float64(scores[keyIdx] - out.ScoreMax)))
		probs[keyIdx] = weight
		expSum += weight
	}
	if expSum <= 0 {
		return out
	}

	top1 := float32(math.Inf(-1))
	top2 := float32(math.Inf(-1))
	for keyIdx := 0; keyIdx < seqLen; keyIdx++ {
		dist := queryIdx - keyIdx
		if dist < 0 {
			dist = -dist
		}
		if attentionMask[batchIdx*seqLen+keyIdx] == 0 || (isLocal && dist > windowHalf) {
			continue
		}
		prob := probs[keyIdx] / expSum
		probs[keyIdx] = prob
		if len(out.ProbSample) < 16 {
			out.ProbSample = append(out.ProbSample, prob)
		}
		if prob > 0 {
			out.ProbEntropy -= float64(prob) * math.Log(float64(prob))
		}
		if prob > out.ProbMax {
			out.ProbMax = prob
			out.ProbArgmax = keyIdx
		}
		if prob > top1 {
			top2 = top1
			top1 = prob
		} else if prob > top2 {
			top2 = prob
		}
	}
	if math.IsInf(float64(top2), -1) {
		out.ProbTop2Gap = top1
	} else {
		out.ProbTop2Gap = top1 - top2
	}

	var outputSum, outputSumSq float64
	for dim := 0; dim < headDim; dim++ {
		var accum float32
		for keyIdx := 0; keyIdx < seqLen; keyIdx++ {
			prob := probs[keyIdx]
			if prob == 0 {
				continue
			}
			vIdx := attentionIndex(layout, batchIdx, keyIdx, headIdx, dim, seqLen, numHeads, headDim)
			accum += prob * v[vIdx]
		}
		if len(out.OutputSample) < 16 {
			out.OutputSample = append(out.OutputSample, accum)
		}
		outputSum += float64(accum)
		outputSumSq += float64(accum) * float64(accum)
		if ax := math.Abs(float64(accum)); ax > float64(out.OutputMaxAbs) {
			out.OutputMaxAbs = float32(ax)
		}
	}
	if headDim > 0 {
		headDenom := float64(headDim)
		out.OutputMean = outputSum / headDenom
		out.OutputRMS = math.Sqrt(outputSumSq / headDenom)
	}
	return out
}

func layerAttentionRowProbes(layerIdx int, q, k, v []float32, attentionMask []int32, batchSize, seqLen, numHeads, headDim int, isLocal bool, localWindow int) map[string]attentionRowProbe {
	out := map[string]attentionRowProbe{}
	if batchSize == 0 || len(attentionMask) < batchSize*seqLen {
		return out
	}
	validPositions := make([]int, 0, seqLen)
	for tokenIdx := 0; tokenIdx < seqLen; tokenIdx++ {
		if attentionMask[tokenIdx] != 0 {
			validPositions = append(validPositions, tokenIdx)
		}
	}
	if len(validPositions) == 0 {
		return out
	}
	queryPositions := []int{
		validPositions[0],
		validPositions[len(validPositions)/2],
		validPositions[len(validPositions)-1],
	}
	names := [][]string{
		{"attn_token_row_h00_first", "attn_token_row_h00_mid", "attn_token_row_h00_last"},
		{"attn_token_row_h01_first", "attn_token_row_h01_mid", "attn_token_row_h01_last"},
	}
	headsToProbe := numHeads
	if headsToProbe > len(names) {
		headsToProbe = len(names)
	}
	for headIdx := 0; headIdx < headsToProbe; headIdx++ {
		for slot, queryIdx := range queryPositions {
			name := fmt.Sprintf("layer_%02d_%s", layerIdx, names[headIdx][slot])
			out[name] = attentionRowProbeStats(q, k, v, attentionMask, batchSize, seqLen, numHeads, headDim, "token", isLocal, localWindow, 0, headIdx, queryIdx)
		}
	}
	return out
}

func readTrainable(t *FusedTrainer, pred func(string) bool) map[string][]float32 {
	out := make(map[string][]float32)
	for _, name := range t.trainableVarNames {
		if !pred(name) {
			continue
		}
		v := t.varMap[name]
		if v == nil {
			continue
		}
		out[name] = readFloat32Slice(v.MustValue())
	}
	return out
}

func aggregateDelta(before map[string][]float32, t *FusedTrainer, pred func(string) bool) parityStats {
	var total parityStats
	for _, name := range t.trainableVarNames {
		if !pred(name) {
			continue
		}
		prev, ok := before[name]
		if !ok {
			continue
		}
		v := t.varMap[name]
		if v == nil {
			continue
		}
		now := readFloat32Slice(v.MustValue())
		s := diffStats(prev, now)
		total.Elems += s.Elems
		total.L2 += s.L2 * s.L2
		total.MeanAbs += s.MeanAbs * float64(s.Elems)
		if s.MaxAbs > total.MaxAbs {
			total.MaxAbs = s.MaxAbs
		}
	}
	if total.Elems > 0 {
		total.L2 = math.Sqrt(total.L2)
		total.MeanAbs /= float64(total.Elems)
	}
	return total
}

func aggregateAdam(t *FusedTrainer, pred func(string) bool, which string) parityStats {
	var total parityStats
	for i, name := range t.trainableVarNames {
		if i >= len(t.adamStates) || !pred(name) {
			continue
		}
		var values []float32
		if which == "m" {
			values = readFloat32Slice(t.adamStates[i].m.MustValue())
		} else {
			values = readFloat32Slice(t.adamStates[i].v.MustValue())
		}
		addStats(&total, values)
	}
	return finishStats(total)
}

func aggregateSegmentedAdam(s *SegmentedTrainer, names []string, pred func(string) bool, which string) parityStats {
	var total parityStats
	for _, name := range names {
		if !pred(name) {
			continue
		}
		idx, ok := s.adamStateMap[name]
		if !ok || idx >= len(s.adamStates) {
			continue
		}
		var values []float32
		if which == "m" {
			values = readFloat32Slice(s.adamStates[idx].m.MustValue())
		} else {
			values = readFloat32Slice(s.adamStates[idx].v.MustValue())
		}
		addStats(&total, values)
	}
	return finishStats(total)
}

func aggregateDeltaByLoRA(before map[string][]float32, t *FusedTrainer) map[string]parityStats {
	out := make(map[string]parityStats)
	for _, name := range t.trainableVarNames {
		key, ok := loraGradMatrixKey(name)
		if !ok {
			continue
		}
		prev, ok := before[name]
		if !ok {
			continue
		}
		v := t.varMap[name]
		if v == nil {
			continue
		}
		out[key] = diffStats(prev, readFloat32Slice(v.MustValue()))
	}
	return out
}

func aggregateSegmentedAdamByLoRA(s *SegmentedTrainer, names []string, which string) map[string]parityStats {
	out := make(map[string]parityStats)
	for _, name := range names {
		key, ok := loraGradMatrixKey(name)
		if !ok {
			continue
		}
		idx, ok := s.adamStateMap[name]
		if !ok || idx >= len(s.adamStates) {
			continue
		}
		var values []float32
		if which == "m" {
			values = readFloat32Slice(s.adamStates[idx].m.MustValue())
		} else {
			values = readFloat32Slice(s.adamStates[idx].v.MustValue())
		}
		out[key] = parityStatsFromSlice(values)
	}
	return out
}

func globalNormFromGrads(grads []gradEntry) float64 {
	var sum float64
	for _, entry := range grads {
		entry.gradTensor.ConstFlatData(func(flat any) {
			if data, ok := flat.([]float32); ok {
				for _, v := range data {
					sum += float64(v) * float64(v)
				}
			}
		})
	}
	return math.Sqrt(sum)
}

func aggregateGradStats(grads []gradEntry, pred func(string) bool) parityStats {
	var total parityStats
	for _, entry := range grads {
		if !pred(entry.name) {
			continue
		}
		values := readFloat32Slice(entry.gradTensor)
		addStats(&total, values)
	}
	return finishStats(total)
}

func pickGradStats(grads []gradEntry, pred func(string) bool) parityStats {
	for _, entry := range grads {
		if pred(entry.name) {
			return tensorStats(entry.gradTensor)
		}
	}
	return parityStats{}
}

func loraGradMatrixKey(name string) (string, bool) {
	const prefix = "var:/fused_chunker_embedder/encoder/layer/"
	if !strings.HasPrefix(name, prefix) {
		return "", false
	}
	parts := strings.Split(strings.TrimPrefix(name, prefix), "/")
	if len(parts) != 4 {
		return "", false
	}
	layerIdx, err := strconv.Atoi(parts[0])
	if err != nil {
		return "", false
	}
	scope := parts[1]
	module := parts[2]
	matrix := ""
	if parts[3] == "lora_A" {
		matrix = "A"
	} else if parts[3] == "lora_B" {
		matrix = "B"
	} else {
		return "", false
	}
	if module == "Wo" {
		if scope == "attn" {
			module = "out_proj"
		} else if scope == "mlp" {
			module = "wo"
		}
	}
	return fmt.Sprintf("layer_%02d_%s_%s", layerIdx, module, matrix), true
}

func loraGradMatrixStats(grads []gradEntry) map[string]parityStats {
	out := make(map[string]parityStats)
	for _, entry := range grads {
		key, ok := loraGradMatrixKey(entry.name)
		if !ok {
			continue
		}
		out[key] = tensorStats(entry.gradTensor)
	}
	return out
}

func captureSegmentVJPProbe(
	probeLayer int,
	seg int,
	upstreamStats parityStats,
	layerBackResults []*tensors.Tensor,
	layerBackGradNames []string,
) *segmentVJPProbe {
	probe := &segmentVJPProbe{
		TargetLayer:         probeLayer,
		SegmentStart:        seg,
		SegmentEnd:          seg + 1,
		IncludeHiddenGrad:   seg > 0,
		IncludeAdapterGrads: len(layerBackResults) > 1,
		Runtime:             "mpsgraph",
		Profile:             segmentVJPProfileProbe{Runtime: "mpsgraph"},
		Upstream:            upstreamStats,
		AdapterAByName:      map[string]parityStats{},
		AdapterBByName:      map[string]parityStats{},
	}
	if len(layerBackResults) > 0 {
		probe.HiddenGrad = tensorStats(layerBackResults[0])
	}

	var adapterA parityStats
	var adapterB parityStats
	for gi, name := range layerBackGradNames {
		if gi+1 >= len(layerBackResults) {
			continue
		}
		key, ok := loraGradMatrixKey(name)
		if !ok {
			continue
		}
		values := readFloat32Slice(layerBackResults[gi+1])
		stats := parityStatsFromSlice(values)
		if strings.HasSuffix(key, "_A") {
			probe.AdapterAByName[key] = stats
			addStats(&adapterA, values)
		} else if strings.HasSuffix(key, "_B") {
			probe.AdapterBByName[key] = stats
			addStats(&adapterB, values)
		}
	}
	probe.AdapterA = finishStats(adapterA)
	probe.AdapterB = finishStats(adapterB)
	return probe
}

func parityStatsPtr(stats parityStats) *parityStats {
	out := stats
	return &out
}

func layerBackwardDecompProbeFromSegment(segmentProbe *segmentVJPProbe, enabled bool) *layerBackwardDecompProbe {
	if !enabled {
		return nil
	}
	if segmentProbe == nil {
		return &layerBackwardDecompProbe{
			Status: "missing",
			Reason: "segment_vjp_probe_not_captured",
		}
	}
	const missingReason = "requires_debug_partial_vjp_boundary_graph"
	stages := map[string]layerBackwardDecompStage{
		"incoming_upstream": {
			Status: "captured",
			Stats:  parityStatsPtr(segmentProbe.Upstream),
		},
		"full_layer_hidden_grad": {
			Status:         "captured",
			Stats:          parityStatsPtr(segmentProbe.HiddenGrad),
			AdapterA:       parityStatsPtr(segmentProbe.AdapterA),
			AdapterB:       parityStatsPtr(segmentProbe.AdapterB),
			AdapterAByName: segmentProbe.AdapterAByName,
			AdapterBByName: segmentProbe.AdapterBByName,
		},
	}
	for _, name := range []string{
		"mlp_wo",
		"mlp_gelu_input",
		"mlp_gate_value",
		"mlp_gate_input",
		"mlp_wi_output",
		"mlp_norm_output",
		"mlp_hidden_after_attn",
		"attn_out_proj",
		"attention_core",
		"attention_core_post_rope",
		"attention_scores_raw",
		"attention_scores_masked",
		"attention_probs",
		"qkv_proj",
		"qkv_proj_split",
		"attn_norm_hidden_in",
	} {
		stages[name] = layerBackwardDecompStage{
			Status: "missing",
			Reason: missingReason,
		}
	}
	return &layerBackwardDecompProbe{
		Status:       "partial",
		Version:      4,
		TargetLayer:  segmentProbe.TargetLayer,
		SegmentStart: segmentProbe.SegmentStart,
		SegmentEnd:   segmentProbe.SegmentEnd,
		Runtime:      segmentProbe.Runtime,
		Stages:       stages,
	}
}

func scaleParityStats(stats parityStats, scale float64) parityStats {
	out := stats
	absScale := math.Abs(scale)
	out.L2 *= absScale
	out.MaxAbs *= absScale
	out.MeanAbs *= absScale
	out.Sample = make([]float32, len(stats.Sample))
	for i, v := range stats.Sample {
		out.Sample[i] = float32(float64(v) * scale)
	}
	return out
}

func scaleParityStatsMap(values map[string]parityStats, scale float64) map[string]parityStats {
	out := make(map[string]parityStats, len(values))
	for key, stats := range values {
		out[key] = scaleParityStats(stats, scale)
	}
	return out
}

func countStrings(values []string, pred func(string) bool) int {
	count := 0
	for _, value := range values {
		if pred(value) {
			count++
		}
	}
	return count
}

func sampleStrings(values []string, max int) []string {
	if len(values) <= max {
		out := make([]string, len(values))
		copy(out, values)
		return out
	}
	out := make([]string, max)
	copy(out, values[:max])
	return out
}

func gradNames(grads []gradEntry) []string {
	names := make([]string, 0, len(grads))
	for _, entry := range grads {
		names = append(names, entry.name)
	}
	return names
}

func readInt32Slice(t *tensors.Tensor) []int32 {
	var data []int32
	t.ConstFlatData(func(flat any) {
		d := flat.([]int32)
		data = make([]int32, len(d))
		copy(data, d)
	})
	return data
}

func hashStepParityU64(h *uint64, v uint64) {
	for i := 0; i < 8; i++ {
		*h ^= v & 0xff
		*h *= stepParityFNVPrime
		v >>= 8
	}
}

func hashStepParityI32(h *uint64, v int32) {
	hashStepParityU64(h, uint64(uint32(v)))
}

func hashStepParityF32(h *uint64, v float32) {
	hashStepParityU64(h, uint64(math.Float32bits(v)))
}

func absoluteBatchSampleIndices(batch *FusedBatch, sourceOffset int) []int {
	out := make([]int, len(batch.SampleIndices))
	for i, idx := range batch.SampleIndices {
		out[i] = sourceOffset + idx
	}
	return out
}

func stepParityBatchHashesFor(batch *FusedBatch, sourceOffset int) stepParityBatchHashes {
	inputIDs := readInt32Slice(batch.InputIDs)
	attentionMask := readInt32Slice(batch.AttentionMask)
	boundaryLabels := readFloat32Slice(batch.BoundaryLabels)
	chunkStarts := readInt32Slice(batch.ChunkStarts)
	chunkEnds := readInt32Slice(batch.ChunkEnds)
	chunkMask := readFloat32Slice(batch.ChunkMask)

	seqLen := batch.InputIDs.Shape().Dimensions[1]
	maxChunks := batch.ChunkStarts.Shape().Dimensions[1]
	idsHash, maskHash := stepParityFNVOffset, stepParityFNVOffset
	labelsHash, chunksHash, sampleIndicesHash := stepParityFNVOffset, stepParityFNVOffset, stepParityFNVOffset

	for _, idx := range batch.SampleIndices {
		hashStepParityU64(&sampleIndicesHash, uint64(sourceOffset+idx))
	}
	totalTokens := len(batch.SampleIndices) * seqLen
	for i := 0; i < totalTokens; i++ {
		hashStepParityI32(&idsHash, inputIDs[i])
		hashStepParityI32(&maskHash, attentionMask[i])
		if boundaryLabels[i] > 0.5 {
			hashStepParityU64(&labelsHash, 1)
		} else {
			hashStepParityU64(&labelsHash, 0)
		}
	}
	totalChunks := len(batch.SampleIndices) * maxChunks
	for i := 0; i < totalChunks; i++ {
		hashStepParityI32(&chunksHash, chunkStarts[i])
		hashStepParityI32(&chunksHash, chunkEnds[i])
		if chunkMask[i] > 0.5 {
			hashStepParityU64(&chunksHash, 1)
		} else {
			hashStepParityU64(&chunksHash, 0)
		}
	}

	return stepParityBatchHashes{
		SampleIndices: fmt.Sprintf("%x", sampleIndicesHash),
		InputIDs:      fmt.Sprintf("%x", idsHash),
		AttentionMask: fmt.Sprintf("%x", maskHash),
		Labels:        fmt.Sprintf("%x", labelsHash),
		Chunks:        fmt.Sprintf("%x", chunksHash),
	}
}

func tensorProbeStatsFromSlice(values []float32) tensorProbeStats {
	stats := tensorProbeStats{Elems: len(values)}
	hash := stepParityFNVOffset
	var sum, sumSq float64
	for i, v := range values {
		hashStepParityF32(&hash, v)
		if i < 16 {
			stats.Sample = append(stats.Sample, v)
		}
		x := float64(v)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			continue
		}
		sum += x
		sumSq += x * x
		ax := math.Abs(x)
		if ax > stats.MaxAbs {
			stats.MaxAbs = ax
			stats.MaxAbsIndex = i
			stats.MaxAbsValue = v
		}
		addTensorProbeTopAbs(&stats, i, v)
	}
	if len(values) > 0 {
		denom := float64(len(values))
		stats.Mean = sum / denom
		stats.RMS = math.Sqrt(sumSq / denom)
	}
	stats.Hash = fmt.Sprintf("%x", hash)
	return stats
}

func tensorProbeStatsFromMaskedHiddenDiff(lhs, rhs []float32, attentionMask []int32, hiddenSize int, wantValid bool) tensorProbeStats {
	stats := tensorProbeStats{}
	hash := stepParityFNVOffset
	if hiddenSize <= 0 {
		return stats
	}
	tokenCount := len(attentionMask)
	maxTokens := len(lhs) / hiddenSize
	if len(rhs)/hiddenSize < maxTokens {
		maxTokens = len(rhs) / hiddenSize
	}
	if tokenCount > maxTokens {
		tokenCount = maxTokens
	}
	var sum, sumSq float64
	for token := 0; token < tokenCount; token++ {
		isValid := attentionMask[token] != 0
		if isValid != wantValid {
			continue
		}
		base := token * hiddenSize
		for h := 0; h < hiddenSize; h++ {
			diff := lhs[base+h] - rhs[base+h]
			hashStepParityF32(&hash, diff)
			if stats.Elems < 16 {
				stats.Sample = append(stats.Sample, diff)
			}
			x := float64(diff)
			if math.IsNaN(x) || math.IsInf(x, 0) {
				continue
			}
			stats.Elems++
			sum += x
			sumSq += x * x
			ax := math.Abs(x)
			if ax > stats.MaxAbs {
				stats.MaxAbs = ax
				stats.MaxAbsIndex = base + h
				stats.MaxAbsValue = diff
			}
			addTensorProbeTopAbs(&stats, base+h, diff)
		}
	}
	if stats.Elems > 0 {
		denom := float64(stats.Elems)
		stats.Mean = sum / denom
		stats.RMS = math.Sqrt(sumSq / denom)
		stats.Hash = fmt.Sprintf("%x", hash)
	}
	return stats
}

func attentionMaskValidLengths(mask []int32, batchSize, seqLen int) []int {
	lengths := make([]int, batchSize)
	if len(mask) < batchSize*seqLen {
		return lengths
	}
	for b := 0; b < batchSize; b++ {
		base := b * seqLen
		for token := 0; token < seqLen; token++ {
			if mask[base+token] != 0 {
				lengths[b]++
			}
		}
	}
	return lengths
}

func addTensorProbeTopAbs(stats *tensorProbeStats, index int, value float32) {
	const topCount = 8
	ax := math.Abs(float64(value))
	insertAt := -1
	if len(stats.TopAbsValues) < topCount {
		insertAt = len(stats.TopAbsValues)
		stats.TopAbsValues = append(stats.TopAbsValues, 0)
		stats.TopAbsIndices = append(stats.TopAbsIndices, 0)
	} else if ax > math.Abs(float64(stats.TopAbsValues[len(stats.TopAbsValues)-1])) {
		insertAt = len(stats.TopAbsValues) - 1
	} else {
		return
	}
	for insertAt > 0 && ax > math.Abs(float64(stats.TopAbsValues[insertAt-1])) {
		stats.TopAbsValues[insertAt] = stats.TopAbsValues[insertAt-1]
		stats.TopAbsIndices[insertAt] = stats.TopAbsIndices[insertAt-1]
		insertAt--
	}
	stats.TopAbsValues[insertAt] = value
	stats.TopAbsIndices[insertAt] = index
}

func floatSliceProbeStatsFromSlice(values []float32) floatSliceProbeStats {
	stats := floatSliceProbeStats{Elems: len(values)}
	for _, v := range values {
		x := float64(v)
		ax := math.Abs(x)
		if v != 0 {
			stats.Nonzero++
		}
		stats.L2 += x * x
		stats.MeanAbs += ax
		if ax > stats.MaxAbs {
			stats.MaxAbs = ax
		}
	}
	if len(values) > 0 {
		stats.L2 = math.Sqrt(stats.L2)
		stats.MeanAbs /= float64(len(values))
	}
	return stats
}

func sampleFloat32(values []float32, max int) []float32 {
	if max <= 0 || len(values) == 0 {
		return nil
	}
	if len(values) < max {
		max = len(values)
	}
	out := make([]float32, max)
	copy(out, values[:max])
	return out
}

func vectorNormFloat32(values []float32) float32 {
	var sumSq float64
	for _, v := range values {
		sumSq += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sumSq))
}

func contrastiveProbeStatsFromSlices(chunkEmbFlat, gradFlat, chunkMaskFlat []float32, docIDs []int, contrastiveLoss, totalLoss float64, n, embeddingDim int) *contrastiveStepProbe {
	probe := &contrastiveStepProbe{
		ContrastiveLoss: contrastiveLoss,
		TotalLoss:       totalLoss,
		Embeddings:      floatSliceProbeStatsFromSlice(chunkEmbFlat),
		Grad:            floatSliceProbeStatsFromSlice(gradFlat),
	}
	if n <= 0 || embeddingDim <= 0 {
		return probe
	}
	limit := n
	if len(chunkMaskFlat) < limit {
		limit = len(chunkMaskFlat)
	}
	for chunkIdx := 0; chunkIdx < limit; chunkIdx++ {
		if chunkMaskFlat[chunkIdx] <= 0.5 {
			continue
		}
		probe.ActiveChunks++
		if len(probe.ActiveDocIDSample) < 8 && chunkIdx < len(docIDs) {
			probe.ActiveDocIDSample = append(probe.ActiveDocIDSample, docIDs[chunkIdx])
		}
		base := chunkIdx * embeddingDim
		if base+embeddingDim > len(chunkEmbFlat) {
			continue
		}
		embeddingVec := chunkEmbFlat[base : base+embeddingDim]
		if len(probe.ActiveEmbeddingNormSample) < 8 {
			probe.ActiveEmbeddingNormSample = append(probe.ActiveEmbeddingNormSample, vectorNormFloat32(embeddingVec))
		}
		if probe.FirstActiveIndex == nil {
			firstIdx := chunkIdx
			probe.FirstActiveIndex = &firstIdx
			if chunkIdx < len(docIDs) {
				firstDocID := docIDs[chunkIdx]
				probe.FirstActiveDocID = &firstDocID
			}
			probe.FirstActiveEmbeddingSample = sampleFloat32(embeddingVec, 16)
			if base+embeddingDim <= len(gradFlat) {
				gradVec := gradFlat[base : base+embeddingDim]
				probe.FirstActiveGradSample = sampleFloat32(gradVec, 16)
			}
		}
		if base+embeddingDim <= len(gradFlat) && len(probe.ActiveGradNormSample) < 8 {
			probe.ActiveGradNormSample = append(probe.ActiveGradNormSample, vectorNormFloat32(gradFlat[base:base+embeddingDim]))
		}
	}
	return probe
}

func tensorProbeStatsFromTransposedSlice(values []float32, rows, cols int) tensorProbeStats {
	if rows <= 0 || cols <= 0 || len(values) != rows*cols {
		return tensorProbeStatsFromSlice(values)
	}
	transposed := make([]float32, len(values))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			transposed[c*rows+r] = values[r*cols+c]
		}
	}
	return tensorProbeStatsFromSlice(transposed)
}

func linearReferenceErrorStats(actual, input, weight, bias []float32, rows, inDim, outDim int) tensorProbeStats {
	if rows <= 0 || inDim <= 0 || outDim <= 0 {
		return tensorProbeStats{}
	}
	if len(input) != rows*inDim || len(weight) != inDim*outDim || len(actual) < rows*outDim {
		return tensorProbeStats{}
	}
	if bias != nil && len(bias) < outDim {
		return tensorProbeStats{}
	}
	count := len(actual)
	if count > 16 {
		count = 16
	}
	errors := make([]float32, count)
	for flatIdx := 0; flatIdx < count; flatIdx++ {
		row := flatIdx / outDim
		col := flatIdx % outDim
		ref := float32(0)
		if bias != nil {
			ref = bias[col]
		}
		for k := 0; k < inDim; k++ {
			ref += input[row*inDim+k] * weight[k*outDim+col]
		}
		errors[flatIdx] = actual[flatIdx] - ref
	}
	return tensorProbeStatsFromSlice(errors)
}

func scaleReferenceErrorStats(actual, input []float32, scale float64) tensorProbeStats {
	count := len(actual)
	if len(input) < count {
		count = len(input)
	}
	if count > 16 {
		count = 16
	}
	errors := make([]float32, count)
	scale32 := float32(scale)
	for i := 0; i < count; i++ {
		errors[i] = actual[i] - input[i]*scale32
	}
	return tensorProbeStatsFromSlice(errors)
}

func addReferenceErrorStats(actual, lhs, rhs []float32) tensorProbeStats {
	count := len(actual)
	if len(lhs) < count {
		count = len(lhs)
	}
	if len(rhs) < count {
		count = len(rhs)
	}
	if count > 16 {
		count = 16
	}
	errors := make([]float32, count)
	for i := 0; i < count; i++ {
		errors[i] = actual[i] - (lhs[i] + rhs[i])
	}
	return tensorProbeStatsFromSlice(errors)
}

func tensorProbeStatsFromTensor(tensor *tensors.Tensor) tensorProbeStats {
	return tensorProbeStatsFromSlice(readFloat32Slice(tensor))
}

func readFloat32LittleEndianFile(t *testing.T, path string, expectedElems int) []float32 {
	t.Helper()
	if path == "" {
		return nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read Zig replay input %s: %v", path, err)
	}
	if expectedElems < 0 {
		t.Fatalf("invalid expected replay elems: %d", expectedElems)
	}
	expectedBytes := expectedElems * 4
	if len(raw) != expectedBytes {
		t.Fatalf("Zig replay input %s has %d bytes, expected %d", path, len(raw), expectedBytes)
	}
	values := make([]float32, expectedElems)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return values
}

func tensorProbeStatsFromTransposedTensor(tensor *tensors.Tensor, rows, cols int) tensorProbeStats {
	return tensorProbeStatsFromTransposedSlice(readFloat32Slice(tensor), rows, cols)
}

func flattenFloat32(node *Node, elems int) *Node {
	if node.DType() != dtypes.Float32 {
		node = ConvertDType(node, dtypes.Float32)
	}
	return Reshape(node, elems)
}

func embeddingRowProbeName(position int, tokenID int32) string {
	return fmt.Sprintf("pos_%03d_id_%d", position, tokenID)
}

func embeddingTableRowProbeStats(t *FusedTrainer, batch *FusedBatch, hiddenSize int, maxRows int) map[string]tensorProbeStats {
	out := map[string]tensorProbeStats{}
	v := t.varMap["var:/fused_chunker_embedder/embeddings/word_embeddings"]
	if v == nil {
		return out
	}
	values := readFloat32Slice(v.MustValue())
	inputIDs := readInt32Slice(batch.InputIDs)
	count := len(inputIDs)
	if count > maxRows {
		count = maxRows
	}
	for i := 0; i < count; i++ {
		tokenID := inputIDs[i]
		name := embeddingRowProbeName(i, tokenID)
		if tokenID < 0 {
			out[name] = tensorProbeStats{}
			continue
		}
		start := int(tokenID) * hiddenSize
		end := start + hiddenSize
		if start < 0 || end > len(values) {
			out[name] = tensorProbeStats{}
			continue
		}
		out[name] = tensorProbeStatsFromSlice(values[start:end])
	}
	return out
}

func embeddingLookupRowProbeStats(inputIDs []int32, lookup []float32, hiddenSize int, maxRows int) map[string]tensorProbeStats {
	out := map[string]tensorProbeStats{}
	count := len(inputIDs)
	if count > maxRows {
		count = maxRows
	}
	for i := 0; i < count; i++ {
		tokenID := inputIDs[i]
		name := embeddingRowProbeName(i, tokenID)
		start := i * hiddenSize
		end := start + hiddenSize
		if end > len(lookup) {
			out[name] = tensorProbeStats{}
			continue
		}
		out[name] = tensorProbeStatsFromSlice(lookup[start:end])
	}
	return out
}

func checkpointProbeStats(t *FusedTrainer, names ...string) tensorProbeStats {
	for _, name := range names {
		if v := t.varMap[name]; v != nil {
			return tensorProbeStatsFromTensor(v.MustValue())
		}
	}
	return tensorProbeStats{}
}

func checkpointFloat32Slice(t *testing.T, trainer *FusedTrainer, names ...string) []float32 {
	t.Helper()
	for _, name := range names {
		if v := trainer.varMap[name]; v != nil {
			return readFloat32Slice(v.MustValue())
		}
	}
	t.Fatalf("missing checkpoint tensor for any of %v", names)
	return nil
}

func boundaryCheckpointStats(t *FusedTrainer) boundaryCheckpointProbe {
	return boundaryCheckpointProbe{
		FinalNormWeight: checkpointProbeStats(t,
			"var:/fused_chunker_embedder/encoder/final_norm/weight",
			"var:/fused_chunker_embedder/encoder/final_norm/gamma",
		),
		FinalNormBias: checkpointProbeStats(t,
			"var:/fused_chunker_embedder/encoder/final_norm/bias",
			"var:/fused_chunker_embedder/encoder/final_norm/beta",
		),
		W1: checkpointProbeStats(t, "var:/fused_chunker_embedder/boundary_head/mlp_dense1/weight"),
		B1: checkpointProbeStats(t, "var:/fused_chunker_embedder/boundary_head/mlp_dense1/bias"),
		W2: checkpointProbeStats(t, "var:/fused_chunker_embedder/boundary_head/mlp_dense2/weight"),
		B2: checkpointProbeStats(t, "var:/fused_chunker_embedder/boundary_head/mlp_dense2/bias"),
	}
}

func embeddingCheckpointStats(t *FusedTrainer) embeddingProbe {
	return embeddingProbe{
		WordEmbeddingWeight: checkpointProbeStats(t, "var:/fused_chunker_embedder/embeddings/word_embeddings"),
		LayerNormWeight: checkpointProbeStats(t, "var:/fused_chunker_embedder/embeddings/LayerNorm/weight"),
		LayerNormBias:   checkpointProbeStats(t, "var:/fused_chunker_embedder/embeddings/LayerNorm/bias"),
	}
}

func sampleGradShapes(grads []gradEntry, pred func(string) bool, max int) []gradShapeSample {
	out := make([]gradShapeSample, 0, max)
	for _, entry := range grads {
		if !pred(entry.name) {
			continue
		}
		dims := entry.gradTensor.Shape().Dimensions
		shape := make([]int, len(dims))
		copy(shape, dims)
		out = append(out, gradShapeSample{Name: entry.name, Shape: shape})
		if len(out) >= max {
			break
		}
	}
	return out
}

func buildEmbeddingProbeExec(s *SegmentedTrainer, seqLen int) *context.Exec {
	hiddenSize := s.Model.Config.HiddenSize
	return context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, input *Node) []*Node {
			g := input.Graph()
			ctx.SetTraining(g, false)
			batchSize := input.Shape().Dimensions[0]
			inputIDs := Reshape(Slice(input, AxisRange(0, batchSize), AxisRange(0, seqLen)), batchSize, seqLen)
			if inputIDs.DType() != dtypes.Int32 {
				inputIDs = ConvertDType(inputIDs, dtypes.Int32)
			}
			embeddingCtx := ctx.In("fused_chunker_embedder").In("embeddings")
			wordEmbeddingsVar := embeddingCtx.VariableWithShape(
				"word_embeddings",
				shapes.Make(dtypes.Float32, s.Model.Config.VocabSize, hiddenSize),
			)
			wordEmbeddings := wordEmbeddingsVar.ValueGraph(g)
			embedded := Gather(wordEmbeddings, Reshape(inputIDs, batchSize, seqLen, 1))
			normed := s.Model.layerNorm(embeddingCtx.In("LayerNorm"), embedded)
			return []*Node{
				Reshape(embedded, batchSize*seqLen*hiddenSize),
				Reshape(normed, batchSize*seqLen*hiddenSize),
			}
		})
}

func buildLayerActivationInputProbeExec(s *SegmentedTrainer, layerIdx int) *context.Exec {
	hiddenSize := s.Model.Config.HiddenSize
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	intermediate := s.Model.Config.IntermediateSize
	layerKey := strconv.Itoa(layerIdx)
	return context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, hidden, attentionMask *Node) []*Node {
			g := hidden.Graph()
			ctx.SetTraining(g, false)
			batchSize := hidden.Shape().Dimensions[0]
			seqLen := hidden.Shape().Dimensions[1]
			layerCtx := ctx.In("fused_chunker_embedder").In("encoder").In("layer").In(layerKey)
			isGlobal := s.Model.Config.GlobalAttnEveryNLayers > 0 && layerIdx%s.Model.Config.GlobalAttnEveryNLayers == 0
			ropeTheta := s.Model.Config.LocalRoPETheta
			if isGlobal {
				ropeTheta = s.Model.Config.GlobalRoPETheta
			}
			if ropeTheta == 0 {
				ropeTheta = s.Model.Config.RoPEBase
			}

			normed := s.Model.layerNorm(layerCtx.In("attn_norm"), hidden)
			if s.Model.Config.UseBF16() {
				normed = ConvertDType(normed, dtypes.BFloat16)
			}
			attnCtx := layerCtx.In("attn")
			queryRaw := s.Model.linearWithAdapter(attnCtx.In("query_proj"), normed, hiddenSize, true, "", s.Model.LoRAConfig)
			keyRaw := s.Model.linearWithAdapter(attnCtx.In("key_proj"), normed, hiddenSize, true, "", s.Model.LoRAConfig)
			valueRaw := s.Model.linearWithAdapter(attnCtx.In("value_proj"), normed, hiddenSize, true, "", s.Model.LoRAConfig)

			query := Reshape(queryRaw, batchSize, seqLen, numHeads, headDim)
			query = TransposeAllDims(query, 0, 2, 1, 3)
			key := Reshape(keyRaw, batchSize, seqLen, numHeads, headDim)
			key = TransposeAllDims(key, 0, 2, 1, 3)
			value := Reshape(valueRaw, batchSize, seqLen, numHeads, headDim)
			value = TransposeAllDims(value, 0, 2, 1, 3)

			query = applyRoPE(query, seqLen, headDim, ropeTheta)
			key = applyRoPE(key, seqLen, headDim, ropeTheta)
			queryRopeBsh := Reshape(TransposeAllDims(query, 0, 2, 1, 3), batchSize*seqLen*hiddenSize)
			keyRopeBsh := Reshape(TransposeAllDims(key, 0, 2, 1, 3), batchSize*seqLen*hiddenSize)

			scale := 1.0 / math.Sqrt(float64(headDim))
			attnScores := Einsum("bhqd,bhkd->bhqk", query, key)
			attnScoresRaw := attnScores
			attnScores = MulScalar(attnScores, scale)
			dtype := attnScores.DType()
			if attentionMask != nil {
				mask := Reshape(attentionMask, batchSize, 1, 1, seqLen)
				maskBias := MulScalar(Sub(Ones(g, mask.Shape()), ConvertDType(mask, dtype)), -10000.0)
				attnScores = Add(attnScores, maskBias)
			}
			if !isGlobal && s.Model.Config.LocalAttention > 0 {
				windowMask := createSlidingWindowMask(g, seqLen, s.Model.Config.LocalAttention, dtype)
				attnScores = Add(attnScores, windowMask)
			}
			attnScoresDType := attnScores.DType()
			if attnScoresDType == dtypes.BFloat16 {
				attnScores = ConvertDType(attnScores, dtypes.Float32)
			}
			attnProbs := Softmax(attnScores, -1)
			if attnScoresDType == dtypes.BFloat16 {
				attnProbs = ConvertDType(attnProbs, dtypes.BFloat16)
			}
			attnOutput := Einsum("bhqk,bhkd->bhqd", attnProbs, value)
			attnOutput = TransposeAllDims(attnOutput, 0, 2, 1, 3)
			attnOutput = Reshape(attnOutput, batchSize, seqLen, hiddenSize)

			projected := s.Model.linearWithAdapter(attnCtx.In("Wo"), attnOutput, hiddenSize, true, "", s.Model.LoRAConfig)
			if s.Model.Config.UseBF16() {
				projected = ConvertDType(projected, dtypes.Float32)
			}
			hiddenAfterAttn := Add(hidden, projected)
			normedFFN := s.Model.layerNorm(layerCtx.In("mlp_norm"), hiddenAfterAttn)
			if s.Model.Config.UseBF16() {
				normedFFN = ConvertDType(normedFFN, dtypes.BFloat16)
			}
			gated := s.Model.linearWithAdapter(layerCtx.In("mlp").In("Wi"), normedFFN, intermediate*2, true, "", s.Model.LoRAConfig)
			gateInput := Reshape(
				Slice(gated, AxisRange(0, batchSize), AxisRange(0, seqLen), AxisRange(0, intermediate)),
				batchSize, seqLen, intermediate)
			gateValue := Reshape(
				Slice(gated, AxisRange(0, batchSize), AxisRange(0, seqLen), AxisRange(intermediate, intermediate*2)),
				batchSize, seqLen, intermediate)
			woInput := Mul(activations.Gelu(gateInput), gateValue)

			return []*Node{
				flattenFloat32(normed, batchSize*seqLen*hiddenSize),
				flattenFloat32(queryRaw, batchSize*seqLen*hiddenSize),
				flattenFloat32(keyRaw, batchSize*seqLen*hiddenSize),
				flattenFloat32(valueRaw, batchSize*seqLen*hiddenSize),
				flattenFloat32(queryRopeBsh, batchSize*seqLen*hiddenSize),
				flattenFloat32(keyRopeBsh, batchSize*seqLen*hiddenSize),
				flattenFloat32(attnOutput, batchSize*seqLen*hiddenSize),
				flattenFloat32(woInput, batchSize*seqLen*intermediate),
				flattenFloat32(attnScoresRaw, batchSize*numHeads*seqLen*seqLen),
				flattenFloat32(attnScores, batchSize*numHeads*seqLen*seqLen),
				flattenFloat32(attnProbs, batchSize*numHeads*seqLen*seqLen),
			}
		})
}

func nodeElemCount(node *Node) int {
	elems := 1
	for _, dim := range node.Shape().Dimensions {
		elems *= dim
	}
	return elems
}

func buildLayerBackwardSubstageExec(s *SegmentedTrainer, layerIdx int, stage string) (*context.Exec, *[]string) {
	hiddenSize := s.Model.Config.HiddenSize
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	intermediate := s.Model.Config.IntermediateSize
	layerKey := strconv.Itoa(layerIdx)
	gradNames := &[]string{}
	exec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, hidden, upstream, attentionMask *Node) []*Node {
			g := hidden.Graph()
			ctx.SetTraining(g, false)
			batchSize := hidden.Shape().Dimensions[0]
			seqLen := hidden.Shape().Dimensions[1]
			layerCtx := ctx.In("fused_chunker_embedder").In("encoder").In("layer").In(layerKey)
			isGlobal := s.Model.Config.GlobalAttnEveryNLayers > 0 && layerIdx%s.Model.Config.GlobalAttnEveryNLayers == 0
			ropeTheta := s.Model.Config.LocalRoPETheta
			if isGlobal {
				ropeTheta = s.Model.Config.GlobalRoPETheta
			}
			if ropeTheta == 0 {
				ropeTheta = s.Model.Config.RoPEBase
			}

			attnNormed := s.Model.layerNorm(layerCtx.In("attn_norm"), hidden)
			if s.Model.Config.UseBF16() {
				attnNormed = ConvertDType(attnNormed, dtypes.BFloat16)
			}
			qAttnNormed := attnNormed
			kAttnNormed := attnNormed
			vAttnNormed := attnNormed
			if stage == "qkv_proj_split" {
				qAttnNormed = MulScalar(attnNormed, 1.0)
				kAttnNormed = MulScalar(attnNormed, 1.0)
				vAttnNormed = MulScalar(attnNormed, 1.0)
			}
			attnCtx := layerCtx.In("attn")
			queryRaw := s.Model.linearWithAdapter(attnCtx.In("query_proj"), qAttnNormed, hiddenSize, true, "", s.Model.LoRAConfig)
			keyRaw := s.Model.linearWithAdapter(attnCtx.In("key_proj"), kAttnNormed, hiddenSize, true, "", s.Model.LoRAConfig)
			valueRaw := s.Model.linearWithAdapter(attnCtx.In("value_proj"), vAttnNormed, hiddenSize, true, "", s.Model.LoRAConfig)

			query := Reshape(queryRaw, batchSize, seqLen, numHeads, headDim)
			query = TransposeAllDims(query, 0, 2, 1, 3)
			key := Reshape(keyRaw, batchSize, seqLen, numHeads, headDim)
			key = TransposeAllDims(key, 0, 2, 1, 3)
			value := Reshape(valueRaw, batchSize, seqLen, numHeads, headDim)
			value = TransposeAllDims(value, 0, 2, 1, 3)

			query = applyRoPE(query, seqLen, headDim, ropeTheta)
			key = applyRoPE(key, seqLen, headDim, ropeTheta)
			var qRopeInput *Node
			var kRopeInput *Node
			var vAttentionInput *Node
			if stage == "attention_core_post_rope" {
				qRopeFlat := Reshape(TransposeAllDims(query, 0, 2, 1, 3), batchSize, seqLen, hiddenSize)
				kRopeFlat := Reshape(TransposeAllDims(key, 0, 2, 1, 3), batchSize, seqLen, hiddenSize)
				qRopeInput = MulScalar(qRopeFlat, 1.0)
				kRopeInput = MulScalar(kRopeFlat, 1.0)
				query = Reshape(qRopeInput, batchSize, seqLen, numHeads, headDim)
				query = TransposeAllDims(query, 0, 2, 1, 3)
				key = Reshape(kRopeInput, batchSize, seqLen, numHeads, headDim)
				key = TransposeAllDims(key, 0, 2, 1, 3)
			}
			if stage == "attention_core_post_rope" || stage == "attention_probs" {
				vAttentionInput = MulScalar(valueRaw, 1.0)
				value = Reshape(vAttentionInput, batchSize, seqLen, numHeads, headDim)
				value = TransposeAllDims(value, 0, 2, 1, 3)
			}
			scale := 1.0 / math.Sqrt(float64(headDim))
			attnScoresRaw := Einsum("bhqd,bhkd->bhqk", query, key)
			var scoresRawInput *Node
			if stage == "attention_scores_raw" {
				scoresRawInput = MulScalar(attnScoresRaw, 1.0)
				attnScoresRaw = scoresRawInput
			}
			attnScores := MulScalar(attnScoresRaw, scale)
			dtype := attnScores.DType()
			if attentionMask != nil {
				mask := Reshape(attentionMask, batchSize, 1, 1, seqLen)
				maskBias := MulScalar(Sub(Ones(g, mask.Shape()), ConvertDType(mask, dtype)), -10000.0)
				attnScores = Add(attnScores, maskBias)
			}
			if !isGlobal && s.Model.Config.LocalAttention > 0 {
				windowMask := createSlidingWindowMask(g, seqLen, s.Model.Config.LocalAttention, dtype)
				attnScores = Add(attnScores, windowMask)
			}
			var scoresMaskedInput *Node
			if stage == "attention_scores_masked" {
				scoresMaskedInput = MulScalar(attnScores, 1.0)
				attnScores = scoresMaskedInput
			}
			attnScoresDType := attnScores.DType()
			if attnScoresDType == dtypes.BFloat16 {
				attnScores = ConvertDType(attnScores, dtypes.Float32)
			}
			attnProbs := Softmax(attnScores, -1)
			if attnScoresDType == dtypes.BFloat16 {
				attnProbs = ConvertDType(attnProbs, dtypes.BFloat16)
			}
			var probsInput *Node
			if stage == "attention_probs" {
				probsInput = MulScalar(attnProbs, 1.0)
				attnProbs = probsInput
			}
			attnMerged := Einsum("bhqk,bhkd->bhqd", attnProbs, value)
			attnMerged = TransposeAllDims(attnMerged, 0, 2, 1, 3)
			attnMerged = Reshape(attnMerged, batchSize, seqLen, hiddenSize)

			projected := s.Model.linearWithAdapter(attnCtx.In("Wo"), attnMerged, hiddenSize, true, "", s.Model.LoRAConfig)
			if s.Model.Config.UseBF16() {
				projected = ConvertDType(projected, dtypes.Float32)
			}
			hiddenAfterAttn := Add(hidden, projected)
			normedFFN := s.Model.layerNorm(layerCtx.In("mlp_norm"), hiddenAfterAttn)
			if s.Model.Config.UseBF16() {
				normedFFN = ConvertDType(normedFFN, dtypes.BFloat16)
			}
			gated := s.Model.linearWithAdapter(layerCtx.In("mlp").In("Wi"), normedFFN, intermediate*2, true, "", s.Model.LoRAConfig)
			gateInput := Reshape(
				Slice(gated, AxisRange(0, batchSize), AxisRange(0, seqLen), AxisRange(0, intermediate)),
				batchSize, seqLen, intermediate)
			gateValue := Reshape(
				Slice(gated, AxisRange(0, batchSize), AxisRange(0, seqLen), AxisRange(intermediate, intermediate*2)),
				batchSize, seqLen, intermediate)
			geluInput := activations.Gelu(gateInput)
			woInput := Mul(geluInput, gateValue)
			ffnOut := s.Model.linearWithAdapter(layerCtx.In("mlp").In("Wo"), woInput, hiddenSize, true, "", s.Model.LoRAConfig)
			if s.Model.Config.UseBF16() {
				ffnOut = ConvertDType(ffnOut, dtypes.Float32)
			}
			output := Add(hiddenAfterAttn, ffnOut)
			loss := ReduceAllSum(Mul(output, upstream))

			gradTargets := make([]*Node, 0, 16)
			names := make([]string, 0, 16)
			addTarget := func(name string, node *Node) {
				gradTargets = append(gradTargets, node)
				names = append(names, name)
			}
			switch stage {
			case "mlp_wo":
				addTarget("mlp_wo_input_grad", woInput)
			case "mlp_gelu_input":
				addTarget("gelu_input_grad", geluInput)
			case "mlp_gate_value":
				addTarget("gate_value_grad", gateValue)
			case "mlp_gate_input":
				addTarget("gate_input_grad", gateInput)
			case "mlp_wi_output":
				addTarget("wi_output_grad", gated)
			case "mlp_norm_output":
				addTarget("mlp_norm_output_grad", normedFFN)
			case "mlp_hidden_after_attn":
				addTarget("hidden_after_attn_grad", hiddenAfterAttn)
			case "attn_out_proj":
				addTarget("attn_merged_grad", attnMerged)
			case "attention_core":
				addTarget("q_raw_grad", queryRaw)
				addTarget("k_raw_grad", keyRaw)
				addTarget("v_raw_grad", valueRaw)
			case "attention_core_post_rope":
				addTarget("q_rope_grad", qRopeInput)
				addTarget("k_rope_grad", kRopeInput)
				addTarget("v_attention_input_grad", vAttentionInput)
			case "attention_scores_raw":
				addTarget("scores_raw_grad", scoresRawInput)
			case "attention_scores_masked":
				addTarget("scores_masked_grad", scoresMaskedInput)
			case "attention_probs":
				addTarget("probs_grad", probsInput)
				addTarget("v_attention_input_grad", vAttentionInput)
			case "qkv_proj":
				addTarget("attn_normed_grad", attnNormed)
			case "qkv_proj_split":
				addTarget("q_attn_normed_grad", qAttnNormed)
				addTarget("k_attn_normed_grad", kAttnNormed)
				addTarget("v_attn_normed_grad", vAttnNormed)
			case "attn_norm_hidden_in":
				addTarget("hidden_in_grad", hidden)
			default:
				panic(fmt.Sprintf("unknown layer backward substage %q", stage))
			}
			ctx.EnumerateVariables(func(v *context.Variable) {
				name := v.ParameterName()
				if v.Trainable && v.InUseByGraph(g) && strings.Contains(name, "lora_") {
					gradTargets = append(gradTargets, v.ValueGraph(g))
					names = append(names, name)
				}
			})
			grads := Gradient(loss, gradTargets...)
			*gradNames = names
			out := make([]*Node, 0, len(grads))
			for _, grad := range grads {
				out = append(out, flattenFloat32(grad, nodeElemCount(grad)))
			}
			return out
		})
	return exec, gradNames
}

func layerBackwardDecompStageFromResults(results []*tensors.Tensor, names []string, stageGradCount int) layerBackwardDecompStage {
	stage := layerBackwardDecompStage{
		Status:         "captured",
		Components:     map[string]parityStats{},
		AdapterAByName: map[string]parityStats{},
		AdapterBByName: map[string]parityStats{},
	}
	var stats parityStats
	var adapterA parityStats
	var adapterB parityStats
	for i, tensor := range results {
		if i >= len(names) {
			break
		}
		values := readFloat32Slice(tensor)
		itemStats := parityStatsFromSlice(values)
		if i < stageGradCount {
			stage.Components[names[i]] = itemStats
			addStats(&stats, values)
			continue
		}
		key, ok := loraGradMatrixKey(names[i])
		if !ok {
			continue
		}
		if strings.HasSuffix(key, "_A") {
			stage.AdapterAByName[key] = itemStats
			addStats(&adapterA, values)
		} else if strings.HasSuffix(key, "_B") {
			stage.AdapterBByName[key] = itemStats
			addStats(&adapterB, values)
		}
	}
	stage.Stats = parityStatsPtr(finishStats(stats))
	stage.AdapterA = parityStatsPtr(finishStats(adapterA))
	stage.AdapterB = parityStatsPtr(finishStats(adapterB))
	return stage
}

func captureLayerBackwardDecompProbe(
	s *SegmentedTrainer,
	probeLayer int,
	seg int,
	hidden *tensors.Tensor,
	upstream *tensors.Tensor,
	attentionMask *tensors.Tensor,
	segmentProbe *segmentVJPProbe,
) *layerBackwardDecompProbe {
	if segmentProbe == nil {
		return &layerBackwardDecompProbe{Status: "missing", Reason: "segment_vjp_probe_not_captured"}
	}
	stages := map[string]layerBackwardDecompStage{
		"incoming_upstream": {
			Status: "captured",
			Stats:  parityStatsPtr(segmentProbe.Upstream),
		},
		"full_layer_hidden_grad": {
			Status:         "captured",
			Stats:          parityStatsPtr(segmentProbe.HiddenGrad),
			AdapterA:       parityStatsPtr(segmentProbe.AdapterA),
			AdapterB:       parityStatsPtr(segmentProbe.AdapterB),
			AdapterAByName: segmentProbe.AdapterAByName,
			AdapterBByName: segmentProbe.AdapterBByName,
		},
	}
	stageOrder := []string{
		"mlp_wo",
		"mlp_gelu_input",
		"mlp_gate_value",
		"mlp_gate_input",
		"mlp_wi_output",
		"mlp_norm_output",
		"mlp_hidden_after_attn",
		"attn_out_proj",
		"attention_core",
		"attention_core_post_rope",
		"attention_scores_raw",
		"attention_scores_masked",
		"attention_probs",
		"qkv_proj",
		"qkv_proj_split",
		"attn_norm_hidden_in",
	}
	status := "captured"
	for _, stageName := range stageOrder {
		exec, namesPtr := buildLayerBackwardSubstageExec(s, probeLayer, stageName)
		results := exec.MustExec(hidden, upstream, attentionMask)
		names := *namesPtr
		stageGradCount := 1
		if stageName == "attention_core" || stageName == "attention_core_post_rope" || stageName == "qkv_proj_split" {
			stageGradCount = 3
		} else if stageName == "attention_probs" {
			stageGradCount = 2
		}
		stages[stageName] = layerBackwardDecompStageFromResults(results, names, stageGradCount)
		for _, tensor := range results {
			tensor.FinalizeAll()
		}
		if stages[stageName].Status != "captured" {
			status = "partial"
		}
	}
	return &layerBackwardDecompProbe{
		Status:       status,
		Version:      4,
		TargetLayer:  segmentProbe.TargetLayer,
		SegmentStart: seg,
		SegmentEnd:   seg + 1,
		Runtime:      "mpsgraph",
		Stages:       stages,
	}
}

type softmaxVJPCaseConfig struct {
	name     string
	outer    int
	queries  int
	keys     int
	hasMask  bool
	maskBias float32
}

func deterministicSoftmaxScore(outer, query, key int) float32 {
	mixed := (outer*37 + query*17 + key*29 + 11) % 127
	centered := mixed - 63
	return float32(centered)*0.03125 + float32((query+key)%7)*0.002
}

func deterministicSoftmaxUpstream(outer, query, key int) float32 {
	mixed := (outer*19 + query*23 + key*13 + 5) % 89
	centered := mixed - 44
	return float32(centered) * 0.00025
}

func deterministicSoftmaxMaskedKey(query, key, queries, keys int, hasMask bool, localWindow int) bool {
	if !hasMask {
		return false
	}
	tailMask := keys / 16
	if tailMask < 1 {
		tailMask = 1
	}
	if key >= keys-tailMask {
		return true
	}
	if queries == keys && localWindow > 0 {
		windowHalf := localWindow / 2
		diff := query - key
		if diff < 0 {
			diff = -diff
		}
		if diff > windowHalf {
			return true
		}
	}
	return false
}

func makeSoftmaxVJPTensors(cfg softmaxVJPCaseConfig, localWindow int) ([]float32, []float32) {
	total := cfg.outer * cfg.queries * cfg.keys
	scores := make([]float32, total)
	upstream := make([]float32, total)
	idx := 0
	for outer := 0; outer < cfg.outer; outer++ {
		for query := 0; query < cfg.queries; query++ {
			for key := 0; key < cfg.keys; key++ {
				score := deterministicSoftmaxScore(outer, query, key)
				if deterministicSoftmaxMaskedKey(query, key, cfg.queries, cfg.keys, cfg.hasMask, localWindow) {
					score += cfg.maskBias
				}
				scores[idx] = score
				upstream[idx] = deterministicSoftmaxUpstream(outer, query, key)
				idx++
			}
		}
	}
	return scores, upstream
}

func addSoftmaxCPUReferenceStats(out *softmaxVJPCase, scores, upstream, mpsGrad []float32, localWindow int) {
	probs := make([]float32, out.Keys)
	rowStart := 0
	for outer := 0; outer < out.Outer; outer++ {
		_ = outer
		for query := 0; query < out.Queries; query++ {
			best := float32(math.Inf(-1))
			for key := 0; key < out.Keys; key++ {
				if scores[rowStart+key] > best {
					best = scores[rowStart+key]
				}
			}
			var sum float32
			for key := 0; key < out.Keys; key++ {
				prob := float32(math.Exp(float64(scores[rowStart+key] - best)))
				probs[key] = prob
				sum += prob
			}
			if sum > 0 {
				invSum := 1 / sum
				for key := 0; key < out.Keys; key++ {
					probs[key] *= invSum
				}
			} else {
				for key := range probs {
					probs[key] = 0
				}
			}

			var dot float32
			for key := 0; key < out.Keys; key++ {
				addStatValue(&out.Probs, probs[key])
				dot += upstream[rowStart+key] * probs[key]
			}
			for key := 0; key < out.Keys; key++ {
				idx := rowStart + key
				cpuGrad := probs[key] * (upstream[idx] - dot)
				errAbs := float32(math.Abs(float64(mpsGrad[idx] - cpuGrad)))
				addStatValue(&out.CPUScoresMaskedGrad, cpuGrad)
				addStatValue(&out.CPUAbsError, errAbs)
				if deterministicSoftmaxMaskedKey(query, key, out.Queries, out.Keys, out.HasMask, localWindow) {
					addStatValue(&out.MaskedScoresMaskedGrad, mpsGrad[idx])
				} else {
					addStatValue(&out.ValidScoresMaskedGrad, mpsGrad[idx])
				}
			}
			rowStart += out.Keys
		}
	}
	out.Probs = finishStats(out.Probs)
	out.CPUScoresMaskedGrad = finishStats(out.CPUScoresMaskedGrad)
	out.CPUAbsError = finishStats(out.CPUAbsError)
	out.ValidScoresMaskedGrad = finishStats(out.ValidScoresMaskedGrad)
	out.MaskedScoresMaskedGrad = finishStats(out.MaskedScoresMaskedGrad)
}

func captureSoftmaxVJPCase(s *SegmentedTrainer, cfg softmaxVJPCaseConfig, localWindow int) softmaxVJPCase {
	scores, upstream := makeSoftmaxVJPTensors(cfg, localWindow)
	total := len(scores)
	exec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, scoresNode, upstreamNode *Node) []*Node {
			ctx.SetTraining(scoresNode.Graph(), false)
			probs := Softmax(scoresNode, -1)
			loss := ReduceAllSum(Mul(probs, upstreamNode))
			grads := Gradient(loss, scoresNode)
			return []*Node{
				flattenFloat32(probs, total),
				flattenFloat32(grads[0], total),
			}
		})
	scoresTensor := tensors.FromFlatDataAndDimensions(scores, cfg.outer, cfg.queries, cfg.keys)
	upstreamTensor := tensors.FromFlatDataAndDimensions(upstream, cfg.outer, cfg.queries, cfg.keys)
	results := exec.MustExec(scoresTensor, upstreamTensor)
	scoresTensor.FinalizeAll()
	upstreamTensor.FinalizeAll()
	probsTensor := results[0]
	gradTensor := results[1]
	defer probsTensor.FinalizeAll()
	defer gradTensor.FinalizeAll()
	grad := readFloat32Slice(gradTensor)
	out := softmaxVJPCase{
		Status:            "captured",
		Reason:            "",
		Outer:             cfg.outer,
		Queries:           cfg.queries,
		Keys:              cfg.keys,
		HasMask:           cfg.hasMask,
		MaskBias:          cfg.maskBias,
		ScoresMasked:      parityStatsFromSlice(scores),
		UpstreamProbsGrad: parityStatsFromSlice(upstream),
		ScoresMaskedGrad:  parityStatsFromSlice(grad),
	}
	addSoftmaxCPUReferenceStats(&out, scores, upstream, grad, localWindow)
	return out
}

func captureSoftmaxVJPProbe(s *SegmentedTrainer, batchSize, seqLen int) *softmaxVJPProbe {
	layerOuter := batchSize * s.Model.Config.NumAttentionHeads
	localWindow := s.Model.Config.LocalAttention
	configs := []softmaxVJPCaseConfig{
		{name: "synthetic_small_nomask", outer: 1, queries: 8, keys: 16, hasMask: false, maskBias: 0},
		{name: "layer14_shape_mask_neg1e9", outer: layerOuter, queries: seqLen, keys: seqLen, hasMask: true, maskBias: -1.0e9},
		{name: "layer14_shape_mask_neg10000", outer: layerOuter, queries: seqLen, keys: seqLen, hasMask: true, maskBias: -10000.0},
	}
	probe := &softmaxVJPProbe{
		Status:  "captured",
		Version: 1,
		Runtime: "mpsgraph",
		Cases:   map[string]softmaxVJPCase{},
	}
	for _, cfg := range configs {
		probe.Cases[cfg.name] = captureSoftmaxVJPCase(s, cfg, localWindow)
	}
	return probe
}

func deterministicQKVSplitValue(tag, a, b, c int) float32 {
	mixed := (tag*1009 + a*131 + b*37 + c*17) % 2003
	return float32(mixed-1001) * 0.00025
}

func makeFlatQKVTensor(rows, cols, tag int) []float32 {
	out := make([]float32, rows*cols)
	idx := 0
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			out[idx] = deterministicQKVSplitValue(tag, row, col, 0)
			idx++
		}
	}
	return out
}

func makeBhsdQKVTensor(outer, seqLen, headDim, tag int) []float32 {
	out := make([]float32, outer*seqLen*headDim)
	idx := 0
	for outerIdx := 0; outerIdx < outer; outerIdx++ {
		for pos := 0; pos < seqLen; pos++ {
			for dim := 0; dim < headDim; dim++ {
				out[idx] = deterministicQKVSplitValue(tag, outerIdx, pos, dim)
				idx++
			}
		}
	}
	return out
}

func makeScoreQKVTensor(outer, seqLen, tag int) []float32 {
	out := make([]float32, outer*seqLen*seqLen)
	idx := 0
	for outerIdx := 0; outerIdx < outer; outerIdx++ {
		for query := 0; query < seqLen; query++ {
			for key := 0; key < seqLen; key++ {
				out[idx] = deterministicQKVSplitValue(tag, outerIdx, query, key)
				idx++
			}
		}
	}
	return out
}

func makeAttentionProbQKVTensor(outer, seqLen, tag int) []float32 {
	out := make([]float32, outer*seqLen*seqLen)
	idx := 0
	for outerIdx := 0; outerIdx < outer; outerIdx++ {
		for query := 0; query < seqLen; query++ {
			var rowSum float32
			for key := 0; key < seqLen; key++ {
				raw := float32(0.1) + float32(math.Abs(float64(deterministicQKVSplitValue(tag, outerIdx, query, key))))
				out[idx+key] = raw
				rowSum += raw
			}
			if rowSum > 0 {
				inv := 1 / rowSum
				for key := 0; key < seqLen; key++ {
					out[idx+key] *= inv
				}
			}
			idx += seqLen
		}
	}
	return out
}

func absErrorStats(lhs, rhs []float32) parityStats {
	var s parityStats
	n := len(lhs)
	if len(rhs) < n {
		n = len(rhs)
	}
	for i := 0; i < n; i++ {
		addStatValue(&s, float32(math.Abs(float64(lhs[i]-rhs[i]))))
	}
	return finishStats(s)
}

func splitHeadsNode(flat *Node, batchSize, seqLen, numHeads, headDim int) *Node {
	reshaped := Reshape(flat, batchSize, seqLen, numHeads, headDim)
	transposed := TransposeAllDims(reshaped, 0, 2, 1, 3)
	return Reshape(transposed, batchSize*numHeads, seqLen, headDim)
}

func mergeHeadsNode(bhsd *Node, batchSize, seqLen, numHeads, headDim int) *Node {
	reshaped := Reshape(bhsd, batchSize, numHeads, seqLen, headDim)
	transposed := TransposeAllDims(reshaped, 0, 2, 1, 3)
	return Reshape(transposed, batchSize*seqLen, numHeads*headDim)
}

func captureQKVSplitHeadsVJPCase(s *SegmentedTrainer, batchSize, seqLen int) qkvSplitVJPCase {
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	hiddenSize := numHeads * headDim
	total := batchSize * seqLen
	outer := batchSize * numHeads
	flat := makeFlatQKVTensor(total, hiddenSize, 1)
	upstream := makeBhsdQKVTensor(outer, seqLen, headDim, 2)
	cpuGrad := make([]float32, len(flat))
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for pos := 0; pos < seqLen; pos++ {
				for dim := 0; dim < headDim; dim++ {
					outIdx := (((b*numHeads+h)*seqLen + pos) * headDim) + dim
					flatIdx := ((b*seqLen+pos)*hiddenSize + h*headDim + dim)
					cpuGrad[flatIdx] += upstream[outIdx]
				}
			}
		}
	}
	exec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, flatNode, upstreamNode *Node) []*Node {
			ctx.SetTraining(flatNode.Graph(), false)
			split := splitHeadsNode(flatNode, batchSize, seqLen, numHeads, headDim)
			loss := ReduceAllSum(Mul(split, upstreamNode))
			grads := Gradient(loss, flatNode)
			return []*Node{flattenFloat32(grads[0], len(flat))}
		})
	flatTensor := tensors.FromFlatDataAndDimensions(flat, total, hiddenSize)
	upstreamTensor := tensors.FromFlatDataAndDimensions(upstream, outer, seqLen, headDim)
	results := exec.MustExec(flatTensor, upstreamTensor)
	flatTensor.FinalizeAll()
	upstreamTensor.FinalizeAll()
	defer results[0].FinalizeAll()
	grad := readFloat32Slice(results[0])
	return qkvSplitVJPCase{
		Status:     "captured",
		Batch:      batchSize,
		SeqLen:     seqLen,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		HiddenSize: hiddenSize,
		Outer:      outer,
		Components: map[string]parityStats{
			"upstream":                parityStatsFromSlice(upstream),
			"flat_grad":               parityStatsFromSlice(grad),
			"cpu_flat_grad":           parityStatsFromSlice(cpuGrad),
			"flat_grad_cpu_abs_error": absErrorStats(grad, cpuGrad),
		},
	}
}

func captureQKVScoreMatmulVJPCase(s *SegmentedTrainer, batchSize, seqLen int) qkvSplitVJPCase {
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	hiddenSize := numHeads * headDim
	outer := batchSize * numHeads
	q := makeBhsdQKVTensor(outer, seqLen, headDim, 3)
	k := makeBhsdQKVTensor(outer, seqLen, headDim, 4)
	upstream := makeScoreQKVTensor(outer, seqLen, 5)
	cpuQGrad := make([]float32, len(q))
	cpuKGrad := make([]float32, len(k))
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for o := 0; o < outer; o++ {
		for query := 0; query < seqLen; query++ {
			for key := 0; key < seqLen; key++ {
				up := upstream[(o*seqLen+query)*seqLen+key] * scale
				for dim := 0; dim < headDim; dim++ {
					qIdx := (o*seqLen+query)*headDim + dim
					kIdx := (o*seqLen+key)*headDim + dim
					cpuQGrad[qIdx] += up * k[kIdx]
					cpuKGrad[kIdx] += up * q[qIdx]
				}
			}
		}
	}
	exec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, qNode, kNode, upstreamNode *Node) []*Node {
			ctx.SetTraining(qNode.Graph(), false)
			scores := Einsum("oqd,okd->oqk", qNode, kNode)
			scores = MulScalar(scores, float64(scale))
			loss := ReduceAllSum(Mul(scores, upstreamNode))
			grads := Gradient(loss, qNode, kNode)
			return []*Node{
				flattenFloat32(grads[0], len(q)),
				flattenFloat32(grads[1], len(k)),
			}
		})
	qTensor := tensors.FromFlatDataAndDimensions(q, outer, seqLen, headDim)
	kTensor := tensors.FromFlatDataAndDimensions(k, outer, seqLen, headDim)
	upstreamTensor := tensors.FromFlatDataAndDimensions(upstream, outer, seqLen, seqLen)
	results := exec.MustExec(qTensor, kTensor, upstreamTensor)
	qTensor.FinalizeAll()
	kTensor.FinalizeAll()
	upstreamTensor.FinalizeAll()
	defer results[0].FinalizeAll()
	defer results[1].FinalizeAll()
	qGrad := readFloat32Slice(results[0])
	kGrad := readFloat32Slice(results[1])
	return qkvSplitVJPCase{
		Status:     "captured",
		Batch:      batchSize,
		SeqLen:     seqLen,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		HiddenSize: hiddenSize,
		Outer:      outer,
		Components: map[string]parityStats{
			"upstream_scores_grad":  parityStatsFromSlice(upstream),
			"q_grad":                parityStatsFromSlice(qGrad),
			"k_grad":                parityStatsFromSlice(kGrad),
			"cpu_q_grad":            parityStatsFromSlice(cpuQGrad),
			"cpu_k_grad":            parityStatsFromSlice(cpuKGrad),
			"q_grad_cpu_abs_error":  absErrorStats(qGrad, cpuQGrad),
			"k_grad_cpu_abs_error":  absErrorStats(kGrad, cpuKGrad),
		},
	}
}

func captureQKVValueContextVJPCase(s *SegmentedTrainer, batchSize, seqLen int) qkvSplitVJPCase {
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	hiddenSize := numHeads * headDim
	outer := batchSize * numHeads
	probs := makeAttentionProbQKVTensor(outer, seqLen, 6)
	v := makeBhsdQKVTensor(outer, seqLen, headDim, 7)
	upstream := makeBhsdQKVTensor(outer, seqLen, headDim, 8)
	cpuProbsGrad := make([]float32, len(probs))
	cpuVGrad := make([]float32, len(v))
	for o := 0; o < outer; o++ {
		for query := 0; query < seqLen; query++ {
			for key := 0; key < seqLen; key++ {
				var sum float32
				for dim := 0; dim < headDim; dim++ {
					upstreamIdx := (o*seqLen+query)*headDim + dim
					vIdx := (o*seqLen+key)*headDim + dim
					sum += upstream[upstreamIdx] * v[vIdx]
					cpuVGrad[vIdx] += probs[(o*seqLen+query)*seqLen+key] * upstream[upstreamIdx]
				}
				cpuProbsGrad[(o*seqLen+query)*seqLen+key] = sum
			}
		}
	}
	exec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, probsNode, vNode, upstreamNode *Node) []*Node {
			ctx.SetTraining(probsNode.Graph(), false)
			contextNode := Einsum("oqk,okd->oqd", probsNode, vNode)
			loss := ReduceAllSum(Mul(contextNode, upstreamNode))
			grads := Gradient(loss, probsNode, vNode)
			return []*Node{
				flattenFloat32(grads[0], len(probs)),
				flattenFloat32(grads[1], len(v)),
			}
		})
	probsTensor := tensors.FromFlatDataAndDimensions(probs, outer, seqLen, seqLen)
	vTensor := tensors.FromFlatDataAndDimensions(v, outer, seqLen, headDim)
	upstreamTensor := tensors.FromFlatDataAndDimensions(upstream, outer, seqLen, headDim)
	results := exec.MustExec(probsTensor, vTensor, upstreamTensor)
	probsTensor.FinalizeAll()
	vTensor.FinalizeAll()
	upstreamTensor.FinalizeAll()
	defer results[0].FinalizeAll()
	defer results[1].FinalizeAll()
	probsGrad := readFloat32Slice(results[0])
	vGrad := readFloat32Slice(results[1])
	return qkvSplitVJPCase{
		Status:     "captured",
		Batch:      batchSize,
		SeqLen:     seqLen,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		HiddenSize: hiddenSize,
		Outer:      outer,
		Components: map[string]parityStats{
			"upstream_context_grad":    parityStatsFromSlice(upstream),
			"probs_grad":               parityStatsFromSlice(probsGrad),
			"v_grad":                   parityStatsFromSlice(vGrad),
			"cpu_probs_grad":           parityStatsFromSlice(cpuProbsGrad),
			"cpu_v_grad":               parityStatsFromSlice(cpuVGrad),
			"probs_grad_cpu_abs_error": absErrorStats(probsGrad, cpuProbsGrad),
			"v_grad_cpu_abs_error":     absErrorStats(vGrad, cpuVGrad),
		},
	}
}

func captureQKVRopeVJPCase(s *SegmentedTrainer, batchSize, seqLen, probeLayer int) qkvSplitVJPCase {
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	hiddenSize := numHeads * headDim
	outer := batchSize * numHeads
	q := makeBhsdQKVTensor(outer, seqLen, headDim, 11)
	k := makeBhsdQKVTensor(outer, seqLen, headDim, 12)
	upstreamQ := makeBhsdQKVTensor(outer, seqLen, headDim, 13)
	upstreamK := makeBhsdQKVTensor(outer, seqLen, headDim, 14)
	ropeTheta := s.Model.Config.LocalRoPETheta
	if s.Model.Config.GlobalAttnEveryNLayers > 0 && probeLayer%s.Model.Config.GlobalAttnEveryNLayers == 0 {
		ropeTheta = s.Model.Config.GlobalRoPETheta
	}
	if ropeTheta == 0 {
		ropeTheta = s.Model.Config.RoPEBase
	}
	exec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, qNode, kNode, upstreamQNode, upstreamKNode *Node) []*Node {
			ctx.SetTraining(qNode.Graph(), false)
			qBhsd := Reshape(qNode, batchSize, numHeads, seqLen, headDim)
			kBhsd := Reshape(kNode, batchSize, numHeads, seqLen, headDim)
			upstreamQBhsd := Reshape(upstreamQNode, batchSize, numHeads, seqLen, headDim)
			upstreamKBhsd := Reshape(upstreamKNode, batchSize, numHeads, seqLen, headDim)
			qRoped := applyRoPE(qBhsd, seqLen, headDim, ropeTheta)
			kRoped := applyRoPE(kBhsd, seqLen, headDim, ropeTheta)
			loss := ReduceAllSum(Add(Mul(qRoped, upstreamQBhsd), Mul(kRoped, upstreamKBhsd)))
			grads := Gradient(loss, qNode, kNode)
			return []*Node{
				flattenFloat32(grads[0], len(q)),
				flattenFloat32(grads[1], len(k)),
			}
		})
	qTensor := tensors.FromFlatDataAndDimensions(q, outer, seqLen, headDim)
	kTensor := tensors.FromFlatDataAndDimensions(k, outer, seqLen, headDim)
	upstreamQTensor := tensors.FromFlatDataAndDimensions(upstreamQ, outer, seqLen, headDim)
	upstreamKTensor := tensors.FromFlatDataAndDimensions(upstreamK, outer, seqLen, headDim)
	results := exec.MustExec(qTensor, kTensor, upstreamQTensor, upstreamKTensor)
	qTensor.FinalizeAll()
	kTensor.FinalizeAll()
	upstreamQTensor.FinalizeAll()
	upstreamKTensor.FinalizeAll()
	defer results[0].FinalizeAll()
	defer results[1].FinalizeAll()
	qGrad := readFloat32Slice(results[0])
	kGrad := readFloat32Slice(results[1])
	return qkvSplitVJPCase{
		Status:     "captured",
		Batch:      batchSize,
		SeqLen:     seqLen,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		HiddenSize: hiddenSize,
		Outer:      outer,
		Components: map[string]parityStats{
			"upstream_q_grad": parityStatsFromSlice(upstreamQ),
			"upstream_k_grad": parityStatsFromSlice(upstreamK),
			"q_grad":          parityStatsFromSlice(qGrad),
			"k_grad":          parityStatsFromSlice(kGrad),
		},
	}
}

func captureQKVSumConsistencyCase(s *SegmentedTrainer, batchSize, seqLen int) qkvSplitVJPCase {
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	hiddenSize := numHeads * headDim
	total := batchSize * seqLen
	outer := batchSize * numHeads
	x := makeFlatQKVTensor(total, hiddenSize, 9)
	upstream := makeFlatQKVTensor(total, hiddenSize, 10)
	scale := 1.0 / math.Sqrt(float64(headDim))
	sharedExec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, xNode, upstreamNode *Node) []*Node {
			ctx.SetTraining(xNode.Graph(), false)
			q := splitHeadsNode(xNode, batchSize, seqLen, numHeads, headDim)
			k := splitHeadsNode(xNode, batchSize, seqLen, numHeads, headDim)
			v := splitHeadsNode(xNode, batchSize, seqLen, numHeads, headDim)
			scores := Einsum("oqd,okd->oqk", q, k)
			scores = MulScalar(scores, scale)
			probs := Softmax(scores, -1)
			contextNode := Einsum("oqk,okd->oqd", probs, v)
			merged := mergeHeadsNode(contextNode, batchSize, seqLen, numHeads, headDim)
			loss := ReduceAllSum(Mul(merged, upstreamNode))
			grads := Gradient(loss, xNode)
			return []*Node{flattenFloat32(grads[0], len(x))}
		})
	separateExec := context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, qNode, kNode, vNode, upstreamNode *Node) []*Node {
			ctx.SetTraining(qNode.Graph(), false)
			q := splitHeadsNode(qNode, batchSize, seqLen, numHeads, headDim)
			k := splitHeadsNode(kNode, batchSize, seqLen, numHeads, headDim)
			v := splitHeadsNode(vNode, batchSize, seqLen, numHeads, headDim)
			scores := Einsum("oqd,okd->oqk", q, k)
			scores = MulScalar(scores, scale)
			probs := Softmax(scores, -1)
			contextNode := Einsum("oqk,okd->oqd", probs, v)
			merged := mergeHeadsNode(contextNode, batchSize, seqLen, numHeads, headDim)
			loss := ReduceAllSum(Mul(merged, upstreamNode))
			grads := Gradient(loss, qNode, kNode, vNode)
			return []*Node{
				flattenFloat32(grads[0], len(x)),
				flattenFloat32(grads[1], len(x)),
				flattenFloat32(grads[2], len(x)),
			}
		})
	xTensor := tensors.FromFlatDataAndDimensions(x, total, hiddenSize)
	upstreamTensor := tensors.FromFlatDataAndDimensions(upstream, total, hiddenSize)
	sharedResults := sharedExec.MustExec(xTensor, upstreamTensor)
	xTensor.FinalizeAll()
	upstreamTensor.FinalizeAll()
	defer sharedResults[0].FinalizeAll()
	sharedGrad := append([]float32(nil), readFloat32Slice(sharedResults[0])...)

	qTensor := tensors.FromFlatDataAndDimensions(x, total, hiddenSize)
	kTensor := tensors.FromFlatDataAndDimensions(x, total, hiddenSize)
	vTensor := tensors.FromFlatDataAndDimensions(x, total, hiddenSize)
	upstreamTensor2 := tensors.FromFlatDataAndDimensions(upstream, total, hiddenSize)
	separateResults := separateExec.MustExec(qTensor, kTensor, vTensor, upstreamTensor2)
	qTensor.FinalizeAll()
	kTensor.FinalizeAll()
	vTensor.FinalizeAll()
	upstreamTensor2.FinalizeAll()
	defer separateResults[0].FinalizeAll()
	defer separateResults[1].FinalizeAll()
	defer separateResults[2].FinalizeAll()
	qGrad := append([]float32(nil), readFloat32Slice(separateResults[0])...)
	kGrad := append([]float32(nil), readFloat32Slice(separateResults[1])...)
	vGrad := append([]float32(nil), readFloat32Slice(separateResults[2])...)
	sumGrad := make([]float32, len(x))
	delta := make([]float32, len(x))
	for i := range sumGrad {
		sumGrad[i] = qGrad[i] + kGrad[i] + vGrad[i]
		delta[i] = sharedGrad[i] - sumGrad[i]
	}
	return qkvSplitVJPCase{
		Status:     "captured",
		Batch:      batchSize,
		SeqLen:     seqLen,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		HiddenSize: hiddenSize,
		Outer:      outer,
		Components: map[string]parityStats{
			"shared_grad":       parityStatsFromSlice(sharedGrad),
			"separate_q_grad":   parityStatsFromSlice(qGrad),
			"separate_k_grad":   parityStatsFromSlice(kGrad),
			"separate_v_grad":   parityStatsFromSlice(vGrad),
			"sum_separate_grad": parityStatsFromSlice(sumGrad),
			"shared_minus_sum":  parityStatsFromSlice(delta),
		},
	}
}

func captureQKVSplitVJPProbe(s *SegmentedTrainer, batchSize, seqLen, probeLayer int) *qkvSplitVJPProbe {
	probe := &qkvSplitVJPProbe{
		Status:  "captured",
		Version: 1,
		Runtime: "mpsgraph",
		Cases:   map[string]qkvSplitVJPCase{},
	}
	probe.Cases["split_heads_vjp"] = captureQKVSplitHeadsVJPCase(s, batchSize, seqLen)
	probe.Cases["score_matmul_vjp"] = captureQKVScoreMatmulVJPCase(s, batchSize, seqLen)
	probe.Cases["value_context_vjp"] = captureQKVValueContextVJPCase(s, batchSize, seqLen)
	probe.Cases["rope_qk_vjp"] = captureQKVRopeVJPCase(s, batchSize, seqLen, probeLayer)
	probe.Cases["qkv_sum_consistency"] = captureQKVSumConsistencyCase(s, batchSize, seqLen)
	return probe
}

const projectionDecompositionNodeCount = 10

func linearBaseProjectionNodes(ctx *context.Context, input *Node, outputDim int) (base *Node, weight *Node, bias *Node) {
	g := input.Graph()
	dtype := input.DType()
	rank := input.Rank()
	inputDim := input.Shape().Dimensions[rank-1]
	weightVar := ctx.VariableWithShape("weight", shapes.Make(dtypes.Float32, inputDim, outputDim))
	weight = weightVar.ValueGraph(g)
	biasVar := ctx.VariableWithShape("bias", shapes.Make(dtypes.Float32, outputDim))
	bias = biasVar.ValueGraph(g)

	weightForMatmul := weight
	biasForAdd := bias
	if weightForMatmul.DType() != dtype {
		weightForMatmul = ConvertDType(weightForMatmul, dtype)
		biasForAdd = ConvertDType(biasForAdd, dtype)
	}

	if rank == 2 {
		base = Dot(input, weightForMatmul).MatMul()
		base = Add(base, Reshape(biasForAdd, 1, outputDim))
		return base, weight, bias
	}
	if rank == 3 {
		batchSize := input.Shape().Dimensions[0]
		seqLen := input.Shape().Dimensions[1]
		input2D := Reshape(input, batchSize*seqLen, inputDim)
		base2D := Dot(input2D, weightForMatmul).MatMul()
		base = Reshape(base2D, batchSize, seqLen, outputDim)
		base = Add(base, Reshape(biasForAdd, 1, 1, outputDim))
		return base, weight, bias
	}
	panic(fmt.Sprintf("linearBaseProjectionNodes: unsupported input rank %d", rank))
}

func loraProjectionNodes(adapterCtx *context.Context, input *Node, inputDim, outputDim int, cfg *lora.Config) (aProj *Node, bProj *Node, delta *Node, loraA *Node, loraB *Node) {
	g := input.Graph()
	dtype := input.DType()
	rank := cfg.Rank()
	loraAVar := adapterCtx.VariableWithShape("lora_A", shapes.Make(dtypes.Float32, inputDim, rank))
	loraBVar := adapterCtx.VariableWithShape("lora_B", shapes.Make(dtypes.Float32, rank, outputDim))
	loraA = loraAVar.ValueGraph(g)
	loraB = loraBVar.ValueGraph(g)
	loraAForMatmul := loraA
	loraBForMatmul := loraB
	if loraAForMatmul.DType() != dtype {
		loraAForMatmul = ConvertDType(loraAForMatmul, dtype)
		loraBForMatmul = ConvertDType(loraBForMatmul, dtype)
	}

	if input.Rank() == 2 {
		aProj = Dot(input, loraAForMatmul).MatMul()
		bProj = Dot(aProj, loraBForMatmul).MatMul()
	} else if input.Rank() == 3 {
		batchSize := input.Shape().Dimensions[0]
		seqLen := input.Shape().Dimensions[1]
		input2D := Reshape(input, batchSize*seqLen, inputDim)
		aProj2D := Dot(input2D, loraAForMatmul).MatMul()
		bProj2D := Dot(aProj2D, loraBForMatmul).MatMul()
		aProj = Reshape(aProj2D, batchSize, seqLen, rank)
		bProj = Reshape(bProj2D, batchSize, seqLen, outputDim)
	} else {
		panic(fmt.Sprintf("loraProjectionNodes: unsupported input rank %d", input.Rank()))
	}

	delta = bProj
	if scale := cfg.Scale(); scale != 1.0 {
		delta = MulScalar(delta, scale)
	}
	return aProj, bProj, delta, loraA, loraB
}

func projectionDecompositionNodes(baseCtx, adapterCtx *context.Context, input *Node, outputDim int, cfg *lora.Config) []*Node {
	inputDim := input.Shape().Dimensions[input.Rank()-1]
	batchSize := input.Shape().Dimensions[0]
	seqLen := input.Shape().Dimensions[1]
	rank := cfg.Rank()
	base, weight, bias := linearBaseProjectionNodes(baseCtx, input, outputDim)
	aProj, bProj, delta, loraA, loraB := loraProjectionNodes(adapterCtx, input, inputDim, outputDim, cfg)
	output := Add(base, delta)
	return []*Node{
		flattenFloat32(input, batchSize*seqLen*inputDim),
		flattenFloat32(base, batchSize*seqLen*outputDim),
		flattenFloat32(aProj, batchSize*seqLen*rank),
		flattenFloat32(bProj, batchSize*seqLen*outputDim),
		flattenFloat32(delta, batchSize*seqLen*outputDim),
		flattenFloat32(output, batchSize*seqLen*outputDim),
		flattenFloat32(weight, inputDim*outputDim),
		flattenFloat32(bias, outputDim),
		flattenFloat32(loraA, inputDim*rank),
		flattenFloat32(loraB, rank*outputDim),
	}
}

func buildLayerProjectionDecompositionProbeExec(s *SegmentedTrainer, layerIdx int) *context.Exec {
	hiddenSize := s.Model.Config.HiddenSize
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	intermediate := s.Model.Config.IntermediateSize
	layerKey := strconv.Itoa(layerIdx)
	return context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, hidden, attentionMask *Node) []*Node {
			g := hidden.Graph()
			ctx.SetTraining(g, false)
			batchSize := hidden.Shape().Dimensions[0]
			seqLen := hidden.Shape().Dimensions[1]
			layerCtx := ctx.In("fused_chunker_embedder").In("encoder").In("layer").In(layerKey)
			isGlobal := s.Model.Config.GlobalAttnEveryNLayers > 0 && layerIdx%s.Model.Config.GlobalAttnEveryNLayers == 0
			ropeTheta := s.Model.Config.LocalRoPETheta
			if isGlobal {
				ropeTheta = s.Model.Config.GlobalRoPETheta
			}
			if ropeTheta == 0 {
				ropeTheta = s.Model.Config.RoPEBase
			}

			normed := s.Model.layerNorm(layerCtx.In("attn_norm"), hidden)
			if s.Model.Config.UseBF16() {
				normed = ConvertDType(normed, dtypes.BFloat16)
			}
			attnCtx := layerCtx.In("attn")
			queryParts := projectionDecompositionNodes(attnCtx.In("query_proj"), attnCtx.In("query_proj"), normed, hiddenSize, s.Model.LoRAConfig)
			keyParts := projectionDecompositionNodes(attnCtx.In("key_proj"), attnCtx.In("key_proj"), normed, hiddenSize, s.Model.LoRAConfig)
			valueParts := projectionDecompositionNodes(attnCtx.In("value_proj"), attnCtx.In("value_proj"), normed, hiddenSize, s.Model.LoRAConfig)
			queryRaw := s.Model.linearWithAdapter(attnCtx.In("query_proj"), normed, hiddenSize, true, "", s.Model.LoRAConfig)
			keyRaw := s.Model.linearWithAdapter(attnCtx.In("key_proj"), normed, hiddenSize, true, "", s.Model.LoRAConfig)
			valueRaw := s.Model.linearWithAdapter(attnCtx.In("value_proj"), normed, hiddenSize, true, "", s.Model.LoRAConfig)

			query := Reshape(queryRaw, batchSize, seqLen, numHeads, headDim)
			query = TransposeAllDims(query, 0, 2, 1, 3)
			key := Reshape(keyRaw, batchSize, seqLen, numHeads, headDim)
			key = TransposeAllDims(key, 0, 2, 1, 3)
			value := Reshape(valueRaw, batchSize, seqLen, numHeads, headDim)
			value = TransposeAllDims(value, 0, 2, 1, 3)

			query = applyRoPE(query, seqLen, headDim, ropeTheta)
			key = applyRoPE(key, seqLen, headDim, ropeTheta)
			scale := 1.0 / math.Sqrt(float64(headDim))
			attnScores := Einsum("bhqd,bhkd->bhqk", query, key)
			attnScores = MulScalar(attnScores, scale)
			dtype := attnScores.DType()
			if attentionMask != nil {
				mask := Reshape(attentionMask, batchSize, 1, 1, seqLen)
				maskBias := MulScalar(Sub(Ones(g, mask.Shape()), ConvertDType(mask, dtype)), -10000.0)
				attnScores = Add(attnScores, maskBias)
			}
			if !isGlobal && s.Model.Config.LocalAttention > 0 {
				windowMask := createSlidingWindowMask(g, seqLen, s.Model.Config.LocalAttention, dtype)
				attnScores = Add(attnScores, windowMask)
			}
			attnScoresDType := attnScores.DType()
			if attnScoresDType == dtypes.BFloat16 {
				attnScores = ConvertDType(attnScores, dtypes.Float32)
			}
			attnProbs := Softmax(attnScores, -1)
			if attnScoresDType == dtypes.BFloat16 {
				attnProbs = ConvertDType(attnProbs, dtypes.BFloat16)
			}
			attnOutput := Einsum("bhqk,bhkd->bhqd", attnProbs, value)
			attnOutput = TransposeAllDims(attnOutput, 0, 2, 1, 3)
			attnOutput = Reshape(attnOutput, batchSize, seqLen, hiddenSize)

			outParts := projectionDecompositionNodes(attnCtx.In("Wo"), attnCtx.In("Wo"), attnOutput, hiddenSize, s.Model.LoRAConfig)
			projected := s.Model.linearWithAdapter(attnCtx.In("Wo"), attnOutput, hiddenSize, true, "", s.Model.LoRAConfig)
			if s.Model.Config.UseBF16() {
				projected = ConvertDType(projected, dtypes.Float32)
			}
			hiddenAfterAttn := Add(hidden, projected)
			normedFFN := s.Model.layerNorm(layerCtx.In("mlp_norm"), hiddenAfterAttn)
			if s.Model.Config.UseBF16() {
				normedFFN = ConvertDType(normedFFN, dtypes.BFloat16)
			}
			gated := s.Model.linearWithAdapter(layerCtx.In("mlp").In("Wi"), normedFFN, intermediate*2, true, "", s.Model.LoRAConfig)
			gateInput := Reshape(
				Slice(gated, AxisRange(0, batchSize), AxisRange(0, seqLen), AxisRange(0, intermediate)),
				batchSize, seqLen, intermediate)
			gateValue := Reshape(
				Slice(gated, AxisRange(0, batchSize), AxisRange(0, seqLen), AxisRange(intermediate, intermediate*2)),
				batchSize, seqLen, intermediate)
			geluInput := activations.Gelu(gateInput)
			woInput := Mul(geluInput, gateValue)
			woParts := projectionDecompositionNodes(layerCtx.In("mlp").In("Wo"), layerCtx.In("mlp").In("Wo"), woInput, hiddenSize, s.Model.LoRAConfig)
			ffnOut := s.Model.linearWithAdapter(layerCtx.In("mlp").In("Wo"), woInput, hiddenSize, true, "", s.Model.LoRAConfig)
			if s.Model.Config.UseBF16() {
				ffnOut = ConvertDType(ffnOut, dtypes.Float32)
			}
			layerOutput := Add(hiddenAfterAttn, ffnOut)

			nodes := make([]*Node, 0, 5*projectionDecompositionNodeCount+9)
			nodes = append(nodes, queryParts...)
			nodes = append(nodes, keyParts...)
			nodes = append(nodes, valueParts...)
			nodes = append(nodes, outParts...)
			nodes = append(nodes, woParts...)
			nodes = append(nodes,
				flattenFloat32(hiddenAfterAttn, batchSize*seqLen*hiddenSize),
				flattenFloat32(normedFFN, batchSize*seqLen*hiddenSize),
				flattenFloat32(gated, batchSize*seqLen*intermediate*2),
				flattenFloat32(gateInput, batchSize*seqLen*intermediate),
				flattenFloat32(gateValue, batchSize*seqLen*intermediate),
				flattenFloat32(geluInput, batchSize*seqLen*intermediate),
				flattenFloat32(woInput, batchSize*seqLen*intermediate),
				flattenFloat32(ffnOut, batchSize*seqLen*hiddenSize),
				flattenFloat32(layerOutput, batchSize*seqLen*hiddenSize),
			)
			return nodes
		})
}

func projectionProbeFromResults(results []*tensors.Tensor, start int, rows, inDim, outDim, rank int, scale float64, hasBias bool) projectionDecompositionProbe {
	input := readFloat32Slice(results[start+0])
	base := readFloat32Slice(results[start+1])
	loraAProj := readFloat32Slice(results[start+2])
	loraBProj := readFloat32Slice(results[start+3])
	delta := readFloat32Slice(results[start+4])
	output := readFloat32Slice(results[start+5])
	weight := readFloat32Slice(results[start+6])
	bias := readFloat32Slice(results[start+7])
	loraAWeight := readFloat32Slice(results[start+8])
	loraBWeight := readFloat32Slice(results[start+9])
	return projectionDecompositionProbe{
		Scale:       scale,
		Rank:        rank,
		Rows:        rows,
		InDim:       inDim,
		OutDim:      outDim,
		HasBias:     hasBias,
		Input:       tensorProbeStatsFromSlice(input),
		Base:        tensorProbeStatsFromSlice(base),
		LoRAA:       tensorProbeStatsFromSlice(loraAProj),
		LoRAB:       tensorProbeStatsFromSlice(loraBProj),
		Delta:       tensorProbeStatsFromSlice(delta),
		Output:      tensorProbeStatsFromSlice(output),
		Weight:      tensorProbeStatsFromTransposedSlice(weight, inDim, outDim),
		Bias:        tensorProbeStatsFromSlice(bias),
		LoRAAWeight: tensorProbeStatsFromTransposedSlice(loraAWeight, inDim, rank),
		LoRABWeight: tensorProbeStatsFromTransposedSlice(loraBWeight, rank, outDim),
		BaseReferenceError:  linearReferenceErrorStats(base, input, weight, bias, rows, inDim, outDim),
		LoRAAReferenceError: linearReferenceErrorStats(loraAProj, input, loraAWeight, nil, rows, inDim, rank),
		LoRABReferenceError: linearReferenceErrorStats(loraBProj, loraAProj, loraBWeight, nil, rows, rank, outDim),
		DeltaReferenceError: scaleReferenceErrorStats(delta, loraBProj, scale),
		OutputReferenceError: addReferenceErrorStats(output, base, delta),
	}
}

func captureEncoderLayerReplayProbe(t *testing.T, s *SegmentedTrainer, batch *FusedBatch, replayPath string, actualInput []float32, probeLayer int) *encoderLayerReplayProbe {
	t.Helper()
	if replayPath == "" {
		return nil
	}
	batchSize := batch.InputIDs.Shape().Dimensions[0]
	seqLen := s.Config.MaxSeqLen
	hiddenSize := s.Model.Config.HiddenSize
	numHeads := s.Model.Config.NumAttentionHeads
	headDim := s.Model.Config.HeadDim()
	intermediate := s.Model.Config.IntermediateSize
	rows := batchSize * seqLen
	expectedElems := rows * hiddenSize
	values := readFloat32LittleEndianFile(t, replayPath, expectedElems)
	var actualInputDiff tensorProbeStats
	var actualInputValidTokenDiff tensorProbeStats
	var actualInputPaddingTokenDiff tensorProbeStats
	if len(actualInput) == len(values) {
		actualInputDiff = tensorProbeStatsFromSlice(diffFloat32Slices(actualInput, values))
	}
	targetHidden := tensors.FromFlatDataAndDimensions(values, batchSize, seqLen, hiddenSize)
	defer targetHidden.FinalizeAll()
	attentionMask := s.extractAttentionMask(batch)
	defer attentionMask.FinalizeAll()
	attentionMaskFlat := readInt32Slice(batch.AttentionMask)
	if len(actualInput) == len(values) {
		actualInputValidTokenDiff = tensorProbeStatsFromMaskedHiddenDiff(actualInput, values, attentionMaskFlat, hiddenSize, true)
		actualInputPaddingTokenDiff = tensorProbeStatsFromMaskedHiddenDiff(actualInput, values, attentionMaskFlat, hiddenSize, false)
	}
	layerPrefix := fmt.Sprintf("layer_%02d", probeLayer)

	layerActivationExec := buildLayerActivationInputProbeExec(s, probeLayer)
	layerActivationResults := layerActivationExec.MustExec(targetHidden, attentionMask)
	layerNormed := readFloat32Slice(layerActivationResults[0])
	layerQRaw := readFloat32Slice(layerActivationResults[1])
	layerKRaw := readFloat32Slice(layerActivationResults[2])
	layerVRaw := readFloat32Slice(layerActivationResults[3])
	layerQRope := readFloat32Slice(layerActivationResults[4])
	layerKRope := readFloat32Slice(layerActivationResults[5])
	layerAttnOut := readFloat32Slice(layerActivationResults[6])
	layerWoInput := readFloat32Slice(layerActivationResults[7])
	layerScoresRaw := readFloat32Slice(layerActivationResults[8])
	layerScoresMasked := readFloat32Slice(layerActivationResults[9])
	layerAttnProbs := readFloat32Slice(layerActivationResults[10])
	layerContextRef := attentionContextReference(layerAttnProbs, layerVRaw, batchSize, seqLen, numHeads, headDim, "token")
	layerContextDelta := diffFloat32Slices(layerAttnOut, layerContextRef)
	isLocalAttention := true
	if s.Model.Config.GlobalAttnEveryNLayers > 0 && probeLayer%s.Model.Config.GlobalAttnEveryNLayers == 0 {
		isLocalAttention = false
	}
	layerTokenRef := sdpaReference(layerQRope, layerKRope, layerVRaw, attentionMaskFlat, batchSize, seqLen, numHeads, headDim, "token", isLocalAttention, s.Model.Config.LocalAttention)
	layerKernelRef := sdpaReference(layerQRope, layerKRope, layerVRaw, attentionMaskFlat, batchSize, seqLen, numHeads, headDim, "kernel", isLocalAttention, s.Model.Config.LocalAttention)
	layerTokenDelta := diffFloat32Slices(layerAttnOut, layerTokenRef)
	layerKernelDelta := diffFloat32Slices(layerAttnOut, layerKernelRef)
	for _, tensor := range layerActivationResults {
		tensor.FinalizeAll()
	}
	encoderActivationInputProbe := map[string]tensorProbeStats{
		layerPrefix + "_query_proj": tensorProbeStatsFromSlice(layerNormed),
		layerPrefix + "_key_proj":   tensorProbeStatsFromSlice(layerNormed),
		layerPrefix + "_value_proj": tensorProbeStatsFromSlice(layerNormed),
		layerPrefix + "_out_proj":   tensorProbeStatsFromSlice(layerAttnOut),
		layerPrefix + "_wo":         tensorProbeStatsFromSlice(layerWoInput),
	}
	encoderAttentionInternalProbe := map[string]tensorProbeStats{
		layerPrefix + "_q_raw":              tensorProbeStatsFromSlice(layerQRaw),
		layerPrefix + "_k_raw":              tensorProbeStatsFromSlice(layerKRaw),
		layerPrefix + "_v_raw":              tensorProbeStatsFromSlice(layerVRaw),
		layerPrefix + "_q_rope":             tensorProbeStatsFromSlice(layerQRope),
		layerPrefix + "_k_rope":             tensorProbeStatsFromSlice(layerKRope),
		layerPrefix + "_attn_scores_raw":    tensorProbeStatsFromSlice(layerScoresRaw),
		layerPrefix + "_attn_scores_masked": tensorProbeStatsFromSlice(layerScoresMasked),
		layerPrefix + "_attn_probs":         tensorProbeStatsFromSlice(layerAttnProbs),
		layerPrefix + "_attn_context_ref":   tensorProbeStatsFromSlice(layerContextRef),
		layerPrefix + "_attn_context_delta": tensorProbeStatsFromSlice(layerContextDelta),
		layerPrefix + "_attn_output":        tensorProbeStatsFromSlice(layerAttnOut),
		layerPrefix + "_attn_token_ref":     tensorProbeStatsFromSlice(layerTokenRef),
		layerPrefix + "_attn_token_delta":   tensorProbeStatsFromSlice(layerTokenDelta),
		layerPrefix + "_attn_kernel_ref":    tensorProbeStatsFromSlice(layerKernelRef),
		layerPrefix + "_attn_kernel_delta":  tensorProbeStatsFromSlice(layerKernelDelta),
	}

	layerProjectionExec := buildLayerProjectionDecompositionProbeExec(s, probeLayer)
	layerProjectionResults := layerProjectionExec.MustExec(targetHidden, attentionMask)
	loraRank := s.Model.LoRAConfig.Rank()
	loraScale := s.Model.LoRAConfig.Scale()
	encoderProjectionDecompositionProbe := map[string]projectionDecompositionProbe{
		layerPrefix + "_query_proj": projectionProbeFromResults(layerProjectionResults, 0*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_key_proj":   projectionProbeFromResults(layerProjectionResults, 1*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_value_proj": projectionProbeFromResults(layerProjectionResults, 2*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_out_proj":   projectionProbeFromResults(layerProjectionResults, 3*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_wo":         projectionProbeFromResults(layerProjectionResults, 4*projectionDecompositionNodeCount, rows, intermediate, hiddenSize, loraRank, loraScale, true),
	}
	layerStateOffset := 5 * projectionDecompositionNodeCount
	layerStateNames := []string{
		"hidden_after_attn",
		"mlp_norm_output",
		"wi_output",
		"gate_input",
		"gate_value",
		"gelu_input",
		"wo_input",
		"ffn_out",
		"layer_output",
	}
	encoderLayerStateProbe := map[string]tensorProbeStats{}
	for i, name := range layerStateNames {
		if layerStateOffset+i >= len(layerProjectionResults) {
			break
		}
		encoderLayerStateProbe[layerPrefix+"_"+name] = tensorProbeStatsFromTensor(layerProjectionResults[layerStateOffset+i])
	}
	for _, tensor := range layerProjectionResults {
		tensor.FinalizeAll()
	}

	return &encoderLayerReplayProbe{
		Status:                              "captured",
		Path:                                replayPath,
		TargetLayer:                         probeLayer,
		BatchSize:                           batchSize,
		SeqLen:                              seqLen,
		HiddenSize:                          hiddenSize,
		Elems:                               expectedElems,
		AttentionMaskValidLengths:           attentionMaskValidLengths(attentionMaskFlat, batchSize, seqLen),
		Input:                               tensorProbeStatsFromSlice(values),
		ActualInputDiff:                     actualInputDiff,
		ActualInputValidTokenDiff:           actualInputValidTokenDiff,
		ActualInputPaddingTokenDiff:         actualInputPaddingTokenDiff,
		EncoderActivationInputProbe:          encoderActivationInputProbe,
		EncoderAttentionInternalProbe:        encoderAttentionInternalProbe,
		EncoderProjectionDecompositionProbe:  encoderProjectionDecompositionProbe,
		EncoderLayerStateProbe:               encoderLayerStateProbe,
	}
}

func captureEncoderLayerBackwardReplayProbe(t *testing.T, s *SegmentedTrainer, batch *FusedBatch, inputPath, upstreamPath string, probeLayer int, layerBackwardDecompEnabled bool) *encoderLayerBackwardReplayProbe {
	t.Helper()
	if upstreamPath == "" {
		return nil
	}
	if inputPath == "" {
		return &encoderLayerBackwardReplayProbe{
			Status:       "missing",
			Reason:       "missing_replay_input",
			UpstreamPath: upstreamPath,
			TargetLayer:  probeLayer,
		}
	}
	batchSize := batch.InputIDs.Shape().Dimensions[0]
	seqLen := s.Config.MaxSeqLen
	hiddenSize := s.Model.Config.HiddenSize
	rows := batchSize * seqLen
	expectedElems := rows * hiddenSize
	inputValues := readFloat32LittleEndianFile(t, inputPath, expectedElems)
	upstreamValues := readFloat32LittleEndianFile(t, upstreamPath, expectedElems)
	hidden := tensors.FromFlatDataAndDimensions(inputValues, batchSize, seqLen, hiddenSize)
	defer hidden.FinalizeAll()
	upstream := tensors.FromFlatDataAndDimensions(upstreamValues, batchSize, seqLen, hiddenSize)
	defer upstream.FinalizeAll()
	attentionMask := s.extractAttentionMask(batch)
	defer attentionMask.FinalizeAll()
	if probeLayer < 0 || probeLayer >= len(s.layerBackwardExecs) {
		t.Fatalf("probe layer %d out of range for %d layer backward execs", probeLayer, len(s.layerBackwardExecs))
	}
	layerBackResults := s.layerBackwardExecs[probeLayer].MustExec(hidden, upstream, attentionMask)
	segmentProbe := captureSegmentVJPProbe(
		probeLayer,
		probeLayer,
		parityStatsFromSlice(upstreamValues),
		layerBackResults,
		s.layerBackGradNames[probeLayer],
	)
	var decompProbe *layerBackwardDecompProbe
	if layerBackwardDecompEnabled {
		decompProbe = captureLayerBackwardDecompProbe(
			s,
			probeLayer,
			probeLayer,
			hidden,
			upstream,
			attentionMask,
			segmentProbe,
		)
	}
	for _, tensor := range layerBackResults {
		tensor.FinalizeAll()
	}
	return &encoderLayerBackwardReplayProbe{
		Status:                   "captured",
		InputPath:                inputPath,
		UpstreamPath:             upstreamPath,
		TargetLayer:              probeLayer,
		BatchSize:                batchSize,
		SeqLen:                   seqLen,
		HiddenSize:               hiddenSize,
		Elems:                    expectedElems,
		Input:                    parityStatsFromSlice(inputValues),
		Upstream:                 parityStatsFromSlice(upstreamValues),
		SegmentVJPProbe:          segmentProbe,
		LayerBackwardDecompProbe: decompProbe,
	}
}

func buildBoundaryForwardProbeExec(s *SegmentedTrainer, seqLen int) *context.Exec {
	hiddenSize := s.Model.Config.HiddenSize
	return context.MustNewExec(s.Backend, s.Ctx.Reuse(),
		func(ctx *context.Context, packed *Node) []*Node {
			g := packed.Graph()
			ctx.SetTraining(g, true)
			batchSize := packed.Shape().Dimensions[0]
			hiddenWidth := hiddenSize * seqLen
			mlpDim := s.Model.Config.BoundaryMLPDim

			hiddenPreNorm := Reshape(
				Slice(packed, AxisRange(0, batchSize), AxisRange(0, hiddenWidth)),
				batchSize, seqLen, hiddenSize)
			hidden := s.Model.FinalNorm(ctx, hiddenPreNorm)
			ctxBoundary := ctx.In("fused_chunker_embedder").In("boundary_head")
			dense1 := s.Model.linear(ctxBoundary.In("mlp_dense1"), hidden, mlpDim)
			dense1Post := activations.Gelu(dense1)
			logits := s.Model.linear(ctxBoundary.In("mlp_dense2"), dense1Post, s.Model.Config.NumBoundaryLabels)
			return []*Node{
				Reshape(hiddenPreNorm, batchSize*seqLen*hiddenSize),
				hidden,
				Reshape(dense1, batchSize*seqLen*mlpDim),
				Reshape(dense1Post, batchSize*seqLen*mlpDim),
				Reshape(logits, batchSize*seqLen*s.Model.Config.NumBoundaryLabels),
			}
		})
}

func computeBoundaryProbeStats(logits []float32, batch *FusedBatch, posWeight float64, batchSize, seqLen int) boundaryProbeStats {
	labels := readFloat32Slice(batch.BoundaryLabels)
	attentionMask := readInt32Slice(batch.AttentionMask)
	stats := boundaryProbeStats{}
	var probPosSum, probNegSum float64
	var marginPosSum, marginNegSum float64
	var negCount int
	var logit0Sum, logit1Sum float64
	var ceSum float64
	eps := EpsilonLoss

	for b := 0; b < batchSize; b++ {
		for j := 0; j < seqLen; j++ {
			idx := b*seqLen + j
			if idx >= len(attentionMask) || attentionMask[idx] == 0 {
				continue
			}
			logitIdx := idx * 2
			if logitIdx+1 >= len(logits) || idx >= len(labels) {
				continue
			}
			z0 := float64(logits[logitIdx])
			z1 := float64(logits[logitIdx+1])
			maxLogit := z0
			if z1 > maxLogit {
				maxLogit = z1
			}
			e0 := math.Exp(z0 - maxLogit)
			e1 := math.Exp(z1 - maxLogit)
			denom := e0 + e1
			p0 := e0 / denom
			p1 := e1 / denom
			if p0 < eps {
				p0 = eps
			}
			if p1 < eps {
				p1 = eps
			}

			stats.ValidTokens++
			logit0Sum += z0
			logit1Sum += z1
			if len(stats.LogitPairSample) < 16 {
				stats.LogitPairSample = append(stats.LogitPairSample, logits[logitIdx], logits[logitIdx+1])
			}
			isPositive := labels[idx] > 0.5
			isPredicted := z1 > z0
			if isPredicted {
				stats.PredictedPositives++
			}
			margin := z1 - z0
			if isPositive {
				stats.GoldPositives++
				probPosSum += p1
				marginPosSum += margin
				ceSum += -math.Log(p1) * posWeight
				if isPredicted {
					stats.TP++
				} else {
					stats.FN++
				}
			} else {
				negCount++
				probNegSum += p1
				marginNegSum += margin
				ceSum += -math.Log(p0)
				if isPredicted {
					stats.FP++
				}
			}
		}
	}

	if stats.ValidTokens > 0 {
		denom := float64(stats.ValidTokens) + eps
		stats.CPUCELoss = ceSum / denom
		stats.Logit0Mean = logit0Sum / float64(stats.ValidTokens)
		stats.Logit1Mean = logit1Sum / float64(stats.ValidTokens)
	}
	if stats.GoldPositives > 0 {
		stats.ProbGoldPos = probPosSum / float64(stats.GoldPositives)
		stats.MarginGoldPos = marginPosSum / float64(stats.GoldPositives)
	}
	if negCount > 0 {
		stats.ProbGoldNeg = probNegSum / float64(negCount)
		stats.MarginGoldNeg = marginNegSum / float64(negCount)
	}
	return stats
}

func runMPSGraphSegmentedParityStep(t *testing.T, trainer *FusedTrainer, s *SegmentedTrainer, batch *FusedBatch, probeLayer int, layerBackwardDecompEnabled bool) ([]float32, []gradEntry, float64, float64, boundaryProbeStats, boundaryForwardProbe, embeddingProbe, map[string]tensorProbeStats, map[string]tensorProbeStats, map[string]projectionDecompositionProbe, map[string]tensorProbeStats, map[string]tensorProbeStats, map[string]tensorProbeStats, map[string]attentionRowProbe, *contrastiveStepProbe, *upstreamGradProbe, *segmentVJPProbe, *layerBackwardDecompProbe, []float32) {
	t.Helper()
	batchSize := batch.InputIDs.Shape().Dimensions[0]
	seqLen := s.Config.MaxSeqLen
	hiddenSize := s.Model.Config.HiddenSize
	epochIdx := 0
	if s.stepsPerEpoch > 0 {
		epochIdx = s.currentStep / s.stepsPerEpoch
	}

	combinedInput := s.combineInputs(batch)
	embeddingProbeExec := buildEmbeddingProbeExec(s, seqLen)
	embeddingProbeResults := embeddingProbeExec.MustExec(combinedInput)
	embeddingTokenLookup := readFloat32Slice(embeddingProbeResults[0])
	embeddingLayerNormOutput := readFloat32Slice(embeddingProbeResults[1])
	inputIDs := readInt32Slice(batch.InputIDs)
	embeddingLookupRowProbe := embeddingLookupRowProbeStats(inputIDs, embeddingTokenLookup, hiddenSize, embeddingRowProbeMax)
	for _, tensor := range embeddingProbeResults {
		tensor.FinalizeAll()
	}
	embeddingProbeStats := embeddingProbe{
		TokenLookup:     tensorProbeStatsFromSlice(embeddingTokenLookup),
		LayerNormOutput: tensorProbeStatsFromSlice(embeddingLayerNormOutput),
	}
	embResults := s.embeddingExec.MustExec(combinedInput)
	combinedInput.FinalizeAll()
	hidden := embResults[0]

	attentionMaskFlat := readInt32Slice(batch.AttentionMask)
	attentionMask := s.extractAttentionMask(batch)
	numHeads := 12
	headDim := hiddenSize / numHeads
	numSegments := len(s.layerForwardExecs)
	if probeLayer < 0 || probeLayer >= numSegments {
		t.Fatalf("FUSED_STEP_PARITY_DEBUG_ENCODER_PROBE_LAYER=%d out of range [0,%d)", probeLayer, numSegments)
	}
	savedHiddens := make([]*tensors.Tensor, numSegments+1)
	savedHiddens[0] = hidden
	for seg := 0; seg < numSegments; seg++ {
		layerResults := s.layerForwardExecs[seg].MustExec(hidden, attentionMask)
		hidden = layerResults[0]
		savedHiddens[seg+1] = hidden
	}
	encoderLayerProbe := make(map[string]tensorProbeStats, len(savedHiddens))
	for i, tensor := range savedHiddens {
		if tensor == nil {
			continue
		}
		encoderLayerProbe[fmt.Sprintf("layer_%02d", i)] = tensorProbeStatsFromTensor(tensor)
	}
	targetHidden := savedHiddens[probeLayer]
	targetLayerInputValues := readFloat32Slice(targetHidden)
	layerPrefix := fmt.Sprintf("layer_%02d", probeLayer)
	layerActivationExec := buildLayerActivationInputProbeExec(s, probeLayer)
	layerActivationResults := layerActivationExec.MustExec(targetHidden, attentionMask)
	layerNormed := readFloat32Slice(layerActivationResults[0])
	layerQRaw := readFloat32Slice(layerActivationResults[1])
	layerKRaw := readFloat32Slice(layerActivationResults[2])
	layerVRaw := readFloat32Slice(layerActivationResults[3])
	layerQRope := readFloat32Slice(layerActivationResults[4])
	layerKRope := readFloat32Slice(layerActivationResults[5])
	layerAttnOut := readFloat32Slice(layerActivationResults[6])
	layerWoInput := readFloat32Slice(layerActivationResults[7])
	layerScoresRaw := readFloat32Slice(layerActivationResults[8])
	layerScoresMasked := readFloat32Slice(layerActivationResults[9])
	layerAttnProbs := readFloat32Slice(layerActivationResults[10])
	layerContextRef := attentionContextReference(layerAttnProbs, layerVRaw, batchSize, seqLen, numHeads, headDim, "token")
	layerContextDelta := diffFloat32Slices(layerAttnOut, layerContextRef)
	for _, tensor := range layerActivationResults {
		tensor.FinalizeAll()
	}
	encoderAttentionInternalProbe := map[string]tensorProbeStats{
		layerPrefix + "_q_raw":              tensorProbeStatsFromSlice(layerQRaw),
		layerPrefix + "_k_raw":              tensorProbeStatsFromSlice(layerKRaw),
		layerPrefix + "_v_raw":              tensorProbeStatsFromSlice(layerVRaw),
		layerPrefix + "_q_rope":             tensorProbeStatsFromSlice(layerQRope),
		layerPrefix + "_k_rope":             tensorProbeStatsFromSlice(layerKRope),
		layerPrefix + "_attn_scores_raw":    tensorProbeStatsFromSlice(layerScoresRaw),
		layerPrefix + "_attn_scores_masked": tensorProbeStatsFromSlice(layerScoresMasked),
		layerPrefix + "_attn_probs":         tensorProbeStatsFromSlice(layerAttnProbs),
		layerPrefix + "_attn_context_ref":   tensorProbeStatsFromSlice(layerContextRef),
		layerPrefix + "_attn_context_delta": tensorProbeStatsFromSlice(layerContextDelta),
		layerPrefix + "_attn_output":        tensorProbeStatsFromSlice(layerAttnOut),
	}
	isLocalAttention := true
	if s.Model.Config.GlobalAttnEveryNLayers > 0 && probeLayer%s.Model.Config.GlobalAttnEveryNLayers == 0 {
		isLocalAttention = false
	}
	layerTokenRef := sdpaReference(layerQRope, layerKRope, layerVRaw, attentionMaskFlat, batchSize, seqLen, numHeads, headDim, "token", isLocalAttention, s.Model.Config.LocalAttention)
	layerKernelRef := sdpaReference(layerQRope, layerKRope, layerVRaw, attentionMaskFlat, batchSize, seqLen, numHeads, headDim, "kernel", isLocalAttention, s.Model.Config.LocalAttention)
	layerTokenDelta := diffFloat32Slices(layerAttnOut, layerTokenRef)
	layerKernelDelta := diffFloat32Slices(layerAttnOut, layerKernelRef)
	encoderAttentionInternalProbe[layerPrefix+"_attn_token_ref"] = tensorProbeStatsFromSlice(layerTokenRef)
	encoderAttentionInternalProbe[layerPrefix+"_attn_token_delta"] = tensorProbeStatsFromSlice(layerTokenDelta)
	encoderAttentionInternalProbe[layerPrefix+"_attn_kernel_ref"] = tensorProbeStatsFromSlice(layerKernelRef)
	encoderAttentionInternalProbe[layerPrefix+"_attn_kernel_delta"] = tensorProbeStatsFromSlice(layerKernelDelta)
	encoderAttentionRowProbe := layerAttentionRowProbes(probeLayer, layerQRope, layerKRope, layerVRaw, attentionMaskFlat, batchSize, seqLen, numHeads, headDim, isLocalAttention, s.Model.Config.LocalAttention)
	encoderActivationInputProbe := map[string]tensorProbeStats{
		layerPrefix + "_query_proj": tensorProbeStatsFromSlice(layerNormed),
		layerPrefix + "_key_proj":   tensorProbeStatsFromSlice(layerNormed),
		layerPrefix + "_value_proj": tensorProbeStatsFromSlice(layerNormed),
		layerPrefix + "_out_proj":   tensorProbeStatsFromSlice(layerAttnOut),
		layerPrefix + "_wo":         tensorProbeStatsFromSlice(layerWoInput),
	}
	layerProjectionExec := buildLayerProjectionDecompositionProbeExec(s, probeLayer)
	layerProjectionResults := layerProjectionExec.MustExec(targetHidden, attentionMask)
	loraRank := s.Model.LoRAConfig.Rank()
	loraScale := s.Model.LoRAConfig.Scale()
	rows := batchSize * seqLen
	encoderProjectionDecompositionProbe := map[string]projectionDecompositionProbe{
		layerPrefix + "_query_proj": projectionProbeFromResults(layerProjectionResults, 0*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_key_proj":   projectionProbeFromResults(layerProjectionResults, 1*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_value_proj": projectionProbeFromResults(layerProjectionResults, 2*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_out_proj":   projectionProbeFromResults(layerProjectionResults, 3*projectionDecompositionNodeCount, rows, hiddenSize, hiddenSize, loraRank, loraScale, true),
		layerPrefix + "_wo":         projectionProbeFromResults(layerProjectionResults, 4*projectionDecompositionNodeCount, rows, s.Model.Config.IntermediateSize, hiddenSize, loraRank, loraScale, true),
	}
	layerStateOffset := 5 * projectionDecompositionNodeCount
	layerStateNames := []string{
		"hidden_after_attn",
		"mlp_norm_output",
		"wi_output",
		"gate_input",
		"gate_value",
		"gelu_input",
		"wo_input",
		"ffn_out",
		"layer_output",
	}
	encoderLayerStateProbe := map[string]tensorProbeStats{}
	for i, name := range layerStateNames {
		if layerStateOffset+i >= len(layerProjectionResults) {
			break
		}
		encoderLayerStateProbe[layerPrefix+"_"+name] = tensorProbeStatsFromTensor(layerProjectionResults[layerStateOffset+i])
	}
	for _, tensor := range layerProjectionResults {
		tensor.FinalizeAll()
	}

	var allGrads []gradEntry
	var boundaryLossVal float32
	var boundaryProbe boundaryProbeStats
	var forwardProbe boundaryForwardProbe
	boundaryForwardExec := buildBoundaryForwardProbeExec(s, seqLen)
	boundaryProbeInput := s.packBoundaryInput(hidden, batch, batchSize, seqLen, hiddenSize)
	boundaryForwardResults := boundaryForwardExec.MustExec(boundaryProbeInput)
	boundaryProbeInput.FinalizeAll()
	finalNormInput := readFloat32Slice(boundaryForwardResults[0])
	boundaryHeadInput := readFloat32Slice(boundaryForwardResults[1])
	boundaryHeadTensor := boundaryForwardResults[1]
	dense1Pre := readFloat32Slice(boundaryForwardResults[2])
	dense1Post := readFloat32Slice(boundaryForwardResults[3])
	boundaryLogits := readFloat32Slice(boundaryForwardResults[4])
	for i, tensor := range boundaryForwardResults {
		if i == 1 {
			continue
		}
		tensor.FinalizeAll()
	}
	forwardProbe = boundaryForwardProbe{
		FinalNormInput:       tensorProbeStatsFromSlice(finalNormInput),
		BoundaryHeadInput:    tensorProbeStatsFromSlice(boundaryHeadInput),
		Dense1PreActivation:  tensorProbeStatsFromSlice(dense1Pre),
		Dense1PostActivation: tensorProbeStatsFromSlice(dense1Post),
		Logits:               tensorProbeStatsFromSlice(boundaryLogits),
	}
	upstreamProbe := &upstreamGradProbe{
		Status:      "captured",
		TargetLayer: probeLayer,
		Stages: map[string]tensorProbeStats{
			"final_norm_input": forwardProbe.FinalNormInput,
		},
		UpperEncoderLadder: map[string]tensorProbeStats{},
	}
	boundaryProbe = computeBoundaryProbeStats(boundaryLogits, batch, s.Config.PosWeight, batchSize, seqLen)

	boundaryInput := s.packBoundaryInput(hidden, batch, batchSize, seqLen, hiddenSize)
	boundaryResults := s.boundaryBackExec.MustExec(boundaryInput)
	boundaryInput.FinalizeAll()
	boundaryResults[0].ConstFlatData(func(flat any) {
		if data, ok := flat.([]float32); ok && len(data) > 0 {
			boundaryLossVal = data[0]
		}
	})
	boundaryResults[0].FinalizeAll()
	hiddenGradTensor := boundaryResults[1]
	upstreamProbe.Stages["boundary_features_grad"] = tensorProbeStatsFromTensor(hiddenGradTensor)
	for gi, name := range s.boundaryBackGradNames {
		if gi+2 < len(boundaryResults) {
			allGrads = append(allGrads, gradEntry{name: name, gradTensor: boundaryResults[gi+2]})
		}
	}

	var contrastiveLossVal float32
	var contrastiveProbe *contrastiveStepProbe
	if s.contrastiveForwardExec != nil {
		lossConfig := s.Config.GetLossConfig(epochIdx)
		lambdaEmbed := lossConfig.LambdaEmbed
		if lambdaEmbed > 0 {
			embeddingDim := s.Model.Config.EmbeddingDim
			maxChunksLocal := s.Config.MaxChunks
			var contrastiveFwdInput *tensors.Tensor
			contrastiveFwdInput, s.stepBuffers.contrastiveFwd = PackContrastiveForwardInput(hidden, batch, batchSize, seqLen, hiddenSize, maxChunksLocal, s.stepBuffers.contrastiveFwd)
			contrastiveFwdResults := s.contrastiveForwardExec.MustExec(contrastiveFwdInput)
			contrastiveFwdInput.FinalizeAll()
			chunkEmbFlat := readFloat32Slice(contrastiveFwdResults[0])
			contrastiveFwdResults[0].FinalizeAll()

			s.stepBuffers.chunkMask = readFloat32SliceInto(batch.ChunkMask, s.stepBuffers.chunkMask)
			chunkMaskFlat := s.stepBuffers.chunkMask
			docIDs := buildBatchChunkDocIDs(batch, batchSize, maxChunksLocal)

			var contrastiveResult *ContrastiveLossResult
			if len(s.Config.MRLDimensions) > 0 {
				contrastiveResult = ComputeContrastiveLossOnCPUMRL(
					chunkEmbFlat, chunkMaskFlat, docIDs,
					s.Config.Temperature, lambdaEmbed,
					batchSize, maxChunksLocal, embeddingDim,
					s.Config.MRLDimensions,
					s.Config.FocalGammaContrastive, s.Config.FocalAlphaContrastive,
				)
			} else {
				contrastiveResult = ComputeContrastiveLossOnCPU(
					chunkEmbFlat, chunkMaskFlat, docIDs,
					s.Config.Temperature, lambdaEmbed,
					batchSize, maxChunksLocal, embeddingDim,
					s.Config.FocalGammaContrastive, s.Config.FocalAlphaContrastive,
				)
			}
			contrastiveLossVal = float32(contrastiveResult.ContrastiveLoss)
			contrastiveProbe = contrastiveProbeStatsFromSlices(
				chunkEmbFlat, contrastiveResult.DLdChunkEmb, chunkMaskFlat, docIDs,
				contrastiveResult.ContrastiveLoss, contrastiveResult.TotalLoss,
				batchSize*maxChunksLocal, embeddingDim,
			)

			var contrastiveBackInput *tensors.Tensor
			contrastiveBackInput, s.stepBuffers.contrastiveBwd = PackContrastiveBackwardInput(
				hidden, batch, contrastiveResult.DLdChunkEmb,
				batchSize, seqLen, hiddenSize, maxChunksLocal, embeddingDim,
				s.stepBuffers.contrastiveBwd,
			)
			contrastiveBackResults := s.contrastiveBackExec.MustExec(contrastiveBackInput)
			contrastiveBackInput.FinalizeAll()

			s.stepBuffers.gradA = readFloat32SliceInto(contrastiveBackResults[0], s.stepBuffers.gradA)
			upstreamProbe.Stages["contrastive_encoder_grad"] = tensorProbeStatsFromSlice(s.stepBuffers.gradA)
			contrastiveBackResults[0].FinalizeAll()
			hiddenGradTensor.MutableFlatData(func(flat any) {
				if data, ok := flat.([]float32); ok {
					for i, v := range s.stepBuffers.gradA {
						data[i] += v
					}
				}
			})
			for gi, name := range s.contrastiveBackGradNames {
				if gi+1 < len(contrastiveBackResults) {
					allGrads = append(allGrads, gradEntry{name: name, gradTensor: contrastiveBackResults[gi+1]})
				}
			}
		}
	}
	boundaryHeadTensor.FinalizeAll()
	upstreamProbe.Stages["combined_features_grad"] = tensorProbeStatsFromTensor(hiddenGradTensor)
	upstreamProbe.Stages["lora_output_grad"] = upstreamProbe.Stages["combined_features_grad"]

	totalLossVal := float64(boundaryLossVal)*s.Config.GetLossConfig(epochIdx).LambdaChunk +
		float64(contrastiveLossVal)*s.Config.GetLossConfig(epochIdx).LambdaEmbed
	metrics := []float32{float32(totalLossVal), boundaryLossVal, contrastiveLossVal, 0}

	var segmentProbe *segmentVJPProbe
	var decompProbe *layerBackwardDecompProbe
	savedHiddens[numSegments].FinalizeAll()
	savedHiddens[numSegments] = nil
	for seg := numSegments - 1; seg >= 0; seg-- {
		var upstreamStats parityStats
		captureSegmentProbe := seg == probeLayer
		if seg >= probeLayer {
			upstreamProbe.UpperEncoderLadder[fmt.Sprintf("after_layer_%d", seg)] = tensorProbeStatsFromTensor(hiddenGradTensor)
		}
		if captureSegmentProbe {
			upstreamStats = tensorStats(hiddenGradTensor)
			upstreamProbe.Stages["target_segment_upstream"] = tensorProbeStatsFromTensor(hiddenGradTensor)
		}
		hiddenGrad3D := s.reshapeHiddenGrad(hiddenGradTensor, batchSize, seqLen, hiddenSize)
		hiddenGradTensor.FinalizeAll()
		layerBackResults := s.layerBackwardExecs[seg].MustExec(savedHiddens[seg], hiddenGrad3D, attentionMask)
		if captureSegmentProbe {
			segmentProbe = captureSegmentVJPProbe(probeLayer, seg, upstreamStats, layerBackResults, s.layerBackGradNames[seg])
			if layerBackwardDecompEnabled {
				decompProbe = captureLayerBackwardDecompProbe(
					s,
					probeLayer,
					seg,
					savedHiddens[seg],
					hiddenGrad3D,
					attentionMask,
					segmentProbe,
				)
			}
		}
		hiddenGrad3D.FinalizeAll()
		savedHiddens[seg].FinalizeAll()
		savedHiddens[seg] = nil
		hiddenGradTensor = layerBackResults[0]
		if s.Config.HiddenGradNorm > 0 {
			oldGrad := hiddenGradTensor
			hiddenGradTensor = s.clipHiddenGrad(hiddenGradTensor, s.Config.HiddenGradNorm)
			if hiddenGradTensor != oldGrad {
				oldGrad.FinalizeAll()
			}
		}
		for gi, name := range s.layerBackGradNames[seg] {
			if gi+1 < len(layerBackResults) {
				allGrads = append(allGrads, gradEntry{name: name, gradTensor: layerBackResults[gi+1]})
			}
		}
	}
	attentionMask.FinalizeAll()
	if hiddenGradTensor != nil {
		hiddenGradTensor.FinalizeAll()
	}

	preClipNorm := globalNormFromGrads(allGrads)
	if s.Config.MaxGradNorm > 0 {
		s.clipGradientsByGlobalNorm(allGrads, s.Config.MaxGradNorm)
	}
	postClipNorm := globalNormFromGrads(allGrads)
	if layerBackwardDecompEnabled && decompProbe == nil {
		decompProbe = &layerBackwardDecompProbe{Status: "missing", Reason: "segment_vjp_probe_not_captured"}
	}
	return metrics, allGrads, preClipNorm, postClipNorm, boundaryProbe, forwardProbe, embeddingProbeStats, embeddingLookupRowProbe, encoderActivationInputProbe, encoderProjectionDecompositionProbe, encoderAttentionInternalProbe, encoderLayerProbe, encoderLayerStateProbe, encoderAttentionRowProbe, contrastiveProbe, upstreamProbe, segmentProbe, decompProbe, targetLayerInputValues
}

func TestFusedStepParityDump(t *testing.T) {
	outJSON := envString("FUSED_STEP_PARITY_OUT_JSON", "")
	if outJSON == "" {
		t.Fatal("FUSED_STEP_PARITY_OUT_JSON is required")
	}
	dataPath := envString("FUSED_STEP_PARITY_DATA", "")
	modelDir := envString("FUSED_STEP_PARITY_MODEL_DIR", "")
	checkpoint := envString("FUSED_STEP_PARITY_CHECKPOINT", "")
	optimizer := envString("FUSED_STEP_PARITY_OPTIMIZER", "")
	zigReplayInput := envString("FUSED_STEP_PARITY_ZIG_REPLAY_INPUT", "")
	zigReplayUpstream := envString("FUSED_STEP_PARITY_ZIG_REPLAY_UPSTREAM", "")
	if dataPath == "" || modelDir == "" || checkpoint == "" {
		t.Fatal("FUSED_STEP_PARITY_DATA, MODEL_DIR, and CHECKPOINT are required")
	}

	offset := envInt(t, "FUSED_STEP_PARITY_OFFSET", 1600)
	limit := envInt(t, "FUSED_STEP_PARITY_LIMIT", 8)
	batchSize := envInt(t, "FUSED_STEP_PARITY_BATCH_SIZE", 8)
	maxSeqLen := envInt(t, "FUSED_STEP_PARITY_MAX_SEQ_LEN", 384)
	maxChunks := envInt(t, "FUSED_STEP_PARITY_MAX_CHUNKS", 32)
	probeLayer := envInt(t, "FUSED_STEP_PARITY_DEBUG_ENCODER_PROBE_LAYER", 0)
	layerBackwardDecompEnabled := false
	switch strings.ToLower(envString("FUSED_STEP_PARITY_LAYER_BACKWARD_DECOMP", "0")) {
	case "1", "true", "yes":
		layerBackwardDecompEnabled = true
	}
	qkvSplitVJPEnabled := false
	switch strings.ToLower(envString("FUSED_STEP_PARITY_QKV_SPLIT_VJP", "0")) {
	case "1", "true", "yes":
		qkvSplitVJPEnabled = true
	}
		checkpointStep := envInt(t, "FUSED_STEP_PARITY_CHECKPOINT_STEP", 200)
		updateStep := envInt(t, "FUSED_STEP_PARITY_UPDATE_STEP", checkpointStep)
		split := envString("FUSED_STEP_PARITY_SPLIT", "train")
		backendName := envString("FUSED_STEP_PARITY_BACKEND", "mpsgraph")
		mixedPrecision := envString("FUSED_STEP_PARITY_GO_MIXED_PRECISION", "none")
		if mixedPrecision != "none" && mixedPrecision != "bf16" {
			t.Fatalf("FUSED_STEP_PARITY_GO_MIXED_PRECISION=%q unsupported; use none or bf16", mixedPrecision)
		}

	allSamples, err := LoadFusedSamples(dataPath, split)
	if err != nil {
		t.Fatalf("load samples: %v", err)
	}
	if offset < 0 || offset+limit > len(allSamples) {
		t.Fatalf("invalid offset/limit %d/%d for %d samples", offset, limit, len(allSamples))
	}
	stepsPerEpoch := (len(allSamples) + batchSize - 1) / batchSize
	samples := allSamples[offset : offset+limit]

	tokenizer, err := NewFusedHFTokenizer(filepath.Join(modelDir, "tokenizer.json"), 50368)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	ds := NewFusedDataset(samples, tokenizer, maxSeqLen, maxChunks, batchSize, false, 42)

	modelConfig := DefaultFusedChunkerEmbedderConfig()
	modelConfig.BoundaryDropout = 0
	modelConfig.EnableSplade = false
	modelConfig.BoundaryMLPDim = 256
	modelConfig.MixedPrecision = mixedPrecision

	loraConfig := lora.NewConfig()
	loraConfig.SetRank(16)
	loraConfig.SetAlpha(32)
	loraConfig.SetDropout(0)
	loraConfig.SetTargetModules("query_proj", "value_proj", "key_proj", "Wo")

	trainingConfig := DefaultFusedTrainingConfig()
	trainingConfig.BatchSize = batchSize
	trainingConfig.NumEpochs = 5
	trainingConfig.MaxSeqLen = maxSeqLen
	trainingConfig.MaxChunks = maxChunks
	trainingConfig.LearningRate = 2e-5
	trainingConfig.WarmupSteps = 200
	trainingConfig.WeightDecay = 0.01
	trainingConfig.MaxGradNorm = 1
	trainingConfig.PosWeight = 1
	trainingConfig.LambdaChunk = 1
	trainingConfig.LambdaEmbed = 0.3
	trainingConfig.BoundaryFocusEpochs = 3
	trainingConfig.MRLDimensions = []int{768, 256, 128}
	trainingConfig.LoRATargets = []string{"query_proj", "value_proj", "key_proj", "Wo"}
	trainingConfig.EnableSplade = false
	trainingConfig.Optimizer = "adamw"

		backend, err := backends.NewWithConfig(backendName)
		if err != nil {
			t.Fatalf("create %s backend: %v", backendName, err)
		}
		defer backend.Finalize()

	model := NewFusedChunkerEmbedder(modelConfig, loraConfig)
	trainer := NewFusedTrainer(model, trainingConfig, backend)
	if backend.Name() == "mpsgraph" {
		cpuBackend, err := xla.New("")
		if err != nil {
			t.Fatalf("create CPU init backend: %v", err)
		}
		defer cpuBackend.Finalize()
		if err := trainer.InitializeVariablesWithBackend(ds, cpuBackend); err != nil {
			t.Fatalf("initialize variables on CPU: %v", err)
		}
	} else {
		if err := trainer.InitializeVariablesWithBackend(ds, backend); err != nil {
			t.Fatalf("initialize variables: %v", err)
		}
	}
	loaded, err := trainer.LoadPretrainedWeights(modelDir)
	if err != nil {
		t.Fatalf("load pretrained weights: %v", err)
	}
	if loaded == 0 {
		t.Fatal("loaded zero pretrained tensors")
	}
	trainer.SanitizeWeights()
	trainer.InitDoRAMagnitudes()
	trainer.FreezeBaseWeights()
	trainer.InitScheduleAndOptimizer()
	if err := trainer.LoadLoRAWeights(checkpoint); err != nil {
		t.Fatalf("load converted checkpoint: %v", err)
	}
	if optimizer != "" {
		if err := trainer.LoadOptimizerState(optimizer); err != nil {
			t.Fatalf("load converted optimizer: %v", err)
		}
	}
	trainer.stepsPerEpoch = stepsPerEpoch
	trainer.totalSteps = stepsPerEpoch * trainingConfig.NumEpochs
	trainer.currentStep = checkpointStep
	trainer.stepVar.SetValue(tensors.FromScalar(float32(checkpointStep)))

	seg := NewSegmentedTrainer(trainer.Model, trainer.Config, trainer.Backend, trainer.Ctx)
	seg.stepVar = trainer.stepVar
	seg.adamStates = trainer.adamStates
	seg.sfStates = trainer.sfStates
	seg.sfGlobals = &trainer.sfGlobals
	seg.OutputDir = trainer.OutputDir
	seg.CheckpointOptimizerState = trainer.CheckpointOptimizerState
	seg.stepsPerEpoch = stepsPerEpoch
	seg.totalSteps = trainer.totalSteps
	seg.currentStep = checkpointStep
	seg.Distributed = trainer.Distributed
	seg.WeightStore = trainer.WeightStore
	seg.BuildExecs()

	ds.Reset()
	batch := ds.NextBatch()
	if batch == nil {
		t.Fatal("missing parity batch")
	}
	batchHashes := stepParityBatchHashesFor(batch, offset)
	batchSampleIndices := absoluteBatchSampleIndices(batch, offset)
	metrics, allGrads, preClipNorm, postClipNorm, boundaryProbe, forwardProbe, embeddingProbeStats, embeddingLookupRowProbe, encoderActivationInputProbe, encoderProjectionDecompositionProbe, encoderAttentionInternalProbe, encoderLayerProbe, encoderLayerStateProbe, encoderAttentionRowProbe, contrastiveProbe, upstreamGradProbe, segmentProbe, layerBackwardDecompProbe, targetLayerInputValues := runMPSGraphSegmentedParityStep(t, trainer, seg, batch, probeLayer, layerBackwardDecompEnabled)
	if len(metrics) < 4 {
		t.Fatalf("expected four metrics, got %d", len(metrics))
	}
	encoderLayerReplayProbe := captureEncoderLayerReplayProbe(t, seg, batch, zigReplayInput, targetLayerInputValues, probeLayer)
	encoderLayerBackwardReplayProbe := captureEncoderLayerBackwardReplayProbe(t, seg, batch, zigReplayInput, zigReplayUpstream, probeLayer, layerBackwardDecompEnabled)
	var softmaxProbe *softmaxVJPProbe
	if layerBackwardDecompEnabled {
		softmaxProbe = captureSoftmaxVJPProbe(seg, batchSize, maxSeqLen)
	}
	var qkvSplitProbe *qkvSplitVJPProbe
	if qkvSplitVJPEnabled {
		qkvSplitProbe = captureQKVSplitVJPProbe(seg, batchSize, maxSeqLen, probeLayer)
	}

	isLoRA := func(name string) bool {
		return strings.Contains(name, "lora_A") || strings.Contains(name, "lora_B")
	}
	boundaryName := func(suffix string) func(string) bool {
		return func(name string) bool {
			return strings.Contains(name, "boundary_head/"+suffix)
		}
	}
	isTask := func(name string) bool {
		return isTaskHead(name)
	}
	allGradNames := gradNames(allGrads)

	activeLoraMatrices := 0
	for _, entry := range allGrads {
		if isLoRA(entry.name) {
			activeLoraMatrices++
		}
	}
	loraGrad := aggregateGradStats(allGrads, isLoRA)
	taskGrad := aggregateGradStats(allGrads, isTask)
	taskGradShapeSample := sampleGradShapes(allGrads, isTask, 8)
	gradW1 := pickGradStats(allGrads, boundaryName("mlp_dense1/weight"))
	gradB1 := pickGradStats(allGrads, boundaryName("mlp_dense1/bias"))
	gradW2 := pickGradStats(allGrads, boundaryName("mlp_dense2/weight"))
	gradB2 := pickGradStats(allGrads, boundaryName("mlp_dense2/bias"))
	loraGradMatrix := loraGradMatrixStats(allGrads)
	gradClipScale := 1.0
	if preClipNorm > 0 {
		gradClipScale = postClipNorm / preClipNorm
	}
	preClipMatrixScale := 1.0
	if gradClipScale > 0 {
		preClipMatrixScale = 1.0 / gradClipScale
	}
	loraGradPreClip := scaleParityStats(loraGrad, preClipMatrixScale)
	taskGradPreClip := scaleParityStats(taskGrad, preClipMatrixScale)
	gradW1PreClip := scaleParityStats(gradW1, preClipMatrixScale)
	gradB1PreClip := scaleParityStats(gradB1, preClipMatrixScale)
	gradW2PreClip := scaleParityStats(gradW2, preClipMatrixScale)
	gradB2PreClip := scaleParityStats(gradB2, preClipMatrixScale)
	loraGradMatrixPreClip := scaleParityStatsMap(loraGradMatrix, preClipMatrixScale)
	checkpointProbe := boundaryCheckpointStats(trainer)
	if upstreamGradProbe != nil {
		upstreamGradProbe.Stages["final_norm_weight"] = checkpointProbe.FinalNormWeight
	}
	embeddingCheckpointProbe := embeddingCheckpointStats(trainer)
	embeddingProbeStats.WordEmbeddingWeight = embeddingCheckpointProbe.WordEmbeddingWeight
	embeddingProbeStats.LayerNormWeight = embeddingCheckpointProbe.LayerNormWeight
	embeddingProbeStats.LayerNormBias = embeddingCheckpointProbe.LayerNormBias
	embeddingTableRowProbe := embeddingTableRowProbeStats(trainer, batch, trainer.Model.Config.HiddenSize, embeddingRowProbeMax)

	loraBefore := readTrainable(trainer, isLoRA)
	trainer.currentStep = updateStep
	seg.currentStep = updateStep
	trainer.stepVar.SetValue(tensors.FromScalar(float32(updateStep)))
	if trainer.Config.Optimizer == "schedule-free" {
		seg.applyScheduleFreeAdamWCPU(allGrads)
	} else {
		seg.applyAdamWCPU(allGrads)
	}
	loraUpdate := aggregateDelta(loraBefore, trainer, isLoRA)
	loraUpdateMatrix := aggregateDeltaByLoRA(loraBefore, trainer)
	adamM := aggregateSegmentedAdam(seg, trainer.trainableVarNames, isLoRA, "m")
	adamV := aggregateSegmentedAdam(seg, trainer.trainableVarNames, isLoRA, "v")
	adamMMatrix := aggregateSegmentedAdamByLoRA(seg, trainer.trainableVarNames, "m")
	adamVMatrix := aggregateSegmentedAdamByLoRA(seg, trainer.trainableVarNames, "v")
	for _, entry := range allGrads {
		entry.gradTensor.FinalizeAll()
	}

	out := parityOutput{
		Tool:                "go_fused_step_parity",
		SchemaVersion:       1,
		Status:              "passed",
		Backend:             backend.Name(),
		MixedPrecision:      model.Config.MixedPrecision,
		UseBF16:             model.Config.UseBF16(),
		Checkpoint:          checkpoint,
		Optimizer:           optimizer,
		TargetProbeLayer:    probeLayer,
		CheckpointStepCount: checkpointStep,
		UpdateStep:          updateStep,
		DiagnosticMode:      "mpsgraph_segmented",
		StepsPerEpoch:       stepsPerEpoch,
		BatchSize:           batchSize,
		Offset:              offset,
		Limit:               limit,
		TrainableTensors:    len(trainer.trainableVarNames),
		NoUpdate: map[string]any{
				"hashes":           batchHashes,
				"batch_sample_indices": batchSampleIndices,
				"target_probe_layer": probeLayer,
				"total_loss":       float64(metrics[0]),
				"boundary_loss":    float64(metrics[1]),
				"contrastive_loss": float64(metrics[2]),
				"contrastive":      contrastiveProbe,
				"upstream_grad_probe": upstreamGradProbe,
				"segment_vjp_probe": segmentProbe,
				"layer_backward_decomp_probe": layerBackwardDecompProbe,
				"softmax_vjp_probe": softmaxProbe,
				"qkv_split_vjp_probe": qkvSplitProbe,
				"coherence_loss":   float64(metrics[3]),
				"grad_norm_pre_clip":  preClipNorm,
				"grad_norm_post_clip": postClipNorm,
				"grad_clip_scale":     gradClipScale,
				"lora_grad_norm_pre_clip": loraGradPreClip.L2,
				"grad_norm_lora":      loraGrad.L2,
				"grad_lora_pre_clip":  loraGradPreClip,
				"grad_lora":           loraGrad,
				"lora_grad_matrix_stats": loraGradMatrix,
				"lora_grad_matrix_stats_post_clip": loraGradMatrix,
				"lora_grad_matrix_stats_pre_clip": loraGradMatrixPreClip,
				"grad_task_head_pre_clip": taskGradPreClip,
				"grad_task_head":      taskGrad,
				"grad_tensors":         len(allGrads),
				"lora_grad_tensors":    activeLoraMatrices,
				"task_head_trainables": countStrings(trainer.trainableVarNames, isTask),
				"task_head_grad_tensors": countStrings(allGradNames, isTask),
				"task_head_grad_shape_sample": taskGradShapeSample,
				"boundary_back_grad_names": len(seg.boundaryBackGradNames),
				"boundary_back_grad_name_sample": sampleStrings(seg.boundaryBackGradNames, 12),
				"grad_name_sample": sampleStrings(allGradNames, 12),
				"boundary_probe":      boundaryProbe,
				"boundary_forward_probe": forwardProbe,
				"embedding_probe":     embeddingProbeStats,
				"embedding_table_row_probe": embeddingTableRowProbe,
				"embedding_lookup_row_probe": embeddingLookupRowProbe,
				"boundary_checkpoint_probe": checkpointProbe,
				"encoder_activation_input_probe": encoderActivationInputProbe,
				"encoder_projection_decomposition_probe": encoderProjectionDecompositionProbe,
				"encoder_attention_internal_probe": encoderAttentionInternalProbe,
				"encoder_attention_row_probe": encoderAttentionRowProbe,
				"encoder_layer_input_probe": encoderLayerProbe,
				"encoder_layer_state_probe": encoderLayerStateProbe,
				"encoder_layer_replay_probe": encoderLayerReplayProbe,
				"encoder_layer_backward_replay_probe": encoderLayerBackwardReplayProbe,
				"grad_w1_pre_clip":    gradW1PreClip,
				"grad_w1":             gradW1,
				"grad_b1_pre_clip":    gradB1PreClip,
				"grad_b1":             gradB1,
				"grad_w2_pre_clip":    gradW2PreClip,
				"grad_w2":             gradW2,
				"grad_b2_pre_clip":    gradB2PreClip,
				"grad_b2":             gradB2,
		},
		ApplyUpdate: map[string]any{
			"target_probe_layer":    probeLayer,
			"active_matrices":      activeLoraMatrices,
			"update_lora":          loraUpdate,
			"update_norm":          loraUpdate.L2,
			"update_max_abs":       loraUpdate.MaxAbs,
			"update_mean_abs":      loraUpdate.MeanAbs,
			"lora_update_matrix_stats": loraUpdateMatrix,
			"adam_m_lora":          adamM,
			"adam_v_lora":          adamV,
			"adam_m_norm":          adamM.L2,
			"adam_v_norm":          adamV.L2,
			"adam_m_lora_matrix_stats": adamMMatrix,
			"adam_v_lora_matrix_stats": adamVMatrix,
			"optimizer_step_count": updateStep,
		},
				Notes: []string{"Generated diagnostic mirrors the production Go MPSGraph segmented path for one frozen batch.", "LoRA optimizer moments are zero-initialized unless the converted optimizer includes LoRA state."},
			}

	bytes, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatalf("marshal output: %v", err)
	}
	if err := os.WriteFile(outJSON, append(bytes, '\n'), 0644); err != nil {
		t.Fatalf("write output: %v", err)
	}
}
GOEOF
}

run_go_step() {
  if [[ "$skip_go_step" == "1" || "$go_step_mode" != "full" ]]; then
    local skipped_mode="$go_step_mode"
    if [[ "$skip_go_step" == "1" ]]; then
      skipped_mode="forced_skip"
    fi
    python3 - "$go_step_json" "$skipped_mode" <<'PYEOF'
import json, sys
path, mode = sys.argv[1:]
with open(path, "w", encoding="utf-8") as f:
    json.dump({
        "tool": "go_fused_step_parity",
        "schema_version": 1,
        "status": "skipped",
        "mode": mode,
        "reason": "Production Go training uses the MPSGraph segmented path; full Go training-step parity is pending a segmented one-step diagnostic extractor.",
    }, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF
    return
  fi

  write_go_test
  printf 'package finetune\n' > "$go_empty_test_src"
  python3 - "$go_test_src" "$go_empty_test_src" "$gopeft_dir/e2e/finetune" "$go_overlay_json" <<'PYEOF'
import json, sys
from pathlib import Path
src, empty, package_dir, out = sys.argv[1:]
package_dir = Path(package_dir)
replace = {}
for path in package_dir.glob("*_test.go"):
    replace[str(path)] = empty
replace[str(package_dir / "fused_step_parity_test.go")] = src
with open(out, "w", encoding="utf-8") as f:
    json.dump({"Replace": replace}, f)
PYEOF

  local optimizer_env=""
  if [[ -f "$go_optimizer" ]]; then
    optimizer_env="$go_optimizer"
  fi
  local zig_replay_input_env=""
  local zig_replay_upstream_env=""
  case "$dump_encoder_replay_input" in
    1|true|TRUE|yes|YES)
      zig_replay_input_env="$zig_no_update_replay_input"
      ;;
  esac
  case "$dump_encoder_replay_upstream" in
    1|true|TRUE|yes|YES)
      zig_replay_upstream_env="$zig_apply_update_replay_upstream"
      ;;
  esac

  echo "running Go frozen step parity"
  set +e
  (
    cd "$gopeft_dir"
    FUSED_STEP_PARITY_OUT_JSON="$go_step_json" \
    FUSED_STEP_PARITY_DATA="$data_path" \
    FUSED_STEP_PARITY_MODEL_DIR="$model_dir" \
    FUSED_STEP_PARITY_CHECKPOINT="$go_checkpoint" \
    FUSED_STEP_PARITY_OPTIMIZER="$optimizer_env" \
    FUSED_STEP_PARITY_OFFSET="$offset" \
    FUSED_STEP_PARITY_LIMIT="$limit" \
    FUSED_STEP_PARITY_BATCH_SIZE="$batch_size" \
    FUSED_STEP_PARITY_MAX_SEQ_LEN="$max_seq_len" \
	    FUSED_STEP_PARITY_MAX_CHUNKS="$max_chunks" \
	    FUSED_STEP_PARITY_CHECKPOINT_STEP="$checkpoint_step_count" \
	    FUSED_STEP_PARITY_UPDATE_STEP="$go_update_step" \
	    FUSED_STEP_PARITY_SPLIT="$split" \
	    FUSED_STEP_PARITY_BACKEND="$go_step_backend" \
	    FUSED_STEP_PARITY_GO_MIXED_PRECISION="$go_mixed_precision" \
	    FUSED_STEP_PARITY_DEBUG_ENCODER_PROBE_LAYER="$debug_encoder_probe_layer" \
	    FUSED_STEP_PARITY_ZIG_REPLAY_INPUT="$zig_replay_input_env" \
	    FUSED_STEP_PARITY_ZIG_REPLAY_UPSTREAM="$zig_replay_upstream_env" \
	    FUSED_STEP_PARITY_LAYER_BACKWARD_DECOMP="$layer_backward_decomp" \
	    FUSED_STEP_PARITY_QKV_SPLIT_VJP="$qkv_split_vjp" \
	      go test -overlay "$go_overlay_json" ./e2e/finetune -run TestFusedStepParityDump -count=1 -timeout "$go_timeout"
  ) > "$go_stdout" 2> "$go_stderr"
  local status=$?
  set -e

  if [[ "$status" != "0" || ! -f "$go_step_json" ]]; then
    echo "Go overlay parity failed, retrying from a temporary gopeft copy" >&2
    local go_tmp="$out_dir/gopeft_tmp"
    rm -rf "$go_tmp"
    mkdir -p "$go_tmp"
    cp -R "$gopeft_dir" "$go_tmp/gopeft"
    if [[ -d "/Users/tim/Documents/af/go-coreml" && ! -e "$go_tmp/go-coreml" ]]; then
      ln -s /Users/tim/Documents/af/go-coreml "$go_tmp/go-coreml"
    fi
    find "$go_tmp/gopeft/e2e/finetune" -maxdepth 1 -name '*_test.go' -type f -delete
    cp "$go_test_src" "$go_tmp/gopeft/e2e/finetune/fused_step_parity_test.go"
    set +e
    (
      cd "$go_tmp/gopeft"
      FUSED_STEP_PARITY_OUT_JSON="$go_step_json" \
      FUSED_STEP_PARITY_DATA="$data_path" \
      FUSED_STEP_PARITY_MODEL_DIR="$model_dir" \
      FUSED_STEP_PARITY_CHECKPOINT="$go_checkpoint" \
      FUSED_STEP_PARITY_OPTIMIZER="$optimizer_env" \
      FUSED_STEP_PARITY_OFFSET="$offset" \
      FUSED_STEP_PARITY_LIMIT="$limit" \
      FUSED_STEP_PARITY_BATCH_SIZE="$batch_size" \
      FUSED_STEP_PARITY_MAX_SEQ_LEN="$max_seq_len" \
	      FUSED_STEP_PARITY_MAX_CHUNKS="$max_chunks" \
	      FUSED_STEP_PARITY_CHECKPOINT_STEP="$checkpoint_step_count" \
	      FUSED_STEP_PARITY_UPDATE_STEP="$go_update_step" \
	      FUSED_STEP_PARITY_SPLIT="$split" \
	      FUSED_STEP_PARITY_BACKEND="$go_step_backend" \
	      FUSED_STEP_PARITY_GO_MIXED_PRECISION="$go_mixed_precision" \
	      FUSED_STEP_PARITY_DEBUG_ENCODER_PROBE_LAYER="$debug_encoder_probe_layer" \
	      FUSED_STEP_PARITY_ZIG_REPLAY_INPUT="$zig_replay_input_env" \
	      FUSED_STEP_PARITY_ZIG_REPLAY_UPSTREAM="$zig_replay_upstream_env" \
	      FUSED_STEP_PARITY_LAYER_BACKWARD_DECOMP="$layer_backward_decomp" \
	      FUSED_STEP_PARITY_QKV_SPLIT_VJP="$qkv_split_vjp" \
	        go test ./e2e/finetune -run TestFusedStepParityDump -count=1 -timeout "$go_timeout"
    ) >> "$go_stdout" 2>> "$go_stderr"
    status=$?
    set -e
  fi

  if [[ "$status" != "0" || ! -f "$go_step_json" ]]; then
    echo "Go frozen step parity failed" >&2
    echo "tail of $go_stderr:" >&2
    tail -n 120 "$go_stderr" >&2 || true
    exit 1
  fi
}

run_go_step

python3 - "$batch_dir/zig_batch.json" "$batch_dir/go_batch.json" "$zig_no_update_json" "$zig_apply_update_json" "$go_step_json" "$adapter_json" "$summary_json" <<'PYEOF'
import json
import math
import os
import re
import sys
from pathlib import Path

zig_batch_path, go_batch_path, zig_no_update_path, zig_apply_update_path, go_step_path, adapter_path, summary_path = map(Path, sys.argv[1:])

def load(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

zig_batch = load(zig_batch_path)
go_batch = load(go_batch_path)
zig_no_update = load(zig_no_update_path)
zig_apply_update = load(zig_apply_update_path)
go_step = load(go_step_path)
adapter = load(adapter_path)
skip_batch_parity = os.environ.get("ANTFLY_FUSED_CHUNKER_STEP_PARITY_SKIP_BATCH", "0").lower() in ("1", "true", "yes")
zig_repeatability_path = summary_path.with_name("zig_repeatability_summary.json")
zig_repeatability = load(zig_repeatability_path)
zig_profile_baseline_path = summary_path.with_name("zig_profile_baseline_summary.json")
zig_profile_baseline = load(zig_profile_baseline_path)

summary = {
    "tool": "fused_chunker_go_zig_step_parity",
    "schema_version": 30,
    "status": "passed",
    "batch_parity": "not_run",
    "checkpoint_adapter": "not_run",
    "zig_step_parity": "not_run",
    "go_training_step_parity": "not_run",
    "failures": [],
	    "notes": [
	        "Go tokenization/batch parity is checked exactly for the frozen batch.",
	        "Zig emits forward/backward/update diagnostics from the resumed checkpoint.",
	        "Production Go fused-chunker training uses MPSGraph segmented execution.",
	        "The opt-in Go full-step diagnostic uses a generated in-package test that mirrors one MPSGraph segmented training step.",
	    ],
    "artifacts": {
        "zig_batch": str(zig_batch_path),
        "go_batch": str(go_batch_path),
        "zig_step_no_update": str(zig_no_update_path),
        "zig_step_apply_update": str(zig_apply_update_path),
        "zig_step_no_update_replay_input": str(Path(str(zig_no_update_path)).with_name("zig_step_no_update_encoder_replay_input.f32")),
        "zig_step_apply_update_replay_input": str(Path(str(zig_apply_update_path)).with_name("zig_step_apply_update_encoder_replay_input.f32")),
        "zig_step_apply_update_replay_upstream": str(Path(str(zig_apply_update_path)).with_name("zig_step_apply_update_encoder_replay_upstream.f32")),
        "go_step": str(go_step_path),
        "checkpoint_adapter": str(adapter_path),
        "zig_repeatability": str(zig_repeatability_path),
        "zig_profile_baseline": str(zig_profile_baseline_path),
    },
}

if isinstance(zig_repeatability, dict):
    summary["zig_repeatability"] = {
        "status": zig_repeatability.get("status"),
        "repeats": zig_repeatability.get("repeats"),
        "field_count": zig_repeatability.get("field_count"),
        "failure_count": zig_repeatability.get("failure_count"),
        "max_abs_diff": zig_repeatability.get("max_abs_diff"),
        "max_rel_diff": zig_repeatability.get("max_rel_diff"),
        "worst_field": zig_repeatability.get("worst_field"),
    }

if isinstance(zig_profile_baseline, dict):
    summary["zig_profile_baseline"] = {
        "status": zig_profile_baseline.get("status"),
        "baseline_path": zig_profile_baseline.get("baseline_path"),
        "observed_paths": zig_profile_baseline.get("observed_paths"),
        "observed_phase_count": zig_profile_baseline.get("observed_phase_count"),
        "phase_status": zig_profile_baseline.get("phase_status"),
        "failed_phases": zig_profile_baseline.get("failed_phases"),
        "field_count": zig_profile_baseline.get("field_count"),
        "comparison_count": zig_profile_baseline.get("comparison_count"),
        "failure_count": zig_profile_baseline.get("failure_count"),
        "max_abs_diff": zig_profile_baseline.get("max_abs_diff"),
        "max_rel_diff": zig_profile_baseline.get("max_rel_diff"),
        "worst_field": zig_profile_baseline.get("worst_field"),
        "baseline_profile": zig_profile_baseline.get("baseline_profile"),
        "observed_profile": zig_profile_baseline.get("observed_profile"),
        "observed_profiles": zig_profile_baseline.get("observed_profiles"),
        "layer_input_signal": zig_profile_baseline.get("layer_input_signal"),
    }

def fail(message):
    summary["status"] = "failed"
    summary["failures"].append(message)

hash_fields = ["sample_indices", "input_ids", "attention_mask", "labels", "chunks"]

def batch_hashes(doc):
    hashes = (doc or {}).get("hashes") if isinstance(doc, dict) else None
    return hashes if isinstance(hashes, dict) else None

def hash_mismatches(lhs, rhs):
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return list(hash_fields)
    return [field for field in hash_fields if lhs.get(field) != rhs.get(field)]

def f1_from_counts(tp, fp, fn):
    if not all(isinstance(v, (int, float)) for v in [tp, fp, fn]):
        return None
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return (2 * tp) / denom

def compare_probe_stat(zig_stat, go_stat):
    if not isinstance(zig_stat, dict) or not isinstance(go_stat, dict):
        return None
    out = {}
    for key in ["mean", "rms", "max_abs", "max_abs_value"]:
        zv = zig_stat.get(key)
        gv = go_stat.get(key)
        if isinstance(zv, (int, float)) and isinstance(gv, (int, float)):
            abs_diff = abs(float(zv) - float(gv))
            denom = max(abs(float(zv)), abs(float(gv)), 1e-12)
            out[key] = {
                "zig": zv,
                "go": gv,
                "abs_diff": abs_diff,
                "rel_diff": abs_diff / denom,
            }
    if isinstance(zig_stat.get("max_abs_index"), int) and isinstance(go_stat.get("max_abs_index"), int):
        out["max_abs_index"] = {
            "zig": zig_stat.get("max_abs_index"),
            "go": go_stat.get("max_abs_index"),
            "match": zig_stat.get("max_abs_index") == go_stat.get("max_abs_index"),
        }
    if isinstance(zig_stat.get("hash"), str) and isinstance(go_stat.get("hash"), str):
        out["hash"] = {
            "zig": zig_stat.get("hash"),
            "go": go_stat.get("hash"),
            "match": zig_stat.get("hash") == go_stat.get("hash"),
        }
    zs = zig_stat.get("sample") or []
    gs = go_stat.get("sample") or []
    sample_count = min(len(zs), len(gs))
    if sample_count:
        diffs = [abs(float(zs[i]) - float(gs[i])) for i in range(sample_count)]
        out["sample_count"] = sample_count
        out["sample_max_abs_diff"] = max(diffs)
        out["sample_mean_abs_diff"] = sum(diffs) / sample_count
    zi = zig_stat.get("top_abs_indices") or []
    gi = go_stat.get("top_abs_indices") or []
    zv = zig_stat.get("top_abs_values") or []
    gv = go_stat.get("top_abs_values") or []
    if isinstance(zi, list) and isinstance(gi, list) and zi and gi:
        overlap = len(set(zi) & set(gi))
        same_order = sum(1 for a, b in zip(zi, gi) if a == b)
        out["top_abs_index_overlap"] = {
            "zig_count": len(zi),
            "go_count": len(gi),
            "intersection_count": overlap,
            "same_order_count": same_order,
        }
        out["top_abs_first"] = {
            "zig_index": zi[0],
            "go_index": gi[0],
            "index_match": zi[0] == gi[0],
            "zig_value": zv[0] if isinstance(zv, list) and zv else None,
            "go_value": gv[0] if isinstance(gv, list) and gv else None,
        }
    out["zig_elems"] = zig_stat.get("elems")
    out["go_elems"] = go_stat.get("elems")
    return out

def compare_probe_maps(zig_map, go_map, stage_names):
    out = {}
    for stage in stage_names:
        cmp = compare_probe_stat((zig_map or {}).get(stage), (go_map or {}).get(stage))
        if cmp is not None:
            out[stage] = cmp
    return out

def upper_encoder_ladder_sort_key(stage):
    match = re.match(r"after_layer_(\d+)$", str(stage))
    if match:
        return -int(match.group(1))
    return 0

upstream_grad_stage_order = [
    "boundary_features_grad",
    "contrastive_features_grad",
    "contrastive_encoder_grad",
    "combined_features_grad",
    "final_norm_input",
    "final_norm_weight",
    "lora_output_grad",
    "target_segment_upstream",
]
upstream_grad_comparable_stage_order = [
    "lora_output_grad",
    "target_segment_upstream",
]
upstream_grad_diagnostic_stage_order = [
    "boundary_features_grad",
    "contrastive_features_grad",
    "contrastive_encoder_grad",
    "combined_features_grad",
    "final_norm_input",
    "final_norm_weight",
]
upstream_grad_stage_notes = {
    "boundary_features_grad": "diagnostic_only: Go captures encoder-space boundary grad; Zig may expose post-final-norm feature grad.",
    "contrastive_features_grad": "diagnostic_only: Zig captures post-final-norm contrastive feature grad before final-norm VJP.",
    "contrastive_encoder_grad": "diagnostic_only: Go captures contrastive grad after its final-norm VJP, in encoder space.",
    "combined_features_grad": "diagnostic_only: Go and Zig names are retained for raw inspection but are not the primary comparable stage.",
    "final_norm_input": "diagnostic_only: activation snapshot, not a gradient parity stage.",
    "final_norm_weight": "diagnostic_only: checkpoint snapshot, not a gradient parity stage.",
    "lora_output_grad": "comparable: gradient entering the upper encoder/LoRA stack.",
    "target_segment_upstream": "comparable: gradient entering the probed encoder segment.",
}

def compare_upstream_grad_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {"metadata": {}, "stages": {}, "diagnostic_only_stages": {}}
    for key in ["status", "target_layer"]:
        out["metadata"][key] = {
            "zig": zig_probe.get(key),
            "go": go_probe.get(key),
            "match": zig_probe.get(key) == go_probe.get(key),
        }
    zig_stages = zig_probe.get("stages") or {}
    go_stages = go_probe.get("stages") or {}
    raw_stage_names = sorted(set(zig_stages.keys()) | set(go_stages.keys()))
    comparable_stage_names = list(upstream_grad_comparable_stage_order)
    diagnostic_stage_names = list(upstream_grad_diagnostic_stage_order)
    for stage in raw_stage_names:
        if stage in upstream_grad_comparable_stage_order:
            continue
        if stage not in diagnostic_stage_names:
            diagnostic_stage_names.append(stage)
    out["stages"] = compare_probe_maps(zig_stages, go_stages, comparable_stage_names)
    out["diagnostic_only_stages"] = compare_probe_maps(zig_stages, go_stages, diagnostic_stage_names)
    out["stage_semantics"] = {
        stage: upstream_grad_stage_notes.get(stage, "diagnostic_only: no parity semantics registered for this stage name.")
        for stage in sorted(set(comparable_stage_names) | set(diagnostic_stage_names))
        if stage in raw_stage_names or stage in upstream_grad_stage_notes
    }
    out["raw_stage_names"] = {
        "zig": sorted(zig_stages.keys()),
        "go": sorted(go_stages.keys()),
        "comparable": comparable_stage_names,
        "diagnostic_only": diagnostic_stage_names,
    }
    zig_ladder = zig_probe.get("upper_encoder_ladder") or {}
    go_ladder = go_probe.get("upper_encoder_ladder") or {}
    ladder_names = sorted(set(zig_ladder.keys()) | set(go_ladder.keys()), key=upper_encoder_ladder_sort_key)
    out["upper_encoder_ladder"] = compare_probe_maps(zig_ladder, go_ladder, ladder_names)
    return out

def upstream_grad_stage_diverged(cmp):
    if not isinstance(cmp, dict):
        return False
    for key in ["rms", "max_abs", "mean"]:
        rel = ((cmp.get(key) or {}).get("rel_diff") or 0.0)
        if rel > 1e-2:
            return True
    if (cmp.get("sample_max_abs_diff") or 0.0) > 1e-5:
        return True
    return False

def upstream_grad_assembly_signal(upstream_grad_comparisons):
    signal = {
        "classification": "missing",
        "first_divergent_stage": None,
        "worst_stage": None,
        "max_rms_rel_diff": 0.0,
        "max_rms_abs_diff": 0.0,
        "max_sample_abs_diff": 0.0,
        "initial_encoder_upstream_rms_rel_diff": None,
        "target_segment_upstream_rms_rel_diff": None,
    }
    if not isinstance(upstream_grad_comparisons, dict):
        return signal
    stages = upstream_grad_comparisons.get("stages") or {}
    for stage in upstream_grad_comparable_stage_order:
        cmp = stages.get(stage) or {}
        rms = cmp.get("rms") or {}
        rel = rms.get("rel_diff") or 0.0
        if rel > signal["max_rms_rel_diff"]:
            signal["worst_stage"] = stage
            signal["max_rms_rel_diff"] = rel
            signal["max_rms_abs_diff"] = rms.get("abs_diff") or 0.0
        signal["max_sample_abs_diff"] = max(signal["max_sample_abs_diff"], cmp.get("sample_max_abs_diff") or 0.0)
    signal["initial_encoder_upstream_rms_rel_diff"] = ((stages.get("lora_output_grad") or {}).get("rms") or {}).get("rel_diff")
    signal["target_segment_upstream_rms_rel_diff"] = ((stages.get("target_segment_upstream") or {}).get("rms") or {}).get("rel_diff")
    for stage in upstream_grad_comparable_stage_order:
        if upstream_grad_stage_diverged(stages.get(stage)):
            signal["first_divergent_stage"] = stage
            break
    initial_rel = signal["initial_encoder_upstream_rms_rel_diff"] or 0.0
    target_rel = signal["target_segment_upstream_rms_rel_diff"] or 0.0
    if initial_rel <= 1e-4 and target_rel <= 1e-4 and signal["max_sample_abs_diff"] <= 1e-5:
        signal["classification"] = "upstream_grad_clean"
    elif initial_rel <= 1e-2 and target_rel > 1e-2:
        signal["classification"] = "upper_encoder_backprop_drift"
    elif initial_rel > 1e-2:
        signal["classification"] = "initial_encoder_upstream_divergent"
    elif signal["max_rms_rel_diff"] <= 1e-2:
        signal["classification"] = "upstream_grad_small_drift"
    else:
        signal["classification"] = "upstream_grad_divergent"
    return signal

def parse_upper_ladder_layer(stage):
    match = re.match(r"after_layer_(\d+)$", str(stage))
    if not match:
        return None
    return int(match.group(1))

def upper_encoder_ladder_signal(upstream_grad_comparisons):
    signal = {
        "classification": "missing",
        "stage_count": 0,
        "first_large_drift_stage": None,
        "first_large_drift_transition": None,
        "first_large_drift_layer": None,
        "worst_stage": None,
        "max_rms_rel_diff": 0.0,
        "max_rms_abs_diff": 0.0,
        "rms_rel_by_stage": {},
    }
    if not isinstance(upstream_grad_comparisons, dict):
        return signal
    ladder = upstream_grad_comparisons.get("upper_encoder_ladder") or {}
    stage_names = sorted(ladder.keys(), key=upper_encoder_ladder_sort_key)
    if not stage_names:
        return signal
    signal["stage_count"] = len(stage_names)
    previous_stage = "lora_output_grad"
    previous_rel = (((upstream_grad_comparisons.get("stages") or {}).get("lora_output_grad") or {}).get("rms") or {}).get("rel_diff") or 0.0
    for stage in stage_names:
        cmp = ladder.get(stage) or {}
        rms = cmp.get("rms") or {}
        rel = rms.get("rel_diff") or 0.0
        signal["rms_rel_by_stage"][stage] = rel
        if rel > signal["max_rms_rel_diff"]:
            signal["worst_stage"] = stage
            signal["max_rms_rel_diff"] = rel
            signal["max_rms_abs_diff"] = rms.get("abs_diff") or 0.0
        if rel > 1e-2 and signal["first_large_drift_stage"] is None:
            signal["first_large_drift_stage"] = stage
            signal["first_large_drift_transition"] = f"{previous_stage}->{stage}"
            if previous_stage.startswith("after_layer_"):
                signal["first_large_drift_layer"] = parse_upper_ladder_layer(previous_stage)
            else:
                signal["first_large_drift_layer"] = parse_upper_ladder_layer(stage)
        previous_stage = stage
        previous_rel = rel
    _ = previous_rel
    if signal["max_rms_rel_diff"] <= 1e-4:
        signal["classification"] = "upper_encoder_ladder_clean"
    elif signal["max_rms_rel_diff"] <= 1e-2:
        signal["classification"] = "upper_encoder_ladder_small_drift"
    else:
        signal["classification"] = "upper_encoder_ladder_divergent"
    return signal

def compare_numeric_value(zig_value, go_value):
    if not isinstance(zig_value, (int, float)) or not isinstance(go_value, (int, float)):
        return None
    abs_diff = abs(float(zig_value) - float(go_value))
    denom = max(abs(float(zig_value)), abs(float(go_value)), 1e-12)
    return {
        "zig": zig_value,
        "go": go_value,
        "abs_diff": abs_diff,
        "rel_diff": abs_diff / denom,
    }

def compare_lora_stat(zig_stat, go_stat):
    if not isinstance(zig_stat, dict) or not isinstance(go_stat, dict):
        return None
    out = {}
    for key in ["l2", "max_abs", "mean_abs"]:
        zv = zig_stat.get(key)
        gv = go_stat.get(key)
        if isinstance(zv, (int, float)) and isinstance(gv, (int, float)):
            abs_diff = abs(float(zv) - float(gv))
            denom = max(abs(float(zv)), abs(float(gv)), 1e-12)
            out[key] = {
                "zig": zv,
                "go": gv,
                "abs_diff": abs_diff,
                "rel_diff": abs_diff / denom,
            }
    sample_cmp = compare_number_sample(zig_stat.get("sample"), go_stat.get("sample"))
    if sample_cmp is not None:
        out["sample"] = sample_cmp
    out["zig_elems"] = zig_stat.get("elems")
    out["go_elems"] = go_stat.get("elems")
    return out

def compare_lora_stat_maps(zig_map, go_map):
    if not isinstance(zig_map, dict):
        zig_map = {}
    if not isinstance(go_map, dict):
        go_map = {}
    out = {}
    for key in sorted(set(zig_map.keys()) | set(go_map.keys())):
        cmp = compare_lora_stat(zig_map.get(key), go_map.get(key))
        if cmp is not None:
            out[key] = cmp
    return out

def lora_map_name_set_comparison(zig_map, go_map):
    if not isinstance(zig_map, dict):
        zig_map = {}
    if not isinstance(go_map, dict):
        go_map = {}
    zig_keys = set(zig_map.keys())
    go_keys = set(go_map.keys())
    common = sorted(zig_keys & go_keys)
    return {
        "zig": sorted(zig_keys),
        "go": sorted(go_keys),
        "match": zig_keys == go_keys,
        "common_adapter_count": len(common),
        "missing_in_zig": sorted(go_keys - zig_keys),
        "missing_in_go": sorted(zig_keys - go_keys),
    }

def compare_segment_vjp_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {"metadata": {}, "stats": {}}
    for key in ["target_layer", "segment_start", "segment_end", "include_hidden_grad", "include_adapter_grads", "runtime"]:
        out["metadata"][key] = {
            "zig": zig_probe.get(key),
            "go": go_probe.get(key),
            "match": zig_probe.get(key) == go_probe.get(key),
        }
    for key in ["upstream", "hidden_grad", "adapter_a", "adapter_b"]:
        cmp = compare_lora_stat(zig_probe.get(key), go_probe.get(key))
        if cmp is not None:
            out["stats"][key] = cmp
    out["adapter_a_by_name"] = compare_lora_stat_maps(
        zig_probe.get("adapter_a_by_name") or {},
        go_probe.get("adapter_a_by_name") or {},
    )
    out["adapter_b_by_name"] = compare_lora_stat_maps(
        zig_probe.get("adapter_b_by_name") or {},
        go_probe.get("adapter_b_by_name") or {},
    )
    return out

backward_decomp_stage_order = [
    "incoming_upstream",
    "full_layer_hidden_grad",
    "mlp_wo",
    "mlp_gelu_input",
    "mlp_gate_value",
    "mlp_gate_input",
    "mlp_wi_output",
    "mlp_norm_output",
    "mlp_hidden_after_attn",
    "attn_out_proj",
    "attention_core",
    "attention_core_post_rope",
    "attention_scores_raw",
    "attention_scores_masked",
    "attention_probs",
    "qkv_proj",
    "qkv_proj_split",
    "attn_norm_hidden_in",
]

def compare_backward_decomp_stage(zig_stage, go_stage):
    if not isinstance(zig_stage, dict) or not isinstance(go_stage, dict):
        return None
    out = {
        "status": {
            "zig": zig_stage.get("status"),
            "go": go_stage.get("status"),
            "match": zig_stage.get("status") == go_stage.get("status"),
        },
        "reason": {
            "zig": zig_stage.get("reason"),
            "go": go_stage.get("reason"),
            "match": zig_stage.get("reason") == go_stage.get("reason"),
        },
    }
    stats_cmp = compare_lora_stat(zig_stage.get("stats"), go_stage.get("stats"))
    if stats_cmp is not None:
        out["stats"] = stats_cmp
    out["components"] = compare_lora_stat_maps(
        zig_stage.get("components") or {},
        go_stage.get("components") or {},
    )
    zig_adapter_a_by_name = zig_stage.get("adapter_a_by_name") or {}
    go_adapter_a_by_name = go_stage.get("adapter_a_by_name") or {}
    zig_adapter_b_by_name = zig_stage.get("adapter_b_by_name") or {}
    go_adapter_b_by_name = go_stage.get("adapter_b_by_name") or {}
    adapter_a_names = lora_map_name_set_comparison(zig_adapter_a_by_name, go_adapter_a_by_name)
    adapter_b_names = lora_map_name_set_comparison(zig_adapter_b_by_name, go_adapter_b_by_name)
    out["adapter_a_names"] = adapter_a_names
    out["adapter_b_names"] = adapter_b_names
    if adapter_a_names["match"]:
        adapter_a_cmp = compare_lora_stat(zig_stage.get("adapter_a"), go_stage.get("adapter_a"))
        if adapter_a_cmp is not None:
            out["adapter_a"] = adapter_a_cmp
    else:
        out["adapter_a"] = {"skipped": "adapter_name_set_mismatch"}
    if adapter_b_names["match"]:
        adapter_b_cmp = compare_lora_stat(zig_stage.get("adapter_b"), go_stage.get("adapter_b"))
        if adapter_b_cmp is not None:
            out["adapter_b"] = adapter_b_cmp
    else:
        out["adapter_b"] = {"skipped": "adapter_name_set_mismatch"}
    out["adapter_a_by_name"] = compare_lora_stat_maps(
        zig_adapter_a_by_name,
        go_adapter_a_by_name,
    )
    out["adapter_b_by_name"] = compare_lora_stat_maps(
        zig_adapter_b_by_name,
        go_adapter_b_by_name,
    )
    return out

def compare_layer_backward_decomp_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {"metadata": {}, "stages": {}}
    for key in ["status", "version", "target_layer", "segment_start", "segment_end", "runtime"]:
        out["metadata"][key] = {
            "zig": zig_probe.get(key),
            "go": go_probe.get(key),
            "match": zig_probe.get(key) == go_probe.get(key),
        }
    zig_stages = zig_probe.get("stages") or {}
    go_stages = go_probe.get("stages") or {}
    stage_names = list(backward_decomp_stage_order)
    for stage in sorted(set(zig_stages.keys()) | set(go_stages.keys())):
        if stage not in stage_names:
            stage_names.append(stage)
    for stage in stage_names:
        cmp = compare_backward_decomp_stage(zig_stages.get(stage), go_stages.get(stage))
        if cmp is not None:
            out["stages"][stage] = cmp
    return out

def backward_component_signal(layer_backward_decomp_comparisons):
    signal = {
        "component_count": 0,
        "worst_stage": None,
        "worst_component": None,
        "max_l2_rel_diff": 0.0,
        "max_l2_abs_diff": 0.0,
        "max_sample_abs_diff": 0.0,
    }
    for stage, cmp in ((layer_backward_decomp_comparisons or {}).get("stages") or {}).items():
        for component, item in (cmp.get("components") or {}).items():
            signal["component_count"] += 1
            l2 = item.get("l2") or {}
            sample = item.get("sample") or {}
            rel = l2.get("rel_diff") or 0.0
            if rel > signal["max_l2_rel_diff"]:
                signal["worst_stage"] = stage
                signal["worst_component"] = component
                signal["max_l2_rel_diff"] = rel
                signal["max_l2_abs_diff"] = l2.get("abs_diff") or 0.0
                signal["max_sample_abs_diff"] = sample.get("max_abs_diff") or 0.0
    return signal

def same_upstream_backward_signal(replay_probe, segment_comparisons, layer_backward_decomp_comparisons):
    signal = {
        "status": (replay_probe or {}).get("status") if isinstance(replay_probe, dict) else None,
        "classification": "missing",
        "target_layer": (replay_probe or {}).get("target_layer") if isinstance(replay_probe, dict) else None,
        "hidden_grad_considered": False,
        "classification_note": None,
        "segment_hidden_grad_l2_abs_diff": None,
        "segment_hidden_grad_l2_rel_diff": None,
        "segment_adapter_a_l2_rel_diff": None,
        "segment_adapter_b_l2_rel_diff": None,
        "max_component_l2_rel_diff": None,
        "worst_component": None,
        "worst_stage": None,
    }
    if not isinstance(replay_probe, dict) or replay_probe.get("status") != "captured":
        return signal
    stats = (segment_comparisons or {}).get("stats") or {}
    hidden_l2 = ((stats.get("hidden_grad") or {}).get("l2") or {})
    adapter_a_l2 = ((stats.get("adapter_a") or {}).get("l2") or {})
    adapter_b_l2 = ((stats.get("adapter_b") or {}).get("l2") or {})
    signal["segment_hidden_grad_l2_abs_diff"] = hidden_l2.get("abs_diff")
    signal["segment_hidden_grad_l2_rel_diff"] = hidden_l2.get("rel_diff")
    signal["segment_adapter_a_l2_rel_diff"] = adapter_a_l2.get("rel_diff")
    signal["segment_adapter_b_l2_rel_diff"] = adapter_b_l2.get("rel_diff")
    metadata = (segment_comparisons or {}).get("metadata") or {}
    include_hidden = (metadata.get("include_hidden_grad") or {})
    if signal["target_layer"] is None:
        target_layer_meta = metadata.get("target_layer") or {}
        signal["target_layer"] = target_layer_meta.get("go") if target_layer_meta.get("match", False) else target_layer_meta.get("zig")
    hidden_grad_requested = bool(include_hidden.get("zig")) or bool(include_hidden.get("go"))
    signal["hidden_grad_considered"] = hidden_grad_requested
    if not hidden_grad_requested and signal["segment_hidden_grad_l2_rel_diff"] is not None:
        signal["classification_note"] = "ignored_hidden_grad_stats_because_replay_metadata_disables_hidden_grad"
    component_signal = backward_component_signal(layer_backward_decomp_comparisons)
    signal["max_component_l2_rel_diff"] = component_signal.get("max_l2_rel_diff")
    signal["worst_component"] = component_signal.get("worst_component")
    signal["worst_stage"] = component_signal.get("worst_stage")
    worst_rel = max(
        (signal["segment_hidden_grad_l2_rel_diff"] or 0.0) if hidden_grad_requested else 0.0,
        signal["segment_adapter_a_l2_rel_diff"] or 0.0,
        signal["segment_adapter_b_l2_rel_diff"] or 0.0,
        signal["max_component_l2_rel_diff"] or 0.0,
    )
    if worst_rel <= 1e-4:
        signal["classification"] = "same_upstream_vjp_clean"
    elif worst_rel <= 1e-2:
        signal["classification"] = "same_upstream_vjp_small_drift"
    else:
        signal["classification"] = "same_upstream_vjp_divergent"
    return signal

def decomp_stage_stat_l2_rel(layer_backward_decomp_comparisons, stage, stat_name="stats"):
    l2 = (((((layer_backward_decomp_comparisons or {}).get("stages") or {}).get(stage) or {}).get(stat_name) or {}).get("l2") or {})
    return l2.get("rel_diff")

def actual_vs_same_upstream_decomp_signal(actual_comparisons, same_upstream_comparisons, segment_comparisons=None):
    actual_component_signal = backward_component_signal(actual_comparisons)
    same_component_signal = backward_component_signal(same_upstream_comparisons)
    actual_hidden_rel = decomp_stage_stat_l2_rel(actual_comparisons, "full_layer_hidden_grad")
    same_hidden_rel = decomp_stage_stat_l2_rel(same_upstream_comparisons, "full_layer_hidden_grad")
    metadata = (segment_comparisons or {}).get("metadata") or {}
    include_hidden = (metadata.get("include_hidden_grad") or {})
    hidden_grad_considered = True
    classification_note = None
    if include_hidden:
        hidden_grad_considered = bool(include_hidden.get("zig")) or bool(include_hidden.get("go"))
    if not hidden_grad_considered:
        classification_note = "ignored_hidden_grad_stats_because_replay_metadata_disables_hidden_grad"
    actual_worst_rel = max(
        (actual_hidden_rel or 0.0) if hidden_grad_considered else 0.0,
        actual_component_signal.get("max_l2_rel_diff") or 0.0,
    )
    same_worst_rel = max(
        (same_hidden_rel or 0.0) if hidden_grad_considered else 0.0,
        same_component_signal.get("max_l2_rel_diff") or 0.0,
    )
    signal = {
        "classification": "missing",
        "hidden_grad_considered": hidden_grad_considered,
        "classification_note": classification_note,
        "actual_full_layer_hidden_grad_l2_rel_diff": actual_hidden_rel,
        "same_upstream_full_layer_hidden_grad_l2_rel_diff": same_hidden_rel,
        "actual_max_component_l2_rel_diff": actual_component_signal.get("max_l2_rel_diff"),
        "same_upstream_max_component_l2_rel_diff": same_component_signal.get("max_l2_rel_diff"),
        "actual_worst_stage": actual_component_signal.get("worst_stage"),
        "actual_worst_component": actual_component_signal.get("worst_component"),
        "same_upstream_worst_stage": same_component_signal.get("worst_stage"),
        "same_upstream_worst_component": same_component_signal.get("worst_component"),
        "actual_worst_l2_rel_diff": actual_worst_rel,
        "same_upstream_worst_l2_rel_diff": same_worst_rel,
        "actual_to_same_worst_l2_rel_ratio": None,
    }
    if same_worst_rel > 0:
        signal["actual_to_same_worst_l2_rel_ratio"] = actual_worst_rel / same_worst_rel
    if not isinstance(actual_comparisons, dict) or not isinstance(same_upstream_comparisons, dict):
        return signal
    if same_worst_rel > 1e-2:
        signal["classification"] = "same_upstream_vjp_divergent"
    elif actual_worst_rel > 1e-2 and same_worst_rel <= 1e-4:
        signal["classification"] = "actual_upstream_sensitivity"
    elif actual_worst_rel > 1e-2:
        signal["classification"] = "actual_path_drift"
    elif actual_worst_rel > 1e-4:
        signal["classification"] = "actual_path_small_drift"
    else:
        signal["classification"] = "clean"
    return signal

def training_parity_next_target_signal(summary):
    optimizer = summary.get("optimizer_update_signal") or {}
    upstream = summary.get("upstream_grad_assembly_signal") or {}
    ladder = summary.get("upper_encoder_ladder_signal") or {}
    same_upstream = summary.get("same_upstream_backward_signal") or {}
    actual_same = summary.get("actual_vs_same_upstream_decomp_signal") or {}
    projection = summary.get("projection_decomposition_signal") or {}
    layer_state = summary.get("layer_state_signal") or {}
    layer_internal = summary.get("layer_internal_jump_signal") or {}
    layer_input = summary.get("layer_input_jump_signal") or {}
    layer_input_direct = summary.get("layer_input_direct_diff_signal") or {}
    layer_replay = summary.get("layer_replay_signal") or {}
    forward_amp = summary.get("forward_amplification_signal") or {}
    attention_context = summary.get("attention_context_replay_signal") or {}
    attention_reference_outlier = summary.get("attention_reference_outlier_signal") or {}
    attention = summary.get("attention_row_signal") or {}
    qkv = summary.get("qkv_split_vjp_signal") or {}
    first_layer = ladder.get("first_large_drift_layer")
    target_layer = summary.get("target_probe_layer")
    signal = {
        "classification": "missing",
        "target_probe_layer": target_layer,
        "first_large_drift_layer": first_layer,
        "first_large_drift_transition": ladder.get("first_large_drift_transition"),
        "optimizer_classification": optimizer.get("classification"),
        "upstream_classification": upstream.get("classification"),
        "same_upstream_classification": same_upstream.get("classification"),
        "actual_vs_same_classification": actual_same.get("classification"),
        "first_forward_divergent_stage": summary.get("first_divergent_stage"),
        "forward_projection_metadata_mismatches": projection.get("metadata_mismatches"),
        "layer_state_worst_rms_stage": layer_state.get("worst_rms_stage"),
        "layer_state_max_rms_rel_diff": layer_state.get("max_rms_rel_diff"),
        "layer_internal_first_abs_over_1e_2": layer_internal.get("first_rms_abs_over_1e_2_stage"),
        "first_forward_input_jump_layer": layer_input.get("first_rms_abs_over_1e_2_layer"),
        "largest_forward_input_jump_layer": layer_input.get("largest_rms_abs_layer"),
        "target_layer_input_direct_diff_classification": layer_input_direct.get("classification"),
        "target_layer_input_direct_diff_rms": layer_input_direct.get("rms"),
        "target_layer_input_direct_diff_max_abs": layer_input_direct.get("max_abs"),
        "target_layer_input_valid_diff_rms": layer_input_direct.get("valid_rms"),
        "target_layer_input_valid_diff_max_abs": layer_input_direct.get("valid_max_abs"),
        "target_layer_input_padding_diff_rms": layer_input_direct.get("padding_rms"),
        "target_layer_input_padding_diff_max_abs": layer_input_direct.get("padding_max_abs"),
        "layer_replay_classification": layer_replay.get("classification"),
        "layer_replay_layer_output_rms_abs_diff": layer_replay.get("layer_output_rms_abs_diff"),
        "forward_amplification_classification": forward_amp.get("classification"),
        "forward_amplification_next_target": forward_amp.get("next_target"),
        "first_material_forward_input_drift_layer": forward_amp.get("first_input_rms_abs_over_5e_4_layer"),
        "target_layer_input_to_next_layer_amplification": forward_amp.get("target_to_next_input_amplification"),
        "attention_context_classification": attention_context.get("classification"),
        "attention_context_rms_abs_diff": attention_context.get("attn_context_rms_abs_diff"),
        "attention_context_max_qkv_probs_rms_abs_diff": attention_context.get("max_qkv_probs_rms_abs_diff"),
        "attention_zig_best_ref_delta_rms": attention_context.get("zig_best_ref_delta_rms"),
        "attention_go_best_ref_delta_rms": attention_context.get("go_best_ref_delta_rms"),
        "attention_reference_outlier_classification": attention_reference_outlier.get("classification"),
        "attention_argmax_mismatches": attention.get("argmax_mismatches"),
        "qkv_worst_case": qkv.get("worst_case"),
        "qkv_worst_component": qkv.get("worst_component"),
        "next_target": None,
    }
    if layer_input_direct.get("classification") == "layer_input_direct_diff_material_drift":
        signal["classification"] = "forward_layer_input_material_drift"
        signal["next_target"] = f"localize the producer of layer_{int(target_layer or 0):02d} input drift before rerunning readiness"
    elif (
        layer_input_direct.get("classification") == "layer_input_direct_diff_padding_only_drift"
        and optimizer.get("classification") in ["optimizer_update_small_drift", "optimizer_update_clean"]
        and upstream.get("classification") in ["upstream_grad_small_drift", "upstream_grad_clean", None]
        and not first_layer
    ):
        signal["classification"] = "training_parity_padding_only_drift"
        signal["next_target"] = "run_short_readiness_probe_with_valid-token quality checks"
    elif optimizer.get("classification") == "optimizer_update_clean":
        signal["classification"] = "optimizer_clean"
        signal["next_target"] = "run_short_readiness_probe"
    elif (
        optimizer.get("classification") == "optimizer_update_small_drift" and
        upstream.get("classification") in ["upstream_grad_small_drift", "upstream_grad_clean", None] and
        attention_context.get("classification") in ["attention_context_clean", None] and
        not first_layer
    ):
        signal["classification"] = "training_parity_small_drift"
        signal["next_target"] = "run_short_readiness_probe"
    elif (
        optimizer.get("classification") == "lora_pre_clip_gradient_drift"
        and same_upstream.get("classification") in ["same_upstream_vjp_clean", "same_upstream_vjp_small_drift"]
        and actual_same.get("classification") == "actual_upstream_sensitivity"
    ):
        if attention_context.get("classification") == "attention_context_accumulation_drift":
            signal["classification"] = "attention_context_drift_drives_lora_gradient_drift"
            signal["next_target"] = attention_context.get("next_target")
        elif (
            attention_context.get("classification") == "attention_reference_contract_mismatch"
            and attention_reference_outlier.get("classification") != "attention_reference_outliers_on_padding"
        ):
            signal["classification"] = "attention_reference_contract_mismatch_drives_lora_gradient_drift"
            signal["next_target"] = attention_context.get("next_target")
        elif forward_amp.get("classification") in [
            "geglu_product_or_tail_input_mismatch",
            "wo_projection_amplifies_subthreshold_input_drift",
            "wo_projection_mismatch",
            "residual_add_or_output_cast_mismatch",
        ]:
            signal["classification"] = "local_forward_replay_mismatch_drives_lora_gradient_drift"
            signal["next_target"] = forward_amp.get("next_target")
        elif forward_amp.get("classification") in [
            "target_layer_amplifies_accumulated_input_drift",
            "target_layer_materializes_accumulated_input_drift",
            "target_layer_introduces_nontrivial_input_drift",
        ]:
            signal["classification"] = "upstream_forward_activation_drift_drives_lora_gradient_drift"
            signal["next_target"] = forward_amp.get("next_target")
        elif layer_replay.get("classification") == "upstream_amplification" and layer_input.get("first_rms_abs_over_1e_2_layer"):
            signal["classification"] = "upstream_forward_activation_drift_drives_lora_gradient_drift"
            signal["next_target"] = f"probe and fix forward drift entering {layer_input.get('first_rms_abs_over_1e_2_layer')} before rerunning readiness"
        else:
            signal["classification"] = "forward_activation_or_upstream_sensitivity_drives_lora_gradient_drift"
            if summary.get("first_divergent_stage"):
                signal["next_target"] = f"localize and fix the forward activation drift at {summary.get('first_divergent_stage')} before rerunning readiness"
            else:
                signal["next_target"] = "localize and fix the forward activation drift around the first large upper-ladder transition before rerunning readiness"
    elif upstream.get("classification") == "upper_encoder_backprop_drift" or ladder.get("classification") == "upper_encoder_ladder_divergent":
        signal["classification"] = "upper_encoder_backprop_drift"
        signal["next_target"] = "probe the first large upper-ladder transition with same-upstream and forward activation dumps"
    elif optimizer.get("classification") == "lora_pre_clip_gradient_drift":
        signal["classification"] = "lora_pre_clip_gradient_drift"
        signal["next_target"] = optimizer.get("next_target")
    else:
        signal["classification"] = "inspect_existing_signals"
        signal["next_target"] = "inspect parity_summary.json signals before choosing a training code change"
    return signal

softmax_vjp_stat_fields = [
    "scores_masked",
    "probs",
    "upstream_probs_grad",
    "scores_masked_grad",
    "cpu_scores_masked_grad",
    "cpu_abs_error",
    "valid_scores_masked_grad",
    "masked_scores_masked_grad",
]

def compare_softmax_vjp_case(zig_case, go_case):
    if not isinstance(zig_case, dict) or not isinstance(go_case, dict):
        return None
    out = {"metadata": {}, "stats": {}}
    for key in ["status", "reason", "outer", "queries", "keys", "has_mask", "mask_bias"]:
        out["metadata"][key] = {
            "zig": zig_case.get(key),
            "go": go_case.get(key),
            "match": zig_case.get(key) == go_case.get(key),
        }
    for key in softmax_vjp_stat_fields:
        cmp = compare_lora_stat(zig_case.get(key), go_case.get(key))
        if cmp is not None:
            out["stats"][key] = cmp
    return out

def compare_softmax_vjp_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {"metadata": {}, "cases": {}}
    for key in ["status", "version", "runtime"]:
        out["metadata"][key] = {
            "zig": zig_probe.get(key),
            "go": go_probe.get(key),
            "match": zig_probe.get(key) == go_probe.get(key),
        }
    zig_cases = zig_probe.get("cases") or {}
    go_cases = go_probe.get("cases") or {}
    for case_name in sorted(set(zig_cases.keys()) | set(go_cases.keys())):
        cmp = compare_softmax_vjp_case(zig_cases.get(case_name), go_cases.get(case_name))
        if cmp is not None:
            out["cases"][case_name] = cmp
    return out

def softmax_vjp_signal(softmax_cmp):
    signal = {
        "case_count": 0,
        "worst_case": None,
        "max_scores_masked_grad_l2_rel_diff": 0.0,
        "max_scores_masked_grad_l2_abs_diff": 0.0,
        "max_valid_scores_masked_grad_l2_rel_diff": 0.0,
        "max_valid_scores_masked_grad_l2_abs_diff": 0.0,
        "max_masked_scores_masked_grad_l2_abs_diff": 0.0,
        "max_cpu_abs_error_l2": 0.0,
    }
    for case_name, cmp in ((softmax_cmp or {}).get("cases") or {}).items():
        signal["case_count"] += 1
        stats = cmp.get("stats") or {}
        scores_cmp = stats.get("valid_scores_masked_grad") or stats.get("scores_masked_grad") or {}
        l2 = scores_cmp.get("l2") or {}
        rel = l2.get("rel_diff") or 0.0
        if rel > signal["max_valid_scores_masked_grad_l2_rel_diff"]:
            signal["worst_case"] = case_name
            signal["max_valid_scores_masked_grad_l2_rel_diff"] = rel
            signal["max_valid_scores_masked_grad_l2_abs_diff"] = l2.get("abs_diff") or 0.0
        full_l2 = ((stats.get("scores_masked_grad") or {}).get("l2") or {})
        signal["max_scores_masked_grad_l2_rel_diff"] = max(
            signal["max_scores_masked_grad_l2_rel_diff"],
            full_l2.get("rel_diff") or 0.0,
        )
        signal["max_scores_masked_grad_l2_abs_diff"] = max(
            signal["max_scores_masked_grad_l2_abs_diff"],
            full_l2.get("abs_diff") or 0.0,
        )
        masked_l2 = ((stats.get("masked_scores_masked_grad") or {}).get("l2") or {})
        signal["max_masked_scores_masked_grad_l2_abs_diff"] = max(
            signal["max_masked_scores_masked_grad_l2_abs_diff"],
            masked_l2.get("abs_diff") or 0.0,
        )
        zig_cpu_error = (((stats.get("cpu_abs_error") or {}).get("l2") or {}).get("zig")) or 0.0
        go_cpu_error = (((stats.get("cpu_abs_error") or {}).get("l2") or {}).get("go")) or 0.0
        signal["max_cpu_abs_error_l2"] = max(signal["max_cpu_abs_error_l2"], zig_cpu_error, go_cpu_error)
    return signal

def compare_qkv_split_vjp_case(zig_case, go_case):
    if not isinstance(zig_case, dict) or not isinstance(go_case, dict):
        return None
    out = {"metadata": {}, "components": {}}
    for key in ["status", "reason", "batch", "seq_len", "num_heads", "head_dim", "hidden_size", "outer"]:
        out["metadata"][key] = {
            "zig": zig_case.get(key),
            "go": go_case.get(key),
            "match": zig_case.get(key) == go_case.get(key),
        }
    out["components"] = compare_lora_stat_maps(
        zig_case.get("components") or {},
        go_case.get("components") or {},
    )
    return out

def compare_qkv_split_vjp_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {"metadata": {}, "cases": {}}
    for key in ["status", "version", "runtime"]:
        out["metadata"][key] = {
            "zig": zig_probe.get(key),
            "go": go_probe.get(key),
            "match": zig_probe.get(key) == go_probe.get(key),
        }
    zig_cases = zig_probe.get("cases") or {}
    go_cases = go_probe.get("cases") or {}
    for case_name in sorted(set(zig_cases.keys()) | set(go_cases.keys())):
        cmp = compare_qkv_split_vjp_case(zig_cases.get(case_name), go_cases.get(case_name))
        if cmp is not None:
            out["cases"][case_name] = cmp
    return out

def qkv_split_vjp_signal(qkv_cmp):
    signal = {
        "case_count": 0,
        "component_count": 0,
        "worst_case": None,
        "worst_component": None,
        "max_l2_rel_diff": 0.0,
        "max_l2_abs_diff": 0.0,
        "max_cpu_abs_error_l2": 0.0,
    }
    for case_name, case_cmp in ((qkv_cmp or {}).get("cases") or {}).items():
        signal["case_count"] += 1
        for component_name, component_cmp in (case_cmp.get("components") or {}).items():
            signal["component_count"] += 1
            l2 = component_cmp.get("l2") or {}
            if component_name.endswith("_cpu_abs_error"):
                signal["max_cpu_abs_error_l2"] = max(
                    signal["max_cpu_abs_error_l2"],
                    l2.get("zig") or 0.0,
                    l2.get("go") or 0.0,
                )
                continue
            rel = l2.get("rel_diff") or 0.0
            if rel > signal["max_l2_rel_diff"]:
                signal["worst_case"] = case_name
                signal["worst_component"] = component_name
                signal["max_l2_rel_diff"] = rel
                signal["max_l2_abs_diff"] = l2.get("abs_diff") or 0.0
    return signal

def qkv_accumulation_signal(qkv_cmp):
    component = ((((qkv_cmp or {}).get("cases") or {}).get("qkv_sum_consistency") or {}).get("components") or {}).get("shared_minus_sum") or {}
    l2 = component.get("l2") or {}
    max_abs = component.get("max_abs") or {}
    mean_abs = component.get("mean_abs") or {}
    return {
        "zig_l2": l2.get("zig"),
        "go_l2": l2.get("go"),
        "l2_abs_diff": l2.get("abs_diff"),
        "l2_rel_diff": l2.get("rel_diff"),
        "zig_max_abs": max_abs.get("zig"),
        "go_max_abs": max_abs.get("go"),
        "zig_mean_abs": mean_abs.get("zig"),
        "go_mean_abs": mean_abs.get("go"),
    }

def lora_stat_comparison_diverged(cmp, abs_threshold=1e-4, rel_threshold=1e-3):
    if not isinstance(cmp, dict):
        return False
    if cmp.get("zig_elems") != cmp.get("go_elems"):
        return True
    sample = cmp.get("sample") or {}
    if sample.get("max_abs_diff", 0.0) > abs_threshold:
        return True
    for key in ["l2", "max_abs", "mean_abs"]:
        item = cmp.get(key)
        if not isinstance(item, dict):
            continue
        if item.get("abs_diff", 0.0) > abs_threshold and item.get("rel_diff", 0.0) > rel_threshold:
            return True
    return False

def backward_decomp_stage_diverged(cmp):
    if not isinstance(cmp, dict):
        return False
    if not (cmp.get("status") or {}).get("match", True):
        return True
    if not (cmp.get("reason") or {}).get("match", True):
        return True
    for key in ["stats", "adapter_a", "adapter_b"]:
        if lora_stat_comparison_diverged(cmp.get(key)):
            return True
    for entry_cmp in (cmp.get("components") or {}).values():
        if lora_stat_comparison_diverged(entry_cmp):
            return True
    for key in ["adapter_a_names", "adapter_b_names"]:
        name_cmp = cmp.get(key)
        if isinstance(name_cmp, dict) and not name_cmp.get("match", True):
            return True
    for map_key in ["adapter_a_by_name", "adapter_b_by_name"]:
        for entry_cmp in (cmp.get(map_key) or {}).values():
            if lora_stat_comparison_diverged(entry_cmp):
                return True
    return False

def first_backward_decomp_divergent_stage(decomp_comparisons):
    stages = (decomp_comparisons or {}).get("stages") or {}
    for stage in backward_decomp_stage_order:
        if backward_decomp_stage_diverged(stages.get(stage)):
            return stage
    for stage in sorted(set(stages.keys()) - set(backward_decomp_stage_order)):
        if backward_decomp_stage_diverged(stages.get(stage)):
            return stage
    return "none"

def worst_lora_sample_comparisons(comparisons, limit=20):
    def score(item):
        cmp = item[1] or {}
        sample = cmp.get("sample") or {}
        l2 = cmp.get("l2") or {}
        return (
            float(sample.get("max_abs_diff") or 0.0),
            float(l2.get("rel_diff") or 0.0),
        )
    return [
        {"name": name, **cmp}
        for name, cmp in sorted((comparisons or {}).items(), key=score, reverse=True)[:limit]
    ]

def comparison_rel(comparisons, name):
    item = (comparisons or {}).get(name) or {}
    return (item.get("rel_diff") if isinstance(item, dict) else None)

def comparison_abs(comparisons, name):
    item = (comparisons or {}).get(name) or {}
    return (item.get("abs_diff") if isinstance(item, dict) else None)

def comparison_values(comparisons, name):
    item = (comparisons or {}).get(name) or {}
    if not isinstance(item, dict):
        return {"zig": None, "go": None}
    return {"zig": item.get("zig"), "go": item.get("go")}

def max_lora_map_l2_rel(comparisons):
    max_rel = 0.0
    worst_name = None
    for name, cmp in (comparisons or {}).items():
        rel = (((cmp or {}).get("l2") or {}).get("rel_diff")) or 0.0
        if rel > max_rel:
            max_rel = rel
            worst_name = name
    return {"name": worst_name, "l2_rel_diff": max_rel}

def l2_square(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) * float(value)

def pre_clip_norm_decomposition(comparisons):
    global_values = comparison_values(comparisons, "grad_norm_pre_clip")
    lora_values = comparison_values(comparisons, "lora_grad_norm_pre_clip")
    task_values = comparison_values(comparisons, "task_head_grad_norm_pre_clip")
    pieces = {
        "global": global_values,
        "lora": lora_values,
        "task_head": task_values,
    }
    squared = {}
    for name, values in pieces.items():
        zig_sq = l2_square(values.get("zig"))
        go_sq = l2_square(values.get("go"))
        squared[name] = {
            "zig_sq": zig_sq,
            "go_sq": go_sq,
            "go_minus_zig_sq": (go_sq - zig_sq) if zig_sq is not None and go_sq is not None else None,
        }
    global_delta = squared["global"].get("go_minus_zig_sq")
    lora_delta = squared["lora"].get("go_minus_zig_sq")
    task_delta = squared["task_head"].get("go_minus_zig_sq")
    residual_delta = None
    if all(isinstance(value, (int, float)) and math.isfinite(value) for value in [global_delta, lora_delta, task_delta]):
        residual_delta = global_delta - lora_delta - task_delta
    denom = max(abs(global_delta or 0.0), 1e-12)
    shares = {
        "lora_abs_share_of_global_sq_diff": abs(lora_delta or 0.0) / denom,
        "task_head_abs_share_of_global_sq_diff": abs(task_delta or 0.0) / denom,
        "residual_abs_share_of_global_sq_diff": abs(residual_delta or 0.0) / denom,
    }
    primary_driver = "missing"
    if global_delta is not None:
        driver_scores = {
            "lora": shares["lora_abs_share_of_global_sq_diff"],
            "task_head": shares["task_head_abs_share_of_global_sq_diff"],
            "residual": shares["residual_abs_share_of_global_sq_diff"],
        }
        primary_driver = max(driver_scores.items(), key=lambda item: item[1])[0]
    return {
        "squared_norms": squared,
        "residual_go_minus_zig_sq": residual_delta,
        "primary_driver": primary_driver,
        **shares,
    }

def optimizer_update_signal(comparisons, pre_clip_matrix_comparisons, update_matrix_comparisons, adam_m_matrix_comparisons, adam_v_matrix_comparisons):
    pre_clip_rel = comparison_rel(comparisons, "grad_norm_pre_clip")
    clip_scale_rel = comparison_rel(comparisons, "grad_clip_scale")
    post_clip_rel = comparison_rel(comparisons, "grad_norm_post_clip")
    lora_pre_rel = comparison_rel(comparisons, "lora_grad_norm_pre_clip")
    lora_post_rel = comparison_rel(comparisons, "lora_grad_norm_post_clip")
    task_pre_rel = comparison_rel(comparisons, "task_head_grad_norm_pre_clip")
    task_post_rel = comparison_rel(comparisons, "task_head_grad_norm_post_clip")
    update_rel = comparison_rel(comparisons, "update_norm")
    update_max_abs_rel = comparison_rel(comparisons, "update_max_abs")
    adam_m_rel = comparison_rel(comparisons, "adam_m_norm")
    adam_v_rel = comparison_rel(comparisons, "adam_v_norm")
    norm_decomposition = pre_clip_norm_decomposition(comparisons)
    pre_clip_worst = max_lora_map_l2_rel(pre_clip_matrix_comparisons)
    update_worst = max_lora_map_l2_rel(update_matrix_comparisons)
    adam_m_worst = max_lora_map_l2_rel(adam_m_matrix_comparisons)
    adam_v_worst = max_lora_map_l2_rel(adam_v_matrix_comparisons)
    scalar_rels = {
        "grad_norm_pre_clip": pre_clip_rel,
        "grad_clip_scale": clip_scale_rel,
        "grad_norm_post_clip": post_clip_rel,
        "lora_grad_norm_pre_clip": lora_pre_rel,
        "lora_grad_norm_post_clip": lora_post_rel,
        "task_head_grad_norm_pre_clip": task_pre_rel,
        "task_head_grad_norm_post_clip": task_post_rel,
        "update_norm": update_rel,
        "update_max_abs": update_max_abs_rel,
        "adam_m_norm": adam_m_rel,
        "adam_v_norm": adam_v_rel,
    }
    finite_rels = {
        key: value for key, value in scalar_rels.items()
        if isinstance(value, (int, float)) and math.isfinite(value)
    }
    worst_scalar = None
    if finite_rels:
        worst_scalar = max(finite_rels.items(), key=lambda item: item[1])
    max_scalar_rel = max(abs(value) for value in finite_rels.values()) if finite_rels else None
    pre_clip_worst_rel = pre_clip_worst.get("l2_rel_diff") if isinstance(pre_clip_worst, dict) else 0.0
    update_worst_rel = update_worst.get("l2_rel_diff") if isinstance(update_worst, dict) else 0.0
    signal = {
        "classification": "missing",
        "next_target": None,
        "scalar_rel_diffs": scalar_rels,
        "scalar_abs_diffs": {
            "grad_norm_pre_clip": comparison_abs(comparisons, "grad_norm_pre_clip"),
            "grad_clip_scale": comparison_abs(comparisons, "grad_clip_scale"),
            "grad_norm_post_clip": comparison_abs(comparisons, "grad_norm_post_clip"),
            "lora_grad_norm_pre_clip": comparison_abs(comparisons, "lora_grad_norm_pre_clip"),
            "lora_grad_norm_post_clip": comparison_abs(comparisons, "lora_grad_norm_post_clip"),
            "task_head_grad_norm_pre_clip": comparison_abs(comparisons, "task_head_grad_norm_pre_clip"),
            "task_head_grad_norm_post_clip": comparison_abs(comparisons, "task_head_grad_norm_post_clip"),
            "update_norm": comparison_abs(comparisons, "update_norm"),
            "update_max_abs": comparison_abs(comparisons, "update_max_abs"),
            "adam_m_norm": comparison_abs(comparisons, "adam_m_norm"),
            "adam_v_norm": comparison_abs(comparisons, "adam_v_norm"),
        },
        "grad_norm_pre_clip": comparison_values(comparisons, "grad_norm_pre_clip"),
        "grad_clip_scale": comparison_values(comparisons, "grad_clip_scale"),
        "grad_norm_post_clip": comparison_values(comparisons, "grad_norm_post_clip"),
        "pre_clip_norm_decomposition": norm_decomposition,
        "worst_scalar": {"name": worst_scalar[0], "rel_diff": worst_scalar[1]} if worst_scalar is not None else None,
        "worst_pre_clip_lora_matrix_l2_rel": pre_clip_worst,
        "worst_update_lora_matrix_l2_rel": update_worst,
        "worst_adam_m_lora_matrix_l2_rel": adam_m_worst,
        "worst_adam_v_lora_matrix_l2_rel": adam_v_worst,
    }
    if not finite_rels:
        return signal
    if max_scalar_rel <= 1e-4 and update_worst_rel <= 1e-3:
        signal["classification"] = "optimizer_update_clean"
        signal["next_target"] = "run_short_readiness_probe"
    elif (update_rel or 0.0) > 1e-3 or (update_max_abs_rel or 0.0) > 1e-3 or update_worst_rel > 1e-3:
        signal["classification"] = "update_norm_drift"
        signal["next_target"] = "compare AdamW update timestep, denominator, epsilon, and weight decay against Go CPU update"
    elif (
        norm_decomposition.get("primary_driver") == "lora" and
        (norm_decomposition.get("lora_abs_share_of_global_sq_diff") or 0.0) >= 0.8 and
        (task_pre_rel or 0.0) <= 1e-3 and
        pre_clip_worst_rel > 1e-3
    ):
        signal["classification"] = "lora_pre_clip_gradient_drift"
        signal["next_target"] = "compare unclipped LoRA matrix gradients before global clipping"
    elif (pre_clip_rel or 0.0) > 1e-2 or (clip_scale_rel or 0.0) > 1e-2 or (task_pre_rel or 0.0) > 1e-2 or (lora_pre_rel or 0.0) > 1e-2:
        signal["classification"] = "pre_clip_gradient_norm_drift"
        signal["next_target"] = "compare unclipped task-head and LoRA gradients plus global-norm membership before clipping"
    elif (adam_v_rel or 0.0) > 1e-2:
        signal["classification"] = "adam_v_moment_drift"
        signal["next_target"] = "compare AdamW second-moment formula, epsilon path, and optimizer-state load mapping"
    elif (adam_m_rel or 0.0) > 1e-2:
        signal["classification"] = "adam_m_moment_drift"
        signal["next_target"] = "compare AdamW first-moment formula and optimizer-state load mapping"
    else:
        signal["classification"] = "optimizer_update_small_drift"
        signal["next_target"] = "tighten tolerances or proceed to a short readiness sanity run"
    return signal

def compare_number_sample(zig_sample, go_sample):
    zig_sample = zig_sample or []
    go_sample = go_sample or []
    sample_count = min(len(zig_sample), len(go_sample))
    if sample_count == 0:
        return None
    diffs = [abs(float(zig_sample[i]) - float(go_sample[i])) for i in range(sample_count)]
    return {
        "count": sample_count,
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / sample_count,
    }

def compare_slice_probe_stat(zig_stat, go_stat):
    if not isinstance(zig_stat, dict) or not isinstance(go_stat, dict):
        return None
    out = {}
    for key in ["elems", "nonzero", "l2", "max_abs", "mean_abs"]:
        cmp = compare_numeric_value(zig_stat.get(key), go_stat.get(key))
        if cmp is not None:
            out[key] = cmp
    return out

def compare_contrastive_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {}
    for key in ["active_chunks", "contrastive_loss", "total_loss", "first_active_index", "first_active_doc_id"]:
        cmp = compare_numeric_value(zig_probe.get(key), go_probe.get(key))
        if cmp is not None:
            out[key] = cmp
    for key in ["embeddings", "grad"]:
        cmp = compare_slice_probe_stat(zig_probe.get(key), go_probe.get(key))
        if cmp is not None:
            out[key] = cmp
    for key in [
        "first_active_embedding_sample",
        "first_active_grad_sample",
        "active_doc_id_sample",
        "active_embedding_norm_sample",
        "active_grad_norm_sample",
    ]:
        cmp = compare_number_sample(zig_probe.get(key), go_probe.get(key))
        if cmp is not None:
            out[key] = cmp
    return out

projection_components = [
    "input", "base", "lora_a", "lora_b", "delta", "output",
    "weight", "bias", "lora_a_weight", "lora_b_weight",
]

projection_reference_error_components = [
    "base_reference_error",
    "lora_a_reference_error",
    "lora_b_reference_error",
    "delta_reference_error",
    "output_reference_error",
]

def compare_projection_probe(zig_probe, go_probe):
    if not isinstance(zig_probe, dict) or not isinstance(go_probe, dict):
        return None
    out = {"metadata_mismatches": [], "ignored_metadata_mismatches": [], "components": {}}
    for key in ["rank", "rows", "in_dim", "out_dim", "has_bias"]:
        zv = zig_probe.get(key)
        gv = go_probe.get(key)
        out[key] = {"zig": zv, "go": gv, "match": zv == gv}
        if zv != gv:
            out["metadata_mismatches"].append(key)
    zscale = zig_probe.get("scale")
    gscale = go_probe.get("scale")
    if isinstance(zscale, (int, float)) and isinstance(gscale, (int, float)):
        abs_diff = abs(float(zscale) - float(gscale))
        out["scale"] = {"zig": zscale, "go": gscale, "abs_diff": abs_diff, "match": abs_diff <= 1e-6}
        if abs_diff > 1e-6:
            out["metadata_mismatches"].append("scale")
    else:
        out["scale"] = {"zig": zscale, "go": gscale, "match": zscale == gscale}
        if zscale != gscale:
            out["metadata_mismatches"].append("scale")
    for component in projection_components:
        cmp = compare_probe_stat(zig_probe.get(component), go_probe.get(component))
        if cmp is not None:
            out["components"][component] = cmp
    if "has_bias" in out["metadata_mismatches"]:
        bias_cmp = (out.get("components") or {}).get("bias") or {}
        zero_bias = True
        for metric in ["rms", "max_abs", "mean"]:
            item = bias_cmp.get(metric) or {}
            for side in ["zig", "go", "abs_diff"]:
                value = item.get(side)
                if not isinstance(value, (int, float)) or abs(float(value)) > 1e-12:
                    zero_bias = False
        if zero_bias:
            out["metadata_mismatches"] = [m for m in out["metadata_mismatches"] if m != "has_bias"]
            out["ignored_metadata_mismatches"].append("has_bias_zero_bias")
            out["has_bias"]["ignored_zero_bias_mismatch"] = True
    return out

def compare_projection_maps(zig_map, go_map, stage_names):
    out = {}
    for stage in stage_names:
        cmp = compare_projection_probe((zig_map or {}).get(stage), (go_map or {}).get(stage))
        if cmp is not None:
            out[stage] = cmp
    return out

def compare_attention_row_probe(zig_row, go_row):
    if not isinstance(zig_row, dict) or not isinstance(go_row, dict):
        return None
    out = {}
    for key in [
        "batch", "head", "query", "valid_keys", "score_mean", "score_rms",
        "score_min", "score_max", "score_argmax", "prob_entropy", "prob_max",
        "prob_argmax", "prob_top2_gap", "query_rms", "query_max_abs",
        "key_query_rms", "key_query_max_abs", "value_query_rms", "value_query_max_abs",
        "output_mean", "output_rms", "output_max_abs",
    ]:
        zv = zig_row.get(key)
        gv = go_row.get(key)
        if isinstance(zv, (int, float)) and isinstance(gv, (int, float)):
            abs_diff = abs(float(zv) - float(gv))
            denom = max(abs(float(zv)), abs(float(gv)), 1e-12)
            out[key] = {
                "zig": zv,
                "go": gv,
                "abs_diff": abs_diff,
                "rel_diff": abs_diff / denom,
            }
    for key in ["query_sample", "key_query_sample", "value_query_sample", "score_sample", "prob_sample", "output_sample"]:
        zs = zig_row.get(key) or []
        gs = go_row.get(key) or []
        sample_count = min(len(zs), len(gs))
        if sample_count:
            diffs = [abs(float(zs[i]) - float(gs[i])) for i in range(sample_count)]
            out[key] = {
                "count": sample_count,
                "max_abs_diff": max(diffs),
                "mean_abs_diff": sum(diffs) / sample_count,
            }
    return out

def compare_attention_row_maps(zig_map, go_map):
    out = {}
    for stage in sorted(set((zig_map or {}).keys()) | set((go_map or {}).keys())):
        cmp = compare_attention_row_probe((zig_map or {}).get(stage), (go_map or {}).get(stage))
        if cmp is not None:
            out[stage] = cmp
    return out

def split_layer_stage(stage):
    text = str(stage)
    match = re.match(r"^layer_(\d+)(?:_(.*))?$", text)
    if not match:
        return (1_000_000, text)
    suffix = match.group(2) or ""
    return (int(match.group(1)), suffix)

def layer_stage_sort_key(stage):
    layer, suffix = split_layer_stage(stage)
    return (layer, suffix)

def activation_stage_sort_key(stage):
    order = {
        "query_proj": 0,
        "key_proj": 1,
        "value_proj": 2,
        "out_proj": 3,
        "wo": 4,
    }
    layer, suffix = split_layer_stage(stage)
    return (layer, order.get(suffix, 1_000_000), suffix)

projection_stage_sort_key = activation_stage_sort_key

def attention_internal_stage_sort_key(stage):
    order = {
        "q_raw": 0,
        "k_raw": 1,
        "v_raw": 2,
        "q_rope": 3,
        "k_rope": 4,
        "attn_scores_raw": 5,
        "attn_scores_masked": 6,
        "attn_probs": 7,
        "attn_context_ref": 8,
        "attn_context_delta": 9,
        "attn_output": 10,
        "attn_token_ref": 11,
        "attn_token_delta": 12,
        "attn_kernel_ref": 13,
        "attn_kernel_delta": 14,
    }
    layer, suffix = split_layer_stage(stage)
    return (layer, order.get(suffix, 1_000_000), suffix)

def layer_state_stage_sort_key(stage):
    order = {
        "hidden_after_attn": 0,
        "mlp_norm_output": 1,
        "wi_output": 2,
        "gate_input": 3,
        "gate_value": 4,
        "gelu_input": 5,
        "wo_input": 6,
        "ffn_out": 7,
        "layer_output": 8,
    }
    layer, suffix = split_layer_stage(stage)
    return (layer, order.get(suffix, 1_000_000), suffix)

def stages_with_suffix(*maps, suffix):
    names = set()
    for stage_map in maps:
        names.update((stage_map or {}).keys())
    return [name for name in sorted(names, key=activation_stage_sort_key) if split_layer_stage(name)[1] == suffix]

def stat_rms(probe_map, name):
    item = (probe_map or {}).get(name) or {}
    value = item.get("rms")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

def pick_sdpa_layout(token_delta, kernel_delta):
    candidates = []
    if token_delta is not None:
        candidates.append(("token_major", token_delta))
    if kernel_delta is not None:
        candidates.append(("kernel_major", kernel_delta))
    if not candidates:
        return "unknown"
    return min(candidates, key=lambda item: item[1])[0]

def probe_stage_diverged(cmp, abs_threshold=1e-4, rel_threshold=1e-3):
    if not isinstance(cmp, dict):
        return False
    if cmp.get("zig_elems") != cmp.get("go_elems"):
        return True
    if cmp.get("sample_max_abs_diff", 0.0) > abs_threshold:
        return True
    for key in ["mean", "rms", "max_abs"]:
        item = cmp.get(key)
        if not isinstance(item, dict):
            continue
        if item.get("abs_diff", 0.0) > abs_threshold and item.get("rel_diff", 0.0) > rel_threshold:
            return True
    return False

def attention_row_stage_diverged(cmp, abs_threshold=1e-4, rel_threshold=1e-3):
    if not isinstance(cmp, dict):
        return False
    for key, item in cmp.items():
        if isinstance(item, dict) and "abs_diff" in item and "rel_diff" in item:
            if key in ["batch", "head", "query", "valid_keys", "score_argmax", "prob_argmax"]:
                if item.get("abs_diff", 0.0) != 0.0:
                    return True
            elif item.get("abs_diff", 0.0) > abs_threshold and item.get("rel_diff", 0.0) > rel_threshold:
                return True
        elif isinstance(item, dict) and item.get("max_abs_diff", 0.0) > abs_threshold:
            return True
    return False

def projection_stage_diverged(cmp, abs_threshold=1e-4, rel_threshold=1e-3):
    if not isinstance(cmp, dict):
        return False
    if cmp.get("metadata_mismatches"):
        return True
    ignore_zero_bias_shape = ((cmp.get("has_bias") or {}).get("ignored_zero_bias_mismatch") is True)
    for component in projection_components:
        if component == "bias" and ignore_zero_bias_shape:
            continue
        if probe_stage_diverged((cmp.get("components") or {}).get(component), abs_threshold, rel_threshold):
            return True
    return False

def layer_input_jump_signal(encoder_comparisons):
    rows = []
    for stage in sorted((encoder_comparisons or {}).keys(), key=layer_stage_sort_key):
        cmp = (encoder_comparisons or {}).get(stage) or {}
        rms = cmp.get("rms") or {}
        value = rms.get("abs_diff")
        rel = rms.get("rel_diff")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            rows.append({
                "stage": stage,
                "rms_abs_diff": float(value),
                "rms_rel_diff": float(rel) if isinstance(rel, (int, float)) and math.isfinite(float(rel)) else None,
                "sample_max_abs_diff": cmp.get("sample_max_abs_diff"),
            })
    signal = {
        "layers_compared": len(rows),
        "largest_rms_abs_layer": None,
        "largest_rms_abs_diff": None,
        "max_rms_abs_jump_from": None,
        "max_rms_abs_jump_to": None,
        "max_rms_abs_jump": None,
        "max_rms_abs_jump_ratio": None,
        "first_rms_abs_over_1e_2_layer": None,
    }
    if not rows:
        return signal
    largest = max(rows, key=lambda item: item["rms_abs_diff"])
    signal["largest_rms_abs_layer"] = largest["stage"]
    signal["largest_rms_abs_diff"] = largest["rms_abs_diff"]
    previous = rows[0]
    for current in rows[1:]:
        jump = current["rms_abs_diff"] - previous["rms_abs_diff"]
        ratio = current["rms_abs_diff"] / max(previous["rms_abs_diff"], 1e-12)
        if signal["max_rms_abs_jump"] is None or jump > signal["max_rms_abs_jump"]:
            signal["max_rms_abs_jump_from"] = previous["stage"]
            signal["max_rms_abs_jump_to"] = current["stage"]
            signal["max_rms_abs_jump"] = jump
            signal["max_rms_abs_jump_ratio"] = ratio
        previous = current
    for row in rows:
        if row["rms_abs_diff"] > 1e-2:
            signal["first_rms_abs_over_1e_2_layer"] = row["stage"]
            break
    return signal

def layer_input_direct_diff_signal(replay_probe):
    diff = (replay_probe or {}).get("actual_input_diff") or {}
    valid_diff = (replay_probe or {}).get("actual_input_valid_token_diff") or {}
    padding_diff = (replay_probe or {}).get("actual_input_padding_token_diff") or {}
    rms = diff.get("rms")
    max_abs = diff.get("max_abs")
    mean_abs = diff.get("mean")
    valid_rms = valid_diff.get("rms")
    valid_max_abs = valid_diff.get("max_abs")
    padding_rms = padding_diff.get("rms")
    padding_max_abs = padding_diff.get("max_abs")
    signal = {
        "status": "missing",
        "classification": "missing_direct_diff",
        "target_layer": (replay_probe or {}).get("target_layer"),
        "rms": rms if isinstance(rms, (int, float)) else None,
        "mean": mean_abs if isinstance(mean_abs, (int, float)) else None,
        "max_abs": max_abs if isinstance(max_abs, (int, float)) else None,
        "valid_rms": valid_rms if isinstance(valid_rms, (int, float)) else None,
        "valid_max_abs": valid_max_abs if isinstance(valid_max_abs, (int, float)) else None,
        "padding_rms": padding_rms if isinstance(padding_rms, (int, float)) else None,
        "padding_max_abs": padding_max_abs if isinstance(padding_max_abs, (int, float)) else None,
        "hash": diff.get("hash") if isinstance(diff.get("hash"), str) else None,
    }
    if not isinstance(diff, dict) or not isinstance(diff.get("elems"), int) or diff.get("elems") <= 0:
        return signal
    signal["status"] = "captured"
    rms_value = abs(float(rms)) if isinstance(rms, (int, float)) and math.isfinite(float(rms)) else 0.0
    max_abs_value = abs(float(max_abs)) if isinstance(max_abs, (int, float)) and math.isfinite(float(max_abs)) else 0.0
    valid_rms_value = abs(float(valid_rms)) if isinstance(valid_rms, (int, float)) and math.isfinite(float(valid_rms)) else None
    valid_max_abs_value = abs(float(valid_max_abs)) if isinstance(valid_max_abs, (int, float)) and math.isfinite(float(valid_max_abs)) else None
    padding_rms_value = abs(float(padding_rms)) if isinstance(padding_rms, (int, float)) and math.isfinite(float(padding_rms)) else None
    if valid_rms_value is not None and valid_rms_value <= 1e-4 and (valid_max_abs_value or 0.0) <= 1e-3:
        if rms_value > 1e-2 and (padding_rms_value or 0.0) > 1e-2:
            signal["classification"] = "layer_input_direct_diff_padding_only_drift"
        else:
            signal["classification"] = "layer_input_direct_diff_clean"
    elif valid_rms_value is not None and valid_rms_value <= 1e-2:
        signal["classification"] = "layer_input_direct_diff_valid_small_drift"
    elif rms_value <= 1e-4 and max_abs_value <= 1e-4:
        signal["classification"] = "layer_input_direct_diff_clean"
    elif rms_value <= 1e-2:
        signal["classification"] = "layer_input_direct_diff_small_drift"
    else:
        signal["classification"] = "layer_input_direct_diff_material_drift"
    return signal

def layer_state_signal(layer_state_comparisons):
    signal = {
        "states_compared": len(layer_state_comparisons or {}),
        "max_sample_abs_diff": 0.0,
        "max_rms_abs_diff": 0.0,
        "max_rms_rel_diff": 0.0,
        "worst_sample_stage": None,
        "worst_rms_stage": None,
    }
    for stage, cmp in (layer_state_comparisons or {}).items():
        sample = cmp.get("sample_max_abs_diff") or 0.0
        rms = cmp.get("rms") or {}
        rms_abs = rms.get("abs_diff") or 0.0
        rms_rel = rms.get("rel_diff") or 0.0
        if sample > signal["max_sample_abs_diff"]:
            signal["max_sample_abs_diff"] = sample
            signal["worst_sample_stage"] = stage
        if rms_rel > signal["max_rms_rel_diff"]:
            signal["max_rms_rel_diff"] = rms_rel
            signal["max_rms_abs_diff"] = rms_abs
            signal["worst_rms_stage"] = stage
    return signal

def layer_internal_jump_signal(layer_state_comparisons):
    rows = []
    for stage in sorted((layer_state_comparisons or {}).keys(), key=layer_state_stage_sort_key):
        cmp = (layer_state_comparisons or {}).get(stage) or {}
        rms = cmp.get("rms") or {}
        rms_abs = rms.get("abs_diff")
        rms_rel = rms.get("rel_diff")
        if isinstance(rms_abs, (int, float)) and math.isfinite(float(rms_abs)):
            rows.append({
                "stage": stage,
                "rms_abs_diff": float(rms_abs),
                "rms_rel_diff": float(rms_rel) if isinstance(rms_rel, (int, float)) and math.isfinite(float(rms_rel)) else None,
                "sample_max_abs_diff": cmp.get("sample_max_abs_diff"),
            })
    signal = {
        "states_compared": len(rows),
        "largest_rms_abs_stage": None,
        "largest_rms_abs_diff": None,
        "max_rms_abs_jump_from": None,
        "max_rms_abs_jump_to": None,
        "max_rms_abs_jump": None,
        "max_rms_abs_jump_ratio": None,
        "first_rms_abs_over_1e_2_stage": None,
        "terminal_stage": None,
        "terminal_rms_abs_diff": None,
    }
    if not rows:
        return signal
    largest = max(rows, key=lambda item: item["rms_abs_diff"])
    signal["largest_rms_abs_stage"] = largest["stage"]
    signal["largest_rms_abs_diff"] = largest["rms_abs_diff"]
    signal["terminal_stage"] = rows[-1]["stage"]
    signal["terminal_rms_abs_diff"] = rows[-1]["rms_abs_diff"]
    previous = rows[0]
    for current in rows[1:]:
        jump = current["rms_abs_diff"] - previous["rms_abs_diff"]
        ratio = current["rms_abs_diff"] / max(previous["rms_abs_diff"], 1e-12)
        if signal["max_rms_abs_jump"] is None or jump > signal["max_rms_abs_jump"]:
            signal["max_rms_abs_jump_from"] = previous["stage"]
            signal["max_rms_abs_jump_to"] = current["stage"]
            signal["max_rms_abs_jump"] = jump
            signal["max_rms_abs_jump_ratio"] = ratio
        previous = current
    for row in rows:
        if row["rms_abs_diff"] > 1e-2:
            signal["first_rms_abs_over_1e_2_stage"] = row["stage"]
            break
    return signal

def layer_state_rms_abs(layer_state_comparisons, layer, suffix):
    cmp = (layer_state_comparisons or {}).get(f"layer_{int(layer):02d}_{suffix}") or {}
    rms = cmp.get("rms") or {}
    value = rms.get("abs_diff")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None

def comparison_rms_abs(comparisons, name):
    cmp = (comparisons or {}).get(name) or {}
    rms = cmp.get("rms") or {}
    value = rms.get("abs_diff")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None

def comparison_rms_pair(comparisons, name):
    cmp = (comparisons or {}).get(name) or {}
    rms = cmp.get("rms") or {}
    out = {}
    for key in ["zig", "go", "abs_diff", "rel_diff"]:
        value = rms.get(key)
        out[key] = float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None
    return out

def min_finite(*values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return None
    return min(float(value) for value in finite)

def decode_attention_flat_index(index, batch_size, seq_len, hidden_size, num_heads=12):
    if not isinstance(index, int) or batch_size <= 0 or seq_len <= 0 or hidden_size <= 0 or num_heads <= 0:
        return None
    head_dim = hidden_size // num_heads
    if head_dim <= 0:
        return None
    row = index // hidden_size
    dim = index % hidden_size
    return {
        "index": index,
        "row": row,
        "dim": dim,
        "batch": row // seq_len,
        "token": row % seq_len,
        "head": dim // head_dim,
        "head_dim": dim % head_dim,
    }

def attention_reference_outlier_signal(go_replay_probe, replay_input, layer):
    prefix = f"layer_{int(layer):02d}"
    internals = (go_replay_probe or {}).get("encoder_attention_internal_probe") or {}
    token_delta = internals.get(f"{prefix}_attn_token_delta") or {}
    indices = token_delta.get("top_abs_indices") or []
    values = token_delta.get("top_abs_values") or []
    batch_size = int((replay_input or {}).get("batch_size") or (go_replay_probe or {}).get("batch_size") or 0)
    seq_len = int((replay_input or {}).get("seq_len") or (go_replay_probe or {}).get("seq_len") or 0)
    hidden_size = int((replay_input or {}).get("hidden_size") or (go_replay_probe or {}).get("hidden_size") or 0)
    valid_lengths = (go_replay_probe or {}).get("attention_mask_valid_lengths") or []
    outliers = []
    for pos, index in enumerate(indices[:8]):
        if not isinstance(index, int):
            continue
        decoded = decode_attention_flat_index(index, batch_size, seq_len, hidden_size)
        if decoded is None:
            continue
        batch = decoded.get("batch")
        token = decoded.get("token")
        valid_length = valid_lengths[batch] if isinstance(batch, int) and 0 <= batch < len(valid_lengths) else None
        on_valid_token = None
        if isinstance(valid_length, int) and isinstance(token, int):
            on_valid_token = token < valid_length
        decoded.update({
            "value": values[pos] if pos < len(values) else None,
            "valid_length": valid_length,
            "on_valid_token": on_valid_token,
            "tokens_past_valid": (token - valid_length) if isinstance(token, int) and isinstance(valid_length, int) else None,
        })
        outliers.append(decoded)
    valid_flags = [item.get("on_valid_token") for item in outliers if item.get("on_valid_token") is not None]
    classification = "missing"
    if outliers and valid_flags and all(flag is False for flag in valid_flags):
        classification = "attention_reference_outliers_on_padding"
    elif outliers and any(flag is False for flag in valid_flags):
        classification = "attention_reference_outliers_include_padding"
    elif outliers and all(flag is True for flag in valid_flags):
        classification = "attention_reference_outliers_on_valid_tokens"
    elif outliers:
        classification = "attention_reference_outliers_unclassified"
    return {
        "classification": classification,
        "target_layer": layer,
        "go_token_delta_rms": token_delta.get("rms"),
        "go_token_delta_max_abs": token_delta.get("max_abs"),
        "attention_mask_valid_lengths": valid_lengths,
        "top_go_token_delta_outliers": outliers,
    }

def layer_input_rms_abs(encoder_comparisons, layer):
    cmp = (encoder_comparisons or {}).get(f"layer_{int(layer):02d}") or {}
    rms = cmp.get("rms") or {}
    value = rms.get("abs_diff")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None

def attention_context_replay_signal(attention_internal_comparisons, layer):
    prefix = f"layer_{int(layer):02d}"
    q = comparison_rms_abs(attention_internal_comparisons, f"{prefix}_q_raw")
    k = comparison_rms_abs(attention_internal_comparisons, f"{prefix}_k_raw")
    v = comparison_rms_abs(attention_internal_comparisons, f"{prefix}_v_raw")
    probs = comparison_rms_abs(attention_internal_comparisons, f"{prefix}_attn_probs")
    context = comparison_rms_abs(attention_internal_comparisons, f"{prefix}_attn_output")
    context_ref = comparison_rms_abs(attention_internal_comparisons, f"{prefix}_attn_context_ref")
    token_delta = comparison_rms_pair(attention_internal_comparisons, f"{prefix}_attn_token_delta")
    kernel_delta = comparison_rms_pair(attention_internal_comparisons, f"{prefix}_attn_kernel_delta")
    token_ref = comparison_rms_pair(attention_internal_comparisons, f"{prefix}_attn_token_ref")
    kernel_ref = comparison_rms_pair(attention_internal_comparisons, f"{prefix}_attn_kernel_ref")
    zig_best_ref_delta = min_finite(token_delta.get("zig"), kernel_delta.get("zig"))
    go_best_ref_delta = min_finite(token_delta.get("go"), kernel_delta.get("go"))
    signal = {
        "classification": "missing",
        "target_layer": layer,
        "q_raw_rms_abs_diff": q,
        "k_raw_rms_abs_diff": k,
        "v_raw_rms_abs_diff": v,
        "attn_probs_rms_abs_diff": probs,
        "attn_context_rms_abs_diff": context,
        "attn_context_ref_rms_abs_diff": context_ref,
        "max_qkv_probs_rms_abs_diff": max([value for value in [q, k, v, probs] if value is not None], default=None),
        "zig_token_ref_delta_rms": token_delta.get("zig"),
        "go_token_ref_delta_rms": token_delta.get("go"),
        "token_ref_delta_rms_abs_diff": token_delta.get("abs_diff"),
        "zig_kernel_ref_delta_rms": kernel_delta.get("zig"),
        "go_kernel_ref_delta_rms": kernel_delta.get("go"),
        "kernel_ref_delta_rms_abs_diff": kernel_delta.get("abs_diff"),
        "token_ref_rms_abs_diff": token_ref.get("abs_diff"),
        "kernel_ref_rms_abs_diff": kernel_ref.get("abs_diff"),
        "zig_best_ref_delta_rms": zig_best_ref_delta,
        "go_best_ref_delta_rms": go_best_ref_delta,
        "next_target": None,
    }
    max_inputs = signal["max_qkv_probs_rms_abs_diff"]
    if context is None:
        return signal
    if context > 5e-5 and max_inputs is not None and max_inputs <= 1e-6:
        if zig_best_ref_delta is not None and zig_best_ref_delta <= 5e-5 and go_best_ref_delta is not None and go_best_ref_delta > 1e-4:
            signal["classification"] = "attention_reference_contract_mismatch"
            signal["next_target"] = f"inspect layer_{int(layer):02d} Go MPSGraph attention reference/decomposition outliers before patching Metal SDPA"
        elif zig_best_ref_delta is not None and go_best_ref_delta is not None and zig_best_ref_delta <= 5e-5 and go_best_ref_delta <= 5e-5:
            signal["classification"] = "attention_context_accumulation_drift"
            signal["next_target"] = f"fix layer_{int(layer):02d} SDPA/context accumulation drift with clean Q/K/V/probs"
        else:
            signal["classification"] = "attention_context_output_drift"
            signal["next_target"] = f"inspect layer_{int(layer):02d} attention context/output parity with local reference deltas"
    elif context > 1e-4:
        signal["classification"] = "attention_context_drift"
        signal["next_target"] = f"inspect layer_{int(layer):02d} attention internals"
    else:
        signal["classification"] = "attention_context_clean"
        signal["next_target"] = None
    return signal

def geglu_product_signal(layer_state_comparisons, replay_layer_state_comparisons, layer):
    native_gelu = layer_state_rms_abs(layer_state_comparisons, layer, "gelu_input")
    native_gate = layer_state_rms_abs(layer_state_comparisons, layer, "gate_value")
    native_wo_input = layer_state_rms_abs(layer_state_comparisons, layer, "wo_input")
    native_ffn_out = layer_state_rms_abs(layer_state_comparisons, layer, "ffn_out")
    replay_wo_input = layer_state_rms_abs(replay_layer_state_comparisons, layer, "wo_input")
    replay_ffn_out = layer_state_rms_abs(replay_layer_state_comparisons, layer, "ffn_out")
    native_product_source = max([v for v in [native_gelu, native_gate] if v is not None], default=None)
    return {
        "target_layer": layer,
        "native_gelu_input_rms_abs_diff": native_gelu,
        "native_gate_value_rms_abs_diff": native_gate,
        "native_wo_input_rms_abs_diff": native_wo_input,
        "native_ffn_out_rms_abs_diff": native_ffn_out,
        "native_product_amplification": (native_wo_input / max(native_product_source, 1e-12)) if native_wo_input is not None and native_product_source is not None else None,
        "native_wo_projection_amplification": (native_ffn_out / max(native_wo_input, 1e-12)) if native_ffn_out is not None and native_wo_input is not None else None,
        "replay_wo_input_rms_abs_diff": replay_wo_input,
        "replay_ffn_out_rms_abs_diff": replay_ffn_out,
        "replay_wo_projection_amplification": (replay_ffn_out / max(replay_wo_input, 1e-12)) if replay_ffn_out is not None and replay_wo_input is not None else None,
    }

def forward_amplification_signal(encoder_comparisons, layer_state_comparisons, layer_replay, geglu_signal, layer):
    target_input = layer_input_rms_abs(encoder_comparisons, layer)
    next_input = layer_input_rms_abs(encoder_comparisons, int(layer) + 1)
    native_layer_output = layer_state_rms_abs(layer_state_comparisons, layer, "layer_output")
    native_ffn_out = (geglu_signal or {}).get("native_ffn_out_rms_abs_diff")
    native_wo_input = (geglu_signal or {}).get("native_wo_input_rms_abs_diff")
    rows = []
    for stage in sorted((encoder_comparisons or {}).keys(), key=layer_stage_sort_key):
        cmp = (encoder_comparisons or {}).get(stage) or {}
        rms = cmp.get("rms") or {}
        value = rms.get("abs_diff")
        rel = rms.get("rel_diff")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            rows.append({
                "stage": stage,
                "rms_abs_diff": float(value),
                "rms_rel_diff": float(rel) if isinstance(rel, (int, float)) and math.isfinite(float(rel)) else None,
            })
    pre_target_rows = []
    for row in rows:
        try:
            index = int(row["stage"].split("_")[1])
        except Exception:
            continue
        if index <= int(layer):
            pre_target_rows.append(row)
    first_over_1e_4 = next((row for row in rows if row["rms_abs_diff"] > 1e-4), None)
    first_over_5e_4 = next((row for row in rows if row["rms_abs_diff"] > 5e-4), None)
    first_over_1e_3 = next((row for row in rows if row["rms_abs_diff"] > 1e-3), None)
    largest_pre_target = max(pre_target_rows, key=lambda item: item["rms_abs_diff"], default=None)
    signal = {
        "classification": "missing",
        "target_layer": layer,
        "target_input_rms_abs_diff": target_input,
        "next_layer_input_rms_abs_diff": next_input,
        "native_layer_output_rms_abs_diff": native_layer_output,
        "native_ffn_out_rms_abs_diff": native_ffn_out,
        "native_wo_input_rms_abs_diff": native_wo_input,
        "target_to_next_input_amplification": (next_input / max(target_input, 1e-12)) if target_input is not None and next_input is not None else None,
        "target_input_to_wo_input_amplification": (native_wo_input / max(target_input, 1e-12)) if target_input is not None and native_wo_input is not None else None,
        "target_input_to_ffn_out_amplification": (native_ffn_out / max(target_input, 1e-12)) if target_input is not None and native_ffn_out is not None else None,
        "first_input_rms_abs_over_1e_4_layer": first_over_1e_4["stage"] if first_over_1e_4 else None,
        "first_input_rms_abs_over_5e_4_layer": first_over_5e_4["stage"] if first_over_5e_4 else None,
        "first_input_rms_abs_over_1e_3_layer": first_over_1e_3["stage"] if first_over_1e_3 else None,
        "largest_pre_target_input_rms_abs_layer": largest_pre_target["stage"] if largest_pre_target else None,
        "largest_pre_target_input_rms_abs_diff": largest_pre_target["rms_abs_diff"] if largest_pre_target else None,
        "layer_replay_classification": (layer_replay or {}).get("classification"),
        "layer_replay_layer_output_rms_abs_diff": (layer_replay or {}).get("layer_output_rms_abs_diff"),
        "next_target": None,
    }
    target_to_next = signal["target_to_next_input_amplification"]
    if (
        target_input is not None
        and next_input is not None
        and target_input <= 1e-3
        and next_input > 1e-2
        and (layer_replay or {}).get("classification") == "upstream_amplification"
    ):
        signal["classification"] = "target_layer_amplifies_accumulated_input_drift"
        source_layer = signal["first_input_rms_abs_over_5e_4_layer"] or signal["first_input_rms_abs_over_1e_4_layer"]
        if source_layer:
            signal["next_target"] = f"localize accumulated forward drift before {source_layer}, then retest layer_{int(layer):02d} amplification"
        else:
            signal["next_target"] = f"localize sub-1e-3 accumulated forward drift before layer_{int(layer):02d}, then retest amplification"
    elif (
        target_input is not None
        and next_input is not None
        and target_input <= 5e-4
        and next_input > 5e-4
        and (target_to_next or 0.0) > 3.0
        and (layer_replay or {}).get("classification") == "upstream_amplification"
    ):
        signal["classification"] = "target_layer_materializes_accumulated_input_drift"
        signal["next_target"] = f"inspect layer_{int(layer):02d} internal forward stages, then localize incoming drift before layer_{int(layer):02d}"
    elif (
        target_input is not None
        and next_input is not None
        and target_input <= 1e-4
        and next_input > 1e-4
        and (target_to_next or 0.0) > 3.0
        and (layer_replay or {}).get("classification") == "upstream_amplification"
    ):
        signal["classification"] = "target_layer_introduces_nontrivial_input_drift"
        signal["next_target"] = f"inspect layer_{int(layer):02d} internal forward stages, then localize incoming drift before layer_{int(layer):02d}"
    elif (layer_replay or {}).get("classification") == "upstream_amplification":
        signal["classification"] = "upstream_amplification"
        signal["next_target"] = f"localize incoming drift before layer_{int(layer):02d}"
    elif (layer_replay or {}).get("classification") == "wo_projection_amplifies_subthreshold_input_drift":
        signal["classification"] = (layer_replay or {}).get("classification")
        signal["next_target"] = f"fix layer_{int(layer):02d} GeGLU/tail input replay drift before Wo projection"
    elif (layer_replay or {}).get("classification"):
        signal["classification"] = (layer_replay or {}).get("classification")
        signal["next_target"] = f"fix layer_{int(layer):02d} local replay mismatch before rerunning readiness"
    return signal

def layer_replay_signal(replay_probe, replay_layer_state_comparisons, replay_projection_comparisons, layer):
    signal = {
        "status": "not_captured",
        "classification": "missing_replay",
        "target_layer": layer,
        "wo_input_rms_abs_diff": None,
        "ffn_out_rms_abs_diff": None,
        "layer_output_rms_abs_diff": None,
        "projection_output_rms_abs_diff": None,
        "projection_weight_rms_abs_diff": None,
        "classification_threshold": 1e-4,
    }
    if not isinstance(replay_probe, dict) or replay_probe.get("status") != "captured":
        if isinstance(replay_probe, dict) and replay_probe.get("status"):
            signal["status"] = replay_probe.get("status")
            signal["classification"] = replay_probe.get("reason") or "missing_replay"
        return signal
    signal["status"] = "captured"
    wo_input = layer_state_rms_abs(replay_layer_state_comparisons, layer, "wo_input")
    ffn_out = layer_state_rms_abs(replay_layer_state_comparisons, layer, "ffn_out")
    layer_output = layer_state_rms_abs(replay_layer_state_comparisons, layer, "layer_output")
    projection = (replay_projection_comparisons or {}).get(f"layer_{int(layer):02d}_wo") or {}
    components = projection.get("components") or {}
    projection_output = ((components.get("output") or {}).get("rms") or {}).get("abs_diff")
    projection_weight = ((components.get("weight") or {}).get("rms") or {}).get("abs_diff")
    signal["wo_input_rms_abs_diff"] = wo_input
    signal["ffn_out_rms_abs_diff"] = ffn_out
    signal["layer_output_rms_abs_diff"] = layer_output
    signal["projection_output_rms_abs_diff"] = projection_output if isinstance(projection_output, (int, float)) else None
    signal["projection_weight_rms_abs_diff"] = projection_weight if isinstance(projection_weight, (int, float)) else None
    threshold = signal["classification_threshold"]
    if wo_input is not None and wo_input > threshold:
        signal["classification"] = "geglu_product_or_tail_input_mismatch"
    elif (
        ffn_out is not None
        and ffn_out > threshold
        and wo_input is not None
        and wo_input > threshold / 10.0
    ):
        signal["classification"] = "wo_projection_amplifies_subthreshold_input_drift"
    elif ffn_out is not None and ffn_out > threshold:
        signal["classification"] = "wo_projection_mismatch"
    elif layer_output is not None and layer_output > threshold:
        signal["classification"] = "residual_add_or_output_cast_mismatch"
    else:
        signal["classification"] = "upstream_amplification"
    return signal

def attention_reference_diagnostic_stages_to_ignore(attention_signal, target_layer):
    if not isinstance(attention_signal, dict):
        return set()
    if attention_signal.get("classification") != "attention_reference_contract_mismatch":
        return set()
    try:
        prefix = f"layer_{int(target_layer):02d}"
    except Exception:
        return set()
    return {
        f"{prefix}_attn_context_ref",
        f"{prefix}_attn_context_delta",
        f"{prefix}_attn_token_ref",
        f"{prefix}_attn_token_delta",
        f"{prefix}_attn_kernel_ref",
        f"{prefix}_attn_kernel_delta",
    }

def first_divergent_stage(checkpoint_comparisons, embedding_comparisons, table_row_comparisons, lookup_row_comparisons, activation_comparisons, projection_comparisons, attention_internal_comparisons, attention_row_comparisons, layer_state_comparisons, encoder_comparisons, forward_comparisons, ignored_attention_internal_stages=None):
    ignored_attention_internal_stages = ignored_attention_internal_stages or set()
    for stage in ["final_norm_weight", "final_norm_bias", "w1", "b1", "w2", "b2"]:
        if probe_stage_diverged((checkpoint_comparisons or {}).get(stage), 1e-5, 1e-5):
            return "checkpoint_tensors"
    for stage in ["word_embedding_weight", "layer_norm_weight", "layer_norm_bias", "token_lookup", "layer_norm_output"]:
        if probe_stage_diverged((embedding_comparisons or {}).get(stage), 1e-5, 1e-5):
            return f"embedding_probe:{stage}"
    for stage in sorted((table_row_comparisons or {}).keys()):
        if probe_stage_diverged((table_row_comparisons or {}).get(stage), 1e-5, 1e-5):
            return f"embedding_table_row:{stage}"
    for stage in sorted((lookup_row_comparisons or {}).keys()):
        if probe_stage_diverged((lookup_row_comparisons or {}).get(stage), 1e-5, 1e-5):
            return f"embedding_lookup_row:{stage}"
    seen_activation = set()
    seen_projection = set()
    for suffix in ["query_proj", "key_proj", "value_proj"]:
        for stage in stages_with_suffix(activation_comparisons, projection_comparisons, suffix=suffix):
            seen_activation.add(stage)
            seen_projection.add(stage)
            if probe_stage_diverged((activation_comparisons or {}).get(stage)):
                return f"encoder_activation_input:{stage}"
            if projection_stage_diverged((projection_comparisons or {}).get(stage)):
                return f"encoder_projection_decomposition:{stage}"
    for stage in sorted((attention_internal_comparisons or {}).keys(), key=attention_internal_stage_sort_key):
        if stage in ignored_attention_internal_stages:
            continue
        if probe_stage_diverged((attention_internal_comparisons or {}).get(stage)):
            return f"encoder_attention_internal:{stage}"
    for stage in sorted((attention_row_comparisons or {}).keys()):
        if attention_row_stage_diverged((attention_row_comparisons or {}).get(stage)):
            return f"encoder_attention_row:{stage}"
    for stage in sorted((layer_state_comparisons or {}).keys(), key=layer_state_stage_sort_key):
        if probe_stage_diverged((layer_state_comparisons or {}).get(stage)):
            return f"encoder_layer_state:{stage}"
    for suffix in ["out_proj", "wo"]:
        for stage in stages_with_suffix(activation_comparisons, projection_comparisons, suffix=suffix):
            seen_activation.add(stage)
            seen_projection.add(stage)
            if probe_stage_diverged((activation_comparisons or {}).get(stage)):
                return f"encoder_activation_input:{stage}"
            if projection_stage_diverged((projection_comparisons or {}).get(stage)):
                return f"encoder_projection_decomposition:{stage}"
    for stage in sorted((activation_comparisons or {}).keys(), key=activation_stage_sort_key):
        if stage in seen_activation:
            continue
        if probe_stage_diverged((activation_comparisons or {}).get(stage)):
            return f"encoder_activation_input:{stage}"
    for stage in sorted((projection_comparisons or {}).keys(), key=projection_stage_sort_key):
        if stage in seen_projection:
            continue
        if projection_stage_diverged((projection_comparisons or {}).get(stage)):
            return f"encoder_projection_decomposition:{stage}"
    for stage in sorted((encoder_comparisons or {}).keys(), key=layer_stage_sort_key):
        if probe_stage_diverged((encoder_comparisons or {}).get(stage)):
            return f"encoder_layer_input:{stage}"
    for stage in ["final_norm_input", "boundary_head_input", "dense1_pre_activation", "dense1_post_activation", "logits"]:
        if probe_stage_diverged((forward_comparisons or {}).get(stage)):
            return stage
    return "none"

if adapter is None:
    fail("missing checkpoint adapter summary")
else:
    summary["checkpoint_adapter"] = "passed" if adapter.get("status") == "passed" else "failed"
    summary["checkpoint_adapter_summary"] = adapter
    if adapter.get("mapped_tensors") != 224:
        fail(f"checkpoint adapter mapped_tensors={adapter.get('mapped_tensors')} expected 224")
    if adapter.get("skipped_unmapped") != 0 or adapter.get("missing_required") != 0:
        fail("checkpoint adapter skipped required trainable tensors")

if zig_batch is not None and go_batch is not None:
    batch_fields = ["offset", "samples", "batches", "max_seq_len", "max_chunks", "valid_tokens", "boundary_gold_tokens", "first_batch"]
    mismatches = [field for field in batch_fields if zig_batch.get(field) != go_batch.get(field)]
    standalone_hash_mismatches = hash_mismatches(batch_hashes(zig_batch), batch_hashes(go_batch))
    if mismatches or standalone_hash_mismatches:
        fail(f"batch parity mismatch fields={mismatches} hashes={standalone_hash_mismatches}")
        summary["batch_parity"] = "failed"
    else:
        summary["batch_parity"] = "passed"
else:
    if skip_batch_parity:
        summary["batch_parity"] = "not_run"
        summary["notes"].append("Batch parity artifacts were skipped by ANTFLY_FUSED_CHUNKER_STEP_PARITY_SKIP_BATCH=1.")
    else:
        fail("missing batch parity artifacts")

zig_step_hashes = batch_hashes(zig_no_update)
go_no_update_for_hashes = (go_step or {}).get("no_update") if isinstance(go_step, dict) else None
go_step_hashes = batch_hashes(go_no_update_for_hashes)
standalone_hashes = batch_hashes(zig_batch)
zig_vs_go_step_hash_mismatches = hash_mismatches(zig_step_hashes, go_step_hashes) if go_step_hashes is not None else list(hash_fields)
standalone_vs_zig_step_hash_mismatches = hash_mismatches(standalone_hashes, zig_step_hashes) if standalone_hashes is not None and zig_step_hashes is not None else list(hash_fields)
standalone_vs_go_step_hash_mismatches = hash_mismatches(standalone_hashes, go_step_hashes) if standalone_hashes is not None and go_step_hashes is not None else list(hash_fields)
go_step_available_for_hashes = isinstance(go_step, dict) and go_step.get("status") == "passed" and go_step_hashes is not None
summary["step_batch_parity"] = {
    "status": "not_checked",
    "hash_fields": hash_fields,
    "standalone_batch_hashes": standalone_hashes,
    "zig_step_hashes": zig_step_hashes,
    "go_step_hashes": go_step_hashes,
    "zig_vs_go_step_hash_mismatches": zig_vs_go_step_hash_mismatches,
    "standalone_vs_zig_step_hash_mismatches": standalone_vs_zig_step_hash_mismatches,
    "standalone_vs_go_step_hash_mismatches": standalone_vs_go_step_hash_mismatches,
    "zig_vs_go_step_hash_match": go_step_available_for_hashes and len(zig_vs_go_step_hash_mismatches) == 0,
    "standalone_vs_zig_step_hash_match": len(standalone_vs_zig_step_hash_mismatches) == 0,
    "standalone_vs_go_step_hash_match": go_step_available_for_hashes and len(standalone_vs_go_step_hash_mismatches) == 0,
}
if zig_step_hashes is not None:
    if go_step_available_for_hashes:
        if zig_vs_go_step_hash_mismatches:
            summary["step_batch_parity"]["status"] = "failed"
            fail(f"Go/Zig training-step batch hash mismatch fields={zig_vs_go_step_hash_mismatches}")
        elif standalone_vs_zig_step_hash_mismatches:
            summary["step_batch_parity"]["status"] = "go_zig_passed_preflight_mismatch"
            summary["notes"].append(
                "Standalone batch parity hashes differ from the actual Go/Zig training-step batch; treating this as a diagnostic preflight mismatch, not a model math mismatch."
            )
        else:
            summary["step_batch_parity"]["status"] = "passed"
    elif standalone_hashes is not None:
        if standalone_vs_zig_step_hash_mismatches:
            summary["step_batch_parity"]["status"] = "failed"
            fail(f"Zig no-update step hash mismatch against standalone batch fields={standalone_vs_zig_step_hash_mismatches}")
        else:
            summary["step_batch_parity"]["status"] = "standalone_zig_passed"
else:
    summary["step_batch_parity"]["status"] = "failed"
    fail("Zig no-update JSON is missing batch hashes")

if zig_no_update is not None and zig_apply_update is not None:
    summary["zig_step_parity"] = "captured"
    summary["target_probe_layer"] = zig_no_update.get("target_probe_layer")
    summary["zig_target_probe_layer"] = zig_no_update.get("target_probe_layer")
    if zig_no_update.get("phase") != "no_update":
        fail("Zig no-update JSON has wrong phase")
    if zig_apply_update.get("phase") != "apply_update":
        fail("Zig apply-update JSON has wrong phase")
    update = zig_apply_update.get("update")
    if not isinstance(update, dict):
        fail("Zig apply-update JSON is missing update diagnostics")
    else:
        for key in ["grad_norm_pre_clip", "grad_norm_post_clip", "update_norm", "adam_m_norm", "adam_v_norm"]:
            value = update.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                fail(f"Zig update field is not finite: {key}={value!r}")
        if update.get("active_matrices", 0) <= 0:
            fail("Zig update diagnostics report no active LoRA matrices")
    boundary = zig_no_update.get("boundary", {})
    summary["zig_quality_signal"] = {
        "loss": boundary.get("loss"),
        "eval_f1": boundary.get("eval_f1"),
        "prob_gold_pos": boundary.get("prob_gold_pos"),
        "prob_gold_neg": boundary.get("prob_gold_neg"),
        "gold_positive_lower_than_negative": (
            isinstance(boundary.get("prob_gold_pos"), (int, float))
            and isinstance(boundary.get("prob_gold_neg"), (int, float))
            and boundary["prob_gold_pos"] < boundary["prob_gold_neg"]
        ),
    }
else:
    fail("missing Zig frozen-step JSON artifacts")

if go_step is None:
    fail("missing Go frozen-step JSON artifact")
    summary["go_training_step_parity"] = "failed"
elif go_step.get("status") == "skipped":
    summary["go_training_step_parity"] = "pending_mpsgraph_segmented"
    if summary["status"] == "passed":
        summary["status"] = "diagnostic_only"
    summary["notes"].append(go_step.get("reason", "Go training-step parity was skipped."))
else:
    if go_step.get("status") != "passed":
        fail(f"Go frozen-step status={go_step.get('status')!r}")
        summary["go_training_step_parity"] = "failed"
    else:
        summary["go_training_step_parity"] = "captured"
        summary["go_step_summary"] = {
            "backend": go_step.get("backend"),
            "diagnostic_mode": go_step.get("diagnostic_mode"),
            "mixed_precision": go_step.get("mixed_precision"),
            "use_bf16": go_step.get("use_bf16"),
            "trainable_tensors": go_step.get("trainable_tensors"),
            "checkpoint_step_count": go_step.get("checkpoint_step_count"),
            "update_step": go_step.get("update_step"),
	            "boundary_loss": (go_step.get("no_update") or {}).get("boundary_loss"),
	            "contrastive": (go_step.get("no_update") or {}).get("contrastive"),
	            "boundary_probe": (go_step.get("no_update") or {}).get("boundary_probe"),
	            "boundary_forward_probe": (go_step.get("no_update") or {}).get("boundary_forward_probe"),
	            "embedding_probe": (go_step.get("no_update") or {}).get("embedding_probe"),
	            "embedding_table_row_probe": (go_step.get("no_update") or {}).get("embedding_table_row_probe"),
	            "embedding_lookup_row_probe": (go_step.get("no_update") or {}).get("embedding_lookup_row_probe"),
	            "boundary_checkpoint_probe": (go_step.get("no_update") or {}).get("boundary_checkpoint_probe"),
	            "encoder_activation_input_probe": (go_step.get("no_update") or {}).get("encoder_activation_input_probe"),
	            "encoder_projection_decomposition_probe": (go_step.get("no_update") or {}).get("encoder_projection_decomposition_probe"),
	            "encoder_attention_internal_probe": (go_step.get("no_update") or {}).get("encoder_attention_internal_probe"),
	            "encoder_attention_row_probe": (go_step.get("no_update") or {}).get("encoder_attention_row_probe"),
	            "encoder_layer_input_probe": (go_step.get("no_update") or {}).get("encoder_layer_input_probe"),
	            "encoder_layer_state_probe": (go_step.get("no_update") or {}).get("encoder_layer_state_probe"),
	            "encoder_layer_backward_replay_probe": (go_step.get("no_update") or {}).get("encoder_layer_backward_replay_probe"),
	            "layer_backward_decomp_probe": (go_step.get("no_update") or {}).get("layer_backward_decomp_probe"),
	            "upstream_grad_probe": (go_step.get("no_update") or {}).get("upstream_grad_probe"),
	            "softmax_vjp_probe": (go_step.get("no_update") or {}).get("softmax_vjp_probe"),
	            "qkv_split_vjp_probe": (go_step.get("no_update") or {}).get("qkv_split_vjp_probe"),
	            "grad_norm_lora": (go_step.get("no_update") or {}).get("grad_norm_lora"),
	            "grad_tensors": (go_step.get("no_update") or {}).get("grad_tensors"),
	            "lora_grad_tensors": (go_step.get("no_update") or {}).get("lora_grad_tensors"),
	            "task_head_trainables": (go_step.get("no_update") or {}).get("task_head_trainables"),
	            "task_head_grad_tensors": (go_step.get("no_update") or {}).get("task_head_grad_tensors"),
	            "boundary_back_grad_names": (go_step.get("no_update") or {}).get("boundary_back_grad_names"),
	            "update_norm": (go_step.get("apply_update") or {}).get("update_norm"),
	            "grad_norm_pre_clip": (go_step.get("no_update") or {}).get("grad_norm_pre_clip"),
	            "grad_norm_post_clip": (go_step.get("no_update") or {}).get("grad_norm_post_clip"),
	            "grad_clip_scale": (go_step.get("no_update") or {}).get("grad_clip_scale"),
	            "lora_grad_norm_pre_clip": (go_step.get("no_update") or {}).get("lora_grad_norm_pre_clip"),
            "active_matrices": (go_step.get("apply_update") or {}).get("active_matrices"),
            "adam_m_norm": (go_step.get("apply_update") or {}).get("adam_m_norm"),
            "adam_v_norm": (go_step.get("apply_update") or {}).get("adam_v_norm"),
        }
        if go_step.get("trainable_tensors") != 224:
            fail(f"Go trainable_tensors={go_step.get('trainable_tensors')} expected 224")
        if go_step.get("backend") != "mpsgraph":
            fail(f"Go backend={go_step.get('backend')!r} expected 'mpsgraph'")
        if go_step.get("diagnostic_mode") != "mpsgraph_segmented":
            fail(f"Go diagnostic_mode={go_step.get('diagnostic_mode')!r} expected 'mpsgraph_segmented'")
        go_target_probe_layer = go_step.get("target_probe_layer", (go_step.get("no_update") or {}).get("target_probe_layer"))
        summary["go_target_probe_layer"] = go_target_probe_layer
        if summary.get("target_probe_layer") != go_target_probe_layer:
            fail(f"Go/Zig target_probe_layer mismatch: zig={summary.get('target_probe_layer')!r} go={go_target_probe_layer!r}")
        if (go_step.get("apply_update") or {}).get("active_matrices") != (zig_apply_update.get("update") or {}).get("active_matrices"):
            fail("Go/Zig active LoRA matrix count differs")
        if (go_step.get("no_update") or {}).get("task_head_trainables", 0) > 0 and (go_step.get("no_update") or {}).get("task_head_grad_tensors", 0) == 0:
            summary["notes"].append("Go MPSGraph segmented diagnostic found task-head trainables but no task-head gradient tensors returned by boundaryBackExec.")

        comparisons = {}
        zb = (zig_no_update or {}).get("boundary") or {}
        zu = (zig_apply_update or {}).get("update") or {}
        gn = go_step.get("no_update") or {}
        gu = go_step.get("apply_update") or {}
        gb = gn.get("boundary_probe") or {}
        summary["go_quality_signal"] = {
            "loss": gn.get("boundary_loss"),
            "cpu_ce_loss": gb.get("cpu_ce_loss"),
            "eval_f1": f1_from_counts(gb.get("tp"), gb.get("fp"), gb.get("fn")),
            "tp": gb.get("tp"),
            "fp": gb.get("fp"),
            "fn": gb.get("fn"),
            "predicted_positives": gb.get("predicted_positives"),
            "gold_positives": gb.get("gold_positives"),
            "prob_gold_pos": gb.get("prob_gold_pos"),
            "prob_gold_neg": gb.get("prob_gold_neg"),
            "gold_positive_lower_than_negative": (
                isinstance(gb.get("prob_gold_pos"), (int, float))
                and isinstance(gb.get("prob_gold_neg"), (int, float))
                and gb["prob_gold_pos"] < gb["prob_gold_neg"]
            ),
        }
        zig_forward = (zig_no_update or {}).get("boundary_forward_probe") or {}
        go_forward = gn.get("boundary_forward_probe") or {}
        zig_embedding = (zig_no_update or {}).get("embedding_probe") or {}
        go_embedding = gn.get("embedding_probe") or {}
        zig_embedding_table_rows = (zig_no_update or {}).get("embedding_table_row_probe") or {}
        go_embedding_table_rows = gn.get("embedding_table_row_probe") or {}
        zig_embedding_lookup_rows = (zig_no_update or {}).get("embedding_lookup_row_probe") or {}
        go_embedding_lookup_rows = gn.get("embedding_lookup_row_probe") or {}
        zig_checkpoint = (zig_no_update or {}).get("boundary_checkpoint_probe") or {}
        go_checkpoint = gn.get("boundary_checkpoint_probe") or {}
        zig_encoder_activations = (zig_no_update or {}).get("encoder_activation_input_probe") or {}
        go_encoder_activations = gn.get("encoder_activation_input_probe") or {}
        zig_encoder_projection_decompositions = (zig_no_update or {}).get("encoder_projection_decomposition_probe") or {}
        go_encoder_projection_decompositions = gn.get("encoder_projection_decomposition_probe") or {}
        zig_encoder_attention_internals = (zig_no_update or {}).get("encoder_attention_internal_probe") or {}
        go_encoder_attention_internals = gn.get("encoder_attention_internal_probe") or {}
        zig_encoder_attention_rows = (zig_no_update or {}).get("encoder_attention_row_probe") or {}
        go_encoder_attention_rows = gn.get("encoder_attention_row_probe") or {}
        zig_encoder_layers = (zig_no_update or {}).get("encoder_layer_input_probe") or {}
        go_encoder_layers = gn.get("encoder_layer_input_probe") or {}
        zig_encoder_layer_states = (zig_no_update or {}).get("encoder_layer_state_probe") or {}
        go_encoder_layer_states = gn.get("encoder_layer_state_probe") or {}
        zig_encoder_replay_input = (zig_no_update or {}).get("encoder_replay_input") or {}
        zig_encoder_replay_upstream = (zig_apply_update or {}).get("encoder_replay_upstream") or {}
        go_encoder_replay = gn.get("encoder_layer_replay_probe") or {}
        go_encoder_replay_activations = (go_encoder_replay or {}).get("encoder_activation_input_probe") or {}
        go_encoder_replay_attention_internals = (go_encoder_replay or {}).get("encoder_attention_internal_probe") or {}
        go_encoder_replay_projection_decompositions = (go_encoder_replay or {}).get("encoder_projection_decomposition_probe") or {}
        go_encoder_replay_layer_states = (go_encoder_replay or {}).get("encoder_layer_state_probe") or {}
        go_encoder_backward_replay = gn.get("encoder_layer_backward_replay_probe") or {}
        go_encoder_backward_replay_segment = (go_encoder_backward_replay or {}).get("segment_vjp_probe") or {}
        go_encoder_backward_replay_decomp = (go_encoder_backward_replay or {}).get("layer_backward_decomp_probe") or {}
        zig_upstream_grad_probe = (zig_apply_update or {}).get("upstream_grad_probe") or {}
        go_upstream_grad_probe = gn.get("upstream_grad_probe") or {}
        zig_contrastive = (zig_no_update or {}).get("contrastive") or {}
        go_contrastive = gn.get("contrastive") or {}
        zig_segment_probe = (zig_apply_update or {}).get("segment_vjp_probe") or {}
        go_segment_probe = gn.get("segment_vjp_probe") or {}
        zig_layer_backward_decomp_probe = (zig_apply_update or {}).get("layer_backward_decomp_probe") or {}
        go_layer_backward_decomp_probe = gn.get("layer_backward_decomp_probe") or {}
        zig_softmax_vjp_probe = (zig_apply_update or {}).get("softmax_vjp_probe") or {}
        go_softmax_vjp_probe = gn.get("softmax_vjp_probe") or {}
        zig_qkv_split_vjp_probe = (zig_apply_update or {}).get("qkv_split_vjp_probe") or {}
        go_qkv_split_vjp_probe = gn.get("qkv_split_vjp_probe") or {}
        checkpoint_comparisons = compare_probe_maps(
            zig_checkpoint,
            go_checkpoint,
            ["final_norm_weight", "final_norm_bias", "w1", "b1", "w2", "b2"],
        )
        embedding_comparisons = compare_probe_maps(
            zig_embedding,
            go_embedding,
            ["word_embedding_weight", "layer_norm_weight", "layer_norm_bias", "token_lookup", "layer_norm_output"],
        )
        embedding_table_row_names = sorted(set(zig_embedding_table_rows.keys()) | set(go_embedding_table_rows.keys()))
        embedding_table_row_comparisons = compare_probe_maps(
            zig_embedding_table_rows,
            go_embedding_table_rows,
            embedding_table_row_names,
        )
        embedding_lookup_row_names = sorted(set(zig_embedding_lookup_rows.keys()) | set(go_embedding_lookup_rows.keys()))
        embedding_lookup_row_comparisons = compare_probe_maps(
            zig_embedding_lookup_rows,
            go_embedding_lookup_rows,
            embedding_lookup_row_names,
        )
        encoder_activation_names = sorted(set(zig_encoder_activations.keys()) | set(go_encoder_activations.keys()), key=activation_stage_sort_key)
        encoder_activation_comparisons = compare_probe_maps(
            zig_encoder_activations,
            go_encoder_activations,
            encoder_activation_names,
        )
        encoder_projection_decomposition_names = sorted(
            set(zig_encoder_projection_decompositions.keys()) | set(go_encoder_projection_decompositions.keys()),
            key=projection_stage_sort_key,
        )
        encoder_projection_decomposition_comparisons = compare_projection_maps(
            zig_encoder_projection_decompositions,
            go_encoder_projection_decompositions,
            encoder_projection_decomposition_names,
        )
        encoder_attention_internal_names = sorted(set(zig_encoder_attention_internals.keys()) | set(go_encoder_attention_internals.keys()), key=attention_internal_stage_sort_key)
        encoder_attention_internal_comparisons = compare_probe_maps(
            zig_encoder_attention_internals,
            go_encoder_attention_internals,
            encoder_attention_internal_names,
        )
        encoder_attention_row_comparisons = compare_attention_row_maps(
            zig_encoder_attention_rows,
            go_encoder_attention_rows,
        )
        encoder_layer_state_names = sorted(set(zig_encoder_layer_states.keys()) | set(go_encoder_layer_states.keys()), key=layer_state_stage_sort_key)
        encoder_layer_state_comparisons = compare_probe_maps(
            zig_encoder_layer_states,
            go_encoder_layer_states,
            encoder_layer_state_names,
        )
        encoder_replay_activation_names = sorted(set(zig_encoder_activations.keys()) | set(go_encoder_replay_activations.keys()), key=activation_stage_sort_key)
        encoder_replay_activation_comparisons = compare_probe_maps(
            zig_encoder_activations,
            go_encoder_replay_activations,
            encoder_replay_activation_names,
        )
        zig_encoder_replay_attention_internals = dict(zig_encoder_attention_internals or {})
        try:
            replay_target_prefix = f"layer_{int(summary.get('target_probe_layer') or 0):02d}"
        except Exception:
            replay_target_prefix = "layer_00"
        replay_zig_attn_out = (zig_encoder_activations or {}).get(f"{replay_target_prefix}_out_proj")
        if replay_zig_attn_out is not None:
            zig_encoder_replay_attention_internals[f"{replay_target_prefix}_attn_output"] = replay_zig_attn_out
        encoder_replay_attention_internal_names = sorted(
            set(zig_encoder_replay_attention_internals.keys()) | set(go_encoder_replay_attention_internals.keys()),
            key=attention_internal_stage_sort_key,
        )
        encoder_replay_attention_internal_comparisons = compare_probe_maps(
            zig_encoder_replay_attention_internals,
            go_encoder_replay_attention_internals,
            encoder_replay_attention_internal_names,
        )
        encoder_replay_projection_names = sorted(
            set(zig_encoder_projection_decompositions.keys()) | set(go_encoder_replay_projection_decompositions.keys()),
            key=projection_stage_sort_key,
        )
        encoder_replay_projection_comparisons = compare_projection_maps(
            zig_encoder_projection_decompositions,
            go_encoder_replay_projection_decompositions,
            encoder_replay_projection_names,
        )
        encoder_replay_layer_state_names = sorted(set(zig_encoder_layer_states.keys()) | set(go_encoder_replay_layer_states.keys()), key=layer_state_stage_sort_key)
        encoder_replay_layer_state_comparisons = compare_probe_maps(
            zig_encoder_layer_states,
            go_encoder_replay_layer_states,
            encoder_replay_layer_state_names,
        )
        same_upstream_segment_vjp_comparisons = compare_segment_vjp_probe(
            zig_segment_probe,
            go_encoder_backward_replay_segment,
        )
        same_upstream_layer_backward_decomp_comparisons = compare_layer_backward_decomp_probe(
            zig_layer_backward_decomp_probe,
            go_encoder_backward_replay_decomp,
        )
        encoder_layer_names = sorted(set(zig_encoder_layers.keys()) | set(go_encoder_layers.keys()), key=layer_stage_sort_key)
        encoder_layer_comparisons = compare_probe_maps(
            zig_encoder_layers,
            go_encoder_layers,
            encoder_layer_names,
        )
        forward_comparisons = compare_probe_maps(
            zig_forward,
            go_forward,
            ["final_norm_input", "boundary_head_input", "dense1_pre_activation", "dense1_post_activation", "logits"],
        )
        summary["boundary_checkpoint_comparisons"] = checkpoint_comparisons
        summary["embedding_probe_comparisons"] = embedding_comparisons
        summary["embedding_table_row_comparisons"] = embedding_table_row_comparisons
        summary["embedding_lookup_row_comparisons"] = embedding_lookup_row_comparisons
        summary["encoder_activation_input_comparisons"] = encoder_activation_comparisons
        summary["encoder_projection_decomposition_comparisons"] = encoder_projection_decomposition_comparisons
        summary["encoder_attention_internal_comparisons"] = encoder_attention_internal_comparisons
        summary["encoder_attention_row_comparisons"] = encoder_attention_row_comparisons
        summary["encoder_layer_input_comparisons"] = encoder_layer_comparisons
        summary["encoder_layer_state_comparisons"] = encoder_layer_state_comparisons
        summary["encoder_replay_input"] = zig_encoder_replay_input
        summary["encoder_layer_replay_probe"] = {
            "status": go_encoder_replay.get("status") if isinstance(go_encoder_replay, dict) else None,
            "path": go_encoder_replay.get("path") if isinstance(go_encoder_replay, dict) else None,
            "target_layer": go_encoder_replay.get("target_layer") if isinstance(go_encoder_replay, dict) else None,
            "elems": go_encoder_replay.get("elems") if isinstance(go_encoder_replay, dict) else None,
        }
        summary["encoder_layer_replay_activation_comparisons"] = encoder_replay_activation_comparisons
        summary["encoder_layer_replay_attention_internal_comparisons"] = encoder_replay_attention_internal_comparisons
        summary["encoder_layer_replay_projection_comparisons"] = encoder_replay_projection_comparisons
        summary["encoder_layer_replay_comparisons"] = encoder_replay_layer_state_comparisons
        summary["encoder_replay_upstream"] = zig_encoder_replay_upstream
        summary["encoder_layer_backward_replay_probe"] = {
            "status": go_encoder_backward_replay.get("status") if isinstance(go_encoder_backward_replay, dict) else None,
            "reason": go_encoder_backward_replay.get("reason") if isinstance(go_encoder_backward_replay, dict) else None,
            "input_path": go_encoder_backward_replay.get("input_path") if isinstance(go_encoder_backward_replay, dict) else None,
            "upstream_path": go_encoder_backward_replay.get("upstream_path") if isinstance(go_encoder_backward_replay, dict) else None,
            "target_layer": go_encoder_backward_replay.get("target_layer") if isinstance(go_encoder_backward_replay, dict) else None,
            "elems": go_encoder_backward_replay.get("elems") if isinstance(go_encoder_backward_replay, dict) else None,
        }
        summary["same_upstream_segment_vjp_probe_comparisons"] = same_upstream_segment_vjp_comparisons
        summary["same_upstream_layer_backward_decomp_probe_comparisons"] = same_upstream_layer_backward_decomp_comparisons
        summary["same_upstream_backward_signal"] = same_upstream_backward_signal(
            go_encoder_backward_replay,
            same_upstream_segment_vjp_comparisons,
            same_upstream_layer_backward_decomp_comparisons,
        )
        upstream_grad_probe_comparisons = compare_upstream_grad_probe(zig_upstream_grad_probe, go_upstream_grad_probe)
        summary["upstream_grad_probe_comparisons"] = upstream_grad_probe_comparisons
        summary["upstream_grad_assembly_signal"] = upstream_grad_assembly_signal(upstream_grad_probe_comparisons)
        summary["first_upstream_grad_divergent_stage"] = (summary["upstream_grad_assembly_signal"] or {}).get("first_divergent_stage")
        summary["upper_encoder_ladder_comparisons"] = (upstream_grad_probe_comparisons or {}).get("upper_encoder_ladder") if isinstance(upstream_grad_probe_comparisons, dict) else None
        summary["upper_encoder_ladder_signal"] = upper_encoder_ladder_signal(upstream_grad_probe_comparisons)
        summary["first_large_drift_transition"] = (summary["upper_encoder_ladder_signal"] or {}).get("first_large_drift_transition")
        summary["first_large_drift_layer"] = (summary["upper_encoder_ladder_signal"] or {}).get("first_large_drift_layer")
        summary["layer_state_signal"] = layer_state_signal(encoder_layer_state_comparisons)
        summary["layer_internal_jump_signal"] = layer_internal_jump_signal(encoder_layer_state_comparisons)
        summary["layer_input_jump_signal"] = layer_input_jump_signal(encoder_layer_comparisons)
        summary["layer_input_direct_diff_signal"] = layer_input_direct_diff_signal(go_encoder_replay)
        summary["layer_replay_signal"] = layer_replay_signal(go_encoder_replay, encoder_replay_layer_state_comparisons, encoder_replay_projection_comparisons, summary.get("target_probe_layer") or 0)
        summary["attention_context_replay_signal"] = attention_context_replay_signal(encoder_replay_attention_internal_comparisons, summary.get("target_probe_layer") or 0)
        summary["attention_reference_outlier_signal"] = attention_reference_outlier_signal(go_encoder_replay, summary.get("encoder_replay_input") or {}, summary.get("target_probe_layer") or 0)
        summary["geglu_product_signal"] = geglu_product_signal(encoder_layer_state_comparisons, encoder_replay_layer_state_comparisons, summary.get("target_probe_layer") or 0)
        summary["forward_amplification_signal"] = forward_amplification_signal(
            encoder_layer_comparisons,
            encoder_layer_state_comparisons,
            summary["layer_replay_signal"],
            summary["geglu_product_signal"],
            summary.get("target_probe_layer") or 0,
        )
        summary["boundary_forward_comparisons"] = forward_comparisons
        summary["contrastive_comparisons"] = compare_contrastive_probe(zig_contrastive, go_contrastive)
        segment_vjp_comparisons = compare_segment_vjp_probe(zig_segment_probe, go_segment_probe)
        summary["segment_vjp_probe_comparisons"] = segment_vjp_comparisons
        layer_backward_decomp_comparisons = compare_layer_backward_decomp_probe(
            zig_layer_backward_decomp_probe,
            go_layer_backward_decomp_probe,
        )
        summary["layer_backward_decomp_probe_comparisons"] = layer_backward_decomp_comparisons
        summary["layer_backward_component_signal"] = backward_component_signal(layer_backward_decomp_comparisons)
        summary["first_backward_decomp_divergent_stage"] = first_backward_decomp_divergent_stage(layer_backward_decomp_comparisons)
        summary["actual_vs_same_upstream_decomp_signal"] = actual_vs_same_upstream_decomp_signal(
            layer_backward_decomp_comparisons,
            same_upstream_layer_backward_decomp_comparisons,
            same_upstream_segment_vjp_comparisons,
        )
        softmax_vjp_comparisons = compare_softmax_vjp_probe(
            zig_softmax_vjp_probe,
            go_softmax_vjp_probe,
        )
        summary["softmax_vjp_probe_comparisons"] = softmax_vjp_comparisons
        summary["softmax_vjp_signal"] = softmax_vjp_signal(softmax_vjp_comparisons)
        qkv_split_vjp_comparisons = compare_qkv_split_vjp_probe(
            zig_qkv_split_vjp_probe,
            go_qkv_split_vjp_probe,
        )
        summary["qkv_split_vjp_probe_comparisons"] = qkv_split_vjp_comparisons
        summary["qkv_split_vjp_signal"] = qkv_split_vjp_signal(qkv_split_vjp_comparisons)
        summary["qkv_accumulation_signal"] = qkv_accumulation_signal(qkv_split_vjp_comparisons)
        if isinstance(segment_vjp_comparisons, dict):
            summary["segment_vjp_adapter_a_worst"] = worst_lora_sample_comparisons(segment_vjp_comparisons.get("adapter_a_by_name") or {})
            summary["segment_vjp_adapter_b_worst"] = worst_lora_sample_comparisons(segment_vjp_comparisons.get("adapter_b_by_name") or {})
        target_layer = summary.get("target_probe_layer")
        try:
            target_prefix = f"layer_{int(target_layer):02d}"
        except Exception:
            target_prefix = "layer_00"
        zig_token_delta = stat_rms(zig_encoder_attention_internals, f"{target_prefix}_attn_token_delta")
        zig_kernel_delta = stat_rms(zig_encoder_attention_internals, f"{target_prefix}_attn_kernel_delta")
        go_token_delta = stat_rms(go_encoder_attention_internals, f"{target_prefix}_attn_token_delta")
        go_kernel_delta = stat_rms(go_encoder_attention_internals, f"{target_prefix}_attn_kernel_delta")
        summary["sdpa_reference_signal"] = {
            "zig_token_delta_rms": zig_token_delta,
            "zig_kernel_delta_rms": zig_kernel_delta,
            "zig_best_layout": pick_sdpa_layout(zig_token_delta, zig_kernel_delta),
            "go_token_delta_rms": go_token_delta,
            "go_kernel_delta_rms": go_kernel_delta,
            "go_best_layout": pick_sdpa_layout(go_token_delta, go_kernel_delta),
            "target_prefix": target_prefix,
            "token_ref_rms_abs_diff": ((encoder_attention_internal_comparisons.get(f"{target_prefix}_attn_token_ref") or {}).get("rms") or {}).get("abs_diff"),
            "kernel_ref_rms_abs_diff": ((encoder_attention_internal_comparisons.get(f"{target_prefix}_attn_kernel_ref") or {}).get("rms") or {}).get("abs_diff"),
        }
        projection_signal = {
            "projections_compared": len(encoder_projection_decomposition_comparisons or {}),
            "metadata_mismatches": [],
            "max_base_sample_abs_diff": 0.0,
            "max_lora_a_sample_abs_diff": 0.0,
            "max_lora_b_sample_abs_diff": 0.0,
            "max_delta_sample_abs_diff": 0.0,
            "max_output_sample_abs_diff": 0.0,
            "max_weight_sample_abs_diff": 0.0,
            "max_lora_a_weight_sample_abs_diff": 0.0,
            "max_lora_b_weight_sample_abs_diff": 0.0,
        }
        for stage, cmp in (encoder_projection_decomposition_comparisons or {}).items():
            for mismatch in cmp.get("metadata_mismatches") or []:
                projection_signal["metadata_mismatches"].append(f"{stage}:{mismatch}")
            components = cmp.get("components") or {}
            projection_signal["max_base_sample_abs_diff"] = max(projection_signal["max_base_sample_abs_diff"], (components.get("base") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_lora_a_sample_abs_diff"] = max(projection_signal["max_lora_a_sample_abs_diff"], (components.get("lora_a") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_lora_b_sample_abs_diff"] = max(projection_signal["max_lora_b_sample_abs_diff"], (components.get("lora_b") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_delta_sample_abs_diff"] = max(projection_signal["max_delta_sample_abs_diff"], (components.get("delta") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_output_sample_abs_diff"] = max(projection_signal["max_output_sample_abs_diff"], (components.get("output") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_weight_sample_abs_diff"] = max(projection_signal["max_weight_sample_abs_diff"], (components.get("weight") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_lora_a_weight_sample_abs_diff"] = max(projection_signal["max_lora_a_weight_sample_abs_diff"], (components.get("lora_a_weight") or {}).get("sample_max_abs_diff") or 0.0)
            projection_signal["max_lora_b_weight_sample_abs_diff"] = max(projection_signal["max_lora_b_weight_sample_abs_diff"], (components.get("lora_b_weight") or {}).get("sample_max_abs_diff") or 0.0)
        summary["projection_decomposition_signal"] = projection_signal
        reference_error_signal = {
            "projections_compared": len(encoder_projection_decomposition_names or []),
        }
        for component in projection_reference_error_components:
            for side_name, probe_map in [
                ("zig", zig_encoder_projection_decompositions),
                ("go", go_encoder_projection_decompositions),
            ]:
                max_abs = 0.0
                max_rms = 0.0
                for probe in (probe_map or {}).values():
                    stat = (probe or {}).get(component) or {}
                    abs_value = stat.get("max_abs")
                    rms_value = stat.get("rms")
                    if isinstance(abs_value, (int, float)) and math.isfinite(abs_value):
                        max_abs = max(max_abs, float(abs_value))
                    if isinstance(rms_value, (int, float)) and math.isfinite(rms_value):
                        max_rms = max(max_rms, float(rms_value))
                reference_error_signal[f"{side_name}_{component}_max_abs"] = max_abs
                reference_error_signal[f"{side_name}_{component}_max_rms"] = max_rms
        summary["projection_reference_error_signal"] = reference_error_signal
        max_score_sample_diff = 0.0
        max_prob_sample_diff = 0.0
        max_output_sample_diff = 0.0
        max_query_sample_diff = 0.0
        max_key_query_sample_diff = 0.0
        max_value_query_sample_diff = 0.0
        argmax_mismatches = 0
        for cmp in (encoder_attention_row_comparisons or {}).values():
            max_query_sample_diff = max(max_query_sample_diff, ((cmp.get("query_sample") or {}).get("max_abs_diff") or 0.0))
            max_key_query_sample_diff = max(max_key_query_sample_diff, ((cmp.get("key_query_sample") or {}).get("max_abs_diff") or 0.0))
            max_value_query_sample_diff = max(max_value_query_sample_diff, ((cmp.get("value_query_sample") or {}).get("max_abs_diff") or 0.0))
            max_score_sample_diff = max(max_score_sample_diff, ((cmp.get("score_sample") or {}).get("max_abs_diff") or 0.0))
            max_prob_sample_diff = max(max_prob_sample_diff, ((cmp.get("prob_sample") or {}).get("max_abs_diff") or 0.0))
            max_output_sample_diff = max(max_output_sample_diff, ((cmp.get("output_sample") or {}).get("max_abs_diff") or 0.0))
            for key in ["score_argmax", "prob_argmax"]:
                item = cmp.get(key) or {}
                if item.get("abs_diff", 0.0) != 0.0:
                    argmax_mismatches += 1
        summary["attention_row_signal"] = {
            "rows_compared": len(encoder_attention_row_comparisons or {}),
            "max_query_sample_abs_diff": max_query_sample_diff,
            "max_key_query_sample_abs_diff": max_key_query_sample_diff,
            "max_value_query_sample_abs_diff": max_value_query_sample_diff,
            "max_score_sample_abs_diff": max_score_sample_diff,
            "max_prob_sample_abs_diff": max_prob_sample_diff,
            "max_output_sample_abs_diff": max_output_sample_diff,
            "argmax_mismatches": argmax_mismatches,
        }
        ignored_attention_stages = attention_reference_diagnostic_stages_to_ignore(
            summary.get("attention_context_replay_signal"),
            summary.get("target_probe_layer") or 0,
        )
        summary["first_divergent_stage_ignored_diagnostics"] = sorted(ignored_attention_stages)
        summary["first_divergent_stage"] = first_divergent_stage(checkpoint_comparisons, embedding_comparisons, embedding_table_row_comparisons, embedding_lookup_row_comparisons, encoder_activation_comparisons, encoder_projection_decomposition_comparisons, encoder_attention_internal_comparisons, encoder_attention_row_comparisons, encoder_layer_state_comparisons, encoder_layer_comparisons, forward_comparisons, ignored_attention_stages)
        pairs = [
            ("boundary_loss", zb.get("loss"), gn.get("boundary_loss")),
            ("contrastive_loss", zig_contrastive.get("contrastive_loss"), gn.get("contrastive_loss")),
            ("contrastive_total_loss", zig_contrastive.get("total_loss"), go_contrastive.get("total_loss")),
            ("contrastive_embedding_l2", (zig_contrastive.get("embeddings") or {}).get("l2"), (go_contrastive.get("embeddings") or {}).get("l2")),
            ("contrastive_embedding_max_abs", (zig_contrastive.get("embeddings") or {}).get("max_abs"), (go_contrastive.get("embeddings") or {}).get("max_abs")),
            ("contrastive_grad_l2", (zig_contrastive.get("grad") or {}).get("l2"), (go_contrastive.get("grad") or {}).get("l2")),
            ("contrastive_grad_max_abs", (zig_contrastive.get("grad") or {}).get("max_abs"), (go_contrastive.get("grad") or {}).get("max_abs")),
            ("upstream_contrastive_features_grad_rms", (((zig_upstream_grad_probe.get("stages") or {}).get("contrastive_features_grad") or {}).get("rms")), (((go_upstream_grad_probe.get("stages") or {}).get("contrastive_features_grad") or {}).get("rms"))),
            ("upstream_contrastive_encoder_grad_rms", (((zig_upstream_grad_probe.get("stages") or {}).get("contrastive_encoder_grad") or {}).get("rms")), (((go_upstream_grad_probe.get("stages") or {}).get("contrastive_encoder_grad") or {}).get("rms"))),
            ("upstream_lora_output_grad_rms", (((zig_upstream_grad_probe.get("stages") or {}).get("lora_output_grad") or {}).get("rms")), (((go_upstream_grad_probe.get("stages") or {}).get("lora_output_grad") or {}).get("rms"))),
            ("upstream_target_segment_rms", (((zig_upstream_grad_probe.get("stages") or {}).get("target_segment_upstream") or {}).get("rms")), (((go_upstream_grad_probe.get("stages") or {}).get("target_segment_upstream") or {}).get("rms"))),
            ("segment_vjp_upstream_l2", (zig_segment_probe.get("upstream") or {}).get("l2"), (go_segment_probe.get("upstream") or {}).get("l2")),
            ("segment_vjp_hidden_grad_l2", (zig_segment_probe.get("hidden_grad") or {}).get("l2"), (go_segment_probe.get("hidden_grad") or {}).get("l2")),
            ("segment_vjp_adapter_a_l2", (zig_segment_probe.get("adapter_a") or {}).get("l2"), (go_segment_probe.get("adapter_a") or {}).get("l2")),
            ("segment_vjp_adapter_b_l2", (zig_segment_probe.get("adapter_b") or {}).get("l2"), (go_segment_probe.get("adapter_b") or {}).get("l2")),
            ("layer_backward_decomp_incoming_upstream_l2", (((zig_layer_backward_decomp_probe.get("stages") or {}).get("incoming_upstream") or {}).get("stats") or {}).get("l2"), (((go_layer_backward_decomp_probe.get("stages") or {}).get("incoming_upstream") or {}).get("stats") or {}).get("l2")),
            ("layer_backward_decomp_full_layer_hidden_grad_l2", (((zig_layer_backward_decomp_probe.get("stages") or {}).get("full_layer_hidden_grad") or {}).get("stats") or {}).get("l2"), (((go_layer_backward_decomp_probe.get("stages") or {}).get("full_layer_hidden_grad") or {}).get("stats") or {}).get("l2")),
            ("layer_backward_decomp_full_layer_adapter_a_l2", (((zig_layer_backward_decomp_probe.get("stages") or {}).get("full_layer_hidden_grad") or {}).get("adapter_a") or {}).get("l2"), (((go_layer_backward_decomp_probe.get("stages") or {}).get("full_layer_hidden_grad") or {}).get("adapter_a") or {}).get("l2")),
            ("layer_backward_decomp_full_layer_adapter_b_l2", (((zig_layer_backward_decomp_probe.get("stages") or {}).get("full_layer_hidden_grad") or {}).get("adapter_b") or {}).get("l2"), (((go_layer_backward_decomp_probe.get("stages") or {}).get("full_layer_hidden_grad") or {}).get("adapter_b") or {}).get("l2")),
            ("boundary_cpu_ce_loss", zb.get("loss"), gb.get("cpu_ce_loss")),
            ("prob_gold_pos", zb.get("prob_gold_pos"), gb.get("prob_gold_pos")),
            ("prob_gold_neg", zb.get("prob_gold_neg"), gb.get("prob_gold_neg")),
            ("predicted_positives", zb.get("predicted_positives"), gb.get("predicted_positives")),
            ("boundary_tp", zb.get("tp"), gb.get("tp")),
            ("boundary_fp", zb.get("fp"), gb.get("fp")),
            ("boundary_fn", zb.get("fn"), gb.get("fn")),
            ("grad_norm_pre_clip", zu.get("grad_norm_pre_clip"), gn.get("grad_norm_pre_clip")),
            ("grad_clip_scale", zu.get("grad_clip_scale"), gn.get("grad_clip_scale")),
            ("lora_grad_norm_pre_clip", zu.get("lora_grad_norm_pre_clip"), gn.get("lora_grad_norm_pre_clip")),
            ("task_head_grad_norm_pre_clip", zu.get("extra_grad_norm_pre_clip"), (gn.get("grad_task_head_pre_clip") or {}).get("l2")),
            ("boundary_dense1_weight_grad_norm_pre_clip", zu.get("boundary_dense1_weight_grad_norm_pre_clip"), (gn.get("grad_w1_pre_clip") or {}).get("l2")),
            ("boundary_dense1_bias_grad_norm_pre_clip", zu.get("boundary_dense1_bias_grad_norm_pre_clip"), (gn.get("grad_b1_pre_clip") or {}).get("l2")),
            ("boundary_dense2_weight_grad_norm_pre_clip", zu.get("boundary_dense2_weight_grad_norm_pre_clip"), (gn.get("grad_w2_pre_clip") or {}).get("l2")),
            ("boundary_dense2_bias_grad_norm_pre_clip", zu.get("boundary_dense2_bias_grad_norm_pre_clip"), (gn.get("grad_b2_pre_clip") or {}).get("l2")),
            ("lora_grad_norm_post_clip", zu.get("lora_grad_norm_post_clip"), gn.get("grad_norm_lora")),
            ("task_head_grad_norm_post_clip", zu.get("extra_grad_norm_post_clip"), (gn.get("grad_task_head") or {}).get("l2")),
            ("boundary_dense1_weight_grad_norm_post_clip", zu.get("boundary_dense1_weight_grad_norm_post_clip"), (gn.get("grad_w1") or {}).get("l2")),
            ("boundary_dense1_bias_grad_norm_post_clip", zu.get("boundary_dense1_bias_grad_norm_post_clip"), (gn.get("grad_b1") or {}).get("l2")),
            ("boundary_dense2_weight_grad_norm_post_clip", zu.get("boundary_dense2_weight_grad_norm_post_clip"), (gn.get("grad_w2") or {}).get("l2")),
            ("boundary_dense2_bias_grad_norm_post_clip", zu.get("boundary_dense2_bias_grad_norm_post_clip"), (gn.get("grad_b2") or {}).get("l2")),
            ("grad_norm_post_clip", zu.get("grad_norm_post_clip"), gn.get("grad_norm_post_clip")),
            ("update_norm", zu.get("update_norm"), gu.get("update_norm")),
            ("update_max_abs", zu.get("update_max_abs"), gu.get("update_max_abs")),
            ("adam_m_norm", zu.get("adam_m_norm"), gu.get("adam_m_norm")),
            ("adam_v_norm", zu.get("adam_v_norm"), gu.get("adam_v_norm")),
        ]
        for name, zig_value, go_value in pairs:
            if isinstance(zig_value, (int, float)) and isinstance(go_value, (int, float)):
                abs_diff = abs(float(zig_value) - float(go_value))
                denom = max(abs(float(zig_value)), abs(float(go_value)), 1e-12)
                comparisons[name] = {
                    "zig": zig_value,
                    "go": go_value,
                    "abs_diff": abs_diff,
                    "rel_diff": abs_diff / denom,
                }
        summary["go_zig_comparisons"] = comparisons
        zig_lora_matrix = (zig_apply_update or {}).get("lora_grad_matrix_stats") or {}
        go_lora_matrix = gn.get("lora_grad_matrix_stats") or {}
        lora_matrix_comparisons = {}
        for key in sorted(set(zig_lora_matrix.keys()) | set(go_lora_matrix.keys())):
            cmp = compare_numeric_value(
                (zig_lora_matrix.get(key) or {}).get("l2"),
                (go_lora_matrix.get(key) or {}).get("l2"),
            )
            if cmp is not None:
                lora_matrix_comparisons[key] = cmp
        summary["lora_grad_matrix_comparisons"] = lora_matrix_comparisons
        summary["lora_grad_matrix_worst"] = [
            {"name": name, **cmp}
            for name, cmp in sorted(
                lora_matrix_comparisons.items(),
                key=lambda item: item[1].get("rel_diff", 0.0),
                reverse=True,
            )[:20]
        ]
        zig_lora_matrix_post = (zig_apply_update or {}).get("lora_grad_matrix_stats_post_clip") or zig_lora_matrix
        go_lora_matrix_post = gn.get("lora_grad_matrix_stats_post_clip") or go_lora_matrix
        zig_lora_matrix_pre = (zig_apply_update or {}).get("lora_grad_matrix_stats_pre_clip") or {}
        go_lora_matrix_pre = gn.get("lora_grad_matrix_stats_pre_clip") or {}
        post_sample_comparisons = compare_lora_stat_maps(zig_lora_matrix_post, go_lora_matrix_post)
        pre_sample_comparisons = compare_lora_stat_maps(zig_lora_matrix_pre, go_lora_matrix_pre)
        summary["lora_grad_matrix_post_clip_comparisons"] = post_sample_comparisons
        summary["lora_grad_matrix_pre_clip_comparisons"] = pre_sample_comparisons
        summary["lora_grad_matrix_sample_comparisons"] = post_sample_comparisons
        summary["lora_grad_matrix_sample_worst"] = worst_lora_sample_comparisons(post_sample_comparisons)
        summary["lora_grad_matrix_pre_clip_sample_worst"] = worst_lora_sample_comparisons(pre_sample_comparisons)
        zig_update_matrix_all = zu.get("lora_update_matrix_stats") or {}
        zig_update_matrix = {
            key: (value or {}).get("update") or {}
            for key, value in zig_update_matrix_all.items()
        } if isinstance(zig_update_matrix_all, dict) else {}
        zig_adam_m_matrix = {
            key: (value or {}).get("adam_m") or {}
            for key, value in zig_update_matrix_all.items()
        } if isinstance(zig_update_matrix_all, dict) else {}
        zig_adam_v_matrix = {
            key: (value or {}).get("adam_v") or {}
            for key, value in zig_update_matrix_all.items()
        } if isinstance(zig_update_matrix_all, dict) else {}
        go_update_matrix = gu.get("lora_update_matrix_stats") or {}
        go_adam_m_matrix = gu.get("adam_m_lora_matrix_stats") or {}
        go_adam_v_matrix = gu.get("adam_v_lora_matrix_stats") or {}
        lora_update_matrix_comparisons = compare_lora_stat_maps(zig_update_matrix, go_update_matrix)
        adam_m_matrix_comparisons = compare_lora_stat_maps(zig_adam_m_matrix, go_adam_m_matrix)
        adam_v_matrix_comparisons = compare_lora_stat_maps(zig_adam_v_matrix, go_adam_v_matrix)
        summary["lora_update_matrix_comparisons"] = lora_update_matrix_comparisons
        summary["adam_m_lora_matrix_comparisons"] = adam_m_matrix_comparisons
        summary["adam_v_lora_matrix_comparisons"] = adam_v_matrix_comparisons
        summary["lora_update_matrix_worst"] = worst_lora_sample_comparisons(lora_update_matrix_comparisons)
        summary["adam_m_lora_matrix_worst"] = worst_lora_sample_comparisons(adam_m_matrix_comparisons)
        summary["adam_v_lora_matrix_worst"] = worst_lora_sample_comparisons(adam_v_matrix_comparisons)
        summary["optimizer_update_signal"] = optimizer_update_signal(
            comparisons,
            pre_sample_comparisons,
            lora_update_matrix_comparisons,
            adam_m_matrix_comparisons,
            adam_v_matrix_comparisons,
        )
        summary["training_parity_next_target_signal"] = training_parity_next_target_signal(summary)

with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")

print(json.dumps(summary, indent=2, sort_keys=True))
if summary["status"] == "failed":
    sys.exit(1)
PYEOF

echo "fused chunker frozen-step parity summary=$summary_json"
