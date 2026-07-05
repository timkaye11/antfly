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
# shellcheck source=scripts/inference_cli.sh
source "$SCRIPT_DIR/inference_cli.sh"

ANTFLY_BIN="$(resolve_antfly_inference_bin)"
MODEL_DIR="${ANTFLY_INFERENCE_GEMMA4_MODEL:-$HOME/.antfly/inference/models/ggml-org/gemma-4-e2b-it-gguf}"
PROMPT="${ANTFLY_INFERENCE_GEMMA4_PREFILL_PROMPT:-hi}"
MAX_TOKENS="${ANTFLY_INFERENCE_GEMMA4_PREFILL_MAX_TOKENS:-4}"
EXPECTED_TOKEN_IDS="${ANTFLY_INFERENCE_GEMMA4_EXPECTED_TOKEN_IDS:-}"
MIN_GENERATED_Q8_0_SMALL_BATCH="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q8_0_SMALL_BATCH:-0}"
MIN_GENERATED_Q4_0_SMALL_BATCH="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q4_0_SMALL_BATCH:-0}"
MIN_GENERATED_Q4_SMALL_BATCH="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q4_SMALL_BATCH:-0}"
MIN_GENERATED_Q5_SMALL_BATCH="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q5_SMALL_BATCH:-0}"
MIN_GENERATED_Q6_SMALL_BATCH="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q6_SMALL_BATCH:-0}"
MIN_GENERATED_COUNTERS="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_COUNTERS:-}"
JSON_TIMING="${ANTFLY_INFERENCE_GEMMA4_JSON_TIMING:-1}"
RAW_PROMPT="${ANTFLY_INFERENCE_GEMMA4_PREFILL_RAW_PROMPT:-0}"
OUT_DIR="${OUT_DIR:-/tmp/antfly-inference-metal-gemma4-prefill-frame-test}"
SELF_TEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --antfly-bin)
      if [[ -z "${2:-}" ]]; then
        echo "--antfly-bin requires a path" >&2
        exit 2
      fi
      ANTFLY_BIN="$2"
      shift 2
      ;;
    --generated-q8-smoke)
      MAX_TOKENS="${ANTFLY_INFERENCE_GEMMA4_PREFILL_MAX_TOKENS:-2}"
      RAW_PROMPT="${ANTFLY_INFERENCE_GEMMA4_PREFILL_RAW_PROMPT:-1}"
      MIN_GENERATED_Q8_0_SMALL_BATCH="${ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q8_0_SMALL_BATCH:-1}"
      if [[ -z "$MIN_GENERATED_COUNTERS" ]]; then
        MIN_GENERATED_COUNTERS="q8_0_small_batch=1"
      fi
      shift
      ;;
    --e4b-smoke)
      MODEL_DIR="${ANTFLY_INFERENCE_GEMMA4_MODEL:-$HOME/.antfly/inference/models/google/gemma-4-E4B-it-qat-q4_0-gguf}"
      MAX_TOKENS="${ANTFLY_INFERENCE_GEMMA4_PREFILL_MAX_TOKENS:-2}"
      RAW_PROMPT="${ANTFLY_INFERENCE_GEMMA4_PREFILL_RAW_PROMPT:-1}"
      shift
      ;;
    --self-test)
      SELF_TEST=1
      shift
      ;;
    *)
      echo "unknown Metal Gemma4 prefill-frame smoke argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$ANTFLY_BIN" in
  /*|"") ;;
  */*) ANTFLY_BIN="$(pwd)/$ANTFLY_BIN" ;;
esac

if (( MIN_GENERATED_Q8_0_SMALL_BATCH > 0 )); then
  export TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH="${TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH:-1}"
fi
if (( MIN_GENERATED_Q4_0_SMALL_BATCH > 0 )); then
  export TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH="${TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH:-1}"
fi
if (( MIN_GENERATED_Q4_SMALL_BATCH > 0 )); then
  export TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH="${TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH:-1}"
fi
if (( MIN_GENERATED_Q6_SMALL_BATCH > 0 )); then
  unset TERMITE_METAL_DISABLE_ANTFLY_Q6_K_SMALL_BATCH
fi

run_case() {
  local label="$1"
  shift
  local out="$OUT_DIR/${label}.txt"
  local json="$OUT_DIR/${label}.json"
  echo "running $label..." >&2
  local cmd=("$ANTFLY_BIN")
  if [[ "$(basename "$ANTFLY_BIN")" == "antfly" ]]; then
    cmd+=(inference)
  fi
  cmd+=(
    generate "$MODEL_DIR" "$PROMPT"
    --backend metal
    --max-tokens "$MAX_TOKENS"
    --print-token-ids
    --print-token-count
    --print-timing
  )
  if [[ "$JSON_TIMING" != "0" ]]; then
    cmd+=(--json-timing "$json")
  fi
  if [[ "$RAW_PROMPT" != "0" ]]; then
    cmd+=(--raw-prompt)
  fi
  set +e
  (
    cd "$ANTFLY_INFERENCE_ZIG_ROOT"
    "$@" "${cmd[@]}"
  ) >"$out" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "Metal Gemma4 prefill-frame case failed: label=$label exit=$rc output=$out" >&2
    sed -n '1,220p' "$out" >&2 || true
    exit "$rc"
  fi
  echo "$out"
}

token_ids_from() {
  local label="$1"
  local out="$2"
  local actual
  actual="$(awk '/^token_ids:/ { sub(/^token_ids:[[:space:]]*/, ""); print; exit }' "$out")"
  if [[ -z "$actual" ]]; then
    echo "missing token_ids for $label" >&2
    echo "output:   $out" >&2
    sed -n '1,220p' "$out" >&2
    exit 1
  fi
  printf '%s\n' "$actual"
}

json_counter_from() {
  local json="$1"
  local key="$2"
  awk -v needle="\"${key}\"" '
    index($0, needle) {
      value = $0
      sub(/^[^:]*:/, "", value)
      gsub(/[,[:space:]]/, "", value)
      if (value ~ /^[0-9]+$/) {
        print value
        exit
      }
    }
  ' "$json"
}

assert_json_counter_at_least() {
  local label="$1"
  local json="$2"
  local key="$3"
  local minimum="$4"
  local actual
  actual="$(json_counter_from "$json" "$key")"
  if [[ -z "$actual" ]]; then
    echo "missing JSON timing counter $key for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  if (( actual < minimum )); then
    echo "JSON timing counter $key below gate for $label" >&2
    echo "expected at least: $minimum" >&2
    echo "actual: $actual" >&2
    echo "json: $json" >&2
    exit 1
  fi
}

assert_json_timing_anchor() {
  local label="$1"
  local out="$2"
  if [[ "$JSON_TIMING" == "0" ]]; then
    return
  fi
  local json="${out%.txt}.json"
  if [[ ! -s "$json" ]]; then
    echo "missing JSON timing output for $label" >&2
    echo "json: $json" >&2
    exit 1
  fi
  if ! grep -q '"backend":"metal"' "$json"; then
    echo "JSON timing did not record the Metal backend for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  if ! grep -q '"metal":{' "$json"; then
    echo "JSON timing did not include Metal counters for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  if ! grep -q '"generated_quant_dispatch":{' "$json"; then
    echo "JSON timing did not include generated quant dispatch counters for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  if ! grep -q '"quant_kernel_plan":{' "$json"; then
    echo "JSON timing did not include quant kernel plan counters for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  assert_json_counter_at_least "$label" "$json" planned 1
  if [[ "$(json_counter_from "$json" fallback)" != "0" ]]; then
    echo "JSON timing reported Metal command fallback for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  if [[ "$(json_counter_from "$json" decode_fallback)" != "0" ]]; then
    echo "JSON timing reported Metal decode fallback for $label" >&2
    echo "json: $json" >&2
    sed -n '1,220p' "$json" >&2 || true
    exit 1
  fi
  if (( MIN_GENERATED_Q8_0_SMALL_BATCH > 0 )); then
    assert_json_counter_at_least "$label" "$json" q8_0_small_batch "$MIN_GENERATED_Q8_0_SMALL_BATCH"
  fi
  if (( MIN_GENERATED_Q4_0_SMALL_BATCH > 0 )); then
    assert_json_counter_at_least "$label" "$json" q4_0_small_batch "$MIN_GENERATED_Q4_0_SMALL_BATCH"
  fi
  if (( MIN_GENERATED_Q4_SMALL_BATCH > 0 )); then
    assert_json_counter_at_least "$label" "$json" q4_k_small_batch "$MIN_GENERATED_Q4_SMALL_BATCH"
  fi
  if (( MIN_GENERATED_Q5_SMALL_BATCH > 0 )); then
    assert_json_counter_at_least "$label" "$json" q5_k_small_batch "$MIN_GENERATED_Q5_SMALL_BATCH"
  fi
  if (( MIN_GENERATED_Q6_SMALL_BATCH > 0 )); then
    assert_json_counter_at_least "$label" "$json" q6_k_small_batch "$MIN_GENERATED_Q6_SMALL_BATCH"
  fi

  local generated_counter_gate generated_counter_key generated_counter_min
  for generated_counter_gate in ${MIN_GENERATED_COUNTERS//,/ }; do
    [[ -z "$generated_counter_gate" ]] && continue
    generated_counter_key="${generated_counter_gate%%=*}"
    generated_counter_min="${generated_counter_gate#*=}"
    assert_json_counter_at_least "$label" "$json" "$generated_counter_key" "$generated_counter_min"
  done
}

assert_anchor() {
  local label="$1"
  local out="$2"
  local actual
  actual="$(token_ids_from "$label" "$out")"
  if [[ "$actual" != "$EXPECTED_TOKEN_IDS" ]]; then
    echo "unexpected token_ids for $label" >&2
    echo "expected: $EXPECTED_TOKEN_IDS" >&2
    echo "actual:   ${actual:-<missing>}" >&2
    echo "output:   $out" >&2
    sed -n '1,220p' "$out" >&2
    exit 1
  fi

  if ! grep -q '^metal_decoder_frame:' "$out"; then
    echo "missing metal_decoder_frame counters for $label" >&2
    echo "output: $out" >&2
    exit 1
  fi

  if ! grep -q 'prefill_direct_family=[1-9]' "$out"; then
    echo "prefill direct family path was not exercised for $label" >&2
    echo "output: $out" >&2
    exit 1
  fi

  if grep -q 'decode_fallback=[1-9]' "$out"; then
    echo "Metal frame decode fallback was exercised for $label" >&2
    echo "output: $out" >&2
    exit 1
  fi

  if ! grep -q 'metal_runtime_command_operators: fallback=0' "$out"; then
    echo "Metal runtime command operators used fallback for $label" >&2
    echo "output: $out" >&2
    exit 1
  fi

  generated_dispatch_count() {
    local key="$1"
    awk -v needle="${key}=" '
    /^metal_generated_quant_dispatch:/ {
      for (i = 1; i <= NF; i++) {
        if ($i == needle || index($i, needle) == 1) {
          split($i, a, "=")
          value = a[2] + 0
          if (!found || value > max) {
            max = value
            found = 1
          }
        }
      }
    }
    END {
      if (found) print max
    }
  ' "$out"
  }

  local gen_q8_0_small_batch gen_q4_0_small_batch gen_q4_small_batch gen_q5_small_batch gen_q6_small_batch
  gen_q8_0_small_batch="$(generated_dispatch_count q8_0_small_batch)"
  gen_q4_0_small_batch="$(generated_dispatch_count q4_0_small_batch)"
  gen_q4_small_batch="$(generated_dispatch_count q4_k_small_batch)"
  gen_q5_small_batch="$(generated_dispatch_count q5_k_small_batch)"
  gen_q6_small_batch="$(generated_dispatch_count q6_k_small_batch)"
  if [[ -z "$gen_q8_0_small_batch" || -z "$gen_q4_0_small_batch" || -z "$gen_q4_small_batch" || -z "$gen_q5_small_batch" || -z "$gen_q6_small_batch" ]]; then
    echo "missing generated quant dispatch counters for $label" >&2
    echo "output: $out" >&2
    exit 1
  fi
  if (( MIN_GENERATED_Q8_0_SMALL_BATCH > 0 && gen_q8_0_small_batch < MIN_GENERATED_Q8_0_SMALL_BATCH )); then
    echo "generated Q8_0 small-batch dispatch below gate for $label" >&2
    echo "expected at least: $MIN_GENERATED_Q8_0_SMALL_BATCH" >&2
    echo "actual: ${gen_q8_0_small_batch:-<missing>}" >&2
    echo "output: $out" >&2
    exit 1
  fi
  if (( MIN_GENERATED_Q4_0_SMALL_BATCH > 0 && gen_q4_0_small_batch < MIN_GENERATED_Q4_0_SMALL_BATCH )); then
    echo "generated Q4_0 small-batch dispatch below gate for $label" >&2
    echo "expected at least: $MIN_GENERATED_Q4_0_SMALL_BATCH" >&2
    echo "actual: ${gen_q4_0_small_batch:-<missing>}" >&2
    echo "output: $out" >&2
    exit 1
  fi
  if (( MIN_GENERATED_Q4_SMALL_BATCH > 0 && gen_q4_small_batch < MIN_GENERATED_Q4_SMALL_BATCH )); then
    echo "generated Q4_K small-batch dispatch below gate for $label" >&2
    echo "expected at least: $MIN_GENERATED_Q4_SMALL_BATCH" >&2
    echo "actual: ${gen_q4_small_batch:-<missing>}" >&2
    echo "output: $out" >&2
    exit 1
  fi
  if (( MIN_GENERATED_Q5_SMALL_BATCH > 0 && gen_q5_small_batch < MIN_GENERATED_Q5_SMALL_BATCH )); then
    echo "generated Q5_K small-batch dispatch below gate for $label" >&2
    echo "expected at least: $MIN_GENERATED_Q5_SMALL_BATCH" >&2
    echo "actual: ${gen_q5_small_batch:-<missing>}" >&2
    echo "output: $out" >&2
    exit 1
  fi
  if (( MIN_GENERATED_Q6_SMALL_BATCH > 0 && gen_q6_small_batch < MIN_GENERATED_Q6_SMALL_BATCH )); then
    echo "generated Q6_K small-batch dispatch below gate for $label" >&2
    echo "expected at least: $MIN_GENERATED_Q6_SMALL_BATCH" >&2
    echo "actual: ${gen_q6_small_batch:-<missing>}" >&2
    echo "output: $out" >&2
    exit 1
  fi
  local generated_counter_gate generated_counter_key generated_counter_min generated_counter_actual
  for generated_counter_gate in ${MIN_GENERATED_COUNTERS//,/ }; do
    [[ -z "$generated_counter_gate" ]] && continue
    if [[ "$generated_counter_gate" != *=* || "$generated_counter_gate" == =* ]]; then
      echo "invalid generated quant counter gate: $generated_counter_gate" >&2
      echo "expected format: counter_name=min_count" >&2
      exit 2
    fi
    generated_counter_key="${generated_counter_gate%%=*}"
    generated_counter_min="${generated_counter_gate#*=}"
    if [[ ! "$generated_counter_min" =~ ^[0-9]+$ ]]; then
      echo "invalid generated quant counter minimum for $generated_counter_key: $generated_counter_min" >&2
      exit 2
    fi
    generated_counter_actual="$(generated_dispatch_count "$generated_counter_key")"
    if [[ -z "$generated_counter_actual" ]]; then
      echo "missing generated quant dispatch counter $generated_counter_key for $label" >&2
      echo "output: $out" >&2
      exit 1
    fi
    if [[ ! "$generated_counter_actual" =~ ^[0-9]+$ ]]; then
      echo "generated quant dispatch counter $generated_counter_key was not numeric for $label" >&2
      echo "actual: $generated_counter_actual" >&2
      echo "output: $out" >&2
      exit 1
    fi
    if (( generated_counter_actual < generated_counter_min )); then
      echo "generated $generated_counter_key dispatch below gate for $label" >&2
      echo "expected at least: $generated_counter_min" >&2
      echo "actual: $generated_counter_actual" >&2
      echo "output: $out" >&2
      exit 1
    fi
  done

  assert_json_timing_anchor "$label" "$out"

  if ! grep -q 'attn_out_linear=0 attn_post_norm=0 attn_residual_add=0' "$out"; then
    echo "attention residual path fell back to split prefill ops for $label" >&2
    echo "output: $out" >&2
    exit 1
  fi
}

if [[ "$SELF_TEST" == "1" ]]; then
  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/antfly-prefill-frame-self-test.XXXXXX")"
  trap 'rm -rf "$tmp_dir"' EXIT
  JSON_TIMING=1
  sample="$tmp_dir/sample.txt"
  cat >"$sample" <<'OUT'
token_ids: 1
metal_decoder_frame: begins=1 submits=1 wait_ms=0 gpu_ms=0
metal_executor_ms: prefill_direct_family=1
metal_runtime_command_operators: fallback=0
metal_generated_quant_dispatch: q8_0_small_batch=1 q8_0_small_batch_bias_gelu=2 q2_k_small_batch=3 q4_0_small_batch=1 q4_k_small_batch=1 q5_k_small_batch=1 q6_k_small_batch=1
metal_generated_quant_dispatch: q8_0_small_batch=2 q8_0_small_batch_bias_gelu=3 q2_k_small_batch=4 q4_0_small_batch=1 q4_k_small_batch=1 q5_k_small_batch=1 q6_k_small_batch=1
decoder_gated_prefill_ops: tokens=3 attn_out_linear=0 attn_post_norm=0 attn_residual_add=0
metal_frame_fallbacks: decode_fallback=0
OUT
  cat >"${sample%.txt}.json" <<'JSON'
{
"backend":"metal",
"metal":{
"runtime_command_operators":{
"fallback":0
},
"generated_quant_dispatch":{
"q8_0_small_batch":2,
"q8_0_small_batch_bias_gelu":3,
"q2_k_small_batch":4,
"q4_0_small_batch":1,
"q4_k_small_batch":1,
"q5_k_small_batch":1,
"q6_k_small_batch":1
},
"quant_kernel_plan":{
"planned":5
},
"frame_fallbacks":{
"decode_fallback":0
}
}
}
JSON
  EXPECTED_TOKEN_IDS=1
  MIN_GENERATED_Q8_0_SMALL_BATCH=1
  MIN_GENERATED_Q4_0_SMALL_BATCH=1
  MIN_GENERATED_Q4_SMALL_BATCH=1
  MIN_GENERATED_Q5_SMALL_BATCH=1
  MIN_GENERATED_Q6_SMALL_BATCH=1
  assert_anchor self-test-pass "$sample"
  MIN_GENERATED_Q8_0_SMALL_BATCH=3
  if ( assert_anchor self-test-q8-0-fail "$sample" ) 2>"$tmp_dir/q8-0-fail.err"; then
    echo "expected generated Q8_0 small-batch gate failure" >&2
    exit 1
  fi
  if ! grep -q 'generated Q8_0 small-batch dispatch below gate' "$tmp_dir/q8-0-fail.err"; then
    cat "$tmp_dir/q8-0-fail.err" >&2
    exit 1
  fi
  MIN_GENERATED_Q8_0_SMALL_BATCH=1
  MIN_GENERATED_Q4_0_SMALL_BATCH=2
  if ( assert_anchor self-test-q4-0-fail "$sample" ) 2>"$tmp_dir/q4-0-fail.err"; then
    echo "expected generated Q4_0 small-batch gate failure" >&2
    exit 1
  fi
  if ! grep -q 'generated Q4_0 small-batch dispatch below gate' "$tmp_dir/q4-0-fail.err"; then
    cat "$tmp_dir/q4-0-fail.err" >&2
    exit 1
  fi
  MIN_GENERATED_Q4_0_SMALL_BATCH=1
  MIN_GENERATED_Q4_SMALL_BATCH=2
  if ( assert_anchor self-test-q4-fail "$sample" ) 2>"$tmp_dir/q4-fail.err"; then
    echo "expected generated Q4_K small-batch gate failure" >&2
    exit 1
  fi
  if ! grep -q 'generated Q4_K small-batch dispatch below gate' "$tmp_dir/q4-fail.err"; then
    cat "$tmp_dir/q4-fail.err" >&2
    exit 1
  fi
  MIN_GENERATED_Q4_SMALL_BATCH=1
  MIN_GENERATED_Q5_SMALL_BATCH=2
  if ( assert_anchor self-test-q5-fail "$sample" ) 2>"$tmp_dir/q5-fail.err"; then
    echo "expected generated Q5_K small-batch gate failure" >&2
    exit 1
  fi
  if ! grep -q 'generated Q5_K small-batch dispatch below gate' "$tmp_dir/q5-fail.err"; then
    cat "$tmp_dir/q5-fail.err" >&2
    exit 1
  fi
  MIN_GENERATED_Q5_SMALL_BATCH=1
  MIN_GENERATED_Q6_SMALL_BATCH=2
  if ( assert_anchor self-test-q6-fail "$sample" ) 2>"$tmp_dir/q6-fail.err"; then
    echo "expected generated Q6_K small-batch gate failure" >&2
    exit 1
  fi
  if ! grep -q 'generated Q6_K small-batch dispatch below gate' "$tmp_dir/q6-fail.err"; then
    cat "$tmp_dir/q6-fail.err" >&2
    exit 1
  fi
  MIN_GENERATED_Q6_SMALL_BATCH=1
  MIN_GENERATED_COUNTERS="q8_0_small_batch_bias_gelu=3,q2_k_small_batch=4"
  assert_anchor self-test-generic-pass "$sample"
  MIN_GENERATED_COUNTERS="q8_0_small_batch_bias_gelu=4"
  if ( assert_anchor self-test-generic-low-fail "$sample" ) 2>"$tmp_dir/generic-low-fail.err"; then
    echo "expected generated counter gate failure" >&2
    exit 1
  fi
  if ! grep -q 'generated q8_0_small_batch_bias_gelu dispatch below gate' "$tmp_dir/generic-low-fail.err"; then
    cat "$tmp_dir/generic-low-fail.err" >&2
    exit 1
  fi
  MIN_GENERATED_COUNTERS="missing_counter=1"
  if ( assert_anchor self-test-generic-missing-fail "$sample" ) 2>"$tmp_dir/generic-missing-fail.err"; then
    echo "expected generated counter missing failure" >&2
    exit 1
  fi
  if ! grep -q 'missing generated quant dispatch counter missing_counter' "$tmp_dir/generic-missing-fail.err"; then
    cat "$tmp_dir/generic-missing-fail.err" >&2
    exit 1
  fi
  echo "metal Gemma4 prefill-frame script self-test passed"
  exit 0
fi

if [[ ! -x "$ANTFLY_BIN" ]]; then
  echo "antfly binary not executable: $ANTFLY_BIN" >&2
  echo "build it first, for example: cd pkg/inference && zig build -Doptimize=ReleaseFast -Dmetal=true -Donnx=false -Dpjrt=false" >&2
  exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Gemma4 model directory not found: $MODEL_DIR" >&2
  echo "set ANTFLY_INFERENCE_GEMMA4_MODEL to the local GGUF model directory" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

default_out="$(run_case default env)" || exit $?
if [[ -z "$EXPECTED_TOKEN_IDS" ]]; then
  EXPECTED_TOKEN_IDS="$(token_ids_from default "$default_out")"
fi
assert_anchor default "$default_out"

sync_out="$(run_case stage-sync env TERMITE_METAL_SYNC_GATED_FAMILY_STAGES=1)" || exit $?
assert_anchor stage-sync "$sync_out"

echo "metal Gemma4 prefill-frame no-fallback/stage-sync smoke passed"
echo "default:    $default_out"
echo "stage-sync: $sync_out"
