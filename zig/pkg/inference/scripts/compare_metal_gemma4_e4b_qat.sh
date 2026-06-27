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
source "$SCRIPT_DIR/inference_cli.sh"
ROOT_OUT_DIR="${OUT_DIR:-/tmp/antfly-inference-gemma4-e4b-qat-compare-$(date -u +%Y%m%d-%H%M%S)}"
DEFAULT_MODELS_DIR="${ANTFLY_INFERENCE_GEMMA4_MODELS_DIR:-$HOME/.antfly/inference/models}"
LEGACY_MODELS_DIR="${ANTFLY_INFERENCE_GEMMA4_LEGACY_MODELS_DIR:-/private/tmp/antfly-inference-models}"
RUNS="${ANTFLY_INFERENCE_GEMMA4_COMPARE_RUNS:-3}"
TOKENS="${ANTFLY_INFERENCE_GEMMA4_COMPARE_TOKENS:-256}"
MIN_SPEEDUP="${ANTFLY_INFERENCE_GEMMA4_COMPARE_MIN_SPEEDUP:-1.10}"
PROMPT="${ANTFLY_INFERENCE_GEMMA4_COMPARE_PROMPT:-Write a numbered list from 1 to 300. Each item must be exactly: local inference benchmark continues.}"
RUN_QAT_ORACLE="${ANTFLY_INFERENCE_GEMMA4_QAT_ORACLE:-1}"
ORACLE_PROMPT="${ANTFLY_INFERENCE_GEMMA4_QAT_ORACLE_PROMPT:-Write one short paragraph about local inference.}"
ORACLE_TOKENS="${ANTFLY_INFERENCE_GEMMA4_QAT_ORACLE_TOKENS:-8}"
ORACLE_RENDERED_PROMPT="${ANTFLY_INFERENCE_GEMMA4_QAT_ORACLE_RENDERED_PROMPT:-}"
if [[ -z "$ORACLE_RENDERED_PROMPT" ]]; then
  printf -v ORACLE_RENDERED_PROMPT '<|turn>user\n%s<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' "$ORACLE_PROMPT"
fi
ORACLE_EXPECTED="${ANTFLY_INFERENCE_GEMMA4_QAT_ORACLE_EXPECTED:-}"
if [[ -z "$ORACLE_EXPECTED" ]]; then
  ORACLE_EXPECTED="Here's a thinking"
fi
DRAFT_MODEL="${ANTFLY_INFERENCE_GEMMA4_COMPARE_DRAFT_MODEL:-${ANTFLY_INFERENCE_GEMMA4_DRAFT_MODEL:-${ANTFLY_INFERENCE_GEMMA4_E4B_DRAFT_MODEL:-}}}"
MTP_POLICY_DRAFT_MODEL="${ANTFLY_INFERENCE_GEMMA4_MTP_POLICY_DRAFT_MODEL:-$DRAFT_MODEL}"
RUN_MTP_POLICY_CHECK="${ANTFLY_INFERENCE_GEMMA4_QAT_MTP_POLICY_CHECK:-1}"
MIN_HOT_DECODE_TOK_S="${ANTFLY_INFERENCE_GEMMA4_MIN_HOT_DECODE_TOK_S:-1}"
MIN_QAT_PLE_ACTIVATION_RHS_F16="${ANTFLY_INFERENCE_GEMMA4_COMPARE_MIN_QAT_PLE_ACTIVATION_RHS_F16:-${ANTFLY_INFERENCE_GEMMA4_MIN_Q4_0_PLE_ACTIVATION_RHS_REDUCE_OUT_F16:-1}}"
MIN_QAT_PLE_LINEAR_F16="${ANTFLY_INFERENCE_GEMMA4_COMPARE_MIN_QAT_PLE_LINEAR_F16:-${ANTFLY_INFERENCE_GEMMA4_MIN_Q4_0_PLE_LINEAR_REDUCE_IN_F16:-1}}"

mkdir -p "$ROOT_OUT_DIR"

resolve_text_gguf() {
  local model="$1"
  if [[ -f "$model" ]]; then
    printf '%s\n' "$model"
    return
  fi
  find "$model" -maxdepth 1 -type f -name '*.gguf' ! -name '*mmproj*' | sort | head -n 1
}

run_qat_oracle() {
  local model="$1"
  if [[ "$RUN_QAT_ORACLE" == "0" ]]; then
    return
  fi
  if ! command -v llama-completion >/dev/null 2>&1; then
    echo "Gemma4 QAT oracle needs llama-completion; set ANTFLY_INFERENCE_GEMMA4_QAT_ORACLE=0 to skip" >&2
    exit 2
  fi
  local gguf
  gguf="$(resolve_text_gguf "$model")"
  if [[ -z "$gguf" ]]; then
    echo "Gemma4 QAT oracle could not find text GGUF under: $model" >&2
    exit 2
  fi
  local out="$ROOT_OUT_DIR/qat-oracle"
  mkdir -p "$out"
  local antfly_out="$out/antfly.txt"
  local llama_out="$out/llama.txt"
  run_antfly_inference generate "$model" "$ORACLE_RENDERED_PROMPT" \
    --backend metal \
    --max-tokens "$ORACLE_TOKENS" \
    --temperature 0 \
    --raw-prompt \
    --print-token-count \
    --print-finish-reason \
    --print-token-ids \
    >"$antfly_out" 2>&1
  llama-completion -m "$gguf" \
    --no-conversation \
    --no-jinja \
    --special \
    -p "$ORACLE_RENDERED_PROMPT" \
    -n "$ORACLE_TOKENS" \
    --temp 0 \
    --no-display-prompt \
    --no-perf \
    >"$llama_out" 2>&1
  if ! grep -Fq "$ORACLE_EXPECTED" "$llama_out"; then
    echo "Gemma4 QAT oracle reference did not contain expected text: $ORACLE_EXPECTED" >&2
    echo "llama output: $llama_out" >&2
    exit 1
  fi
  if ! grep -Fq "$ORACLE_EXPECTED" "$antfly_out"; then
    echo "Gemma4 QAT Metal output failed oracle text check: expected $ORACLE_EXPECTED" >&2
    echo "antfly output: $antfly_out" >&2
    echo "llama output: $llama_out" >&2
    exit 1
  fi
}

token_ids_from_output() {
  awk '/^token_ids:/ { sub(/^token_ids:[[:space:]]*/, ""); print; exit }' "$1"
}

run_mtp_policy_check() {
  local model="$1"
  local draft="$2"
  if [[ "$RUN_MTP_POLICY_CHECK" == "0" || -z "$draft" || ! -e "$draft" ]]; then
    return
  fi
  local out="$ROOT_OUT_DIR/qat-mtp-policy"
  mkdir -p "$out"
  local target_out="$out/target.txt"
  local auto_out="$out/auto.txt"
  local force_out="$out/force.txt"
  local force_k1_out="$out/force-k1.txt"
  run_antfly_inference generate "$model" "$ORACLE_RENDERED_PROMPT" \
    --backend metal \
    --max-tokens "$ORACLE_TOKENS" \
    --temperature 0 \
    --raw-prompt \
    --print-token-count \
    --print-finish-reason \
    --print-token-ids \
    >"$target_out" 2>&1
  ANTFLY_GEMMA4_MTP_AUTO_MIN_TOKENS=1 \
  ANTFLY_GEMMA4_MTP_ENABLE_METAL_AUTO=0 \
  ANTFLY_GEMMA4_MTP_PROFILE=1 \
  run_antfly_inference generate "$model" "$ORACLE_RENDERED_PROMPT" \
    --backend metal \
    --draft-model "$draft" \
    --speculative-k "${ANTFLY_INFERENCE_GEMMA4_COMPARE_SPECULATIVE_K:-${ANTFLY_INFERENCE_GEMMA4_SPECULATIVE_K:-2}}" \
    --speculation-policy auto \
    --speculation-calibration positive \
    --max-tokens "$ORACLE_TOKENS" \
    --temperature 0 \
    --raw-prompt \
    --print-token-count \
    --print-finish-reason \
    --print-token-ids \
    --print-timing \
    >"$auto_out" 2>&1
  local target_ids auto_ids
  target_ids="$(token_ids_from_output "$target_out")"
  auto_ids="$(token_ids_from_output "$auto_out")"
  if [[ -z "$target_ids" || -z "$auto_ids" || "$target_ids" != "$auto_ids" ]]; then
    echo "Gemma4 QAT MTP auto policy changed token IDs" >&2
    echo "target: ${target_ids:-<missing>} ($target_out)" >&2
    echo "auto:   ${auto_ids:-<missing>} ($auto_out)" >&2
    exit 1
  fi
  if grep -Eq '^speculative: .*mtp_enabled=true' "$auto_out"; then
    echo "Gemma4 QAT Metal MTP auto enabled without explicit opt-in: $auto_out" >&2
    exit 1
  fi
  ANTFLY_GEMMA4_MTP_PROFILE=1 \
  run_antfly_inference generate "$model" "$ORACLE_RENDERED_PROMPT" \
    --backend metal \
    --draft-model "$draft" \
    --speculative-k "${ANTFLY_INFERENCE_GEMMA4_COMPARE_SPECULATIVE_K:-${ANTFLY_INFERENCE_GEMMA4_SPECULATIVE_K:-2}}" \
    --speculation-policy force \
    --max-tokens "$ORACLE_TOKENS" \
    --temperature 0 \
    --raw-prompt \
    --print-token-count \
    --print-finish-reason \
    --print-token-ids \
    --print-timing \
    >"$force_out" 2>&1
  local force_ids
  force_ids="$(token_ids_from_output "$force_out")"
  if [[ -z "$force_ids" || "$target_ids" != "$force_ids" ]]; then
    echo "Gemma4 QAT forced MTP changed token IDs" >&2
    echo "target: ${target_ids:-<missing>} ($target_out)" >&2
    echo "force:  ${force_ids:-<missing>} ($force_out)" >&2
    exit 1
  fi
  if ! grep -Eq '^speculative: .*mtp_enabled=true' "$force_out"; then
    echo "Gemma4 QAT forced MTP did not enable MTP: $force_out" >&2
    exit 1
  fi
  ANTFLY_GEMMA4_MTP_PROFILE=1 \
  run_antfly_inference generate "$model" "$ORACLE_RENDERED_PROMPT" \
    --backend metal \
    --draft-model "$draft" \
    --speculative-k 1 \
    --speculation-policy force \
    --max-tokens "$ORACLE_TOKENS" \
    --temperature 0 \
    --raw-prompt \
    --print-token-count \
    --print-finish-reason \
    --print-token-ids \
    --print-timing \
    >"$force_k1_out" 2>&1
  local force_k1_ids
  force_k1_ids="$(token_ids_from_output "$force_k1_out")"
  if [[ -z "$force_k1_ids" || "$target_ids" != "$force_k1_ids" ]]; then
    echo "Gemma4 QAT forced MTP k=1 changed token IDs" >&2
    echo "target:   ${target_ids:-<missing>} ($target_out)" >&2
    echo "force k1: ${force_k1_ids:-<missing>} ($force_k1_out)" >&2
    exit 1
  fi
  if ! grep -Eq '^speculative: .*mtp_enabled=true' "$force_k1_out"; then
    echo "Gemma4 QAT forced MTP k=1 did not enable MTP: $force_k1_out" >&2
    exit 1
  fi
}

run_variant() {
  local variant="$1"
  local model="$2"
  local out="$3"
  local q4_pair_gate=0
  local q6_gate=0
  local q4_0_ple_activation_gate=0
  local q4_0_ple_linear_gate=0
  if [[ "$variant" == "standard" ]]; then
    q4_pair_gate="${ANTFLY_INFERENCE_GEMMA4_MIN_Q4_PAIR_ACT_REDUCE_OUT_F16:-1}"
    q6_gate="${ANTFLY_INFERENCE_GEMMA4_MIN_Q6_REDUCE_IN_F16:-1}"
  elif [[ "$variant" == "qat-q4_0" ]]; then
    q4_0_ple_activation_gate="$MIN_QAT_PLE_ACTIVATION_RHS_F16"
    q4_0_ple_linear_gate="$MIN_QAT_PLE_LINEAR_F16"
  fi
  env \
    ANTFLY_INFERENCE_GEMMA4_E4B_VARIANT="$variant" \
    ANTFLY_INFERENCE_GEMMA4_MODEL="$model" \
    ANTFLY_INFERENCE_GEMMA4_E4B_REQUIRE_FULL_TOKENS=1 \
    ANTFLY_INFERENCE_GEMMA4_BENCH_PROMPT="$PROMPT" \
    ANTFLY_INFERENCE_GEMMA4_BENCH_MAX_TOKENS="$TOKENS" \
    ANTFLY_INFERENCE_GEMMA4_BENCH_WARMUP_TOKENS="${ANTFLY_INFERENCE_GEMMA4_COMPARE_WARMUP_TOKENS:-16}" \
    ANTFLY_INFERENCE_GEMMA4_BENCH_RUNS="$RUNS" \
    ANTFLY_INFERENCE_GEMMA4_REUSE_PROBE="${ANTFLY_INFERENCE_GEMMA4_REUSE_PROBE:-0}" \
    ANTFLY_INFERENCE_GEMMA4_DRAFT_MODEL="" \
    ANTFLY_INFERENCE_GEMMA4_SPECULATIVE_K="${ANTFLY_INFERENCE_GEMMA4_COMPARE_SPECULATIVE_K:-${ANTFLY_INFERENCE_GEMMA4_SPECULATIVE_K:-4}}" \
    ANTFLY_INFERENCE_GEMMA4_SPECULATION_POLICY="${ANTFLY_INFERENCE_GEMMA4_COMPARE_SPECULATION_POLICY:-${ANTFLY_INFERENCE_GEMMA4_SPECULATION_POLICY:-auto}}" \
    ANTFLY_INFERENCE_GEMMA4_SPECULATION_CALIBRATION="${ANTFLY_INFERENCE_GEMMA4_COMPARE_SPECULATION_CALIBRATION:-${ANTFLY_INFERENCE_GEMMA4_SPECULATION_CALIBRATION:-positive}}" \
    ANTFLY_INFERENCE_GEMMA4_REQUIRE_MTP_ENABLED=0 \
    ANTFLY_INFERENCE_GEMMA4_MIN_SPECULATIVE_ROUNDS=0 \
    ANTFLY_INFERENCE_GEMMA4_MIN_SPECULATIVE_DRAFTED=0 \
    ANTFLY_INFERENCE_GEMMA4_MIN_SPECULATIVE_ACCEPTED=0 \
    ANTFLY_INFERENCE_GEMMA4_MIN_SPECULATIVE_MATCHED=0 \
    ANTFLY_INFERENCE_GEMMA4_MIN_Q4_PAIR_ACT_REDUCE_OUT_F16="$q4_pair_gate" \
    ANTFLY_INFERENCE_GEMMA4_MIN_Q6_REDUCE_IN_F16="$q6_gate" \
    ANTFLY_INFERENCE_GEMMA4_MIN_Q4_0_PLE_ACTIVATION_RHS_REDUCE_OUT_F16="$q4_0_ple_activation_gate" \
    ANTFLY_INFERENCE_GEMMA4_MIN_Q4_0_PLE_LINEAR_REDUCE_IN_F16="$q4_0_ple_linear_gate" \
    ANTFLY_INFERENCE_GEMMA4_MAX_Q4_0_LINEAR_REDUCE_SUMSQ=0 \
    ANTFLY_INFERENCE_GEMMA4_MAX_RMS_NORM_ADD_SUMSQ=0 \
    ANTFLY_INFERENCE_GEMMA4_MIN_DECODE_TOK_S="${ANTFLY_INFERENCE_GEMMA4_MIN_DECODE_TOK_S:-1}" \
    ANTFLY_INFERENCE_GEMMA4_MIN_HOT_DECODE_TOK_S="$MIN_HOT_DECODE_TOK_S" \
    TERMITE_METAL_DISABLE_Q4_0_LINEAR_RMS_ADD_SUMSQ=1 \
    TERMITE_METAL_ENABLE_Q4_0_LINEAR_RMS_ADD_SUMSQ=0 \
    TERMITE_METAL_Q4_0_LINEAR_RMS_ADD_SUMSQ_EXPERIMENT=0 \
    TERMITE_METAL_ENABLE_Q4_0_F16_FFN=0 \
    TERMITE_METAL_Q4_0_F16_FFN_EXPERIMENT=0 \
    ANTFLY_INFERENCE_GEMMA4_ALLOW_UNSAFE_Q4_0_F16_FFN=0 \
    TERMITE_METAL_Q4_0_LINEAR_RMS_ADD_F16_PROJECT=0 \
    TERMITE_METAL_Q4_0_LINEAR_RMS_ADD_F16_PROJECT_EXPERIMENT=0 \
    ANTFLY_GEMMA4_MTP_ENABLE_METAL_AUTO=0 \
    OUT_DIR="$out" \
    "$SCRIPT_DIR/bench_metal_gemma4_e4b.sh"
}

QAT_MODEL_DEFAULT="$DEFAULT_MODELS_DIR/google/gemma-4-E4B-it-qat-q4_0-gguf"
Q4K_MODEL_DEFAULT="$DEFAULT_MODELS_DIR/ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
[[ -e "$QAT_MODEL_DEFAULT" ]] || QAT_MODEL_DEFAULT="$LEGACY_MODELS_DIR/google/gemma-4-E4B-it-qat-q4_0-gguf"
[[ -e "$Q4K_MODEL_DEFAULT" ]] || Q4K_MODEL_DEFAULT="$LEGACY_MODELS_DIR/ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
QAT_MODEL="${ANTFLY_INFERENCE_GEMMA4_QAT_MODEL:-$QAT_MODEL_DEFAULT}"
Q4K_MODEL="${ANTFLY_INFERENCE_GEMMA4_Q4K_MODEL:-$Q4K_MODEL_DEFAULT}"
if [[ -z "$MTP_POLICY_DRAFT_MODEL" ]]; then
  candidate="${QAT_MODEL%-gguf}-unquantized-assistant"
  [[ -e "$candidate" ]] && MTP_POLICY_DRAFT_MODEL="$candidate"
fi

run_qat_oracle "$QAT_MODEL"
run_mtp_policy_check "$QAT_MODEL" "$MTP_POLICY_DRAFT_MODEL"
run_variant qat-q4_0 "$QAT_MODEL" "$ROOT_OUT_DIR/qat-q4_0"
run_variant standard "$Q4K_MODEL" "$ROOT_OUT_DIR/q4_k_m"

python3 - "$ROOT_OUT_DIR" "$MIN_SPEEDUP" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_speedup = float(sys.argv[2])
qat = json.loads((root / "qat-q4_0" / "summary.json").read_text())
q4k = json.loads((root / "q4_k_m" / "summary.json").read_text())
qat_decode = float(qat["median_decode_tok_s"])
q4k_decode = float(q4k["median_decode_tok_s"])
speedup = qat_decode / q4k_decode if q4k_decode else 0.0
qat_rows = qat["rows"]
qat_ple_activation_min = min(int(r.get("q4_0_ple_activation_rhs_reduce_out_f16", 0)) for r in qat_rows)
qat_ple_linear_min = min(int(r.get("q4_0_ple_linear_reduce_in_f16", 0)) for r in qat_rows)

def mtp_policy_summary(path: Path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    line = next((ln for ln in text.splitlines() if ln.startswith("speculative:")), "")
    return {
        "enabled": "mtp_enabled=true" in line,
        "accepted": int((re.search(r"\baccepted=(\d+)", line) or [None, 0])[1]),
        "drafted": int((re.search(r"\bdrafted=(\d+)", line) or [None, 0])[1]),
        "matched": int((re.search(r"\bmatched=(\d+)", line) or [None, 0])[1]),
    }

summary = {
    "qat_median_decode_tok_s": qat_decode,
    "q4k_median_decode_tok_s": q4k_decode,
    "speedup": speedup,
    "min_speedup": min_speedup,
    "qat_ple_activation_rhs_reduce_out_f16_min": qat_ple_activation_min,
    "qat_ple_linear_reduce_in_f16_min": qat_ple_linear_min,
    "mtp_force": mtp_policy_summary(root / "qat-mtp-policy" / "force.txt"),
    "mtp_force_k1": mtp_policy_summary(root / "qat-mtp-policy" / "force-k1.txt"),
    "qat_speculative_decision": qat["rows"][0].get("speculative_decision", ""),
    "q4k_speculative_decision": q4k["rows"][0].get("speculative_decision", ""),
    "qat_runtime_toggles": qat.get("runtime_toggles", {}),
    "q4k_runtime_toggles": q4k.get("runtime_toggles", {}),
    "qat_summary": str(root / "qat-q4_0" / "summary.tsv"),
    "q4k_summary": str(root / "q4_k_m" / "summary.tsv"),
}
(root / "compare-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"compare summary: {root / 'compare-summary.json'}")
print(f"qat_decode_tok_s={qat_decode:.3f} q4k_decode_tok_s={q4k_decode:.3f} speedup={speedup:.3f} min_speedup={min_speedup:.3f}")
if speedup < min_speedup:
    raise SystemExit(f"QAT speedup {speedup:.3f} below gate {min_speedup:.3f}")
PY

echo "raw output: $ROOT_OUT_DIR"
