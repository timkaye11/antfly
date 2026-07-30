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

# Paired production gate for SearchAF's long-answer path. It compares the
# compiled whole-model generation pipeline with llama.cpp using the same raw 2,003-token
# prompt, exact output length, greedy sampling, cache precision, and GPU backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/inference_cli.sh
source "$SCRIPT_DIR/inference_cli.sh"

MODEL="${MODEL:-$HOME/.antfly/inference/models/google/gemma-4-E4B-it-qat-q4_0-gguf}"
ANTFLY_BIN_OVERRIDE="${ANTFLY_BIN:-}"
LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-llama-completion}"
EXPECTED_LLAMA_CPP_BUILD="${EXPECTED_LLAMA_CPP_BUILD:-10182}"
EXPECTED_LLAMA_CPP_SHA256="${EXPECTED_LLAMA_CPP_SHA256:-${LLAMA_CPP_EXPECTED_SHA256:-}}"
EXPECTED_TOKEN_IDS_SHA256="${EXPECTED_TOKEN_IDS_SHA256:-}"
OUT_DIR="${OUT_DIR:-/tmp/antfly-gemma4-metal-long-output-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-300}"
PROMPT_REPEAT="${PROMPT_REPEAT:-36}"
WARMUPS="${WARMUPS:-1}"
WARMUP_OUTPUT_TOKENS="${WARMUP_OUTPUT_TOKENS:-4}"
RUNS="${RUNS:-5}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-45}"
MAX_TOTAL_RATIO="${MAX_TOTAL_RATIO:-1.10}"
MIN_DECODE_RATIO="${MIN_DECODE_RATIO:-0.90}"
MAX_CV="${MAX_CV:-0.03}"
ANTFLY_CACHE_DTYPE="${ANTFLY_CACHE_DTYPE:-f16}"
LLAMA_CACHE_TYPE_K="${LLAMA_CACHE_TYPE_K:-f16}"
LLAMA_CACHE_TYPE_V="${LLAMA_CACHE_TYPE_V:-f16}"
LLAMA_CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-4096}"
EXPECT_GENERATED_FLASH_PREFILL_CALLS="${EXPECT_GENERATED_FLASH_PREFILL_CALLS:-35}"
EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS="${EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS:-7}"
EXPECT_PREFILL_DIRECT_KV="${EXPECT_PREFILL_DIRECT_KV:-0}"
EXPECT_FAST_PREPARED_FRAME="${EXPECT_FAST_PREPARED_FRAME:-1}"
EXPECT_Q4_0_MMV_VARIANT="${EXPECT_Q4_0_MMV_VARIANT:-nr4-nsg2}"
EXPECT_SWA_SCAN_CLAMP="${EXPECT_SWA_SCAN_CLAMP:-1}"
EXPECT_LLAMA_METAL_DEVICE="${EXPECT_LLAMA_METAL_DEVICE:-Apple M4}"
EXPECT_LLAMA_OFFLOADED_LAYERS="${EXPECT_LLAMA_OFFLOADED_LAYERS:-43}"

export EXPECT_GENERATED_FLASH_PREFILL_CALLS
export EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS
export EXPECT_PREFILL_DIRECT_KV
export EXPECT_FAST_PREPARED_FRAME
export EXPECT_Q4_0_MMV_VARIANT
export EXPECT_SWA_SCAN_CLAMP
export EXPECT_LLAMA_METAL_DEVICE
export EXPECT_LLAMA_OFFLOADED_LAYERS

for value in "$OUTPUT_TOKENS" "$WARMUP_OUTPUT_TOKENS" "$PROMPT_REPEAT" "$RUNS"; do
  case "$value" in
    ''|*[!0-9]*|0)
      echo "OUTPUT_TOKENS, WARMUP_OUTPUT_TOKENS, PROMPT_REPEAT, and RUNS must be positive integers" >&2
      exit 2
      ;;
  esac
done
for value in "$WARMUPS" "$COOLDOWN_SECONDS"; do
  case "$value" in
    ''|*[!0-9]*)
      echo "WARMUPS and COOLDOWN_SECONDS must be non-negative integers" >&2
      exit 2
      ;;
  esac
done

resolve_text_gguf() {
  local model="$1"
  if [[ -f "$model" ]]; then
    printf '%s\n' "$model"
    return
  fi
  find "$model" -maxdepth 1 -type f -name '*.gguf' ! -name '*mmproj*' | sort | head -n 1
}

GGUF="${GGUF:-$(resolve_text_gguf "$MODEL")}"
if [[ ! -e "$MODEL" || -z "$GGUF" || ! -f "$GGUF" ]]; then
  echo "missing Gemma4 model or text GGUF: $MODEL" >&2
  exit 2
fi
ANTFLY_BIN_RESOLVED="$(resolve_antfly_inference_bin)"
if [[ ! -x "$ANTFLY_BIN_RESOLVED" ]]; then
  echo "Antfly inference binary is not executable: $ANTFLY_BIN_RESOLVED" >&2
  exit 2
fi
LLAMA_CPP_BIN_RESOLVED="$(command -v "$LLAMA_CPP_BIN" 2>/dev/null || true)"
if [[ -z "$LLAMA_CPP_BIN_RESOLVED" || ! -x "$LLAMA_CPP_BIN_RESOLVED" ]]; then
  echo "llama-completion not found: $LLAMA_CPP_BIN" >&2
  exit 2
fi

PROMPT="${PROMPT:-}"
if [[ -z "$PROMPT" ]]; then
  evidence=""
  sentence='You answer questions about indexed files using only evidence. Evidence: Spella Caffe Logo.pdf is in /Users/timkaye/Downloads. Spella Caffe Logo Two Color.pdf is in /Users/timkaye/Downloads. Ignore unrelated source code. '
  for ((i = 0; i < PROMPT_REPEAT; i++)); do
    evidence+="$sentence"
  done
  printf -v PROMPT '<|turn>user\n%s\n\nwhere are my Spella coffee assets?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' "$evidence"
fi

mkdir -p "$OUT_DIR"

repo_root="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
model_sha256="$(shasum -a 256 "$GGUF" | awk '{print $1}')"
git_revision="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)"
git_status="$(LC_ALL=C git -C "$repo_root" status --porcelain=v1 --untracked-files=all)"
git_dirty=false
if [[ -n "$git_status" ]]; then
  git_dirty=true
fi
git_tracked_diff_sha256="$(LC_ALL=C git -C "$repo_root" diff --binary --no-ext-diff HEAD -- | shasum -a 256 | awk '{print $1}')"
git_status_sha256="$(LC_ALL=C git -C "$repo_root" status --porcelain=v1 --untracked-files=all | shasum -a 256 | awk '{print $1}')"
benchmark_harness_sha256="$(shasum -a 256 "$SCRIPT_DIR/benchmark_metal_gemma4_long_output.sh" | awk '{print $1}')"
benchmark_parser_sha256="$(shasum -a 256 "$SCRIPT_DIR/gemma4_metal_long_output.py" | awk '{print $1}')"
antfly_binary_sha256="$(shasum -a 256 "$ANTFLY_BIN_RESOLVED" | awk '{print $1}')"
llama_binary_sha256="$(shasum -a 256 "$LLAMA_CPP_BIN_RESOLVED" | awk '{print $1}')"
prompt_sha256="$(printf '%s' "$PROMPT" | shasum -a 256 | awk '{print $1}')"
llama_version_output="$("$LLAMA_CPP_BIN_RESOLVED" --version 2>&1 || true)"
if [[ -z "$llama_version_output" ]]; then
  echo "llama.cpp comparator returned empty --version output: $LLAMA_CPP_BIN_RESOLVED" >&2
  exit 2
fi
expected_llama_cpp_sha256_normalized="$(printf '%s' "$EXPECTED_LLAMA_CPP_SHA256" | tr '[:upper:]' '[:lower:]')"
if [[ -n "$expected_llama_cpp_sha256_normalized" && "$llama_binary_sha256" != "$expected_llama_cpp_sha256_normalized" ]]; then
  echo "llama.cpp binary SHA-256 mismatch: expected $expected_llama_cpp_sha256_normalized, got $llama_binary_sha256" >&2
  exit 2
fi
python3 - "$OUT_DIR/metadata.json" "$git_revision" "$MODEL" "$GGUF" "$model_sha256" \
  "$OUTPUT_TOKENS" "$PROMPT_REPEAT" "$RUNS" "$WARMUPS" "$COOLDOWN_SECONDS" \
  "$LLAMA_CPP_BIN" "$LLAMA_CPP_BIN_RESOLVED" "$llama_version_output" "$llama_binary_sha256" \
  "$EXPECTED_LLAMA_CPP_BUILD" "$expected_llama_cpp_sha256_normalized" "$ANTFLY_BIN_RESOLVED" \
  "$antfly_binary_sha256" "$prompt_sha256" "$ANTFLY_CACHE_DTYPE" "$LLAMA_CACHE_TYPE_K" \
  "$LLAMA_CACHE_TYPE_V" "$LLAMA_CONTEXT_SIZE" "$WARMUP_OUTPUT_TOKENS" "$git_dirty" \
  "$git_tracked_diff_sha256" "$git_status_sha256" "$benchmark_harness_sha256" \
  "$benchmark_parser_sha256" <<'PY'
import json
import os
import platform
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
version_output = sys.argv[13]
version_match = re.search(r"\bversion:\s*(\d+)\s*\(([0-9a-fA-F]+)\)", version_output)
if not version_match:
    raise SystemExit(f"unrecognized llama.cpp --version output: {version_output!r}")
build = int(version_match.group(1))
commit = version_match.group(2).lower()
expected_build = int(sys.argv[15])
if build != expected_build:
    raise SystemExit(f"llama.cpp build mismatch: expected b{expected_build}, got b{build}")
known_full_commits = {
    (10182, "afeebe103"): "afeebe103bd99cda8f5dfaefcabadf890db7fda7",
}
path.write_text(json.dumps({
    "schema": "antfly.gemma4_metal_long_output.metadata.v2",
    "git_revision": sys.argv[2],
    "git_dirty": sys.argv[25] == "true",
    "git_tracked_diff_sha256": sys.argv[26],
    "git_status_sha256": sys.argv[27],
    "benchmark_harness_sha256": sys.argv[28],
    "benchmark_parser_sha256": sys.argv[29],
    "model": sys.argv[3],
    "gguf": sys.argv[4],
    "gguf_sha256": sys.argv[5],
    "host": platform.platform(),
    "output_tokens": int(sys.argv[6]),
    "prompt_repeat": int(sys.argv[7]),
    "runs": int(sys.argv[8]),
    "warmups": int(sys.argv[9]),
    "cooldown_seconds": int(sys.argv[10]),
    "prompt_sha256": sys.argv[19],
    "llama_cpp_bin": sys.argv[11],
    "llama_cpp_resolved_bin": sys.argv[12],
    "llama_cpp_version": f"version: {build} ({commit})",
    "llama_cpp_version_output": version_output,
    "llama_cpp_build": build,
    "llama_cpp_commit": commit,
    "llama_cpp_full_commit": known_full_commits.get((build, commit)),
    "llama_cpp_binary_sha256": sys.argv[14],
    "llama_cpp_expected_build": expected_build,
    "llama_cpp_expected_sha256": sys.argv[16] or None,
    "llama_cpp_comparator_id": f"llama.cpp-b{build}-{commit}-{sys.argv[14][:12]}",
    "antfly_bin": sys.argv[17],
    "antfly_binary_sha256": sys.argv[18],
    "antfly_route": "compiled_whole_model",
    "antfly_cache_dtype": sys.argv[20],
    "llama_cache_type_k": sys.argv[21],
    "llama_cache_type_v": sys.argv[22],
    "llama_context_size": int(sys.argv[23]),
    "warmup_output_tokens": int(sys.argv[24]),
    "split_gqa_enable": os.environ.get("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT"),
    "split_gqa_disable": os.environ.get("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT"),
    "pipelined_decode_frame_enable": os.environ.get("TERMITE_METAL_ENABLE_PIPELINED_DECODE_FRAME"),
    "pipelined_decode_frame_disable": os.environ.get("TERMITE_METAL_DISABLE_PIPELINED_DECODE_FRAME"),
    "metal_policy_env": {
        name: os.environ.get(name)
        for name in (
            "TERMITE_METAL_ENABLE_PREFILL_SG_DIRECT_LOAD",
            "TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD",
            "TERMITE_METAL_Q4_0_MMV_VARIANT",
            "TERMITE_METAL_DISABLE_Q4_0_MMV_PORTFOLIO",
            "TERMITE_METAL_TRACE_Q4_0_MMV_VARIANT",
            "TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP",
            "TERMITE_METAL_DISABLE_FAST_PREPARED_FRAME",
            "TERMITE_METAL_FORCE_DIAGNOSTIC_COMMAND_BUFFERS",
            "EXPECT_GENERATED_FLASH_PREFILL_CALLS",
            "EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS",
            "EXPECT_PREFILL_DIRECT_KV",
            "EXPECT_FAST_PREPARED_FRAME",
            "EXPECT_Q4_0_MMV_VARIANT",
            "EXPECT_SWA_SCAN_CLAMP",
            "EXPECT_LLAMA_METAL_DEVICE",
            "EXPECT_LLAMA_OFFLOADED_LAYERS",
        )
    },
}, indent=2) + "\n")
PY

run_antfly_binary() {
  if [[ -n "$ANTFLY_BIN_OVERRIDE" ]]; then
    "$ANTFLY_BIN_OVERRIDE" "$@"
  else
    run_antfly_inference "$@"
  fi
}

cooldown() {
  if (( COOLDOWN_SECONDS > 0 )); then
    sleep "$COOLDOWN_SECONDS"
  fi
}

run_antfly() {
  local sample="$1"
  local output_tokens="${2:-$OUTPUT_TOKENS}"
  local json="$OUT_DIR/antfly-$sample.json"
  local log="$OUT_DIR/antfly-$sample.log"
  TERMITE_GEN_STAGE_DEBUG=1 \
  run_antfly_binary generate "$MODEL" "$PROMPT" \
    --backend metal \
    --mode compiled \
    --compiled-target whole-model \
    --max-tokens "$output_tokens" \
    --temperature 0 \
    --raw-prompt \
    --ignore-eos \
    --cache-dtype "$ANTFLY_CACHE_DTYPE" \
    --print-token-count \
    --print-finish-reason \
    --print-token-ids \
    --print-prompt-token-ids \
    --print-timing \
    --json-timing "$json" >"$log" 2>&1
}

run_llama() {
  local sample="$1"
  local output_tokens="${2:-$OUTPUT_TOKENS}"
  local log="$OUT_DIR/llama-$sample.log"
  "$LLAMA_CPP_BIN_RESOLVED" \
    -m "$GGUF" \
    --no-conversation \
    --no-jinja \
    --special \
    -p "$PROMPT" \
    -n "$output_tokens" \
    -c "$LLAMA_CONTEXT_SIZE" \
    -b 512 \
    -ub 512 \
    -ngl 999 \
    -lv 4 \
    --log-colors off \
    -ctk "$LLAMA_CACHE_TYPE_K" \
    -ctv "$LLAMA_CACHE_TYPE_V" \
    --temp 0 \
    --ignore-eos \
    --no-display-prompt >"$log" 2>&1
}

for ((i = 1; i <= WARMUPS; i++)); do
  run_antfly "warmup-$i" "$WARMUP_OUTPUT_TOKENS"
  cooldown
  run_llama "warmup-$i" "$WARMUP_OUTPUT_TOKENS"
  cooldown
done

for ((i = 1; i <= RUNS; i++)); do
  if (( i % 2 == 1 )); then
    run_antfly "$i"
    cooldown
    run_llama "$i"
  else
    run_llama "$i"
    cooldown
    run_antfly "$i"
  fi
  cooldown
done

python3 "$SCRIPT_DIR/gemma4_metal_long_output.py" \
  --out-dir "$OUT_DIR" \
  --runs "$RUNS" \
  --output-tokens "$OUTPUT_TOKENS" \
  --max-total-ratio "$MAX_TOTAL_RATIO" \
  --min-decode-ratio "$MIN_DECODE_RATIO" \
  --max-cv "$MAX_CV" \
  --expected-token-ids-sha256 "$EXPECTED_TOKEN_IDS_SHA256"

echo "output: $OUT_DIR"
