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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PKG_DIR="$ROOT_DIR/pkg/inference"

if [[ -n "${ANTFLY_BIN:-}" ]]; then
  ANTFLY_BIN="$ANTFLY_BIN"
elif [[ -x "$PKG_DIR/zig-out/bin/antfly" ]]; then
  ANTFLY_BIN="$PKG_DIR/zig-out/bin/antfly"
else
  ANTFLY_BIN="$PKG_DIR/zig-out/bin/antfly-inference"
fi
TARGET_MODEL_DIR="${ANTFLY_INFERENCE_GEMMA4_TARGET_MODEL:-$HOME/.antfly/inference/models/google/gemma-4-E2B-it}"
DRAFT_MODEL_DIR="${ANTFLY_INFERENCE_GEMMA4_DFLASH_DRAFT_MODEL:-$HOME/.antfly/inference/models/z-lab/gemma-4-E2B-it-dflash}"
PROMPT="${ANTFLY_INFERENCE_GEMMA4_DFLASH_PROMPT:-hi}"
MAX_TOKENS="${ANTFLY_INFERENCE_GEMMA4_DFLASH_MAX_TOKENS:-4}"
SPECULATIVE_K="${ANTFLY_INFERENCE_GEMMA4_DFLASH_SPECULATIVE_K:-16}"
EXPECTED_TOKEN_IDS="${ANTFLY_INFERENCE_GEMMA4_DFLASH_EXPECTED_TOKEN_IDS:-}"
BACKEND="${ANTFLY_INFERENCE_GEMMA4_DFLASH_BACKEND:-auto}"
HOST_BUDGET_MB="${ANTFLY_INFERENCE_GEMMA4_DFLASH_HOST_BUDGET_MB:-12288}"
COMBINED_BUDGET_MB="${ANTFLY_INFERENCE_GEMMA4_DFLASH_COMBINED_BUDGET_MB:-17408}"
OUT_DIR="${OUT_DIR:-/tmp/antfly-inference-metal-gemma4-dflash-speculative}"
DEBUG_METAL_SCRIPT="$PKG_DIR/scripts/debug_metal_command.sh"

if [[ ! -x "$ANTFLY_BIN" ]]; then
  echo "antfly binary not executable: $ANTFLY_BIN" >&2
  echo "build it first, for example: cd pkg/inference && zig build -Doptimize=ReleaseFast -Dmetal=true -Dmlx=false -Donnx=false -Dpjrt=false" >&2
  exit 2
fi

if [[ "$(basename "$ANTFLY_BIN")" == "antfly-inference" ]]; then
  GENERATE_COMMAND=(generate)
else
  GENERATE_COMMAND=(inference generate)
fi

if [[ ! -d "$TARGET_MODEL_DIR" ]]; then
  echo "Gemma4 target model directory not found: $TARGET_MODEL_DIR" >&2
  echo "set ANTFLY_INFERENCE_GEMMA4_TARGET_MODEL to the local target model directory" >&2
  exit 2
fi

if [[ ! -d "$DRAFT_MODEL_DIR" ]]; then
  echo "Gemma4 DFlash draft model directory not found: $DRAFT_MODEL_DIR" >&2
  echo "set ANTFLY_INFERENCE_GEMMA4_DFLASH_DRAFT_MODEL to the local DFlash draft model directory" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
out="$OUT_DIR/dflash.txt"
debug_out="$OUT_DIR/debug"

set +e
TERMITE_DEBUG_METAL_TIMING=1 \
bash "$DEBUG_METAL_SCRIPT" command \
  --label metal-gemma4-dflash-speculative \
  --out-dir "$debug_out" \
  --timeout "${ANTFLY_INFERENCE_GEMMA4_DFLASH_TIMEOUT_SECS:-90}" \
  --api-validate \
  --cwd "$ROOT_DIR" \
  -- "$ANTFLY_BIN" "${GENERATE_COMMAND[@]}" "$TARGET_MODEL_DIR" "$PROMPT" \
  --backend "$BACKEND" \
  --draft-model "$DRAFT_MODEL_DIR" \
  --speculative-method dflash \
  --speculative-k "$SPECULATIVE_K" \
  --max-tokens "$MAX_TOKENS" \
  --host-budget-mb "$HOST_BUDGET_MB" \
  --combined-budget-mb "$COMBINED_BUDGET_MB" \
  --print-token-ids \
  --print-token-count \
  --print-timing >"$out" 2>&1
rc=$?
set -e

if [[ "$rc" != "0" ]]; then
  echo "Gemma4 DFlash speculative run failed; output: $out debug: $debug_out" >&2
  sed -n '1,260p' "$out" >&2
  if [[ -f "$debug_out/stdout.txt" ]]; then
    sed -n '1,260p' "$debug_out/stdout.txt" >&2
  fi
  exit 1
fi

token_ids="$(awk '/^token_ids:/ { sub(/^token_ids:[[:space:]]*/, ""); print; exit }' "$debug_out/stdout.txt")"
if [[ -n "$EXPECTED_TOKEN_IDS" && "$token_ids" != "$EXPECTED_TOKEN_IDS" ]]; then
  echo "Gemma4 DFlash token anchor failed" >&2
  echo "expected: $EXPECTED_TOKEN_IDS" >&2
  echo "actual:   ${token_ids:-<missing>}" >&2
  echo "output:   $debug_out/stdout.txt" >&2
  exit 1
fi

if ! grep -Eq 'speculative: method=dflash rounds=[1-9][0-9]* drafted=[1-9][0-9]* matched=[1-9][0-9]*' "$debug_out/stdout.txt"; then
  echo "Gemma4 DFlash did not produce accepted speculative drafts; output: $debug_out/stdout.txt" >&2
  grep -E 'speculative:|dflash:|token_ids:' "$debug_out/stdout.txt" >&2 || true
  exit 1
fi

if ! grep -Eq 'dflash: .*host_fallbacks=0' "$debug_out/stdout.txt"; then
  echo "Gemma4 DFlash used a host fallback; output: $debug_out/stdout.txt" >&2
  grep -E 'dflash:|speculative:' "$debug_out/stdout.txt" >&2 || true
  exit 1
fi

if ! grep -Eq 'dflash: .*full_tensor_download_bytes=0' "$debug_out/stdout.txt"; then
  echo "Gemma4 DFlash downloaded full tensors; output: $debug_out/stdout.txt" >&2
  grep -E 'dflash:|speculative:' "$debug_out/stdout.txt" >&2 || true
  exit 1
fi

if grep -Eq 'assistant|MTP|metal decoder-runtime prewarm failed' "$debug_out/stdout.txt"; then
  echo "Gemma4 DFlash emitted stale assistant/MTP/prewarm warnings; output: $debug_out/stdout.txt" >&2
  exit 1
fi

echo "metal Gemma4 DFlash speculative smoke passed"
echo "output: $debug_out/stdout.txt"
echo "debug: $debug_out"
