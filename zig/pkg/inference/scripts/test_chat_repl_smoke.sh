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

# Non-interactive smoke test for `antfly inference chat`: scripts two turns
# plus /bye through the REPL and asserts that the model answered, the stats
# footer appeared, and the second turn reused the cached KV prefix.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PKG_DIR="$ROOT_DIR/pkg/inference"

ANTFLY_BIN="${ANTFLY_BIN:-$PKG_DIR/zig-out/bin/antfly-inference}"
MODEL_DIR="${ANTFLY_INFERENCE_CHAT_SMOKE_MODEL:-$HOME/.antfly/inference/models/ggml-org/gemma-4-e2b-it-gguf}"
MAX_TOKENS="${ANTFLY_INFERENCE_CHAT_SMOKE_MAX_TOKENS:-24}"
BACKEND="${ANTFLY_INFERENCE_CHAT_SMOKE_BACKEND:-auto}"
OUT_DIR="${OUT_DIR:-/tmp/antfly-inference-chat-smoke}"

if [[ ! -x "$ANTFLY_BIN" ]]; then
  echo "antfly binary not executable: $ANTFLY_BIN" >&2
  echo "build it first, for example: cd pkg/inference && zig build -Doptimize=ReleaseFast -Dmetal=true -Donnx=false -Dpjrt=false" >&2
  exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "chat smoke model not installed: $MODEL_DIR" >&2
  echo "install it first: $ANTFLY_BIN pull ggml-org/gemma-4-e2b-it-gguf:gguf" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/chat-smoke.log"

# The first turn must exceed the prompt-cache store threshold (32 tokens) so
# the second turn can attach the cached prefix.
printf 'Please explain in a few sentences what a key-value store is and why many databases use log-structured merge trees for storage.\nThanks. Now explain what a vector index is for in one sentence.\n/bye\n' |
  "$ANTFLY_BIN" chat "$MODEL_DIR" \
    --backend "$BACKEND" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0 \
    >"$LOG_FILE" 2>&1 || {
  echo "chat smoke run failed; log follows" >&2
  cat "$LOG_FILE" >&2
  exit 1
}

fail() {
  echo "chat smoke assertion failed: $1" >&2
  cat "$LOG_FILE" >&2
  exit 1
}

# Each turn must either stream public text or explicitly report that the model
# kept its reply in the thought channel (current gemma4 E2B GGUF behavior —
# see GEMMA4.md "Chat REPL"). A silent blank turn is the failure mode.
if ! grep -qE 'key-value|store|thought-channel' "$LOG_FILE"; then
  fail "first turn produced neither reply text nor a thought-channel notice"
fi
# Stats footer: "(N tok · X tok/s · ..." once per turn.
[[ "$(grep -c 'tok/s' "$LOG_FILE")" -ge 2 ]] || fail "expected a stats footer for both turns"
# The second turn must reuse the cached KV prefix from the first.
grep -E '[1-9][0-9]* cached' "$LOG_FILE" >/dev/null || fail "second turn reported no cached prompt tokens"

echo "chat smoke passed ($LOG_FILE)"
