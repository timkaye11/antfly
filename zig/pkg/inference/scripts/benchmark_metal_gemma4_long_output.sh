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
if [[ -n "$ANTFLY_BIN_OVERRIDE" && ! -x "$ANTFLY_BIN_OVERRIDE" ]]; then
  echo "ANTFLY_BIN is not executable: $ANTFLY_BIN_OVERRIDE" >&2
  exit 2
fi
if ! command -v "$LLAMA_CPP_BIN" >/dev/null 2>&1; then
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
llama_version="$("$LLAMA_CPP_BIN" --version 2>&1 | head -n 1 || true)"
python3 - "$OUT_DIR/metadata.json" "$git_revision" "$MODEL" "$GGUF" "$model_sha256" \
  "$OUTPUT_TOKENS" "$PROMPT_REPEAT" "$RUNS" "$WARMUPS" "$COOLDOWN_SECONDS" \
  "$LLAMA_CPP_BIN" "$llama_version" "$ANTFLY_CACHE_DTYPE" "$LLAMA_CACHE_TYPE_K" \
  "$LLAMA_CACHE_TYPE_V" "$LLAMA_CONTEXT_SIZE" "$WARMUP_OUTPUT_TOKENS" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "git_revision": sys.argv[2],
    "model": sys.argv[3],
    "gguf": sys.argv[4],
    "gguf_sha256": sys.argv[5],
    "host": platform.platform(),
    "output_tokens": int(sys.argv[6]),
    "prompt_repeat": int(sys.argv[7]),
    "runs": int(sys.argv[8]),
    "warmups": int(sys.argv[9]),
    "cooldown_seconds": int(sys.argv[10]),
    "llama_cpp_bin": sys.argv[11],
    "llama_cpp_version": sys.argv[12],
    "antfly_route": "compiled_whole_model",
    "antfly_cache_dtype": sys.argv[13],
    "llama_cache_type_k": sys.argv[14],
    "llama_cache_type_v": sys.argv[15],
    "llama_context_size": int(sys.argv[16]),
    "warmup_output_tokens": int(sys.argv[17]),
    "split_gqa_enable": os.environ.get("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT"),
    "split_gqa_disable": os.environ.get("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT"),
    "pipelined_decode_frame_enable": os.environ.get("TERMITE_METAL_ENABLE_PIPELINED_DECODE_FRAME"),
    "pipelined_decode_frame_disable": os.environ.get("TERMITE_METAL_DISABLE_PIPELINED_DECODE_FRAME"),
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
  "$LLAMA_CPP_BIN" \
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

python3 - "$OUT_DIR" "$RUNS" "$OUTPUT_TOKENS" "$MAX_TOTAL_RATIO" "$MIN_DECODE_RATIO" "$MAX_CV" <<'PY'
import json
import os
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = int(sys.argv[2])
requested_tokens = int(sys.argv[3])
max_total_ratio = float(sys.argv[4])
min_decode_ratio = float(sys.argv[5])
max_cv = float(sys.argv[6])

def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

split_enable = os.environ.get("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT")
expect_split_gqa = (split_enable is None or env_flag("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT")) and not env_flag("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT")

def metric(text, pattern, label, path):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {label}: {path}")
    return match

def summary(values):
    mean = statistics.fmean(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }

rows = []
reference_ids = None
for index in range(1, runs + 1):
    antfly_path = root / f"antfly-{index}.json"
    antfly_log_path = root / f"antfly-{index}.log"
    llama_path = root / f"llama-{index}.log"
    antfly = json.loads(antfly_path.read_text())
    antfly_log = antfly_log_path.read_text(errors="replace")
    llama_log = llama_path.read_text(errors="replace")

    if antfly.get("tokens") != requested_tokens or antfly.get("finish_reason") != "length":
        raise SystemExit(f"Antfly did not generate exactly {requested_tokens} tokens: {antfly_path}")
    if "generate-setup: live whole-model executor skipped" not in antfly_log:
        raise SystemExit(f"Antfly did not enter the compiled generation pipeline: {antfly_log_path}")
    if "gen_debug: executePrefill whole-model fast path" not in antfly_log:
        raise SystemExit(f"Antfly silently fell back from compiled whole-model prefill: {antfly_log_path}")

    ids = metric(antfly_log, r"^token_ids:\s*(.*)$", "Antfly token IDs", antfly_log_path).group(1).strip()
    if len(ids.split()) != requested_tokens:
        raise SystemExit(f"Antfly token ID count mismatch: {antfly_log_path}")
    if reference_ids is None:
        reference_ids = ids
    elif ids != reference_ids:
        raise SystemExit(f"Antfly greedy token IDs changed between paired runs: {antfly_log_path}")

    prompt_ids = metric(antfly_log, r"^prompt_token_ids:\s*(.*)$", "Antfly prompt token IDs", antfly_log_path).group(1).split()
    timing = antfly.get("timing_ms") or {}
    antfly_total = float(timing.get("generate") or 0)
    antfly_prefill = float(timing.get("prefill_inner") or 0)
    antfly_decode = float(timing.get("decode_inner") or 0)
    if min(antfly_total, antfly_prefill, antfly_decode) <= 0:
        raise SystemExit(f"invalid Antfly timing: {antfly_path}")

    llama_prompt = metric(
        llama_log,
        r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens",
        "llama prompt timing",
        llama_path,
    )
    llama_eval = metric(
        llama_log,
        r"(?m)^common_perf_print:\s+eval time =\s*([0-9.]+) ms /\s*(\d+) runs",
        "llama eval timing",
        llama_path,
    )
    llama_sampling = metric(
        llama_log,
        r"sampling time =\s*([0-9.]+) ms",
        "llama sampling timing",
        llama_path,
    )
    llama_total_match = metric(
        llama_log,
        r"total time =\s*([0-9.]+) ms",
        "llama total timing",
        llama_path,
    )
    llama_prompt_tokens = int(llama_prompt.group(2))
    llama_eval_runs = int(llama_eval.group(2))
    if llama_prompt_tokens != len(prompt_ids):
        raise SystemExit(
            f"prompt token accounting differs: Antfly={len(prompt_ids)} llama={llama_prompt_tokens}"
        )
    if llama_eval_runs != requested_tokens - 1:
        raise SystemExit(
            f"llama eval runs={llama_eval_runs}, expected {requested_tokens - 1}: {llama_path}"
        )

    llama_eval_ms = float(llama_eval.group(1))
    llama_sampling_ms = float(llama_sampling.group(1))
    llama_total = float(llama_total_match.group(1))
    antfly_decode_tps = (requested_tokens - 1) * 1000.0 / antfly_decode
    llama_decode_tps = llama_eval_runs * 1000.0 / (llama_eval_ms + llama_sampling_ms)

    dispatch = re.search(
        r"^metal_attention_dispatch:.*\bpaged_1x=(\d+).*\bdecode_gqa_split=(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    if not dispatch:
        raise SystemExit(f"missing decode attention route counters: {antfly_log_path}")
    paged_calls = int(dispatch.group(1))
    split_calls = int(dispatch.group(2))
    # Compiled prefill produces the first output token. The production decoder
    # then executes one 42-layer attention frame for each remaining token; it must
    # not retain or speculatively execute an additional next-token frame.
    decode_frames = requested_tokens - 1
    expected_attention = decode_frames * 42
    expected_paged = 0 if expect_split_gqa else expected_attention
    expected_split = expected_attention if expect_split_gqa else 0
    if paged_calls != expected_paged or split_calls != expected_split:
        raise SystemExit(
            f"decode attention routes paged/split={paged_calls}/{split_calls}, "
            f"expected production paged/split={expected_paged}/{expected_split}: {antfly_log_path}"
        )

    runtime_memory = re.search(
        r"^metal_runtime_memory:.*\bframe_retained_mb=(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    if not runtime_memory or int(runtime_memory.group(1)) != 0:
        raise SystemExit(f"compiled decoder retained a speculative frame: {antfly_log_path}")

    q4_rows = re.search(
        r"^metal_q4_0_dispatch:.*\blinear_reduce_rows=(\d+)/(\d+)/(\d+)/(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    q6_rows = re.search(
        r"^metal_q4_q6_k_dispatch:.*\bq6_linear_reduce_rows=(\d+)/(\d+)/(\d+)/(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    q4_encode = re.search(
        r"^metal_q4_0_encode_us:\s+linear_reduce=(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    q4_exact = re.search(
        r"^metal_jit_exact_dispatch:\s+q4_0=(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    q4_pair_activation = re.search(
        r"^metal_q4_0_dispatch:.*\bpair_act_reduce=(\d+)",
        antfly_log,
        re.MULTILINE,
    )
    exact_q4_calls = int(q4_exact.group(1)) if q4_exact else 0
    fused_q4_pairs = int(q4_pair_activation.group(1)) if q4_pair_activation else 0
    if not q4_rows or int(q4_rows.group(1)) + exact_q4_calls + 2 * fused_q4_pairs < decode_frames * 210:
        raise SystemExit(f"missing expected Q4_0 row-one decode route: {antfly_log_path}")
    if not q6_rows or int(q6_rows.group(1)) < requested_tokens:
        raise SystemExit(f"missing expected Q6_K row-one LM-head route: {antfly_log_path}")

    rows.append({
        "sample": index,
        "prompt_tokens": len(prompt_ids),
        "antfly_total_ms": antfly_total,
        "antfly_prefill_ms": antfly_prefill,
        "antfly_decode_ms": antfly_decode,
        "antfly_decode_tok_s": antfly_decode_tps,
        "llama_total_ms": llama_total,
        "llama_prompt_ms": float(llama_prompt.group(1)),
        "llama_decode_ms": llama_eval_ms + llama_sampling_ms,
        "llama_decode_tok_s": llama_decode_tps,
        "total_ratio": antfly_total / llama_total,
        "decode_ratio": antfly_decode_tps / llama_decode_tps,
        "paged_1x_calls": paged_calls,
        "decode_gqa_split_calls": split_calls,
        "frame_retained_mb": int(runtime_memory.group(1)),
        "q4_0_linear_reduce_rows_1": int(q4_rows.group(1)) if q4_rows else None,
        "q4_0_exact_dispatches": exact_q4_calls,
        "q4_0_pair_activation_dispatches": fused_q4_pairs,
        "q6_k_linear_reduce_rows_1": int(q6_rows.group(1)) if q6_rows else None,
        "q4_0_linear_reduce_encode_us": int(q4_encode.group(1)) if q4_encode else None,
    })

antfly_total = summary([row["antfly_total_ms"] for row in rows])
llama_total = summary([row["llama_total_ms"] for row in rows])
antfly_decode_tps = summary([row["antfly_decode_tok_s"] for row in rows])
llama_decode_tps = summary([row["llama_decode_tok_s"] for row in rows])
total_ratio = antfly_total["median"] / llama_total["median"]
decode_ratio = antfly_decode_tps["median"] / llama_decode_tps["median"]

metadata = json.loads((root / "metadata.json").read_text())
result = {
    "metadata": metadata,
    "runs": runs,
    "output_tokens": requested_tokens,
    "prompt_tokens": rows[0]["prompt_tokens"],
    "antfly_total_ms": antfly_total,
    "llama_total_ms": llama_total,
    "antfly_decode_tok_s": antfly_decode_tps,
    "llama_decode_tok_s": llama_decode_tps,
    "total_ratio": total_ratio,
    "max_total_ratio": max_total_ratio,
    "decode_ratio": decode_ratio,
    "min_decode_ratio": min_decode_ratio,
    "max_cv": max_cv,
    "token_ids": reference_ids,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    f"Gemma4 long-output: prompt={result['prompt_tokens']} output={requested_tokens} "
    f"Antfly={antfly_total['median']:.1f}ms llama={llama_total['median']:.1f}ms "
    f"total_ratio={total_ratio:.3f} decode_ratio={decode_ratio:.3f}"
)

if antfly_total["cv"] > max_cv or llama_total["cv"] > max_cv:
    raise SystemExit(
        f"benchmark CV exceeded {max_cv:.3f}: Antfly={antfly_total['cv']:.3f} llama={llama_total['cv']:.3f}"
    )
if total_ratio > max_total_ratio:
    raise SystemExit(f"Antfly/llama total ratio {total_ratio:.3f} exceeds {max_total_ratio:.3f}")
if decode_ratio < min_decode_ratio:
    raise SystemExit(f"Antfly/llama decode ratio {decode_ratio:.3f} below {min_decode_ratio:.3f}")
PY

echo "output: $OUT_DIR"
