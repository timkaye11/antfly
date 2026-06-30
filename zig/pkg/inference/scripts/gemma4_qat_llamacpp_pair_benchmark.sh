#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: gemma4_qat_llamacpp_pair_benchmark.sh

Alternates Antfly and llama.cpp on the Gemma 4 E4B QAT Q4_0 CUDA benchmark
and writes raw outputs plus paired summary artifacts.

Environment overrides:
  ANTFLY_BIN              antfly-inference binary
  LLAMA_CPP_BIN           llama.cpp llama-completion binary
  MODEL                   Gemma 4 E4B QAT q4_0 GGUF file or model directory
  OUT_DIR                 output directory
  REPEATS                 paired samples per engine (default: 5)
  PROMPT                  raw prompt
  ANTFLY_TOKENS           Antfly generated-token request (default: 511)
  LLAMA_TOKENS            llama.cpp n_predict request (default: 512)
  TIMEOUT                 per-command timeout, or off (default: 360s)
  REQUIRE_ANTFLY_WIN      1 to fail when Antfly median is not faster (default: 0)
  MIN_WIN_MS              required median E2E win in ms when enforcing (default: 0)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
inference_dir="$repo_root/zig/pkg/inference"
antfly_bin="${ANTFLY_BIN:-${ANTFY_BIN:-$inference_dir/zig-out/bin/antfly-inference}}"
llama_cpp_bin="${LLAMA_CPP_BIN:-/tmp/llama.cpp/build/bin/llama-completion}"
model="${MODEL:-$repo_root/.models/google/gemma-4-E4B-it-qat-q4_0-gguf/gemma-4-E4B_q4_0-it.gguf}"
out_dir="${OUT_DIR:-/tmp/antfly-gemma4-qat-llamacpp-paired-$(date -u +%Y%m%dT%H%M%SZ)}"
repeats="${REPEATS:-5}"
prompt="${PROMPT:-Here is a sentence about ants:}"
antfly_tokens="${ANTFLY_TOKENS:-511}"
llama_tokens="${LLAMA_TOKENS:-512}"
command_timeout="${TIMEOUT:-360s}"
require_antfly_win="${REQUIRE_ANTFLY_WIN:-0}"
min_win_ms="${MIN_WIN_MS:-0}"

case "$repeats" in
  ''|*[!0-9]*)
    echo "REPEATS must be a positive integer" >&2
    exit 2
    ;;
esac
if [[ "$repeats" -lt 1 ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 1
  fi
}

run_maybe_timeout() {
  if [[ -n "$command_timeout" && "$command_timeout" != "0" && "$command_timeout" != "off" && "$command_timeout" != "none" ]]; then
    timeout "$command_timeout" "$@"
  else
    "$@"
  fi
}

require_path "antfly-inference binary" "$antfly_bin"
require_path "llama.cpp binary" "$llama_cpp_bin"
require_path "Gemma 4 E4B QAT model" "$model"
mkdir -p "$out_dir"

common_antfly_env=(
  ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING=1
  ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION=1
  ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION_CHUNK=12
  ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_PRECOMPUTE=1
  ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE=1
  ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_TILE4_W8=1
  ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_TILE4_W10_E4B_DOWN=1
  ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_TILE4_W8=1
  ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK=1
  ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A=1
  ANTFLY_INFERENCE_CUDA_Q4_0_QKV_Q8_1_DP4A=1
  ANTFLY_INFERENCE_CUDA_Q4_0_QKV_Q8_1_TILE8=1
  ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_DP4A=1
  ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A=1
  ANTFLY_INFERENCE_CUDA_Q4_0_ACTIVATION_SLICE_Q8_1_DP4A=1
  ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_Q8_1_DP4A=1
  ANTFLY_INFERENCE_CUDA_Q6_K_LM_HEAD_Q8_1=1
  ANTFLY_INFERENCE_CUDA_Q6_K_LM_HEAD_Q8_1_TILE8_EXACT_THREADS=1
  ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY=required
  ANTFLY_INFERENCE_CUDA_CAPTURE_FINAL_HIDDEN=1
  ANTFLY_INFERENCE_CUDA_CAPTURE_UPDATE_EXEC=1
  ANTFLY_INFERENCE_CUDA_CAPTURE_DEVICE_SCALARS=1
  ANTFLY_INFERENCE_CUDA_CAPTURE_PERSISTENT_REPLAY=1
  ANTFLY_INFERENCE_CUDA_CAPTURE_GREEDY_TOKEN=1
  ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD=863
  ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP=2500
  ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY=544
)

run_antfly_command() {
  if [[ -n "$command_timeout" && "$command_timeout" != "0" && "$command_timeout" != "off" && "$command_timeout" != "none" ]]; then
    env "${common_antfly_env[@]}" timeout "$command_timeout" "$@"
  else
    env "${common_antfly_env[@]}" "$@"
  fi
}

run_antfly() {
  local index="$1"
  local json_path="$out_dir/antfly_${index}.json"
  local log_path="$out_dir/antfly_${index}.log"
  rm -f "$json_path" "$log_path"
  echo "RUN antfly sample=$index"
  run_antfly_command "$antfly_bin" generate "$model" "$prompt" \
    --backend cuda \
    --combined-budget-mb 22000 \
    --backend-budget-mb 19000 \
    --kv-budget-mb 1024 \
    --scratch-budget-mb 2048 \
    --prefill-chunk-size 32 \
    --max-tokens "$antfly_tokens" \
    --temperature 0 \
    --raw-prompt \
    --no-chat-template \
    --ignore-eos \
    --cache-dtype polar4 \
    --print-timing \
    --print-token-count \
    --json-timing "$json_path" >"$log_path" 2>&1
}

run_llama() {
  local index="$1"
  local log_path="$out_dir/llama_${index}.log"
  rm -f "$log_path"
  echo "RUN llama.cpp sample=$index"
  run_maybe_timeout "$llama_cpp_bin" \
    -m "$model" \
    -p "$prompt" \
    -n "$llama_tokens" \
    -c 2048 \
    -ngl 999 \
    -ctk q4_0 \
    -ctv q4_0 \
    --temp 0 \
    --top-k 64 \
    --top-p 0.95 \
    --min-p 0.05 \
    -s 3060418694 \
    -no-cnv \
    --no-display-prompt \
    --ignore-eos >"$log_path" 2>&1
}

for ((i = 1; i <= repeats; i++)); do
  run_antfly "$i"
  run_llama "$i"
done

python3 - "$out_dir" "$repeats" "$require_antfly_win" "$min_win_ms" <<'PY'
import json
import math
import pathlib
import re
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1])
repeats = int(sys.argv[2])
require_win = sys.argv[3].lower() not in {"0", "false", "off", "no"}
min_win_ms = float(sys.argv[4])

llama_patterns = {
    "prompt_eval_ms": re.compile(r"(?m)^[^\n]*perf_print:\s+prompt eval time =\s+([0-9.]+) ms"),
    "eval_ms": re.compile(r"(?m)^[^\n]*perf_print:\s+eval time =\s+([0-9.]+) ms"),
    "total_ms": re.compile(r"(?m)^[^\n]*perf_print:\s+total time =\s+([0-9.]+) ms"),
    "graphs_reused": re.compile(r"(?m)^[^\n]*perf_print:\s+graphs reused =\s+([0-9]+)"),
}

def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]

def stats(values):
    return {
        "min": min(values),
        "median": statistics.median(values),
        "avg": sum(values) / len(values),
        "max": max(values),
        "p95": percentile(values, 95),
    }

rows = []
antfly_totals = []
llama_totals = []
errors = []
for index in range(1, repeats + 1):
    antfly_path = out_dir / f"antfly_{index}.json"
    llama_path = out_dir / f"llama_{index}.log"
    if not antfly_path.exists():
        errors.append(f"missing {antfly_path}")
        continue
    if not llama_path.exists():
        errors.append(f"missing {llama_path}")
        continue

    antfly = json.loads(antfly_path.read_text(encoding="utf-8"))
    timing = antfly.get("timing_ms") or {}
    antfly_total = float(timing.get("generate") or timing.get("total_inner") or 0.0)
    antfly_prefill = float(timing.get("prefill_inner") or 0.0)
    antfly_decode = float(timing.get("decode_inner") or 0.0)
    antfly_tps = float(antfly.get("decode_tok_per_s") or 0.0)
    cuda = antfly.get("cuda") or {}
    antfly_replays = int(cuda.get("graph_capture_replays") or 0)
    antfly_discards = int(cuda.get("graph_capture_discards") or 0)
    antfly_precompute = int(cuda.get("gated_down_fused_q4_0_precompute") or 0)
    antfly_tile4 = int(cuda.get("gated_down_fused_q4_0_tile4") or 0)

    text = llama_path.read_text(encoding="utf-8", errors="replace")
    parsed = {}
    for name, pattern in llama_patterns.items():
        match = pattern.search(text)
        if not match:
            errors.append(f"missing {name} in {llama_path}")
            parsed[name] = 0.0
            continue
        parsed[name] = float(match.group(1))

    llama_total = parsed["total_ms"]
    llama_eval = parsed["eval_ms"]
    llama_prompt = parsed["prompt_eval_ms"]
    llama_graphs = int(parsed["graphs_reused"])
    antfly_totals.append(antfly_total)
    llama_totals.append(llama_total)
    rows.append(
        {
            "sample": index,
            "antfly_total_ms": antfly_total,
            "antfly_prefill_ms": antfly_prefill,
            "antfly_decode_ms": antfly_decode,
            "antfly_tok_s": antfly_tps,
            "antfly_replays": antfly_replays,
            "antfly_discards": antfly_discards,
            "antfly_gated_down_precompute": antfly_precompute,
            "antfly_gated_down_tile4": antfly_tile4,
            "llama_total_ms": llama_total,
            "llama_prompt_eval_ms": llama_prompt,
            "llama_eval_ms": llama_eval,
            "llama_graphs_reused": llama_graphs,
            "antfly_win_ms": llama_total - antfly_total,
        }
    )

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

summary = {
    "repeats": repeats,
    "antfly_total_ms": stats(antfly_totals),
    "llama_total_ms": stats(llama_totals),
    "win_ms": stats([row["antfly_win_ms"] for row in rows]),
    "rows": rows,
}
summary["antfly_median_win_ms"] = summary["llama_total_ms"]["median"] - summary["antfly_total_ms"]["median"]
summary["ok"] = (not require_win) or summary["antfly_median_win_ms"] >= min_win_ms

(out_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (out_dir / "paired_summary.tsv").open("w", encoding="utf-8") as f:
    fields = [
        "sample",
        "antfly_total_ms",
        "antfly_prefill_ms",
        "antfly_decode_ms",
        "antfly_tok_s",
        "antfly_replays",
        "antfly_discards",
        "antfly_gated_down_precompute",
        "antfly_gated_down_tile4",
        "llama_total_ms",
        "llama_prompt_eval_ms",
        "llama_eval_ms",
        "llama_graphs_reused",
        "antfly_win_ms",
    ]
    f.write("\t".join(fields) + "\n")
    for row in rows:
        f.write("\t".join(str(row[field]) for field in fields) + "\n")

print(
    "paired_benchmark "
    f"antfly_median_ms={summary['antfly_total_ms']['median']:.2f} "
    f"llama_median_ms={summary['llama_total_ms']['median']:.2f} "
    f"median_win_ms={summary['antfly_median_win_ms']:.2f} "
    f"out_dir={out_dir}"
)
if not summary["ok"]:
    print(
        f"Antfly median win {summary['antfly_median_win_ms']:.2f} ms "
        f"is below required {min_win_ms:.2f} ms",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
