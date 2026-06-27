#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: bench_qwen36_cuda_production.sh [--quick|--full|--production-gate] [--graph] [--skip-build]

Runs the Qwen3.6 CUDA production-throughput harness for one NVIDIA L4 stream.

Modes:
  --quick             2-token smoke, 64-token baseline, 64-token decode profile (default)
  --full             quick mode plus 3x 256-token production runs and 10 tok/s gate
  --production-gate  alias for --full
  --graph            also run a 64-token graph-replay candidate and validate token parity
  --skip-build       use the existing antfly-inference binary

Environment overrides:
  ANTFLY_QWEN36_MODEL             Qwen3.6 GGUF path
  ANTFLY_QWEN36_OUT_DIR           output dir (default: /tmp/qwen36-prod-<timestamp>)
  ANTFLY_QWEN36_MIN_TOK_PER_S     production median floor (default: 10.0)
  ANTFLY_QWEN36_MAX_DEVICE_MB     device allocation ceiling (default: 20000)
  ANTFLY_QWEN36_PROMPT            benchmark prompt
  ANTFLY_QWEN36_TIMEOUT           per-generate timeout seconds (default: 1800)
  ANTFLY_QWEN36_CACHE_DTYPE       KV cache dtype (default: f32)
  ANTFLY_QWEN36_COMBINED_MB       combined memory budget (default: 28000)
  ANTFLY_QWEN36_HOST_MB           host lazy-weight budget (default: 6144)
  ANTFLY_QWEN36_BACKEND_MB        backend budget (default: 20000)
  ANTFLY_QWEN36_KV_MB             KV budget (default: 512)
  ANTFLY_QWEN36_SCRATCH_MB        scratch budget (default: 1024)
  ANTFLY_QWEN36_GRAPH_TEMP_PERIOD graph temp slot period (default: 1)
  ANTFLY_QWEN36_GRAPH_TEMP_SKIP   graph temp slot skip (default: 0)
  ANTFLY_QWEN36_ALLOW_EXPERIMENTAL_GRAPH_INSTANTIATE
                                  run the bounded CUDA graph replay path for --graph (default: 1)
  ZIG                             path to Zig
USAGE
}

mode="quick"
run_graph=0
skip_build=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --quick)
      mode="quick"
      ;;
    --full|--production-gate)
      mode="full"
      ;;
    --graph)
      run_graph=1
      ;;
    --skip-build)
      skip_build=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

pkg_root="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "$pkg_root/../../.." && pwd)"
model_path="${ANTFLY_QWEN36_MODEL:-$repo_root/.models/ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM/Qwen3.6-27B-4bpw-16GB-VRAM.gguf}"
out_dir="${ANTFLY_QWEN36_OUT_DIR:-${TMPDIR:-/tmp}/qwen36-prod-$(date -u +%Y%m%dT%H%M%SZ)}"
prompt="${ANTFLY_QWEN36_PROMPT:-Write a concise technical paragraph about CUDA inference throughput.}"
min_tok_per_s="${ANTFLY_QWEN36_MIN_TOK_PER_S:-10.0}"
max_device_mb="${ANTFLY_QWEN36_MAX_DEVICE_MB:-20000}"
command_timeout="${ANTFLY_QWEN36_TIMEOUT:-1800}"
cache_dtype="${ANTFLY_QWEN36_CACHE_DTYPE:-f32}"
combined_budget_mb="${ANTFLY_QWEN36_COMBINED_MB:-28000}"
host_budget_mb="${ANTFLY_QWEN36_HOST_MB:-6144}"
backend_budget_mb="${ANTFLY_QWEN36_BACKEND_MB:-20000}"
kv_budget_mb="${ANTFLY_QWEN36_KV_MB:-512}"
scratch_budget_mb="${ANTFLY_QWEN36_SCRATCH_MB:-1024}"
graph_temp_period="${ANTFLY_QWEN36_GRAPH_TEMP_PERIOD:-1}"
graph_temp_skip="${ANTFLY_QWEN36_GRAPH_TEMP_SKIP:-0}"
allow_experimental_graph_instantiate="${ANTFLY_QWEN36_ALLOW_EXPERIMENTAL_GRAPH_INSTANTIATE:-1}"
zig_global_cache_dir="${ZIG_GLOBAL_CACHE_DIR:-${TMPDIR:-/tmp}/antfly-zig-global-cache}"

mkdir -p "$out_dir"
summary="$out_dir/summary.txt"
: > "$summary"

log() {
  printf '%s\n' "$*" | tee -a "$summary"
}

resolve_zig() {
  if [ -n "${ZIG:-}" ]; then
    printf '%s\n' "$ZIG"
  elif command -v zig >/dev/null 2>&1; then
    command -v zig
  elif [ -x "$repo_root/.tools/zig-x86_64-linux-0.16.0/zig" ]; then
    printf '%s\n' "$repo_root/.tools/zig-x86_64-linux-0.16.0/zig"
  else
    echo "zig not found; set ZIG=/path/to/zig" >&2
    return 1
  fi
}

require_model() {
  if [ ! -f "$model_path" ]; then
    log "missing Qwen3.6 GGUF: $model_path"
    exit 1
  fi
  python3 - "$model_path" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
if size < 10 * 1000 * 1000 * 1000:
    raise SystemExit(f"{path} is only {size} bytes; expected a full Qwen3.6 GGUF")
print(f"model_size_bytes={size}")
PY
}

build_binary() {
  if [ "$skip_build" = "1" ]; then
    return
  fi
  local zig_bin
  zig_bin="$(resolve_zig)"
  log "building ReleaseFast CUDA binary"
  (cd "$pkg_root" && "$zig_bin" build --global-cache-dir "$zig_global_cache_dir" \
    -Dcuda=true \
    -Dcuda-artifacts=fatbin \
    -Dcuda-libs=auto \
    -Doptimize=ReleaseFast)
}

run_generate() {
  local label="$1"
  local tokens="$2"
  shift 2
  local json="$out_dir/$label.json"
  local run_log="$out_dir/$label.log"
  local -a env_args=("$@")

  log "running $label max_tokens=$tokens"
  env "${env_args[@]}" timeout "$command_timeout" "$bin" generate "$model_path" "$prompt" \
    --backend cuda \
    --max-tokens "$tokens" \
    --temperature 0 \
    --raw-prompt \
    --no-chat-template \
    --print-token-count \
    --print-token-ids \
    --print-finish-reason \
    --print-timing \
    --json-timing "$json" \
    --cache-dtype "$cache_dtype" \
    --combined-budget-mb "$combined_budget_mb" \
    --host-budget-mb "$host_budget_mb" \
    --backend-budget-mb "$backend_budget_mb" \
    --kv-budget-mb "$kv_budget_mb" \
    --scratch-budget-mb "$scratch_budget_mb" 2>&1 | tee "$run_log"
}

validate_json() {
  local require_profile="$1"
  local require_graph="$2"
  shift 2
  python3 - "$require_profile" "$require_graph" "$min_tok_per_s" "$max_device_mb" "$mode" "$@" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

require_profile = sys.argv[1] == "1"
graph_mode = sys.argv[2]
require_graph = graph_mode != "0"
require_graph_replay = graph_mode == "replay"
min_tok_per_s = float(sys.argv[3])
max_device_bytes = int(float(sys.argv[4]) * 1024 * 1024)
mode = sys.argv[5]
paths = [Path(p) for p in sys.argv[6:]]

fallback_zero_keys = [
    "qwen35_decode_core_fallbacks",
    "qwen35_decode_core_ab_fallbacks",
    "qwen36_mlp_fused_fallbacks",
    "qwen36_mlp_pre_rms_fused_fallbacks",
    "lm_head_argmax_fallbacks",
    "gated_down_fallbacks",
    "qkv_fallback_unsupported",
    "qkv_kernel_unavailable",
    "decoder_runtime_gated_ffn_misses",
    "device_kv_fail_batch",
    "device_kv_fail_no_cache",
    "device_kv_fail_no_storage",
    "device_kv_fail_no_hook",
    "device_kv_fail_write",
    "device_kv_fail_read",
    "device_kv_fail_shape",
]

profile_keys = [
    "decode_profile_qkv_us",
    "decode_profile_gqa_attention_us",
    "decode_profile_attention_output_us",
    "decode_profile_attention_norm_residual_us",
    "decode_profile_ffn_gate_up_us",
    "decode_profile_ffn_gated_down_us",
    "decode_profile_ffn_post_norm_us",
    "decode_profile_lm_head_argmax_us",
    "decode_profile_graph_replay_us",
]

payloads = []
errors = []
for path in paths:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payloads.append((path, payload))
    cuda = payload.get("cuda")
    if payload.get("backend") != "cuda":
        errors.append(f"{path}: backend={payload.get('backend')!r}, expected cuda")
    if not isinstance(payload.get("decode_tok_per_s"), (int, float)) or payload["decode_tok_per_s"] <= 0:
        errors.append(f"{path}: invalid decode_tok_per_s={payload.get('decode_tok_per_s')!r}")
    if not isinstance(cuda, dict):
        errors.append(f"{path}: missing cuda stats")
        continue
    if cuda.get("qwen36_mlp_fused_hits", 0) <= 0:
        errors.append(f"{path}: qwen36_mlp_fused_hits must be > 0")
    if cuda.get("qwen35_decode_core_ab_fused", 0) <= 0:
        errors.append(f"{path}: qwen35_decode_core_ab_fused must be > 0")
    if cuda.get("qwen36_mlp_pre_rms_fused_hits", 0) != 0:
        errors.append(f"{path}: pre-RMS MLP fusion must remain default-off")
    for key in fallback_zero_keys:
        value = cuda.get(key)
        if value not in (None, 0):
            errors.append(f"{path}: cuda.{key}={value}, expected 0")
    allocated = cuda.get("device_allocated_bytes")
    if allocated is not None and allocated > max_device_bytes:
        errors.append(f"{path}: device_allocated_bytes={allocated}, ceiling={max_device_bytes}")

if require_profile:
    profile_payload = payloads[-1][1]
    cuda = profile_payload.get("cuda") or {}
    if cuda.get("decode_profile_events", 0) <= 0:
        errors.append(f"{payloads[-1][0]}: decode_profile_events must be > 0")
    ranked = sorted(
        ((key, int(cuda.get(key, 0))) for key in profile_keys),
        key=lambda item: item[1],
        reverse=True,
    )
    print("decode_profile_ranked_us:")
    for key, value in ranked:
        print(f"  {key}={value}")

if require_graph:
    graph_payload = payloads[-1][1]
    cuda = graph_payload.get("cuda") or {}
    persistent_replays = cuda.get("graph_capture_persistent_replays", 0)
    unsafe_aborts = cuda.get("graph_capture_unsafe_aborts", 0)
    if require_graph_replay and persistent_replays <= 0:
        errors.append(f"{payloads[-1][0]}: graph_capture_persistent_replays must be > 0")
    if cuda.get("graph_capture_scalar_updates", 0) <= 0:
        errors.append(f"{payloads[-1][0]}: graph_capture_scalar_updates must be > 0")
    if cuda.get("graph_capture_capacity_skips", 0) != 0:
        errors.append(f"{payloads[-1][0]}: graph_capture_capacity_skips must be 0")
    if require_graph_replay and unsafe_aborts != 0:
        errors.append(f"{payloads[-1][0]}: graph_capture_unsafe_aborts must be 0")
    if not require_graph_replay and persistent_replays == 0 and unsafe_aborts <= 0:
        errors.append(f"{payloads[-1][0]}: guarded graph mode should either replay or record an unsafe abort")
    if cuda.get("graph_capture_input_mismatch_skips", 0) != 0:
        errors.append(f"{payloads[-1][0]}: graph_capture_input_mismatch_skips must be 0")

production = [p for p in payloads if p[0].name.startswith("production-")]
if mode == "full":
    if len(production) != 3:
        errors.append(f"expected 3 production runs, found {len(production)}")
    else:
        rates = [float(payload["decode_tok_per_s"]) for _, payload in production]
        median = statistics.median(rates)
        print(f"production_tok_s: values={rates} median={median:.6f} gate={min_tok_per_s:.6f}")
        if not math.isfinite(median) or median < min_tok_per_s:
            errors.append(f"production median tok/s {median:.6f} < gate {min_tok_per_s:.6f}")

for path, payload in payloads:
    cuda = payload.get("cuda") or {}
    print(
        f"{path.name}: tok_s={payload.get('decode_tok_per_s')} "
        f"tokens={payload.get('tokens')} launches={cuda.get('kernel_launches')} "
        f"launches_per_token={cuda.get('launches_per_token')} "
        f"persistent_replays={cuda.get('graph_capture_persistent_replays', 0)}"
    )

if errors:
    print("FAIL qwen36 production harness:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

token_ids_line() {
  local path="$1"
  grep '^token_ids:' "$path" | tail -1 | sed 's/[[:space:]]*$//'
}

require_model
build_binary

bin="$pkg_root/zig-out/bin/antfly-inference"
if [ ! -x "$bin" ]; then
  echo "missing binary: $bin" >&2
  exit 1
fi

log "output_dir=$out_dir"
log "running cuda-info --smoke"
"$bin" cuda-info --smoke 2>&1 | tee "$out_dir/cuda-info-smoke.log"

run_generate smoke-2 2
validate_json 0 0 "$out_dir/smoke-2.json" | tee -a "$summary"

run_generate baseline-64 64
validate_json 0 0 "$out_dir/baseline-64.json" | tee -a "$summary"

run_generate profile-64 64 ANTFLY_INFERENCE_CUDA_PROFILE_DECODE=1
validate_json 1 0 "$out_dir/profile-64.json" | tee -a "$summary"

if [ "$run_graph" = "1" ]; then
  graph_validation_mode="guard"
  if [ "$allow_experimental_graph_instantiate" = "1" ]; then
    graph_validation_mode="replay"
  fi
  run_generate graph-64 64 \
    ANTFLY_INFERENCE_CUDA_QWEN36_DECODE_GRAPH=1 \
    ANTFLY_INFERENCE_CUDA_CAPTURE_DEVICE_SCALARS=1 \
    ANTFLY_INFERENCE_CUDA_QWEN36_DECODE_GRAPH_INSTANTIATE="$allow_experimental_graph_instantiate" \
    ANTFLY_INFERENCE_CUDA_PROFILE_DECODE=1 \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$graph_temp_period" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP="$graph_temp_skip"
  baseline_ids="$(token_ids_line "$out_dir/baseline-64.log")"
  graph_ids="$(token_ids_line "$out_dir/graph-64.log")"
  if [ -z "$baseline_ids" ] || [ "$baseline_ids" != "$graph_ids" ]; then
    log "FAIL graph token parity"
    log "baseline: $baseline_ids"
    log "graph:    $graph_ids"
    exit 1
  fi
  validate_json 1 "$graph_validation_mode" "$out_dir/baseline-64.json" "$out_dir/graph-64.json" | tee -a "$summary"
fi

if [ "$mode" = "full" ]; then
  run_generate production-1 256
  run_generate production-2 256
  run_generate production-3 256
  validate_json 0 0 \
    "$out_dir/production-1.json" \
    "$out_dir/production-2.json" \
    "$out_dir/production-3.json" | tee -a "$summary"
fi

log "PASS qwen36 cuda production harness mode=$mode graph=$run_graph"
log "outputs: $out_dir"
