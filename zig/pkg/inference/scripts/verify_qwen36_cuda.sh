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

usage() {
  cat <<'USAGE'
usage: verify_qwen36_cuda.sh [--download] [--skip-build] [--smoke-only] [--production-gate]

Builds the CUDA binary, verifies Qwen3.6 CUDA capability, and runs a bounded
real-model generation smoke against the Qwen3.6 27B 4bpw GGUF.

Environment overrides:
  ANTFLY_QWEN36_MODEL          path to the GGUF file
  ANTFLY_QWEN36_MODEL_URL      download URL used with --download
  ANTFLY_QWEN36_PROMPT         generation prompt
  ANTFLY_QWEN36_MAX_TOKENS     generated token count (default: 2)
  ANTFLY_QWEN36_PROD_PROMPT    production-gate prompt
  ANTFLY_QWEN36_PROD_MAX_TOKENS production-gate token count (default: 64)
  ANTFLY_QWEN36_MIN_TOK_PER_S  production-gate decode tok/s floor (default: 10)
  ANTFLY_QWEN36_TIMEOUT        command timeout seconds (default: 1800)
  ANTFLY_QWEN36_OUT_DIR        output directory for logs/json timing
  ANTFLY_QWEN36_COMBINED_MB    combined memory budget (default: 28000)
  ANTFLY_QWEN36_HOST_MB        host lazy-weight budget (default: 6144)
  ANTFLY_QWEN36_BACKEND_MB     backend budget (default: 20000)
  ANTFLY_QWEN36_KV_MB          KV budget (default: 512)
  ANTFLY_QWEN36_SCRATCH_MB     scratch budget (default: 1024)
  ANTFLY_QWEN36_CACHE_DTYPE    KV cache dtype (default: f32)
  ZIG                          path to Zig
USAGE
}

download=0
skip_build=0
smoke_only=0
production_gate=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --download)
      download=1
      ;;
    --skip-build)
      skip_build=1
      ;;
    --smoke-only)
      smoke_only=1
      ;;
    --production-gate)
      production_gate=1
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
cd "$pkg_root"

model_url="${ANTFLY_QWEN36_MODEL_URL:-https://huggingface.co/ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM/resolve/main/Qwen3.6-27B-4bpw-16GB-VRAM.gguf}"
default_model="$repo_root/.models/ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM/Qwen3.6-27B-4bpw-16GB-VRAM.gguf"
model_path="${ANTFLY_QWEN36_MODEL:-$default_model}"
out_dir="${ANTFLY_QWEN36_OUT_DIR:-${TMPDIR:-/tmp}/antfly-qwen36-cuda-$(date -u +%Y%m%dT%H%M%SZ)}"
prompt="${ANTFLY_QWEN36_PROMPT:-Write one concise sentence about CUDA inference.}"
max_tokens="${ANTFLY_QWEN36_MAX_TOKENS:-2}"
prod_prompt="${ANTFLY_QWEN36_PROD_PROMPT:-Write a short paragraph about CUDA inference.}"
prod_max_tokens="${ANTFLY_QWEN36_PROD_MAX_TOKENS:-64}"
min_tok_per_s="${ANTFLY_QWEN36_MIN_TOK_PER_S:-10}"
command_timeout="${ANTFLY_QWEN36_TIMEOUT:-1800}"
combined_budget_mb="${ANTFLY_QWEN36_COMBINED_MB:-28000}"
host_budget_mb="${ANTFLY_QWEN36_HOST_MB:-6144}"
backend_budget_mb="${ANTFLY_QWEN36_BACKEND_MB:-20000}"
kv_budget_mb="${ANTFLY_QWEN36_KV_MB:-512}"
scratch_budget_mb="${ANTFLY_QWEN36_SCRATCH_MB:-1024}"
cache_dtype="${ANTFLY_QWEN36_CACHE_DTYPE:-f32}"
cuda_artifacts="${ANTFLY_CUDA_ARTIFACTS:-fatbin}"
cuda_libraries="${ANTFLY_CUDA_LIBS:-auto}"
optimize="${ANTFLY_QWEN36_OPTIMIZE:-ReleaseFast}"
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

ensure_download_space() {
  python3 - "$model_path" <<'PY'
import os
import shutil
import sys

path = os.path.abspath(sys.argv[1])
parent = os.path.dirname(path)
os.makedirs(parent, exist_ok=True)
free = shutil.disk_usage(parent).free
minimum = 16 * 1024 * 1024 * 1024
if free < minimum:
    raise SystemExit(f"need at least 16 GiB free in {parent}, found {free / (1024**3):.2f} GiB")
PY
}

if [ "$download" = "1" ]; then
  ensure_download_space
  mkdir -p "$(dirname "$model_path")"
  log "downloading Qwen3.6 GGUF to $model_path"
  curl -L --fail --continue-at - --output "$model_path" "$model_url"
fi

if [ ! -f "$model_path" ]; then
  log "missing Qwen3.6 GGUF: $model_path"
  log "rerun with --download, or set ANTFLY_QWEN36_MODEL to an existing file"
  exit 1
fi

python3 - "$model_path" <<'PY'
import os
import sys

path = sys.argv[1]
size = os.path.getsize(path)
minimum = 10 * 1000 * 1000 * 1000
if size < minimum:
    raise SystemExit(f"{path} is only {size} bytes; expected a full Qwen3.6 GGUF")
print(f"model_size_bytes={size}")
PY

zig_bin="$(resolve_zig)"
if [ "$skip_build" = "0" ]; then
  log "building CUDA binary"
  "$zig_bin" build --global-cache-dir "$zig_global_cache_dir" \
    -Dcuda=true \
    -Dcuda-artifacts="$cuda_artifacts" \
    -Dcuda-libs="$cuda_libraries" \
    -Doptimize="$optimize"
fi

bin="$pkg_root/zig-out/bin/antfly-inference"
if [ ! -x "$bin" ]; then
  echo "missing binary: $bin" >&2
  exit 1
fi

cuda_info_log="$out_dir/cuda-info-smoke.log"
log "running cuda-info --smoke"
"$bin" cuda-info --smoke 2>&1 | tee "$cuda_info_log"
grep -q "capability_qwen35: true" "$cuda_info_log"
grep -q "smoke: qwen35_linear_attention ok" "$cuda_info_log"

if [ "$smoke_only" = "1" ]; then
  log "PASS qwen36 smoke-only gate"
  log "outputs: $out_dir"
  exit 0
fi

generate_log="$out_dir/qwen36-generate.log"
json_file="$out_dir/qwen36-generate.json"
log "running Qwen3.6 CUDA generate smoke"
timeout "$command_timeout" "$bin" generate "$model_path" "$prompt" \
  --backend cuda \
  --max-tokens "$max_tokens" \
  --temperature 0 \
  --raw-prompt \
  --no-chat-template \
  --print-token-count \
  --print-finish-reason \
  --print-timing \
  --json-timing "$json_file" \
  --cache-dtype "$cache_dtype" \
  --combined-budget-mb "$combined_budget_mb" \
  --host-budget-mb "$host_budget_mb" \
  --backend-budget-mb "$backend_budget_mb" \
  --kv-budget-mb "$kv_budget_mb" \
  --scratch-budget-mb "$scratch_budget_mb" 2>&1 | tee "$generate_log"

python3 - "$json_file" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

errors = []
if payload.get("backend") != "cuda":
    errors.append(f"backend={payload.get('backend')!r}, expected cuda")
tokens = payload.get("tokens")
if not isinstance(tokens, int) or tokens <= 0:
    errors.append(f"tokens={tokens!r}, expected > 0")
tok_s = payload.get("decode_tok_per_s")
if not isinstance(tok_s, (int, float)) or not math.isfinite(tok_s) or tok_s <= 0:
    errors.append(f"decode_tok_per_s={tok_s!r}, expected finite > 0")
cuda_generate = payload.get("cuda_generate")
if not isinstance(cuda_generate, dict):
    errors.append("missing cuda_generate stats")
else:
    launches = cuda_generate.get("kernel_launches")
    linear = cuda_generate.get("launch_linear")
    if not isinstance(launches, int) or launches <= 0:
        errors.append(f"cuda_generate.kernel_launches={launches!r}, expected > 0")
    if not isinstance(linear, int) or linear <= 0:
        errors.append(f"cuda_generate.launch_linear={linear!r}, expected > 0")

cuda_stats = payload.get("cuda")
if not isinstance(cuda_stats, dict):
    errors.append("missing cuda stats")
else:
    device_kv_successes = cuda_stats.get("device_kv_successes")
    if not isinstance(device_kv_successes, int) or device_kv_successes <= 0:
        errors.append(f"cuda.device_kv_successes={device_kv_successes!r}, expected > 0")
    for key in (
        "device_kv_fail_batch",
        "device_kv_fail_no_cache",
        "device_kv_fail_no_storage",
        "device_kv_fail_no_hook",
        "device_kv_fail_write",
        "device_kv_fail_read",
        "device_kv_fail_shape",
    ):
        value = cuda_stats.get(key)
        if value != 0:
            errors.append(f"cuda.{key}={value!r}, expected 0")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

print(
    "PASS qwen36_cuda_generate: "
    f"tokens={tokens} finish_reason={payload.get('finish_reason')!r} "
    f"decode_tok_per_s={tok_s:.3f} json={path}"
)
PY

if [ "$production_gate" = "1" ]; then
  chat_log="$out_dir/qwen36-chat-template.log"
  log "running Qwen3.6 chat-template gate"
  timeout "$command_timeout" "$bin" generate "$model_path" "$prod_prompt" \
    --backend cuda \
    --max-tokens 0 \
    --temperature 0 \
    --print-chat-template-status \
    --cache-dtype "$cache_dtype" \
    --combined-budget-mb "$combined_budget_mb" \
    --host-budget-mb "$host_budget_mb" \
    --backend-budget-mb "$backend_budget_mb" \
    --kv-budget-mb "$kv_budget_mb" \
    --scratch-budget-mb "$scratch_budget_mb" 2>&1 | tee "$chat_log"
  if grep -q "chat template init failed" "$chat_log"; then
    echo "chat template init failed" >&2
    exit 1
  fi
  grep -q "chat_template=true" "$chat_log"

  prod_log="$out_dir/qwen36-production.log"
  prod_json="$out_dir/qwen36-production.json"
  log "running Qwen3.6 CUDA production gate"
  timeout "$command_timeout" "$bin" generate "$model_path" "$prod_prompt" \
    --backend cuda \
    --max-tokens "$prod_max_tokens" \
    --temperature 0 \
    --raw-prompt \
    --no-chat-template \
    --print-token-count \
    --print-finish-reason \
    --print-timing \
    --json-timing "$prod_json" \
    --cache-dtype "$cache_dtype" \
    --combined-budget-mb "$combined_budget_mb" \
    --host-budget-mb "$host_budget_mb" \
    --backend-budget-mb "$backend_budget_mb" \
    --kv-budget-mb "$kv_budget_mb" \
    --scratch-budget-mb "$scratch_budget_mb" 2>&1 | tee "$prod_log"

  python3 - "$prod_json" "$prod_max_tokens" "$min_tok_per_s" <<'PY'
import json
import math
import sys

path = sys.argv[1]
expected_tokens = int(sys.argv[2])
min_tok_per_s = float(sys.argv[3])
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

errors = []
if payload.get("backend") != "cuda":
    errors.append(f"backend={payload.get('backend')!r}, expected cuda")
tokens = payload.get("tokens")
if not isinstance(tokens, int) or tokens < expected_tokens:
    errors.append(f"tokens={tokens!r}, expected >= {expected_tokens}")
tok_s = payload.get("decode_tok_per_s")
if not isinstance(tok_s, (int, float)) or not math.isfinite(tok_s) or tok_s < min_tok_per_s:
    errors.append(f"decode_tok_per_s={tok_s!r}, expected >= {min_tok_per_s}")

cuda_generate = payload.get("cuda_generate")
if not isinstance(cuda_generate, dict):
    errors.append("missing cuda_generate stats")
else:
    if cuda_generate.get("qwen35_decode_core_fused", 0) <= 0:
        errors.append(f"cuda_generate.qwen35_decode_core_fused={cuda_generate.get('qwen35_decode_core_fused')!r}, expected > 0")
    if cuda_generate.get("qwen35_decode_core_fallbacks", 0) != 0:
        errors.append(f"cuda_generate.qwen35_decode_core_fallbacks={cuda_generate.get('qwen35_decode_core_fallbacks')!r}, expected 0")
    if cuda_generate.get("linear_pair_fused_iq3_s", 0) <= 0:
        errors.append(f"cuda_generate.linear_pair_fused_iq3_s={cuda_generate.get('linear_pair_fused_iq3_s')!r}, expected > 0")

cuda_stats = payload.get("cuda")
if not isinstance(cuda_stats, dict):
    errors.append("missing cuda stats")
else:
    if cuda_stats.get("host_attention_fallbacks", 0) != 0:
        errors.append(f"cuda.host_attention_fallbacks={cuda_stats.get('host_attention_fallbacks')!r}, expected 0")
    device_kv_successes = cuda_stats.get("device_kv_successes")
    if not isinstance(device_kv_successes, int) or device_kv_successes <= 0:
        errors.append(f"cuda.device_kv_successes={device_kv_successes!r}, expected > 0")
    for key in (
        "device_kv_fail_batch",
        "device_kv_fail_no_cache",
        "device_kv_fail_no_storage",
        "device_kv_fail_no_hook",
        "device_kv_fail_write",
        "device_kv_fail_read",
        "device_kv_fail_shape",
    ):
        value = cuda_stats.get(key)
        if value != 0:
            errors.append(f"cuda.{key}={value!r}, expected 0")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

print(
    "PASS qwen36_cuda_production_gate: "
    f"tokens={tokens} decode_tok_per_s={tok_s:.3f} "
    f"qwen35_decode_core_fused={cuda_generate.get('qwen35_decode_core_fused')} "
    f"linear_pair_fused_iq3_s={cuda_generate.get('linear_pair_fused_iq3_s')} "
    f"json={path}"
)
PY
fi

log "PASS qwen36 CUDA verifier"
log "outputs: $out_dir"
