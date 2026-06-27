#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/gemma4_cuda_production_gate.sh

Runs the Gemma4 CUDA production-readiness gate against local models.

Environment:
  ROOT_DIR=/path/to/antfly
  ZIG=.tools/zig-x86_64-linux-0.16.0/zig
  BIN=zig/pkg/inference/zig-out/bin/antfly-inference
  OUT_DIR=/tmp/gemma4-cuda-production-gate-<timestamp>

  RUN_BUILD=1
  RUN_SMOKE=1
  RUN_MICROBENCH=0
  RUN_DEFAULT_POLICY=1
  RUN_TARGET_ONLY=1
  RUN_E4B_QAT=auto
  RUN_E4B_QAT_LONG=off
  RUN_E4B_QAT_RESIDENT=off
  RUN_E4B_QAT_RESIDENT_SOAK=off
  RUN_E4B_QAT_RESIDENT_BACKPRESSURE=off
  RUN_E4B_QAT_COMPRESSED_KV=off
  RUN_E4B_Q4K_BASELINE=auto
  RUN_E4B_Q4K_RESIDENT_BASELINE=auto
  RUN_E4B_QAT_PROVIDER_COMPARISON=off
  RUN_E4B_QAT_PROVIDER_BENCHMARK=off
  RUN_E4B_QAT_COMPETITIVE_FLOOR=off
  RUN_E4B_QAT_MTP=auto
  RUN_E4B_QAT_MTP_TARGET_EQUIV=off
  RUN_E4B_QAT_MTP_REPLAY_STABILITY=auto
  RUN_E4B_QAT_MTP_REPLAY_512=off
  RUN_E4B_QAT_MTP_HIDDEN_AB=off
  RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX=off
  RUN_E4B_QAT_MTP_DONOR_MATRIX=off
  RUN_E4B_QAT_MTP_BENEFIT=off
  RUN_12B_MTP=1
  RUN_E2B_MTP=1

  MAX_TOKENS=16
  E4B_QAT_TOKENS=512
  E4B_QAT_REPEATS=2
  E4B_QAT_LONG_TOKENS=1024
  E4B_QAT_LONG_MIN_TOKENS=768
  E4B_QAT_LONG_KV_BUDGET_MB=1024
  E4B_QAT_LONG_FORCE_KV_CAPACITY=2048
  E4B_QAT_LONG_MIN_GRAPH_REPLAYS=auto
  E4B_QAT_LONG_MAX_LAUNCHES_PER_TOKEN=22
  E4B_QAT_LONG_MAX_DOWNLOAD_SYNCS=4
  E4B_QAT_COMPRESSED_KV_DTYPE=polar4
  E4B_QAT_COMPRESSED_KV_TOKENS=512
  E4B_QAT_COMPRESSED_KV_MIN_TOKENS=512
  E4B_QAT_COMPRESSED_KV_MIN_TOK_S=36.0
  E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS=auto
  E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS=4
  E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS=0
  E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS=1
  E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES=1
  E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS=1
  E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS=1
  E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA=1
  E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES=0
  E4B_QAT_COMPRESSED_KV_TURBOQUANT_MIN_TOKENS=0
  E4B_QAT_RESIDENT_TOKENS=512
  E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS=512
  E4B_QAT_RESIDENT_WARM_REPEATS=2
  E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY=required
  E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS=auto
  E4B_QAT_RESIDENT_PORT=auto
  E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=
  E4B_QAT_RESIDENT_SOAK_REQUESTS=6
  E4B_QAT_RESIDENT_SOAK_CONCURRENCY=2
  E4B_QAT_RESIDENT_SOAK_TOKENS=256
  E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS=256
  E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS=auto
  E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS=4
  E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY=4
  E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS=256
  E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED=1
  E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED=1
  E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS=2000
  E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS=auto
  E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY=required
  MIN_E4B_QAT_RESIDENT_WARM_TOK_S=24.0
  MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S=14.0
  MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S=8.0
  E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS=35000
  MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO=1.25
  MIN_E4B_QAT_PROVIDER_RATIO=1.0
  E4B_QAT_COMPETITIVE_FLOORS="compressed_kv_decode_tok_s=36.0"
  E4B_QAT_REQUIRE_PROVIDER_METADATA=1
  E4B_QAT_PROVIDER_BASELINE_JSON=
  E4B_QAT_PROVIDER_BASELINE_INLINE=
  E4B_QAT_PROVIDER_BASE_URL=
  E4B_QAT_PROVIDER_API_KEY_ENV=PROVIDER_API_KEY
  E4B_QAT_PROVIDER_NAME=
  E4B_QAT_PROVIDER_MODEL=google/gemma-4-E4B-it-qat-q4_0-gguf
  E4B_QAT_PROVIDER_HARDWARE=
  E4B_QAT_PROVIDER_SOURCE_URL=
  E4B_QAT_PROVIDER_STREAM=0
  E4B_QAT_PROVIDER_RATE_SOURCE=e2e
  E4B_QAT_PROVIDER_REPEATS=2
  E4B_QAT_PROVIDER_WARMUP=1
  E4B_QAT_PROVIDER_BASELINE_STATS=avg,median,min
  E4B_QAT_MTP_TOKENS=128
  E4B_QAT_MTP_SPEC_KS="1 2 4"
  E4B_QAT_MTP_PROMPT_FILTER="ants_chat code_chat"
  E4B_QAT_MTP_MODE=production
  E4B_QAT_MTP_CACHE_DTYPE=
  E4B_QAT_MTP_TURBOQUANT_MIN_TOKENS=0
  E4B_QAT_MTP_REPLAY_STABILITY_TOKENS=128
  E4B_QAT_MTP_REPLAY_STABILITY_SPEC_KS="1 2"
  E4B_QAT_MTP_REPLAY_STABILITY_PROMPT_FILTER="ants_chat code_chat"
  E4B_QAT_MTP_REPLAY_TOKENS=512
  E4B_QAT_MTP_REPLAY_SPEC_KS="2"
  E4B_QAT_MTP_REPLAY_PROMPT_FILTER="ants_chat code_chat"
  E4B_QAT_MTP_HIDDEN_AB_REPEATS=2
  E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO=1.03
  E4B_QAT_MTP_TARGET_EQUIV_TOKENS=16
  E4B_QAT_MTP_TARGET_EQUIV_SPEC_KS="1 2"
  E4B_QAT_MTP_TARGET_EQUIV_PROMPT_FILTER="ants_chat factual_chat explain_chat code_chat"
  E4B_QAT_MTP_ACCEPTANCE_TOKENS=128
  E4B_QAT_MTP_ACCEPTANCE_SPEC_KS="1 2"
  E4B_QAT_MTP_ACCEPTANCE_PROMPT_FILTER="ants_chat code_chat"
  E4B_QAT_MTP_DONOR_MATRIX_TOKENS=128
  E4B_QAT_MTP_DONOR_MATRIX_SPEC_KS="1 2"
  E4B_QAT_MTP_DONOR_MATRIX_PROMPT_FILTER="ants_chat code_chat"
  E4B_QAT_MTP_DONOR_MATRIX_MODES="shared_type tail_base non_shared_tail_base first_shared_base"
  PROMPT_FILTER="ants_chat code_chat"
  SPEC_KS="1 2 4"
  MTP_MIN_ACTIVE_SPEED_RATIO=1.0
  MTP_MIN_BENEFIT_RATIO=1.02
  MTP_REPLAY_512_MIN_BENEFIT_RATIO=1.02
  E4B_QAT_MTP_REQUIRE_PREPROJECT_FUSION=1
  E4B_QAT_MTP_REQUIRE_MASKED_SELECT_FUSION=0
  RUN_TIMEOUT=420s   # set to off to avoid wrapping CUDA commands in timeout
  E4B_QAT_REQUIRE_FUSED=1
  E4B_QAT_Q4_0_GATED_DOWN_TILE8=0
  E4B_QAT_REQUIRE_GATED_DOWN_TILE8=0
  E4B_QAT_Q4_0_PLE_GATE_FUSION=1
  E4B_QAT_PLE_RMS_EMBED_FUSION=0
  E4B_QAT_REQUIRE_FAST_GQA=1
  E4B_QAT_DECODE_GRAPH_REPLAY=required
  E4B_QAT_REQUIRE_GRAPH_REPLAY=1
  E4B_QAT_REQUIRE_DEVICE_TOKEN_HANDOFF=1
  E4B_QAT_REQUIRE_RAW_TOKEN_EXPORT=1
  E4B_QAT_PENDING_TOKEN_READBACK=1
  E4B_QAT_MAX_DOWNLOAD_SYNCS=4
  E4B_QAT_REQUIRE_PLE_FUSION=1
  E4B_QAT_MIN_GRAPH_REPLAYS=auto
  E4B_QAT_MAX_LAUNCHES_PER_TOKEN=22.5
  E4B_QAT_TEMP_SLOT_PERIOD=0
  E4B_QAT_CAPTURE_ALLOW_UNPINNED=1
  E4B_QAT_CAPTURE_MIN_ALLOC_SEQ=10000
  E4B_QAT_FORCE_KV_CAPACITY=1024
  MIN_E4B_QAT_OVER_Q4K_RATIO=1.25
  MTP_VERIFY_DEVICE_RESULT=auto|0|1
  MIN_12B_Q8_TOK_S=13.0
  MIN_12B_Q4K_TOK_S=8.5
  MIN_E4B_QAT_TOK_S=24.0
  MIN_E4B_QAT_RUN_TOK_S=24.0
  MIN_E4B_QAT_LONG_TOK_S=15.0
  MIN_E2B_TOK_S=18.0

  GEMMA12_Q8=.models/google/gemma-4-12B-it-q8_0
  GEMMA12_Q4=.models/google/gemma-4-12B-it-q4_k
  E4B_QAT=.models/google/gemma-4-E4B-it-qat-q4_0-gguf
  E4B_Q4K_BASELINE=.models/google/gemma-4-E4B-it-q4_k
  E4B_QAT_ASSISTANT_Q8=.models/google/gemma-4-E4B-it-assistant
  E4B_QAT_ASSISTANT_Q4=
  GEMMA12_ASSISTANT_Q8=.models/google/gemma-4-12B-it-assistant
  GEMMA12_ASSISTANT_Q4=
  E2B_TARGET=.models/unsloth/gemma-4-E2B-it-qat-GGUF
  E2B_ASSISTANT_Q8=.models/unsloth/gemma-4-E2B-it-qat-GGUF/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf
  E2B_ASSISTANT_Q4=.models/unsloth/gemma-4-E2B-it-qat-GGUF/MTP/gemma-4-E2B-it-Q4_0-MTP.gguf
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ZIG="${ZIG:-$ROOT_DIR/.tools/zig-x86_64-linux-0.16.0/zig}"
BIN="${BIN:-$ROOT_DIR/zig/pkg/inference/zig-out/bin/antfly-inference}"
OUT_DIR="${OUT_DIR:-/tmp/gemma4-cuda-production-gate-$(date +%Y%m%d-%H%M%S)}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_MICROBENCH="${RUN_MICROBENCH:-0}"
RUN_DEFAULT_POLICY="${RUN_DEFAULT_POLICY:-1}"
RUN_TARGET_ONLY="${RUN_TARGET_ONLY:-1}"
RUN_E4B_QAT="${RUN_E4B_QAT:-auto}"
RUN_E4B_QAT_LONG="${RUN_E4B_QAT_LONG:-off}"
RUN_E4B_QAT_RESIDENT="${RUN_E4B_QAT_RESIDENT:-off}"
RUN_E4B_QAT_RESIDENT_SOAK="${RUN_E4B_QAT_RESIDENT_SOAK:-off}"
RUN_E4B_QAT_RESIDENT_BACKPRESSURE="${RUN_E4B_QAT_RESIDENT_BACKPRESSURE:-off}"
RUN_E4B_QAT_COMPRESSED_KV="${RUN_E4B_QAT_COMPRESSED_KV:-off}"
RUN_E4B_Q4K_BASELINE="${RUN_E4B_Q4K_BASELINE:-auto}"
RUN_E4B_Q4K_RESIDENT_BASELINE="${RUN_E4B_Q4K_RESIDENT_BASELINE:-auto}"
RUN_E4B_QAT_PROVIDER_COMPARISON="${RUN_E4B_QAT_PROVIDER_COMPARISON:-off}"
RUN_E4B_QAT_PROVIDER_BENCHMARK="${RUN_E4B_QAT_PROVIDER_BENCHMARK:-off}"
RUN_E4B_QAT_MTP="${RUN_E4B_QAT_MTP:-auto}"
RUN_E4B_QAT_MTP_TARGET_EQUIV="${RUN_E4B_QAT_MTP_TARGET_EQUIV:-off}"
RUN_E4B_QAT_MTP_REPLAY_STABILITY="${RUN_E4B_QAT_MTP_REPLAY_STABILITY:-auto}"
RUN_E4B_QAT_MTP_REPLAY_512="${RUN_E4B_QAT_MTP_REPLAY_512:-off}"
RUN_E4B_QAT_MTP_HIDDEN_AB="${RUN_E4B_QAT_MTP_HIDDEN_AB:-off}"
RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX="${RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX:-off}"
RUN_E4B_QAT_MTP_DONOR_MATRIX="${RUN_E4B_QAT_MTP_DONOR_MATRIX:-off}"
RUN_E4B_QAT_MTP_BENEFIT="${RUN_E4B_QAT_MTP_BENEFIT:-off}"
E4B_QAT_MTP_REQUIRE_PREPROJECT_FUSION="${E4B_QAT_MTP_REQUIRE_PREPROJECT_FUSION:-1}"
E4B_QAT_MTP_REQUIRE_MASKED_SELECT_FUSION="${E4B_QAT_MTP_REQUIRE_MASKED_SELECT_FUSION:-0}"
RUN_12B_MTP="${RUN_12B_MTP:-1}"
RUN_E2B_MTP="${RUN_E2B_MTP:-1}"

MAX_TOKENS="${MAX_TOKENS:-16}"
E4B_QAT_TOKENS="${E4B_QAT_TOKENS:-512}"
E4B_QAT_REPEATS="${E4B_QAT_REPEATS:-${RUN_E4B_QAT_REPEATS:-2}}"
E4B_QAT_LONG_TOKENS="${E4B_QAT_LONG_TOKENS:-1024}"
E4B_QAT_LONG_MIN_TOKENS="${E4B_QAT_LONG_MIN_TOKENS:-768}"
E4B_QAT_LONG_KV_BUDGET_MB="${E4B_QAT_LONG_KV_BUDGET_MB:-1024}"
E4B_QAT_LONG_FORCE_KV_CAPACITY="${E4B_QAT_LONG_FORCE_KV_CAPACITY:-2048}"
E4B_QAT_LONG_MIN_GRAPH_REPLAYS="${E4B_QAT_LONG_MIN_GRAPH_REPLAYS:-auto}"
E4B_QAT_LONG_MAX_LAUNCHES_PER_TOKEN="${E4B_QAT_LONG_MAX_LAUNCHES_PER_TOKEN:-22}"
E4B_QAT_LONG_MAX_DOWNLOAD_SYNCS="${E4B_QAT_LONG_MAX_DOWNLOAD_SYNCS:-4}"
E4B_QAT_COMPRESSED_KV_DTYPE="${E4B_QAT_COMPRESSED_KV_DTYPE:-polar4}"
E4B_QAT_COMPRESSED_KV_TOKENS="${E4B_QAT_COMPRESSED_KV_TOKENS:-512}"
E4B_QAT_COMPRESSED_KV_MIN_TOKENS="${E4B_QAT_COMPRESSED_KV_MIN_TOKENS:-$E4B_QAT_COMPRESSED_KV_TOKENS}"
E4B_QAT_COMPRESSED_KV_MIN_TOK_S="${E4B_QAT_COMPRESSED_KV_MIN_TOK_S:-36.0}"
E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS="${E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS:-auto}"
E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS="${E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS:-4}"
E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS="${E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS:-0}"
E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS="${E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS:-1}"
E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES="${E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES:-$E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS}"
E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS="${E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS:-1}"
E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS="${E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS:-1}"
E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA="${E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA:-1}"
E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES="${E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES:-$E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS}"
E4B_QAT_COMPRESSED_KV_TURBOQUANT_MIN_TOKENS="${E4B_QAT_COMPRESSED_KV_TURBOQUANT_MIN_TOKENS:-0}"
E4B_QAT_RESIDENT_TOKENS="${E4B_QAT_RESIDENT_TOKENS:-512}"
E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS="${E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS:-$E4B_QAT_RESIDENT_TOKENS}"
E4B_QAT_RESIDENT_WARM_REPEATS="${E4B_QAT_RESIDENT_WARM_REPEATS:-${RUN_E4B_QAT_RESIDENT_WARM_REPEATS:-2}}"
E4B_QAT_RESIDENT_PROMPT="${E4B_QAT_RESIDENT_PROMPT:-Write a detailed technical explanation of how database indexes improve read queries while slowing down writes. Include examples, tradeoffs, tuning advice, and operational caveats.}"
E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY="${E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY:-required}"
E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS="${E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS:-auto}"
E4B_QAT_RESIDENT_PORT="${E4B_QAT_RESIDENT_PORT:-}"
E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS="${E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS:-}"
E4B_QAT_RESIDENT_SOAK_REQUESTS="${E4B_QAT_RESIDENT_SOAK_REQUESTS:-6}"
E4B_QAT_RESIDENT_SOAK_CONCURRENCY="${E4B_QAT_RESIDENT_SOAK_CONCURRENCY:-2}"
E4B_QAT_RESIDENT_SOAK_TOKENS="${E4B_QAT_RESIDENT_SOAK_TOKENS:-256}"
E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS="${E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS:-$E4B_QAT_RESIDENT_SOAK_TOKENS}"
E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS="${E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS:-auto}"
E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS="${E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS:-4}"
E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY="${E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY:-4}"
E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS="${E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS:-256}"
E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED="${E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED:-1}"
E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED="${E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED:-1}"
E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS="${E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS:-2000}"
E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS="${E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS:-auto}"
E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY="${E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY:-required}"
PROMPT_FILTER="${PROMPT_FILTER:-ants_chat code_chat}"
SPEC_KS="${SPEC_KS:-1 2 4}"
MTP_MIN_ACTIVE_SPEED_RATIO="${MTP_MIN_ACTIVE_SPEED_RATIO:-1.0}"
MTP_MIN_BENEFIT_RATIO="${MTP_MIN_BENEFIT_RATIO:-1.02}"
MTP_REPLAY_512_MIN_BENEFIT_RATIO="${MTP_REPLAY_512_MIN_BENEFIT_RATIO:-1.02}"
RUN_E4B_QAT_COMPETITIVE_FLOOR="${RUN_E4B_QAT_COMPETITIVE_FLOOR:-off}"
E4B_QAT_COMPETITIVE_FLOORS="${E4B_QAT_COMPETITIVE_FLOORS:-compressed_kv_decode_tok_s=36.0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-420s}"
E4B_QAT_REQUIRE_FUSED="${E4B_QAT_REQUIRE_FUSED:-1}"
E4B_QAT_Q4_0_GATED_DOWN_TILE8="${E4B_QAT_Q4_0_GATED_DOWN_TILE8:-0}"
E4B_QAT_REQUIRE_GATED_DOWN_TILE8="${E4B_QAT_REQUIRE_GATED_DOWN_TILE8:-$E4B_QAT_Q4_0_GATED_DOWN_TILE8}"
E4B_QAT_Q4_0_PLE_GATE_FUSION="${E4B_QAT_Q4_0_PLE_GATE_FUSION:-1}"
E4B_QAT_PLE_RMS_EMBED_FUSION="${E4B_QAT_PLE_RMS_EMBED_FUSION:-0}"
E4B_QAT_REQUIRE_FAST_GQA="${E4B_QAT_REQUIRE_FAST_GQA:-1}"
E4B_QAT_DECODE_GRAPH_REPLAY="${E4B_QAT_DECODE_GRAPH_REPLAY:-required}"
E4B_QAT_REQUIRE_GRAPH_REPLAY="${E4B_QAT_REQUIRE_GRAPH_REPLAY:-1}"
E4B_QAT_REQUIRE_DEVICE_TOKEN_HANDOFF="${E4B_QAT_REQUIRE_DEVICE_TOKEN_HANDOFF:-1}"
E4B_QAT_REQUIRE_RAW_TOKEN_EXPORT="${E4B_QAT_REQUIRE_RAW_TOKEN_EXPORT:-1}"
E4B_QAT_PENDING_TOKEN_READBACK="${E4B_QAT_PENDING_TOKEN_READBACK:-1}"
E4B_QAT_MAX_DOWNLOAD_SYNCS="${E4B_QAT_MAX_DOWNLOAD_SYNCS:-4}"
E4B_QAT_REQUIRE_PLE_FUSION="${E4B_QAT_REQUIRE_PLE_FUSION:-1}"
E4B_QAT_MIN_GRAPH_REPLAYS="${E4B_QAT_MIN_GRAPH_REPLAYS:-auto}"
E4B_QAT_MAX_LAUNCHES_PER_TOKEN="${E4B_QAT_MAX_LAUNCHES_PER_TOKEN:-22.5}"
E4B_QAT_TEMP_SLOT_PERIOD="${E4B_QAT_TEMP_SLOT_PERIOD:-0}"
E4B_QAT_CAPTURE_ALLOW_UNPINNED="${E4B_QAT_CAPTURE_ALLOW_UNPINNED:-1}"
E4B_QAT_CAPTURE_MIN_ALLOC_SEQ="${E4B_QAT_CAPTURE_MIN_ALLOC_SEQ:-10000}"
E4B_QAT_FORCE_KV_CAPACITY="${E4B_QAT_FORCE_KV_CAPACITY:-1024}"
MICROBENCH_WARMUP_ITERS="${MICROBENCH_WARMUP_ITERS:-2}"
MICROBENCH_MEASURE_ITERS="${MICROBENCH_MEASURE_ITERS:-5}"
MTP_VERIFY_DEVICE_RESULT="${MTP_VERIFY_DEVICE_RESULT:-auto}"

COMBINED_BUDGET_MB="${COMBINED_BUDGET_MB:-22000}"
BACKEND_BUDGET_MB="${BACKEND_BUDGET_MB:-19000}"
KV_BUDGET_MB="${KV_BUDGET_MB:-512}"
SCRATCH_BUDGET_MB="${SCRATCH_BUDGET_MB:-1024}"
E2B_COMBINED_BUDGET_MB="${E2B_COMBINED_BUDGET_MB:-12000}"
E2B_BACKEND_BUDGET_MB="${E2B_BACKEND_BUDGET_MB:-9000}"
E2B_KV_BUDGET_MB="${E2B_KV_BUDGET_MB:-256}"
E2B_SCRATCH_BUDGET_MB="${E2B_SCRATCH_BUDGET_MB:-512}"
E4B_QAT_HOST_BUDGET_MB="${E4B_QAT_HOST_BUDGET_MB:-8000}"
E4B_QAT_COMBINED_BUDGET_MB="${E4B_QAT_COMBINED_BUDGET_MB:-18000}"
E4B_QAT_BACKEND_BUDGET_MB="${E4B_QAT_BACKEND_BUDGET_MB:-12000}"
E4B_QAT_KV_BUDGET_MB="${E4B_QAT_KV_BUDGET_MB:-512}"
E4B_QAT_SCRATCH_BUDGET_MB="${E4B_QAT_SCRATCH_BUDGET_MB:-1024}"
E4B_Q4K_HOST_BUDGET_MB="${E4B_Q4K_HOST_BUDGET_MB:-8000}"
E4B_Q4K_COMBINED_BUDGET_MB="${E4B_Q4K_COMBINED_BUDGET_MB:-18000}"
E4B_Q4K_BACKEND_BUDGET_MB="${E4B_Q4K_BACKEND_BUDGET_MB:-12000}"
E4B_Q4K_KV_BUDGET_MB="${E4B_Q4K_KV_BUDGET_MB:-512}"
E4B_Q4K_SCRATCH_BUDGET_MB="${E4B_Q4K_SCRATCH_BUDGET_MB:-1024}"

TARGET_BASELINE_MIN_RATIO="${TARGET_BASELINE_MIN_RATIO:-0.90}"
AUTO_MIN_RATIO="${AUTO_MIN_RATIO:-0.95}"
PROMOTION_RATIO="${PROMOTION_RATIO:-1.05}"
MIN_E4B_QAT_OVER_Q4K_RATIO="${MIN_E4B_QAT_OVER_Q4K_RATIO:-1.25}"
MIN_12B_Q8_TOK_S="${MIN_12B_Q8_TOK_S:-13.0}"
MIN_12B_Q4K_TOK_S="${MIN_12B_Q4K_TOK_S:-8.5}"
MIN_E4B_QAT_TOK_S="${MIN_E4B_QAT_TOK_S:-24.0}"
MIN_E4B_QAT_RUN_TOK_S="${MIN_E4B_QAT_RUN_TOK_S:-24.0}"
MIN_E4B_QAT_LONG_TOK_S="${MIN_E4B_QAT_LONG_TOK_S:-15.0}"
MIN_E4B_QAT_RESIDENT_WARM_TOK_S="${MIN_E4B_QAT_RESIDENT_WARM_TOK_S:-24.0}"
MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S="${MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S:-14.0}"
MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S="${MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S:-8.0}"
E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS="${E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS:-35000}"
MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO="${MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO:-1.25}"
MIN_E4B_QAT_PROVIDER_RATIO="${MIN_E4B_QAT_PROVIDER_RATIO:-1.0}"
E4B_QAT_REQUIRE_PROVIDER_METADATA="${E4B_QAT_REQUIRE_PROVIDER_METADATA:-1}"
MIN_E2B_TOK_S="${MIN_E2B_TOK_S:-18.0}"

GEMMA12_Q8="${GEMMA12_Q8:-$ROOT_DIR/.models/google/gemma-4-12B-it-q8_0}"
GEMMA12_Q4="${GEMMA12_Q4:-$ROOT_DIR/.models/google/gemma-4-12B-it-q4_k}"
E4B_QAT="${E4B_QAT:-$ROOT_DIR/.models/google/gemma-4-E4B-it-qat-q4_0-gguf}"
E4B_Q4K_BASELINE="${E4B_Q4K_BASELINE:-$ROOT_DIR/.models/google/gemma-4-E4B-it-q4_k}"
E4B_QAT_PROVIDER_BASELINE_JSON="${E4B_QAT_PROVIDER_BASELINE_JSON:-}"
E4B_QAT_PROVIDER_BASELINE_INLINE="${E4B_QAT_PROVIDER_BASELINE_INLINE:-}"
E4B_QAT_PROVIDER_BASE_URL="${E4B_QAT_PROVIDER_BASE_URL:-}"
E4B_QAT_PROVIDER_API_KEY_ENV="${E4B_QAT_PROVIDER_API_KEY_ENV:-PROVIDER_API_KEY}"
E4B_QAT_PROVIDER_ENDPOINT="${E4B_QAT_PROVIDER_ENDPOINT:-chat}"
E4B_QAT_PROVIDER_NO_AUTH="${E4B_QAT_PROVIDER_NO_AUTH:-0}"
E4B_QAT_PROVIDER_HEADER="${E4B_QAT_PROVIDER_HEADER:-}"
E4B_QAT_PROVIDER_NAME="${E4B_QAT_PROVIDER_NAME:-}"
E4B_QAT_PROVIDER_MODEL="${E4B_QAT_PROVIDER_MODEL:-google/gemma-4-E4B-it-qat-q4_0-gguf}"
E4B_QAT_PROVIDER_HARDWARE="${E4B_QAT_PROVIDER_HARDWARE:-}"
E4B_QAT_PROVIDER_SOURCE_URL="${E4B_QAT_PROVIDER_SOURCE_URL:-}"
E4B_QAT_PROVIDER_METRIC="${E4B_QAT_PROVIDER_METRIC:-resident_e2e_tok_s}"
E4B_QAT_PROVIDER_STREAM="${E4B_QAT_PROVIDER_STREAM:-0}"
E4B_QAT_PROVIDER_STREAM_INCLUDE_USAGE="${E4B_QAT_PROVIDER_STREAM_INCLUDE_USAGE:-1}"
E4B_QAT_PROVIDER_RATE_SOURCE="${E4B_QAT_PROVIDER_RATE_SOURCE:-e2e}"
E4B_QAT_PROVIDER_WORKLOAD="${E4B_QAT_PROVIDER_WORKLOAD:-antfly-resident-index-explanation-512}"
E4B_QAT_PROVIDER_TOKENS="${E4B_QAT_PROVIDER_TOKENS:-$E4B_QAT_RESIDENT_TOKENS}"
E4B_QAT_PROVIDER_MIN_COMPLETION_TOKENS="${E4B_QAT_PROVIDER_MIN_COMPLETION_TOKENS:-$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS}"
E4B_QAT_PROVIDER_REPEATS="${E4B_QAT_PROVIDER_REPEATS:-2}"
E4B_QAT_PROVIDER_WARMUP="${E4B_QAT_PROVIDER_WARMUP:-1}"
E4B_QAT_PROVIDER_BASELINE_STAT="${E4B_QAT_PROVIDER_BASELINE_STAT:-avg}"
E4B_QAT_PROVIDER_BASELINE_STATS="${E4B_QAT_PROVIDER_BASELINE_STATS:-avg,median,min}"
E4B_QAT_PROVIDER_MAX_TOKEN_FIELD="${E4B_QAT_PROVIDER_MAX_TOKEN_FIELD:-max_completion_tokens}"
E4B_QAT_PROVIDER_NOTES="${E4B_QAT_PROVIDER_NOTES:-gate-collected provider baseline}"
E4B_QAT_MTP_TOKENS="${E4B_QAT_MTP_TOKENS:-128}"
E4B_QAT_MTP_SPEC_KS="${E4B_QAT_MTP_SPEC_KS:-$SPEC_KS}"
E4B_QAT_MTP_PROMPT_FILTER="${E4B_QAT_MTP_PROMPT_FILTER:-$PROMPT_FILTER}"
E4B_QAT_MTP_MODE="${E4B_QAT_MTP_MODE:-production}"
E4B_QAT_MTP_CACHE_DTYPE="${E4B_QAT_MTP_CACHE_DTYPE:-}"
E4B_QAT_MTP_TURBOQUANT_MIN_TOKENS="${E4B_QAT_MTP_TURBOQUANT_MIN_TOKENS:-0}"
E4B_QAT_MTP_REPLAY_STABILITY_TOKENS="${E4B_QAT_MTP_REPLAY_STABILITY_TOKENS:-128}"
E4B_QAT_MTP_REPLAY_STABILITY_SPEC_KS="${E4B_QAT_MTP_REPLAY_STABILITY_SPEC_KS:-1 2}"
E4B_QAT_MTP_REPLAY_STABILITY_PROMPT_FILTER="${E4B_QAT_MTP_REPLAY_STABILITY_PROMPT_FILTER:-ants_chat code_chat}"
E4B_QAT_MTP_REPLAY_TOKENS="${E4B_QAT_MTP_REPLAY_TOKENS:-512}"
E4B_QAT_MTP_REPLAY_SPEC_KS="${E4B_QAT_MTP_REPLAY_SPEC_KS:-2}"
E4B_QAT_MTP_REPLAY_PROMPT_FILTER="${E4B_QAT_MTP_REPLAY_PROMPT_FILTER:-ants_chat code_chat}"
E4B_QAT_MTP_HIDDEN_AB_REPEATS="${E4B_QAT_MTP_HIDDEN_AB_REPEATS:-2}"
E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO="${E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO:-1.03}"
E4B_QAT_MTP_TARGET_EQUIV_TOKENS="${E4B_QAT_MTP_TARGET_EQUIV_TOKENS:-16}"
E4B_QAT_MTP_TARGET_EQUIV_SPEC_KS="${E4B_QAT_MTP_TARGET_EQUIV_SPEC_KS:-1 2}"
E4B_QAT_MTP_TARGET_EQUIV_PROMPT_FILTER="${E4B_QAT_MTP_TARGET_EQUIV_PROMPT_FILTER:-ants_chat factual_chat explain_chat code_chat}"
E4B_QAT_MTP_ACCEPTANCE_TOKENS="${E4B_QAT_MTP_ACCEPTANCE_TOKENS:-128}"
E4B_QAT_MTP_ACCEPTANCE_SPEC_KS="${E4B_QAT_MTP_ACCEPTANCE_SPEC_KS:-1 2}"
E4B_QAT_MTP_ACCEPTANCE_PROMPT_FILTER="${E4B_QAT_MTP_ACCEPTANCE_PROMPT_FILTER:-ants_chat code_chat}"
E4B_QAT_MTP_DONOR_MATRIX_TOKENS="${E4B_QAT_MTP_DONOR_MATRIX_TOKENS:-128}"
E4B_QAT_MTP_DONOR_MATRIX_SPEC_KS="${E4B_QAT_MTP_DONOR_MATRIX_SPEC_KS:-1 2}"
E4B_QAT_MTP_DONOR_MATRIX_PROMPT_FILTER="${E4B_QAT_MTP_DONOR_MATRIX_PROMPT_FILTER:-ants_chat code_chat}"
E4B_QAT_MTP_DONOR_MATRIX_MODES="${E4B_QAT_MTP_DONOR_MATRIX_MODES:-shared_type tail_base non_shared_tail_base first_shared_base}"
GEMMA12_ASSISTANT_Q8="${GEMMA12_ASSISTANT_Q8:-$ROOT_DIR/.models/google/gemma-4-12B-it-assistant}"
GEMMA12_ASSISTANT_Q4="${GEMMA12_ASSISTANT_Q4:-}"
E4B_QAT_ASSISTANT_Q8="${E4B_QAT_ASSISTANT_Q8:-$ROOT_DIR/.models/google/gemma-4-E4B-it-assistant}"
E4B_QAT_ASSISTANT_Q4="${E4B_QAT_ASSISTANT_Q4:-}"
E2B_TARGET="${E2B_TARGET:-$ROOT_DIR/.models/unsloth/gemma-4-E2B-it-qat-GGUF}"
E2B_ASSISTANT_Q8="${E2B_ASSISTANT_Q8:-$ROOT_DIR/.models/unsloth/gemma-4-E2B-it-qat-GGUF/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf}"
E2B_ASSISTANT_Q4="${E2B_ASSISTANT_Q4:-$ROOT_DIR/.models/unsloth/gemma-4-E2B-it-qat-GGUF/MTP/gemma-4-E2B-it-Q4_0-MTP.gguf}"

mkdir -p "$OUT_DIR"
STEPS_TSV="$OUT_DIR/steps.tsv"
printf "step\tstatus\tdetail\n" >"$STEPS_TSV"

FAILED=0
E4B_QAT_RESIDENT_SERVER_PID=""
E4B_QAT_PROVIDER_GENERATED_BASELINE_JSON=""

cleanup_e4b_qat_resident_server() {
  if [[ -n "${E4B_QAT_RESIDENT_SERVER_PID:-}" ]]; then
    if kill -0 "$E4B_QAT_RESIDENT_SERVER_PID" >/dev/null 2>&1; then
      kill "$E4B_QAT_RESIDENT_SERVER_PID" >/dev/null 2>&1 || true
      wait "$E4B_QAT_RESIDENT_SERVER_PID" >/dev/null 2>&1 || true
    fi
    E4B_QAT_RESIDENT_SERVER_PID=""
  fi
}

trap cleanup_e4b_qat_resident_server EXIT

record() {
  local step="$1"
  local status="$2"
  local detail="${3:-}"
  printf "%s\t%s\t%s\n" "$step" "$status" "$detail" >>"$STEPS_TSV"
  if [[ "$status" == "fail" ]]; then
    FAILED=1
  fi
}

write_cuda_environment() {
  local json_path="$OUT_DIR/cuda_environment.json"
  local parser="$ROOT_DIR/zig/pkg/inference/scripts/cuda_environment_from_smoke.py"
  if ! command -v python3 >/dev/null 2>&1; then
    record "cuda_environment" "skip" "python3 unavailable"
    return
  fi
  if [[ ! -e "$parser" ]]; then
    record "cuda_environment" "fail" "missing $parser"
    return
  fi
  if python3 "$parser" \
    --log "$OUT_DIR/cuda_smoke.log" \
    --out "$json_path" \
    --steps "$STEPS_TSV" \
    --step cuda_smoke; then
    local smoke_status
    smoke_status="$(python3 - "$json_path" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
print(data.get("status") or "unknown")
PY
)"
    if [[ "$smoke_status" == "ok" ]]; then
      record "cuda_environment" "ok" "$json_path"
    elif [[ "$smoke_status" == "fail" ]]; then
      record "cuda_environment" "fail" "$json_path status=$smoke_status"
    else
      record "cuda_environment" "skip" "$json_path status=$smoke_status"
    fi
  else
    record "cuda_environment" "fail" "$json_path parse_failed"
  fi
}

run_e4b_qat_provider_benchmark() {
  case "$RUN_E4B_QAT_PROVIDER_BENCHMARK" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_provider_benchmark" "fail" "invalid RUN_E4B_QAT_PROVIDER_BENCHMARK=$RUN_E4B_QAT_PROVIDER_BENCHMARK"
      return
      ;;
  esac
  if [[ "$RUN_E4B_QAT_PROVIDER_BENCHMARK" == "off" ]]; then
    record "e4b_qat_provider_benchmark" "skip" "RUN_E4B_QAT_PROVIDER_BENCHMARK=off"
    return
  fi
  if [[ -z "$E4B_QAT_PROVIDER_BASE_URL" ]]; then
    if [[ "$RUN_E4B_QAT_PROVIDER_BENCHMARK" == "required" ]]; then
      record "e4b_qat_provider_benchmark" "fail" "missing E4B_QAT_PROVIDER_BASE_URL"
    else
      record "e4b_qat_provider_benchmark" "skip" "missing E4B_QAT_PROVIDER_BASE_URL"
    fi
    return
  fi
  local missing=()
  [[ -n "$E4B_QAT_PROVIDER_NAME" ]] || missing+=(E4B_QAT_PROVIDER_NAME)
  [[ -n "$E4B_QAT_PROVIDER_HARDWARE" ]] || missing+=(E4B_QAT_PROVIDER_HARDWARE)
  [[ -n "$E4B_QAT_PROVIDER_SOURCE_URL" ]] || missing+=(E4B_QAT_PROVIDER_SOURCE_URL)
  if (( ${#missing[@]} )); then
    record "e4b_qat_provider_benchmark" "fail" "missing ${missing[*]}"
    return
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    record "e4b_qat_provider_benchmark" "fail" "python3 unavailable"
    return
  fi
  local benchmarker="$ROOT_DIR/zig/pkg/inference/scripts/gemma4_qat_provider_benchmark.py"
  if [[ ! -e "$benchmarker" ]]; then
    record "e4b_qat_provider_benchmark" "fail" "missing $benchmarker"
    return
  fi
  local summarizer="$ROOT_DIR/zig/pkg/inference/scripts/gemma4_qat_production_summary.py"
  if [[ ! -e "$summarizer" ]]; then
    record "e4b_qat_provider_benchmark" "fail" "missing $summarizer"
    return
  fi
  local json_path="$OUT_DIR/e4b_qat_provider_baselines.json"
  local rows_path="$OUT_DIR/e4b_qat_provider_baselines.tsv"
  local log_path="$OUT_DIR/e4b_qat_provider_benchmark.log"
  local validation_path="$OUT_DIR/e4b_qat_provider_baseline_validation.json"
  local validation_log="$OUT_DIR/e4b_qat_provider_baseline_validation.log"
  local -a benchmark_args=(
    --base-url "$E4B_QAT_PROVIDER_BASE_URL"
    --endpoint "$E4B_QAT_PROVIDER_ENDPOINT"
    --api-key-env "$E4B_QAT_PROVIDER_API_KEY_ENV"
    --provider "$E4B_QAT_PROVIDER_NAME"
    --model "$E4B_QAT_PROVIDER_MODEL"
    --hardware "$E4B_QAT_PROVIDER_HARDWARE"
    --source-url "$E4B_QAT_PROVIDER_SOURCE_URL"
    --metric "$E4B_QAT_PROVIDER_METRIC"
    --rate-source "$E4B_QAT_PROVIDER_RATE_SOURCE"
    --workload "$E4B_QAT_PROVIDER_WORKLOAD"
    --prompt "$E4B_QAT_RESIDENT_PROMPT"
    --tokens "$E4B_QAT_PROVIDER_TOKENS"
    --min-completion-tokens "$E4B_QAT_PROVIDER_MIN_COMPLETION_TOKENS"
    --repeats "$E4B_QAT_PROVIDER_REPEATS"
    --warmup "$E4B_QAT_PROVIDER_WARMUP"
    --baseline-stat "$E4B_QAT_PROVIDER_BASELINE_STAT"
    --baseline-stats "$E4B_QAT_PROVIDER_BASELINE_STATS"
    --max-token-field "$E4B_QAT_PROVIDER_MAX_TOKEN_FIELD"
    --min-ratio "$MIN_E4B_QAT_PROVIDER_RATIO"
    --notes "$E4B_QAT_PROVIDER_NOTES"
    --output "$json_path"
    --rows-tsv "$rows_path"
  )
  case "$E4B_QAT_PROVIDER_NO_AUTH" in
    1|true|True|on|ON|yes|YES)
      benchmark_args+=(--no-auth)
      ;;
  esac
  case "$E4B_QAT_PROVIDER_STREAM" in
    1|true|True|on|ON|yes|YES)
      benchmark_args+=(--stream)
      ;;
  esac
  case "$E4B_QAT_PROVIDER_STREAM_INCLUDE_USAGE" in
    0|false|False|off|OFF|no|NO)
      benchmark_args+=(--no-stream-include-usage)
      ;;
  esac
  if [[ -n "$E4B_QAT_PROVIDER_HEADER" ]]; then
    benchmark_args+=(--header "$E4B_QAT_PROVIDER_HEADER")
  fi
  if python3 "$benchmarker" "${benchmark_args[@]}" >"$log_path" 2>&1; then
    local detail
    detail="$(sed -n '1p' "$log_path" 2>/dev/null || true)"
    record "e4b_qat_provider_benchmark" "ok" "${detail:-$json_path}"
    if python3 "$summarizer" \
      --validate-provider-baselines-only \
      --provider-baseline "$json_path" \
      --output "$validation_path" >"$validation_log" 2>&1; then
      E4B_QAT_PROVIDER_GENERATED_BASELINE_JSON="$json_path"
      detail="$(sed -n '1p' "$validation_log" 2>/dev/null || true)"
      record "e4b_qat_provider_baseline_validation" "ok" "${detail:-$validation_path}"
    else
      detail="$(sed -n '1p' "$validation_log" 2>/dev/null || true)"
      record "e4b_qat_provider_baseline_validation" "fail" "${detail:-$validation_log}"
    fi
  else
    local detail
    detail="$(sed -n '1p' "$log_path" 2>/dev/null || true)"
    record "e4b_qat_provider_benchmark" "fail" "${detail:-$log_path}"
  fi
}

write_qat_production_summary() {
  local json_path="$OUT_DIR/e4b_qat_production_summary.json"
  local log_path="$OUT_DIR/e4b_qat_production_summary.log"
  local summarizer="$ROOT_DIR/zig/pkg/inference/scripts/gemma4_qat_production_summary.py"
  if ! command -v python3 >/dev/null 2>&1; then
    record "qat_production_summary" "skip" "python3 unavailable"
    return
  fi
  if [[ ! -e "$summarizer" ]]; then
    record "qat_production_summary" "fail" "missing $summarizer"
    return
  fi
  case "$RUN_E4B_QAT_PROVIDER_COMPARISON" in
    auto|required|off)
      ;;
    *)
      record "qat_production_summary" "fail" "invalid RUN_E4B_QAT_PROVIDER_COMPARISON=$RUN_E4B_QAT_PROVIDER_COMPARISON"
      return
      ;;
  esac
  case "$RUN_E4B_QAT_COMPETITIVE_FLOOR" in
    auto|required|off)
      ;;
    *)
      record "qat_production_summary" "fail" "invalid RUN_E4B_QAT_COMPETITIVE_FLOOR=$RUN_E4B_QAT_COMPETITIVE_FLOOR"
      return
      ;;
  esac
  case "$RUN_E4B_QAT_MTP_BENEFIT" in
    auto|required|off)
      ;;
    *)
      record "qat_production_summary" "fail" "invalid RUN_E4B_QAT_MTP_BENEFIT=$RUN_E4B_QAT_MTP_BENEFIT"
      return
      ;;
  esac
  local compressed_kv_summary_min_graph_replays="$E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS"
  if [[ "$compressed_kv_summary_min_graph_replays" == "auto" ]]; then
    if (( E4B_QAT_COMPRESSED_KV_MIN_TOKENS > 32 )); then
      compressed_kv_summary_min_graph_replays=$((E4B_QAT_COMPRESSED_KV_MIN_TOKENS - 32))
    else
      compressed_kv_summary_min_graph_replays=1
    fi
  fi
  local -a summary_args=(
    --out-dir "$OUT_DIR"
    --output "$json_path"
    --min-provider-ratio "$MIN_E4B_QAT_PROVIDER_RATIO"
    --min-compressed-kv-tok-s "$E4B_QAT_COMPRESSED_KV_MIN_TOK_S"
    --min-compressed-kv-tokens "$E4B_QAT_COMPRESSED_KV_MIN_TOKENS"
    --min-compressed-kv-graph-replays "$compressed_kv_summary_min_graph_replays"
    --max-compressed-kv-download-syncs "$E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS"
    --max-compressed-kv-capacity-skips "$E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS"
    --min-compressed-kv-compressed-v-reads "$E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS"
    --min-compressed-kv-compressed-v-writes "$E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES"
    --min-compressed-kv-paged-uploads "$E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS"
    --min-compressed-kv-identity-attention-reads "$E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS"
    --min-compressed-kv-fast-gqa "$E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA"
    --max-compressed-kv-fail-writes "$E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES"
    --min-target-qat-tok-s "$MIN_E4B_QAT_TOK_S"
    --min-target-qat-run-tok-s "$MIN_E4B_QAT_RUN_TOK_S"
    --min-target-qat-over-q4k-ratio "$MIN_E4B_QAT_OVER_Q4K_RATIO"
    --min-target-qat-tokens "$E4B_QAT_TOKENS"
    --min-target-qat-repeats "$E4B_QAT_REPEATS"
    --min-long-qat-tok-s "$MIN_E4B_QAT_LONG_TOK_S"
    --min-long-qat-tokens "$E4B_QAT_LONG_MIN_TOKENS"
    --min-resident-qat-tok-s "$MIN_E4B_QAT_RESIDENT_WARM_TOK_S"
    --min-resident-qat-over-q4k-ratio "$MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO"
    --min-resident-qat-tokens "$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS"
    --min-resident-qat-repeats "$E4B_QAT_RESIDENT_WARM_REPEATS"
    --min-soak-requests "$E4B_QAT_RESIDENT_SOAK_REQUESTS"
    --min-soak-aggregate-tok-s "$MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S"
    --min-soak-request-tok-s "$MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S"
    --max-soak-p95-e2e-ms "$E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS"
    --min-backpressure-accepted "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED"
    --min-backpressure-rejected "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED"
    --max-backpressure-reject-ms "$E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS"
    --min-mtp-active-speed-ratio "$MTP_MIN_ACTIVE_SPEED_RATIO"
    --min-mtp-benefit-ratio "$MTP_MIN_BENEFIT_RATIO"
    --min-mtp-replay-512-benefit-ratio "$MTP_REPLAY_512_MIN_BENEFIT_RATIO"
    --min-mtp-hidden-ab-ratio "$E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO"
    --min-mtp-hidden-ab-pairs "$E4B_QAT_MTP_HIDDEN_AB_REPEATS"
  )
  if [[ "$RUN_SMOKE" == "1" ]]; then
    summary_args+=(--require-cuda-environment)
  fi
  if [[ "$RUN_TARGET_ONLY" == "1" && "$RUN_E4B_QAT" != "off" && "$RUN_E4B_Q4K_BASELINE" != "off" && -e "$E4B_QAT" && -e "$E4B_Q4K_BASELINE" ]]; then
    summary_args+=(--require-target)
  fi
  if [[ "$RUN_E4B_QAT_COMPRESSED_KV" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-compressed-kv)
  fi
  if [[ "$RUN_E4B_QAT_LONG" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-long)
  fi
  if [[ "$RUN_E4B_QAT_RESIDENT" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-resident)
    if [[ "$RUN_E4B_Q4K_RESIDENT_BASELINE" != "off" && -e "$E4B_Q4K_BASELINE" ]]; then
      summary_args+=(--require-resident-q4k)
    fi
    if [[ "$RUN_E4B_QAT_RESIDENT_SOAK" != "off" ]]; then
      summary_args+=(--require-soak)
    fi
    if [[ "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" != "off" ]]; then
      summary_args+=(--require-backpressure)
    fi
  fi
  if [[ "$RUN_E4B_QAT_MTP" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-mtp)
    if [[ "$E4B_QAT_MTP_REQUIRE_PREPROJECT_FUSION" == "1" ]]; then
      summary_args+=(--require-mtp-preproject-fusion)
    fi
    if [[ "$E4B_QAT_MTP_REQUIRE_MASKED_SELECT_FUSION" == "1" ]]; then
      summary_args+=(--require-mtp-masked-select-fusion)
    fi
  fi
  if [[ "$RUN_E4B_QAT_MTP_TARGET_EQUIV" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-mtp-target-equivalence)
  fi
  if [[ "$RUN_E4B_QAT_MTP_HIDDEN_AB" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-mtp-hidden-ab)
  fi
  if [[ "$RUN_E4B_QAT_MTP_DONOR_MATRIX" != "off" && -e "$E4B_QAT" ]]; then
    summary_args+=(--require-mtp-donor-matrix)
  fi
  if [[ "$RUN_E4B_QAT_MTP_BENEFIT" == "required" ]]; then
    summary_args+=(--require-mtp-benefit)
  fi
  if [[ "$RUN_E4B_QAT_PROVIDER_COMPARISON" != "off" || -n "$E4B_QAT_PROVIDER_GENERATED_BASELINE_JSON" || "$RUN_E4B_QAT_PROVIDER_BENCHMARK" == "required" ]]; then
    if [[ -n "$E4B_QAT_PROVIDER_BASELINE_JSON" ]]; then
      summary_args+=(--provider-baseline "$E4B_QAT_PROVIDER_BASELINE_JSON")
    fi
    if [[ -n "$E4B_QAT_PROVIDER_GENERATED_BASELINE_JSON" ]]; then
      summary_args+=(--provider-baseline "$E4B_QAT_PROVIDER_GENERATED_BASELINE_JSON")
    fi
    if [[ -n "$E4B_QAT_PROVIDER_BASELINE_INLINE" ]]; then
      summary_args+=(--provider-baseline-json "$E4B_QAT_PROVIDER_BASELINE_INLINE")
    fi
    case "$E4B_QAT_REQUIRE_PROVIDER_METADATA" in
      0|false|False|off|OFF|no|NO)
        ;;
      *)
        summary_args+=(--require-provider-metadata)
        ;;
    esac
    if [[ "$RUN_E4B_QAT_PROVIDER_COMPARISON" == "required" || "$RUN_E4B_QAT_PROVIDER_BENCHMARK" == "required" || -n "$E4B_QAT_PROVIDER_BASELINE_JSON" || -n "$E4B_QAT_PROVIDER_GENERATED_BASELINE_JSON" || -n "$E4B_QAT_PROVIDER_BASELINE_INLINE" ]]; then
      summary_args+=(--require-provider-comparison)
    fi
  fi
  if [[ "$RUN_E4B_QAT_COMPETITIVE_FLOOR" != "off" ]]; then
    for floor in $E4B_QAT_COMPETITIVE_FLOORS; do
      summary_args+=(--competitive-floor "$floor")
    done
    if [[ "$RUN_E4B_QAT_COMPETITIVE_FLOOR" == "required" ]]; then
      summary_args+=(--require-competitive-floor)
    fi
  fi
  if python3 "$summarizer" "${summary_args[@]}" >"$log_path" 2>&1; then
    local summary_line
    summary_line="$(sed -n '1p' "$log_path" 2>/dev/null || true)"
    record "qat_production_summary" "ok" "${summary_line:-$json_path}"
  else
    local detail
    detail="$(sed -n '1p' "$log_path" 2>/dev/null || true)"
    record "qat_production_summary" "fail" "${detail:-$log_path}"
  fi
}

exists_path() {
  [[ -e "$1" ]]
}

run_logged() {
  local step="$1"
  local log="$2"
  shift 2
  echo "gate: $step"
  if "$@" >"$log" 2>&1; then
    record "$step" "ok" "$log"
  else
    local status=$?
    record "$step" "fail" "$log exit_$status"
  fi
}

run_logged_in_dir() {
  local step="$1"
  local dir="$2"
  local log="$3"
  shift 3
  echo "gate: $step"
  if (cd "$dir" && "$@") >"$log" 2>&1; then
    record "$step" "ok" "$log"
  else
    local status=$?
    record "$step" "fail" "$log exit_$status"
  fi
}

run_with_optional_timeout() {
  if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" && "$RUN_TIMEOUT" != "off" && "$RUN_TIMEOUT" != "none" ]]; then
    timeout "$RUN_TIMEOUT" "$@"
  else
    "$@"
  fi
}

choose_resident_port() {
  python3 - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

wait_for_e4b_qat_resident_server() {
  local url="$1"
  local log_path="$2"
  local attempts=0
  while [[ "$attempts" -lt 120 ]]; do
    if python3 - "$url" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
        response.read()
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    if ! kill -0 "$E4B_QAT_RESIDENT_SERVER_PID" >/dev/null 2>&1; then
      sed -n '1,220p' "$log_path" >&2 || true
      return 1
    fi
    attempts=$((attempts + 1))
    sleep 0.5
  done
  sed -n '1,220p' "$log_path" >&2 || true
  return 1
}

e4b_qat_resident_generate_request() {
  local label="$1"
  local endpoint="$2"
  local response_json="$3"
  local model="${4:-$E4B_QAT}"
  python3 - "$endpoint" "$response_json" "$model" "$E4B_QAT_RESIDENT_PROMPT" "$E4B_QAT_RESIDENT_TOKENS" "$label" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

url, response_path, model, prompt, tokens, label = sys.argv[1:7]
body = {
    "model": model,
    "backend": "cuda",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(tokens),
    "temperature": 0,
    "stream": False,
    "cache_dtype": "f32",
}
request = urllib.request.Request(
    url,
    data=json.dumps(body).encode("utf-8"),
    headers={"content-type": "application/json"},
    method="POST",
)
start = time.monotonic()
try:
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        status = response.status
except urllib.error.HTTPError as exc:
    raw = exc.read()
    status = exc.code
except Exception as exc:
    raise SystemExit(f"{label}: request failed: {exc}") from exc
elapsed_ms = (time.monotonic() - start) * 1000.0
with open(response_path, "wb") as f:
    f.write(raw)
if status != 200:
    raise SystemExit(f"{label}: HTTP {status}: {raw.decode('utf-8', errors='replace')[:1000]}")
try:
    data = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"{label}: response was not JSON: {exc}") from exc
if data.get("error"):
    raise SystemExit(f"{label}: API error: {data.get('error')}")
usage = data.get("usage") or {}
completion_tokens = int(usage.get("completion_tokens") or 0)
if completion_tokens <= 0:
    raise SystemExit(f"{label}: completion_tokens={completion_tokens}, expected > 0")
tok_s = completion_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
print(f"{label}\t{elapsed_ms:.1f}\t{completion_tokens}\t{tok_s:.3f}")
PY
}

e4b_qat_resident_soak_requests() {
  local endpoint="$1"
  local response_prefix="$2"
  local model="$3"
  local tokens="$4"
  local request_count="$5"
  local concurrency="$6"
  local min_completion_tokens="$7"
  local tsv_path="$8"
  local meta_json="$9"
  python3 - "$endpoint" "$response_prefix" "$model" "$E4B_QAT_RESIDENT_PROMPT" "$tokens" "$request_count" "$concurrency" "$min_completion_tokens" "$tsv_path" "$meta_json" <<'PY'
import concurrent.futures
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

(
    url,
    response_prefix,
    model,
    prompt,
    tokens_arg,
    request_count_arg,
    concurrency_arg,
    min_completion_tokens_arg,
    tsv_path,
    meta_json,
) = sys.argv[1:11]
tokens = int(tokens_arg)
request_count = int(request_count_arg)
concurrency = int(concurrency_arg)
min_completion_tokens = int(min_completion_tokens_arg)
response_prefix = pathlib.Path(response_prefix)
timeout_s = max(300, tokens * request_count * 3)

if request_count < 1:
    raise SystemExit("request_count must be positive")
if concurrency < 1:
    raise SystemExit("concurrency must be positive")

def run_one(index):
    label = f"e4b_qat_resident_soak{index}"
    body = {
        "model": model,
        "backend": "cuda",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": False,
        "cache_dtype": "f32",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except Exception as exc:
        raise RuntimeError(f"{label}: request failed: {exc}") from exc
    elapsed_ms = (time.monotonic() - start) * 1000.0
    response_path = response_prefix.with_name(f"{response_prefix.name}_{index}.json")
    response_path.write_bytes(raw)
    if status != 200:
        text = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {status}: {text[:1000]}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}: response was not JSON: {exc}") from exc
    if data.get("error"):
        raise RuntimeError(f"{label}: API error: {data.get('error')}")
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens < min_completion_tokens:
        raise RuntimeError(f"{label}: completion_tokens={completion_tokens} < {min_completion_tokens}")
    tok_s = completion_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    return {
        "index": index,
        "case": label,
        "e2e_ms": elapsed_ms,
        "completion_tokens": completion_tokens,
        "e2e_tok_s": tok_s,
    }

wall_start = time.monotonic()
rows = []
errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
    future_to_index = {executor.submit(run_one, index): index for index in range(1, request_count + 1)}
    for future in concurrent.futures.as_completed(future_to_index):
        try:
            rows.append(future.result())
        except Exception as exc:
            errors.append(str(exc))
wall_ms = (time.monotonic() - wall_start) * 1000.0
if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

rows.sort(key=lambda row: row["index"])
with open(tsv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case", "e2e_ms", "completion_tokens", "e2e_tok_s"], delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "case": row["case"],
                "e2e_ms": f"{row['e2e_ms']:.1f}",
                "completion_tokens": row["completion_tokens"],
                "e2e_tok_s": f"{row['e2e_tok_s']:.3f}",
            }
        )

total_completion_tokens = sum(row["completion_tokens"] for row in rows)
aggregate_tok_s = total_completion_tokens / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
pathlib.Path(meta_json).write_text(
    json.dumps(
        {
            "aggregate_tok_s": aggregate_tok_s,
            "concurrency": concurrency,
            "request_count": request_count,
            "tokens": tokens,
            "total_completion_tokens": total_completion_tokens,
            "wall_ms": wall_ms,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(
    f"requests={request_count} concurrency={concurrency} "
    f"total_completion_tokens={total_completion_tokens} wall_ms={wall_ms:.1f} "
    f"aggregate_tok_s={aggregate_tok_s:.3f}"
)
PY
}

run_e4b_qat_resident_soak() {
  local endpoint="$1"
  local server_log="$2"
  case "$RUN_E4B_QAT_RESIDENT_SOAK" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_resident_soak" "fail" "invalid RUN_E4B_QAT_RESIDENT_SOAK=$RUN_E4B_QAT_RESIDENT_SOAK"
      return 1
      ;;
  esac
  if [[ "$RUN_E4B_QAT_RESIDENT_SOAK" == "off" ]]; then
    record "e4b_qat_resident_soak" "skip" "RUN_E4B_QAT_RESIDENT_SOAK=off"
    return 0
  fi
  if ! [[ "$E4B_QAT_RESIDENT_SOAK_REQUESTS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_SOAK_REQUESTS" -lt 1 ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid E4B_QAT_RESIDENT_SOAK_REQUESTS=$E4B_QAT_RESIDENT_SOAK_REQUESTS"
    return 1
  fi
  if ! [[ "$E4B_QAT_RESIDENT_SOAK_CONCURRENCY" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_SOAK_CONCURRENCY" -lt 1 ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid E4B_QAT_RESIDENT_SOAK_CONCURRENCY=$E4B_QAT_RESIDENT_SOAK_CONCURRENCY"
    return 1
  fi
  if ! [[ "$E4B_QAT_RESIDENT_SOAK_TOKENS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_SOAK_TOKENS" -lt 1 ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid E4B_QAT_RESIDENT_SOAK_TOKENS=$E4B_QAT_RESIDENT_SOAK_TOKENS"
    return 1
  fi
  if ! [[ "$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS" -lt 1 ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS=$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS"
    return 1
  fi
  if [[ "$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS" -gt "$E4B_QAT_RESIDENT_SOAK_TOKENS" ]]; then
    record "e4b_qat_resident_soak" "fail" "E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS=$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS exceeds E4B_QAT_RESIDENT_SOAK_TOKENS=$E4B_QAT_RESIDENT_SOAK_TOKENS"
    return 1
  fi
  if [[ "$E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS" != "auto" && ! "$E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS" =~ ^[0-9]+$ ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS=$E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS"
    return 1
  fi
  if ! [[ "$MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S=$MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S"
    return 1
  fi
  if ! [[ "$E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    record "e4b_qat_resident_soak" "fail" "invalid E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS=$E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS"
    return 1
  fi

  local tsv_path="$OUT_DIR/e4b_qat_resident_soak.tsv"
  local meta_json="$OUT_DIR/e4b_qat_resident_soak_meta.json"
  local summary="$OUT_DIR/e4b_qat_resident_soak_summary.txt"
  local err_path="$OUT_DIR/e4b_qat_resident_soak.err"
  if ! e4b_qat_resident_soak_requests \
    "$endpoint" \
    "$OUT_DIR/e4b_qat_resident_soak" \
    "$E4B_QAT" \
    "$E4B_QAT_RESIDENT_SOAK_TOKENS" \
    "$E4B_QAT_RESIDENT_SOAK_REQUESTS" \
    "$E4B_QAT_RESIDENT_SOAK_CONCURRENCY" \
    "$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS" \
    "$tsv_path" \
    "$meta_json" >"$summary" 2>"$err_path"; then
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    record "e4b_qat_resident_soak" "fail" "${detail:-request_failed}"
    return 1
  fi

  if python3 - "$tsv_path" "$meta_json" "$server_log" "$MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S" "$MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S" "$E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS" "$E4B_QAT_RESIDENT_SOAK_REQUESTS" "$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS" "$E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY" "$E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS" "$E4B_QAT_RESIDENT_WARM_REPEATS" "$E4B_QAT_RESIDENT_TOKENS" "$E4B_QAT_RESIDENT_SOAK_TOKENS" >"$summary" 2>"$err_path" <<'PY'
import csv
import json
import math
import sys

(
    tsv_path,
    meta_json,
    log_path,
    min_agg_tok_s_arg,
    min_request_tok_s_arg,
    max_p95_e2e_ms_arg,
    expected_requests_arg,
    min_completion_tokens_arg,
    replay_mode,
    graph_floor_arg,
    warm_repeats_arg,
    warm_tokens_arg,
    soak_tokens_arg,
) = sys.argv[1:14]
min_agg_tok_s = float(min_agg_tok_s_arg)
min_request_tok_s = float(min_request_tok_s_arg)
max_p95_e2e_ms = float(max_p95_e2e_ms_arg)
expected_requests = int(expected_requests_arg)
min_completion_tokens = int(min_completion_tokens_arg)
warm_repeats = int(warm_repeats_arg)
warm_tokens = int(warm_tokens_arg)
soak_tokens = int(soak_tokens_arg)

with open(tsv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
meta = json.loads(open(meta_json, "r", encoding="utf-8").read())

errors = []
if len(rows) != expected_requests:
    errors.append(f"soak_rows={len(rows)} != {expected_requests}")
rates = []
latencies_ms = []
for row in rows:
    label = row.get("case", "unknown")
    try:
        completion_tokens = int(row["completion_tokens"])
        latency_ms = float(row["e2e_ms"])
        rate = float(row["e2e_tok_s"])
    except Exception as exc:
        errors.append(f"{label}: invalid row: {exc}")
        continue
    if completion_tokens < min_completion_tokens:
        errors.append(f"{label}: completion_tokens={completion_tokens} < {min_completion_tokens}")
    if rate <= 0:
        errors.append(f"{label}: e2e_tok_s={rate:.3f} <= 0")
    if rate < min_request_tok_s:
        errors.append(f"{label}: e2e_tok_s={rate:.3f} < {min_request_tok_s:.3f}")
    rates.append(rate)
    latencies_ms.append(latency_ms)

aggregate_tok_s = float(meta.get("aggregate_tok_s") or 0.0)
if aggregate_tok_s < min_agg_tok_s:
    errors.append(f"aggregate_tok_s={aggregate_tok_s:.3f} < {min_agg_tok_s:.3f}")

def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]

p50_e2e_ms = percentile(latencies_ms, 50)
p95_e2e_ms = percentile(latencies_ms, 95)
p99_e2e_ms = percentile(latencies_ms, 99)
max_e2e_ms = max(latencies_ms) if latencies_ms else 0.0
if p95_e2e_ms > max_p95_e2e_ms:
    errors.append(f"p95_e2e_ms={p95_e2e_ms:.1f} > {max_p95_e2e_ms:.1f}")

replays = 0
graph_floor = 0
if replay_mode == "required":
    if graph_floor_arg == "auto":
        requested_tokens = warm_repeats * warm_tokens + expected_requests * soak_tokens
        request_count = warm_repeats + expected_requests
        graph_floor = max(1, requested_tokens - request_count * 8)
    else:
        graph_floor = int(graph_floor_arg)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    replays = log_text.count("persistent_replayed") + log_text.count("instantiated_cached_replayed")
    if replays < graph_floor:
        errors.append(f"graph replays={replays} < {graph_floor}")
    for marker in (
        "unsafe_d2h_copy",
        "unsafe_h2d_copy",
        "unsafe_temp_alloc",
        "CudaGraphCaptureUnsafe",
        "persistent_replay_kv_capacity_exceeded",
        "cuda_graph_capture_probe: discarded",
        "CUDA_ERROR_ILLEGAL_ADDRESS",
    ):
        if marker in log_text:
            errors.append(f"server log contains {marker}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

min_rate = min(rates) if rates else 0.0
avg_rate = sum(rates) / len(rates) if rates else 0.0
print(
    f"requests={len(rows)} concurrency={meta.get('concurrency')} "
    f"aggregate_tok_s={aggregate_tok_s:.3f} min_e2e_tok_s={min_rate:.3f} "
    f"avg_e2e_tok_s={avg_rate:.3f} p50_e2e_ms={p50_e2e_ms:.1f} "
    f"p95_e2e_ms={p95_e2e_ms:.1f} p99_e2e_ms={p99_e2e_ms:.1f} "
    f"max_e2e_ms={max_e2e_ms:.1f} graph_replays={replays} graph_floor={graph_floor}"
)
PY
  then
    local summary_line
    summary_line="$(sed -n '1p' "$summary" 2>/dev/null || true)"
    record "e4b_qat_resident_soak" "ok" "${summary_line:-$summary}"
    return 0
  else
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    record "e4b_qat_resident_soak" "fail" "${detail:-$summary}"
    return 1
  fi
}

e4b_qat_resident_backpressure_requests() {
  local endpoint="$1"
  local response_prefix="$2"
  local model="$3"
  local tokens="$4"
  local request_count="$5"
  local concurrency="$6"
  local tsv_path="$7"
  python3 - "$endpoint" "$response_prefix" "$model" "$E4B_QAT_RESIDENT_PROMPT" "$tokens" "$request_count" "$concurrency" "$tsv_path" <<'PY'
import concurrent.futures
import csv
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

url, response_prefix, model, prompt, tokens_arg, request_count_arg, concurrency_arg, tsv_path = sys.argv[1:9]
tokens = int(tokens_arg)
request_count = int(request_count_arg)
concurrency = int(concurrency_arg)
response_prefix = pathlib.Path(response_prefix)

def run_one(index):
    label = f"e4b_qat_resident_backpressure{index}"
    body = {
        "model": model,
        "backend": "cuda",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0,
        "stream": False,
        "cache_dtype": "f32",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    except Exception as exc:
        raise RuntimeError(f"{label}: request failed: {exc}") from exc
    elapsed_ms = (time.monotonic() - start) * 1000.0
    response_path = response_prefix.with_name(f"{response_prefix.name}_{index}.json")
    response_path.write_bytes(raw)
    completion_tokens = 0
    tok_s = 0.0
    if status == 200:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label}: response was not JSON: {exc}") from exc
        if data.get("error"):
            raise RuntimeError(f"{label}: API error: {data.get('error')}")
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if completion_tokens <= 0:
            raise RuntimeError(f"{label}: completion_tokens={completion_tokens}, expected > 0")
        tok_s = completion_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    elif status != 503:
        text = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"{label}: HTTP {status}: {text[:1000]}")
    return {
        "index": index,
        "case": label,
        "status": status,
        "e2e_ms": elapsed_ms,
        "completion_tokens": completion_tokens,
        "e2e_tok_s": tok_s,
    }

rows = []
errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
    future_to_index = {executor.submit(run_one, index): index for index in range(1, request_count + 1)}
    for future in concurrent.futures.as_completed(future_to_index):
        try:
            rows.append(future.result())
        except Exception as exc:
            errors.append(str(exc))
if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

rows.sort(key=lambda row: row["index"])
with open(tsv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["case", "status", "e2e_ms", "completion_tokens", "e2e_tok_s"], delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "case": row["case"],
                "status": row["status"],
                "e2e_ms": f"{row['e2e_ms']:.1f}",
                "completion_tokens": row["completion_tokens"],
                "e2e_tok_s": f"{row['e2e_tok_s']:.3f}",
            }
        )
PY
}

run_e4b_qat_resident_backpressure() {
  local endpoint="$1"
  local server_log="$2"
  local metrics_url="$3"
  case "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_resident_backpressure" "fail" "invalid RUN_E4B_QAT_RESIDENT_BACKPRESSURE=$RUN_E4B_QAT_RESIDENT_BACKPRESSURE"
      return 1
      ;;
  esac
  if [[ "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" == "off" ]]; then
    record "e4b_qat_resident_backpressure" "skip" "RUN_E4B_QAT_RESIDENT_BACKPRESSURE=off"
    return 0
  fi
  if [[ -z "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" ]]; then
    record "e4b_qat_resident_backpressure" "fail" "E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS is required for backpressure gate"
    return 1
  fi
  for pair in \
    "E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS:$E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY:$E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS:$E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED:$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED:$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED"; do
    local name="${pair%%:*}"
    local value="${pair#*:}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
      record "e4b_qat_resident_backpressure" "fail" "invalid $name=$value"
      return 1
    fi
  done
  if ! [[ "$E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    record "e4b_qat_resident_backpressure" "fail" "invalid E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS=$E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS"
    return 1
  fi
  if [[ "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS" != "auto" && ! "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS" =~ ^[0-9]+$ ]]; then
    record "e4b_qat_resident_backpressure" "fail" "invalid E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS=$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS"
    return 1
  fi

  local tsv_path="$OUT_DIR/e4b_qat_resident_backpressure.tsv"
  local metrics_path="$OUT_DIR/e4b_qat_resident_backpressure_metrics.txt"
  local summary="$OUT_DIR/e4b_qat_resident_backpressure_summary.txt"
  local err_path="$OUT_DIR/e4b_qat_resident_backpressure.err"
  if ! e4b_qat_resident_backpressure_requests \
    "$endpoint" \
    "$OUT_DIR/e4b_qat_resident_backpressure" \
    "$E4B_QAT" \
    "$E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS" \
    "$E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS" \
    "$E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY" \
    "$tsv_path" >"$summary" 2>"$err_path"; then
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    record "e4b_qat_resident_backpressure" "fail" "${detail:-request_failed}"
    return 1
  fi

  if python3 - "$tsv_path" "$server_log" "$metrics_url" "$metrics_path" "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED" "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED" "$E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS" "$E4B_QAT_RESIDENT_WARM_REPEATS" "$E4B_QAT_RESIDENT_TOKENS" "$E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS" "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" "$E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY" "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS" >"$summary" 2>"$err_path" <<'PY'
import csv
import sys
import urllib.request

(
    tsv_path,
    log_path,
    metrics_url,
    metrics_path,
    min_accepted_arg,
    min_rejected_arg,
    max_reject_ms_arg,
    warm_repeats_arg,
    warm_tokens_arg,
    backpressure_tokens_arg,
    max_concurrent_arg,
    replay_mode,
    graph_floor_arg,
) = sys.argv[1:14]
min_accepted = int(min_accepted_arg)
min_rejected = int(min_rejected_arg)
max_reject_ms = float(max_reject_ms_arg)
warm_repeats = int(warm_repeats_arg)
warm_tokens = int(warm_tokens_arg)
backpressure_tokens = int(backpressure_tokens_arg)
max_concurrent = int(max_concurrent_arg)

with open(tsv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

errors = []
accepted = []
rejected = []
for row in rows:
    status = int(row["status"])
    elapsed_ms = float(row["e2e_ms"])
    if status == 200:
        tokens = int(row["completion_tokens"])
        rate = float(row["e2e_tok_s"])
        if tokens <= 0:
            errors.append(f"{row['case']}: completion_tokens={tokens}")
        if rate <= 0:
            errors.append(f"{row['case']}: e2e_tok_s={rate:.3f}")
        accepted.append(row)
    elif status == 503:
        if elapsed_ms > max_reject_ms:
            errors.append(f"{row['case']}: reject_ms={elapsed_ms:.1f} > {max_reject_ms:.1f}")
        rejected.append(row)
    else:
        errors.append(f"{row['case']}: unexpected status={status}")

if len(accepted) < min_accepted:
    errors.append(f"accepted={len(accepted)} < {min_accepted}")
if len(rejected) < min_rejected:
    errors.append(f"rejected={len(rejected)} < {min_rejected}")

try:
    with urllib.request.urlopen(metrics_url, timeout=5) as response:
        metrics_text = response.read().decode("utf-8", errors="replace")
except Exception as exc:
    errors.append(f"metrics scrape failed: {exc}")
    metrics_text = ""
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(metrics_text)

def metric_value(name):
    prefix = name + " "
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            try:
                return float(line[len(prefix):].strip())
            except ValueError:
                return None
    return None

metric_expectations = {
    "antfly_inference_request_queue_capacity": float(max_concurrent),
    "antfly_inference_request_queue_depth": 0.0,
    "antfly_inference_request_queue_available": float(max_concurrent),
    "antfly_inference_request_queue_active_requests": 0.0,
}
for name, expected in metric_expectations.items():
    value = metric_value(name)
    if value is None:
        errors.append(f"missing metric {name}")
    elif value != expected:
        errors.append(f"{name}={value:g} != {expected:g}")

rejection_metric = metric_value("antfly_inference_request_queue_rejections_total")
if rejection_metric is None:
    errors.append("missing metric antfly_inference_request_queue_rejections_total")
elif rejection_metric < len(rejected):
    errors.append(f"queue_rejections_total={rejection_metric:g} < rejected={len(rejected)}")

rejected_units_metric = metric_value("antfly_inference_request_queue_rejected_units_total")
queue_units_per_backpressure = 1 + (1 + 0) + max(backpressure_tokens // 256, 0)
rejected_units_floor = len(rejected) * min(max_concurrent, queue_units_per_backpressure)
if rejected_units_metric is None:
    errors.append("missing metric antfly_inference_request_queue_rejected_units_total")
elif rejected_units_metric < rejected_units_floor:
    errors.append(f"queue_rejected_units_total={rejected_units_metric:g} < floor={rejected_units_floor}")

replays = 0
graph_floor = 0
if replay_mode == "required":
    accepted_tokens = sum(int(row["completion_tokens"]) for row in accepted)
    if graph_floor_arg == "auto":
        graph_floor = max(1, warm_repeats * warm_tokens + accepted_tokens - (warm_repeats + len(accepted)) * 8)
    else:
        graph_floor = int(graph_floor_arg)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    replays = log_text.count("persistent_replayed") + log_text.count("instantiated_cached_replayed")
    if replays < graph_floor:
        errors.append(f"graph replays={replays} < {graph_floor}")
    for marker in (
        "unsafe_d2h_copy",
        "unsafe_h2d_copy",
        "unsafe_temp_alloc",
        "CudaGraphCaptureUnsafe",
        "persistent_replay_kv_capacity_exceeded",
        "cuda_graph_capture_probe: discarded",
        "CUDA_ERROR_ILLEGAL_ADDRESS",
    ):
        if marker in log_text:
            errors.append(f"server log contains {marker}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

accepted_rates = [float(row["e2e_tok_s"]) for row in accepted]
reject_ms = [float(row["e2e_ms"]) for row in rejected]
avg_accepted = sum(accepted_rates) / len(accepted_rates) if accepted_rates else 0.0
max_reject = max(reject_ms) if reject_ms else 0.0
print(
    f"requests={len(rows)} accepted={len(accepted)} rejected={len(rejected)} "
    f"avg_accepted_tok_s={avg_accepted:.3f} max_reject_ms={max_reject:.1f} "
    f"graph_replays={replays} graph_floor={graph_floor} "
    f"queue_rejections={rejection_metric:.0f} queue_rejected_units={rejected_units_metric:.0f}"
)
PY
  then
    local summary_line
    summary_line="$(sed -n '1p' "$summary" 2>/dev/null || true)"
    record "e4b_qat_resident_backpressure" "ok" "${summary_line:-$summary}"
    return 0
  else
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    record "e4b_qat_resident_backpressure" "fail" "${detail:-$summary}"
    return 1
  fi
}

run_target_generate() {
  local label="$1"
  local model="$2"
  local combined="$3"
  local backend="$4"
  local kv="$5"
  local scratch="$6"
  local host="${7:-}"
  local tokens="${8:-$MAX_TOKENS}"
  local json_path="$OUT_DIR/${label}.json"
  local log_path="$OUT_DIR/${label}.log"
  rm -f "$json_path"
  if ! exists_path "$model"; then
    record "$label" "skip" "missing $model"
    return
  fi
  local -a budget_args=(
    --combined-budget-mb "$combined"
    --backend-budget-mb "$backend"
    --kv-budget-mb "$kv"
    --scratch-budget-mb "$scratch"
  )
  if [[ -n "$host" ]]; then
    budget_args+=(--host-budget-mb "$host")
  fi
  run_logged "$label" "$log_path" \
    run_with_optional_timeout "$BIN" generate "$model" "Write one sentence about ants." \
      --backend cuda \
      --max-tokens "$tokens" \
      --temperature 0 \
      "${budget_args[@]}" \
      --print-timing \
      --print-token-count \
      --json-timing "$json_path" \
      --raw-prompt \
      --no-chat-template
}

run_e4b_qat_target() {
  case "$RUN_E4B_QAT" in
    auto|required|off)
      ;;
    *)
      record "target_e4b_qat" "fail" "invalid RUN_E4B_QAT=$RUN_E4B_QAT"
      return
      ;;
  esac
  if [[ "$RUN_E4B_QAT" == "off" ]]; then
    record "target_e4b_qat" "skip" "RUN_E4B_QAT=off"
    return
  fi
  if ! exists_path "$E4B_QAT"; then
    if [[ "$RUN_E4B_QAT" == "required" ]]; then
      record "target_e4b_qat" "fail" "missing $E4B_QAT"
    else
      record "target_e4b_qat" "skip" "missing $E4B_QAT"
    fi
    return
  fi
  if ! [[ "$E4B_QAT_REPEATS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_REPEATS" -lt 1 ]]; then
    record "target_e4b_qat" "fail" "invalid E4B_QAT_REPEATS=$E4B_QAT_REPEATS"
    return
  fi

  local repeat
  for ((repeat = 1; repeat <= E4B_QAT_REPEATS; repeat++)); do
    local label="target_e4b_qat"
    if [[ "$E4B_QAT_REPEATS" -gt 1 ]]; then
      label="target_e4b_qat_run${repeat}"
    fi
    run_e4b_qat_target_once "$label"
  done
}

run_e4b_qat_target_once() {
  local label="$1"
  local json_path="$OUT_DIR/${label}.json"
  local log_path="$OUT_DIR/${label}.log"
  rm -f "$json_path"

  local -a generate_cmd=(
    "$BIN" generate "$E4B_QAT" "Write one sentence about ants."
    --backend cuda
    --max-tokens "$E4B_QAT_TOKENS"
    --temperature 0
    --host-budget-mb "$E4B_QAT_HOST_BUDGET_MB"
    --combined-budget-mb "$E4B_QAT_COMBINED_BUDGET_MB"
    --backend-budget-mb "$E4B_QAT_BACKEND_BUDGET_MB"
    --kv-budget-mb "$E4B_QAT_KV_BUDGET_MB"
    --scratch-budget-mb "$E4B_QAT_SCRATCH_BUDGET_MB"
    --print-timing
    --print-token-count
    --json-timing "$json_path"
    --raw-prompt
    --no-chat-template
  )
  local -a env_cmd=(
    env
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$E4B_QAT_DECODE_GRAPH_REPLAY"
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$E4B_QAT_TEMP_SLOT_PERIOD"
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$E4B_QAT_CAPTURE_ALLOW_UNPINNED"
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$E4B_QAT_CAPTURE_MIN_ALLOC_SEQ"
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$E4B_QAT_FORCE_KV_CAPACITY"
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$E4B_QAT_Q4_0_GATED_DOWN_TILE8"
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$E4B_QAT_Q4_0_PLE_GATE_FUSION"
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$E4B_QAT_PLE_RMS_EMBED_FUSION"
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$E4B_QAT_PENDING_TOKEN_READBACK"
  )
  if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" && "$RUN_TIMEOUT" != "off" && "$RUN_TIMEOUT" != "none" ]]; then
    run_logged "$label" "$log_path" "${env_cmd[@]}" timeout "$RUN_TIMEOUT" "${generate_cmd[@]}"
  else
    run_logged "$label" "$log_path" "${env_cmd[@]}" "${generate_cmd[@]}"
  fi
}

run_e4b_qat_compressed_kv() {
  case "$RUN_E4B_QAT_COMPRESSED_KV" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_compressed_kv" "fail" "invalid RUN_E4B_QAT_COMPRESSED_KV=$RUN_E4B_QAT_COMPRESSED_KV"
      return
      ;;
  esac
  if [[ "$RUN_E4B_QAT_COMPRESSED_KV" == "off" ]]; then
    record "e4b_qat_compressed_kv" "skip" "RUN_E4B_QAT_COMPRESSED_KV=off"
    return
  fi
  if ! exists_path "$E4B_QAT"; then
    if [[ "$RUN_E4B_QAT_COMPRESSED_KV" == "required" ]]; then
      record "e4b_qat_compressed_kv" "fail" "missing $E4B_QAT"
    else
      record "e4b_qat_compressed_kv" "skip" "missing $E4B_QAT"
    fi
    return
  fi

  local label="e4b_qat_compressed_kv"
  local json_path="$OUT_DIR/${label}.json"
  local log_path="$OUT_DIR/${label}.log"
  local validation_path="$OUT_DIR/${label}_validation.txt"
  rm -f "$json_path" "$validation_path"

  local -a generate_cmd=(
    "$BIN" generate "$E4B_QAT" "Here is a sentence about ants:"
    --backend cuda
    --cache-dtype "$E4B_QAT_COMPRESSED_KV_DTYPE"
    --max-tokens "$E4B_QAT_COMPRESSED_KV_TOKENS"
    --temperature 0
    --host-budget-mb "$E4B_QAT_HOST_BUDGET_MB"
    --combined-budget-mb "$E4B_QAT_COMBINED_BUDGET_MB"
    --backend-budget-mb "$E4B_QAT_BACKEND_BUDGET_MB"
    --kv-budget-mb "$E4B_QAT_KV_BUDGET_MB"
    --scratch-budget-mb "$E4B_QAT_SCRATCH_BUDGET_MB"
    --print-timing
    --print-token-count
    --json-timing "$json_path"
    --raw-prompt
    --no-chat-template
  )
  local -a env_cmd=(
    env
    ANTFLY_INFERENCE_CUDA_TURBOQUANT_MIN_TOKENS="$E4B_QAT_COMPRESSED_KV_TURBOQUANT_MIN_TOKENS"
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$E4B_QAT_DECODE_GRAPH_REPLAY"
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$E4B_QAT_TEMP_SLOT_PERIOD"
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$E4B_QAT_CAPTURE_ALLOW_UNPINNED"
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$E4B_QAT_CAPTURE_MIN_ALLOC_SEQ"
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$E4B_QAT_FORCE_KV_CAPACITY"
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$E4B_QAT_Q4_0_GATED_DOWN_TILE8"
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$E4B_QAT_Q4_0_PLE_GATE_FUSION"
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$E4B_QAT_PLE_RMS_EMBED_FUSION"
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$E4B_QAT_PENDING_TOKEN_READBACK"
  )
  if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" && "$RUN_TIMEOUT" != "off" && "$RUN_TIMEOUT" != "none" ]]; then
    run_logged "$label" "$log_path" "${env_cmd[@]}" timeout "$RUN_TIMEOUT" "${generate_cmd[@]}"
  else
    run_logged "$label" "$log_path" "${env_cmd[@]}" "${generate_cmd[@]}"
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    if [[ "$RUN_E4B_QAT_COMPRESSED_KV" == "required" ]]; then
      record "e4b_qat_compressed_kv_validation" "fail" "python3 unavailable"
    else
      record "e4b_qat_compressed_kv_validation" "skip" "python3 unavailable"
    fi
    return
  fi
  if python3 - "$json_path" "$E4B_QAT_COMPRESSED_KV_MIN_TOKENS" "$E4B_QAT_COMPRESSED_KV_MIN_TOK_S" "$E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS" "$E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS" "$E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS" "$E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS" "$E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES" "$E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS" "$E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS" "$E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA" "$E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES" >"$validation_path" 2>&1 <<'PY'
import json
import math
import sys

(
    path,
    min_tokens_s,
    min_rate_s,
    min_replays_s,
    max_download_s,
    max_capacity_s,
    min_compressed_reads_s,
    min_compressed_writes_s,
    min_paged_uploads_s,
    min_identity_attention_reads_s,
    min_fast_gqa_s,
    max_fail_writes_s,
) = sys.argv[1:13]
with open(path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

tokens = int(data.get("tokens") or data.get("generated_tokens") or 0)
rate = float(data.get("decode_tok_per_s") or 0.0)
cuda = data.get("cuda") or {}
replays = int(cuda.get("graph_capture_persistent_replays") or 0)
downloads = int(cuda.get("download_syncs") or 0)
capacity_skips = int(cuda.get("graph_capture_capacity_skips") or 0)
compressed_reads = int(cuda.get("device_kv_compressed_v_reads") or 0)
compressed_writes = int(cuda.get("device_kv_compressed_v_writes") or 0)
paged_uploads = int(cuda.get("device_kv_paged_block_table_uploads") or 0)
identity_attention_reads = int(cuda.get("device_kv_paged_identity_attention_reads") or 0)
fast_gqa = int(cuda.get("launch_attention_gqa_decode_fast") or 0)
fail_write = int(cuda.get("device_kv_fail_write") or 0)

min_tokens = int(min_tokens_s)
min_rate = float(min_rate_s)
max_downloads = int(max_download_s)
max_capacity = int(max_capacity_s)
min_compressed_reads = int(min_compressed_reads_s)
min_compressed_writes = int(min_compressed_writes_s)
min_paged_uploads = int(min_paged_uploads_s)
min_identity_attention_reads = int(min_identity_attention_reads_s)
min_fast_gqa = int(min_fast_gqa_s)
max_fail_writes = int(max_fail_writes_s)
if min_replays_s == "auto":
    min_replays = max(1, min_tokens - 32)
else:
    min_replays = int(min_replays_s)

checks = [
    ("tokens", tokens >= min_tokens, f"{tokens} >= {min_tokens}"),
    ("decode_tok_per_s", rate >= min_rate and math.isfinite(rate), f"{rate:.3f} >= {min_rate:.3f}"),
    ("graph_replays", replays >= min_replays, f"{replays} >= {min_replays}"),
    ("download_syncs", downloads <= max_downloads, f"{downloads} <= {max_downloads}"),
    ("capacity_skips", capacity_skips <= max_capacity, f"{capacity_skips} <= {max_capacity}"),
    ("compressed_v_reads", compressed_reads >= min_compressed_reads, f"{compressed_reads} >= {min_compressed_reads}"),
    ("compressed_v_writes", compressed_writes >= min_compressed_writes, f"{compressed_writes} >= {min_compressed_writes}"),
    ("paged_uploads", paged_uploads >= min_paged_uploads, f"{paged_uploads} >= {min_paged_uploads}"),
    ("identity_attention_reads", identity_attention_reads >= min_identity_attention_reads, f"{identity_attention_reads} >= {min_identity_attention_reads}"),
    ("fast_gqa", fast_gqa >= min_fast_gqa, f"{fast_gqa} >= {min_fast_gqa}"),
    ("write_fallbacks", fail_write <= max_fail_writes, f"{fail_write} <= {max_fail_writes}"),
]

failed = [f"{name}: {detail}" for name, ok, detail in checks if not ok]
print(
    "compressed_kv "
    f"tokens={tokens} tok_s={rate:.3f} replays={replays} downloads={downloads} "
    f"capacity_skips={capacity_skips} compressed_reads={compressed_reads} compressed_writes={compressed_writes} "
    f"paged_uploads={paged_uploads} identity_attention_reads={identity_attention_reads} fast_gqa={fast_gqa} fail_write={fail_write}"
)
if failed:
    print("failed: " + "; ".join(failed))
    sys.exit(1)
PY
  then
    local detail
    detail="$(sed -n '1p' "$validation_path" 2>/dev/null || true)"
    record "e4b_qat_compressed_kv_validation" "ok" "${detail:-$validation_path}"
  else
    local detail
    detail="$(tail -1 "$validation_path" 2>/dev/null || true)"
    record "e4b_qat_compressed_kv_validation" "fail" "${detail:-$validation_path}"
  fi
}

run_e4b_qat_long_target() {
  case "$RUN_E4B_QAT_LONG" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_long" "fail" "invalid RUN_E4B_QAT_LONG=$RUN_E4B_QAT_LONG"
      return
      ;;
  esac
  if [[ "$RUN_E4B_QAT_LONG" == "off" ]]; then
    record "e4b_qat_long" "skip" "RUN_E4B_QAT_LONG=off"
    return
  fi
  if ! exists_path "$E4B_QAT"; then
    if [[ "$RUN_E4B_QAT_LONG" == "required" ]]; then
      record "e4b_qat_long" "fail" "missing $E4B_QAT"
    else
      record "e4b_qat_long" "skip" "missing $E4B_QAT"
    fi
    return
  fi
  if ! [[ "$E4B_QAT_LONG_TOKENS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_LONG_TOKENS" -lt 1 ]]; then
    record "e4b_qat_long" "fail" "invalid E4B_QAT_LONG_TOKENS=$E4B_QAT_LONG_TOKENS"
    return
  fi
  if ! [[ "$E4B_QAT_LONG_MIN_TOKENS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_LONG_MIN_TOKENS" -lt 1 ]]; then
    record "e4b_qat_long" "fail" "invalid E4B_QAT_LONG_MIN_TOKENS=$E4B_QAT_LONG_MIN_TOKENS"
    return
  fi
  if [[ "$E4B_QAT_LONG_MIN_TOKENS" -gt "$E4B_QAT_LONG_TOKENS" ]]; then
    record "e4b_qat_long" "fail" "E4B_QAT_LONG_MIN_TOKENS=$E4B_QAT_LONG_MIN_TOKENS exceeds E4B_QAT_LONG_TOKENS=$E4B_QAT_LONG_TOKENS"
    return
  fi

  local label="e4b_qat_long"
  local json_path="$OUT_DIR/${label}.json"
  local log_path="$OUT_DIR/${label}.log"
  rm -f "$json_path"

  local -a generate_cmd=(
    "$BIN" generate "$E4B_QAT" "Write one sentence about ants."
    --backend cuda
    --max-tokens "$E4B_QAT_LONG_TOKENS"
    --temperature 0
    --host-budget-mb "$E4B_QAT_HOST_BUDGET_MB"
    --combined-budget-mb "$E4B_QAT_COMBINED_BUDGET_MB"
    --backend-budget-mb "$E4B_QAT_BACKEND_BUDGET_MB"
    --kv-budget-mb "$E4B_QAT_LONG_KV_BUDGET_MB"
    --scratch-budget-mb "$E4B_QAT_SCRATCH_BUDGET_MB"
    --print-timing
    --print-token-count
    --json-timing "$json_path"
    --raw-prompt
    --no-chat-template
  )
  local -a env_cmd=(
    env
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$E4B_QAT_DECODE_GRAPH_REPLAY"
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$E4B_QAT_TEMP_SLOT_PERIOD"
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$E4B_QAT_CAPTURE_ALLOW_UNPINNED"
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$E4B_QAT_CAPTURE_MIN_ALLOC_SEQ"
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$E4B_QAT_LONG_FORCE_KV_CAPACITY"
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$E4B_QAT_Q4_0_GATED_DOWN_TILE8"
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$E4B_QAT_Q4_0_PLE_GATE_FUSION"
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$E4B_QAT_PLE_RMS_EMBED_FUSION"
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$E4B_QAT_PENDING_TOKEN_READBACK"
  )
  if [[ -n "$RUN_TIMEOUT" && "$RUN_TIMEOUT" != "0" && "$RUN_TIMEOUT" != "off" && "$RUN_TIMEOUT" != "none" ]]; then
    run_logged "$label" "$log_path" "${env_cmd[@]}" timeout "$RUN_TIMEOUT" "${generate_cmd[@]}"
  else
    run_logged "$label" "$log_path" "${env_cmd[@]}" "${generate_cmd[@]}"
  fi
}

run_e4b_qat_resident() {
  case "$RUN_E4B_QAT_RESIDENT_SOAK" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_resident_soak" "fail" "invalid RUN_E4B_QAT_RESIDENT_SOAK=$RUN_E4B_QAT_RESIDENT_SOAK"
      return
      ;;
  esac
  case "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_resident_backpressure" "fail" "invalid RUN_E4B_QAT_RESIDENT_BACKPRESSURE=$RUN_E4B_QAT_RESIDENT_BACKPRESSURE"
      return
      ;;
  esac
  case "$RUN_E4B_QAT_RESIDENT" in
    auto|required|off)
      ;;
    *)
      record "e4b_qat_resident" "fail" "invalid RUN_E4B_QAT_RESIDENT=$RUN_E4B_QAT_RESIDENT"
      return
      ;;
  esac
  if [[ "$RUN_E4B_QAT_RESIDENT" == "off" ]]; then
    local dependent_failed=0
    if [[ "$RUN_E4B_QAT_RESIDENT_SOAK" != "off" ]]; then
      record "e4b_qat_resident_soak" "fail" "RUN_E4B_QAT_RESIDENT_SOAK=$RUN_E4B_QAT_RESIDENT_SOAK requires RUN_E4B_QAT_RESIDENT"
      dependent_failed=1
    fi
    if [[ "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" != "off" ]]; then
      record "e4b_qat_resident_backpressure" "fail" "RUN_E4B_QAT_RESIDENT_BACKPRESSURE=$RUN_E4B_QAT_RESIDENT_BACKPRESSURE requires RUN_E4B_QAT_RESIDENT"
      dependent_failed=1
    fi
    if [[ "$dependent_failed" -ne 0 ]]; then
      return
    fi
    record "e4b_qat_resident" "skip" "RUN_E4B_QAT_RESIDENT=off"
    return
  fi
  if [[ "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" != "off" && -z "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" ]]; then
    record "e4b_qat_resident_backpressure" "fail" "E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS is required for backpressure gate"
    return
  fi
  if ! exists_path "$E4B_QAT"; then
    if [[ "$RUN_E4B_QAT_RESIDENT" == "required" ]]; then
      record "e4b_qat_resident" "fail" "missing $E4B_QAT"
    else
      record "e4b_qat_resident" "skip" "missing $E4B_QAT"
    fi
    return
  fi
  if ! [[ "$E4B_QAT_RESIDENT_TOKENS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_TOKENS" -lt 1 ]]; then
    record "e4b_qat_resident" "fail" "invalid E4B_QAT_RESIDENT_TOKENS=$E4B_QAT_RESIDENT_TOKENS"
    return
  fi
  if ! [[ "$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS" -lt 1 ]]; then
    record "e4b_qat_resident" "fail" "invalid E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS=$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS"
    return
  fi
  if [[ "$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS" -gt "$E4B_QAT_RESIDENT_TOKENS" ]]; then
    record "e4b_qat_resident" "fail" "E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS=$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS exceeds E4B_QAT_RESIDENT_TOKENS=$E4B_QAT_RESIDENT_TOKENS"
    return
  fi
  if ! [[ "$E4B_QAT_RESIDENT_WARM_REPEATS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_WARM_REPEATS" -lt 1 ]]; then
    record "e4b_qat_resident" "fail" "invalid E4B_QAT_RESIDENT_WARM_REPEATS=$E4B_QAT_RESIDENT_WARM_REPEATS"
    return
  fi
  if [[ "$E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS" != "auto" && ! "$E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS" =~ ^[0-9]+$ ]]; then
    record "e4b_qat_resident" "fail" "invalid E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS=$E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS"
    return
  fi
  local -a server_capacity_args=()
  if [[ -n "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" ]]; then
    if ! [[ "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" =~ ^[0-9]+$ ]] || [[ "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" -lt 1 ]]; then
      record "e4b_qat_resident" "fail" "invalid E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS"
      return
    fi
    server_capacity_args=(--max-concurrent-requests "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS")
  fi

  local port="$E4B_QAT_RESIDENT_PORT"
  if [[ -z "$port" || "$port" == "auto" ]]; then
    port="$(choose_resident_port)"
  fi
  local host="127.0.0.1"
  local server_log="$OUT_DIR/e4b_qat_resident_server.log"
  local resident_tsv="$OUT_DIR/e4b_qat_resident_cuda_server.tsv"
  local summary="$OUT_DIR/e4b_qat_resident_summary.txt"
  local err_path="$OUT_DIR/e4b_qat_resident.err"
  local endpoint="http://$host:$port/ai/v1/generate"
  local ready_url="http://$host:$port/healthz"
  local resident_graph_probe_trace="${ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE:-0}"
  if [[ "$E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY" == "required" ]]; then
    resident_graph_probe_trace=1
  fi

  echo "gate: e4b_qat_resident"
  env \
    ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE="$resident_graph_probe_trace" \
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$E4B_QAT_TEMP_SLOT_PERIOD" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$E4B_QAT_CAPTURE_ALLOW_UNPINNED" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$E4B_QAT_CAPTURE_MIN_ALLOC_SEQ" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$E4B_QAT_FORCE_KV_CAPACITY" \
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1 \
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$E4B_QAT_Q4_0_GATED_DOWN_TILE8" \
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$E4B_QAT_Q4_0_PLE_GATE_FUSION" \
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$E4B_QAT_PLE_RMS_EMBED_FUSION" \
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$E4B_QAT_PENDING_TOKEN_READBACK" \
    "$BIN" run \
      --host "$host" \
      --port "$port" \
      "${server_capacity_args[@]}" \
      --models-dir "$ROOT_DIR/.models" \
      --preload-model "generator:cuda:$E4B_QAT" >"$server_log" 2>&1 &
  E4B_QAT_RESIDENT_SERVER_PID=$!

  if ! wait_for_e4b_qat_resident_server "$ready_url" "$server_log"; then
    cleanup_e4b_qat_resident_server
    record "e4b_qat_resident" "fail" "$server_log server_not_ready"
    return
  fi

  printf 'case\te2e_ms\tcompletion_tokens\te2e_tok_s\n' >"$resident_tsv"
  local repeat
  for ((repeat = 1; repeat <= E4B_QAT_RESIDENT_WARM_REPEATS; repeat++)); do
    if ! e4b_qat_resident_generate_request \
      "e4b_qat_resident_warm${repeat}" \
      "$endpoint" \
      "$OUT_DIR/e4b_qat_resident_warm${repeat}.json" >>"$resident_tsv" 2>"$err_path"; then
      local detail
      detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
      cleanup_e4b_qat_resident_server
      record "e4b_qat_resident" "fail" "${detail:-request_failed}"
      return
    fi
  done

  if python3 - "$resident_tsv" "$server_log" "$MIN_E4B_QAT_RESIDENT_WARM_TOK_S" "$E4B_QAT_RESIDENT_WARM_REPEATS" "$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS" "$E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY" "$E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS" "$E4B_QAT_RESIDENT_TOKENS" >"$summary" 2>"$err_path" <<'PY'
import csv
import sys

tsv_path, log_path, min_tok_s, expected_repeats, min_completion_tokens, replay_mode, graph_floor_arg, tokens_arg = sys.argv[1:9]
min_tok_s = float(min_tok_s)
expected_repeats = int(expected_repeats)
min_completion_tokens = int(min_completion_tokens)
tokens = int(tokens_arg)

with open(tsv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

errors = []
rates = []
if len(rows) < expected_repeats:
    errors.append(f"warm_rows={len(rows)} < {expected_repeats}")
for row in rows:
    label = row.get("case", "unknown")
    try:
        completion_tokens = int(row["completion_tokens"])
        rate = float(row["e2e_tok_s"])
    except Exception as exc:
        errors.append(f"{label}: invalid row: {exc}")
        continue
    if completion_tokens < min_completion_tokens:
        errors.append(f"{label}: completion_tokens={completion_tokens} < {min_completion_tokens}")
    if rate < min_tok_s:
        errors.append(f"{label}: e2e_tok_s={rate:.3f} < {min_tok_s:.3f}")
    rates.append(rate)

replays = 0
if replay_mode == "required":
    if graph_floor_arg == "auto":
        graph_floor = max(1, tokens // 3)
    else:
        graph_floor = int(graph_floor_arg)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    replays = log_text.count("persistent_replayed") + log_text.count("instantiated_cached_replayed")
    if replays < graph_floor:
        errors.append(f"graph replays={replays} < {graph_floor}")
    for marker in (
        "unsafe_d2h_copy",
        "unsafe_h2d_copy",
        "unsafe_temp_alloc",
        "CudaGraphCaptureUnsafe",
        "persistent_replay_kv_capacity_exceeded",
        "cuda_graph_capture_probe: discarded",
        "CUDA_ERROR_ILLEGAL_ADDRESS",
    ):
        if marker in log_text:
            errors.append(f"server log contains {marker}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

min_rate = min(rates)
avg_rate = sum(rates) / len(rates)
print(f"warm_repeats={len(rates)} min_e2e_tok_s={min_rate:.3f} avg_e2e_tok_s={avg_rate:.3f} graph_replays={replays}")
PY
  then
    local summary_line
    summary_line="$(sed -n '1p' "$summary" 2>/dev/null || true)"
    record "e4b_qat_resident" "ok" "${summary_line:-$summary}"
    local resident_subgate_failed=0
    if ! run_e4b_qat_resident_soak "$endpoint" "$server_log"; then
      resident_subgate_failed=1
    fi
    local metrics_url="http://$host:$port/ml/v1/metrics"
    if ! run_e4b_qat_resident_backpressure "$endpoint" "$server_log" "$metrics_url"; then
      resident_subgate_failed=1
    fi
    cleanup_e4b_qat_resident_server
    run_e4b_q4k_resident_baseline "$resident_tsv"
    if [[ "$resident_subgate_failed" -ne 0 ]]; then
      return
    fi
  else
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    cleanup_e4b_qat_resident_server
    record "e4b_qat_resident" "fail" "${detail:-$summary}"
  fi
}

run_e4b_q4k_resident_baseline() {
  local qat_tsv="$1"
  case "$RUN_E4B_Q4K_RESIDENT_BASELINE" in
    auto|required|off)
      ;;
    *)
      record "e4b_q4k_resident_baseline" "fail" "invalid RUN_E4B_Q4K_RESIDENT_BASELINE=$RUN_E4B_Q4K_RESIDENT_BASELINE"
      return
      ;;
  esac
  if [[ "$RUN_E4B_Q4K_RESIDENT_BASELINE" == "off" ]]; then
    record "e4b_q4k_resident_baseline" "skip" "RUN_E4B_Q4K_RESIDENT_BASELINE=off"
    return
  fi
  if ! exists_path "$E4B_Q4K_BASELINE"; then
    if [[ "$RUN_E4B_Q4K_RESIDENT_BASELINE" == "required" ]]; then
      record "e4b_q4k_resident_baseline" "fail" "missing $E4B_Q4K_BASELINE"
    else
      record "e4b_q4k_resident_baseline" "skip" "missing $E4B_Q4K_BASELINE"
    fi
    return
  fi

  local port
  port="$(choose_resident_port)"
  local host="127.0.0.1"
  local server_log="$OUT_DIR/e4b_q4k_resident_server.log"
  local resident_tsv="$OUT_DIR/e4b_q4k_resident_cuda_server.tsv"
  local summary="$OUT_DIR/e4b_q4k_resident_summary.txt"
  local ratio_summary="$OUT_DIR/e4b_qat_resident_over_q4k_summary.txt"
  local err_path="$OUT_DIR/e4b_q4k_resident.err"
  local endpoint="http://$host:$port/ai/v1/generate"
  local ready_url="http://$host:$port/healthz"
  local graph_probe_trace="${ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE:-0}"
  if [[ "$E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY" == "required" ]]; then
    graph_probe_trace=1
  fi

  echo "gate: e4b_q4k_resident_baseline"
  env \
    ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE="$graph_probe_trace" \
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$E4B_QAT_TEMP_SLOT_PERIOD" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$E4B_QAT_CAPTURE_ALLOW_UNPINNED" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$E4B_QAT_CAPTURE_MIN_ALLOC_SEQ" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$E4B_QAT_FORCE_KV_CAPACITY" \
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1 \
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$E4B_QAT_PENDING_TOKEN_READBACK" \
    "$BIN" run \
      --host "$host" \
      --port "$port" \
      --models-dir "$ROOT_DIR/.models" \
      --preload-model "generator:cuda:$E4B_Q4K_BASELINE" >"$server_log" 2>&1 &
  E4B_QAT_RESIDENT_SERVER_PID=$!

  if ! wait_for_e4b_qat_resident_server "$ready_url" "$server_log"; then
    cleanup_e4b_qat_resident_server
    record "e4b_q4k_resident_baseline" "fail" "$server_log server_not_ready"
    return
  fi

  printf 'case\te2e_ms\tcompletion_tokens\te2e_tok_s\n' >"$resident_tsv"
  local repeat
  for ((repeat = 1; repeat <= E4B_QAT_RESIDENT_WARM_REPEATS; repeat++)); do
    if ! e4b_qat_resident_generate_request \
      "e4b_q4k_resident_warm${repeat}" \
      "$endpoint" \
      "$OUT_DIR/e4b_q4k_resident_warm${repeat}.json" \
      "$E4B_Q4K_BASELINE" >>"$resident_tsv" 2>"$err_path"; then
      local detail
      detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
      cleanup_e4b_qat_resident_server
      record "e4b_q4k_resident_baseline" "fail" "${detail:-request_failed}"
      return
    fi
  done

  if python3 - "$resident_tsv" "$server_log" "$E4B_QAT_RESIDENT_WARM_REPEATS" "$E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS" "$E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY" "$E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS" "$E4B_QAT_RESIDENT_TOKENS" >"$summary" 2>"$err_path" <<'PY'
import csv
import sys

tsv_path, log_path, expected_repeats, min_completion_tokens, replay_mode, graph_floor_arg, tokens_arg = sys.argv[1:8]
expected_repeats = int(expected_repeats)
min_completion_tokens = int(min_completion_tokens)
tokens = int(tokens_arg)

with open(tsv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

errors = []
rates = []
if len(rows) < expected_repeats:
    errors.append(f"warm_rows={len(rows)} < {expected_repeats}")
for row in rows:
    label = row.get("case", "unknown")
    try:
        completion_tokens = int(row["completion_tokens"])
        rate = float(row["e2e_tok_s"])
    except Exception as exc:
        errors.append(f"{label}: invalid row: {exc}")
        continue
    if completion_tokens < min_completion_tokens:
        errors.append(f"{label}: completion_tokens={completion_tokens} < {min_completion_tokens}")
    if rate <= 0:
        errors.append(f"{label}: e2e_tok_s={rate:.3f} <= 0")
    rates.append(rate)

replays = 0
if replay_mode == "required":
    graph_floor = max(1, tokens // 3) if graph_floor_arg == "auto" else int(graph_floor_arg)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    replays = log_text.count("persistent_replayed") + log_text.count("instantiated_cached_replayed")
    if replays < graph_floor:
        errors.append(f"graph replays={replays} < {graph_floor}")
    for marker in (
        "unsafe_d2h_copy",
        "unsafe_h2d_copy",
        "unsafe_temp_alloc",
        "CudaGraphCaptureUnsafe",
        "persistent_replay_kv_capacity_exceeded",
        "cuda_graph_capture_probe: discarded",
        "CUDA_ERROR_ILLEGAL_ADDRESS",
    ):
        if marker in log_text:
            errors.append(f"server log contains {marker}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)

min_rate = min(rates)
avg_rate = sum(rates) / len(rates)
print(f"warm_repeats={len(rates)} min_e2e_tok_s={min_rate:.3f} avg_e2e_tok_s={avg_rate:.3f} graph_replays={replays}")
PY
  then
    local summary_line
    summary_line="$(sed -n '1p' "$summary" 2>/dev/null || true)"
    cleanup_e4b_qat_resident_server
    record "e4b_q4k_resident_baseline" "ok" "${summary_line:-$summary}"
  else
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    cleanup_e4b_qat_resident_server
    record "e4b_q4k_resident_baseline" "fail" "${detail:-$summary}"
    return
  fi

  if python3 - "$qat_tsv" "$resident_tsv" "$MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO" >"$ratio_summary" 2>"$err_path" <<'PY'
import csv
import sys

qat_path, q4k_path, floor_arg = sys.argv[1:4]
floor = float(floor_arg)

def avg_rate(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rates = [float(row["e2e_tok_s"]) for row in rows if row.get("e2e_tok_s")]
    if not rates:
        raise SystemExit(f"{path}: no rate rows")
    return sum(rates) / len(rates)

qat = avg_rate(qat_path)
q4k = avg_rate(q4k_path)
ratio = qat / q4k if q4k > 0 else 0.0
if ratio < floor:
    print(f"qat_avg={qat:.3f} q4k_avg={q4k:.3f} ratio={ratio:.3f} floor={floor}", file=sys.stderr)
    raise SystemExit(1)
print(f"qat_avg={qat:.3f} q4k_avg={q4k:.3f} ratio={ratio:.3f} floor={floor}")
PY
  then
    local summary_line
    summary_line="$(sed -n '1p' "$ratio_summary" 2>/dev/null || true)"
    record "e4b_qat_resident_over_q4k" "ok" "${summary_line:-$ratio_summary}"
  else
    local detail
    detail="$(sed -n '1p' "$err_path" 2>/dev/null || true)"
    record "e4b_qat_resident_over_q4k" "fail" "${detail:-$ratio_summary}"
  fi
}

run_e4b_q4k_baseline() {
  case "$RUN_E4B_Q4K_BASELINE" in
    auto|required|off)
      ;;
    *)
      record "target_e4b_q4k" "fail" "invalid RUN_E4B_Q4K_BASELINE=$RUN_E4B_Q4K_BASELINE"
      return
      ;;
  esac
  if [[ "$RUN_E4B_Q4K_BASELINE" == "off" ]]; then
    record "target_e4b_q4k" "skip" "RUN_E4B_Q4K_BASELINE=off"
    return
  fi
  if ! exists_path "$E4B_Q4K_BASELINE"; then
    if [[ "$RUN_E4B_Q4K_BASELINE" == "required" ]]; then
      record "target_e4b_q4k" "fail" "missing $E4B_Q4K_BASELINE"
    else
      record "target_e4b_q4k" "skip" "missing $E4B_Q4K_BASELINE"
    fi
    return
  fi
  run_target_generate \
    "target_e4b_q4k" \
    "$E4B_Q4K_BASELINE" \
    "$E4B_Q4K_COMBINED_BUDGET_MB" \
    "$E4B_Q4K_BACKEND_BUDGET_MB" \
    "$E4B_Q4K_KV_BUDGET_MB" \
    "$E4B_Q4K_SCRATCH_BUDGET_MB" \
    "$E4B_Q4K_HOST_BUDGET_MB" \
    "$E4B_QAT_TOKENS"
}

run_default_policy_check() {
  local target="$GEMMA12_Q8"
  local assistant="$GEMMA12_ASSISTANT_Q8"
  local json_path="$OUT_DIR/default_auto_uncalibrated.json"
  local log_path="$OUT_DIR/default_auto_uncalibrated.log"
  rm -f "$json_path"
  if ! exists_path "$target" || ! exists_path "$assistant"; then
    record "default_auto_uncalibrated" "skip" "missing 12B target or assistant"
    return
  fi
  run_logged "default_auto_uncalibrated" "$log_path" \
    run_with_optional_timeout "$BIN" generate "$target" "Ants" \
      --backend cuda \
      --draft-model "$assistant" \
      --speculative-k 1 \
      --speculation-policy auto \
      --max-tokens 1 \
      --temperature 0 \
      --combined-budget-mb "$COMBINED_BUDGET_MB" \
      --backend-budget-mb "$BACKEND_BUDGET_MB" \
      --kv-budget-mb "$KV_BUDGET_MB" \
      --scratch-budget-mb "$SCRATCH_BUDGET_MB" \
      --print-timing \
      --print-token-count \
      --json-timing "$json_path" \
      --raw-prompt \
      --no-chat-template
}

run_mtp_matrix() {
  local label="$1"
  local target="$2"
  local assistant_q8="$3"
  local assistant_q4="$4"
  local combined="$5"
  local backend="$6"
  local kv="$7"
  local scratch="$8"
  local mode="${9:-auto}"
  local matrix_max_tokens="${10:-$MAX_TOKENS}"
  local matrix_spec_ks="${11:-$SPEC_KS}"
  local matrix_prompt_filter="${12:-$PROMPT_FILTER}"
  local matrix_mode="${13:-production}"
  local matrix_cache_dtype="${14:-}"
  local matrix_speculation_policy="${15:-auto}"
  local matrix_speculation_calibration="${16:-probe}"
  local matrix_target_replay="${17:-}"
  local matrix_replay_context_key="${18:-}"
  local matrix_unsafe_target_replay="${19:-auto}"
  local matrix_assistant_replay="${20:-auto}"
  local matrix_materialize_replay="${21:-auto}"
  local matrix_capture_persistent_replay="${22:-auto}"
  local matrix_temp_slot_period="${23:-}"
  local matrix_temp_slot_skip="${24:-}"
  local matrix_position_mode="${25:-}"
  local matrix_hidden_source="${26:-}"
  local matrix_concat_order="${27:-}"
  local matrix_kv_donor_mode="${28:-}"
  local matrix_json_token_ids="${29:-0}"
  local matrix_hidden_select_fusion="${30:-auto}"
  local matrix_dir="$OUT_DIR/$label"
  local log_path="$OUT_DIR/${label}.log"
  case "$mode" in
    auto|required|off)
      ;;
    *)
      record "$label" "fail" "invalid mode=$mode; expected auto|required|off"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "$label" "skip" "mode=off"
    return
  fi
  if ! exists_path "$target" || ! exists_path "$assistant_q8"; then
    if [[ "$mode" == "required" ]]; then
      record "$label" "fail" "missing target or q8 assistant target=$target assistant=$assistant_q8"
    else
      record "$label" "skip" "missing target or q8 assistant"
    fi
    return
  fi
  mkdir -p "$matrix_dir"
  echo "gate: $label"
  local -a args=("$target" "$assistant_q8")
  if [[ -n "$assistant_q4" && -e "$assistant_q4" ]]; then
    args+=("$assistant_q4")
  fi
  if OUT_DIR="$matrix_dir" \
    BIN="$BIN" \
    MODE="$matrix_mode" \
    SPECULATION_POLICY="$matrix_speculation_policy" \
    SPECULATION_CALIBRATION="$matrix_speculation_calibration" \
    SPEC_KS="$matrix_spec_ks" \
    PROMPT_FILTER="$matrix_prompt_filter" \
    MAX_TOKENS="$matrix_max_tokens" \
    RUN_TIMEOUT="$RUN_TIMEOUT" \
    MTP_VERIFY_DEVICE_RESULT="$MTP_VERIFY_DEVICE_RESULT" \
    MTP_MASKED_SELECT_HIDDEN_FUSION="$matrix_hidden_select_fusion" \
    MTP_TARGET_REPLAY="$matrix_target_replay" \
    MTP_REPLAY_CONTEXT_KEY="$matrix_replay_context_key" \
    MTP_UNSAFE_TARGET_REPLAY="$matrix_unsafe_target_replay" \
    MTP_ASSISTANT_REPLAY="$matrix_assistant_replay" \
    MTP_MATERIALIZE_REPLAY="$matrix_materialize_replay" \
    CUDA_CAPTURE_PERSISTENT_REPLAY="$matrix_capture_persistent_replay" \
    CUDA_TEMP_SLOT_PERIOD="$matrix_temp_slot_period" \
    CUDA_TEMP_SLOT_SKIP="$matrix_temp_slot_skip" \
    MTP_POSITION_MODE="$matrix_position_mode" \
    MTP_TARGET_HIDDEN_SOURCE="$matrix_hidden_source" \
    MTP_CONCAT_ORDER="$matrix_concat_order" \
    MTP_KV_DONOR_MODE="$matrix_kv_donor_mode" \
    JSON_TOKEN_IDS="$matrix_json_token_ids" \
    RUN_TARGET_ONLY=1 \
    COMBINED_BUDGET_MB="$combined" \
    BACKEND_BUDGET_MB="$backend" \
    KV_BUDGET_MB="$kv" \
    SCRATCH_BUDGET_MB="$scratch" \
    CACHE_DTYPE="$matrix_cache_dtype" \
    TURBOQUANT_MIN_TOKENS="$E4B_QAT_MTP_TURBOQUANT_MIN_TOKENS" \
    "$ROOT_DIR/scripts/bench_gemma4_mtp.sh" "${args[@]}" >"$log_path" 2>&1; then
    record "$label" "ok" "$matrix_dir/summary.tsv"
  else
    local status=$?
    record "$label" "fail" "$log_path exit_$status"
  fi
}

compare_mtp_replay_stability() {
  local baseline_summary="$1"
  local replay_summary="$2"
  python3 - "$baseline_summary" "$replay_summary" <<'PY'
import csv
import json
import pathlib
import sys

baseline_path = pathlib.Path(sys.argv[1])
replay_path = pathlib.Path(sys.argv[2])
fields = ["status", "tokens", "finish_reason", "drafted", "matched", "accepted", "accept_permille"]

def load(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for row in rows:
        if row.get("assistant") == "target":
            continue
        key = (row.get("assistant"), row.get("spec_k"), row.get("case"))
        out[key] = row
    return out

base = load(baseline_path)
replay = load(replay_path)
errors = []
for key, base_row in sorted(base.items()):
    replay_row = replay.get(key)
    if replay_row is None:
        errors.append(f"missing replay row key={key}")
        continue
    for field in fields:
        if str(base_row.get(field, "")) != str(replay_row.get(field, "")):
            errors.append(
                f"field drift key={key} field={field} baseline={base_row.get(field)} replay={replay_row.get(field)}"
            )
    spec_k = key[1] or ""
    if spec_k == "2":
        try:
            draft_replays = int(replay_row.get("draft_graph_persistent_replays") or "0")
        except ValueError:
            draft_replays = 0
        if draft_replays <= 0:
            errors.append(f"missing strict assistant replay hits key={key} draft_graph_persistent_replays={draft_replays}")

extra = sorted(set(replay) - set(base))
for key in extra:
    errors.append(f"unexpected replay row key={key}")

result = {
    "baseline": str(baseline_path),
    "replay": str(replay_path),
    "row_count": len(base),
    "ok": not errors,
    "errors": errors,
}
print(json.dumps(result, sort_keys=True))
if errors:
    raise SystemExit(1)
PY
}

compare_mtp_target_equivalence() {
  local matrix_dir="$1"
  python3 - "$matrix_dir" <<'PY'
import json
import pathlib
import re
import sys

matrix_dir = pathlib.Path(sys.argv[1])
stem_re = re.compile(r"^(.+)_k([0-9]+)_(.+)$")

targets = {}
candidate_count = 0
errors = []

for json_path in sorted(matrix_dir.glob("target_k0_*.json")):
    stem = json_path.stem
    match = stem_re.match(stem)
    if not match:
        continue
    _, _, case = match.groups()
    status_path = json_path.with_suffix(".status")
    status = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "unknown"
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{stem}: json load failed: {exc}")
        continue
    token_ids = data.get("token_ids")
    if status != "ok":
        errors.append(f"{stem}: status={status}")
    if not isinstance(token_ids, list):
        errors.append(f"{stem}: token_ids missing; set ANTFLY_INFERENCE_JSON_TOKEN_IDS=1")
        continue
    targets[case] = {
        "stem": stem,
        "tokens": data.get("tokens"),
        "finish_reason": data.get("finish_reason"),
        "token_ids": token_ids,
    }

for json_path in sorted(matrix_dir.glob("*_k*_*.json")):
    stem = json_path.stem
    match = stem_re.match(stem)
    if not match:
        continue
    label, spec_k, case = match.groups()
    if label == "target":
        continue
    candidate_count += 1
    target = targets.get(case)
    if target is None:
        errors.append(f"{stem}: missing target row for case={case}")
        continue
    status_path = json_path.with_suffix(".status")
    status = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else "unknown"
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{stem}: json load failed: {exc}")
        continue
    token_ids = data.get("token_ids")
    if status != "ok":
        errors.append(f"{stem}: status={status}")
    if not isinstance(token_ids, list):
        errors.append(f"{stem}: token_ids missing")
        continue
    if data.get("tokens") != target["tokens"]:
        errors.append(f"{stem}: tokens={data.get('tokens')} target={target['tokens']}")
    if data.get("finish_reason") != target["finish_reason"]:
        errors.append(f"{stem}: finish_reason={data.get('finish_reason')} target={target['finish_reason']}")
    if token_ids != target["token_ids"]:
        first_diff = None
        for idx, (actual, expected) in enumerate(zip(token_ids, target["token_ids"])):
            if actual != expected:
                first_diff = (idx, actual, expected)
                break
        if first_diff is None and len(token_ids) != len(target["token_ids"]):
            first_diff = (min(len(token_ids), len(target["token_ids"])), "eof", "eof")
        errors.append(
            f"{stem}: token_ids drift first_diff={first_diff} "
            f"len={len(token_ids)} target_len={len(target['token_ids'])}"
        )

result = {
    "matrix_dir": str(matrix_dir),
    "target_case_count": len(targets),
    "candidate_count": candidate_count,
    "ok": not errors and bool(targets) and candidate_count > 0,
    "errors": errors,
}
print(json.dumps(result, sort_keys=True))
if not result["ok"]:
    raise SystemExit(1)
PY
}

run_e4b_qat_mtp_target_equivalence() {
  local mode="$RUN_E4B_QAT_MTP_TARGET_EQUIV"
  case "$mode" in
    auto|required|off) ;;
    *)
      record "mtp_target_equivalence" "fail" "invalid RUN_E4B_QAT_MTP_TARGET_EQUIV=$mode"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "mtp_target_equivalence" "skip" "RUN_E4B_QAT_MTP_TARGET_EQUIV=off"
    return
  fi
  if ! exists_path "$E4B_QAT" || ! exists_path "$E4B_QAT_ASSISTANT_Q8"; then
    if [[ "$mode" == "required" ]]; then
      record "mtp_target_equivalence" "fail" "missing E4B QAT target or assistant"
    else
      record "mtp_target_equivalence" "skip" "missing E4B QAT target or assistant"
    fi
    return
  fi

  run_mtp_matrix \
    "mtp_target_equivalence" \
    "$E4B_QAT" \
    "$E4B_QAT_ASSISTANT_Q8" \
    "$E4B_QAT_ASSISTANT_Q4" \
    "$E4B_QAT_COMBINED_BUDGET_MB" \
    "$E4B_QAT_BACKEND_BUDGET_MB" \
    "$E4B_QAT_KV_BUDGET_MB" \
    "$E4B_QAT_SCRATCH_BUDGET_MB" \
    required \
    "$E4B_QAT_MTP_TARGET_EQUIV_TOKENS" \
    "$E4B_QAT_MTP_TARGET_EQUIV_SPEC_KS" \
    "$E4B_QAT_MTP_TARGET_EQUIV_PROMPT_FILTER" \
    production \
    f32 \
    force \
    positive \
    off \
    1 \
    0 \
    1 \
    auto \
    1 \
    1 \
    0 \
    "" \
    "" \
    "" \
    "" \
    1

  local compare_json="$OUT_DIR/mtp_target_equivalence.json"
  local compare_log="$OUT_DIR/mtp_target_equivalence_compare.log"
  if compare_mtp_target_equivalence "$OUT_DIR/mtp_target_equivalence" >"$compare_json" 2>"$compare_log"; then
    record "mtp_target_equivalence" "ok" "$compare_json"
  else
    local detail
    detail="$(sed -n '1p' "$compare_json" 2>/dev/null || sed -n '1p' "$compare_log" 2>/dev/null || true)"
    record "mtp_target_equivalence" "fail" "${detail:-$compare_log}"
  fi
}

run_e4b_qat_mtp_replay_stability() {
  local mode="$RUN_E4B_QAT_MTP_REPLAY_STABILITY"
  case "$mode" in
    auto|required|off) ;;
    *)
      record "mtp_replay_stability" "fail" "invalid RUN_E4B_QAT_MTP_REPLAY_STABILITY=$mode"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "mtp_replay_stability" "skip" "RUN_E4B_QAT_MTP_REPLAY_STABILITY=off"
    return
  fi
  if ! exists_path "$E4B_QAT" || ! exists_path "$E4B_QAT_ASSISTANT_Q8"; then
    if [[ "$mode" == "required" ]]; then
      record "mtp_replay_stability" "fail" "missing E4B QAT target or assistant"
    else
      record "mtp_replay_stability" "skip" "missing E4B QAT target or assistant"
    fi
    return
  fi

  run_mtp_matrix \
    "mtp_replay_stability_baseline" \
    "$E4B_QAT" \
    "$E4B_QAT_ASSISTANT_Q8" \
    "$E4B_QAT_ASSISTANT_Q4" \
    "$E4B_QAT_COMBINED_BUDGET_MB" \
    "$E4B_QAT_BACKEND_BUDGET_MB" \
    "$E4B_QAT_KV_BUDGET_MB" \
    "$E4B_QAT_SCRATCH_BUDGET_MB" \
    required \
    "$E4B_QAT_MTP_REPLAY_STABILITY_TOKENS" \
    "$E4B_QAT_MTP_REPLAY_STABILITY_SPEC_KS" \
    "$E4B_QAT_MTP_REPLAY_STABILITY_PROMPT_FILTER" \
    profile \
    f32 \
    auto \
    probe \
    off \
    1 \
    0 \
    0

  run_mtp_matrix \
    "mtp_replay_stability_assistant" \
    "$E4B_QAT" \
    "$E4B_QAT_ASSISTANT_Q8" \
    "$E4B_QAT_ASSISTANT_Q4" \
    "$E4B_QAT_COMBINED_BUDGET_MB" \
    "$E4B_QAT_BACKEND_BUDGET_MB" \
    "$E4B_QAT_KV_BUDGET_MB" \
    "$E4B_QAT_SCRATCH_BUDGET_MB" \
    required \
    "$E4B_QAT_MTP_REPLAY_STABILITY_TOKENS" \
    "$E4B_QAT_MTP_REPLAY_STABILITY_SPEC_KS" \
    "$E4B_QAT_MTP_REPLAY_STABILITY_PROMPT_FILTER" \
    profile \
    f32 \
    auto \
    probe \
    off \
    1 \
    0 \
    1 \
    auto \
    1 \
    1 \
    0

  local compare_json="$OUT_DIR/mtp_replay_stability.json"
  local compare_log="$OUT_DIR/mtp_replay_stability_compare.log"
  if compare_mtp_replay_stability \
    "$OUT_DIR/mtp_replay_stability_baseline/summary.tsv" \
    "$OUT_DIR/mtp_replay_stability_assistant/summary.tsv" >"$compare_json" 2>"$compare_log"; then
    record "mtp_replay_stability" "ok" "$compare_json"
  else
    local detail
    detail="$(sed -n '1p' "$compare_json" 2>/dev/null || sed -n '1p' "$compare_log" 2>/dev/null || true)"
    record "mtp_replay_stability" "fail" "${detail:-$compare_log}"
  fi
}

run_e4b_qat_mtp_replay_512() {
  local mode="$RUN_E4B_QAT_MTP_REPLAY_512"
  case "$mode" in
    auto|required|off) ;;
    *)
      record "mtp_replay_512" "fail" "invalid RUN_E4B_QAT_MTP_REPLAY_512=$mode"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "mtp_replay_512" "skip" "RUN_E4B_QAT_MTP_REPLAY_512=off"
    return
  fi
  if ! exists_path "$E4B_QAT" || ! exists_path "$E4B_QAT_ASSISTANT_Q8"; then
    if [[ "$mode" == "required" ]]; then
      record "mtp_replay_512" "fail" "missing E4B QAT target or assistant"
    else
      record "mtp_replay_512" "skip" "missing E4B QAT target or assistant"
    fi
    return
  fi

  run_mtp_matrix \
    "mtp_replay_512" \
    "$E4B_QAT" \
    "$E4B_QAT_ASSISTANT_Q8" \
    "$E4B_QAT_ASSISTANT_Q4" \
    "$E4B_QAT_COMBINED_BUDGET_MB" \
    "$E4B_QAT_BACKEND_BUDGET_MB" \
    "$E4B_QAT_KV_BUDGET_MB" \
    "$E4B_QAT_SCRATCH_BUDGET_MB" \
    "$mode" \
    "$E4B_QAT_MTP_REPLAY_TOKENS" \
    "$E4B_QAT_MTP_REPLAY_SPEC_KS" \
    "$E4B_QAT_MTP_REPLAY_PROMPT_FILTER" \
    production \
    f32 \
    auto \
    probe \
    off \
    1 \
    0 \
    1 \
    auto \
    1 \
    1 \
    0
}

run_e4b_qat_mtp_hidden_ab() {
  local mode="$RUN_E4B_QAT_MTP_HIDDEN_AB"
  case "$mode" in
    auto|required|off) ;;
    *)
      record "mtp_hidden_ab" "fail" "invalid RUN_E4B_QAT_MTP_HIDDEN_AB=$mode"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "mtp_hidden_ab" "skip" "RUN_E4B_QAT_MTP_HIDDEN_AB=off"
    return
  fi
  if ! [[ "$E4B_QAT_MTP_HIDDEN_AB_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    record "mtp_hidden_ab" "fail" "invalid E4B_QAT_MTP_HIDDEN_AB_REPEATS=$E4B_QAT_MTP_HIDDEN_AB_REPEATS"
    return
  fi
  if ! exists_path "$E4B_QAT" || ! exists_path "$E4B_QAT_ASSISTANT_Q8"; then
    if [[ "$mode" == "required" ]]; then
      record "mtp_hidden_ab" "fail" "missing E4B QAT target or assistant"
    else
      record "mtp_hidden_ab" "skip" "missing E4B QAT target or assistant"
    fi
    return
  fi

  local failures=0
  local repeat label
  for ((repeat = 1; repeat <= E4B_QAT_MTP_HIDDEN_AB_REPEATS; repeat++)); do
    for label in default hidden; do
      local fusion=0
      if [[ "$label" == "hidden" ]]; then
        fusion=1
      fi
      run_mtp_matrix \
        "mtp_hidden_ab_${label}_r${repeat}" \
        "$E4B_QAT" \
        "$E4B_QAT_ASSISTANT_Q8" \
        "$E4B_QAT_ASSISTANT_Q4" \
        "$E4B_QAT_COMBINED_BUDGET_MB" \
        "$E4B_QAT_BACKEND_BUDGET_MB" \
        "$E4B_QAT_KV_BUDGET_MB" \
        "$E4B_QAT_SCRATCH_BUDGET_MB" \
        required \
        "$E4B_QAT_MTP_REPLAY_TOKENS" \
        "$E4B_QAT_MTP_REPLAY_SPEC_KS" \
        "$E4B_QAT_MTP_REPLAY_PROMPT_FILTER" \
        production \
        f32 \
        auto \
        probe \
        off \
        1 \
        0 \
        1 \
        auto \
        1 \
        1 \
        0 \
        "" \
        "" \
        "" \
        "" \
        0 \
        "$fusion"
      if [[ ! -f "$OUT_DIR/mtp_hidden_ab_${label}_r${repeat}/summary.tsv" ]]; then
        failures=$((failures + 1))
      fi
    done
  done
  if (( failures == 0 )); then
    record "mtp_hidden_ab" "ok" "repeats=$E4B_QAT_MTP_HIDDEN_AB_REPEATS min_ratio=$E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO"
  else
    record "mtp_hidden_ab" "fail" "matrix_failures=$failures"
  fi
}

run_e4b_qat_mtp_acceptance_matrix() {
  local mode="$RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX"
  case "$mode" in
    auto|required|off) ;;
    *)
      record "mtp_acceptance_matrix" "fail" "invalid RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX=$mode"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "mtp_acceptance_matrix" "skip" "RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX=off"
    return
  fi
  if ! exists_path "$E4B_QAT" || ! exists_path "$E4B_QAT_ASSISTANT_Q8"; then
    if [[ "$mode" == "required" ]]; then
      record "mtp_acceptance_matrix" "fail" "missing E4B QAT target or assistant"
    else
      record "mtp_acceptance_matrix" "skip" "missing E4B QAT target or assistant"
    fi
    return
  fi

  local failures=0
  local position_mode hidden_source concat_order combo_label
  for position_mode in target_constant target_absolute; do
    for hidden_source in final pre_norm; do
      for concat_order in embedding_activation activation_embedding; do
        combo_label="mtp_acceptance_matrix_${position_mode}_${hidden_source}_${concat_order}"
        run_mtp_matrix \
          "$combo_label" \
          "$E4B_QAT" \
          "$E4B_QAT_ASSISTANT_Q8" \
          "$E4B_QAT_ASSISTANT_Q4" \
          "$E4B_QAT_COMBINED_BUDGET_MB" \
          "$E4B_QAT_BACKEND_BUDGET_MB" \
          "$E4B_QAT_KV_BUDGET_MB" \
          "$E4B_QAT_SCRATCH_BUDGET_MB" \
          required \
          "$E4B_QAT_MTP_ACCEPTANCE_TOKENS" \
          "$E4B_QAT_MTP_ACCEPTANCE_SPEC_KS" \
          "$E4B_QAT_MTP_ACCEPTANCE_PROMPT_FILTER" \
          profile \
          f32 \
          force \
          positive \
          off \
          1 \
          0 \
          0 \
          auto \
          auto \
          "" \
          "" \
          "$position_mode" \
          "$hidden_source" \
          "$concat_order"
        if [[ ! -f "$OUT_DIR/$combo_label/summary.tsv" ]]; then
          failures=$((failures + 1))
        fi
      done
    done
  done
  if (( failures == 0 )); then
    record "mtp_acceptance_matrix" "ok" "$OUT_DIR/mtp_acceptance_matrix_*"
  else
    record "mtp_acceptance_matrix" "fail" "combo_failures=$failures"
  fi
}

run_e4b_qat_mtp_donor_matrix() {
  local mode="$RUN_E4B_QAT_MTP_DONOR_MATRIX"
  case "$mode" in
    auto|required|off) ;;
    *)
      record "mtp_donor_matrix" "fail" "invalid RUN_E4B_QAT_MTP_DONOR_MATRIX=$mode"
      return
      ;;
  esac
  if [[ "$mode" == "off" ]]; then
    record "mtp_donor_matrix" "skip" "RUN_E4B_QAT_MTP_DONOR_MATRIX=off"
    return
  fi
  if ! exists_path "$E4B_QAT" || ! exists_path "$E4B_QAT_ASSISTANT_Q8"; then
    if [[ "$mode" == "required" ]]; then
      record "mtp_donor_matrix" "fail" "missing E4B QAT target or assistant"
    else
      record "mtp_donor_matrix" "skip" "missing E4B QAT target or assistant"
    fi
    return
  fi

  local failures=0
  local donor_mode combo_label
  for donor_mode in $E4B_QAT_MTP_DONOR_MATRIX_MODES; do
    combo_label="mtp_donor_matrix_${donor_mode}"
    run_mtp_matrix \
      "$combo_label" \
      "$E4B_QAT" \
      "$E4B_QAT_ASSISTANT_Q8" \
      "$E4B_QAT_ASSISTANT_Q4" \
      "$E4B_QAT_COMBINED_BUDGET_MB" \
      "$E4B_QAT_BACKEND_BUDGET_MB" \
      "$E4B_QAT_KV_BUDGET_MB" \
      "$E4B_QAT_SCRATCH_BUDGET_MB" \
      required \
      "$E4B_QAT_MTP_DONOR_MATRIX_TOKENS" \
      "$E4B_QAT_MTP_DONOR_MATRIX_SPEC_KS" \
      "$E4B_QAT_MTP_DONOR_MATRIX_PROMPT_FILTER" \
      profile \
      f32 \
      force \
      positive \
      off \
      1 \
      0 \
      0 \
      auto \
      auto \
      "" \
      "" \
      target_constant \
      final \
      embedding_activation \
      "$donor_mode"
    if [[ ! -f "$OUT_DIR/$combo_label/summary.tsv" ]]; then
      failures=$((failures + 1))
    fi
  done
  if (( failures == 0 )); then
    record "mtp_donor_matrix" "ok" "$OUT_DIR/mtp_donor_matrix_*"
  else
    record "mtp_donor_matrix" "fail" "combo_failures=$failures"
  fi
}

if [[ "$RUN_BUILD" == "1" ]]; then
  run_logged_in_dir "build_cuda_sm89" "$ROOT_DIR/zig/pkg/inference" "$OUT_DIR/build_cuda_sm89.log" "$ZIG" build -Dcuda=true -Dcuda-artifacts=sm89
else
  record "build_cuda_sm89" "skip" "RUN_BUILD=0"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  run_logged "cuda_smoke" "$OUT_DIR/cuda_smoke.log" "$BIN" cuda-info --smoke
else
  record "cuda_smoke" "skip" "RUN_SMOKE=0"
fi
write_cuda_environment

if [[ "$RUN_MICROBENCH" == "1" ]]; then
  run_logged "bench_cuda_gemma4_shapes" "$OUT_DIR/bench_cuda_gemma4_shapes.log" \
    "$BIN" bench-cuda \
    --warmup-iters "$MICROBENCH_WARMUP_ITERS" \
    --measure-iters "$MICROBENCH_MEASURE_ITERS" \
    --gemma4-shapes \
    --json-out "$OUT_DIR/bench-cuda-gemma4-shapes.json"
else
  record "bench_cuda_gemma4_shapes" "skip" "RUN_MICROBENCH=0"
fi

if [[ "$RUN_DEFAULT_POLICY" == "1" ]]; then
  run_default_policy_check
else
  record "default_auto_uncalibrated" "skip" "RUN_DEFAULT_POLICY=0"
fi

if [[ "$RUN_TARGET_ONLY" == "1" ]]; then
  run_target_generate "target_12b_q8" "$GEMMA12_Q8" "$COMBINED_BUDGET_MB" "$BACKEND_BUDGET_MB" "$KV_BUDGET_MB" "$SCRATCH_BUDGET_MB"
  run_target_generate "target_12b_q4k" "$GEMMA12_Q4" "$COMBINED_BUDGET_MB" "$BACKEND_BUDGET_MB" "$KV_BUDGET_MB" "$SCRATCH_BUDGET_MB"
  run_target_generate "target_e2b" "$E2B_TARGET" "$E2B_COMBINED_BUDGET_MB" "$E2B_BACKEND_BUDGET_MB" "$E2B_KV_BUDGET_MB" "$E2B_SCRATCH_BUDGET_MB"
  run_e4b_q4k_baseline
  run_e4b_qat_target
else
  record "target_only" "skip" "RUN_TARGET_ONLY=0"
fi
run_e4b_qat_long_target
run_e4b_qat_compressed_kv
run_e4b_qat_resident

if [[ "$RUN_12B_MTP" == "1" ]]; then
  run_mtp_matrix "mtp_12b" "$GEMMA12_Q8" "$GEMMA12_ASSISTANT_Q8" "$GEMMA12_ASSISTANT_Q4" "$COMBINED_BUDGET_MB" "$BACKEND_BUDGET_MB" "$KV_BUDGET_MB" "$SCRATCH_BUDGET_MB"
else
  record "mtp_12b" "skip" "RUN_12B_MTP=0"
fi

if [[ "$RUN_E2B_MTP" == "1" ]]; then
  run_mtp_matrix "mtp_e2b" "$E2B_TARGET" "$E2B_ASSISTANT_Q8" "$E2B_ASSISTANT_Q4" "$E2B_COMBINED_BUDGET_MB" "$E2B_BACKEND_BUDGET_MB" "$E2B_KV_BUDGET_MB" "$E2B_SCRATCH_BUDGET_MB"
else
  record "mtp_e2b" "skip" "RUN_E2B_MTP=0"
fi

run_mtp_matrix \
  "mtp_e4b_qat" \
  "$E4B_QAT" \
  "$E4B_QAT_ASSISTANT_Q8" \
  "$E4B_QAT_ASSISTANT_Q4" \
  "$E4B_QAT_COMBINED_BUDGET_MB" \
  "$E4B_QAT_BACKEND_BUDGET_MB" \
  "$E4B_QAT_KV_BUDGET_MB" \
  "$E4B_QAT_SCRATCH_BUDGET_MB" \
  "$RUN_E4B_QAT_MTP" \
  "$E4B_QAT_MTP_TOKENS" \
  "$E4B_QAT_MTP_SPEC_KS" \
  "$E4B_QAT_MTP_PROMPT_FILTER" \
  "$E4B_QAT_MTP_MODE" \
  "$E4B_QAT_MTP_CACHE_DTYPE"
run_e4b_qat_mtp_target_equivalence
run_e4b_qat_mtp_replay_stability
run_e4b_qat_mtp_replay_512
run_e4b_qat_mtp_hidden_ab
run_e4b_qat_mtp_acceptance_matrix
run_e4b_qat_mtp_donor_matrix
run_e4b_qat_provider_benchmark
write_qat_production_summary

if command -v python3 >/dev/null 2>&1; then
  python3 - "$OUT_DIR" "$TARGET_BASELINE_MIN_RATIO" "$AUTO_MIN_RATIO" "$PROMOTION_RATIO" "$MIN_12B_Q8_TOK_S" "$MIN_12B_Q4K_TOK_S" "$MIN_E2B_TOK_S" "$MIN_E4B_QAT_TOK_S" "$MIN_E4B_QAT_RUN_TOK_S" "$MIN_E4B_QAT_OVER_Q4K_RATIO" "$E4B_QAT_TOKENS" "$E4B_QAT_REPEATS" "$E4B_QAT_REQUIRE_FUSED" "$E4B_QAT_REQUIRE_FAST_GQA" "$E4B_QAT_REQUIRE_GRAPH_REPLAY" "$E4B_QAT_REQUIRE_DEVICE_TOKEN_HANDOFF" "$E4B_QAT_MIN_GRAPH_REPLAYS" "$E4B_QAT_MAX_LAUNCHES_PER_TOKEN" "$E4B_QAT_REQUIRE_GATED_DOWN_TILE8" "$E4B_QAT_REQUIRE_RAW_TOKEN_EXPORT" "$E4B_QAT_MAX_DOWNLOAD_SYNCS" "$E4B_QAT_REQUIRE_PLE_FUSION" "$E4B_QAT_LONG_TOKENS" "$E4B_QAT_LONG_MIN_TOKENS" "$MIN_E4B_QAT_LONG_TOK_S" "$E4B_QAT_LONG_MIN_GRAPH_REPLAYS" "$E4B_QAT_LONG_MAX_LAUNCHES_PER_TOKEN" "$E4B_QAT_LONG_MAX_DOWNLOAD_SYNCS" "$MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO" "$RUN_E4B_QAT_RESIDENT_SOAK" "$E4B_QAT_RESIDENT_SOAK_REQUESTS" "$E4B_QAT_RESIDENT_SOAK_CONCURRENCY" "$E4B_QAT_RESIDENT_SOAK_TOKENS" "$E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS" "$MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S" "$MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S" "$E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS" "$E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS" "$RUN_E4B_QAT_RESIDENT_BACKPRESSURE" "$E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS" "$E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS" "$E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY" "$E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS" "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED" "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED" "$E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS" "$E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS" "$MTP_MIN_ACTIVE_SPEED_RATIO" <<'PY'
import csv
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
target_baseline_min_ratio = float(sys.argv[2])
auto_min_ratio = float(sys.argv[3])
promotion_ratio = float(sys.argv[4])
target_rate_floors = {
    "target_12b_q8": float(sys.argv[5]),
    "target_12b_q4k": float(sys.argv[6]),
    "target_e2b": float(sys.argv[7]),
    "target_e4b_qat": float(sys.argv[8]),
}
e4b_qat_run_min_tok_s = float(sys.argv[9])
e4b_qat_over_q4k_min_ratio = float(sys.argv[10])
e4b_qat_min_tokens = int(float(sys.argv[11]))
e4b_qat_repeats = int(float(sys.argv[12]))
e4b_qat_require_fused = sys.argv[13].lower() not in {"0", "false", "off", "no"}
e4b_qat_require_fast_gqa = sys.argv[14].lower() not in {"0", "false", "off", "no"}
e4b_qat_require_graph_replay = sys.argv[15].lower() not in {"0", "false", "off", "no"}
e4b_qat_require_device_token_handoff = sys.argv[16].lower() not in {"0", "false", "off", "no"}
e4b_qat_min_graph_replays_arg = sys.argv[17].lower()
e4b_qat_max_launches_per_token = float(sys.argv[18])
e4b_qat_require_gated_down_tile8 = sys.argv[19].lower() not in {"0", "false", "off", "no"}
e4b_qat_require_raw_token_export = sys.argv[20].lower() not in {"0", "false", "off", "no"}
e4b_qat_max_download_syncs_arg = sys.argv[21].lower()
e4b_qat_require_ple_fusion = sys.argv[22].lower() not in {"0", "false", "off", "no"}
e4b_qat_long_tokens = int(float(sys.argv[23]))
e4b_qat_long_min_tokens = int(float(sys.argv[24]))
e4b_qat_long_min_tok_s = float(sys.argv[25])
e4b_qat_long_min_graph_replays_arg = sys.argv[26].lower()
e4b_qat_long_max_launches_per_token = float(sys.argv[27])
e4b_qat_long_max_download_syncs_arg = sys.argv[28].lower()
e4b_qat_resident_over_q4k_min_ratio = float(sys.argv[29])
e4b_qat_resident_soak_mode = sys.argv[30]
e4b_qat_resident_soak_requests = int(float(sys.argv[31]))
e4b_qat_resident_soak_concurrency = int(float(sys.argv[32]))
e4b_qat_resident_soak_tokens = int(float(sys.argv[33]))
e4b_qat_resident_soak_min_completion_tokens = int(float(sys.argv[34]))
e4b_qat_resident_soak_min_agg_tok_s = float(sys.argv[35])
e4b_qat_resident_soak_min_request_tok_s = float(sys.argv[36])
e4b_qat_resident_soak_max_p95_e2e_ms = float(sys.argv[37])
e4b_qat_resident_soak_min_graph_replays = sys.argv[38]
e4b_qat_resident_backpressure_mode = sys.argv[39]
e4b_qat_resident_max_concurrent_requests_arg = sys.argv[40]
e4b_qat_resident_backpressure_requests = int(float(sys.argv[41]))
e4b_qat_resident_backpressure_concurrency = int(float(sys.argv[42]))
e4b_qat_resident_backpressure_tokens = int(float(sys.argv[43]))
e4b_qat_resident_backpressure_min_accepted = int(float(sys.argv[44]))
e4b_qat_resident_backpressure_min_rejected = int(float(sys.argv[45]))
e4b_qat_resident_backpressure_max_reject_ms = float(sys.argv[46])
e4b_qat_resident_backpressure_min_graph_replays = sys.argv[47]
mtp_min_active_speed_ratio = float(sys.argv[48])
if e4b_qat_resident_max_concurrent_requests_arg == "":
    e4b_qat_resident_max_concurrent_requests = None
else:
    e4b_qat_resident_max_concurrent_requests = int(float(e4b_qat_resident_max_concurrent_requests_arg))
if e4b_qat_min_graph_replays_arg in {"", "auto"}:
    e4b_qat_min_graph_replays = max(1, e4b_qat_min_tokens - 64)
else:
    e4b_qat_min_graph_replays = int(float(e4b_qat_min_graph_replays_arg))
e4b_qat_max_download_syncs = None if e4b_qat_max_download_syncs_arg in {"", "off", "none", "no"} else int(float(e4b_qat_max_download_syncs_arg))
e4b_qat_long_max_download_syncs = None if e4b_qat_long_max_download_syncs_arg in {"", "off", "none", "no"} else int(float(e4b_qat_long_max_download_syncs_arg))

checks = []
failures = []
warnings = []
promotions = []

def add_check(name, ok, detail):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        failures.append(f"{name}: {detail}")

steps_path = out_dir / "steps.tsv"
step_rows = []
if steps_path.exists():
    with steps_path.open(newline="") as f:
        step_rows = list(csv.DictReader(f, delimiter="\t"))
    for row in step_rows:
        if row.get("status") == "fail":
            add_check(f"step_{row.get('step')}", False, row.get("detail", "failed"))
else:
    add_check("steps_tsv", False, "missing steps.tsv")

default_json = out_dir / "default_auto_uncalibrated.json"
if default_json.exists():
    data = json.loads(default_json.read_text())
    spec = data.get("speculative") or {}
    add_check(
        "default_auto_uncalibrated_policy",
        spec.get("speculation_policy") == "auto"
        and spec.get("speculation_calibration") == "none"
        and spec.get("speculation_policy_decision") == "disabled_uncalibrated"
        and spec.get("mtp_enabled") is False,
        json.dumps(spec, sort_keys=True),
    )
else:
    skipped = any(row.get("step") == "default_auto_uncalibrated" and row.get("status") == "skip" for row in step_rows)
    if not skipped:
        add_check("default_auto_uncalibrated_policy", False, "missing JSON")

def to_float(value):
    try:
        return float(value)
    except Exception:
        return None

def to_int(value):
    try:
        return int(value)
    except Exception:
        return None

microbench_json = out_dir / "bench-cuda-gemma4-shapes.json"
if microbench_json.exists():
    try:
        microbench = json.loads(microbench_json.read_text())
        shapes = microbench.get("gemma4_shapes") or []
        add_check("bench_cuda_gemma4_shapes_nonempty", bool(shapes), f"{len(shapes)} shapes")
        for row in shapes:
            label = row.get("label", "unknown")
            q8_speedup = to_float(row.get("q8_candidate_speedup"))
            q4_speedup = to_float(row.get("q4_candidate_speedup"))
            add_check(
                f"bench_cuda_gemma4_{label}_candidate_fields",
                bool(row.get("q8_candidate") and row.get("q4_candidate") and q8_speedup and q4_speedup),
                json.dumps({k: row.get(k) for k in ("q8_candidate", "q8_candidate_speedup", "q4_candidate", "q4_candidate_speedup")}, sort_keys=True),
            )
            if q8_speedup is not None:
                add_check(f"bench_cuda_gemma4_{label}_q8_speedup_positive", q8_speedup > 0, f"{q8_speedup:.3f}")
            if q4_speedup is not None:
                add_check(f"bench_cuda_gemma4_{label}_q4_speedup_positive", q4_speedup > 0, f"{q4_speedup:.3f}")
    except Exception as exc:
        add_check("bench_cuda_gemma4_shapes_json", False, repr(exc))

def analyze_summary(summary_path):
    matrix_name = summary_path.parent.name
    diagnostic_matrix = (
        matrix_name.startswith("mtp_acceptance_matrix_")
        or matrix_name.startswith("mtp_donor_matrix_")
        or matrix_name.startswith("mtp_target_equivalence")
    )
    rows = []
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        add_check(f"{summary_path.parent.name}_summary_nonempty", False, "no rows")
        return

    targets = {}
    for row in rows:
        if row.get("assistant") == "target" and row.get("status") == "ok":
            rate = to_float(row.get("decode_tok_s"))
            if rate is not None:
                targets[row.get("case")] = rate

    add_check(f"{summary_path.parent.name}_target_rows", bool(targets), f"{len(targets)} target rows")

    for row in rows:
        label = row.get("assistant", "")
        status = row.get("status", "")
        case = row.get("case", "")
        if status != "ok":
            add_check(f"{summary_path.parent.name}_{label}_{case}_status", False, status)
            continue
        if label == "target":
            continue

        policy = row.get("policy", "")
        calibration = row.get("calibration", "")
        decision = row.get("policy_decision", "")
        graph_replay = row.get("graph_replay", "")
        add_check(
            f"{summary_path.parent.name}_{label}_{case}_policy_metadata",
            bool(policy and calibration and decision and graph_replay),
            f"policy={policy} calibration={calibration} decision={decision} graph={graph_replay}",
        )

        rate = to_float(row.get("decode_tok_s"))
        target_rate = targets.get(case)
        if rate is None or target_rate is None:
            warnings.append(f"{summary_path.parent.name}:{label}:{case}: missing rate comparison")
            continue

        ratio = rate / target_rate if target_rate > 0 else 0.0
        if decision in {"active", "forced"}:
            if not diagnostic_matrix:
                add_check(
                    f"{summary_path.parent.name}_{label}_{case}_active_ratio",
                    ratio >= mtp_min_active_speed_ratio,
                    f"ratio={ratio:.3f} floor={mtp_min_active_speed_ratio:.3f} rate={rate:.3f} target={target_rate:.3f}",
                )
            fallbacks = to_int(row.get("dedicated_runtime_fallbacks"))
            if fallbacks is not None:
                add_check(
                    f"{summary_path.parent.name}_{label}_{case}_dedicated_runtime",
                    fallbacks == 0,
                    f"dedicated_runtime_fallbacks={fallbacks}",
                )
            if not diagnostic_matrix and ratio >= promotion_ratio:
                promotions.append(
                    {
                        "matrix": summary_path.parent.name,
                        "assistant": label,
                        "case": case,
                        "spec_k": row.get("spec_k"),
                        "ratio": round(ratio, 3),
                    }
                )
        elif decision.startswith("disabled_"):
            if decision == "disabled_slow":
                add_check(
                    f"{summary_path.parent.name}_{label}_{case}_slow_disabled",
                    ratio <= mtp_min_active_speed_ratio,
                    f"ratio={ratio:.3f} correctly disabled",
                )
        else:
            add_check(f"{summary_path.parent.name}_{label}_{case}_decision", False, decision)

for summary_path in sorted(out_dir.glob("*/summary.tsv")):
    analyze_summary(summary_path)

target_jsons = sorted(out_dir.glob("target_*.json"))
target_only_skipped = any(row.get("step") == "target_only" and row.get("status") == "skip" for row in step_rows)
if target_jsons or not target_only_skipped:
    add_check("target_only_jsons", bool(target_jsons), f"{len(target_jsons)} target-only JSON files")
e4b_qat_rates = []
for path in target_jsons:
    data = json.loads(path.read_text())
    rate = to_float(data.get("decode_tok_per_s"))
    add_check(path.stem + "_rate", rate is not None and rate > 0, f"decode_tok_per_s={rate}")
    is_e4b_qat = path.stem == "target_e4b_qat" or path.stem.startswith("target_e4b_qat_")
    floor = target_rate_floors.get(path.stem, target_rate_floors.get("target_e4b_qat", 0.0) if is_e4b_qat else 0.0)
    if is_e4b_qat:
        floor = e4b_qat_run_min_tok_s
    if floor > 0:
        add_check(
            path.stem + "_rate_floor",
            rate is not None and rate >= floor,
            f"decode_tok_per_s={rate} floor={floor}",
        )
    if is_e4b_qat:
        if rate is not None:
            e4b_qat_rates.append(rate)
        tokens = to_int(data.get("tokens"))
        add_check(
            path.stem + "_tokens",
            tokens is not None and tokens >= e4b_qat_min_tokens,
            f"tokens={tokens} min={e4b_qat_min_tokens}",
        )
        cuda = data.get("cuda") or {}
        if e4b_qat_max_download_syncs is not None:
            download_syncs = to_int(cuda.get("download_syncs"))
            add_check(
                path.stem + "_download_syncs",
                download_syncs is not None and download_syncs <= e4b_qat_max_download_syncs,
                f"download_syncs={download_syncs} max={e4b_qat_max_download_syncs}",
            )
        if e4b_qat_require_ple_fusion:
            add_mul_fused = to_int(cuda.get("add_mul_scalar_fused"))
            add_check(
                path.stem + "_ple_add_mul_scalar_fused",
                add_mul_fused is not None and add_mul_fused >= e4b_qat_min_tokens,
                f"add_mul_scalar_fused={add_mul_fused} min={e4b_qat_min_tokens}",
            )
        scalar_launches = to_int(cuda.get("launch_scalar"))
        add_check(
            path.stem + "_cuda_scalar_launches",
            scalar_launches == 0,
            f"launch_scalar={scalar_launches} max=0",
        )
        embedding_launches = to_int(cuda.get("launch_embedding"))
        add_check(
            path.stem + "_cuda_embedding_launches",
            embedding_launches is not None and embedding_launches <= e4b_qat_min_tokens + 1,
            f"launch_embedding={embedding_launches} max={e4b_qat_min_tokens + 1}",
        )
        if e4b_qat_require_fused:
            fused_fields = (
                "qkv_fused_q4_0",
                "linear_pair_fused_q4_0",
                "gated_down_fused_q4_0",
            )
            fallback_fields = (
                "qkv_fallback_unsupported",
                "qkv_kernel_unavailable",
                "linear_pair_fallbacks",
                "gated_down_fallbacks",
            )
            for field in fused_fields:
                value = to_int(cuda.get(field))
                add_check(path.stem + "_" + field, value is not None and value > 0, f"{field}={value}")
            for field in fallback_fields:
                value = to_int(cuda.get(field))
                add_check(path.stem + "_" + field, value == 0, f"{field}={value}")
            if e4b_qat_require_gated_down_tile8:
                tile8_hits = to_int(cuda.get("gated_down_fused_q4_0_tile8"))
                add_check(
                    path.stem + "_gated_down_fused_q4_0_tile8",
                    tile8_hits is not None and tile8_hits > 0,
                    f"gated_down_fused_q4_0_tile8={tile8_hits}",
                )
            q4_0_lm_head = to_int(cuda.get("lm_head_argmax_fused_q4_0"))
            add_check(
                path.stem + "_lm_head_argmax_fused_q4_0_disabled",
                q4_0_lm_head == 0,
                f"lm_head_argmax_fused_q4_0={q4_0_lm_head} max=0",
            )
            q6_lm_head = to_int(cuda.get("lm_head_argmax_fused_q6"))
            add_check(
                path.stem + "_lm_head_argmax_fused_q6",
                q6_lm_head is not None and q6_lm_head > 0,
                f"lm_head_argmax_fused_q6={q6_lm_head}",
            )
            lm_head_fallbacks = to_int(cuda.get("lm_head_argmax_fallbacks"))
            add_check(
                path.stem + "_lm_head_argmax_fallbacks",
                lm_head_fallbacks == 0,
                f"lm_head_argmax_fallbacks={lm_head_fallbacks}",
            )
        if e4b_qat_require_fast_gqa:
            fast_hits = to_int(cuda.get("launch_attention_gqa_decode_fast"))
            add_check(
                path.stem + "_launch_attention_gqa_decode_fast",
                fast_hits is not None and fast_hits > 0,
                f"launch_attention_gqa_decode_fast={fast_hits}",
            )
            fast_fallbacks = to_int(cuda.get("launch_attention_gqa_decode_fast_fallbacks"))
            add_check(
                path.stem + "_launch_attention_gqa_decode_fast_fallbacks",
                fast_fallbacks == 0,
                f"launch_attention_gqa_decode_fast_fallbacks={fast_fallbacks}",
            )
        if e4b_qat_require_raw_token_export:
            cuda_generate = data.get("cuda_generate") or {}
            to_float32_calls = to_int(cuda_generate.get("to_float32_calls"))
            to_float32_bytes = to_int(cuda_generate.get("to_float32_bytes"))
            add_check(
                path.stem + "_raw_token_export_to_float32_calls",
                to_float32_calls == 0,
                f"cuda_generate.to_float32_calls={to_float32_calls}",
            )
            add_check(
                path.stem + "_raw_token_export_to_float32_bytes",
                to_float32_bytes == 0,
                f"cuda_generate.to_float32_bytes={to_float32_bytes}",
            )
        if e4b_qat_require_graph_replay:
            graph_replays = to_int(cuda.get("graph_capture_persistent_replays"))
            add_check(
                path.stem + "_graph_capture_persistent_replays",
                graph_replays is not None and graph_replays >= e4b_qat_min_graph_replays,
                f"graph_capture_persistent_replays={graph_replays} min={e4b_qat_min_graph_replays}",
            )
            capacity_skips = to_int(cuda.get("graph_capture_capacity_skips"))
            add_check(
                path.stem + "_graph_capture_capacity_skips",
                capacity_skips == 0,
                f"graph_capture_capacity_skips={capacity_skips}",
            )
            launches_per_token = to_float(cuda.get("launches_per_token"))
            add_check(
                path.stem + "_launches_per_token",
                launches_per_token is not None and launches_per_token <= e4b_qat_max_launches_per_token,
                f"launches_per_token={launches_per_token} max={e4b_qat_max_launches_per_token}",
            )
        if e4b_qat_require_device_token_handoff:
            generation_runtime = data.get("generation_decoder_runtime") or {}
            handoff_attempts = to_int(generation_runtime.get("device_token_handoff_attempts"))
            handoff_hits = to_int(generation_runtime.get("device_token_handoff_hits"))
            handoff_fallbacks = to_int(generation_runtime.get("device_token_handoff_fallbacks"))
            handoff_seeds = to_int(generation_runtime.get("device_token_handoff_seeds"))
            min_handoffs = max(0, e4b_qat_min_tokens - 1)
            add_check(
                path.stem + "_device_token_handoff_attempts",
                handoff_attempts is not None and handoff_attempts >= min_handoffs,
                f"device_token_handoff_attempts={handoff_attempts} min={min_handoffs}",
            )
            add_check(
                path.stem + "_device_token_handoff_hits",
                handoff_hits is not None and handoff_hits >= min_handoffs,
                f"device_token_handoff_hits={handoff_hits} min={min_handoffs}",
            )
            add_check(
                path.stem + "_device_token_handoff_fallbacks",
                handoff_fallbacks == 0,
                f"device_token_handoff_fallbacks={handoff_fallbacks}",
            )
            add_check(
                path.stem + "_device_token_handoff_seeds",
                handoff_seeds is not None and handoff_seeds >= 1,
                f"device_token_handoff_seeds={handoff_seeds}",
            )

long_path = out_dir / "e4b_qat_long.json"
if long_path.exists():
    data = json.loads(long_path.read_text())
    rate = to_float(data.get("decode_tok_per_s"))
    tokens = to_int(data.get("tokens"))
    long_counter_floor = max(e4b_qat_long_min_tokens, tokens or 0)
    long_graph_floor = (
        max(1, long_counter_floor - 64)
        if e4b_qat_long_min_graph_replays_arg in {"", "auto"}
        else int(float(e4b_qat_long_min_graph_replays_arg))
    )
    add_check(
        "e4b_qat_long_requested_tokens",
        e4b_qat_long_tokens >= e4b_qat_long_min_tokens,
        f"tokens={e4b_qat_long_tokens} min={e4b_qat_long_min_tokens}",
    )
    add_check(
        "e4b_qat_long_tokens",
        tokens is not None and tokens >= e4b_qat_long_min_tokens,
        f"tokens={tokens} min={e4b_qat_long_min_tokens}",
    )
    add_check(
        "e4b_qat_long_rate_floor",
        rate is not None and rate >= e4b_qat_long_min_tok_s,
        f"decode_tok_per_s={rate} floor={e4b_qat_long_min_tok_s}",
    )
    cuda = data.get("cuda") or {}
    if e4b_qat_long_max_download_syncs is not None:
        download_syncs = to_int(cuda.get("download_syncs"))
        add_check(
            "e4b_qat_long_download_syncs",
            download_syncs is not None and download_syncs <= e4b_qat_long_max_download_syncs,
            f"download_syncs={download_syncs} max={e4b_qat_long_max_download_syncs}",
        )
    if e4b_qat_require_ple_fusion:
        add_mul_fused = to_int(cuda.get("add_mul_scalar_fused"))
        add_check(
            "e4b_qat_long_ple_add_mul_scalar_fused",
            add_mul_fused is not None and add_mul_fused >= long_counter_floor,
            f"add_mul_scalar_fused={add_mul_fused} min={long_counter_floor}",
        )
    scalar_launches = to_int(cuda.get("launch_scalar"))
    add_check(
        "e4b_qat_long_cuda_scalar_launches",
        scalar_launches == 0,
        f"launch_scalar={scalar_launches} max=0",
    )
    embedding_launches = to_int(cuda.get("launch_embedding"))
    # EOS rollback can leave two lookahead embedding launches in this long run.
    add_check(
        "e4b_qat_long_cuda_embedding_launches",
        embedding_launches is not None and embedding_launches <= long_counter_floor + 3,
        f"launch_embedding={embedding_launches} max={long_counter_floor + 3}",
    )
    if e4b_qat_require_fused:
        for field in ("qkv_fused_q4_0", "linear_pair_fused_q4_0", "gated_down_fused_q4_0"):
            value = to_int(cuda.get(field))
            add_check("e4b_qat_long_" + field, value is not None and value > 0, f"{field}={value}")
        for field in ("qkv_fallback_unsupported", "qkv_kernel_unavailable", "linear_pair_fallbacks", "gated_down_fallbacks"):
            value = to_int(cuda.get(field))
            add_check("e4b_qat_long_" + field, value == 0, f"{field}={value}")
        if e4b_qat_require_gated_down_tile8:
            tile8_hits = to_int(cuda.get("gated_down_fused_q4_0_tile8"))
            add_check(
                "e4b_qat_long_gated_down_fused_q4_0_tile8",
                tile8_hits is not None and tile8_hits > 0,
                f"gated_down_fused_q4_0_tile8={tile8_hits}",
            )
        q4_0_lm_head = to_int(cuda.get("lm_head_argmax_fused_q4_0"))
        add_check(
            "e4b_qat_long_lm_head_argmax_fused_q4_0_disabled",
            q4_0_lm_head == 0,
            f"lm_head_argmax_fused_q4_0={q4_0_lm_head} max=0",
        )
        q6_lm_head = to_int(cuda.get("lm_head_argmax_fused_q6"))
        add_check(
            "e4b_qat_long_lm_head_argmax_fused_q6",
            q6_lm_head is not None and q6_lm_head > 0,
            f"lm_head_argmax_fused_q6={q6_lm_head}",
        )
        lm_head_fallbacks = to_int(cuda.get("lm_head_argmax_fallbacks"))
        add_check(
            "e4b_qat_long_lm_head_argmax_fallbacks",
            lm_head_fallbacks == 0,
            f"lm_head_argmax_fallbacks={lm_head_fallbacks}",
        )
    if e4b_qat_require_fast_gqa:
        fast_hits = to_int(cuda.get("launch_attention_gqa_decode_fast"))
        add_check(
            "e4b_qat_long_launch_attention_gqa_decode_fast",
            fast_hits is not None and fast_hits > 0,
            f"launch_attention_gqa_decode_fast={fast_hits}",
        )
        fast_fallbacks = to_int(cuda.get("launch_attention_gqa_decode_fast_fallbacks"))
        add_check(
            "e4b_qat_long_launch_attention_gqa_decode_fast_fallbacks",
            fast_fallbacks == 0,
            f"launch_attention_gqa_decode_fast_fallbacks={fast_fallbacks}",
        )
    if e4b_qat_require_raw_token_export:
        cuda_generate = data.get("cuda_generate") or {}
        to_float32_calls = to_int(cuda_generate.get("to_float32_calls"))
        to_float32_bytes = to_int(cuda_generate.get("to_float32_bytes"))
        add_check(
            "e4b_qat_long_raw_token_export_to_float32_calls",
            to_float32_calls == 0,
            f"cuda_generate.to_float32_calls={to_float32_calls}",
        )
        add_check(
            "e4b_qat_long_raw_token_export_to_float32_bytes",
            to_float32_bytes == 0,
            f"cuda_generate.to_float32_bytes={to_float32_bytes}",
        )
    if e4b_qat_require_graph_replay:
        graph_replays = to_int(cuda.get("graph_capture_persistent_replays"))
        add_check(
            "e4b_qat_long_graph_capture_persistent_replays",
            graph_replays is not None and graph_replays >= long_graph_floor,
            f"graph_capture_persistent_replays={graph_replays} min={long_graph_floor}",
        )
        capacity_skips = to_int(cuda.get("graph_capture_capacity_skips"))
        add_check(
            "e4b_qat_long_graph_capture_capacity_skips",
            capacity_skips == 0,
            f"graph_capture_capacity_skips={capacity_skips}",
        )
        launches_per_token = to_float(cuda.get("launches_per_token"))
        add_check(
            "e4b_qat_long_launches_per_token",
            launches_per_token is not None and launches_per_token <= e4b_qat_long_max_launches_per_token,
            f"launches_per_token={launches_per_token} max={e4b_qat_long_max_launches_per_token}",
        )
    if e4b_qat_require_device_token_handoff:
        generation_runtime = data.get("generation_decoder_runtime") or {}
        handoff_attempts = to_int(generation_runtime.get("device_token_handoff_attempts"))
        handoff_hits = to_int(generation_runtime.get("device_token_handoff_hits"))
        handoff_fallbacks = to_int(generation_runtime.get("device_token_handoff_fallbacks"))
        handoff_seeds = to_int(generation_runtime.get("device_token_handoff_seeds"))
        min_handoffs = max(0, long_counter_floor - 1)
        add_check(
            "e4b_qat_long_device_token_handoff_attempts",
            handoff_attempts is not None and handoff_attempts >= min_handoffs,
            f"device_token_handoff_attempts={handoff_attempts} min={min_handoffs}",
        )
        add_check(
            "e4b_qat_long_device_token_handoff_hits",
            handoff_hits is not None and handoff_hits >= min_handoffs,
            f"device_token_handoff_hits={handoff_hits} min={min_handoffs}",
        )
        add_check(
            "e4b_qat_long_device_token_handoff_fallbacks",
            handoff_fallbacks == 0,
            f"device_token_handoff_fallbacks={handoff_fallbacks}",
        )
        add_check(
            "e4b_qat_long_device_token_handoff_seeds",
            handoff_seeds is not None and handoff_seeds >= 1,
            f"device_token_handoff_seeds={handoff_seeds}",
        )

if e4b_qat_rates:
    e4b_qat_floor = target_rate_floors.get("target_e4b_qat", 0.0)
    e4b_qat_min_rate = min(e4b_qat_rates)
    e4b_qat_avg_rate = sum(e4b_qat_rates) / len(e4b_qat_rates)
    add_check(
        "target_e4b_qat_repeat_count",
        len(e4b_qat_rates) >= e4b_qat_repeats,
        f"runs={len(e4b_qat_rates)} expected={e4b_qat_repeats}",
    )
    add_check(
        "target_e4b_qat_repeat_min_rate",
        e4b_qat_min_rate >= e4b_qat_run_min_tok_s,
        f"min_decode_tok_per_s={e4b_qat_min_rate:.6f} floor={e4b_qat_run_min_tok_s}",
    )
    add_check(
        "target_e4b_qat_repeat_avg_rate",
        e4b_qat_avg_rate >= e4b_qat_floor,
        f"avg_decode_tok_per_s={e4b_qat_avg_rate:.6f} floor={e4b_qat_floor}",
    )
    q4k_path = out_dir / "target_e4b_q4k.json"
    if q4k_path.exists():
        q4k_data = json.loads(q4k_path.read_text())
        q4k_rate = to_float(q4k_data.get("decode_tok_per_s"))
        ratio = e4b_qat_avg_rate / q4k_rate if q4k_rate and q4k_rate > 0 else 0.0
        add_check(
            "target_e4b_qat_over_q4k_ratio",
            q4k_rate is not None and ratio >= e4b_qat_over_q4k_min_ratio,
            f"qat_avg={e4b_qat_avg_rate:.6f} q4k={q4k_rate} ratio={ratio:.3f} floor={e4b_qat_over_q4k_min_ratio}",
        )

environment = {}
cuda_environment_path = out_dir / "cuda_environment.json"
if cuda_environment_path.exists():
    try:
        environment["cuda_smoke"] = json.loads(cuda_environment_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_check("cuda_environment_json", False, repr(exc))
else:
    add_check("cuda_environment_json", False, "missing cuda_environment.json")

qat_production_summary = None
qat_production_summary_path = out_dir / "e4b_qat_production_summary.json"
if qat_production_summary_path.exists():
    try:
        qat_production_summary = json.loads(qat_production_summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_check("qat_production_summary_json", False, repr(exc))

step_status = {row.get("step"): row.get("status") for row in step_rows}
provider_benchmark_paths = {
    "baseline_json": out_dir / "e4b_qat_provider_baselines.json",
    "rows_tsv": out_dir / "e4b_qat_provider_baselines.tsv",
    "benchmark_log": out_dir / "e4b_qat_provider_benchmark.log",
    "validation_json": out_dir / "e4b_qat_provider_baseline_validation.json",
    "validation_log": out_dir / "e4b_qat_provider_baseline_validation.log",
}
provider_benchmark = {
    "step_status": step_status.get("e4b_qat_provider_benchmark"),
    "validation_step_status": step_status.get("e4b_qat_provider_baseline_validation"),
    "artifacts": {name: str(path) for name, path in provider_benchmark_paths.items()},
    "baseline": None,
    "validation": None,
}
if provider_benchmark["step_status"] == "ok":
    add_check(
        "provider_benchmark_baseline_json",
        provider_benchmark_paths["baseline_json"].exists(),
        str(provider_benchmark_paths["baseline_json"]),
    )
    add_check(
        "provider_benchmark_rows_tsv",
        provider_benchmark_paths["rows_tsv"].exists(),
        str(provider_benchmark_paths["rows_tsv"]),
    )
    if provider_benchmark_paths["baseline_json"].exists():
        try:
            provider_benchmark["baseline"] = json.loads(provider_benchmark_paths["baseline_json"].read_text(encoding="utf-8"))
        except Exception as exc:
            add_check("provider_benchmark_baseline_json_parse", False, repr(exc))
    if provider_benchmark_paths["validation_json"].exists():
        try:
            provider_benchmark["validation"] = json.loads(provider_benchmark_paths["validation_json"].read_text(encoding="utf-8"))
            validation_ok = bool((provider_benchmark["validation"].get("verdict") or {}).get("ok"))
            add_check(
                "provider_benchmark_baseline_validation",
                validation_ok,
                f"ok={validation_ok}",
            )
        except Exception as exc:
            add_check("provider_benchmark_validation_json_parse", False, repr(exc))
    else:
        add_check("provider_benchmark_validation_json", False, str(provider_benchmark_paths["validation_json"]))

report = {
    "ok": not failures,
    "out_dir": str(out_dir),
    "environment": environment,
    "provider_benchmark": provider_benchmark,
    "qat_production_summary": qat_production_summary,
    "thresholds": {
        "target_baseline_min_ratio": target_baseline_min_ratio,
        "auto_min_ratio": auto_min_ratio,
        "promotion_ratio": promotion_ratio,
        "target_rate_floors_tok_s": target_rate_floors,
        "e4b_qat_run_min_tok_s": e4b_qat_run_min_tok_s,
        "e4b_qat_over_q4k_min_ratio": e4b_qat_over_q4k_min_ratio,
        "e4b_qat_min_tokens": e4b_qat_min_tokens,
        "e4b_qat_repeats": e4b_qat_repeats,
        "e4b_qat_require_fused": e4b_qat_require_fused,
        "e4b_qat_require_fast_gqa": e4b_qat_require_fast_gqa,
        "e4b_qat_require_graph_replay": e4b_qat_require_graph_replay,
        "e4b_qat_require_device_token_handoff": e4b_qat_require_device_token_handoff,
        "e4b_qat_require_raw_token_export": e4b_qat_require_raw_token_export,
        "e4b_qat_require_ple_fusion": e4b_qat_require_ple_fusion,
        "e4b_qat_require_gated_down_tile8": e4b_qat_require_gated_down_tile8,
        "e4b_qat_min_graph_replays": e4b_qat_min_graph_replays,
        "e4b_qat_max_launches_per_token": e4b_qat_max_launches_per_token,
        "e4b_qat_max_download_syncs": e4b_qat_max_download_syncs,
        "e4b_qat_long_tokens": e4b_qat_long_tokens,
        "e4b_qat_long_min_tokens": e4b_qat_long_min_tokens,
        "e4b_qat_long_min_tok_s": e4b_qat_long_min_tok_s,
        "e4b_qat_long_min_graph_replays": e4b_qat_long_min_graph_replays_arg,
        "e4b_qat_long_max_launches_per_token": e4b_qat_long_max_launches_per_token,
        "e4b_qat_long_max_download_syncs": e4b_qat_long_max_download_syncs,
        "e4b_qat_resident_over_q4k_min_ratio": e4b_qat_resident_over_q4k_min_ratio,
        "e4b_qat_resident_soak_mode": e4b_qat_resident_soak_mode,
        "e4b_qat_resident_soak_requests": e4b_qat_resident_soak_requests,
        "e4b_qat_resident_soak_concurrency": e4b_qat_resident_soak_concurrency,
        "e4b_qat_resident_soak_tokens": e4b_qat_resident_soak_tokens,
        "e4b_qat_resident_soak_min_completion_tokens": e4b_qat_resident_soak_min_completion_tokens,
        "e4b_qat_resident_soak_min_agg_tok_s": e4b_qat_resident_soak_min_agg_tok_s,
        "e4b_qat_resident_soak_min_request_tok_s": e4b_qat_resident_soak_min_request_tok_s,
        "e4b_qat_resident_soak_max_p95_e2e_ms": e4b_qat_resident_soak_max_p95_e2e_ms,
        "e4b_qat_resident_soak_min_graph_replays": e4b_qat_resident_soak_min_graph_replays,
        "e4b_qat_resident_backpressure_mode": e4b_qat_resident_backpressure_mode,
        "e4b_qat_resident_max_concurrent_requests": e4b_qat_resident_max_concurrent_requests,
        "e4b_qat_resident_backpressure_requests": e4b_qat_resident_backpressure_requests,
        "e4b_qat_resident_backpressure_concurrency": e4b_qat_resident_backpressure_concurrency,
        "e4b_qat_resident_backpressure_tokens": e4b_qat_resident_backpressure_tokens,
        "e4b_qat_resident_backpressure_min_accepted": e4b_qat_resident_backpressure_min_accepted,
        "e4b_qat_resident_backpressure_min_rejected": e4b_qat_resident_backpressure_min_rejected,
        "e4b_qat_resident_backpressure_max_reject_ms": e4b_qat_resident_backpressure_max_reject_ms,
        "e4b_qat_resident_backpressure_min_graph_replays": e4b_qat_resident_backpressure_min_graph_replays,
    },
    "steps": step_rows,
    "checks": checks,
    "warnings": warnings,
    "promotions": promotions,
    "failures": failures,
}
(out_dir / "readiness.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
with (out_dir / "readiness.tsv").open("w") as f:
    f.write("ok\tname\tdetail\n")
    for check in checks:
        f.write(f"{check['ok']}\t{check['name']}\t{check['detail']}\n")

print(f"readiness_json={out_dir / 'readiness.json'}")
print(f"readiness_tsv={out_dir / 'readiness.tsv'}")
if failures:
    print("readiness=fail")
    for failure in failures:
        print(f"failure: {failure}")
    sys.exit(1)
print("readiness=ok")
if promotions:
    print("promotions=" + json.dumps(promotions, sort_keys=True))
PY
else
  record "readiness_summary" "skip" "python3 unavailable"
fi

echo "out_dir=$OUT_DIR"
if [[ "$FAILED" != "0" ]]; then
  exit 1
fi
