#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: gemma4_cuda_production_gate.sh [--quick|--full|--bench-only|--mtp-only]

Runs the Gemma4 CUDA production-readiness gate for resident target inference,
TurboQuant compressed KV, and MTP auto-policy behavior.

Environment overrides:
  ZIG_BIN                       path to Zig 0.16 binary
  ANTFLY_BIN                    path to antfly-inference binary
  OUT_DIR                       output directory for logs/json timing
  E2B_MODEL                     Gemma4 E2B target model directory
  E4B_QAT_MODEL                 Gemma4 E4B QAT q4_0 GGUF model directory
  RUN_E4B_QAT                   auto|required|off (default: auto)
  RUN_E4B_QAT_LONG              auto|required|off (default: off)
  RUN_E4B_QAT_RESIDENT          auto|required|off (default: off)
  RUN_E4B_QAT_RESIDENT_SOAK     auto|required|off (default: off)
  RUN_E4B_QAT_RESIDENT_BACKPRESSURE auto|required|off (default: off)
  RUN_E4B_Q4K_BASELINE          auto|required|off (default: auto)
  RUN_E4B_Q4K_RESIDENT_BASELINE auto|required|off (default: auto)
  RUN_CUDA_ENV                  auto|required|off capture cuda-info --smoke metadata (default: auto)
  RUN_E4B_QAT_PROVIDER_COMPARISON auto|required|off require local QAT to clear provider baselines (default: off)
  RUN_E4B_QAT_PROVIDER_BENCHMARK auto|required|off collect OpenAI-compatible provider baseline (default: off)
  RUN_E4B_QAT_COMPETITIVE_FLOOR auto|required|off require local QAT to clear named tok/s floors (default: off)
  E4B_QAT_COMPETITIVE_FLOORS space-separated metric=tok/s floors (default: compressed_kv_decode_tok_s=36.0)
  E4B_QAT_PROVIDER_BASELINE_JSON provider baseline JSON path
  E4B_QAT_PROVIDER_BASELINE_INLINE inline provider baseline JSON
  MIN_E4B_QAT_PROVIDER_RATIO    minimum local/provider tok/s ratio (default: 1.0)
  E4B_QAT_REQUIRE_PROVIDER_METADATA 1 to require provider baseline provenance fields (default: 1)
  E4B_QAT_PROVIDER_BASE_URL     OpenAI-compatible provider base URL for provider benchmark
  E4B_QAT_PROVIDER_API_KEY_ENV  API key environment variable name (default: PROVIDER_API_KEY)
  E4B_QAT_PROVIDER_NAME         provider name for benchmark provenance
  E4B_QAT_PROVIDER_MODEL        provider model id (default: google/gemma-4-E4B-it-qat-q4_0-gguf)
  E4B_QAT_PROVIDER_HARDWARE     provider hardware/instance class for provenance
  E4B_QAT_PROVIDER_SOURCE_URL   provider run/dashboard URL for provenance
  E4B_QAT_PROVIDER_STREAM       1 to benchmark streaming completions (default: 0)
  E4B_QAT_PROVIDER_RATE_SOURCE  e2e or stream_decode baseline rate source (default: e2e)
  E4B_QAT_PROVIDER_BASELINE_STATS comma-separated provider stats to emit (default: avg,median,min)
  E4B_Q4K_BASELINE_MODEL        Gemma4 E4B Q4_K baseline model directory
  E4B_QAT_TOKENS                E4B QAT target-only generated tokens (default: 512)
  E4B_QAT_REPEATS               E4B QAT target-only repeat runs (default: 2)
  E4B_QAT_LONG_TOKENS           E4B QAT long generated-token request (default: 1024)
  E4B_QAT_LONG_MIN_TOKENS       E4B QAT long generated-token floor, EOS-aware (default: 768)
  E4B_QAT_LONG_KV_BUDGET_MB     E4B QAT long KV budget (default: 1024)
  E4B_QAT_LONG_FORCE_KV_CAPACITY forced long graph replay KV capacity (default: 2048)
  E4B_QAT_LONG_MIN_GRAPH_REPLAYS long replay floor, or auto for generated tokens-64 (default: auto)
  E4B_QAT_LONG_MAX_LAUNCHES_PER_TOKEN long launch density ceiling (default: 22)
  E4B_QAT_LONG_MAX_DOWNLOAD_SYNCS long download-sync ceiling, off to disable (default: 4)
  E4B_QAT_RESIDENT_TOKENS       preloaded server generated tokens per request (default: 512)
  E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS minimum completion tokens per warm request (default: tokens)
  E4B_QAT_RESIDENT_WARM_REPEATS preloaded server measured warm requests (default: 2)
  E4B_QAT_RESIDENT_PROMPT       prompt for preloaded server warm requests
  E4B_QAT_RESIDENT_PORT         preloaded server port (default: auto)
  E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS weighted in-flight request capacity for server overload checks
  E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY off|auto|required for server gate (default: required)
  E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS server replay floor, or auto for tokens/3 (default: auto)
  E4B_QAT_RESIDENT_SOAK_REQUESTS concurrent soak requests after warm pass (default: 6)
  E4B_QAT_RESIDENT_SOAK_CONCURRENCY soak request concurrency (default: 2)
  E4B_QAT_RESIDENT_SOAK_TOKENS soak generated tokens per request (default: 256)
  E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS soak completion token floor (default: tokens)
  E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS soak replay floor, or auto (default: auto)
  E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS overload probe requests (default: 4)
  E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY overload probe concurrency (default: 4)
  E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS overload probe generated tokens per request (default: 256)
  E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED minimum accepted overload requests (default: 1)
  E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED minimum HTTP 503 overload requests (default: 1)
  E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS HTTP 503 latency ceiling (default: 2000)
  E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS overload replay floor, or auto (default: auto)
  E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY off|auto|required for Q4_K server baseline (default: required)
  E4B_QAT_REQUIRE_FUSED         1 to require Q4_0 fused CUDA counters (default: 1)
  E4B_QAT_Q4_0_GATED_DOWN_TILE8 1 to enable Q4_0 gated-down tile8 for QAT (default: 0)
  E4B_QAT_REQUIRE_GATED_DOWN_TILE8 1 to require gated-down tile8 hits (default: follows enable)
  E4B_QAT_Q4_0_PLE_GATE_FUSION 1 to enable Q4_0 fused PLE gate projection (default: 1)
  E4B_QAT_PLE_RMS_EMBED_FUSION 1 to enable fused PLE RMS/embed construction (default: 0)
  E4B_QAT_REQUIRE_FAST_GQA      1 to require fast f32 GQA decode counters (default: 1)
  E4B_QAT_DECODE_GRAPH_REPLAY   off|auto|required for QAT decode graph replay (default: required)
  E4B_QAT_REQUIRE_GRAPH_REPLAY  1 to require persistent replay telemetry (default: 1)
  E4B_QAT_REQUIRE_DEVICE_TOKEN_HANDOFF 1 to require greedy token handoff telemetry (default: 1)
  E4B_QAT_REQUIRE_RAW_TOKEN_EXPORT 1 to require raw i32 token export/no toFloat32 token path (default: 1)
  E4B_QAT_PENDING_TOKEN_READBACK 1 to enable delayed CUDA token readback for QAT (default: 1)
  E4B_QAT_MAX_DOWNLOAD_SYNCS    QAT generate download-sync ceiling, off to disable (default: 4)
  E4B_QAT_REQUIRE_PLE_FUSION    1 to require fused PLE add-multiply counters (default: 1)
  E4B_QAT_MIN_GRAPH_REPLAYS     replay floor, or auto for tokens-64 (default: auto)
  E4B_QAT_MAX_LAUNCHES_PER_TOKEN QAT launch density ceiling (default: 22.5)
  E4B_QAT_TEMP_SLOT_PERIOD      pinned temp slot period for graph capture (default: 0)
  E4B_QAT_CAPTURE_ALLOW_UNPINNED allow stable-reuse graph capture without pinned slots (default: 1)
  E4B_QAT_CAPTURE_MIN_ALLOC_SEQ delayed capture warmup allocation sequence (default: 10000)
  E4B_QAT_FORCE_KV_CAPACITY     forced graph replay KV capacity tokens (default: 1024)
  GEMMA12B_Q4_MODEL             Gemma4 12B Q4 model path/directory
  LONG_CONTEXT_TOKENS           full-mode E2B polar4 stress tokens (default: 512)
  REQUIRE_SPEED_THRESHOLDS      1 to enforce tok/s floors (default: full only)
  MIN_E2B_TEXT_TOK_S            full-mode E2B floor for real_bench_128 (default: 15.0)
  MIN_12B_Q4_TOK_S              full-mode 12B Q4 floor for 32-token runs (default: 8.0)
  MIN_E4B_QAT_TOK_S             full-mode E4B QAT floor (default: 24.0)
  MIN_E4B_QAT_RUN_TOK_S         per-run E4B QAT jitter floor (default: 24.0)
  MIN_E4B_QAT_LONG_TOK_S        long E4B QAT floor (default: 15.0)
  MIN_E4B_QAT_RESIDENT_WARM_TOK_S preloaded server warm e2e floor (default: 24.0)
  MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S resident soak aggregate floor (default: 14.0)
  MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S resident soak per-request floor (default: 8.0)
  E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS resident soak p95 latency ceiling (default: 35000)
  MIN_E4B_QAT_OVER_Q4K_RATIO    QAT avg tok/s over E4B Q4_K baseline floor (default: 1.25)
  MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO resident QAT avg tok/s over Q4_K floor (default: 1.25)
  RUN_RESIDENT                  auto|required|off (default: auto)
  RESIDENT_MODEL                resident server model (default: E2B_MODEL)
  RESIDENT_PROMPT               resident server prompt
  RESIDENT_TOKENS               resident server generated tokens (default: 32)
  RESIDENT_CACHE_DTYPE          resident server cache dtype (default: f32)
  RESIDENT_MODELS_DIR           models dir for resident server (default: .models)
  RESIDENT_PORT                 resident server port (default: auto)
  MAX_RESIDENT_WARM_COLD_RATIO  warm/cold E2E ratio ceiling (default: 0.75)
  MIN_RESIDENT_WARM_TOK_S       warm resident E2E tok/s floor (default: 10.0)
  RUN_MTP                       auto|required|off (default: auto)
  MTP_TARGET_MODEL              MTP target model (default: E2B_MODEL)
  MTP_DRAFT_MODEL               MTP assistant GGUF
  MTP_TOKENS                    MTP comparison tokens (default: 128)
  MTP_SPECULATIVE_K             MTP speculative window (default: 2)
  MTP_CACHE_DTYPE               target/draft KV cache dtype for MTP (default: f32)
  MTP_TURBOQUANT_MIN_TOKENS     TurboQuant min token threshold for MTP (default: 0)
  MTP_MIN_ACTIVE_SPEED_RATIO    active-MTP tok/s floor vs target-only (default: 1.0)
  MTP_TARGET_REPLAY             off|auto|required target-side replay policy (default: auto)
  MTP_REPLAY_CONTEXT_KEY        auto|0|1 context-keyed replay cache (default: auto)
  MTP_UNSAFE_TARGET_REPLAY      auto|0|1 allow unsafe target replay testing (default: auto)
  MTP_ASSISTANT_REPLAY          auto|0|1 strict assistant replay testing (default: auto)
  MTP_MATERIALIZE_REPLAY        auto|0|1 materialize replay cache setup (default: auto)
  CUDA_CAPTURE_PERSISTENT_REPLAY auto|0|1 persistent replay capture for MTP (default: auto)
  CUDA_TEMP_SLOT_PERIOD         pinned temp slot period override for MTP
  CUDA_TEMP_SLOT_SKIP           pinned temp slot skip override for MTP
  MTP_POSITION_MODE             target_constant|target_absolute|legacy_one diagnostic override
  MTP_TARGET_HIDDEN_SOURCE      final|pre_norm diagnostic override
  MTP_CONCAT_ORDER              embedding_activation|activation_embedding diagnostic override
  MTP_KV_DONOR_MODE             shared_type|tail_base|assistant_index|non_shared_tail_base|first_shared_base diagnostic override
USAGE
}

mode="quick"
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)
      mode="quick"
      ;;
    --full)
      mode="full"
      ;;
    --bench-only)
      mode="bench-only"
      ;;
    --mtp-only)
      mode="mtp-only"
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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
inference_dir="$repo_root/zig/pkg/inference"
antfly_bin="${ANTFLY_BIN:-${ANTFY_BIN:-$inference_dir/zig-out/bin/antfly-inference}}"
out_dir="${OUT_DIR:-/tmp/antfly-gemma4-cuda-production-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out_dir"

e2b_model="${E2B_MODEL:-$repo_root/.models/unsloth/gemma-4-E2B-it-qat-GGUF}"
e4b_qat_model="${E4B_QAT_MODEL:-$repo_root/.models/google/gemma-4-E4B-it-qat-q4_0-gguf}"
e4b_q4k_baseline_model="${E4B_Q4K_BASELINE_MODEL:-${E4B_Q4K_BASELINE:-$repo_root/.models/google/gemma-4-E4B-it-q4_k}}"
run_e4b_qat="${RUN_E4B_QAT:-auto}"
run_e4b_qat_long="${RUN_E4B_QAT_LONG:-off}"
run_e4b_qat_resident="${RUN_E4B_QAT_RESIDENT:-off}"
run_e4b_qat_resident_soak="${RUN_E4B_QAT_RESIDENT_SOAK:-off}"
run_e4b_qat_resident_backpressure="${RUN_E4B_QAT_RESIDENT_BACKPRESSURE:-off}"
run_e4b_q4k_baseline="${RUN_E4B_Q4K_BASELINE:-auto}"
run_e4b_q4k_resident_baseline="${RUN_E4B_Q4K_RESIDENT_BASELINE:-auto}"
run_cuda_env="${RUN_CUDA_ENV:-auto}"
run_e4b_qat_provider_comparison="${RUN_E4B_QAT_PROVIDER_COMPARISON:-off}"
run_e4b_qat_provider_benchmark="${RUN_E4B_QAT_PROVIDER_BENCHMARK:-off}"
run_e4b_qat_competitive_floor="${RUN_E4B_QAT_COMPETITIVE_FLOOR:-off}"
e4b_qat_competitive_floors="${E4B_QAT_COMPETITIVE_FLOORS:-compressed_kv_decode_tok_s=36.0}"
e4b_qat_provider_baseline_json="${E4B_QAT_PROVIDER_BASELINE_JSON:-}"
e4b_qat_provider_baseline_inline="${E4B_QAT_PROVIDER_BASELINE_INLINE:-}"
e4b_qat_require_provider_metadata="${E4B_QAT_REQUIRE_PROVIDER_METADATA:-1}"
e4b_qat_tokens="${E4B_QAT_TOKENS:-512}"
e4b_qat_repeats="${E4B_QAT_REPEATS:-${RUN_E4B_QAT_REPEATS:-2}}"
e4b_qat_long_tokens="${E4B_QAT_LONG_TOKENS:-1024}"
e4b_qat_long_min_tokens="${E4B_QAT_LONG_MIN_TOKENS:-768}"
e4b_qat_long_kv_budget_mb="${E4B_QAT_LONG_KV_BUDGET_MB:-1024}"
e4b_qat_long_force_kv_capacity="${E4B_QAT_LONG_FORCE_KV_CAPACITY:-2048}"
e4b_qat_long_min_graph_replays="${E4B_QAT_LONG_MIN_GRAPH_REPLAYS:-auto}"
e4b_qat_long_max_launches_per_token="${E4B_QAT_LONG_MAX_LAUNCHES_PER_TOKEN:-22}"
e4b_qat_long_max_download_syncs="${E4B_QAT_LONG_MAX_DOWNLOAD_SYNCS:-4}"
e4b_qat_resident_tokens="${E4B_QAT_RESIDENT_TOKENS:-512}"
e4b_qat_resident_min_completion_tokens="${E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS:-$e4b_qat_resident_tokens}"
e4b_qat_resident_warm_repeats="${E4B_QAT_RESIDENT_WARM_REPEATS:-${RUN_E4B_QAT_RESIDENT_WARM_REPEATS:-2}}"
e4b_qat_resident_prompt="${E4B_QAT_RESIDENT_PROMPT:-Write a detailed technical explanation of how database indexes improve read queries while slowing down writes. Include examples, tradeoffs, tuning advice, and operational caveats.}"
e4b_qat_resident_port="${E4B_QAT_RESIDENT_PORT:-}"
e4b_qat_resident_max_concurrent_requests="${E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS:-}"
e4b_qat_resident_decode_graph_replay="${E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY:-required}"
e4b_qat_resident_min_graph_replays="${E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS:-auto}"
e4b_qat_resident_soak_request_count="${E4B_QAT_RESIDENT_SOAK_REQUESTS:-6}"
e4b_qat_resident_soak_concurrency="${E4B_QAT_RESIDENT_SOAK_CONCURRENCY:-2}"
e4b_qat_resident_soak_tokens="${E4B_QAT_RESIDENT_SOAK_TOKENS:-256}"
e4b_qat_resident_soak_min_completion_tokens="${E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS:-$e4b_qat_resident_soak_tokens}"
e4b_qat_resident_soak_min_graph_replays="${E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS:-auto}"
e4b_qat_resident_backpressure_request_count="${E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS:-4}"
e4b_qat_resident_backpressure_concurrency="${E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY:-4}"
e4b_qat_resident_backpressure_tokens="${E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS:-256}"
e4b_qat_resident_backpressure_min_accepted="${E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED:-1}"
e4b_qat_resident_backpressure_min_rejected="${E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED:-1}"
e4b_qat_resident_backpressure_max_reject_ms="${E4B_QAT_RESIDENT_BACKPRESSURE_MAX_REJECT_MS:-2000}"
e4b_qat_resident_backpressure_min_graph_replays="${E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS:-auto}"
e4b_q4k_resident_decode_graph_replay="${E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY:-required}"
e4b_qat_require_fused="${E4B_QAT_REQUIRE_FUSED:-1}"
e4b_qat_q4_0_gated_down_tile8="${E4B_QAT_Q4_0_GATED_DOWN_TILE8:-0}"
e4b_qat_require_gated_down_tile8="${E4B_QAT_REQUIRE_GATED_DOWN_TILE8:-$e4b_qat_q4_0_gated_down_tile8}"
e4b_qat_q4_0_ple_gate_fusion="${E4B_QAT_Q4_0_PLE_GATE_FUSION:-1}"
e4b_qat_ple_rms_embed_fusion="${E4B_QAT_PLE_RMS_EMBED_FUSION:-0}"
e4b_qat_require_fast_gqa="${E4B_QAT_REQUIRE_FAST_GQA:-1}"
e4b_qat_decode_graph_replay="${E4B_QAT_DECODE_GRAPH_REPLAY:-required}"
e4b_qat_require_graph_replay="${E4B_QAT_REQUIRE_GRAPH_REPLAY:-1}"
e4b_qat_require_device_token_handoff="${E4B_QAT_REQUIRE_DEVICE_TOKEN_HANDOFF:-1}"
e4b_qat_require_raw_token_export="${E4B_QAT_REQUIRE_RAW_TOKEN_EXPORT:-1}"
e4b_qat_pending_token_readback="${E4B_QAT_PENDING_TOKEN_READBACK:-1}"
e4b_qat_max_download_syncs="${E4B_QAT_MAX_DOWNLOAD_SYNCS:-4}"
e4b_qat_require_ple_fusion="${E4B_QAT_REQUIRE_PLE_FUSION:-1}"
e4b_qat_min_graph_replays="${E4B_QAT_MIN_GRAPH_REPLAYS:-auto}"
e4b_qat_max_launches_per_token="${E4B_QAT_MAX_LAUNCHES_PER_TOKEN:-22.5}"
e4b_qat_temp_slot_period="${E4B_QAT_TEMP_SLOT_PERIOD:-0}"
e4b_qat_capture_allow_unpinned="${E4B_QAT_CAPTURE_ALLOW_UNPINNED:-1}"
e4b_qat_capture_min_alloc_seq="${E4B_QAT_CAPTURE_MIN_ALLOC_SEQ:-10000}"
e4b_qat_force_kv_capacity="${E4B_QAT_FORCE_KV_CAPACITY:-1024}"
min_e4b_qat_tok_s="${MIN_E4B_QAT_TOK_S:-24.0}"
min_e4b_qat_run_tok_s="${MIN_E4B_QAT_RUN_TOK_S:-24.0}"
min_e4b_qat_long_tok_s="${MIN_E4B_QAT_LONG_TOK_S:-15.0}"
min_e4b_qat_resident_warm_tok_s="${MIN_E4B_QAT_RESIDENT_WARM_TOK_S:-24.0}"
min_e4b_qat_resident_soak_agg_tok_s="${MIN_E4B_QAT_RESIDENT_SOAK_AGG_TOK_S:-14.0}"
min_e4b_qat_resident_soak_request_tok_s="${MIN_E4B_QAT_RESIDENT_SOAK_REQUEST_TOK_S:-8.0}"
e4b_qat_resident_soak_max_p95_e2e_ms="${E4B_QAT_RESIDENT_SOAK_MAX_P95_E2E_MS:-35000}"
min_e4b_qat_over_q4k_ratio="${MIN_E4B_QAT_OVER_Q4K_RATIO:-1.25}"
min_e4b_qat_resident_over_q4k_ratio="${MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO:-1.25}"
min_e4b_qat_provider_ratio="${MIN_E4B_QAT_PROVIDER_RATIO:-1.0}"
e4b_qat_provider_base_url="${E4B_QAT_PROVIDER_BASE_URL:-}"
e4b_qat_provider_api_key_env="${E4B_QAT_PROVIDER_API_KEY_ENV:-PROVIDER_API_KEY}"
e4b_qat_provider_endpoint="${E4B_QAT_PROVIDER_ENDPOINT:-chat}"
e4b_qat_provider_no_auth="${E4B_QAT_PROVIDER_NO_AUTH:-0}"
e4b_qat_provider_header="${E4B_QAT_PROVIDER_HEADER:-}"
e4b_qat_provider_name="${E4B_QAT_PROVIDER_NAME:-}"
e4b_qat_provider_model="${E4B_QAT_PROVIDER_MODEL:-google/gemma-4-E4B-it-qat-q4_0-gguf}"
e4b_qat_provider_hardware="${E4B_QAT_PROVIDER_HARDWARE:-}"
e4b_qat_provider_source_url="${E4B_QAT_PROVIDER_SOURCE_URL:-}"
e4b_qat_provider_metric="${E4B_QAT_PROVIDER_METRIC:-resident_e2e_tok_s}"
e4b_qat_provider_stream="${E4B_QAT_PROVIDER_STREAM:-0}"
e4b_qat_provider_stream_include_usage="${E4B_QAT_PROVIDER_STREAM_INCLUDE_USAGE:-1}"
e4b_qat_provider_rate_source="${E4B_QAT_PROVIDER_RATE_SOURCE:-e2e}"
e4b_qat_provider_workload="${E4B_QAT_PROVIDER_WORKLOAD:-antfly-resident-index-explanation-512}"
e4b_qat_provider_tokens="${E4B_QAT_PROVIDER_TOKENS:-$e4b_qat_resident_tokens}"
e4b_qat_provider_min_completion_tokens="${E4B_QAT_PROVIDER_MIN_COMPLETION_TOKENS:-$e4b_qat_resident_min_completion_tokens}"
e4b_qat_provider_repeats="${E4B_QAT_PROVIDER_REPEATS:-2}"
e4b_qat_provider_warmup="${E4B_QAT_PROVIDER_WARMUP:-1}"
e4b_qat_provider_baseline_stat="${E4B_QAT_PROVIDER_BASELINE_STAT:-avg}"
e4b_qat_provider_baseline_stats="${E4B_QAT_PROVIDER_BASELINE_STATS:-avg,median,min}"
e4b_qat_provider_max_token_field="${E4B_QAT_PROVIDER_MAX_TOKEN_FIELD:-max_completion_tokens}"
e4b_qat_provider_notes="${E4B_QAT_PROVIDER_NOTES:-gate-collected provider baseline}"
e4b_qat_prompt="${E4B_QAT_PROMPT:-Write one sentence about ants.}"
e4b_qat_host_budget_mb="${E4B_QAT_HOST_BUDGET_MB:-8000}"
e4b_qat_combined_budget_mb="${E4B_QAT_COMBINED_BUDGET_MB:-18000}"
e4b_qat_backend_budget_mb="${E4B_QAT_BACKEND_BUDGET_MB:-12000}"
e4b_qat_kv_budget_mb="${E4B_QAT_KV_BUDGET_MB:-512}"
e4b_qat_scratch_budget_mb="${E4B_QAT_SCRATCH_BUDGET_MB:-1024}"
e4b_q4k_host_budget_mb="${E4B_Q4K_HOST_BUDGET_MB:-$e4b_qat_host_budget_mb}"
e4b_q4k_combined_budget_mb="${E4B_Q4K_COMBINED_BUDGET_MB:-$e4b_qat_combined_budget_mb}"
e4b_q4k_backend_budget_mb="${E4B_Q4K_BACKEND_BUDGET_MB:-$e4b_qat_backend_budget_mb}"
e4b_q4k_kv_budget_mb="${E4B_Q4K_KV_BUDGET_MB:-$e4b_qat_kv_budget_mb}"
e4b_q4k_scratch_budget_mb="${E4B_Q4K_SCRATCH_BUDGET_MB:-$e4b_qat_scratch_budget_mb}"
gemma12b_q4_model="${GEMMA12B_Q4_MODEL:-$repo_root/.models/google/gemma-4-12B-it-q4_k}"
mtp_target_model="${MTP_TARGET_MODEL:-$e2b_model}"
mtp_draft_model="${MTP_DRAFT_MODEL:-$repo_root/.models/unsloth/gemma-4-E2B-it-qat-GGUF/MTP/gemma-4-E2B-it-Q8_0-MTP.gguf}"
mtp_prompt="${MTP_PROMPT:-Explain why database indexes improve reads but slow down writes.}"
mtp_tokens="${MTP_TOKENS:-128}"
mtp_speculative_k="${MTP_SPECULATIVE_K:-2}"
mtp_cache_dtype="${MTP_CACHE_DTYPE:-f32}"
mtp_turboquant_min_tokens="${MTP_TURBOQUANT_MIN_TOKENS:-0}"
mtp_min_active_speed_ratio="${MTP_MIN_ACTIVE_SPEED_RATIO:-1.0}"
mtp_target_replay="${MTP_TARGET_REPLAY:-auto}"
mtp_replay_context_key="${MTP_REPLAY_CONTEXT_KEY:-auto}"
mtp_unsafe_target_replay="${MTP_UNSAFE_TARGET_REPLAY:-auto}"
mtp_assistant_replay="${MTP_ASSISTANT_REPLAY:-auto}"
mtp_materialize_replay="${MTP_MATERIALIZE_REPLAY:-auto}"
mtp_capture_persistent_replay="${CUDA_CAPTURE_PERSISTENT_REPLAY:-auto}"
mtp_temp_slot_period="${CUDA_TEMP_SLOT_PERIOD:-}"
mtp_temp_slot_skip="${CUDA_TEMP_SLOT_SKIP:-}"
mtp_position_mode="${MTP_POSITION_MODE:-}"
mtp_target_hidden_source="${MTP_TARGET_HIDDEN_SOURCE:-}"
mtp_concat_order="${MTP_CONCAT_ORDER:-}"
mtp_kv_donor_mode="${MTP_KV_DONOR_MODE:-}"
run_mtp="${RUN_MTP:-auto}"
run_resident="${RUN_RESIDENT:-auto}"
resident_model="${RESIDENT_MODEL:-$e2b_model}"
resident_prompt="${RESIDENT_PROMPT:-Write one sentence about ants.}"
resident_tokens="${RESIDENT_TOKENS:-32}"
resident_cache_dtype="${RESIDENT_CACHE_DTYPE:-f32}"
resident_models_dir="${RESIDENT_MODELS_DIR:-$repo_root/.models}"
resident_host="${RESIDENT_HOST:-127.0.0.1}"
resident_port="${RESIDENT_PORT:-}"
resident_max_warm_cold_ratio="${MAX_RESIDENT_WARM_COLD_RATIO:-0.75}"
resident_min_warm_tok_s="${MIN_RESIDENT_WARM_TOK_S:-10.0}"
require_speed_thresholds="${REQUIRE_SPEED_THRESHOLDS:-}"
if [ -z "$require_speed_thresholds" ]; then
  if [ "$mode" = "full" ]; then
    require_speed_thresholds=1
  else
    require_speed_thresholds=0
  fi
fi

summary="$out_dir/summary.txt"
: > "$summary"
e4b_qat_provider_generated_baseline_json=""

log() {
  printf '%s\n' "$*" | tee -a "$summary"
}

capture_cuda_environment() {
  case "$run_cuda_env" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_CUDA_ENV=$run_cuda_env; expected auto|required|off"
      exit 2
      ;;
  esac

  local log_path="$out_dir/cuda_smoke.log"
  local json_path="$out_dir/cuda_environment.json"
  local parser="$inference_dir/scripts/cuda_environment_from_smoke.py"

  if [ "$run_cuda_env" = "off" ]; then
    if [ -e "$parser" ]; then
      python3 "$parser" --log "$log_path" --out "$json_path" --status skip --detail RUN_CUDA_ENV=off >/dev/null
    fi
    log "SKIP cuda_environment: RUN_CUDA_ENV=off"
    return 0
  fi
  if [ ! -e "$parser" ]; then
    log "missing cuda environment parser: $parser"
    exit 1
  fi

  log "RUN cuda_environment: $antfly_bin cuda-info --smoke"
  if "$antfly_bin" cuda-info --smoke >"$log_path" 2>&1; then
    python3 "$parser" --log "$log_path" --out "$json_path" --status ok --detail "$log_path"
    local detail
    detail="$(python3 - "$json_path" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
device = data.get("device_name") or "unknown"
sm = data.get("compute_capability") or "unknown"
driver = data.get("driver_version") or "unknown"
artifacts = data.get("artifacts") or "unknown"
smoke_count = len(data.get("smoke") or {})
print(f"device={device} sm={sm} driver={driver} artifacts={artifacts} smoke_checks={smoke_count}")
PY
)"
    log "PASS cuda_environment: $detail"
    return 0
  fi

  local status=$?
  local detail="cuda-info --smoke exit_$status"
  if [ "$run_cuda_env" = "required" ]; then
    python3 "$parser" --log "$log_path" --out "$json_path" --status fail --detail "$detail" >/dev/null
    log "FAIL cuda_environment: $detail"
    exit "$status"
  fi
  python3 "$parser" --log "$log_path" --out "$json_path" --status skip --detail "$detail" >/dev/null
  log "SKIP cuda_environment: $detail"
}

run_e4b_qat_provider_benchmark() {
  case "$run_e4b_qat_provider_benchmark" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_PROVIDER_BENCHMARK=$run_e4b_qat_provider_benchmark; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_qat_provider_benchmark" = "off" ]; then
    log "SKIP e4b_qat_provider_benchmark: RUN_E4B_QAT_PROVIDER_BENCHMARK=off"
    return 0
  fi
  if [ -z "$e4b_qat_provider_base_url" ]; then
    if [ "$run_e4b_qat_provider_benchmark" = "required" ]; then
      log "FAIL e4b_qat_provider_benchmark: missing E4B_QAT_PROVIDER_BASE_URL"
      exit 1
    fi
    log "SKIP e4b_qat_provider_benchmark: missing E4B_QAT_PROVIDER_BASE_URL"
    return 0
  fi
  local missing=()
  [ -n "$e4b_qat_provider_name" ] || missing+=(E4B_QAT_PROVIDER_NAME)
  [ -n "$e4b_qat_provider_hardware" ] || missing+=(E4B_QAT_PROVIDER_HARDWARE)
  [ -n "$e4b_qat_provider_source_url" ] || missing+=(E4B_QAT_PROVIDER_SOURCE_URL)
  if [ "${#missing[@]}" -ne 0 ]; then
    log "FAIL e4b_qat_provider_benchmark: missing ${missing[*]}"
    exit 1
  fi
  local benchmarker="$inference_dir/scripts/gemma4_qat_provider_benchmark.py"
  if [ ! -e "$benchmarker" ]; then
    log "FAIL e4b_qat_provider_benchmark: missing $benchmarker"
    exit 1
  fi
  local summarizer="$inference_dir/scripts/gemma4_qat_production_summary.py"
  if [ ! -e "$summarizer" ]; then
    log "FAIL e4b_qat_provider_benchmark: missing $summarizer"
    exit 1
  fi
  local json_path="$out_dir/e4b_qat_provider_baselines.json"
  local rows_path="$out_dir/e4b_qat_provider_baselines.tsv"
  local log_path="$out_dir/e4b_qat_provider_benchmark.log"
  local validation_path="$out_dir/e4b_qat_provider_baseline_validation.json"
  local validation_log="$out_dir/e4b_qat_provider_baseline_validation.log"
  local -a benchmark_args=(
    --base-url "$e4b_qat_provider_base_url"
    --endpoint "$e4b_qat_provider_endpoint"
    --api-key-env "$e4b_qat_provider_api_key_env"
    --provider "$e4b_qat_provider_name"
    --model "$e4b_qat_provider_model"
    --hardware "$e4b_qat_provider_hardware"
    --source-url "$e4b_qat_provider_source_url"
    --metric "$e4b_qat_provider_metric"
    --rate-source "$e4b_qat_provider_rate_source"
    --workload "$e4b_qat_provider_workload"
    --prompt "$e4b_qat_resident_prompt"
    --tokens "$e4b_qat_provider_tokens"
    --min-completion-tokens "$e4b_qat_provider_min_completion_tokens"
    --repeats "$e4b_qat_provider_repeats"
    --warmup "$e4b_qat_provider_warmup"
    --baseline-stat "$e4b_qat_provider_baseline_stat"
    --baseline-stats "$e4b_qat_provider_baseline_stats"
    --max-token-field "$e4b_qat_provider_max_token_field"
    --min-ratio "$min_e4b_qat_provider_ratio"
    --notes "$e4b_qat_provider_notes"
    --output "$json_path"
    --rows-tsv "$rows_path"
  )
  case "$e4b_qat_provider_no_auth" in
    1|true|True|on|ON|yes|YES)
      benchmark_args+=(--no-auth)
      ;;
  esac
  case "$e4b_qat_provider_stream" in
    1|true|True|on|ON|yes|YES)
      benchmark_args+=(--stream)
      ;;
  esac
  case "$e4b_qat_provider_stream_include_usage" in
    0|false|False|off|OFF|no|NO)
      benchmark_args+=(--no-stream-include-usage)
      ;;
  esac
  if [ -n "$e4b_qat_provider_header" ]; then
    benchmark_args+=(--header "$e4b_qat_provider_header")
  fi
  log "RUN e4b_qat_provider_benchmark: provider=$e4b_qat_provider_name model=$e4b_qat_provider_model tokens=$e4b_qat_provider_tokens"
  if python3 "$benchmarker" "${benchmark_args[@]}" >"$log_path" 2>&1; then
    local detail
    detail="$(sed -n '1p' "$log_path" 2>/dev/null || true)"
    log "PASS e4b_qat_provider_benchmark: ${detail:-$json_path}"
    if python3 "$summarizer" \
      --validate-provider-baselines-only \
      --provider-baseline "$json_path" \
      --output "$validation_path" >"$validation_log" 2>&1; then
      e4b_qat_provider_generated_baseline_json="$json_path"
      detail="$(sed -n '1p' "$validation_log" 2>/dev/null || true)"
      log "PASS e4b_qat_provider_baseline_validation: ${detail:-$validation_path}"
      return 0
    fi
    detail="$(sed -n '1p' "$validation_log" 2>/dev/null || true)"
    log "FAIL e4b_qat_provider_baseline_validation: ${detail:-$validation_log}"
    exit 1
  fi
  local detail
  detail="$(sed -n '1p' "$log_path" 2>/dev/null || true)"
  log "FAIL e4b_qat_provider_benchmark: ${detail:-$log_path}"
  exit 1
}

write_qat_production_summary() {
  local json_path="$out_dir/e4b_qat_production_summary.json"
  local summarizer="$inference_dir/scripts/gemma4_qat_production_summary.py"
  if [ ! -e "$summarizer" ]; then
    log "missing QAT production summary parser: $summarizer"
    exit 1
  fi
  case "$run_e4b_qat_provider_comparison" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_PROVIDER_COMPARISON=$run_e4b_qat_provider_comparison; expected auto|required|off"
      exit 2
      ;;
  esac
  case "$run_e4b_qat_competitive_floor" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_COMPETITIVE_FLOOR=$run_e4b_qat_competitive_floor; expected auto|required|off"
      exit 2
      ;;
  esac
  local -a summary_args=(
    --out-dir "$out_dir"
    --output "$json_path"
    --min-provider-ratio "$min_e4b_qat_provider_ratio"
    --min-target-qat-tok-s "$min_e4b_qat_tok_s"
    --min-target-qat-run-tok-s "$min_e4b_qat_run_tok_s"
    --min-target-qat-over-q4k-ratio "$min_e4b_qat_over_q4k_ratio"
    --min-target-qat-tokens "$e4b_qat_tokens"
    --min-target-qat-repeats "$e4b_qat_repeats"
    --min-long-qat-tok-s "$min_e4b_qat_long_tok_s"
    --min-long-qat-tokens "$e4b_qat_long_min_tokens"
    --min-resident-qat-tok-s "$min_e4b_qat_resident_warm_tok_s"
    --min-resident-qat-over-q4k-ratio "$min_e4b_qat_resident_over_q4k_ratio"
    --min-resident-qat-tokens "$e4b_qat_resident_min_completion_tokens"
    --min-resident-qat-repeats "$e4b_qat_resident_warm_repeats"
    --min-soak-requests "$e4b_qat_resident_soak_request_count"
    --min-soak-aggregate-tok-s "$min_e4b_qat_resident_soak_agg_tok_s"
    --min-soak-request-tok-s "$min_e4b_qat_resident_soak_request_tok_s"
    --max-soak-p95-e2e-ms "$e4b_qat_resident_soak_max_p95_e2e_ms"
    --min-backpressure-accepted "$e4b_qat_resident_backpressure_min_accepted"
    --min-backpressure-rejected "$e4b_qat_resident_backpressure_min_rejected"
    --max-backpressure-reject-ms "$e4b_qat_resident_backpressure_max_reject_ms"
    --min-mtp-active-speed-ratio "$mtp_min_active_speed_ratio"
  )
  if [ "$run_cuda_env" = "required" ]; then
    summary_args+=(--require-cuda-environment)
  fi
  if [ "$run_e4b_qat" != "off" ] && [ "$run_e4b_q4k_baseline" != "off" ] && [ "$mode" != "mtp-only" ] && [ -e "$e4b_qat_model" ] && [ -e "$e4b_q4k_baseline_model" ]; then
    summary_args+=(--require-target)
  fi
  if [ "$run_e4b_qat_long" != "off" ] && [ -e "$e4b_qat_model" ]; then
    summary_args+=(--require-long)
  fi
  if [ "$run_e4b_qat_resident" != "off" ] && [ -e "$e4b_qat_model" ]; then
    summary_args+=(--require-resident)
    if [ "$run_e4b_q4k_resident_baseline" != "off" ] && [ -e "$e4b_q4k_baseline_model" ]; then
      summary_args+=(--require-resident-q4k)
    fi
    if [ "$run_e4b_qat_resident_soak" != "off" ]; then
      summary_args+=(--require-soak)
    fi
    if [ "$run_e4b_qat_resident_backpressure" != "off" ]; then
      summary_args+=(--require-backpressure)
    fi
  fi
  if [ "$run_mtp" != "off" ] || [ "$mode" = "mtp-only" ]; then
    summary_args+=(--require-mtp)
  fi
  if [ "$run_e4b_qat_provider_comparison" != "off" ] || [ -n "$e4b_qat_provider_generated_baseline_json" ] || [ "$run_e4b_qat_provider_benchmark" = "required" ]; then
    if [ -n "$e4b_qat_provider_baseline_json" ]; then
      summary_args+=(--provider-baseline "$e4b_qat_provider_baseline_json")
    fi
    if [ -n "$e4b_qat_provider_generated_baseline_json" ]; then
      summary_args+=(--provider-baseline "$e4b_qat_provider_generated_baseline_json")
    fi
    if [ -n "$e4b_qat_provider_baseline_inline" ]; then
      summary_args+=(--provider-baseline-json "$e4b_qat_provider_baseline_inline")
    fi
    case "$e4b_qat_require_provider_metadata" in
      0|false|False|off|OFF|no|NO)
        ;;
      *)
        summary_args+=(--require-provider-metadata)
        ;;
    esac
    if [ "$run_e4b_qat_provider_comparison" = "required" ] || [ "$run_e4b_qat_provider_benchmark" = "required" ] || [ -n "$e4b_qat_provider_baseline_json" ] || [ -n "$e4b_qat_provider_generated_baseline_json" ] || [ -n "$e4b_qat_provider_baseline_inline" ]; then
      summary_args+=(--require-provider-comparison)
    fi
  fi
  if [ "$run_e4b_qat_competitive_floor" != "off" ]; then
    for floor in $e4b_qat_competitive_floors; do
      summary_args+=(--competitive-floor "$floor")
    done
    if [ "$run_e4b_qat_competitive_floor" = "required" ]; then
      summary_args+=(--require-competitive-floor)
    fi
  fi
  python3 "$summarizer" "${summary_args[@]}" | tee -a "$summary"
}

require_path() {
  local label="$1"
  local path="$2"
  if [ ! -e "$path" ]; then
    log "missing $label: $path"
    exit 1
  fi
}

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
}

choose_port() {
  python3 - <<'PY'
import socket

s = socket.socket()
s.bind(("127.0.0.1", 0))
try:
    print(s.getsockname()[1])
finally:
    s.close()
PY
}

resident_server_pid=""
cleanup_resident_server() {
  if [ -n "${resident_server_pid:-}" ]; then
    kill "$resident_server_pid" >/dev/null 2>&1 || true
    wait "$resident_server_pid" >/dev/null 2>&1 || true
    resident_server_pid=""
  fi
}
trap cleanup_resident_server EXIT

run_generate_json() {
  local name="$1"
  shift
  local log_file="$out_dir/$name.log"
  local json_file="$out_dir/$name.json"
  local -a args=("$@")
  rm -f "$json_file"
  log "RUN $name: ${args[*]}"
  "${args[@]}" --json-timing "$json_file" 2>&1 | tee "$log_file"
  last_json_file="$json_file"
}

write_json_table_and_check_speed() {
  local json_dir="$1"
  local require_speed="$2"
  local min_e2b="${MIN_E2B_TEXT_TOK_S:-15.0}"
  local min_12b="${MIN_12B_Q4_TOK_S:-8.0}"
  local min_e4b_qat="${MIN_E4B_QAT_TOK_S:-24.0}"
  local min_e4b_qat_run="${MIN_E4B_QAT_RUN_TOK_S:-24.0}"
  python3 - "$json_dir" "$out_dir/gemma4_cuda_timings.tsv" "$require_speed" "$min_e2b" "$min_12b" "$min_e4b_qat" "$min_e4b_qat_run" "$e4b_qat_tokens" "$e4b_qat_repeats" "$e4b_qat_require_fused" "$e4b_qat_require_fast_gqa" "$e4b_qat_require_graph_replay" "$e4b_qat_require_device_token_handoff" "$e4b_qat_min_graph_replays" "$e4b_qat_max_launches_per_token" "$e4b_qat_require_gated_down_tile8" "$e4b_qat_require_raw_token_export" "$e4b_qat_max_download_syncs" "$e4b_qat_require_ple_fusion" <<'PY'
import glob
import json
import os
import sys

json_dir, out_path, require_speed, min_e2b, min_12b, min_e4b_qat, min_e4b_qat_run, e4b_qat_tokens, e4b_qat_repeats, e4b_qat_require_fused, e4b_qat_require_fast_gqa, e4b_qat_require_graph_replay, e4b_qat_require_device_token_handoff, e4b_qat_min_graph_replays, e4b_qat_max_launches_per_token, e4b_qat_require_gated_down_tile8, e4b_qat_require_raw_token_export, e4b_qat_max_download_syncs, e4b_qat_require_ple_fusion = sys.argv[1:20]
require_speed = require_speed == "1"
min_e2b = float(min_e2b)
min_12b = float(min_12b)
min_e4b_qat = float(min_e4b_qat)
min_e4b_qat_run = float(min_e4b_qat_run)
e4b_qat_tokens = int(float(e4b_qat_tokens))
e4b_qat_repeats = int(float(e4b_qat_repeats))
e4b_qat_require_fused = e4b_qat_require_fused.lower() not in {"0", "false", "off", "no"}
e4b_qat_require_fast_gqa = e4b_qat_require_fast_gqa.lower() not in {"0", "false", "off", "no"}
e4b_qat_require_graph_replay = e4b_qat_require_graph_replay.lower() not in {"0", "false", "off", "no"}
e4b_qat_require_device_token_handoff = e4b_qat_require_device_token_handoff.lower() not in {"0", "false", "off", "no"}
e4b_qat_require_gated_down_tile8 = e4b_qat_require_gated_down_tile8.lower() not in {"0", "false", "off", "no"}
e4b_qat_require_raw_token_export = e4b_qat_require_raw_token_export.lower() not in {"0", "false", "off", "no"}
e4b_qat_max_download_syncs = None if e4b_qat_max_download_syncs.lower() in {"", "off", "none", "no"} else int(float(e4b_qat_max_download_syncs))
e4b_qat_require_ple_fusion = e4b_qat_require_ple_fusion.lower() not in {"0", "false", "off", "no"}
if e4b_qat_min_graph_replays.lower() in {"", "auto"}:
    e4b_qat_min_graph_replays = max(1, e4b_qat_tokens - 64)
else:
    e4b_qat_min_graph_replays = int(float(e4b_qat_min_graph_replays))
e4b_qat_max_launches_per_token = float(e4b_qat_max_launches_per_token)

rows = []
by_name = {}
for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    by_name[name] = data
    timing = data.get("timing_ms") or {}
    load_ms = timing.get("load_model", 0)
    warm_ttft_ms = (timing.get("prefill_inner", 0) or 0) + (timing.get("runtime_prepare_inner", 0) or 0)
    rows.append((
        name,
        str(data.get("tokens", "")),
        f"{load_ms / 1000.0:.2f}s",
        f"{warm_ttft_ms / 1000.0:.2f}s",
        f"{(load_ms + warm_ttft_ms) / 1000.0:.2f}s",
        f"{float(data.get('decode_tok_per_s', 0.0)):.3f}",
    ))

with open(out_path, "w", encoding="utf-8") as f:
    f.write("case\ttokens\tload\twarm_ttft\tcold_ttft\tdecode_tok_s\n")
    for row in rows:
        f.write("\t".join(row) + "\n")

print(f"timing_table={out_path}")
errors = []
if require_speed:
    e2b = by_name.get("real_bench_128")
    if not e2b:
        errors.append("missing real_bench_128.json for E2B speed gate")
    elif float(e2b.get("decode_tok_per_s", 0.0)) < min_e2b:
        errors.append(f"real_bench_128 decode_tok_per_s={e2b.get('decode_tok_per_s')} < {min_e2b}")

    q4 = by_name.get("gemma12b_q4_f32_32")
    if not q4:
        errors.append("missing gemma12b_q4_f32_32.json for 12B Q4 speed gate")
    elif float(q4.get("decode_tok_per_s", 0.0)) < min_12b:
        errors.append(f"gemma12b_q4_f32_32 decode_tok_per_s={q4.get('decode_tok_per_s')} < {min_12b}")

    e4b_qat_runs = [(name, data) for name, data in by_name.items() if name == "e4b_qat_cuda" or name.startswith("e4b_qat_cuda_run")]
    for name, e4b_qat in e4b_qat_runs:
        if float(e4b_qat.get("decode_tok_per_s", 0.0)) < min_e4b_qat_run:
            errors.append(f"{name} decode_tok_per_s={e4b_qat.get('decode_tok_per_s')} < {min_e4b_qat_run}")

    polar_runs = [data for name, data in by_name.items() if name.startswith("gemma12b_q4_polar4_32_run")]
    if not polar_runs:
        errors.append("missing gemma12b_q4_polar4_32_run*.json for polar4 speed gate")
    else:
        avg = sum(float(data.get("decode_tok_per_s", 0.0)) for data in polar_runs) / len(polar_runs)
        if avg < min_12b:
            errors.append(f"gemma12b_q4_polar4_32 average decode_tok_per_s={avg:.3f} < {min_12b}")

e4b_qat_runs = sorted((name, data) for name, data in by_name.items() if name == "e4b_qat_cuda" or name.startswith("e4b_qat_cuda_run"))
e4b_qat_rates = []
for name, e4b_qat in e4b_qat_runs:
    e4b_qat_rates.append(float(e4b_qat.get("decode_tok_per_s", 0.0)))
    tokens = int(e4b_qat.get("tokens") or 0)
    if tokens < e4b_qat_tokens:
        errors.append(f"{name} tokens={tokens} < {e4b_qat_tokens}")
    cuda = e4b_qat.get("cuda") or {}
    if e4b_qat_max_download_syncs is not None:
        download_syncs = int(cuda.get("download_syncs") or 0)
        if download_syncs > e4b_qat_max_download_syncs:
            errors.append(f"{name} download_syncs={download_syncs} > {e4b_qat_max_download_syncs}")
    if e4b_qat_require_ple_fusion:
        add_mul_fused = int(cuda.get("add_mul_scalar_fused") or 0)
        if add_mul_fused < e4b_qat_tokens:
            errors.append(f"{name} add_mul_scalar_fused={add_mul_fused} < {e4b_qat_tokens}")
    scalar_launches_raw = cuda.get("launch_scalar")
    if scalar_launches_raw is None:
        errors.append(f"{name} launch_scalar missing")
    else:
        scalar_launches = int(scalar_launches_raw)
        if scalar_launches != 0:
            errors.append(f"{name} launch_scalar={scalar_launches} > 0")
    embedding_launches_raw = cuda.get("launch_embedding")
    if embedding_launches_raw is None:
        errors.append(f"{name} launch_embedding missing")
    else:
        embedding_launches = int(embedding_launches_raw)
        if embedding_launches > e4b_qat_tokens + 1:
            errors.append(f"{name} launch_embedding={embedding_launches} > {e4b_qat_tokens + 1}")
    if e4b_qat_require_fused:
        cuda = e4b_qat.get("cuda") or {}
        for field in ("qkv_fused_q4_0", "linear_pair_fused_q4_0", "gated_down_fused_q4_0"):
            value = int(cuda.get(field) or 0)
            if value <= 0:
                errors.append(f"{name} {field}={value}")
        for field in ("qkv_fallback_unsupported", "qkv_kernel_unavailable", "linear_pair_fallbacks", "gated_down_fallbacks"):
            value = int(cuda.get(field) or 0)
            if value != 0:
                errors.append(f"{name} {field}={value}")
        if e4b_qat_require_gated_down_tile8:
            tile8_hits = int(cuda.get("gated_down_fused_q4_0_tile8") or 0)
            if tile8_hits <= 0:
                errors.append(f"{name} gated_down_fused_q4_0_tile8={tile8_hits}")
        q4_0_lm_head = int(cuda.get("lm_head_argmax_fused_q4_0") or 0)
        if q4_0_lm_head != 0:
            errors.append(f"{name} lm_head_argmax_fused_q4_0={q4_0_lm_head} > 0")
        q6_lm_head = int(cuda.get("lm_head_argmax_fused_q6") or 0)
        if q6_lm_head <= 0:
            errors.append(f"{name} lm_head_argmax_fused_q6={q6_lm_head}")
        lm_head_fallbacks = int(cuda.get("lm_head_argmax_fallbacks") or 0)
        if lm_head_fallbacks != 0:
            errors.append(f"{name} lm_head_argmax_fallbacks={lm_head_fallbacks}")
    if e4b_qat_require_fast_gqa:
        cuda = e4b_qat.get("cuda") or {}
        fast_hits = int(cuda.get("launch_attention_gqa_decode_fast") or 0)
        if fast_hits <= 0:
            errors.append(f"{name} launch_attention_gqa_decode_fast={fast_hits}")
        fast_fallbacks = int(cuda.get("launch_attention_gqa_decode_fast_fallbacks") or 0)
        if fast_fallbacks != 0:
            errors.append(f"{name} launch_attention_gqa_decode_fast_fallbacks={fast_fallbacks}")
    if e4b_qat_require_raw_token_export:
        cuda_generate = e4b_qat.get("cuda_generate") or {}
        to_float32_calls = cuda_generate.get("to_float32_calls")
        to_float32_bytes = cuda_generate.get("to_float32_bytes")
        if to_float32_calls is None or int(to_float32_calls) != 0:
            errors.append(f"{name} cuda_generate.to_float32_calls={to_float32_calls}")
        if to_float32_bytes is None or int(to_float32_bytes) != 0:
            errors.append(f"{name} cuda_generate.to_float32_bytes={to_float32_bytes}")
    if e4b_qat_require_graph_replay:
        cuda = e4b_qat.get("cuda") or {}
        graph_replays = int(cuda.get("graph_capture_persistent_replays") or 0)
        if graph_replays < e4b_qat_min_graph_replays:
            errors.append(f"{name} graph_capture_persistent_replays={graph_replays} < {e4b_qat_min_graph_replays}")
        launches_per_token = float(cuda.get("launches_per_token") or 0.0)
        if launches_per_token <= 0 or launches_per_token > e4b_qat_max_launches_per_token:
            errors.append(f"{name} launches_per_token={launches_per_token} > {e4b_qat_max_launches_per_token}")
    if e4b_qat_require_device_token_handoff:
        generation_runtime = e4b_qat.get("generation_decoder_runtime") or {}
        min_handoffs = max(0, e4b_qat_tokens - 1)
        handoff_attempts = int(generation_runtime.get("device_token_handoff_attempts") or 0)
        handoff_hits = int(generation_runtime.get("device_token_handoff_hits") or 0)
        handoff_fallbacks = int(generation_runtime.get("device_token_handoff_fallbacks") or 0)
        handoff_seeds = int(generation_runtime.get("device_token_handoff_seeds") or 0)
        if handoff_attempts < min_handoffs:
            errors.append(f"{name} device_token_handoff_attempts={handoff_attempts} < {min_handoffs}")
        if handoff_hits < min_handoffs:
            errors.append(f"{name} device_token_handoff_hits={handoff_hits} < {min_handoffs}")
        if handoff_fallbacks != 0:
            errors.append(f"{name} device_token_handoff_fallbacks={handoff_fallbacks}")
        if handoff_seeds < 1:
            errors.append(f"{name} device_token_handoff_seeds={handoff_seeds}")

if e4b_qat_rates:
    if len(e4b_qat_rates) < e4b_qat_repeats:
        errors.append(f"e4b_qat_cuda repeat_count={len(e4b_qat_rates)} < {e4b_qat_repeats}")
    min_rate = min(e4b_qat_rates)
    avg_rate = sum(e4b_qat_rates) / len(e4b_qat_rates)
    if min_rate < min_e4b_qat_run:
        errors.append(f"e4b_qat_cuda repeat_min_decode_tok_per_s={min_rate:.3f} < {min_e4b_qat_run}")
    if avg_rate < min_e4b_qat:
        errors.append(f"e4b_qat_cuda repeat_avg_decode_tok_per_s={avg_rate:.3f} < {min_e4b_qat}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(1)
PY
}

run_e4b_qat_gate_if_enabled() {
  case "$run_e4b_qat" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT=$run_e4b_qat; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_qat" = "off" ] || [ "$mode" = "mtp-only" ]; then
    return 0
  fi
  if [ ! -e "$e4b_qat_model" ]; then
    if [ "$run_e4b_qat" = "required" ]; then
      require_path "E4B QAT model" "$e4b_qat_model"
    fi
    log "SKIP e4b_qat_cuda: model missing: $e4b_qat_model"
    return 0
  fi
  case "$e4b_qat_repeats" in
    ''|*[!0-9]*)
      log "invalid E4B_QAT_REPEATS=$e4b_qat_repeats; expected positive integer"
      exit 2
      ;;
  esac
  if [ "$e4b_qat_repeats" -lt 1 ]; then
    log "invalid E4B_QAT_REPEATS=$e4b_qat_repeats; expected positive integer"
    exit 2
  fi

  local repeat=1
  while [ "$repeat" -le "$e4b_qat_repeats" ]; do
    local label="e4b_qat_cuda"
    if [ "$e4b_qat_repeats" -gt 1 ]; then
      label="e4b_qat_cuda_run${repeat}"
    fi
    run_e4b_qat_gate_once "$label"
    repeat=$((repeat + 1))
  done
}

run_e4b_qat_gate_once() {
  local label="$1"
  run_generate_json "$label" env \
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$e4b_qat_decode_graph_replay" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$e4b_qat_temp_slot_period" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$e4b_qat_capture_allow_unpinned" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$e4b_qat_capture_min_alloc_seq" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$e4b_qat_force_kv_capacity" \
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1 \
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$e4b_qat_q4_0_gated_down_tile8" \
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$e4b_qat_q4_0_ple_gate_fusion" \
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$e4b_qat_ple_rms_embed_fusion" \
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$e4b_qat_pending_token_readback" \
    "$antfly_bin" generate "$e4b_qat_model" "$e4b_qat_prompt" \
    --backend cuda \
    --cache-dtype f32 \
    --max-tokens "$e4b_qat_tokens" \
    --temperature 0 \
    --raw-prompt \
    --no-chat-template \
    --print-token-count \
    --print-token-ids \
    --print-timing \
    --host-budget-mb "$e4b_qat_host_budget_mb" \
    --combined-budget-mb "$e4b_qat_combined_budget_mb" \
    --backend-budget-mb "$e4b_qat_backend_budget_mb" \
    --kv-budget-mb "$e4b_qat_kv_budget_mb" \
    --scratch-budget-mb "$e4b_qat_scratch_budget_mb"
}

run_e4b_qat_long_gate_if_enabled() {
  case "$run_e4b_qat_long" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_LONG=$run_e4b_qat_long; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_qat_long" = "off" ] || [ "$mode" = "mtp-only" ]; then
    return 0
  fi
  if [ ! -e "$e4b_qat_model" ]; then
    if [ "$run_e4b_qat_long" = "required" ]; then
      require_path "E4B QAT model" "$e4b_qat_model"
    fi
    log "SKIP e4b_qat_cuda_long: model missing: $e4b_qat_model"
    return 0
  fi
  case "$e4b_qat_long_tokens" in
    ''|*[!0-9]*)
      log "invalid E4B_QAT_LONG_TOKENS=$e4b_qat_long_tokens; expected positive integer"
      exit 2
      ;;
  esac
  case "$e4b_qat_long_min_tokens" in
    ''|*[!0-9]*)
      log "invalid E4B_QAT_LONG_MIN_TOKENS=$e4b_qat_long_min_tokens; expected positive integer"
      exit 2
      ;;
  esac
  if [ "$e4b_qat_long_tokens" -lt 1 ]; then
    log "invalid E4B_QAT_LONG_TOKENS=$e4b_qat_long_tokens; expected positive integer"
    exit 2
  fi
  if [ "$e4b_qat_long_min_tokens" -lt 1 ]; then
    log "invalid E4B_QAT_LONG_MIN_TOKENS=$e4b_qat_long_min_tokens; expected positive integer"
    exit 2
  fi
  if [ "$e4b_qat_long_min_tokens" -gt "$e4b_qat_long_tokens" ]; then
    log "invalid E4B_QAT_LONG_MIN_TOKENS=$e4b_qat_long_min_tokens exceeds E4B_QAT_LONG_TOKENS=$e4b_qat_long_tokens"
    exit 2
  fi

  run_generate_json e4b_qat_cuda_long env \
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$e4b_qat_decode_graph_replay" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$e4b_qat_temp_slot_period" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$e4b_qat_capture_allow_unpinned" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$e4b_qat_capture_min_alloc_seq" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$e4b_qat_long_force_kv_capacity" \
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1 \
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$e4b_qat_q4_0_gated_down_tile8" \
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$e4b_qat_q4_0_ple_gate_fusion" \
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$e4b_qat_ple_rms_embed_fusion" \
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$e4b_qat_pending_token_readback" \
    "$antfly_bin" generate "$e4b_qat_model" "$e4b_qat_prompt" \
    --backend cuda \
    --cache-dtype f32 \
    --max-tokens "$e4b_qat_long_tokens" \
    --temperature 0 \
    --raw-prompt \
    --no-chat-template \
    --print-token-count \
    --print-token-ids \
    --print-timing \
    --host-budget-mb "$e4b_qat_host_budget_mb" \
    --combined-budget-mb "$e4b_qat_combined_budget_mb" \
    --backend-budget-mb "$e4b_qat_backend_budget_mb" \
    --kv-budget-mb "$e4b_qat_long_kv_budget_mb" \
    --scratch-budget-mb "$e4b_qat_scratch_budget_mb"
}

run_e4b_q4k_baseline_if_enabled() {
  case "$run_e4b_q4k_baseline" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_Q4K_BASELINE=$run_e4b_q4k_baseline; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_q4k_baseline" = "off" ] || [ "$mode" = "mtp-only" ]; then
    return 0
  fi
  if [ ! -e "$e4b_q4k_baseline_model" ]; then
    if [ "$run_e4b_q4k_baseline" = "required" ]; then
      require_path "E4B Q4_K baseline model" "$e4b_q4k_baseline_model"
    fi
    log "SKIP e4b_q4k_baseline: model missing: $e4b_q4k_baseline_model"
    return 0
  fi

  run_generate_json e4b_q4k_baseline "$antfly_bin" generate "$e4b_q4k_baseline_model" "$e4b_qat_prompt" \
    --backend cuda \
    --cache-dtype f32 \
    --max-tokens "$e4b_qat_tokens" \
    --temperature 0 \
    --raw-prompt \
    --no-chat-template \
    --print-token-count \
    --print-token-ids \
    --print-timing \
    --host-budget-mb "$e4b_q4k_host_budget_mb" \
    --combined-budget-mb "$e4b_q4k_combined_budget_mb" \
    --backend-budget-mb "$e4b_q4k_backend_budget_mb" \
    --kv-budget-mb "$e4b_q4k_kv_budget_mb" \
    --scratch-budget-mb "$e4b_q4k_scratch_budget_mb"
}

check_e4b_qat_gate_json_if_present() {
  python3 - "$out_dir" "$min_e4b_qat_tok_s" "$min_e4b_qat_run_tok_s" "$min_e4b_qat_over_q4k_ratio" "$e4b_qat_tokens" "$e4b_qat_repeats" "$e4b_qat_require_fused" "$e4b_qat_require_fast_gqa" "$e4b_qat_require_graph_replay" "$e4b_qat_require_device_token_handoff" "$e4b_qat_min_graph_replays" "$e4b_qat_max_launches_per_token" "$e4b_qat_require_gated_down_tile8" "$e4b_qat_require_raw_token_export" "$e4b_qat_max_download_syncs" "$e4b_qat_require_ple_fusion" <<'PY'
import glob
import json
import os
import sys

out_dir, min_tok_s, min_run_tok_s, min_qat_over_q4k_ratio, min_tokens, repeats, require_fused, require_fast_gqa, require_graph, require_device_token_handoff, min_graph_replays, max_launches, require_gated_down_tile8, require_raw_token_export, max_download_syncs, require_ple_fusion = sys.argv[1:17]
min_tok_s = float(min_tok_s)
min_run_tok_s = float(min_run_tok_s)
min_qat_over_q4k_ratio = float(min_qat_over_q4k_ratio)
min_tokens = int(float(min_tokens))
repeats = int(float(repeats))
require_fused = require_fused.lower() not in {"0", "false", "off", "no"}
require_fast_gqa = require_fast_gqa.lower() not in {"0", "false", "off", "no"}
require_graph = require_graph.lower() not in {"0", "false", "off", "no"}
require_device_token_handoff = require_device_token_handoff.lower() not in {"0", "false", "off", "no"}
require_gated_down_tile8 = require_gated_down_tile8.lower() not in {"0", "false", "off", "no"}
require_raw_token_export = require_raw_token_export.lower() not in {"0", "false", "off", "no"}
max_download_syncs = None if max_download_syncs.lower() in {"", "off", "none", "no"} else int(float(max_download_syncs))
require_ple_fusion = require_ple_fusion.lower() not in {"0", "false", "off", "no"}
if min_graph_replays.lower() in {"", "auto"}:
    min_graph_replays = max(1, min_tokens - 64)
else:
    min_graph_replays = int(float(min_graph_replays))
max_launches = float(max_launches)

paths = [
    path for path in sorted(glob.glob(os.path.join(out_dir, "e4b_qat_cuda*.json")))
    if os.path.splitext(os.path.basename(path))[0] == "e4b_qat_cuda"
    or os.path.splitext(os.path.basename(path))[0].startswith("e4b_qat_cuda_run")
]
if not paths:
    sys.exit(0)

errors = []
rates = []
for path in paths:
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = int(data.get("tokens") or 0)
    rate = float(data.get("decode_tok_per_s") or 0.0)
    rates.append(rate)
    cuda = data.get("cuda") or {}
    if tokens < min_tokens:
        errors.append(f"{name}: tokens={tokens} < {min_tokens}")
    if rate < min_run_tok_s:
        errors.append(f"{name}: decode_tok_per_s={rate:.3f} < {min_run_tok_s:.3f}")
    if max_download_syncs is not None:
        download_syncs = int(cuda.get("download_syncs") or 0)
        if download_syncs > max_download_syncs:
            errors.append(f"{name}: download_syncs={download_syncs} > {max_download_syncs}")
    if require_ple_fusion:
        add_mul_fused = int(cuda.get("add_mul_scalar_fused") or 0)
        if add_mul_fused < min_tokens:
            errors.append(f"{name}: add_mul_scalar_fused={add_mul_fused} < {min_tokens}")
    scalar_launches_raw = cuda.get("launch_scalar")
    if scalar_launches_raw is None:
        errors.append(f"{name}: launch_scalar missing")
    elif int(scalar_launches_raw) != 0:
        errors.append(f"{name}: launch_scalar={int(scalar_launches_raw)} > 0")
    embedding_launches_raw = cuda.get("launch_embedding")
    if embedding_launches_raw is None:
        errors.append(f"{name}: launch_embedding missing")
    elif int(embedding_launches_raw) > min_tokens + 1:
        errors.append(f"{name}: launch_embedding={int(embedding_launches_raw)} > {min_tokens + 1}")
    if require_fused:
        for field in ("qkv_fused_q4_0", "linear_pair_fused_q4_0", "gated_down_fused_q4_0"):
            value = int(cuda.get(field) or 0)
            if value <= 0:
                errors.append(f"{name}: {field}={value}")
        for field in ("qkv_fallback_unsupported", "qkv_kernel_unavailable", "linear_pair_fallbacks", "gated_down_fallbacks"):
            value = int(cuda.get(field) or 0)
            if value != 0:
                errors.append(f"{name}: {field}={value}")
        if require_gated_down_tile8:
            tile8_hits = int(cuda.get("gated_down_fused_q4_0_tile8") or 0)
            if tile8_hits <= 0:
                errors.append(f"{name}: gated_down_fused_q4_0_tile8={tile8_hits}")
        q4_0_lm_head = int(cuda.get("lm_head_argmax_fused_q4_0") or 0)
        if q4_0_lm_head != 0:
            errors.append(f"{name}: lm_head_argmax_fused_q4_0={q4_0_lm_head} > 0")
        q6_lm_head = int(cuda.get("lm_head_argmax_fused_q6") or 0)
        if q6_lm_head <= 0:
            errors.append(f"{name}: lm_head_argmax_fused_q6={q6_lm_head}")
        lm_head_fallbacks = int(cuda.get("lm_head_argmax_fallbacks") or 0)
        if lm_head_fallbacks != 0:
            errors.append(f"{name}: lm_head_argmax_fallbacks={lm_head_fallbacks}")
    if require_fast_gqa:
        fast_hits = int(cuda.get("launch_attention_gqa_decode_fast") or 0)
        if fast_hits <= 0:
            errors.append(f"{name}: launch_attention_gqa_decode_fast={fast_hits}")
        fast_fallbacks = int(cuda.get("launch_attention_gqa_decode_fast_fallbacks") or 0)
        if fast_fallbacks != 0:
            errors.append(f"{name}: launch_attention_gqa_decode_fast_fallbacks={fast_fallbacks}")
    if require_graph:
        graph_replays = int(cuda.get("graph_capture_persistent_replays") or 0)
        if graph_replays < min_graph_replays:
            errors.append(f"{name}: graph_capture_persistent_replays={graph_replays} < {min_graph_replays}")
        capacity_skips = int(cuda.get("graph_capture_capacity_skips") or 0)
        if capacity_skips != 0:
            errors.append(f"{name}: graph_capture_capacity_skips={capacity_skips}")
        launches = float(cuda.get("launches_per_token") or 0.0)
        if launches <= 0 or launches > max_launches:
            errors.append(f"{name}: launches_per_token={launches:.3f} > {max_launches:.3f}")
    if require_device_token_handoff:
        generation_runtime = data.get("generation_decoder_runtime") or {}
        min_handoffs = max(0, min_tokens - 1)
        handoff_attempts = int(generation_runtime.get("device_token_handoff_attempts") or 0)
        handoff_hits = int(generation_runtime.get("device_token_handoff_hits") or 0)
        handoff_fallbacks = int(generation_runtime.get("device_token_handoff_fallbacks") or 0)
        handoff_seeds = int(generation_runtime.get("device_token_handoff_seeds") or 0)
        if handoff_attempts < min_handoffs:
            errors.append(f"{name}: device_token_handoff_attempts={handoff_attempts} < {min_handoffs}")
        if handoff_hits < min_handoffs:
            errors.append(f"{name}: device_token_handoff_hits={handoff_hits} < {min_handoffs}")
        if handoff_fallbacks != 0:
            errors.append(f"{name}: device_token_handoff_fallbacks={handoff_fallbacks}")
        if handoff_seeds < 1:
            errors.append(f"{name}: device_token_handoff_seeds={handoff_seeds}")
    if require_raw_token_export:
        cuda_generate = data.get("cuda_generate") or {}
        to_float32_calls = cuda_generate.get("to_float32_calls")
        to_float32_bytes = cuda_generate.get("to_float32_bytes")
        if to_float32_calls is None or int(to_float32_calls) != 0:
            errors.append(f"{name}: cuda_generate.to_float32_calls={to_float32_calls}")
        if to_float32_bytes is None or int(to_float32_bytes) != 0:
            errors.append(f"{name}: cuda_generate.to_float32_bytes={to_float32_bytes}")

if len(rates) < repeats:
    errors.append(f"repeat_count={len(rates)} < {repeats}")
min_rate = min(rates)
avg_rate = sum(rates) / len(rates)
if min_rate < min_run_tok_s:
    errors.append(f"repeat_min_decode_tok_per_s={min_rate:.3f} < {min_run_tok_s:.3f}")
if avg_rate < min_tok_s:
    errors.append(f"repeat_avg_decode_tok_per_s={avg_rate:.3f} < {min_tok_s:.3f}")
baseline_path = os.path.join(out_dir, "e4b_q4k_baseline.json")
if os.path.exists(baseline_path):
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    q4k_rate = float(baseline.get("decode_tok_per_s") or 0.0)
    ratio = avg_rate / q4k_rate if q4k_rate > 0 else 0.0
    if ratio < min_qat_over_q4k_ratio:
        errors.append(f"qat_over_q4k_ratio={ratio:.3f} < {min_qat_over_q4k_ratio:.3f} qat_avg={avg_rate:.3f} q4k={q4k_rate:.3f}")

if errors:
    for error in errors:
        print(f"e4b_qat_cuda: {error}", file=sys.stderr)
    sys.exit(1)

print(
    f"PASS e4b_qat_cuda: runs={len(rates)} min_decode_tok_per_s={min_rate:.3f} "
    f"avg_decode_tok_per_s={avg_rate:.3f}"
)
PY
}

check_e4b_qat_long_gate_json_if_present() {
  python3 - "$out_dir" "$e4b_qat_long_min_tokens" "$min_e4b_qat_long_tok_s" "$e4b_qat_long_min_graph_replays" "$e4b_qat_long_max_launches_per_token" "$e4b_qat_long_max_download_syncs" "$e4b_qat_require_fused" "$e4b_qat_require_fast_gqa" "$e4b_qat_require_graph_replay" "$e4b_qat_require_device_token_handoff" "$e4b_qat_require_gated_down_tile8" "$e4b_qat_require_raw_token_export" "$e4b_qat_require_ple_fusion" <<'PY'
import json
import os
import sys

(
    out_dir,
    min_tokens,
    min_tok_s,
    min_graph_replays,
    max_launches,
    max_download_syncs,
    require_fused,
    require_fast_gqa,
    require_graph,
    require_device_token_handoff,
    require_gated_down_tile8,
    require_raw_token_export,
    require_ple_fusion,
) = sys.argv[1:14]
min_tokens = int(float(min_tokens))
min_tok_s = float(min_tok_s)
max_launches = float(max_launches)
max_download_syncs = None if max_download_syncs.lower() in {"", "off", "none", "no"} else int(float(max_download_syncs))
require_fused = require_fused.lower() not in {"0", "false", "off", "no"}
require_fast_gqa = require_fast_gqa.lower() not in {"0", "false", "off", "no"}
require_graph = require_graph.lower() not in {"0", "false", "off", "no"}
require_device_token_handoff = require_device_token_handoff.lower() not in {"0", "false", "off", "no"}
require_gated_down_tile8 = require_gated_down_tile8.lower() not in {"0", "false", "off", "no"}
require_raw_token_export = require_raw_token_export.lower() not in {"0", "false", "off", "no"}
require_ple_fusion = require_ple_fusion.lower() not in {"0", "false", "off", "no"}

path = os.path.join(out_dir, "e4b_qat_cuda_long.json")
if not os.path.exists(path):
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

errors = []
tokens = int(data.get("tokens") or 0)
rate = float(data.get("decode_tok_per_s") or 0.0)
counter_floor = max(min_tokens, tokens)
if min_graph_replays.lower() in {"", "auto"}:
    graph_floor = max(1, counter_floor - 64)
else:
    graph_floor = int(float(min_graph_replays))

if tokens < min_tokens:
    errors.append(f"tokens={tokens} < {min_tokens}")
if rate < min_tok_s:
    errors.append(f"decode_tok_per_s={rate:.3f} < {min_tok_s:.3f}")

cuda = data.get("cuda") or {}
if max_download_syncs is not None:
    download_syncs = int(cuda.get("download_syncs") or 0)
    if download_syncs > max_download_syncs:
        errors.append(f"download_syncs={download_syncs} > {max_download_syncs}")
if require_ple_fusion:
    add_mul_fused = int(cuda.get("add_mul_scalar_fused") or 0)
    if add_mul_fused < counter_floor:
        errors.append(f"add_mul_scalar_fused={add_mul_fused} < {counter_floor}")

scalar_launches_raw = cuda.get("launch_scalar")
if scalar_launches_raw is None:
    errors.append("launch_scalar missing")
elif int(scalar_launches_raw) != 0:
    errors.append(f"launch_scalar={int(scalar_launches_raw)} > 0")
embedding_launches_raw = cuda.get("launch_embedding")
# EOS rollback can leave two lookahead embedding launches in this long run.
if embedding_launches_raw is None:
    errors.append("launch_embedding missing")
elif int(embedding_launches_raw) > counter_floor + 3:
    errors.append(f"launch_embedding={int(embedding_launches_raw)} > {counter_floor + 3}")

if require_fused:
    for field in ("qkv_fused_q4_0", "linear_pair_fused_q4_0", "gated_down_fused_q4_0"):
        value = int(cuda.get(field) or 0)
        if value <= 0:
            errors.append(f"{field}={value}")
    for field in ("qkv_fallback_unsupported", "qkv_kernel_unavailable", "linear_pair_fallbacks", "gated_down_fallbacks"):
        value = int(cuda.get(field) or 0)
        if value != 0:
            errors.append(f"{field}={value}")
        if require_gated_down_tile8:
            tile8_hits = int(cuda.get("gated_down_fused_q4_0_tile8") or 0)
            if tile8_hits <= 0:
                errors.append(f"gated_down_fused_q4_0_tile8={tile8_hits}")
        q4_0_lm_head = int(cuda.get("lm_head_argmax_fused_q4_0") or 0)
        if q4_0_lm_head != 0:
            errors.append(f"lm_head_argmax_fused_q4_0={q4_0_lm_head} > 0")
        q6_lm_head = int(cuda.get("lm_head_argmax_fused_q6") or 0)
        if q6_lm_head <= 0:
            errors.append(f"lm_head_argmax_fused_q6={q6_lm_head}")
        lm_head_fallbacks = int(cuda.get("lm_head_argmax_fallbacks") or 0)
        if lm_head_fallbacks != 0:
            errors.append(f"lm_head_argmax_fallbacks={lm_head_fallbacks}")

if require_fast_gqa:
    fast_hits = int(cuda.get("launch_attention_gqa_decode_fast") or 0)
    if fast_hits <= 0:
        errors.append(f"launch_attention_gqa_decode_fast={fast_hits}")
    fast_fallbacks = int(cuda.get("launch_attention_gqa_decode_fast_fallbacks") or 0)
    if fast_fallbacks != 0:
        errors.append(f"launch_attention_gqa_decode_fast_fallbacks={fast_fallbacks}")

if require_raw_token_export:
    cuda_generate = data.get("cuda_generate") or {}
    to_float32_calls = cuda_generate.get("to_float32_calls")
    to_float32_bytes = cuda_generate.get("to_float32_bytes")
    if to_float32_calls is None or int(to_float32_calls) != 0:
        errors.append(f"cuda_generate.to_float32_calls={to_float32_calls}")
    if to_float32_bytes is None or int(to_float32_bytes) != 0:
        errors.append(f"cuda_generate.to_float32_bytes={to_float32_bytes}")

if require_graph:
    graph_replays = int(cuda.get("graph_capture_persistent_replays") or 0)
    if graph_replays < graph_floor:
        errors.append(f"graph_capture_persistent_replays={graph_replays} < {graph_floor}")
    capacity_skips = int(cuda.get("graph_capture_capacity_skips") or 0)
    if capacity_skips != 0:
        errors.append(f"graph_capture_capacity_skips={capacity_skips}")
    launches = float(cuda.get("launches_per_token") or 0.0)
    if launches <= 0 or launches > max_launches:
        errors.append(f"launches_per_token={launches:.3f} > {max_launches:.3f}")

if require_device_token_handoff:
    generation_runtime = data.get("generation_decoder_runtime") or {}
    min_handoffs = max(0, counter_floor - 1)
    handoff_attempts = int(generation_runtime.get("device_token_handoff_attempts") or 0)
    handoff_hits = int(generation_runtime.get("device_token_handoff_hits") or 0)
    handoff_fallbacks = int(generation_runtime.get("device_token_handoff_fallbacks") or 0)
    handoff_seeds = int(generation_runtime.get("device_token_handoff_seeds") or 0)
    if handoff_attempts < min_handoffs:
        errors.append(f"device_token_handoff_attempts={handoff_attempts} < {min_handoffs}")
    if handoff_hits < min_handoffs:
        errors.append(f"device_token_handoff_hits={handoff_hits} < {min_handoffs}")
    if handoff_fallbacks != 0:
        errors.append(f"device_token_handoff_fallbacks={handoff_fallbacks}")
    if handoff_seeds < 1:
        errors.append(f"device_token_handoff_seeds={handoff_seeds}")

if errors:
    for error in errors:
        print(f"e4b_qat_cuda_long: {error}", file=sys.stderr)
    sys.exit(1)

print(
    f"PASS e4b_qat_cuda_long: tokens={tokens} decode_tok_per_s={rate:.3f} "
    f"graph_floor={graph_floor}"
)
PY
}

wait_for_resident_server() {
  local url="$1"
  local log_file="$2"
  local attempts=0
  while [ "$attempts" -lt 120 ]; do
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
    if ! kill -0 "$resident_server_pid" >/dev/null 2>&1; then
      log "resident server exited before becoming ready"
      sed -n '1,220p' "$log_file" >&2 || true
      return 1
    fi
    attempts=$((attempts + 1))
    sleep 0.5
  done
  log "resident server did not become ready at $url"
  sed -n '1,220p' "$log_file" >&2 || true
  return 1
}

resident_generate_request() {
  local label="$1"
  local url="$2"
  local response_json="$3"
  local model="${4:-$resident_model}"
  local prompt="${5:-$resident_prompt}"
  local tokens="${6:-$resident_tokens}"
  local cache_dtype="${7:-$resident_cache_dtype}"
  python3 - "$url" "$response_json" "$model" "$prompt" "$tokens" "$cache_dtype" "$label" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

url, response_path, model, prompt, tokens, cache_dtype, label = sys.argv[1:8]
body = {
    "model": model,
    "backend": "cuda",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(tokens),
    "temperature": 0,
    "stream": False,
}
if cache_dtype:
    body["cache_dtype"] = cache_dtype

payload = json.dumps(body).encode("utf-8")
request = urllib.request.Request(
    url,
    data=payload,
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
    text = raw.decode("utf-8", errors="replace")
    raise SystemExit(f"{label}: HTTP {status}: {text[:1000]}")

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
  python3 - "$endpoint" "$response_prefix" "$model" "$e4b_qat_resident_prompt" "$tokens" "$request_count" "$concurrency" "$min_completion_tokens" "$tsv_path" "$meta_json" <<'PY'
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
    f"RUN e4b_qat_resident_soak_requests: requests={request_count} concurrency={concurrency} "
    f"total_completion_tokens={total_completion_tokens} wall_ms={wall_ms:.1f} "
    f"aggregate_tok_s={aggregate_tok_s:.3f}"
)
PY
}

run_e4b_qat_resident_soak_if_enabled() {
  local endpoint="$1"
  local server_log="$2"
  local qat_model="$3"
  case "$run_e4b_qat_resident_soak" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_RESIDENT_SOAK=$run_e4b_qat_resident_soak; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_qat_resident_soak" = "off" ]; then
    return 0
  fi
  for pair in \
    "E4B_QAT_RESIDENT_SOAK_REQUESTS:$e4b_qat_resident_soak_request_count" \
    "E4B_QAT_RESIDENT_SOAK_CONCURRENCY:$e4b_qat_resident_soak_concurrency" \
    "E4B_QAT_RESIDENT_SOAK_TOKENS:$e4b_qat_resident_soak_tokens" \
    "E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS:$e4b_qat_resident_soak_min_completion_tokens"; do
    local name="${pair%%:*}"
    local value="${pair#*:}"
    case "$value" in
      ''|*[!0-9]*)
        log "invalid $name=$value; expected positive integer"
        exit 2
        ;;
    esac
    if [ "$value" -lt 1 ]; then
      log "invalid $name=$value; expected positive integer"
      exit 2
    fi
  done
  if [ "$e4b_qat_resident_soak_min_completion_tokens" -gt "$e4b_qat_resident_soak_tokens" ]; then
    log "invalid E4B_QAT_RESIDENT_SOAK_MIN_COMPLETION_TOKENS=$e4b_qat_resident_soak_min_completion_tokens exceeds E4B_QAT_RESIDENT_SOAK_TOKENS=$e4b_qat_resident_soak_tokens"
    exit 2
  fi
  case "$e4b_qat_resident_soak_min_graph_replays" in
    auto|''|*[!0-9]*)
      if [ "$e4b_qat_resident_soak_min_graph_replays" != "auto" ]; then
        log "invalid E4B_QAT_RESIDENT_SOAK_MIN_GRAPH_REPLAYS=$e4b_qat_resident_soak_min_graph_replays; expected auto or non-negative integer"
        exit 2
      fi
      ;;
  esac

  local tsv_path="$out_dir/e4b_qat_resident_soak.tsv"
  local meta_json="$out_dir/e4b_qat_resident_soak_meta.json"
  e4b_qat_resident_soak_requests \
    "$endpoint" \
    "$out_dir/e4b_qat_resident_soak" \
    "$qat_model" \
    "$e4b_qat_resident_soak_tokens" \
    "$e4b_qat_resident_soak_request_count" \
    "$e4b_qat_resident_soak_concurrency" \
    "$e4b_qat_resident_soak_min_completion_tokens" \
    "$tsv_path" \
    "$meta_json" | tee -a "$summary"

  python3 - "$tsv_path" "$meta_json" "$server_log" "$min_e4b_qat_resident_soak_agg_tok_s" "$min_e4b_qat_resident_soak_request_tok_s" "$e4b_qat_resident_soak_max_p95_e2e_ms" "$e4b_qat_resident_soak_request_count" "$e4b_qat_resident_soak_min_completion_tokens" "$e4b_qat_resident_decode_graph_replay" "$e4b_qat_resident_soak_min_graph_replays" "$e4b_qat_resident_warm_repeats" "$e4b_qat_resident_tokens" "$e4b_qat_resident_soak_tokens" <<'PY' | tee -a "$summary"
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
with open(meta_json, "r", encoding="utf-8") as f:
    meta = json.load(f)

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
        print(f"e4b_qat_resident_soak: {error}", file=sys.stderr)
    raise SystemExit(1)

min_rate = min(rates) if rates else 0.0
avg_rate = sum(rates) / len(rates) if rates else 0.0
print(
    f"PASS e4b_qat_resident_soak: requests={len(rows)} concurrency={meta.get('concurrency')} "
    f"aggregate_tok_s={aggregate_tok_s:.3f} min_e2e_tok_s={min_rate:.3f} "
    f"avg_e2e_tok_s={avg_rate:.3f} p50_e2e_ms={p50_e2e_ms:.1f} "
    f"p95_e2e_ms={p95_e2e_ms:.1f} p99_e2e_ms={p99_e2e_ms:.1f} "
    f"max_e2e_ms={max_e2e_ms:.1f} graph_replays={replays} graph_floor={graph_floor}"
)
PY
}

e4b_qat_resident_backpressure_requests() {
  local endpoint="$1"
  local response_prefix="$2"
  local model="$3"
  local tokens="$4"
  local request_count="$5"
  local concurrency="$6"
  local tsv_path="$7"
  python3 - "$endpoint" "$response_prefix" "$model" "$e4b_qat_resident_prompt" "$tokens" "$request_count" "$concurrency" "$tsv_path" <<'PY'
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

run_e4b_qat_resident_backpressure_if_enabled() {
  local endpoint="$1"
  local server_log="$2"
  local qat_model="$3"
  local metrics_url="$4"
  case "$run_e4b_qat_resident_backpressure" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_RESIDENT_BACKPRESSURE=$run_e4b_qat_resident_backpressure; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_qat_resident_backpressure" = "off" ]; then
    return 0
  fi
  if [ -z "$e4b_qat_resident_max_concurrent_requests" ]; then
    log "invalid E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=; required for backpressure gate"
    exit 2
  fi
  for pair in \
    "E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS:$e4b_qat_resident_backpressure_request_count" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY:$e4b_qat_resident_backpressure_concurrency" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_TOKENS:$e4b_qat_resident_backpressure_tokens" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_MIN_ACCEPTED:$e4b_qat_resident_backpressure_min_accepted" \
    "E4B_QAT_RESIDENT_BACKPRESSURE_MIN_REJECTED:$e4b_qat_resident_backpressure_min_rejected"; do
    local name="${pair%%:*}"
    local value="${pair#*:}"
    case "$value" in
      ''|*[!0-9]*)
        log "invalid $name=$value; expected positive integer"
        exit 2
        ;;
    esac
    if [ "$value" -lt 1 ]; then
      log "invalid $name=$value; expected positive integer"
      exit 2
    fi
  done
  case "$e4b_qat_resident_backpressure_min_graph_replays" in
    auto|''|*[!0-9]*)
      if [ "$e4b_qat_resident_backpressure_min_graph_replays" != "auto" ]; then
        log "invalid E4B_QAT_RESIDENT_BACKPRESSURE_MIN_GRAPH_REPLAYS=$e4b_qat_resident_backpressure_min_graph_replays; expected auto or non-negative integer"
        exit 2
      fi
      ;;
  esac

  local tsv_path="$out_dir/e4b_qat_resident_backpressure.tsv"
  local metrics_path="$out_dir/e4b_qat_resident_backpressure_metrics.txt"
  e4b_qat_resident_backpressure_requests \
    "$endpoint" \
    "$out_dir/e4b_qat_resident_backpressure" \
    "$qat_model" \
    "$e4b_qat_resident_backpressure_tokens" \
    "$e4b_qat_resident_backpressure_request_count" \
    "$e4b_qat_resident_backpressure_concurrency" \
    "$tsv_path"

  python3 - "$tsv_path" "$server_log" "$metrics_url" "$metrics_path" "$e4b_qat_resident_backpressure_min_accepted" "$e4b_qat_resident_backpressure_min_rejected" "$e4b_qat_resident_backpressure_max_reject_ms" "$e4b_qat_resident_warm_repeats" "$e4b_qat_resident_tokens" "$e4b_qat_resident_backpressure_tokens" "$e4b_qat_resident_max_concurrent_requests" "$e4b_qat_resident_decode_graph_replay" "$e4b_qat_resident_backpressure_min_graph_replays" <<'PY' | tee -a "$summary"
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

for name, expected in {
    "antfly_inference_request_queue_capacity": float(max_concurrent),
    "antfly_inference_request_queue_depth": 0.0,
    "antfly_inference_request_queue_available": float(max_concurrent),
    "antfly_inference_request_queue_active_requests": 0.0,
}.items():
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
        print(f"e4b_qat_resident_backpressure: {error}", file=sys.stderr)
    raise SystemExit(1)

accepted_rates = [float(row["e2e_tok_s"]) for row in accepted]
reject_ms = [float(row["e2e_ms"]) for row in rejected]
avg_accepted = sum(accepted_rates) / len(accepted_rates) if accepted_rates else 0.0
max_reject = max(reject_ms) if reject_ms else 0.0
print(
    f"PASS e4b_qat_resident_backpressure: requests={len(rows)} accepted={len(accepted)} "
    f"rejected={len(rejected)} avg_accepted_tok_s={avg_accepted:.3f} "
    f"max_reject_ms={max_reject:.1f} graph_replays={replays} graph_floor={graph_floor} "
    f"queue_rejections={rejection_metric:.0f} queue_rejected_units={rejected_units_metric:.0f}"
)
PY
}

run_e4b_qat_resident_gate_if_enabled() {
  case "$run_e4b_qat_resident_soak" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_RESIDENT_SOAK=$run_e4b_qat_resident_soak; expected auto|required|off"
      exit 2
      ;;
  esac
  case "$run_e4b_qat_resident_backpressure" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_RESIDENT_BACKPRESSURE=$run_e4b_qat_resident_backpressure; expected auto|required|off"
      exit 2
      ;;
  esac
  case "$run_e4b_qat_resident" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_QAT_RESIDENT=$run_e4b_qat_resident; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_qat_resident" = "off" ]; then
    if [ "$run_e4b_qat_resident_soak" != "off" ]; then
      log "invalid RUN_E4B_QAT_RESIDENT_SOAK=$run_e4b_qat_resident_soak; requires RUN_E4B_QAT_RESIDENT"
      exit 2
    fi
    if [ "$run_e4b_qat_resident_backpressure" != "off" ]; then
      log "invalid RUN_E4B_QAT_RESIDENT_BACKPRESSURE=$run_e4b_qat_resident_backpressure; requires RUN_E4B_QAT_RESIDENT"
      exit 2
    fi
    return 0
  fi
  if [ "$run_e4b_qat_resident_backpressure" != "off" ] && [ -z "$e4b_qat_resident_max_concurrent_requests" ]; then
    log "invalid E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=; required for backpressure gate"
    exit 2
  fi
  if [ ! -e "$e4b_qat_model" ]; then
    if [ "$run_e4b_qat_resident" = "required" ]; then
      require_path "E4B QAT model" "$e4b_qat_model"
    fi
    log "SKIP e4b_qat_resident_cuda_server: model missing: $e4b_qat_model"
    return 0
  fi
  case "$e4b_qat_resident_tokens" in
    ''|*[!0-9]*)
      log "invalid E4B_QAT_RESIDENT_TOKENS=$e4b_qat_resident_tokens; expected positive integer"
      exit 2
      ;;
  esac
  case "$e4b_qat_resident_warm_repeats" in
    ''|*[!0-9]*)
      log "invalid E4B_QAT_RESIDENT_WARM_REPEATS=$e4b_qat_resident_warm_repeats; expected positive integer"
      exit 2
      ;;
  esac
  case "$e4b_qat_resident_min_completion_tokens" in
    ''|*[!0-9]*)
      log "invalid E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS=$e4b_qat_resident_min_completion_tokens; expected positive integer"
      exit 2
      ;;
  esac
  if [ "$e4b_qat_resident_tokens" -lt 1 ]; then
    log "invalid E4B_QAT_RESIDENT_TOKENS=$e4b_qat_resident_tokens; expected positive integer"
    exit 2
  fi
  if [ "$e4b_qat_resident_min_completion_tokens" -lt 1 ]; then
    log "invalid E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS=$e4b_qat_resident_min_completion_tokens; expected positive integer"
    exit 2
  fi
  if [ "$e4b_qat_resident_min_completion_tokens" -gt "$e4b_qat_resident_tokens" ]; then
    log "invalid E4B_QAT_RESIDENT_MIN_COMPLETION_TOKENS=$e4b_qat_resident_min_completion_tokens exceeds E4B_QAT_RESIDENT_TOKENS=$e4b_qat_resident_tokens"
    exit 2
  fi
  if [ "$e4b_qat_resident_warm_repeats" -lt 1 ]; then
    log "invalid E4B_QAT_RESIDENT_WARM_REPEATS=$e4b_qat_resident_warm_repeats; expected positive integer"
    exit 2
  fi
  case "$e4b_qat_resident_min_graph_replays" in
    auto|''|*[!0-9]*)
      if [ "$e4b_qat_resident_min_graph_replays" != "auto" ]; then
        log "invalid E4B_QAT_RESIDENT_MIN_GRAPH_REPLAYS=$e4b_qat_resident_min_graph_replays; expected auto or non-negative integer"
        exit 2
      fi
      ;;
  esac
  local -a server_capacity_args=()
  if [ -n "$e4b_qat_resident_max_concurrent_requests" ]; then
    case "$e4b_qat_resident_max_concurrent_requests" in
      ''|*[!0-9]*)
        log "invalid E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=$e4b_qat_resident_max_concurrent_requests; expected positive integer"
        exit 2
        ;;
    esac
    if [ "$e4b_qat_resident_max_concurrent_requests" -lt 1 ]; then
      log "invalid E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=$e4b_qat_resident_max_concurrent_requests; expected positive integer"
      exit 2
    fi
    server_capacity_args=(--max-concurrent-requests "$e4b_qat_resident_max_concurrent_requests")
  fi

  local qat_model
  qat_model="$(abs_path "$e4b_qat_model")"
  local models_dir
  models_dir="$(abs_path "$resident_models_dir")"
  local port="$e4b_qat_resident_port"
  if [ -z "$port" ]; then
    port="$(choose_port)"
  fi

  local server_log="$out_dir/e4b_qat_resident_server.log"
  local endpoint="http://$resident_host:$port/ai/v1/generate"
  local metrics_url="http://$resident_host:$port/ml/v1/metrics"
  local ready_url="http://$resident_host:$port/healthz"
  local resident_tsv="$out_dir/e4b_qat_resident_cuda_server.tsv"
  local resident_graph_probe_trace="${ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE:-0}"
  if [ "$e4b_qat_resident_decode_graph_replay" = "required" ]; then
    resident_graph_probe_trace=1
  fi
  log "RUN e4b_qat_resident_cuda_server: model=$qat_model tokens=$e4b_qat_resident_tokens repeats=$e4b_qat_resident_warm_repeats port=$port"
  env \
    ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE="$resident_graph_probe_trace" \
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$e4b_qat_resident_decode_graph_replay" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$e4b_qat_temp_slot_period" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$e4b_qat_capture_allow_unpinned" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$e4b_qat_capture_min_alloc_seq" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$e4b_qat_force_kv_capacity" \
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1 \
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8="$e4b_qat_q4_0_gated_down_tile8" \
    ANTFLY_INFERENCE_CUDA_Q4_0_PLE_GATE_FUSION="$e4b_qat_q4_0_ple_gate_fusion" \
    ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION="$e4b_qat_ple_rms_embed_fusion" \
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$e4b_qat_pending_token_readback" \
    "$antfly_bin" run \
      --host "$resident_host" \
      --port "$port" \
      "${server_capacity_args[@]}" \
      --models-dir "$models_dir" \
      --preload-model "generator:cuda:$qat_model" >"$server_log" 2>&1 &
  resident_server_pid=$!
  wait_for_resident_server "$ready_url" "$server_log"

  {
    printf 'case\te2e_ms\tcompletion_tokens\te2e_tok_s\n'
    local repeat=1
    while [ "$repeat" -le "$e4b_qat_resident_warm_repeats" ]; do
      resident_generate_request "e4b_qat_resident_warm${repeat}" "$endpoint" "$out_dir/e4b_qat_resident_warm${repeat}.json" "$qat_model" "$e4b_qat_resident_prompt" "$e4b_qat_resident_tokens" "f32"
      repeat=$((repeat + 1))
    done
  } | tee "$resident_tsv"

  python3 - "$resident_tsv" "$min_e4b_qat_resident_warm_tok_s" "$e4b_qat_resident_warm_repeats" "$e4b_qat_resident_min_completion_tokens" <<'PY' | tee -a "$summary"
import csv
import sys

path, min_warm_tok_s, expected_repeats, min_completion_tokens = sys.argv[1:5]
min_warm_tok_s = float(min_warm_tok_s)
expected_repeats = int(expected_repeats)
min_completion_tokens = int(min_completion_tokens)
rows = []
with open(path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

errors = []
if len(rows) < expected_repeats:
    errors.append(f"warm_rows={len(rows)} < {expected_repeats}")
rates = []
for row in rows:
    try:
        tokens = int(row["completion_tokens"])
        rate = float(row["e2e_tok_s"])
    except Exception as exc:
        errors.append(f"{row.get('case', 'unknown')}: invalid row: {exc}")
        continue
    if tokens <= 0:
        errors.append(f"{row.get('case')}: completion_tokens={tokens}")
    if tokens < min_completion_tokens:
        errors.append(f"{row.get('case')}: completion_tokens={tokens} < {min_completion_tokens}")
    if rate < min_warm_tok_s:
        errors.append(f"{row.get('case')}: e2e_tok_s={rate:.3f} < {min_warm_tok_s:.3f}")
    rates.append(rate)

if errors:
    for error in errors:
        print(f"e4b_qat_resident_cuda_server: {error}", file=sys.stderr)
    sys.exit(1)

min_rate = min(rates)
avg_rate = sum(rates) / len(rates)
print(
    f"PASS e4b_qat_resident_cuda_server: warm_repeats={len(rates)} "
    f"min_e2e_tok_s={min_rate:.3f} avg_e2e_tok_s={avg_rate:.3f}"
)
PY
  if [ "$e4b_qat_resident_decode_graph_replay" = "required" ]; then
    python3 - "$server_log" "$e4b_qat_resident_min_graph_replays" "$e4b_qat_resident_tokens" <<'PY' | tee -a "$summary"
import sys

log_path, floor_arg, token_arg = sys.argv[1:4]
tokens = int(token_arg)
if floor_arg == "auto":
    floor = max(1, tokens // 3)
else:
    floor = int(floor_arg)

with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

replays = text.count("persistent_replayed") + text.count("instantiated_cached_replayed")
errors = []
if replays < floor:
    errors.append(f"resident graph replays={replays} < {floor}")
for marker in (
    "unsafe_d2h_copy",
    "unsafe_h2d_copy",
    "unsafe_temp_alloc",
    "CudaGraphCaptureUnsafe",
    "persistent_replay_kv_capacity_exceeded",
    "cuda_graph_capture_probe: discarded",
):
    if marker in text:
        errors.append(f"resident graph log contains {marker}")

if errors:
    for error in errors:
        print(f"e4b_qat_resident_cuda_server: {error}", file=sys.stderr)
    sys.exit(1)

print(f"PASS e4b_qat_resident_cuda_graph_replay: replays={replays} floor={floor}")
PY
  fi
  local resident_subgate_failed=0
  if ! run_e4b_qat_resident_soak_if_enabled "$endpoint" "$server_log" "$qat_model"; then
    resident_subgate_failed=1
  fi
  if ! run_e4b_qat_resident_backpressure_if_enabled "$endpoint" "$server_log" "$qat_model" "$metrics_url"; then
    resident_subgate_failed=1
  fi
  cleanup_resident_server
  run_e4b_q4k_resident_baseline_if_enabled "$resident_tsv"
  if [ "$resident_subgate_failed" -ne 0 ]; then
    return
  fi
}

run_e4b_q4k_resident_baseline_if_enabled() {
  local qat_tsv="$1"
  case "$run_e4b_q4k_resident_baseline" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_E4B_Q4K_RESIDENT_BASELINE=$run_e4b_q4k_resident_baseline; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_e4b_q4k_resident_baseline" = "off" ]; then
    return 0
  fi
  if [ ! -e "$e4b_q4k_baseline_model" ]; then
    if [ "$run_e4b_q4k_resident_baseline" = "required" ]; then
      require_path "E4B Q4_K resident baseline model" "$e4b_q4k_baseline_model"
    fi
    log "SKIP e4b_q4k_resident_cuda_server: model missing: $e4b_q4k_baseline_model"
    return 0
  fi

  local q4k_model
  q4k_model="$(abs_path "$e4b_q4k_baseline_model")"
  local models_dir
  models_dir="$(abs_path "$resident_models_dir")"
  local port
  port="$(choose_port)"

  local server_log="$out_dir/e4b_q4k_resident_server.log"
  local endpoint="http://$resident_host:$port/ai/v1/generate"
  local ready_url="http://$resident_host:$port/healthz"
  local resident_tsv="$out_dir/e4b_q4k_resident_cuda_server.tsv"
  local resident_graph_probe_trace="${ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE:-0}"
  if [ "$e4b_q4k_resident_decode_graph_replay" = "required" ]; then
    resident_graph_probe_trace=1
  fi
  log "RUN e4b_q4k_resident_cuda_server: model=$q4k_model tokens=$e4b_qat_resident_tokens repeats=$e4b_qat_resident_warm_repeats port=$port"
  env \
    ANTFLY_INFERENCE_CUDA_GRAPH_CAPTURE_PROBE_TRACE="$resident_graph_probe_trace" \
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$e4b_q4k_resident_decode_graph_replay" \
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$e4b_qat_temp_slot_period" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED="$e4b_qat_capture_allow_unpinned" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ="$e4b_qat_capture_min_alloc_seq" \
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$e4b_qat_force_kv_capacity" \
    ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1 \
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$e4b_qat_pending_token_readback" \
    "$antfly_bin" run \
      --host "$resident_host" \
      --port "$port" \
      --models-dir "$models_dir" \
      --preload-model "generator:cuda:$q4k_model" >"$server_log" 2>&1 &
  resident_server_pid=$!
  wait_for_resident_server "$ready_url" "$server_log"

  {
    printf 'case\te2e_ms\tcompletion_tokens\te2e_tok_s\n'
    local repeat=1
    while [ "$repeat" -le "$e4b_qat_resident_warm_repeats" ]; do
      resident_generate_request "e4b_q4k_resident_warm${repeat}" "$endpoint" "$out_dir/e4b_q4k_resident_warm${repeat}.json" "$q4k_model" "$e4b_qat_resident_prompt" "$e4b_qat_resident_tokens" "f32"
      repeat=$((repeat + 1))
    done
  } | tee "$resident_tsv"

  python3 - "$resident_tsv" "$e4b_qat_resident_warm_repeats" "$e4b_qat_resident_min_completion_tokens" <<'PY' | tee -a "$summary"
import csv
import sys

path, expected_repeats, min_completion_tokens = sys.argv[1:4]
expected_repeats = int(expected_repeats)
min_completion_tokens = int(min_completion_tokens)
with open(path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

errors = []
rates = []
if len(rows) < expected_repeats:
    errors.append(f"warm_rows={len(rows)} < {expected_repeats}")
for row in rows:
    label = row.get("case", "unknown")
    try:
        tokens = int(row["completion_tokens"])
        rate = float(row["e2e_tok_s"])
    except Exception as exc:
        errors.append(f"{label}: invalid row: {exc}")
        continue
    if tokens < min_completion_tokens:
        errors.append(f"{label}: completion_tokens={tokens} < {min_completion_tokens}")
    if rate <= 0:
        errors.append(f"{label}: e2e_tok_s={rate:.3f} <= 0")
    rates.append(rate)

if errors:
    for error in errors:
        print(f"e4b_q4k_resident_cuda_server: {error}", file=sys.stderr)
    sys.exit(1)

print(
    f"PASS e4b_q4k_resident_cuda_server: warm_repeats={len(rates)} "
    f"min_e2e_tok_s={min(rates):.3f} avg_e2e_tok_s={sum(rates) / len(rates):.3f}"
)
PY
  if [ "$e4b_q4k_resident_decode_graph_replay" = "required" ]; then
    python3 - "$server_log" "$e4b_qat_resident_min_graph_replays" "$e4b_qat_resident_tokens" <<'PY' | tee -a "$summary"
import sys

log_path, floor_arg, token_arg = sys.argv[1:4]
tokens = int(token_arg)
floor = max(1, tokens // 3) if floor_arg == "auto" else int(floor_arg)
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

replays = text.count("persistent_replayed") + text.count("instantiated_cached_replayed")
errors = []
if replays < floor:
    errors.append(f"resident graph replays={replays} < {floor}")
for marker in (
    "unsafe_d2h_copy",
    "unsafe_h2d_copy",
    "unsafe_temp_alloc",
    "CudaGraphCaptureUnsafe",
    "persistent_replay_kv_capacity_exceeded",
    "cuda_graph_capture_probe: discarded",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
):
    if marker in text:
        errors.append(f"resident graph log contains {marker}")

if errors:
    for error in errors:
        print(f"e4b_q4k_resident_cuda_server: {error}", file=sys.stderr)
    sys.exit(1)

print(f"PASS e4b_q4k_resident_cuda_graph_replay: replays={replays} floor={floor}")
PY
  fi
  cleanup_resident_server

  python3 - "$qat_tsv" "$resident_tsv" "$min_e4b_qat_resident_over_q4k_ratio" <<'PY' | tee -a "$summary"
import csv
import sys

qat_path, q4k_path, floor_arg = sys.argv[1:4]
floor = float(floor_arg)

def avg_rate(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rates = [float(row["e2e_tok_s"]) for row in rows if row.get("e2e_tok_s")]
    if not rates:
        raise SystemExit(f"{path}: no e2e_tok_s rows")
    return sum(rates) / len(rates)

qat = avg_rate(qat_path)
q4k = avg_rate(q4k_path)
ratio = qat / q4k if q4k > 0 else 0.0
if ratio < floor:
    print(f"e4b_qat_resident_over_q4k: qat_avg={qat:.3f} q4k_avg={q4k:.3f} ratio={ratio:.3f} floor={floor}", file=sys.stderr)
    sys.exit(1)

print(f"PASS e4b_qat_resident_over_q4k: qat_avg={qat:.3f} q4k_avg={q4k:.3f} ratio={ratio:.3f} floor={floor}")
PY
}

run_resident_gate_if_enabled() {
  case "$run_resident" in
    auto|required|off)
      ;;
    *)
      log "invalid RUN_RESIDENT=$run_resident; expected auto|required|off"
      exit 2
      ;;
  esac
  if [ "$run_resident" = "off" ] || [ "$mode" = "mtp-only" ]; then
    return 0
  fi
  if [ ! -e "$resident_model" ]; then
    if [ "$run_resident" = "required" ]; then
      require_path "resident model" "$resident_model"
    fi
    log "SKIP resident_cuda_server: model missing: $resident_model"
    return 0
  fi

  resident_model="$(abs_path "$resident_model")"
  resident_models_dir="$(abs_path "$resident_models_dir")"
  if [ -z "$resident_port" ]; then
    resident_port="$(choose_port)"
  fi

  local server_log="$out_dir/resident_server.log"
  local endpoint="http://$resident_host:$resident_port/ai/v1/generate"
  local ready_url="http://$resident_host:$resident_port/healthz"
  local resident_tsv="$out_dir/resident_cuda_server.tsv"
  log "RUN resident_cuda_server: model=$resident_model cache_dtype=$resident_cache_dtype tokens=$resident_tokens port=$resident_port"
  "$antfly_bin" run \
    --host "$resident_host" \
    --port "$resident_port" \
    --models-dir "$resident_models_dir" >"$server_log" 2>&1 &
  resident_server_pid=$!
  wait_for_resident_server "$ready_url" "$server_log"

  {
    printf 'case\te2e_ms\tcompletion_tokens\te2e_tok_s\n'
    resident_generate_request resident_cold "$endpoint" "$out_dir/resident_cold.json"
    resident_generate_request resident_warm "$endpoint" "$out_dir/resident_warm.json"
  } | tee "$resident_tsv"

  python3 - "$resident_tsv" "$resident_max_warm_cold_ratio" "$resident_min_warm_tok_s" <<'PY' | tee -a "$summary"
import csv
import sys

path, max_ratio, min_warm_tok_s = sys.argv[1:4]
max_ratio = float(max_ratio)
min_warm_tok_s = float(min_warm_tok_s)
rows = {}
with open(path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        rows[row["case"]] = row

errors = []
try:
    cold_ms = float(rows["resident_cold"]["e2e_ms"])
    warm_ms = float(rows["resident_warm"]["e2e_ms"])
    warm_tok_s = float(rows["resident_warm"]["e2e_tok_s"])
except KeyError as exc:
    raise SystemExit(f"missing resident timing row: {exc}") from exc

if warm_ms >= cold_ms * max_ratio:
    errors.append(f"resident warm request {warm_ms:.1f}ms is not below cold*{max_ratio:.2f} ({cold_ms * max_ratio:.1f}ms)")
if warm_tok_s < min_warm_tok_s:
    errors.append(f"resident warm e2e_tok_s={warm_tok_s:.3f} < {min_warm_tok_s:.3f}")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(1)

print(
    f"PASS resident_cuda_server: cold_e2e={cold_ms / 1000.0:.2f}s "
    f"warm_e2e={warm_ms / 1000.0:.2f}s warm_e2e_tok_s={warm_tok_s:.3f}"
)
PY
  cleanup_resident_server
}

check_mtp_policy() {
  local target_json="$1"
  local mtp_json="$2"
  python3 - "$target_json" "$mtp_json" "$mtp_min_active_speed_ratio" <<'PY'
import json
import sys

target_path, mtp_path, min_ratio = sys.argv[1:4]
min_ratio = float(min_ratio)
with open(target_path, "r", encoding="utf-8") as f:
    target = json.load(f)
with open(mtp_path, "r", encoding="utf-8") as f:
    mtp = json.load(f)

errors = []
target_tps = float(target.get("decode_tok_per_s", 0.0))
mtp_tps = float(mtp.get("decode_tok_per_s", 0.0))
spec = mtp.get("speculative")
if target_tps <= 0:
    errors.append(f"target decode_tok_per_s={target_tps}, expected > 0")
if mtp_tps <= 0:
    errors.append(f"mtp decode_tok_per_s={mtp_tps}, expected > 0")
if not isinstance(spec, dict):
    errors.append("MTP run did not emit speculative telemetry")
    spec = {}

if spec.get("speculation_policy") != "auto":
    errors.append(f"speculation_policy={spec.get('speculation_policy')!r}, expected auto")
if spec.get("speculation_calibration") != "probe":
    errors.append(f"speculation_calibration={spec.get('speculation_calibration')!r}, expected probe")

decision = spec.get("speculation_policy_decision")
allowed = {
    "active",
    "disabled_low_acceptance",
    "disabled_zero_match",
    "disabled_slow",
    "disabled_uncalibrated",
    "disabled_unavailable",
}
if decision not in allowed:
    errors.append(f"speculation_policy_decision={decision!r}, expected one of {sorted(allowed)}")

if spec.get("mtp_enabled"):
    if int(spec.get("rounds", 0)) <= 0:
        errors.append("mtp_enabled=true but rounds <= 0")
    if int(spec.get("drafted", 0)) <= 0:
        errors.append("mtp_enabled=true but drafted <= 0")
    cuda = mtp.get("cuda") or {}
    cuda_generate = mtp.get("cuda_generate") or {}
    device_fallbacks = int(cuda.get("mtp_verify_commit_device_fallbacks", 0)) + int(cuda_generate.get("mtp_verify_commit_device_fallbacks", 0))
    if device_fallbacks != 0:
        errors.append(f"mtp_verify_commit_device_fallbacks={device_fallbacks}, expected 0")
    if decision == "active" and mtp_tps < target_tps * min_ratio:
        errors.append(
            f"MTP auto remained active at {mtp_tps:.3f} tok/s, below target-only floor "
            f"{target_tps * min_ratio:.3f} tok/s"
        )
elif decision == "active":
    errors.append("decision=active but mtp_enabled=false")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(1)

print(
    f"PASS mtp_policy: target_tok_s={target_tps:.3f} mtp_tok_s={mtp_tps:.3f} "
    f"decision={decision} acceptance_permille={spec.get('mtp_acceptance_permille')}"
)
PY
}

require_path "antfly-inference binary" "$antfly_bin"
capture_cuda_environment

if [ "$mode" != "mtp-only" ]; then
  validate_mode="$mode"
  if [ "$mode" = "bench-only" ]; then
    validate_mode="bench-only"
  fi
  turboquant_out="$out_dir/turboquant"
  mkdir -p "$turboquant_out"
  log "RUN turboquant_gate: $validate_mode"
  OUT_DIR="$turboquant_out" \
    E2B_MODEL="$e2b_model" \
    GEMMA12B_Q4_MODEL="$gemma12b_q4_model" \
    REAL_BENCH_MODEL="$e2b_model" \
    LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS:-512}" \
    "$inference_dir/scripts/validate_cuda_turboquant_gemma4.sh" "--$validate_mode" 2>&1 | tee "$out_dir/turboquant_gate.log"
  write_json_table_and_check_speed "$turboquant_out" "$require_speed_thresholds" | tee -a "$summary"
fi

run_e4b_q4k_baseline_if_enabled
run_e4b_qat_gate_if_enabled
run_e4b_qat_long_gate_if_enabled
check_e4b_qat_gate_json_if_present
check_e4b_qat_long_gate_json_if_present
run_e4b_qat_resident_gate_if_enabled
run_resident_gate_if_enabled

case "$run_mtp" in
  auto|required|off)
    ;;
  *)
    log "invalid RUN_MTP=$run_mtp; expected auto|required|off"
    exit 2
    ;;
esac

last_json_file=""
if [ "$run_mtp" != "off" ]; then
  if [ ! -e "$mtp_target_model" ] || [ ! -e "$mtp_draft_model" ]; then
    if [ "$run_mtp" = "required" ]; then
      require_path "MTP target model" "$mtp_target_model"
      require_path "MTP draft model" "$mtp_draft_model"
    fi
    log "SKIP mtp_policy: target or draft model missing"
  else
    run_generate_json mtp_target_only env ANTFLY_INFERENCE_CUDA_TURBOQUANT_MIN_TOKENS="$mtp_turboquant_min_tokens" "$antfly_bin" generate "$mtp_target_model" "$mtp_prompt" \
      --backend cuda \
      --cache-dtype "$mtp_cache_dtype" \
      --max-tokens "$mtp_tokens" \
      --temperature 0 \
      --raw-prompt \
      --no-chat-template \
      --print-token-count \
      --print-timing \
      --combined-budget-mb 16000 \
      --backend-budget-mb 12000 \
      --kv-budget-mb 512 \
      --scratch-budget-mb 512
    target_json="$last_json_file"

    mtp_env=(
      ANTFLY_GEMMA4_MTP_PROFILE=1
      ANTFLY_INFERENCE_CUDA_TURBOQUANT_MIN_TOKENS="$mtp_turboquant_min_tokens"
    )
    if [ "$mtp_target_replay" != "auto" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_TARGET_REPLAY="$mtp_target_replay")
    fi
    if [ "$mtp_replay_context_key" != "auto" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_REPLAY_CONTEXT_KEY="$mtp_replay_context_key")
    fi
    if [ "$mtp_unsafe_target_replay" != "auto" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_UNSAFE_TARGET_REPLAY="$mtp_unsafe_target_replay")
    fi
    if [ "$mtp_assistant_replay" != "auto" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_ASSISTANT_REPLAY="$mtp_assistant_replay")
    fi
    if [ "$mtp_materialize_replay" != "auto" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_MATERIALIZE_REPLAY="$mtp_materialize_replay")
    fi
    if [ "$mtp_capture_persistent_replay" != "auto" ]; then
      mtp_env+=(ANTFLY_INFERENCE_CUDA_CAPTURE_PERSISTENT_REPLAY="$mtp_capture_persistent_replay")
    fi
    if [ -n "$mtp_temp_slot_period" ]; then
      mtp_env+=(ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$mtp_temp_slot_period")
    fi
    if [ -n "$mtp_temp_slot_skip" ]; then
      mtp_env+=(ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP="$mtp_temp_slot_skip")
    fi
    if [ -n "$mtp_position_mode" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_POSITION_MODE="$mtp_position_mode")
    fi
    if [ -n "$mtp_target_hidden_source" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_TARGET_HIDDEN_SOURCE="$mtp_target_hidden_source")
    fi
    if [ -n "$mtp_concat_order" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_CONCAT_ORDER="$mtp_concat_order")
    fi
    if [ -n "$mtp_kv_donor_mode" ]; then
      mtp_env+=(ANTFLY_GEMMA4_MTP_KV_DONOR_MODE="$mtp_kv_donor_mode")
    fi

    run_generate_json mtp_auto_probe env "${mtp_env[@]}" "$antfly_bin" generate "$mtp_target_model" "$mtp_prompt" \
      --backend cuda \
      --draft-model "$mtp_draft_model" \
      --speculative-k "$mtp_speculative_k" \
      --speculation-policy auto \
      --speculation-calibration probe \
      --cache-dtype "$mtp_cache_dtype" \
      --max-tokens "$mtp_tokens" \
      --temperature 0 \
      --raw-prompt \
      --no-chat-template \
      --print-token-count \
      --print-timing \
      --combined-budget-mb 16000 \
      --backend-budget-mb 12000 \
      --kv-budget-mb 512 \
      --scratch-budget-mb 512
    check_mtp_policy "$target_json" "$last_json_file" | tee -a "$summary"
  fi
fi

run_e4b_qat_provider_benchmark
write_qat_production_summary
log "summary: $summary"
log "outputs: $out_dir"
