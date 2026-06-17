#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gliner2_metal_train_tests.sh [mode] [options] [-- extra readiness args]

Runs repeatable GLiNER2 Metal finetuning checks from one stable script entrypoint.

Modes:
  unit                  Focused Metal unit tests touched by training graph work
  suite                 Existing smoke + parity + readiness suite
  train-profile         Production-shaped training-only profile with executor stats
  batch32               Effective batch-size-32 Metal readiness gate with activation checkpointing
  profile              Short production-gate-shaped run with graph op profiling
  diagnostic           Full production-gate-shaped diagnostic, semantic eval skipped
  production           Canonical production gate
  all                  unit, suite, then diagnostic

Options:
  --mode NAME           Same as passing NAME as the first positional arg
  --train-data FILE     Required for train-profile/profile/diagnostic/production unless --dry-run
  --eval-data FILE      Required for profile/diagnostic/production unless --dry-run
  --model-dir DIR       Model directory (default: /private/tmp/termite-models/gliner2)
  --out-suffix NAME     Output suffix base under /private/tmp
  --trace START:END     Trace compiled train/Metal progress for a node range
  --graph-stats         Enable training graph executor stats
  --op-profile          Enable graph op profile timings
  --partition-op-stats  Enable Metal partition command/fallback/output op stats
  --op-runs             Print compact reachable-op counts and contiguous run summary
  --concat-trace        Enable Metal concat_prim device/fallback trace diagnostics
  --reduce-trace        Enable Metal reduce_sum/reduce_mean device/fallback trace diagnostics
  --broadcast-trace     Enable Metal broadcast_in_dim device/fallback trace diagnostics
  --skip-semantic-eval  Skip semantic eval in production mode too
  --batch-size N        Training batch size for train-profile/batch32/profile gates
  --seq-len N           Sequence length for train-profile/batch32/profile gates
  --max-examples N      Max train examples for train-profile/batch32/profile gates
  --max-metal-runtime-total-bytes N
                        Fail readiness if Metal runtime allocation snapshot exceeds N
  --max-metal-tensor-device-owned-peak-live-bytes N
                        Fail readiness if owned Metal device tensor peak exceeds N
  --max-metal-eager-arena-peak-bytes N
                        Fail readiness if Metal eager arena peak exceeds N
  --max-metal-eager-arena-spill-bytes N
                        Fail readiness if Metal eager arena spills exceed N
  --max-metal-chunk-local-output-peak-bytes N
                        Fail readiness if chunk-local output pool peak exceeds N
  --max-metal-chunk-local-output-spill-bytes N
                        Fail readiness if chunk-local output spills exceed N
  --max-metal-chunk-local-output-unconsumed-hints N
                        Fail readiness if output hints are allocated but unused
  --min-metal-chunk-local-output-consumed-hints N
                        Fail readiness unless chunk-local output hints are consumed
  TERMITE_METAL_CHUNK_LOCAL_OUTPUTS=1
                        Enable chunk-local command-output backing storage; batch32
                        asserts allocations, resets, and freed bytes when set
  --min-metal-runtime-reuse-hit-count N
                        Fail readiness unless in-frame buffer reuse hits at least N
  --max-graph-command-dispatch-count N
                        Fail readiness if graph command dispatches exceed N
  --max-graph-host-output-count N
                        Fail readiness if graph host outputs exceed N
  --max-metal-frame-gpu-ms N
                        Fail readiness if any Metal frame GPU duration exceeds N ms
  --max-metal-last-frame-compute-encoder-count N
                        Fail readiness if the last Metal frame compute-encoder count exceeds N
  --min-metal-frame-chunk-boundary-count N
                        Fail readiness unless Metal frame chunk boundaries execute at least N times
  --min-metal-frame-chunk-promoted-value-count N
                        Fail readiness unless chunk boundaries promote at least N live values
  --min-metal-frame-chunk-swept-value-count N
                        Fail readiness unless chunk boundaries sweep at least N expired values
  --min-graph-runtime-region-dispatch-count N
                        Fail readiness unless graph runtime regions execute at least N times
  --max-graph-runtime-region-fallback-count N
                        Fail readiness if graph runtime-region fallbacks exceed N
  --min-graph-runtime-region-elided-node-count N
                        Fail readiness unless runtime regions elide at least N graph nodes
  --min-metal-deberta-ffn-forward-region-count N
                        Fail readiness unless graph-level DeBERTa FFN forward regions run at least N times
  --min-metal-deberta-encoder-lora-layer-region-count N
                        Fail readiness unless LoRA-aware DeBERTa encoder-layer regions run at least N times
  --min-metal-deberta-encoder-lora-residual-layernorm-region-count N
                        Fail readiness unless encoder-layer regions include residual+LayerNorm at least N times
  --max-metal-deberta-encoder-lora-layer-scaffold-count N
                        Fail readiness if scaffold-only encoder-layer regions exceed N
  --max-metal-deberta-encoder-lora-layer-fallback-count N
                        Fail readiness if LoRA-aware DeBERTa encoder-layer fallbacks exceed N
  --min-metal-deberta-attention-flash-call-count N
                        Fail readiness unless fused DeBERTa attention runs at least N times
  --max-metal-deberta-attention-gemm-fallback-count N
                        Fail readiness if fused DeBERTa attention GEMM fallbacks exceed N
  --min-metal-deberta-encoder-layer-success-count N
                        Fail readiness unless DeBERTa encoder-layer runtime calls succeed at least N times
  --min-metal-deberta-ffn-fused-call-count N
                        Fail readiness unless fused DeBERTa FFN runs at least N times
  --max-metal-deberta-ffn-fused-fallback-count N
                        Fail readiness if fused DeBERTa FFN fallbacks exceed N
  --max-runtime-frame-ineligible-missing-model-metadata N
                        Fail readiness if runtime-frame candidates are blocked by missing model metadata more than N times
  --max-commands N      train-profile graph-exec command ceiling (default: 6200)
  --max-host-outputs N  train-profile host-output ceiling (default: 500)
  --max-runtime-region-fallbacks N
                        train-profile runtime-region fallback ceiling (default: 0)
  --min-graph-regions N train-profile minimum graph regions (default: 1)
  --min-runtime-plan-dispatches N
                        train-profile minimum runtime plan dispatches (default: 1)
  --dry-run             Print commands without running them
  --help                Show this help

Examples:
  scripts/run_gliner2_metal_train_tests.sh unit
  scripts/run_gliner2_metal_train_tests.sh suite
  scripts/run_gliner2_metal_train_tests.sh train-profile \
    --train-data /private/tmp/termite-gliner2-production-diagnostic/train.jsonl
  scripts/run_gliner2_metal_train_tests.sh batch32 \
    --train-data /private/tmp/termite-gliner2-production-diagnostic/train.jsonl \
    --eval-data /private/tmp/termite-gliner2-production-diagnostic/eval.jsonl
  scripts/run_gliner2_metal_train_tests.sh profile \
    --train-data /private/tmp/termite-gliner2-production-diagnostic/train.jsonl \
    --eval-data /private/tmp/termite-gliner2-production-diagnostic/eval.jsonl
  scripts/run_gliner2_metal_train_tests.sh diagnostic \
    --train-data /private/tmp/termite-gliner2-production-diagnostic/train.jsonl \
    --eval-data /private/tmp/termite-gliner2-production-diagnostic/eval.jsonl
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd "${script_dir}/.." && pwd)"
parity_script="${script_dir}/run_gliner2_metal_train_parity.sh"

mode="suite"
model_dir="/private/tmp/termite-models/gliner2"
train_data=""
eval_data=""
out_suffix=""
trace_range=""
graph_stats=0
op_profile=0
partition_op_stats=0
op_runs=0
concat_trace=0
reduce_trace=0
broadcast_trace=0
skip_semantic_eval=0
dry_run=0
batch_size=1
seq_len=32
max_examples=5
batch_size_explicit=0
seq_len_explicit=0
max_examples_explicit=0
max_metal_runtime_total_bytes=""
min_metal_runtime_reuse_hit_count=""
max_commands=6200
max_host_outputs=500
max_runtime_region_fallbacks=0
min_graph_regions=1
min_runtime_plan_dispatches=1
max_graph_command_dispatch_count=""
max_graph_host_output_count=""
max_metal_frame_gpu_ms=""
max_metal_last_frame_compute_encoder_count=""
max_metal_tensor_device_owned_peak_live_bytes=""
max_metal_eager_arena_peak_bytes=""
max_metal_eager_arena_spill_bytes=""
max_metal_chunk_local_output_peak_bytes=""
max_metal_chunk_local_output_spill_bytes=""
max_metal_chunk_local_output_unconsumed_hints=""
min_metal_chunk_local_output_consumed_hints=""
min_metal_frame_chunk_boundary_count=""
min_metal_frame_chunk_promoted_value_count=""
min_metal_frame_chunk_swept_value_count=""
min_graph_runtime_region_dispatch_count=""
max_graph_runtime_region_fallback_count=""
min_graph_runtime_region_elided_node_count=""
min_metal_deberta_ffn_forward_region_count=""
min_metal_deberta_encoder_lora_layer_region_count=""
min_metal_deberta_encoder_lora_residual_layernorm_region_count=""
max_metal_deberta_encoder_lora_layer_scaffold_count=""
max_metal_deberta_encoder_lora_layer_fallback_count=""
min_metal_deberta_attention_flash_call_count=""
max_metal_deberta_attention_gemm_fallback_count=""
min_metal_deberta_encoder_layer_success_count=""
min_metal_deberta_ffn_fused_call_count=""
max_metal_deberta_ffn_fused_fallback_count=""
max_runtime_frame_ineligible_missing_model_metadata=""
require_slot_bound_outputs=0
require_eager_arena_outputs=0
require_chunk_local_outputs=0
extra_args=()

is_mode() {
  case "$1" in
    unit|suite|train-profile|batch32|profile|diagnostic|production|all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:?missing value for --mode}"
      shift 2
      ;;
    --train-data)
      train_data="${2:?missing value for --train-data}"
      shift 2
      ;;
    --eval-data)
      eval_data="${2:?missing value for --eval-data}"
      shift 2
      ;;
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --out-suffix)
      out_suffix="${2:?missing value for --out-suffix}"
      shift 2
      ;;
    --trace)
      trace_range="${2:?missing value for --trace}"
      shift 2
      ;;
    --graph-stats)
      graph_stats=1
      shift
      ;;
    --op-profile)
      op_profile=1
      shift
      ;;
    --partition-op-stats)
      partition_op_stats=1
      shift
      ;;
    --op-runs)
      op_runs=1
      shift
      ;;
    --concat-trace)
      concat_trace=1
      shift
      ;;
    --reduce-trace)
      reduce_trace=1
      shift
      ;;
    --broadcast-trace)
      broadcast_trace=1
      shift
      ;;
    --skip-semantic-eval)
      skip_semantic_eval=1
      shift
      ;;
    --batch-size)
      batch_size="${2:?missing value for --batch-size}"
      batch_size_explicit=1
      shift 2
      ;;
    --seq-len)
      seq_len="${2:?missing value for --seq-len}"
      seq_len_explicit=1
      shift 2
      ;;
    --max-examples)
      max_examples="${2:?missing value for --max-examples}"
      max_examples_explicit=1
      shift 2
      ;;
    --max-metal-runtime-total-bytes)
      max_metal_runtime_total_bytes="${2:?missing value for --max-metal-runtime-total-bytes}"
      shift 2
      ;;
    --max-metal-tensor-device-owned-peak-live-bytes)
      max_metal_tensor_device_owned_peak_live_bytes="${2:?missing value for --max-metal-tensor-device-owned-peak-live-bytes}"
      shift 2
      ;;
    --max-metal-eager-arena-peak-bytes)
      max_metal_eager_arena_peak_bytes="${2:?missing value for --max-metal-eager-arena-peak-bytes}"
      shift 2
      ;;
    --max-metal-eager-arena-spill-bytes)
      max_metal_eager_arena_spill_bytes="${2:?missing value for --max-metal-eager-arena-spill-bytes}"
      shift 2
      ;;
    --max-metal-chunk-local-output-peak-bytes)
      max_metal_chunk_local_output_peak_bytes="${2:?missing value for --max-metal-chunk-local-output-peak-bytes}"
      shift 2
      ;;
    --max-metal-chunk-local-output-spill-bytes)
      max_metal_chunk_local_output_spill_bytes="${2:?missing value for --max-metal-chunk-local-output-spill-bytes}"
      shift 2
      ;;
    --max-metal-chunk-local-output-unconsumed-hints)
      max_metal_chunk_local_output_unconsumed_hints="${2:?missing value for --max-metal-chunk-local-output-unconsumed-hints}"
      shift 2
      ;;
    --min-metal-chunk-local-output-consumed-hints)
      min_metal_chunk_local_output_consumed_hints="${2:?missing value for --min-metal-chunk-local-output-consumed-hints}"
      shift 2
      ;;
    --min-metal-runtime-reuse-hit-count)
      min_metal_runtime_reuse_hit_count="${2:?missing value for --min-metal-runtime-reuse-hit-count}"
      shift 2
      ;;
    --max-graph-command-dispatch-count)
      max_graph_command_dispatch_count="${2:?missing value for --max-graph-command-dispatch-count}"
      shift 2
      ;;
    --max-graph-host-output-count)
      max_graph_host_output_count="${2:?missing value for --max-graph-host-output-count}"
      shift 2
      ;;
    --max-metal-frame-gpu-ms)
      max_metal_frame_gpu_ms="${2:?missing value for --max-metal-frame-gpu-ms}"
      shift 2
      ;;
    --max-metal-last-frame-compute-encoder-count)
      max_metal_last_frame_compute_encoder_count="${2:?missing value for --max-metal-last-frame-compute-encoder-count}"
      shift 2
      ;;
    --min-metal-frame-chunk-boundary-count)
      min_metal_frame_chunk_boundary_count="${2:?missing value for --min-metal-frame-chunk-boundary-count}"
      shift 2
      ;;
    --min-metal-frame-chunk-promoted-value-count)
      min_metal_frame_chunk_promoted_value_count="${2:?missing value for --min-metal-frame-chunk-promoted-value-count}"
      shift 2
      ;;
    --min-metal-frame-chunk-swept-value-count)
      min_metal_frame_chunk_swept_value_count="${2:?missing value for --min-metal-frame-chunk-swept-value-count}"
      shift 2
      ;;
    --min-graph-runtime-region-dispatch-count)
      min_graph_runtime_region_dispatch_count="${2:?missing value for --min-graph-runtime-region-dispatch-count}"
      shift 2
      ;;
    --max-graph-runtime-region-fallback-count)
      max_graph_runtime_region_fallback_count="${2:?missing value for --max-graph-runtime-region-fallback-count}"
      shift 2
      ;;
    --min-graph-runtime-region-elided-node-count)
      min_graph_runtime_region_elided_node_count="${2:?missing value for --min-graph-runtime-region-elided-node-count}"
      shift 2
      ;;
    --min-metal-deberta-ffn-forward-region-count)
      min_metal_deberta_ffn_forward_region_count="${2:?missing value for --min-metal-deberta-ffn-forward-region-count}"
      shift 2
      ;;
    --min-metal-deberta-encoder-lora-layer-region-count)
      min_metal_deberta_encoder_lora_layer_region_count="${2:?missing value for --min-metal-deberta-encoder-lora-layer-region-count}"
      shift 2
      ;;
    --min-metal-deberta-encoder-lora-residual-layernorm-region-count)
      min_metal_deberta_encoder_lora_residual_layernorm_region_count="${2:?missing value for --min-metal-deberta-encoder-lora-residual-layernorm-region-count}"
      shift 2
      ;;
    --max-metal-deberta-encoder-lora-layer-scaffold-count)
      max_metal_deberta_encoder_lora_layer_scaffold_count="${2:?missing value for --max-metal-deberta-encoder-lora-layer-scaffold-count}"
      shift 2
      ;;
    --max-metal-deberta-encoder-lora-layer-fallback-count)
      max_metal_deberta_encoder_lora_layer_fallback_count="${2:?missing value for --max-metal-deberta-encoder-lora-layer-fallback-count}"
      shift 2
      ;;
    --min-metal-deberta-attention-flash-call-count)
      min_metal_deberta_attention_flash_call_count="${2:?missing value for --min-metal-deberta-attention-flash-call-count}"
      shift 2
      ;;
    --max-metal-deberta-attention-gemm-fallback-count)
      max_metal_deberta_attention_gemm_fallback_count="${2:?missing value for --max-metal-deberta-attention-gemm-fallback-count}"
      shift 2
      ;;
    --min-metal-deberta-encoder-layer-success-count)
      min_metal_deberta_encoder_layer_success_count="${2:?missing value for --min-metal-deberta-encoder-layer-success-count}"
      shift 2
      ;;
    --min-metal-deberta-ffn-fused-call-count)
      min_metal_deberta_ffn_fused_call_count="${2:?missing value for --min-metal-deberta-ffn-fused-call-count}"
      shift 2
      ;;
    --max-metal-deberta-ffn-fused-fallback-count)
      max_metal_deberta_ffn_fused_fallback_count="${2:?missing value for --max-metal-deberta-ffn-fused-fallback-count}"
      shift 2
      ;;
    --max-runtime-frame-ineligible-missing-model-metadata)
      max_runtime_frame_ineligible_missing_model_metadata="${2:?missing value for --max-runtime-frame-ineligible-missing-model-metadata}"
      shift 2
      ;;
    --max-commands)
      max_commands="${2:?missing value for --max-commands}"
      shift 2
      ;;
    --max-host-outputs)
      max_host_outputs="${2:?missing value for --max-host-outputs}"
      shift 2
      ;;
    --max-runtime-region-fallbacks)
      max_runtime_region_fallbacks="${2:?missing value for --max-runtime-region-fallbacks}"
      shift 2
      ;;
    --min-graph-regions)
      min_graph_regions="${2:?missing value for --min-graph-regions}"
      shift 2
      ;;
    --min-runtime-plan-dispatches)
      min_runtime_plan_dispatches="${2:?missing value for --min-runtime-plan-dispatches}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      if is_mode "$1"; then
        mode="$1"
        shift
      else
        extra_args+=("$1")
        shift
      fi
      ;;
  esac
done

if ! is_mode "${mode}"; then
  echo "error: unsupported mode '${mode}'" >&2
  usage >&2
  exit 2
fi

if [[ -n "${trace_range}" && ! "${trace_range}" =~ ^[0-9]+:[0-9]+$ ]]; then
  echo "error: --trace must be START:END" >&2
  exit 2
fi

quote_command() {
  printf 'cmd'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  quote_command "$@"
  if [[ "${dry_run}" -eq 0 ]]; then
    "$@"
  fi
}

run_env_cmd() {
  local -a local_env=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --)
        shift
        break
        ;;
      *)
        local_env+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${#local_env[@]}" -gt 0 ]]; then
    printf 'env'
    printf ' %q' "${local_env[@]}"
    printf '\n'
    quote_command "$@"
    if [[ "${dry_run}" -eq 0 ]]; then
      env "${local_env[@]}" "$@"
    fi
  else
    run_cmd "$@"
  fi
}

profile_env_args() {
  local -a local_env=()
  if [[ "${graph_stats}" -eq 1 ]]; then
    local_env+=("TERMITE_GRAPH_EXECUTOR_STATS=1")
  fi
  if [[ "${op_profile}" -eq 1 ]]; then
    local_env+=("TERMITE_GRAPH_OP_PROFILE=1")
  fi
  if [[ "${partition_op_stats}" -eq 1 ]]; then
    local_env+=("TERMITE_METAL_PARTITION_OP_STATS=1")
  fi
  if [[ "${op_runs}" -eq 1 ]]; then
    local_env+=("TERMITE_METAL_PARTITION_OP_RUNS=1")
  fi
  if [[ "${concat_trace}" -eq 1 ]]; then
    local_env+=("TERMITE_METAL_TRACE_CONCAT_PRIM=1")
  fi
  if [[ "${reduce_trace}" -eq 1 ]]; then
    local_env+=("TERMITE_METAL_TRACE_REDUCE_PRIM=1")
  fi
  if [[ "${broadcast_trace}" -eq 1 ]]; then
    local_env+=("TERMITE_METAL_TRACE_BROADCAST_PRIM=1")
  fi
  if [[ -n "${TERMITE_METAL_TRACE_OWNED_ALLOC_LIMIT:-}" ]]; then
    local_env+=("TERMITE_METAL_TRACE_OWNED_ALLOC_LIMIT=${TERMITE_METAL_TRACE_OWNED_ALLOC_LIMIT}")
  fi
  if [[ -n "${TERMITE_METAL_TRACE_OWNED_ALLOC_MIN_BYTES:-}" ]]; then
    local_env+=("TERMITE_METAL_TRACE_OWNED_ALLOC_MIN_BYTES=${TERMITE_METAL_TRACE_OWNED_ALLOC_MIN_BYTES}")
  fi
  if [[ -n "${TERMITE_METAL_OWNED_PEAK_SNAPSHOT_TOP:-}" ]]; then
    local_env+=("TERMITE_METAL_OWNED_PEAK_SNAPSHOT_TOP=${TERMITE_METAL_OWNED_PEAK_SNAPSHOT_TOP}")
  fi
  if [[ -n "${TERMITE_METAL_OWNED_PEAK_SNAPSHOT_LIMIT:-}" ]]; then
    local_env+=("TERMITE_METAL_OWNED_PEAK_SNAPSHOT_LIMIT=${TERMITE_METAL_OWNED_PEAK_SNAPSHOT_LIMIT}")
  fi
  if [[ -n "${TERMITE_METAL_OWNED_PEAK_SNAPSHOT_MIN_BYTES:-}" ]]; then
    local_env+=("TERMITE_METAL_OWNED_PEAK_SNAPSHOT_MIN_BYTES=${TERMITE_METAL_OWNED_PEAK_SNAPSHOT_MIN_BYTES}")
  fi
  if [[ "${#local_env[@]}" -gt 0 ]]; then
    printf '%s\n' "${local_env[@]}"
  fi
}

append_profile_env_args() {
  local env_arg
  while IFS= read -r env_arg; do
    if [[ -n "${env_arg}" ]]; then
      profile_env_result+=("${env_arg}")
    fi
  done < <(profile_env_args)
}

append_metal_runtime_threshold_args() {
  if [[ -n "${max_metal_runtime_total_bytes}" ]]; then
    cmd+=("--max-metal-runtime-total-bytes" "${max_metal_runtime_total_bytes}")
  fi
  if [[ -n "${max_metal_tensor_device_owned_peak_live_bytes}" ]]; then
    cmd+=("--max-metal-tensor-device-owned-peak-live-bytes" "${max_metal_tensor_device_owned_peak_live_bytes}")
  fi
  if [[ -n "${max_metal_eager_arena_peak_bytes}" ]]; then
    cmd+=("--max-metal-eager-arena-peak-bytes" "${max_metal_eager_arena_peak_bytes}")
  fi
  if [[ -n "${max_metal_eager_arena_spill_bytes}" ]]; then
    cmd+=("--max-metal-eager-arena-spill-bytes" "${max_metal_eager_arena_spill_bytes}")
  fi
  if [[ -n "${max_metal_chunk_local_output_peak_bytes}" ]]; then
    cmd+=("--max-metal-chunk-local-output-peak-bytes" "${max_metal_chunk_local_output_peak_bytes}")
  fi
  if [[ -n "${max_metal_chunk_local_output_spill_bytes}" ]]; then
    cmd+=("--max-metal-chunk-local-output-spill-bytes" "${max_metal_chunk_local_output_spill_bytes}")
  fi
  if [[ -n "${max_metal_chunk_local_output_unconsumed_hints}" ]]; then
    cmd+=("--max-metal-chunk-local-output-unconsumed-hints" "${max_metal_chunk_local_output_unconsumed_hints}")
  fi
  if [[ -n "${min_metal_chunk_local_output_consumed_hints}" ]]; then
    cmd+=("--min-metal-chunk-local-output-consumed-hints" "${min_metal_chunk_local_output_consumed_hints}")
  fi
  if [[ -n "${min_metal_runtime_reuse_hit_count}" ]]; then
    cmd+=("--min-metal-runtime-reuse-hit-count" "${min_metal_runtime_reuse_hit_count}")
  fi
  if [[ -n "${max_graph_command_dispatch_count}" ]]; then
    cmd+=("--max-graph-command-dispatch-count" "${max_graph_command_dispatch_count}")
  fi
  if [[ -n "${max_graph_host_output_count}" ]]; then
    cmd+=("--max-graph-host-output-count" "${max_graph_host_output_count}")
  fi
  if [[ -n "${max_metal_frame_gpu_ms}" ]]; then
    cmd+=("--max-metal-frame-gpu-ms" "${max_metal_frame_gpu_ms}")
  fi
  if [[ -n "${max_metal_last_frame_compute_encoder_count}" ]]; then
    cmd+=("--max-metal-last-frame-compute-encoder-count" "${max_metal_last_frame_compute_encoder_count}")
  fi
  if [[ -n "${min_metal_frame_chunk_boundary_count}" ]]; then
    cmd+=("--min-metal-frame-chunk-boundary-count" "${min_metal_frame_chunk_boundary_count}")
  fi
  if [[ -n "${min_metal_frame_chunk_promoted_value_count}" ]]; then
    cmd+=("--min-metal-frame-chunk-promoted-value-count" "${min_metal_frame_chunk_promoted_value_count}")
  fi
  if [[ -n "${min_metal_frame_chunk_swept_value_count}" ]]; then
    cmd+=("--min-metal-frame-chunk-swept-value-count" "${min_metal_frame_chunk_swept_value_count}")
  fi
  if [[ -n "${min_graph_runtime_region_dispatch_count}" ]]; then
    cmd+=("--min-graph-runtime-region-dispatch-count" "${min_graph_runtime_region_dispatch_count}")
  fi
  if [[ -n "${max_graph_runtime_region_fallback_count}" ]]; then
    cmd+=("--max-graph-runtime-region-fallback-count" "${max_graph_runtime_region_fallback_count}")
  fi
  if [[ -n "${min_graph_runtime_region_elided_node_count}" ]]; then
    cmd+=("--min-graph-runtime-region-elided-node-count" "${min_graph_runtime_region_elided_node_count}")
  fi
  if [[ -n "${min_metal_deberta_ffn_forward_region_count}" ]]; then
    cmd+=("--min-metal-deberta-ffn-forward-region-count" "${min_metal_deberta_ffn_forward_region_count}")
  fi
  if [[ -n "${min_metal_deberta_encoder_lora_layer_region_count}" ]]; then
    cmd+=("--min-metal-deberta-encoder-lora-layer-region-count" "${min_metal_deberta_encoder_lora_layer_region_count}")
  fi
  if [[ -n "${min_metal_deberta_encoder_lora_residual_layernorm_region_count}" ]]; then
    cmd+=("--min-metal-deberta-encoder-lora-residual-layernorm-region-count" "${min_metal_deberta_encoder_lora_residual_layernorm_region_count}")
  fi
  if [[ -n "${max_metal_deberta_encoder_lora_layer_scaffold_count}" ]]; then
    cmd+=("--max-metal-deberta-encoder-lora-layer-scaffold-count" "${max_metal_deberta_encoder_lora_layer_scaffold_count}")
  fi
  if [[ -n "${max_metal_deberta_encoder_lora_layer_fallback_count}" ]]; then
    cmd+=("--max-metal-deberta-encoder-lora-layer-fallback-count" "${max_metal_deberta_encoder_lora_layer_fallback_count}")
  fi
  if [[ -n "${min_metal_deberta_attention_flash_call_count}" ]]; then
    cmd+=("--min-metal-deberta-attention-flash-call-count" "${min_metal_deberta_attention_flash_call_count}")
  fi
  if [[ -n "${max_metal_deberta_attention_gemm_fallback_count}" ]]; then
    cmd+=("--max-metal-deberta-attention-gemm-fallback-count" "${max_metal_deberta_attention_gemm_fallback_count}")
  fi
  if [[ -n "${min_metal_deberta_encoder_layer_success_count}" ]]; then
    cmd+=("--min-metal-deberta-encoder-layer-success-count" "${min_metal_deberta_encoder_layer_success_count}")
  fi
  if [[ -n "${min_metal_deberta_ffn_fused_call_count}" ]]; then
    cmd+=("--min-metal-deberta-ffn-fused-call-count" "${min_metal_deberta_ffn_fused_call_count}")
  fi
  if [[ -n "${max_metal_deberta_ffn_fused_fallback_count}" ]]; then
    cmd+=("--max-metal-deberta-ffn-fused-fallback-count" "${max_metal_deberta_ffn_fused_fallback_count}")
  fi
  if [[ -n "${max_runtime_frame_ineligible_missing_model_metadata}" ]]; then
    cmd+=("--max-runtime-frame-ineligible-missing-model-metadata" "${max_runtime_frame_ineligible_missing_model_metadata}")
  fi
}

require_data() {
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi
  if [[ -z "${train_data}" || -z "${eval_data}" ]]; then
    echo "error: ${mode} requires --train-data and --eval-data" >&2
    exit 2
  fi
}

require_train_data() {
  if [[ "${dry_run}" -eq 1 ]]; then
    return 0
  fi
  if [[ -z "${train_data}" ]]; then
    echo "error: ${mode} requires --train-data" >&2
    exit 2
  fi
}

suffix_arg() {
  local default_suffix="$1"
  if [[ -n "${out_suffix}" ]]; then
    printf '%s\n' "${out_suffix}"
  else
    printf '%s\n' "${default_suffix}"
  fi
}

stat_value() {
  local line="$1"
  local key="$2"
  if [[ "${line}" =~ (^|[[:space:]])${key}=([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
    return 0
  fi
  return 1
}

json_log_value() {
  local log_file="$1"
  local key="$2"
  local value
  value="$(sed -nE "s/^[[:space:]]*\"${key}\":[[:space:]]*\"?([^\",}]+)\"?.*$/\\1/p" "${log_file}" | tail -n 1)"
  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
    return 0
  fi
  return 1
}

assert_production_summary_log() {
  local log_file="$1"
  local status manifest_backend optimizer_backend optimizer_mismatch trainable_transfers resident_transfers adapter_tensors task_head_tensors steps supervised_tokens execute_ms graph_commands graph_planned owned_peak_bytes metal_gpu_ms metal_compute_encoders chunk_boundaries chunk_promoted chunk_swept runtime_regions runtime_region_fallbacks runtime_region_elided_nodes deberta_ffn_forward_regions deberta_lora_layer_regions deberta_lora_layer_fallbacks deberta_flash_calls deberta_gemm_fallbacks deberta_ffn_fused_calls deberta_ffn_fused_fallbacks

  status="$(json_log_value "${log_file}" "status")" || return 1
  [[ "${status}" == "passed" ]] || return 1

  manifest_backend="$(json_log_value "${log_file}" "manifest_backend")" || return 1
  optimizer_backend="$(json_log_value "${log_file}" "optimizer_backend")" || return 1
  optimizer_mismatch="$(json_log_value "${log_file}" "optimizer_backend_mismatch")" || return 1
  trainable_transfers="$(json_log_value "${log_file}" "max_device_trainable_transfer_count")" || return 1
  resident_transfers="$(json_log_value "${log_file}" "max_device_resident_transfer_count")" || return 1
  adapter_tensors="$(json_log_value "${log_file}" "peft_adapter_tensor_count")" || return 1
  task_head_tensors="$(json_log_value "${log_file}" "task_head_tensor_count")" || return 1
  steps="$(json_log_value "${log_file}" "step_record_count")" || return 1
  supervised_tokens="$(json_log_value "${log_file}" "supervised_token_count")" || return 1
  execute_ms="$(json_log_value "${log_file}" "total_execute_ms")" || return 1
  graph_commands="$(json_log_value "${log_file}" "total_graph_command_dispatches")" || return 1
  graph_planned="$(json_log_value "${log_file}" "total_graph_planned_dispatches")" || return 1
  owned_peak_bytes="$(json_log_value "${log_file}" "max_metal_tensor_device_owned_peak_live_bytes")" || return 1
  metal_gpu_ms="$(json_log_value "${log_file}" "max_metal_frame_gpu_ms")" || return 1
  metal_compute_encoders="$(json_log_value "${log_file}" "max_metal_last_frame_compute_encoders")" || return 1
  chunk_boundaries="$(json_log_value "${log_file}" "total_metal_frame_chunk_boundaries")" || return 1
  chunk_promoted="$(json_log_value "${log_file}" "total_metal_frame_chunk_promoted_values")" || return 1
  chunk_swept="$(json_log_value "${log_file}" "total_metal_frame_chunk_swept_values")" || return 1
  runtime_regions="$(json_log_value "${log_file}" "total_graph_runtime_region_dispatches")" || return 1
  runtime_region_fallbacks="$(json_log_value "${log_file}" "total_graph_runtime_region_fallbacks")" || return 1
  runtime_region_elided_nodes="$(json_log_value "${log_file}" "total_graph_runtime_region_elided_nodes")" || return 1
  deberta_ffn_forward_regions="$(json_log_value "${log_file}" "total_metal_deberta_ffn_forward_regions")" || return 1
  deberta_lora_layer_regions="$(json_log_value "${log_file}" "total_metal_deberta_encoder_lora_layer_regions")" || return 1
  deberta_lora_residual_layernorm_regions="$(json_log_value "${log_file}" "total_metal_deberta_encoder_lora_residual_layernorm_regions")" || return 1
  deberta_lora_layer_scaffold_regions="$(json_log_value "${log_file}" "total_metal_deberta_encoder_lora_layer_scaffold_regions")" || return 1
  deberta_lora_layer_fallbacks="$(json_log_value "${log_file}" "total_metal_deberta_encoder_lora_layer_fallbacks")" || return 1
  deberta_flash_calls="$(json_log_value "${log_file}" "total_metal_deberta_attention_flash_calls")" || return 1
  deberta_gemm_fallbacks="$(json_log_value "${log_file}" "total_metal_deberta_attention_gemm_fallbacks")" || return 1
  deberta_ffn_fused_calls="$(json_log_value "${log_file}" "total_metal_deberta_ffn_fused_calls")" || return 1
  deberta_ffn_fused_fallbacks="$(json_log_value "${log_file}" "total_metal_deberta_ffn_fused_fallbacks")" || return 1

  if [[ "${manifest_backend}" != "Metal" ||
    "${optimizer_backend}" != "metal" ||
    "${optimizer_mismatch}" != "false" ||
    "${trainable_transfers}" != "0" ||
    "${resident_transfers}" != "0" ]]; then
    echo "error: production profile summary did not prove resident Metal training" >&2
    echo "       manifest_backend=${manifest_backend} optimizer_backend=${optimizer_backend} optimizer_backend_mismatch=${optimizer_mismatch} max_device_trainable_transfer_count=${trainable_transfers} max_device_resident_transfer_count=${resident_transfers}" >&2
    return 1
  fi

  if (( adapter_tensors <= 0 || task_head_tensors <= 0 || steps <= 0 || supervised_tokens <= 0 )); then
    echo "error: production profile summary is missing required training artifacts or metrics" >&2
    echo "       peft_adapter_tensor_count=${adapter_tensors} task_head_tensor_count=${task_head_tensors} step_record_count=${steps} supervised_token_count=${supervised_tokens}" >&2
    return 1
  fi

  if (( graph_commands == 0 && graph_planned == 0 )); then
    echo "error: production profile summary reported zero graph executor command+planned dispatches (interpreter-only execution; check for silent graph executor fallbacks)" >&2
    echo "       total_graph_command_dispatches=${graph_commands} total_graph_planned_dispatches=${graph_planned}" >&2
    return 1
  fi

  printf 'production_profile_summary_assertions: status=%s manifest_backend=%s optimizer_backend=%s max_device_trainable_transfer_count=%s max_device_resident_transfer_count=%s peft_adapter_tensor_count=%s task_head_tensor_count=%s step_record_count=%s supervised_token_count=%s total_execute_ms=%s total_graph_command_dispatches=%s total_graph_planned_dispatches=%s max_metal_tensor_device_owned_peak_live_bytes=%s max_metal_frame_gpu_ms=%s max_metal_last_frame_compute_encoders=%s total_metal_frame_chunk_boundaries=%s total_metal_frame_chunk_promoted_values=%s total_metal_frame_chunk_swept_values=%s total_graph_runtime_region_dispatches=%s total_graph_runtime_region_fallbacks=%s total_graph_runtime_region_elided_nodes=%s total_metal_deberta_ffn_forward_regions=%s total_metal_deberta_encoder_lora_layer_regions=%s total_metal_deberta_encoder_lora_residual_layernorm_regions=%s total_metal_deberta_encoder_lora_layer_scaffold_regions=%s total_metal_deberta_encoder_lora_layer_fallbacks=%s total_metal_deberta_attention_flash_calls=%s total_metal_deberta_attention_gemm_fallbacks=%s total_metal_deberta_ffn_fused_calls=%s total_metal_deberta_ffn_fused_fallbacks=%s\n' \
    "${status}" "${manifest_backend}" "${optimizer_backend}" "${trainable_transfers}" "${resident_transfers}" "${adapter_tensors}" "${task_head_tensors}" "${steps}" "${supervised_tokens}" "${execute_ms}" "${graph_commands}" "${graph_planned}" "${owned_peak_bytes}" "${metal_gpu_ms}" "${metal_compute_encoders}" "${chunk_boundaries}" "${chunk_promoted}" "${chunk_swept}" "${runtime_regions}" "${runtime_region_fallbacks}" "${runtime_region_elided_nodes}" "${deberta_ffn_forward_regions}" "${deberta_lora_layer_regions}" "${deberta_lora_residual_layernorm_regions}" "${deberta_lora_layer_scaffold_regions}" "${deberta_lora_layer_fallbacks}" "${deberta_flash_calls}" "${deberta_gemm_fallbacks}" "${deberta_ffn_fused_calls}" "${deberta_ffn_fused_fallbacks}"
}

assert_graph_exec_profile_log() {
  local log_file="$1"
  local stats_line=""
  local matched_line=""
  local commands host_outputs graph_regions runtime_plan_dispatches runtime_region_fallbacks
  while IFS= read -r line; do
    [[ "${line}" == *graph_executor_stats:* ]] || continue
    line="graph_executor_stats:${line#*graph_executor_stats:}"
    stats_line="${line}"
    commands="$(stat_value "${line}" commands)" || continue
    host_outputs="$(stat_value "${line}" host_outputs)" || continue
    graph_regions="$(stat_value "${line}" graph_regions)" || continue
    runtime_plan_dispatches="$(stat_value "${line}" runtime_plan_dispatches)" || continue
    runtime_region_fallbacks="$(stat_value "${line}" runtime_region_fallbacks)" || continue

    if (( commands <= max_commands &&
      host_outputs <= max_host_outputs &&
      graph_regions >= min_graph_regions &&
      runtime_plan_dispatches >= min_runtime_plan_dispatches &&
      runtime_region_fallbacks <= max_runtime_region_fallbacks )); then
      matched_line="${line}"
      break
    fi
  done < "${log_file}"

  if [[ -z "${stats_line}" ]]; then
    if assert_production_summary_log "${log_file}"; then
      return 0
    fi
    echo "error: profile did not emit graph_executor_stats or a passing production readiness summary" >&2
    return 1
  fi

  if [[ -z "${matched_line}" ]]; then
    echo "error: no graph_executor_stats line satisfied training graph-exec assertions" >&2
    echo "       max_commands=${max_commands} max_host_outputs=${max_host_outputs} min_graph_regions=${min_graph_regions} min_runtime_plan_dispatches=${min_runtime_plan_dispatches} max_runtime_region_fallbacks=${max_runtime_region_fallbacks}" >&2
    echo "       last_stats: ${stats_line}" >&2
    return 1
  fi

  commands="$(stat_value "${matched_line}" commands)"
  host_outputs="$(stat_value "${matched_line}" host_outputs)"
  graph_regions="$(stat_value "${matched_line}" graph_regions)"
  runtime_plan_dispatches="$(stat_value "${matched_line}" runtime_plan_dispatches)"
  runtime_region_fallbacks="$(stat_value "${matched_line}" runtime_region_fallbacks)"

  printf 'train_profile_graph_exec_assertions: commands=%s host_outputs=%s graph_regions=%s runtime_plan_dispatches=%s runtime_region_fallbacks=%s\n' \
    "${commands}" "${host_outputs}" "${graph_regions}" "${runtime_plan_dispatches}" "${runtime_region_fallbacks}"
}

assert_slot_bound_output_log() {
  local log_file="$1"
  local line="" consumed="" pool_allocs="" pool_reuses="" pool_peak_bytes="" pool_budget_declines=""
  while IFS= read -r candidate; do
    [[ "${candidate}" == *metal_slot_bound_outputs:* ]] || continue
    line="metal_slot_bound_outputs:${candidate#*metal_slot_bound_outputs:}"
  done < "${log_file}"

  if [[ -z "${line}" ]]; then
    echo "error: batch32 did not emit metal_slot_bound_outputs diagnostics" >&2
    return 1
  fi

  consumed="$(stat_value "${line}" consumed)" || return 1
  pool_allocs="$(stat_value "${line}" pool_allocs)" || return 1
  pool_reuses="$(stat_value "${line}" pool_reuses)" || return 1
  pool_budget_declines="$(stat_value "${line}" pool_budget_declines)" || pool_budget_declines=0
  pool_peak_bytes="$(stat_value "${line}" pool_peak_bytes)" || return 1
  if (( consumed <= 0 || pool_allocs <= 0 || pool_reuses <= 0 )); then
    echo "error: slot-bound output pool did not prove active allocation reuse" >&2
    echo "       ${line}" >&2
    return 1
  fi
  if [[ -n "${TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES:-}" && "${TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES}" != "0" ]]; then
    if (( pool_peak_bytes > TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES )); then
      echo "error: slot-bound output pool exceeded configured byte budget" >&2
      echo "       budget=${TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES} ${line}" >&2
      return 1
    fi
  fi

  printf 'slot_bound_output_assertions: consumed=%s pool_allocs=%s pool_reuses=%s pool_budget_declines=%s pool_peak_bytes=%s\n' \
    "${consumed}" "${pool_allocs}" "${pool_reuses}" "${pool_budget_declines}" "${pool_peak_bytes}"
}

assert_eager_arena_log() {
  local log_file="$1"
  local line="" peak_bytes="" allocations="" reuse_hits="" spill_bytes="" hazard_declines="" alias_conflicts="" alias_reclaims="" alias_reclaim_bytes=""
  while IFS= read -r candidate; do
    [[ "${candidate}" == *metal_eager_arena:* ]] || continue
    line="metal_eager_arena:${candidate#*metal_eager_arena:}"
  done < "${log_file}"

  if [[ -z "${line}" ]]; then
    echo "error: batch32 did not emit metal_eager_arena diagnostics" >&2
    return 1
  fi

  peak_bytes="$(stat_value "${line}" peak_bytes)" || return 1
  allocations="$(stat_value "${line}" allocations)" || return 1
  reuse_hits="$(stat_value "${line}" reuse_hits)" || reuse_hits=0
  spill_bytes="$(stat_value "${line}" spill_bytes)" || spill_bytes=0
  hazard_declines="$(stat_value "${line}" hazard_declines)" || hazard_declines=0
  alias_conflicts="$(stat_value "${line}" alias_conflicts)" || alias_conflicts=0
  alias_reclaims="$(stat_value "${line}" alias_reclaims)" || alias_reclaims=0
  alias_reclaim_bytes="$(stat_value "${line}" alias_reclaim_bytes)" || alias_reclaim_bytes=0
  if (( peak_bytes <= 0 || allocations <= 0 )); then
    echo "error: Metal eager arena did not prove active allocation" >&2
    echo "       ${line}" >&2
    return 1
  fi
  if [[ -n "${TERMITE_METAL_EAGER_ARENA_MAX_BYTES:-}" && "${TERMITE_METAL_EAGER_ARENA_MAX_BYTES}" != "0" ]]; then
    if (( peak_bytes > TERMITE_METAL_EAGER_ARENA_MAX_BYTES )); then
      echo "error: Metal eager arena exceeded configured byte budget" >&2
      echo "       budget=${TERMITE_METAL_EAGER_ARENA_MAX_BYTES} ${line}" >&2
      return 1
    fi
  fi

  printf 'eager_arena_assertions: peak_bytes=%s allocations=%s reuse_hits=%s spill_bytes=%s hazard_declines=%s alias_conflicts=%s alias_reclaims=%s alias_reclaim_bytes=%s\n' \
    "${peak_bytes}" "${allocations}" "${reuse_hits}" "${spill_bytes}" "${hazard_declines}" "${alias_conflicts}" "${alias_reclaims}" "${alias_reclaim_bytes}"
}

assert_chunk_local_output_log() {
  local log_file="$1"
  local line="" peak_bytes="" allocations="" reuse_hits="" consumed_hints="" unconsumed_hints="" spill_bytes="" alias_conflicts="" resets="" reset_freed_bytes="" discard_freed_bytes="" reset_live_carry_values=""
  while IFS= read -r candidate; do
    [[ "${candidate}" == *metal_chunk_local_outputs:* ]] || continue
    line="metal_chunk_local_outputs:${candidate#*metal_chunk_local_outputs:}"
  done < "${log_file}"

  if [[ -z "${line}" ]]; then
    echo "error: batch32 did not emit metal_chunk_local_outputs diagnostics" >&2
    return 1
  fi

  peak_bytes="$(stat_value "${line}" peak_bytes)" || return 1
  allocations="$(stat_value "${line}" allocations)" || return 1
  reuse_hits="$(stat_value "${line}" reuse_hits)" || reuse_hits=0
  consumed_hints="$(stat_value "${line}" consumed_hints)" || consumed_hints=0
  unconsumed_hints="$(stat_value "${line}" unconsumed_hints)" || unconsumed_hints=0
  spill_bytes="$(stat_value "${line}" spill_bytes)" || spill_bytes=0
  alias_conflicts="$(stat_value "${line}" alias_conflicts)" || alias_conflicts=0
  resets="$(stat_value "${line}" resets)" || return 1
  reset_freed_bytes="$(stat_value "${line}" reset_freed_bytes)" || return 1
  discard_freed_bytes="$(stat_value "${line}" discard_freed_bytes)" || discard_freed_bytes=0
  reset_live_carry_values="$(stat_value "${line}" reset_live_carry_values)" || reset_live_carry_values=0
  if (( peak_bytes <= 0 || allocations <= 0 || resets <= 0 || reset_freed_bytes <= 0 )); then
    echo "error: chunk-local outputs did not prove active mid-step release" >&2
    echo "       ${line}" >&2
    return 1
  fi
  if [[ -n "${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES:-}" && "${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES}" != "0" ]]; then
    if (( peak_bytes > TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES )); then
      echo "error: chunk-local output pool exceeded configured byte budget" >&2
      echo "       budget=${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES} ${line}" >&2
      return 1
    fi
  fi

  printf 'chunk_local_output_assertions: peak_bytes=%s allocations=%s reuse_hits=%s consumed_hints=%s unconsumed_hints=%s spill_bytes=%s alias_conflicts=%s resets=%s reset_freed_bytes=%s discard_freed_bytes=%s reset_live_carry_values=%s\n' \
    "${peak_bytes}" "${allocations}" "${reuse_hits}" "${consumed_hints}" "${unconsumed_hints}" "${spill_bytes}" "${alias_conflicts}" "${resets}" "${reset_freed_bytes}" "${discard_freed_bytes}" "${reset_live_carry_values}"
}

run_env_cmd_with_profile_assertions() {
  local -a local_env=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --)
        shift
        break
        ;;
      *)
        local_env+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${dry_run}" -eq 1 ]]; then
    run_env_cmd "${local_env[@]}" -- "$@"
    return 0
  fi

  local log_file="/private/tmp/termite-gliner2-metal-train-profile-${out_suffix:-train-profile}.log"
  if [[ "${#local_env[@]}" -gt 0 ]]; then
    printf 'env'
    printf ' %q' "${local_env[@]}"
    printf '\n'
    quote_command "$@"
    if env "${local_env[@]}" "$@" > "${log_file}" 2>&1; then
      cat "${log_file}"
    else
      cat "${log_file}"
      return 1
    fi
  else
    quote_command "$@"
    if "$@" > "${log_file}" 2>&1; then
      cat "${log_file}"
    else
      cat "${log_file}"
      return 1
    fi
  fi
  assert_graph_exec_profile_log "${log_file}"
  if [[ "${require_slot_bound_outputs}" -eq 1 ]]; then
    assert_slot_bound_output_log "${log_file}"
  fi
  if [[ "${require_eager_arena_outputs}" -eq 1 ]]; then
    assert_eager_arena_log "${log_file}"
  fi
  if [[ "${require_chunk_local_outputs}" -eq 1 ]]; then
    assert_chunk_local_output_log "${log_file}"
  fi
}

run_unit() {
  cd "${pkg_root}"
  run_cmd zig build -Dmetal=true -Druntime-test-filter=true test -- \
    "metal_compute: device transpose matches gliner2 flattened attention key shape"
  run_cmd zig build -Dmetal=true -Druntime-test-filter=true test -- \
    "metal_compute: lazy multiply reduce last dim stays resident"
}

run_suite() {
  local -a args=("--model-dir" "${model_dir}")
  local -a cmd=("${parity_script}" --suite "${args[@]}" \
    --out-suffix "$(suffix_arg train-suite)"
  )
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("--" "${extra_args[@]}")
  fi
  run_cmd "${cmd[@]}"
}

run_train_profile() {
  require_train_data
  graph_stats=1
  op_profile=1
  partition_op_stats=1
  local -a cmd=(
    "${parity_script}"
    --model-dir "${model_dir}"
    --train-data "${train_data}"
    --out-suffix "$(suffix_arg train-profile)"
    --batch-size "${batch_size}"
    --max-examples "${max_examples}"
    --seq-len "${seq_len}"
    --max-span-width 4
    --lora-rank 16
    --lora-alpha 32
    --lora-dropout 0
  )
  if [[ -n "${trace_range}" ]]; then
    cmd+=(--trace "${trace_range}")
  fi
  cmd+=(
    --
    --span-loss bce
    --span-positive-weight 32
    --span-hard-negative-weight 1
    --span-negative-weight 1
    --max-grad-norm 1.0
    --compiled-required
  )
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  local -a profile_env_result=()
  append_profile_env_args
  run_env_cmd_with_profile_assertions "${profile_env_result[@]}" -- "${cmd[@]}"
}

run_batch32() {
  require_data
  : "${TERMITE_METAL_ENABLE_DEBERTA_ENCODER_LAYER_LORA_REGION:=1}"
  export TERMITE_METAL_ENABLE_DEBERTA_ENCODER_LAYER_LORA_REGION
  graph_stats=1
  op_profile=1
  partition_op_stats=1
  require_slot_bound_outputs=0
  require_eager_arena_outputs=0
  require_chunk_local_outputs=0
  if [[ "${TERMITE_METAL_SLOT_BOUND_OUTPUTS:-0}" == "1" ]]; then
    require_slot_bound_outputs=1
  fi
  if [[ "${TERMITE_METAL_EAGER_ARENA:-0}" == "1" ]]; then
    require_eager_arena_outputs=1
  fi
  if [[ "${TERMITE_METAL_CHUNK_LOCAL_OUTPUTS:-0}" == "1" ]]; then
    require_chunk_local_outputs=1
    if [[ -z "${min_metal_chunk_local_output_consumed_hints}" ]]; then
      min_metal_chunk_local_output_consumed_hints=1
    fi
    if [[ -z "${max_metal_chunk_local_output_unconsumed_hints}" ]]; then
      max_metal_chunk_local_output_unconsumed_hints=0
    fi
  fi

  local gate_batch_size="${batch_size}"
  local gate_seq_len="${seq_len}"
  local gate_max_examples="${max_examples}"
  if [[ "${batch_size_explicit}" -eq 0 ]]; then
    gate_batch_size=32
  fi
  if [[ "${seq_len_explicit}" -eq 0 ]]; then
    gate_seq_len=128
  fi
  if [[ "${max_examples_explicit}" -eq 0 ]]; then
    gate_max_examples=32
  fi
  if [[ -z "${min_metal_runtime_reuse_hit_count}" ]]; then
    min_metal_runtime_reuse_hit_count=1
  fi
  if [[ -z "${max_metal_runtime_total_bytes}" ]]; then
    max_metal_runtime_total_bytes=536870912
  fi
  if [[ -z "${max_metal_tensor_device_owned_peak_live_bytes}" ]]; then
    max_metal_tensor_device_owned_peak_live_bytes=5583457485
  fi
  if [[ -z "${min_graph_runtime_region_dispatch_count}" ]]; then
    min_graph_runtime_region_dispatch_count=1
  fi
  if [[ -z "${min_metal_frame_chunk_boundary_count}" ]]; then
    min_metal_frame_chunk_boundary_count=1
  fi
  if [[ -z "${min_metal_frame_chunk_swept_value_count}" ]]; then
    min_metal_frame_chunk_swept_value_count=1
  fi
  if [[ -z "${max_graph_runtime_region_fallback_count}" ]]; then
    max_graph_runtime_region_fallback_count=0
  fi
  if [[ -z "${max_graph_host_output_count}" ]]; then
    # Current safe batch32 graph path still materializes host outputs; keep the
    # ceiling explicit so the production target can be tightened as residency
    # work lands.
    max_graph_host_output_count=1000
  fi
  if [[ -z "${min_graph_runtime_region_elided_node_count}" ]]; then
    min_graph_runtime_region_elided_node_count=1
  fi
  if [[ -z "${min_metal_deberta_ffn_forward_region_count}" ]]; then
    # The FFN-forward runtime region is available as an opt-in perf experiment,
    # but it currently raises batch32 checkpointed residency enough to re-OOM.
    min_metal_deberta_ffn_forward_region_count=0
  fi
  if [[ -z "${min_metal_deberta_encoder_lora_layer_region_count}" ]]; then
    min_metal_deberta_encoder_lora_layer_region_count=12
  fi
  if [[ -z "${min_metal_deberta_encoder_lora_residual_layernorm_region_count}" ]]; then
    min_metal_deberta_encoder_lora_residual_layernorm_region_count=12
  fi
  if [[ -z "${max_metal_deberta_encoder_lora_layer_scaffold_count}" ]]; then
    max_metal_deberta_encoder_lora_layer_scaffold_count=0
  fi
  if [[ -z "${max_metal_deberta_encoder_lora_layer_fallback_count}" ]]; then
    max_metal_deberta_encoder_lora_layer_fallback_count=0
  fi
  if [[ -z "${min_metal_deberta_attention_flash_call_count}" ]]; then
    min_metal_deberta_attention_flash_call_count=1
  fi
  if [[ -z "${max_metal_deberta_attention_gemm_fallback_count}" ]]; then
    max_metal_deberta_attention_gemm_fallback_count=0
  fi
  if [[ -z "${min_metal_deberta_encoder_layer_success_count}" ]]; then
    # The full DeBERTa encoder-layer runtime path is the next speed milestone;
    # keep the gate at zero until the opt-in region lands and parity passes.
    min_metal_deberta_encoder_layer_success_count=0
  fi
  if [[ -z "${max_runtime_frame_ineligible_missing_model_metadata}" ]]; then
    max_runtime_frame_ineligible_missing_model_metadata=1
  fi

  local -a cmd=(
    "${parity_script}"
    --production-gate
    --model-dir "${model_dir}"
    --train-data "${train_data}"
    --eval-data "${eval_data}"
    --out-suffix "$(suffix_arg production-batch32)"
    --skip-semantic-eval
    --batch-size "${gate_batch_size}"
    --max-examples "${gate_max_examples}"
    --seq-len "${gate_seq_len}"
    --max-span-width 4
    --lora-rank 16
    --lora-alpha 32
    --lora-dropout 0
    --activation-checkpointing
    --activation-checkpoint-interval 1
    --activation-checkpoint-strategy parameters-only
    --
    --allow-flat-loss
    --skip-quality-eval
    --min-train-examples "${gate_max_examples}"
    --min-eval-examples 1
    --min-total-entities 1
    --min-unique-labels 1
    --min-target-coverage-ratio 0.1
    --min-positive-span-labels 1
    --min-steps 1
    --min-supervised-tokens 1
    --min-entity-tokens 1
    --max-avg-step-wall-ms 120000
    --max-total-execute-ms 120000
  )
  append_metal_runtime_threshold_args
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  local -a profile_env_result=("TERMITE_METAL_BUFFER_REUSE_STATS=1")
  if [[ -z "${TERMITE_METAL_FRAME_CHUNK_OPS:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_FRAME_CHUNK_OPS=128")
  fi
  if [[ -n "${TERMITE_METAL_SLOT_BOUND_OUTPUTS:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_SLOT_BOUND_OUTPUTS=${TERMITE_METAL_SLOT_BOUND_OUTPUTS}")
  fi
  if [[ -n "${TERMITE_METAL_SLOT_BOUND_MIN_BYTES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_SLOT_BOUND_MIN_BYTES=${TERMITE_METAL_SLOT_BOUND_MIN_BYTES}")
  fi
  if [[ -n "${TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES=${TERMITE_METAL_SLOT_BOUND_POOL_MAX_BYTES}")
  fi
  if [[ -n "${TERMITE_METAL_EAGER_ARENA:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_EAGER_ARENA=${TERMITE_METAL_EAGER_ARENA}")
  fi
  if [[ -n "${TERMITE_METAL_EAGER_ARENA_MIN_BYTES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_EAGER_ARENA_MIN_BYTES=${TERMITE_METAL_EAGER_ARENA_MIN_BYTES}")
  fi
  if [[ -n "${TERMITE_METAL_EAGER_ARENA_MAX_BYTES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_EAGER_ARENA_MAX_BYTES=${TERMITE_METAL_EAGER_ARENA_MAX_BYTES}")
  fi
  if [[ -n "${TERMITE_METAL_EAGER_ARENA_RECLAIM_ALIASES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_EAGER_ARENA_RECLAIM_ALIASES=${TERMITE_METAL_EAGER_ARENA_RECLAIM_ALIASES}")
  fi
  if [[ -n "${TERMITE_METAL_CHUNK_LOCAL_OUTPUTS:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_CHUNK_LOCAL_OUTPUTS=${TERMITE_METAL_CHUNK_LOCAL_OUTPUTS}")
  fi
  if [[ -n "${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MIN_BYTES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MIN_BYTES=${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MIN_BYTES}")
  fi
  if [[ -n "${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES:-}" ]]; then
    profile_env_result+=("TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES=${TERMITE_METAL_CHUNK_LOCAL_OUTPUT_MAX_BYTES}")
  fi
  append_profile_env_args
  run_env_cmd_with_profile_assertions "${profile_env_result[@]}" -- "${cmd[@]}"
}

run_profile() {
  require_data
  graph_stats=1
  op_profile=1
  local -a cmd
  local -a args=(
    "--production-gate"
    "--model-dir" "${model_dir}"
    "--train-data" "${train_data}"
    "--eval-data" "${eval_data}"
    "--batch-size" "${batch_size}"
    "--max-examples" "${max_examples}"
    "--seq-len" "${seq_len}"
  )
  if [[ -n "${trace_range}" ]]; then
    args+=(--trace "${trace_range}")
  fi
  cmd=(
    "${parity_script}" "${args[@]}" \
    --out-suffix "$(suffix_arg production-profile)" \
    --skip-semantic-eval \
    -- \
    --max-examples "${max_examples}" \
    --min-steps 5 \
    --allow-flat-loss \
    --skip-quality-eval \
    --max-avg-step-wall-ms 10000 \
    --min-train-examples "${max_examples}" \
    --min-eval-examples 1 \
    --min-total-entities 1 \
    --min-positive-span-labels 1 \
    --min-supervised-tokens 1 \
    --min-entity-tokens 1
  )
  append_metal_runtime_threshold_args
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  local -a profile_env_result=()
  append_profile_env_args
  run_env_cmd_with_profile_assertions "${profile_env_result[@]}" -- "${cmd[@]}"
}

run_diagnostic() {
  require_data
  local -a profile_env_result=()
  append_profile_env_args
  local -a cmd
  local -a args=(
    "--production-gate"
    "--model-dir" "${model_dir}"
    "--train-data" "${train_data}"
    "--eval-data" "${eval_data}"
    "--batch-size" "${batch_size}"
    "--max-examples" "${max_examples}"
    "--seq-len" "${seq_len}"
  )
  cmd=(
    "${parity_script}" "${args[@]}" \
    --out-suffix "$(suffix_arg production-diagnostic)" \
    --skip-semantic-eval \
    -- \
    --skip-quality-eval
  )
  append_metal_runtime_threshold_args
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  if [[ "${#profile_env_result[@]}" -gt 0 ]]; then
    run_env_cmd_with_profile_assertions "${profile_env_result[@]}" -- "${cmd[@]}"
  else
    run_env_cmd_with_profile_assertions -- "${cmd[@]}"
  fi
}

run_production() {
  require_data
  local -a profile_env_result=()
  append_profile_env_args
  local -a args cmd
  args=(
    "--production-gate"
    "--model-dir" "${model_dir}"
    "--train-data" "${train_data}"
    "--eval-data" "${eval_data}"
    "--batch-size" "${batch_size}"
    "--max-examples" "${max_examples}"
    "--seq-len" "${seq_len}"
  )
  if [[ "${skip_semantic_eval}" -eq 1 ]]; then
    args+=("--skip-semantic-eval")
  fi
  cmd=(
    "${parity_script}" "${args[@]}" \
    --out-suffix "$(suffix_arg production-gate)"
  )
  if [[ "${#extra_args[@]}" -gt 0 ||
    -n "${max_metal_runtime_total_bytes}" ||
    -n "${max_metal_tensor_device_owned_peak_live_bytes}" ||
    -n "${max_metal_eager_arena_peak_bytes}" ||
    -n "${max_metal_eager_arena_spill_bytes}" ||
    -n "${max_metal_chunk_local_output_peak_bytes}" ||
    -n "${max_metal_chunk_local_output_spill_bytes}" ||
    -n "${max_metal_chunk_local_output_unconsumed_hints}" ||
    -n "${min_metal_chunk_local_output_consumed_hints}" ||
    -n "${min_metal_runtime_reuse_hit_count}" ||
    -n "${max_graph_command_dispatch_count}" ||
    -n "${max_graph_host_output_count}" ||
    -n "${max_metal_frame_gpu_ms}" ||
    -n "${max_metal_last_frame_compute_encoder_count}" ||
    -n "${min_metal_frame_chunk_boundary_count}" ||
    -n "${min_metal_frame_chunk_promoted_value_count}" ||
    -n "${min_metal_frame_chunk_swept_value_count}" ||
    -n "${min_graph_runtime_region_dispatch_count}" ||
    -n "${max_graph_runtime_region_fallback_count}" ||
    -n "${min_graph_runtime_region_elided_node_count}" ||
    -n "${min_metal_deberta_ffn_forward_region_count}" ||
    -n "${min_metal_deberta_encoder_lora_layer_region_count}" ||
    -n "${min_metal_deberta_encoder_lora_residual_layernorm_region_count}" ||
    -n "${max_metal_deberta_encoder_lora_layer_scaffold_count}" ||
    -n "${max_metal_deberta_encoder_lora_layer_fallback_count}" ||
    -n "${min_metal_deberta_attention_flash_call_count}" ||
    -n "${max_metal_deberta_attention_gemm_fallback_count}" ||
    -n "${min_metal_deberta_encoder_layer_success_count}" ||
    -n "${min_metal_deberta_ffn_fused_call_count}" ||
    -n "${max_metal_deberta_ffn_fused_fallback_count}" ||
    -n "${max_runtime_frame_ineligible_missing_model_metadata}" ]]; then
    cmd+=("--")
    append_metal_runtime_threshold_args
    if [[ "${#extra_args[@]}" -gt 0 ]]; then
      cmd+=("${extra_args[@]}")
    fi
  fi
  if [[ "${#profile_env_result[@]}" -gt 0 ]]; then
    run_env_cmd_with_profile_assertions "${profile_env_result[@]}" -- "${cmd[@]}"
  else
    run_env_cmd_with_profile_assertions -- "${cmd[@]}"
  fi
}

echo "package_root=${pkg_root}"
echo "mode=${mode}"

case "${mode}" in
  unit)
    run_unit
    ;;
  suite)
    run_suite
    ;;
  profile)
    run_profile
    ;;
  train-profile)
    run_train_profile
    ;;
  batch32)
    run_batch32
    ;;
  diagnostic)
    run_diagnostic
    ;;
  production)
    run_production
    ;;
  all)
    run_unit
    run_suite
    run_diagnostic
    ;;
esac
