#!/usr/bin/env bash
# Copyright 2026 Antfly
#
# Fused chunker release-candidate checklist. Runs every release gate lane in
# dependency order and stops at the first failure:
#
#   parity     Go<->Zig batch parity (run_fused_chunker_go_zig_batch_parity.sh)
#   readiness  full-mode training + validation gates
#              (run_fused_chunker_production_readiness.sh, FULL mode:
#              fixed & best & last F1 >= 0.766, MPSGraph-only, perf/memory)
#   boundary   4-dataset chonky boundary benchmark + verifier
#              (run_fused_chunker_benchmark.sh)
#   retrieval  retrieval NDCG@10 lanes + verifier
#              (run_fused_chunker_retrieval_benchmark.sh)
#   export     package the gated checkpoint for serving
#              (export-fused-chunker-model)
#
# Select stages with ANTFLY_FUSED_CHUNKER_RC_STAGES (comma-separated, default
# all). Point ANTFLY_FUSED_CHUNKER_RC_CHECKPOINT at an existing gated
# checkpoint to skip retraining and reuse it for boundary/retrieval/export.
# Every stage writes under ANTFLY_FUSED_CHUNKER_RC_OUT.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stages="${ANTFLY_FUSED_CHUNKER_RC_STAGES:-parity,readiness,boundary,retrieval,export}"
rc_out="${ANTFLY_FUSED_CHUNKER_RC_OUT:-/private/tmp/fused-chunker-release-candidate/$(date +%Y%m%d-%H%M)}"
checkpoint="${ANTFLY_FUSED_CHUNKER_RC_CHECKPOINT:-}"
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-base}"
model_version="${ANTFLY_FUSED_CHUNKER_RC_MODEL_VERSION:-}"

mkdir -p "$rc_out"

has_stage() {
  case ",$stages," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

banner() {
  echo
  echo "==== release-candidate stage: $1 ===="
}

record() {
  echo "$1" >> "$rc_out/rc_stage_results.txt"
}

if has_stage parity; then
  banner parity
  bash "$script_dir/run_fused_chunker_go_zig_batch_parity.sh"
  record "parity: passed"
fi

if has_stage readiness; then
  banner readiness
  if [[ -n "$checkpoint" ]]; then
    echo "readiness: skipped (ANTFLY_FUSED_CHUNKER_RC_CHECKPOINT provided: $checkpoint)"
    record "readiness: skipped (existing checkpoint)"
  else
    export ANTFLY_FUSED_CHUNKER_READINESS_MODE=full
    export ANTFLY_FUSED_CHUNKER_OUTPUT="$rc_out/readiness"
    bash "$script_dir/run_fused_chunker_production_readiness.sh"
    checkpoint="$rc_out/readiness/checkpoint_final.safetensors"
    record "readiness: passed ($checkpoint)"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "release candidate checkpoint missing: $checkpoint" >&2
    exit 1
  fi
fi

if has_stage boundary; then
  banner boundary
  if [[ -z "$checkpoint" ]]; then
    echo "boundary stage needs ANTFLY_FUSED_CHUNKER_RC_CHECKPOINT or a readiness run" >&2
    exit 1
  fi
  ANTFLY_FUSED_CHUNKER_CHECKPOINT="$checkpoint" \
  ANTFLY_FUSED_CHUNKER_MODEL_DIR="$model_dir" \
  ANTFLY_FUSED_CHUNKER_BENCHMARK_OUT="$rc_out/boundary" \
    bash "$script_dir/run_fused_chunker_benchmark.sh"
  record "boundary: passed"
fi

if has_stage retrieval; then
  banner retrieval
  if [[ -z "$checkpoint" ]]; then
    echo "retrieval stage needs ANTFLY_FUSED_CHUNKER_RC_CHECKPOINT or a readiness run" >&2
    exit 1
  fi
  ANTFLY_FUSED_CHUNKER_CHECKPOINT="$checkpoint" \
  ANTFLY_FUSED_CHUNKER_MODEL_DIR="$model_dir" \
  ANTFLY_FUSED_CHUNKER_RETRIEVAL_OUT="$rc_out/retrieval" \
    bash "$script_dir/run_fused_chunker_retrieval_benchmark.sh"
  record "retrieval: passed"
fi

if has_stage export; then
  banner export
  if [[ -z "$checkpoint" ]]; then
    echo "export stage needs ANTFLY_FUSED_CHUNKER_RC_CHECKPOINT or a readiness run" >&2
    exit 1
  fi
  version_args=()
  if [[ -n "$model_version" ]]; then
    version_args+=(--model-version "$model_version")
  fi
  (cd "$script_dir/.." && zig build -Dskip-openapi=true export-fused-chunker-model -- \
    --checkpoint "$checkpoint" \
    --model-dir "$model_dir" \
    --out "$rc_out/export/fused-chunker" \
    --force "${version_args[@]}")
  record "export: passed ($rc_out/export/fused-chunker)"
fi

echo
echo "release-candidate stages complete:"
cat "$rc_out/rc_stage_results.txt"
