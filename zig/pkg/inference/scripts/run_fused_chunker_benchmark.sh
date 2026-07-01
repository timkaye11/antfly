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
#
# Boundary-F1 benchmark lane for the fused chunker release gate
# (evals/chunker/manifest.json). Orchestrates:
#
#   prepare   Materialize chonky token-label JSONL into fused eval JSONL
#             (first 1M tokens per dataset) for each manifest dataset.
#   eval      eval-fused-chunker per dataset with --results-out.
#   aggregate Merge per-dataset results into one benchmark result file.
#   verify    verify-fused-chunker-benchmark against the manifest.
#
# Chonky-format inputs (single-line {"tokens":[...],"ner_tags":[...]} JSONL,
# see github.com/mirth/chonky generate_eval_datasets.py) are expected at
# $ANTFLY_FUSED_CHUNKER_CHONKY_DATA_DIR/chonky_<dataset>.jsonl.
#
# Stages can be run independently, e.g. prepare-only on a CPU box:
#   ANTFLY_FUSED_CHUNKER_BENCHMARK_STAGES=prepare scripts/run_fused_chunker_benchmark.sh
# then later:
#   ANTFLY_FUSED_CHUNKER_BENCHMARK_STAGES=eval,aggregate,verify \
#   ANTFLY_FUSED_CHUNKER_CHECKPOINT=/path/to/checkpoint_final.safetensors \
#   scripts/run_fused_chunker_benchmark.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd -- "$script_dir/.." && pwd)"

zig_bin="${ANTFLY_ZIG_BIN:-}"
if [[ -z "$zig_bin" ]]; then
  if command -v zig >/dev/null 2>&1; then
    zig_bin="$(command -v zig)"
  elif [[ -x "$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig" ]]; then
    zig_bin="$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig"
  else
    echo "missing Zig compiler; set ANTFLY_ZIG_BIN=/path/to/zig" >&2
    exit 1
  fi
fi

stages="${ANTFLY_FUSED_CHUNKER_BENCHMARK_STAGES:-prepare,eval,aggregate,verify}"
datasets="${ANTFLY_FUSED_CHUNKER_BENCHMARK_DATASETS:-bookcorpus en_judgements paul_graham 20_newsgroups}"
chonky_data_dir="${ANTFLY_FUSED_CHUNKER_CHONKY_DATA_DIR:-/Users/tim/Documents/af/gopeft/data}"
out_dir="${ANTFLY_FUSED_CHUNKER_BENCHMARK_OUT:-/private/tmp/fused-chunker-benchmark}"
manifest_path="${ANTFLY_FUSED_CHUNKER_BENCHMARK_MANIFEST:-$pkg_root/evals/chunker/manifest.json}"
verify_lane="${ANTFLY_FUSED_CHUNKER_BENCHMARK_LANE:-boundary_f1}"

# prepare stage
max_total_tokens="${ANTFLY_FUSED_CHUNKER_BENCHMARK_MAX_TOTAL_TOKENS:-1000000}"
max_record_tokens="${ANTFLY_FUSED_CHUNKER_BENCHMARK_MAX_RECORD_TOKENS:-384}"

# eval stage
checkpoint="${ANTFLY_FUSED_CHUNKER_CHECKPOINT:-}"
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-base}"
backend="${ANTFLY_FUSED_CHUNKER_BACKEND:-metal}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
num_layers="${ANTFLY_FUSED_CHUNKER_NUM_LAYERS:-22}"
max_examples="${ANTFLY_FUSED_CHUNKER_BENCHMARK_MAX_EXAMPLES:-0}"
internal_phase20_best_f1="${ANTFLY_FUSED_CHUNKER_INTERNAL_PHASE20_BEST_F1:-}"

prepared_dir="$out_dir/prepared"
results_dir="$out_dir/results"
aggregate_results="$out_dir/fused_chunker_benchmark_results.json"

has_stage() {
  case ",$stages," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

zig_tool() {
  local build_args=(-Dskip-openapi=true -Doptimize=ReleaseFast)
  if [[ "$1" == "metal" ]]; then
    build_args+=(-Dmetal=true)
  fi
  shift
  (
    cd "$pkg_root"
    "$zig_bin" build "${build_args[@]}" "$@"
  )
}

echo "fused chunker boundary-F1 benchmark lane"
echo "  manifest=$manifest_path"
echo "  datasets=$datasets"
echo "  stages=$stages"
echo "  chonky_data_dir=$chonky_data_dir"
echo "  out_dir=$out_dir"
echo "  max_total_tokens=$max_total_tokens max_record_tokens=$max_record_tokens"

if [[ ! -f "$manifest_path" ]]; then
  echo "missing benchmark manifest at $manifest_path" >&2
  exit 1
fi

mkdir -p "$prepared_dir" "$results_dir"

if has_stage prepare; then
  echo "== stage: prepare =="
  for dataset in $datasets; do
    input_path="$chonky_data_dir/chonky_$dataset.jsonl"
    prepared_path="$prepared_dir/$dataset.fused_eval.jsonl"
    if [[ ! -f "$input_path" ]]; then
      echo "missing chonky input at $input_path" >&2
      echo "materialize it in chonky generate_eval_datasets format first" >&2
      exit 1
    fi
    echo "-- prepare $dataset"
    zig_tool cpu prepare-fused-chunker-boundary-eval -- \
      --input "$input_path" \
      --out "$prepared_path" \
      --dataset-name "$dataset" \
      --max-total-tokens "$max_total_tokens" \
      --max-record-tokens "$max_record_tokens"
  done
fi

if has_stage eval; then
  echo "== stage: eval (backend=$backend) =="
  if [[ -z "$checkpoint" ]]; then
    echo "missing checkpoint; set ANTFLY_FUSED_CHUNKER_CHECKPOINT=/path/to/checkpoint_final.safetensors" >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint at $checkpoint" >&2
    exit 1
  fi
  if [[ ! -f "$model_dir/model.safetensors" || ! -f "$model_dir/tokenizer.json" ]]; then
    echo "missing ModernBERT weights/tokenizer under $model_dir" >&2
    exit 1
  fi
  for dataset in $datasets; do
    prepared_path="$prepared_dir/$dataset.fused_eval.jsonl"
    results_path="$results_dir/$dataset.fused_chunker_benchmark_results.json"
    if [[ ! -f "$prepared_path" ]]; then
      echo "missing prepared eval data at $prepared_path (run the prepare stage first)" >&2
      exit 1
    fi
    echo "-- eval $dataset"
    eval_args=(
      --data "$prepared_path"
      --checkpoint "$checkpoint"
      --model-dir "$model_dir"
      --backend "$backend"
      --batch-size "$batch_size"
      --max-seq-len "$max_seq_len"
      --max-chunks "$max_chunks"
      --num-layers "$num_layers"
      --benchmark-dataset-name "$dataset"
      --results-out "$results_path"
    )
    if [[ "$max_examples" != "0" ]]; then
      eval_args+=(--max-examples "$max_examples")
    fi
    if [[ -n "$internal_phase20_best_f1" ]]; then
      eval_args+=(--internal-phase20-best-f1 "$internal_phase20_best_f1")
    fi
    zig_tool "$backend" eval-fused-chunker -- "${eval_args[@]}"
  done
fi

if has_stage aggregate; then
  echo "== stage: aggregate =="
  aggregate_args=(--out "$aggregate_results")
  for dataset in $datasets; do
    results_path="$results_dir/$dataset.fused_chunker_benchmark_results.json"
    if [[ ! -f "$results_path" ]]; then
      echo "missing per-dataset results at $results_path (run the eval stage first)" >&2
      exit 1
    fi
    aggregate_args+=(--input "$results_path")
  done
  zig_tool cpu aggregate-fused-chunker-benchmark -- "${aggregate_args[@]}"
fi

if has_stage verify; then
  echo "== stage: verify =="
  if [[ ! -f "$aggregate_results" ]]; then
    echo "missing aggregated results at $aggregate_results (run the aggregate stage first)" >&2
    exit 1
  fi
  zig_tool cpu verify-fused-chunker-benchmark -- \
    --manifest "$manifest_path" \
    --results "$aggregate_results" \
    --lane "$verify_lane"
  echo "benchmark verification passed lane=$verify_lane"
  echo "  results=$aggregate_results"
fi
