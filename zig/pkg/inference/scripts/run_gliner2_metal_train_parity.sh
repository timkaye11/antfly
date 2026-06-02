#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gliner2_metal_train_parity.sh [options] [-- extra train-gliner2-autodiff args]

Runs the GLiNER2 Metal graph-exec training smoke with stable defaults.

Options:
  --nodes IDS             Enable selected-node parity diagnostics, e.g. 61,1464,1465
  --trace START:END       Enable compiled train trace and Metal progress for a node range
  --full-parity           Enable full direct-vs-graph-exec parity gate
  --out-suffix NAME       Output under /private/tmp/termite-gliner2-metal-NAME
  --out-dir DIR           Explicit output directory
  --model-dir DIR         Model directory (default: /private/tmp/termite-models/gliner2)
  --train-data FILE       Train data (default: testdata/gliner2_ner_smoke.jsonl)
  --seq-len N             Sequence length (default: 16)
  --max-span-width N      Max span width (default: 2)
  --lora-rank N           LoRA rank (default: 1)
  --lora-alpha N          LoRA alpha (default: 2)
  --lora-dropout N        LoRA dropout (default: 0)
  --max-examples N        Max examples (default: 1)
  --batch-size N          Batch size (default: 1)
  --learning-rate N       Learning rate (default: 1e-3)
  --entity-types CSV      Entity types (default: person,organization,location)
  --num-classes N         Number of classes (default: 4)
  --no-graph-exec         Disable graph executor env flag
  --help                  Show this help

Examples:
  scripts/run_gliner2_metal_train_parity.sh --nodes 1457,1461,1462,1463,1464 --out-suffix gated1464
  scripts/run_gliner2_metal_train_parity.sh --trace 1462:1467 --out-suffix trace-1467
  scripts/run_gliner2_metal_train_parity.sh --full-parity --out-suffix full-parity
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd "${script_dir}/.." && pwd)"

model_dir="/private/tmp/termite-models/gliner2"
train_data="testdata/gliner2_ner_smoke.jsonl"
out_dir=""
out_suffix=""
nodes=""
trace_range=""
full_parity=0
graph_exec=1
seq_len=16
max_span_width=2
lora_rank=1
lora_alpha=2
lora_dropout=0
max_examples=1
batch_size=1
learning_rate="1e-3"
entity_types="person,organization,location"
num_classes=4
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      nodes="${2:?missing value for --nodes}"
      shift 2
      ;;
    --trace)
      trace_range="${2:?missing value for --trace}"
      shift 2
      ;;
    --full-parity)
      full_parity=1
      shift
      ;;
    --out-suffix)
      out_suffix="${2:?missing value for --out-suffix}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --model-dir)
      model_dir="${2:?missing value for --model-dir}"
      shift 2
      ;;
    --train-data)
      train_data="${2:?missing value for --train-data}"
      shift 2
      ;;
    --seq-len)
      seq_len="${2:?missing value for --seq-len}"
      shift 2
      ;;
    --max-span-width)
      max_span_width="${2:?missing value for --max-span-width}"
      shift 2
      ;;
    --lora-rank)
      lora_rank="${2:?missing value for --lora-rank}"
      shift 2
      ;;
    --lora-alpha)
      lora_alpha="${2:?missing value for --lora-alpha}"
      shift 2
      ;;
    --lora-dropout)
      lora_dropout="${2:?missing value for --lora-dropout}"
      shift 2
      ;;
    --max-examples)
      max_examples="${2:?missing value for --max-examples}"
      shift 2
      ;;
    --batch-size)
      batch_size="${2:?missing value for --batch-size}"
      shift 2
      ;;
    --learning-rate)
      learning_rate="${2:?missing value for --learning-rate}"
      shift 2
      ;;
    --entity-types)
      entity_types="${2:?missing value for --entity-types}"
      shift 2
      ;;
    --num-classes)
      num_classes="${2:?missing value for --num-classes}"
      shift 2
      ;;
    --no-graph-exec)
      graph_exec=0
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
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${trace_range}" && ! "${trace_range}" =~ ^[0-9]+:[0-9]+$ ]]; then
  echo "error: --trace must be START:END" >&2
  exit 2
fi

if [[ -z "${out_dir}" ]]; then
  if [[ -z "${out_suffix}" ]]; then
    if [[ -n "${nodes}" ]]; then
      out_suffix="nodes-${nodes//,/-}"
    elif [[ -n "${trace_range}" ]]; then
      out_suffix="trace-${trace_range/:/-}"
    elif [[ "${full_parity}" -eq 1 ]]; then
      out_suffix="full-parity"
    else
      out_suffix="smoke"
    fi
  fi
  out_dir="/private/tmp/termite-gliner2-metal-${out_suffix}"
fi

env_args=()
if [[ "${graph_exec}" -eq 1 ]]; then
  env_args+=("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1")
fi
if [[ -n "${nodes}" ]]; then
  env_args+=("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS=${nodes}")
fi
if [[ "${full_parity}" -eq 1 ]]; then
  env_args+=("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK=1")
fi
if [[ -n "${trace_range}" ]]; then
  env_args+=("TERMITE_COMPILED_TRAIN_TRACE=1")
  env_args+=("TERMITE_METAL_PARTITION_PROGRESS_START=${trace_range%%:*}")
  env_args+=("TERMITE_METAL_PARTITION_PROGRESS_END=${trace_range##*:}")
fi

cmd=(
  zig build -Dmetal=true train-gliner2-autodiff --
  --model-dir "${model_dir}"
  --train-data "${train_data}"
  --out-dir "${out_dir}"
  --epochs 1
  --batch-size "${batch_size}"
  --max-examples "${max_examples}"
  --seq-len "${seq_len}"
  --learning-rate "${learning_rate}"
  --backend metal
  --objective span-start
  --entity-types "${entity_types}"
  --num-classes "${num_classes}"
  --lora-rank "${lora_rank}"
  --lora-alpha "${lora_alpha}"
  --lora-dropout "${lora_dropout}"
  --max-span-width "${max_span_width}"
)

if [[ "${#extra_args[@]}" -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo "package_root=${pkg_root}"
echo "out_dir=${out_dir}"
if [[ "${#env_args[@]}" -gt 0 ]]; then
  printf 'env'
  printf ' %q' "${env_args[@]}"
  printf '\n'
fi
printf 'cmd'
printf ' %q' "${cmd[@]}"
printf '\n'

cd "${pkg_root}"
if [[ "${#env_args[@]}" -gt 0 ]]; then
  env "${env_args[@]}" "${cmd[@]}"
else
  "${cmd[@]}"
fi
