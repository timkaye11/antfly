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
  --suite                 Run the repeat validation suite instead of one train run
  --production-gate       Run the canonical gliner2-production-readiness Metal gate
  --dry-run               Print the production gate shape without training
  --out-suffix NAME       Output under /private/tmp/termite-gliner2-metal-NAME
  --out-dir DIR           Explicit output directory
  --model-dir DIR         Model directory (default: /private/tmp/termite-models/gliner2)
  --train-data FILE       Train data (default for smoke/suite: testdata/gliner2_ner_smoke.jsonl)
  --eval-data FILE        Eval data for readiness gates (default: train data)
  --seq-len N             Sequence length (default: 32)
  --max-span-width N      Max span width (default: 2)
  --zig-optimize MODE     Zig optimize mode (default: ReleaseFast)
  --lora-rank N           LoRA rank (default: 1)
  --lora-alpha N          LoRA alpha (default: 2)
  --lora-dropout N        LoRA dropout (default: 0)
  --train-regular-head    Train regular task-head params too (default: LoRA-only)
  --max-examples N        Max examples (default: 1)
  --batch-size N          Batch size (default: 1)
  --learning-rate N       Learning rate (default: 1e-3)
  --objective NAME        Training objective (default: gliner2-total-loss)
  --entity-types CSV      Entity types (default: person,organization,location)
  --num-classes N         Number of classes (default: 4)
  --semantic-golden TEXT EXPECT_TEXT LABEL MIN_SCORE
                          Add a semantic golden for production gate eval
  --eval-text TEXT        Single semantic eval text for production gate eval
  --expect-text TEXT      Expected text for --eval-text
  --expect-label LABEL    Expected label for --eval-text
  --min-score FLOAT       Minimum score for --eval-text
  --skip-semantic-eval    Skip semantic eval for readiness gate diagnostics
  --no-graph-exec         Disable graph executor env flag
  --help                  Show this help

Examples:
  scripts/run_gliner2_metal_train_parity.sh --nodes 1457,1461,1462,1463,1464 --out-suffix gated1464
  scripts/run_gliner2_metal_train_parity.sh --trace 1462:1467 --out-suffix trace-1467
  scripts/run_gliner2_metal_train_parity.sh --full-parity --out-suffix full-parity
  scripts/run_gliner2_metal_train_parity.sh --suite
  scripts/run_gliner2_metal_train_parity.sh --production-gate --train-data train.jsonl --eval-data eval.jsonl \
    --semantic-golden "Alice works at Acme in Paris." Alice person 0.03
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd "${script_dir}/.." && pwd)"

model_dir="/private/tmp/termite-models/gliner2"
train_data="testdata/gliner2_ner_smoke.jsonl"
eval_data=""
out_dir=""
out_suffix=""
nodes=""
trace_range=""
full_parity=0
suite=0
production_gate=0
graph_exec=1
skip_semantic_eval=0
dry_run=0
train_data_explicit=0
eval_data_explicit=0
seq_len=32
max_span_width=2
zig_optimize="ReleaseFast"
lora_rank=1
lora_alpha=2
lora_dropout=0
lora_only_trainables=1
max_examples=1
batch_size=1
learning_rate="1e-3"
objective="gliner2-total-loss"
entity_types="person,organization,location"
num_classes=4
entity_types_explicit=0
num_classes_explicit=0
extra_args=()
semantic_args=()

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
    --suite)
      suite=1
      shift
      ;;
    --production-gate)
      production_gate=1
      shift
      ;;
    --dry-run)
      dry_run=1
      extra_args+=("--dry-run")
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
      train_data_explicit=1
      shift 2
      ;;
    --eval-data)
      eval_data="${2:?missing value for --eval-data}"
      eval_data_explicit=1
      shift 2
      ;;
    --seq-len)
      seq_len="${2:?missing value for --seq-len}"
      shift 2
      ;;
    --zig-optimize)
      zig_optimize="${2:?missing value for --zig-optimize}"
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
    --train-regular-head)
      lora_only_trainables=0
      shift
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
    --objective)
      objective="${2:?missing value for --objective}"
      shift 2
      ;;
    --entity-types)
      entity_types="${2:?missing value for --entity-types}"
      entity_types_explicit=1
      shift 2
      ;;
    --num-classes)
      num_classes="${2:?missing value for --num-classes}"
      num_classes_explicit=1
      shift 2
      ;;
    --semantic-golden)
      semantic_args+=("--semantic-golden" "${2:?missing TEXT for --semantic-golden}" "${3:?missing EXPECT_TEXT for --semantic-golden}" "${4:?missing LABEL for --semantic-golden}" "${5:?missing MIN_SCORE for --semantic-golden}")
      shift 5
      ;;
    --eval-text)
      semantic_args+=("--eval-text" "${2:?missing value for --eval-text}")
      shift 2
      ;;
    --expect-text)
      semantic_args+=("--expect-text" "${2:?missing value for --expect-text}")
      shift 2
      ;;
    --expect-label)
      semantic_args+=("--expect-label" "${2:?missing value for --expect-label}")
      shift 2
      ;;
    --min-score)
      semantic_args+=("--min-score" "${2:?missing value for --min-score}")
      shift 2
      ;;
    --skip-semantic-eval)
      skip_semantic_eval=1
      shift
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

if [[ -z "${eval_data}" ]]; then
  eval_data="${train_data}"
fi

if [[ "${#extra_args[@]}" -gt 0 ]]; then
  for arg in "${extra_args[@]}"; do
    if [[ "${arg}" == "--dry-run" ]]; then
      dry_run=1
    fi
  done
fi

if [[ "${production_gate}" -eq 1 && "${dry_run}" -eq 0 ]]; then
  if [[ "${train_data_explicit}" -eq 0 || "${eval_data_explicit}" -eq 0 ]]; then
    echo "error: --production-gate requires explicit --train-data and --eval-data unless --dry-run is set" >&2
    echo "       use --suite for the smoke/parity readiness checks" >&2
    exit 2
  fi
fi

if [[ "${dry_run}" -eq 1 && "${production_gate}" -eq 0 ]]; then
  echo "error: --dry-run is only supported with --production-gate" >&2
  exit 2
fi

if [[ -z "${out_dir}" ]]; then
  if [[ -z "${out_suffix}" ]]; then
    if [[ "${production_gate}" -eq 1 ]]; then
      out_suffix="production-gate"
    elif [[ "${suite}" -eq 1 ]]; then
      out_suffix="suite"
    elif [[ -n "${nodes}" ]]; then
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
  zig build -Dmetal=true "-Doptimize=${zig_optimize}" train-gliner2-autodiff --
  --model-dir "${model_dir}"
  --train-data "${train_data}"
  --out-dir "${out_dir}"
  --epochs 1
  --batch-size "${batch_size}"
  --max-examples "${max_examples}"
  --seq-len "${seq_len}"
  --learning-rate "${learning_rate}"
  --backend metal
  --objective "${objective}"
  --lora-rank "${lora_rank}"
  --lora-alpha "${lora_alpha}"
  --lora-dropout "${lora_dropout}"
  --max-span-width "${max_span_width}"
)
if [[ "${objective}" != "gliner2-total-loss" || "${entity_types_explicit}" -eq 1 ]]; then
  cmd+=("--entity-types" "${entity_types}")
fi
if [[ "${objective}" != "gliner2-total-loss" || "${num_classes_explicit}" -eq 1 ]]; then
  cmd+=("--num-classes" "${num_classes}")
fi
if [[ "${lora_only_trainables}" -eq 1 ]]; then
  cmd+=("--lora-only-trainables")
fi

if [[ "${#extra_args[@]}" -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

cd "${pkg_root}"

run_cmd() {
  printf 'cmd'
  printf ' %q' "$@"
  printf '\n'
  "$@"
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
    printf 'cmd'
    printf ' %q' "$@"
    printf '\n'
    env "${local_env[@]}" "$@"
  else
    run_cmd "$@"
  fi
}

run_train() {
  local run_out_dir="$1"
  shift
  local -a run_env=("${env_args[@]}")
  local -a run_cmd_args=(
    zig build -Dmetal=true "-Doptimize=${zig_optimize}" train-gliner2-autodiff --
    --model-dir "${model_dir}"
    --train-data "${train_data}"
    --out-dir "${run_out_dir}"
    --epochs 1
    --batch-size "${batch_size}"
    --max-examples "${max_examples}"
    --seq-len "${seq_len}"
    --learning-rate "${learning_rate}"
    --backend metal
    --objective "${objective}"
    --lora-rank "${lora_rank}"
    --lora-alpha "${lora_alpha}"
    --lora-dropout "${lora_dropout}"
    --max-span-width "${max_span_width}"
  )
  if [[ "${objective}" != "gliner2-total-loss" || "${entity_types_explicit}" -eq 1 ]]; then
    run_cmd_args+=("--entity-types" "${entity_types}")
  fi
  if [[ "${objective}" != "gliner2-total-loss" || "${num_classes_explicit}" -eq 1 ]]; then
    run_cmd_args+=("--num-classes" "${num_classes}")
  fi
  if [[ "${lora_only_trainables}" -eq 1 ]]; then
    run_cmd_args+=("--lora-only-trainables")
  fi
  if [[ $# -gt 0 ]]; then
    run_cmd_args+=("$@")
  fi
  run_env_cmd "${run_env[@]}" -- "${run_cmd_args[@]}"
}

run_readiness_smoke() {
  local -a readiness_smoke_args=(
    zig build -Dmetal=true "-Doptimize=${zig_optimize}" gliner2-production-readiness --
    "${model_dir}" "${train_data}" "${eval_data}" "${out_dir}-readiness" "${entity_types}"
    --epochs 1
    --batch-size "${batch_size}"
    --max-examples "${max_examples}"
    --seq-len "${seq_len}"
    --learning-rate "${learning_rate}"
    --backend metal
    --objective "${objective}"
    --compiled-required
    --skip-semantic-eval
    --allow-flat-loss
    --min-train-examples 1
    --min-eval-examples 1
    --min-total-entities 1
    --min-unique-labels 1
    --min-target-coverage-ratio 0.1
    --min-positive-span-labels 1
    --min-steps 1
    --min-supervised-tokens 1
    --min-entity-tokens 1
    --max-avg-step-wall-ms 10000
    --max-total-execute-ms 10000
  )
  if [[ "${lora_only_trainables}" -eq 1 ]]; then
    readiness_smoke_args+=("--lora-only-trainables")
  fi
  run_env_cmd TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 -- "${readiness_smoke_args[@]}"
}

run_production_gate() {
  local -a readiness_args=(
    zig build -Dmetal=true "-Doptimize=${zig_optimize}" gliner2-production-readiness --
    "${model_dir}" "${train_data}" "${eval_data}" "${out_dir}" "${entity_types}"
    --production-metal-gate
    --batch-size "${batch_size}"
    --max-examples "${max_examples}"
    --seq-len "${seq_len}"
    --learning-rate "${learning_rate}"
    --lora-rank "${lora_rank}"
    --lora-alpha "${lora_alpha}"
    --lora-dropout "${lora_dropout}"
    --max-span-width "${max_span_width}"
    --objective "${objective}"
  )
  if [[ "${lora_only_trainables}" -eq 1 ]]; then
    readiness_args+=("--lora-only-trainables")
  fi
  if [[ "${skip_semantic_eval}" -eq 1 ]]; then
    readiness_args+=("--skip-semantic-eval")
  fi
  if [[ "${#semantic_args[@]}" -gt 0 ]]; then
    readiness_args+=("${semantic_args[@]}")
  fi
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    readiness_args+=("${extra_args[@]}")
  fi
  run_env_cmd TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 -- "${readiness_args[@]}"
}

echo "package_root=${pkg_root}"
echo "out_dir=${out_dir}"

if [[ "${production_gate}" -eq 1 ]]; then
  run_production_gate
elif [[ "${suite}" -eq 1 ]]; then
  run_cmd zig build -Dmetal=true "-Doptimize=${zig_optimize}" train-gliner2-autodiff -- --help
  run_cmd zig build -Dmetal=true "-Doptimize=${zig_optimize}" -Druntime-test-filter=true test -- "metal_compute: device transpose matches gliner2 flattened attention key shape"
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    run_train "${out_dir}-smoke" "${extra_args[@]}"
  else
    run_train "${out_dir}-smoke"
  fi
  env_args+=("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK=1")
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    run_train "${out_dir}-full-parity" "${extra_args[@]}"
  else
    run_train "${out_dir}-full-parity"
  fi
  unset 'env_args[${#env_args[@]}-1]'
  run_readiness_smoke
else
  if [[ "${#env_args[@]}" -gt 0 ]]; then
    run_env_cmd "${env_args[@]}" -- "${cmd[@]}"
  else
    run_cmd "${cmd[@]}"
  fi
fi
