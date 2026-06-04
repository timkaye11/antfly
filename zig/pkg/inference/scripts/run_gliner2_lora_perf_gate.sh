#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gliner2_lora_perf_gate.sh [options] [-- extra compare args]

Runs the repeated GLiNER2 LoRA Python/Zig benchmark harness with stable Metal
training defaults and optional pass/fail thresholds.

Options:
  --runs N                         Repeated comparison runs (default: 3)
  --out-dir DIR                    Output directory (default: /private/tmp/termite-gliner2-lora-perf-gate)
  --model-dir DIR                  Zig model dir (default: /private/tmp/termite-models/gliner2)
  --python-model PATH_OR_ID        Python model path/id (default: model dir)
  --train-data FILE                Training JSONL (default: production diagnostic train data)
  --python-bin FILE                Python executable (default: /private/tmp/gliner2-parity-venv/bin/python)
  --include-python                 Run upstream Python side as timing target
  --op-stats                       Enable Metal partition op-stats collection
  --op-runs                        Enable grouped-dot candidate shape summaries
  --max-zig-median-ms N            Fail if Zig median trainer ms exceeds N
  --max-host-output-median N       Fail if median host outputs exceeds N
  --max-fallback-median N          Fail if median interpreter fallbacks exceeds N
  --max-dot-general-count-median N Fail if median dot_general command count exceeds N
  --max-dot-general-ms-median N    Fail if median dot_general total ms exceeds N
  --require-zig-beats-python       Fail unless Zig median step beats Python median step
  --require-loss-parity            Fail unless all runs report valid_loss_parity=true
  --help                           Show this help

Examples:
  scripts/run_gliner2_lora_perf_gate.sh --op-stats --max-host-output-median 388
  scripts/run_gliner2_lora_perf_gate.sh --include-python --require-zig-beats-python
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

runs=3
out_dir="/private/tmp/termite-gliner2-lora-perf-gate"
model_dir="/private/tmp/termite-models/gliner2"
python_model=""
train_data="/private/tmp/termite-gliner2-production-diagnostic/train.jsonl"
python_bin="/private/tmp/gliner2-parity-venv/bin/python"
include_python=0
op_stats=0
op_runs=0
max_zig_median_ms=""
max_host_output_median=""
max_fallback_median=""
max_dot_general_count_median=""
max_dot_general_ms_median=""
require_zig_beats_python=0
require_loss_parity=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      runs="${2:?missing value for --runs}"
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
    --python-model)
      python_model="${2:?missing value for --python-model}"
      shift 2
      ;;
    --train-data)
      train_data="${2:?missing value for --train-data}"
      shift 2
      ;;
    --python-bin)
      python_bin="${2:?missing value for --python-bin}"
      shift 2
      ;;
    --include-python)
      include_python=1
      shift
      ;;
    --op-stats)
      op_stats=1
      shift
      ;;
    --op-runs)
      op_runs=1
      shift
      ;;
    --max-zig-median-ms)
      max_zig_median_ms="${2:?missing value for --max-zig-median-ms}"
      shift 2
      ;;
    --max-host-output-median)
      max_host_output_median="${2:?missing value for --max-host-output-median}"
      shift 2
      ;;
    --max-fallback-median)
      max_fallback_median="${2:?missing value for --max-fallback-median}"
      shift 2
      ;;
    --max-dot-general-count-median)
      max_dot_general_count_median="${2:?missing value for --max-dot-general-count-median}"
      shift 2
      ;;
    --max-dot-general-ms-median)
      max_dot_general_ms_median="${2:?missing value for --max-dot-general-ms-median}"
      shift 2
      ;;
    --require-zig-beats-python)
      require_zig_beats_python=1
      shift
      ;;
    --require-loss-parity)
      require_loss_parity=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${python_model}" ]]; then
  python_model="${model_dir}"
fi

bench_args=(
  "--runs" "${runs}"
  "--out-dir" "${out_dir}"
)
if (( op_stats )); then
  bench_args+=("--op-stats")
fi
if (( op_runs )); then
  bench_args+=("--op-runs")
fi
if [[ -n "${max_zig_median_ms}" ]]; then
  bench_args+=("--max-zig-median-ms" "${max_zig_median_ms}")
fi
if [[ -n "${max_host_output_median}" ]]; then
  bench_args+=("--max-host-output-median" "${max_host_output_median}")
fi
if [[ -n "${max_fallback_median}" ]]; then
  bench_args+=("--max-fallback-median" "${max_fallback_median}")
fi
if [[ -n "${max_dot_general_count_median}" ]]; then
  bench_args+=("--max-dot-general-count-median" "${max_dot_general_count_median}")
fi
if [[ -n "${max_dot_general_ms_median}" ]]; then
  bench_args+=("--max-dot-general-ms-median" "${max_dot_general_ms_median}")
fi
if (( require_zig_beats_python )); then
  bench_args+=("--require-zig-beats-python")
fi
if (( require_loss_parity )); then
  bench_args+=("--require-loss-parity")
fi

compare_args=(
  "--zig-backend" "metal"
  "--zig-build-metal"
  "--model-dir" "${model_dir}"
  "--python-model" "${python_model}"
  "--python-bin" "${python_bin}"
  "--train-data" "${train_data}"
  "--entity-types" "person,organization,location"
  "--steps" "1"
  "--seq-len" "16"
  "--max-span-width" "2"
  "--lora-rank" "1"
  "--lora-alpha" "2"
  "--lora-dropout" "0"
  "--span-positive-weight" "1"
  "--span-negative-weight" "1"
  "--span-hard-negative-weight" "1"
  "--span-loss-reduction" "sum"
  "--timeout-seconds" "900"
)
if (( include_python )); then
  compare_args+=("--perf-target-only-python")
else
  compare_args+=("--skip-python")
fi
compare_args+=("${extra_args[@]}")

cd "${repo_root}"
exec python3 "${script_dir}/benchmark_gliner2_lora_perf.py" "${bench_args[@]}" -- "${compare_args[@]}"
