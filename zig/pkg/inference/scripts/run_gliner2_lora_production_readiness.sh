#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gliner2_lora_production_readiness.sh [options] [-- extra compare args]

Runs the strict GLiNER2 LoRA batch-32/seq-128 production readiness gate and an
opt-in head-MLP fusion guard. The default production path decides the script
exit code; use --require-head-opt-in to also require the experimental head path.

Options:
  --runs N                 Production gate repeated runs (default: 5)
  --head-runs N            Opt-in head fusion guard runs (default: 1)
  --out-dir DIR            Output directory (default: /private/tmp/termite-gliner2-production-readiness)
  --model-dir DIR          Zig model dir forwarded to the perf gate
  --python-model PATH      Python model path/id forwarded to the perf gate
  --train-data FILE        Training JSONL forwarded to the perf gate
  --python-bin FILE        Python executable forwarded to the perf gate
  --compare-steps N        Steps per comparison run forwarded to the perf gate
  --warm-production-ready  Use strict warm-step production target defaults
  --loop-profile           Enable executor loop-profile timing summaries
  --hazard-profile         Enable planned-access hazard timing summaries
  --max-zig-python-warm-step-ratio-median N
                           Forward warm median Zig/Python ratio limit
  --max-zig-python-warm-step-ratio-any-run N
                           Forward warm any-run Zig/Python ratio limit
  --skip-head-opt-in       Skip the experimental head-MLP fusion guard
  --require-head-opt-in    Fail if the experimental head-MLP fusion guard fails
  --help                   Show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

runs=5
runs_explicit=0
head_runs=1
out_dir="/private/tmp/termite-gliner2-production-readiness"
skip_head_opt_in=0
require_head_opt_in=0
warm_production_ready=0
gate_args=()
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      runs="${2:?missing value for --runs}"
      runs_explicit=1
      shift 2
      ;;
    --head-runs)
      head_runs="${2:?missing value for --head-runs}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing value for --out-dir}"
      shift 2
      ;;
    --model-dir | --python-model | --train-data | --python-bin)
      gate_args+=("$1" "${2:?missing value for $1}")
      shift 2
      ;;
    --compare-steps | --max-zig-python-warm-step-ratio-median | --max-zig-python-warm-step-ratio-any-run)
      gate_args+=("$1" "${2:?missing value for $1}")
      shift 2
      ;;
    --warm-production-ready)
      warm_production_ready=1
      gate_args+=("$1")
      shift
      ;;
    --loop-profile | --hazard-profile)
      gate_args+=("$1")
      shift
      ;;
    --skip-head-opt-in)
      skip_head_opt_in=1
      shift
      ;;
    --require-head-opt-in)
      require_head_opt_in=1
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

if (( warm_production_ready && ! runs_explicit )); then
  runs=3
fi

mkdir -p "${out_dir}"

default_dir="${out_dir}/default"
head_dir="${out_dir}/head-mlp-opt-in"
default_rc=0
head_rc=0

default_cmd=(
  "${script_dir}/run_gliner2_lora_perf_gate.sh"
  "--production-ready"
  "--runs" "${runs}"
  "--out-dir" "${default_dir}"
)
head_cmd=(
  "${script_dir}/run_gliner2_lora_perf_gate.sh"
  "--production-batch32"
  "--op-stats"
  "--runs" "${head_runs}"
  "--out-dir" "${head_dir}"
  "--max-zig-metal-peak-live-bytes-median" "1717986918"
  "--max-zig-metal-planned-barriers-median" "40"
  "--max-zig-metal-planned-scopes-median" "55"
  "--max-command-dispatch-median" "6250"
  "--max-fallback-median" "0"
  "--max-true-host-output-median" "0"
  "--min-zig-head-mlp-forward-region-median" "1"
)
if ((${#gate_args[@]})); then
  default_cmd+=("${gate_args[@]}")
  head_cmd+=("${gate_args[@]}")
fi
if ((${#extra_args[@]})); then
  default_cmd+=("--" "${extra_args[@]}")
  head_cmd+=("--" "${extra_args[@]}")
fi

set +e
"${default_cmd[@]}"
default_rc=$?

if (( skip_head_opt_in )); then
  head_rc=0
else
  TERMITE_METAL_ENABLE_HEAD_MLP_FORWARD_RUNTIME_REGION=1 "${head_cmd[@]}"
  head_rc=$?
fi
set -e

python3 - "${out_dir}" "${default_rc}" "${head_rc}" "${skip_head_opt_in}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
default_rc = int(sys.argv[2])
head_rc = int(sys.argv[3])
skip_head_opt_in = bool(int(sys.argv[4]))


def load_summary(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"error": f"invalid json: {exc}"}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


default_summary_path = out_dir / "default" / "perf_summary.json"
head_summary_path = out_dir / "head-mlp-opt-in" / "perf_summary.json"
summary = {
    "production_ready": default_rc == 0,
    "head_opt_in_ready": None if skip_head_opt_in else head_rc == 0,
    "default_rc": default_rc,
    "head_opt_in_rc": None if skip_head_opt_in else head_rc,
    "default_summary_path": str(default_summary_path),
    "head_opt_in_summary_path": None if skip_head_opt_in else str(head_summary_path),
    "default_summary": load_summary(default_summary_path),
    "head_opt_in_summary": None if skip_head_opt_in else load_summary(head_summary_path),
}
out_path = out_dir / "readiness_summary.json"
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"readiness summary: {out_path}")
PY

if (( default_rc != 0 )); then
  exit "${default_rc}"
fi
if (( require_head_opt_in && head_rc != 0 )); then
  exit "${head_rc}"
fi
exit 0
