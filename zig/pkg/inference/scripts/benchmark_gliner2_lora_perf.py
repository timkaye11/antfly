#!/usr/bin/env python3
"""Run repeated GLiNER2 LoRA Python/Zig comparisons and summarize medians."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def metric(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def run_compare(argv: list[str], out_dir: Path, run_index: int, timeout: int | None, env: dict[str, str] | None) -> dict[str, Any]:
    run_dir = out_dir / f"run-{run_index:02d}"
    cmd = [
        sys.executable,
        str(script_dir() / "compare_gliner2_lora_python_zig.py"),
        "--out-dir",
        str(run_dir),
        "--keep-out-dir",
        *argv,
    ]
    started = time.time()
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=run_env,
    )
    report_path = run_dir / "comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return {
        "run": run_index,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "argv": cmd,
        "report_path": str(report_path),
        "summary": report.get("summary", {}),
        "output_tail": proc.stdout[-8000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat compare_gliner2_lora_python_zig.py and write median/p90 performance summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("/private/tmp/termite-gliner2-lora-perf-runs"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--op-stats", action="store_true", help="Set TERMITE_METAL_PARTITION_OP_STATS=1 for each comparison run")
    parser.add_argument("--op-runs", action="store_true", help="Set TERMITE_METAL_PARTITION_OP_RUNS=1 for grouped-dot candidate summaries")
    parser.add_argument("--max-zig-median-ms", type=float, default=None)
    parser.add_argument("--max-host-output-median", type=float, default=None)
    parser.add_argument("--max-fallback-median", type=float, default=None)
    parser.add_argument("--max-dot-general-count-median", type=float, default=None)
    parser.add_argument("--max-dot-general-ms-median", type=float, default=None)
    parser.add_argument("--require-loss-parity", action="store_true")
    parser.add_argument("--require-zig-beats-python", action="store_true")
    parser.add_argument(
        "compare_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to compare_gliner2_lora_python_zig.py. Prefix with -- before forwarded args.",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    forwarded = args.compare_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    run_env: dict[str, str] = {}
    if args.op_stats:
        run_env["TERMITE_METAL_PARTITION_OP_STATS"] = "1"
    if args.op_runs:
        run_env["TERMITE_METAL_PARTITION_OP_RUNS"] = "1"
    for idx in range(1, args.runs + 1):
        result = run_compare(forwarded, args.out_dir, idx, args.timeout_seconds, run_env)
        runs.append(result)
        print(json.dumps({
            "event": "perf_run",
            "run": idx,
            "returncode": result["returncode"],
            "summary": result["summary"],
        }, sort_keys=True))
        if result["returncode"] != 0 and not args.allow_failures:
            break

    successful = [run for run in runs if run["returncode"] == 0]
    keys = [
        "python_avg_step_wall_ms",
        "zig_avg_trainer_ms",
        "zig_total_trainer_ms",
        "zig_epoch_wall_ms",
        "zig_graph_executor_command_dispatches_avg",
        "zig_graph_executor_interpreter_fallbacks_avg",
        "zig_graph_executor_host_outputs_avg",
        "zig_graph_executor_regions_avg",
        "zig_graph_executor_runtime_region_dispatches_avg",
        "zig_graph_executor_runtime_region_plan_compiles_avg",
        "zig_graph_executor_runtime_region_plan_reuses_avg",
        "zig_graph_executor_plan_build_ms_avg",
        "zig_graph_executor_buffer_plan_build_ms_avg",
        "zig_graph_executor_plan_cache_hits_avg",
        "zig_graph_executor_plan_cache_misses_avg",
        "zig_metal_frame_wait_ms_avg",
        "zig_metal_frame_gpu_ms_avg",
        "zig_metal_last_frame_compute_encoders_avg",
        "zig_metal_last_frame_blit_encoders_avg",
        "zig_metal_last_frame_planned_scopes_avg",
        "zig_metal_last_frame_planned_barriers_avg",
        "zig_metal_last_frame_planned_command_ops_avg",
        "zig_metal_deberta_encoder_plan_attempts_avg",
        "zig_metal_deberta_encoder_plan_successes_avg",
        "zig_metal_deberta_encoder_plan_reuses_avg",
        "zig_metal_deberta_encoder_plan_failures_avg",
        "zig_metal_deberta_encoder_layer_attempts_avg",
        "zig_metal_deberta_encoder_layer_successes_avg",
        "zig_metal_deberta_encoder_layer_fallbacks_avg",
        "zig_metal_deberta_relative_qk_pair_calls_avg",
        "zig_metal_deberta_relative_qk_pair_fallbacks_avg",
        "zig_metal_deberta_ffn_fused_calls_avg",
        "zig_metal_deberta_ffn_fused_mps_matmuls_avg",
        "zig_metal_deberta_ffn_fused_fallbacks_avg",
        "zig_metal_deberta_attention_flash_calls_avg",
        "zig_metal_deberta_attention_gemm_calls_avg",
        "zig_metal_deberta_attention_gemm_fallbacks_avg",
        "zig_metal_deberta_attention_legacy_calls_avg",
        "zig_dot_general_command_count",
        "zig_dot_general_command_total_ms",
        "zig_dot_general_command_avg_ms",
        "zig_gather_fallback_count",
        "zig_gather_fallback_total_ms",
        "zig_gather_host_output_count",
        "zig_gather_host_output_total_ms",
        "zig_final_avg_loss",
        "loss_delta_zig_minus_python",
        "valid_loss_parity",
    ]
    metrics = {
        key: summarize([value for run in successful if (value := metric(run["summary"], key)) is not None])
        for key in keys
    }
    zig_median = metrics["zig_avg_trainer_ms"]["median"]
    py_median = metrics["python_avg_step_wall_ms"]["median"]
    summary = {
        "runs_requested": args.runs,
        "runs_completed": len(runs),
        "runs_successful": len(successful),
        "compare_args": forwarded,
        "metrics": metrics,
        "zig_beats_python_median_step_time": (
            zig_median < py_median if zig_median is not None and py_median is not None else None
        ),
        "op_stats_enabled": args.op_stats,
        "op_runs_enabled": args.op_runs,
        "top_dot_shapes": next((run["summary"].get("zig_top_dot_shapes") for run in successful if run["summary"].get("zig_top_dot_shapes")), []),
        "run_reports": [run["report_path"] for run in runs],
    }
    failures: list[str] = []
    if len(successful) != args.runs:
        failures.append(f"only {len(successful)}/{args.runs} runs succeeded")
    threshold_checks = [
        ("zig_avg_trainer_ms", args.max_zig_median_ms, "Zig median trainer ms"),
        ("zig_graph_executor_host_outputs_avg", args.max_host_output_median, "host output median"),
        ("zig_graph_executor_interpreter_fallbacks_avg", args.max_fallback_median, "fallback median"),
        ("zig_dot_general_command_count", args.max_dot_general_count_median, "dot_general command-count median"),
        ("zig_dot_general_command_total_ms", args.max_dot_general_ms_median, "dot_general total-ms median"),
    ]
    for key, limit, label in threshold_checks:
        median = metrics[key]["median"]
        if limit is not None and (median is None or median > limit):
            failures.append(f"{label} {median} exceeds limit {limit}")
    valid_loss_values = [run["summary"].get("valid_loss_parity") for run in successful if "valid_loss_parity" in run["summary"]]
    if args.require_loss_parity and (not valid_loss_values or not all(bool(value) for value in valid_loss_values)):
        failures.append("loss parity was required but not all successful runs reported valid_loss_parity=true")
    if args.require_zig_beats_python and summary["zig_beats_python_median_step_time"] is not True:
        failures.append("Zig median step time did not beat Python median step time")
    summary["pass"] = not failures
    summary["failures"] = failures
    out_path = args.out_dir / "perf_summary.json"
    out_path.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2), encoding="utf-8")
    print(f"perf summary: {out_path}")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
