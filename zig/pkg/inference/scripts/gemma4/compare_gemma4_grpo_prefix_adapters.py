#!/usr/bin/env python3
"""Compare matched-horizon Metal trainer and MLX LoRA adapter deltas."""

from __future__ import annotations

from gemma4_files import sha256_file

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.numpy import load_file

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_gemma4_lora_mlx_benchmark as locked  # noqa: E402

WEIGHT_PREFIX = "weight::"
USE_SITE = re.compile(r"^(?P<base>.+)\.use_(?P<index>[0-9]+)\.lora_(?P<role>[AB])$")


class PrefixAdapterParityError(RuntimeError):
    """A matched-prefix adapter artifact or numerical gate drifted."""


def _adapter_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise PrefixAdapterParityError(f"adapter must not be a symlink: {candidate}")
    if candidate.is_dir():
        candidate = candidate / "adapter_model.safetensors"
    if candidate.is_symlink() or not candidate.is_file():
        raise PrefixAdapterParityError(f"adapter is not a regular file: {candidate}")
    return candidate.resolve()


def _regular_json(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise PrefixAdapterParityError(f"{label} is not a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrefixAdapterParityError(
            f"{label} is not valid JSON: {resolved}"
        ) from exc
    if not isinstance(payload, dict):
        raise PrefixAdapterParityError(f"{label} must contain a JSON object")
    return resolved, payload


def validate_mlx_run(
    status_path: Path,
    mlx_adapter_path: Path,
    expected_groups: int,
    expected_optimizer_steps: int,
    expected_tensors: int,
) -> Mapping[str, Any]:
    """Bind an MLX adapter to its guarded run and exact replay boundary."""
    adapter_path = _adapter_path(mlx_adapter_path)
    adapter_digest = sha256_file(adapter_path)
    status_path, status = _regular_json(status_path, "MLX run status")
    if (
        status.get("schema_version") != "antfly_gemma4_guarded_execution/v1"
        or status.get("status") != "completed"
        or status.get("returncode") != 0
    ):
        raise PrefixAdapterParityError("MLX guarded run did not complete successfully")
    if (
        Path(str(status.get("adapter_output_path", ""))).expanduser().resolve()
        != adapter_path
    ):
        raise PrefixAdapterParityError("MLX status names a different adapter output")
    if status.get("adapter_output_sha256") != adapter_digest:
        raise PrefixAdapterParityError("MLX adapter digest differs from guarded status")
    report_value = status.get("output_path")
    if not isinstance(report_value, str) or not report_value:
        raise PrefixAdapterParityError("MLX guarded status lacks its run report")
    report_path, report = _regular_json(Path(report_value), "MLX run report")
    report_digest = sha256_file(report_path)
    if status.get("output_sha256") != report_digest:
        raise PrefixAdapterParityError("MLX report digest differs from guarded status")
    runner_start = status.get("runner_sha256_at_start")
    if (
        not isinstance(runner_start, str)
        or status.get("runner_sha256_at_finish") != runner_start
        or report.get("runner_sha256") != f"sha256:{runner_start}"
    ):
        raise PrefixAdapterParityError(
            "MLX runner identity is not stable across the guarded run"
        )
    contract = report.get("contract")
    trace = report.get("mlx", {}).get("trace_replay", {}).get("training")
    adapter = trace.get("adapter_output") if isinstance(trace, dict) else None
    if (
        report.get("schema_version") != "antfly_gemma4_grpo_boolq_mlx_multitoken/v2"
        or report.get("status") != "diagnostic-lane-completed"
        or not isinstance(contract, dict)
        or contract.get("execution_lane") != "trace-replay"
        or contract.get("executed_train_groups") != expected_groups
        or not isinstance(trace, dict)
        or trace.get("mode") != "trace_replay"
        or trace.get("groups") != expected_groups
        or trace.get("optimizer_steps") != expected_optimizer_steps
        or not isinstance(adapter, dict)
        or Path(str(adapter.get("path", ""))).expanduser().resolve() != adapter_path
        or adapter.get("sha256") != adapter_digest
        or adapter.get("tensor_count") != expected_tensors
    ):
        raise PrefixAdapterParityError(
            "MLX report is not the requested trace-replay boundary"
        )
    return {
        "status_path": str(status_path),
        "status_sha256": sha256_file(status_path),
        "report_path": str(report_path),
        "report_sha256": report_digest,
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_digest,
    }


def _checkpoint_candidate(slot_name: str) -> str:
    use_site = USE_SITE.fullmatch(slot_name)
    if use_site:
        return (
            f"{use_site.group('base')}.loop_{use_site.group('index')}."
            f"lora_{use_site.group('role')}.weight"
        )
    if slot_name.endswith((".lora_A", ".lora_B")):
        return slot_name + ".weight"
    raise PrefixAdapterParityError(f"unsupported Metal trainer slot: {slot_name}")


def _finite_f32(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or not np.all(np.isfinite(array)):
        raise PrefixAdapterParityError(f"{label} must be finite float32")
    return array


def compare_adapter_files(
    seed_path: Path,
    metal_checkpoint_path: Path,
    mlx_path: Path,
) -> Mapping[str, Any]:
    seed_path = _adapter_path(seed_path)
    mlx_path = _adapter_path(mlx_path)
    metal_checkpoint_path = metal_checkpoint_path.expanduser().resolve()
    if metal_checkpoint_path.is_symlink() or not metal_checkpoint_path.is_file():
        raise PrefixAdapterParityError("Metal trainer checkpoint is not a regular file")
    seed = load_file(str(seed_path))
    mlx = load_file(str(mlx_path))
    checkpoint = load_file(str(metal_checkpoint_path))
    if set(mlx) != set(seed):
        raise PrefixAdapterParityError("MLX and seed adapter inventories differ")
    seed_by_identity: dict[tuple[str, str], str] = {}
    for name in seed:
        identity = locked.canonicalize_adapter_tensor_name(name)
        if identity in seed_by_identity:
            raise PrefixAdapterParityError(
                f"duplicate seed adapter identity: {identity}"
            )
        seed_by_identity[identity] = name
    metal_by_seed_name: dict[str, np.ndarray] = {}
    for checkpoint_name, values in checkpoint.items():
        if not checkpoint_name.startswith(WEIGHT_PREFIX):
            continue
        candidate = _checkpoint_candidate(checkpoint_name[len(WEIGHT_PREFIX) :])
        identity = locked.canonicalize_adapter_tensor_name(candidate)
        try:
            seed_name = seed_by_identity[identity]
        except KeyError as exc:
            raise PrefixAdapterParityError(
                f"Metal checkpoint tensor is outside the seed inventory: {checkpoint_name}"
            ) from exc
        if seed_name in metal_by_seed_name:
            raise PrefixAdapterParityError(
                f"duplicate Metal checkpoint identity: {identity}"
            )
        seed_values = _finite_f32(seed[seed_name], f"seed tensor {seed_name}")
        flat = _finite_f32(values, f"Metal tensor {checkpoint_name}")
        if flat.size != seed_values.size:
            raise PrefixAdapterParityError(
                f"Metal checkpoint size differs for {seed_name}"
            )
        metal_by_seed_name[seed_name] = flat.reshape(seed_values.shape)
    if set(metal_by_seed_name) != set(seed):
        missing = sorted(set(seed) - set(metal_by_seed_name))
        raise PrefixAdapterParityError(
            f"Metal checkpoint is missing {len(missing)} adapter tensors: {missing[:3]}"
        )

    metal_squares = mlx_squares = dot = difference_squares = 0.0
    max_abs_difference = 0.0
    nonzero_metal = nonzero_mlx = 0
    per_tensor: list[dict[str, Any]] = []
    for name in sorted(seed):
        initial = _finite_f32(seed[name], f"seed tensor {name}")
        metal_final = metal_by_seed_name[name]
        mlx_final = _finite_f32(mlx[name], f"MLX tensor {name}")
        if metal_final.shape != initial.shape or mlx_final.shape != initial.shape:
            raise PrefixAdapterParityError(f"adapter tensor shape differs for {name}")
        metal_delta = np.subtract(metal_final, initial, dtype=np.float32)
        mlx_delta = np.subtract(mlx_final, initial, dtype=np.float32)
        difference = np.subtract(mlx_delta, metal_delta, dtype=np.float32)
        metal_sq = float(np.sum(metal_delta.astype(np.float64) ** 2))
        mlx_sq = float(np.sum(mlx_delta.astype(np.float64) ** 2))
        tensor_dot = float(
            np.sum(metal_delta.astype(np.float64) * mlx_delta.astype(np.float64))
        )
        difference_sq = float(np.sum(difference.astype(np.float64) ** 2))
        tensor_max = float(np.max(np.abs(difference))) if difference.size else 0.0
        metal_squares += metal_sq
        mlx_squares += mlx_sq
        dot += tensor_dot
        difference_squares += difference_sq
        max_abs_difference = max(max_abs_difference, tensor_max)
        nonzero_metal += metal_sq > 0.0
        nonzero_mlx += mlx_sq > 0.0
        metal_norm = math.sqrt(metal_sq)
        mlx_norm = math.sqrt(mlx_sq)
        difference_norm = math.sqrt(difference_sq)
        per_tensor.append(
            {
                "name": name,
                "elements": int(initial.size),
                "metal_delta_l2": metal_norm,
                "mlx_delta_l2": mlx_norm,
                "delta_cosine_similarity": (
                    tensor_dot / (metal_norm * mlx_norm)
                    if metal_norm > 0.0 and mlx_norm > 0.0
                    else None
                ),
                "delta_vector_l2_error": difference_norm,
                "delta_vector_l2_relative_error": (
                    difference_norm / metal_norm if metal_norm > 0.0 else None
                ),
                "delta_max_abs_difference": tensor_max,
            }
        )
    metal_norm = math.sqrt(metal_squares)
    mlx_norm = math.sqrt(mlx_squares)
    difference_norm = math.sqrt(difference_squares)
    if metal_norm == 0.0 or mlx_norm == 0.0:
        raise PrefixAdapterParityError("matched adapters contain a zero global update")
    metrics = {
        "tensor_count": len(seed),
        "nonzero_metal_delta_tensors": nonzero_metal,
        "nonzero_mlx_delta_tensors": nonzero_mlx,
        "metal_delta_l2": metal_norm,
        "mlx_delta_l2": mlx_norm,
        "delta_cosine_similarity": dot / (metal_norm * mlx_norm),
        "delta_l2_relative_difference": abs(mlx_norm - metal_norm) / metal_norm,
        "delta_vector_l2_error": difference_norm,
        "delta_vector_l2_relative_error": difference_norm / metal_norm,
        "delta_max_abs_difference": max_abs_difference,
    }
    checks = {
        "adapter_delta_direction": metrics["delta_cosine_similarity"] >= 0.95,
        "adapter_delta_norm": metrics["delta_l2_relative_difference"] <= 0.1,
        "adapter_delta_vector": metrics["delta_vector_l2_relative_error"] <= 0.1,
    }
    return {
        "passed": all(checks.values()),
        "thresholds": {"min_cosine": 0.95, "max_relative_error": 0.1},
        "checks": checks,
        "metrics": metrics,
        "per_tensor": per_tensor,
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise PrefixAdapterParityError(f"output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seed-adapter", type=Path, required=True)
    result.add_argument("--metal-checkpoint", type=Path, required=True)
    result.add_argument("--metal-capture-summary", type=Path, required=True)
    result.add_argument("--mlx-adapter", type=Path, required=True)
    result.add_argument("--mlx-run-status", type=Path, required=True)
    result.add_argument("--expected-groups", type=int, default=260)
    result.add_argument("--expected-optimizer-steps", type=int, default=242)
    result.add_argument("--expected-tensors", type=int, default=686)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    summary_path = args.metal_capture_summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    checkpoint_path = args.metal_checkpoint.expanduser().resolve()
    state = summary.get("checkpoint_state")
    if not isinstance(state, dict):
        raise PrefixAdapterParityError("Metal capture summary lacks checkpoint state")
    if (
        summary.get("status") != "captured"
        or state.get("examples_into_epoch") != args.expected_groups
        or state.get("examples_seen") != args.expected_groups
        or state.get("optimizer_steps") != args.expected_optimizer_steps
    ):
        raise PrefixAdapterParityError(
            "Metal capture is not the requested group boundary"
        )
    checkpoint_artifact = summary.get("artifacts", {}).get("checkpoint", {})
    if checkpoint_artifact.get("sha256") != sha256_file(checkpoint_path):
        raise PrefixAdapterParityError(
            "Metal checkpoint digest differs from capture summary"
        )
    mlx_run = validate_mlx_run(
        args.mlx_run_status,
        args.mlx_adapter,
        args.expected_groups,
        args.expected_optimizer_steps,
        args.expected_tensors,
    )
    comparison = compare_adapter_files(
        args.seed_adapter,
        checkpoint_path,
        args.mlx_adapter,
    )
    if comparison["metrics"]["tensor_count"] != args.expected_tensors:
        raise PrefixAdapterParityError("adapter tensor count differs from expectation")
    payload = {
        "schema_version": "antfly_gemma4_grpo_matched_prefix_adapter_parity/v1",
        "status": "passed" if comparison["passed"] else "failed-parity",
        "scope": f"E4B seed-17 matched {args.expected_groups}-group Metal/MLX adapter deltas",
        "training_boundary": {
            "groups": args.expected_groups,
            "optimizer_steps": args.expected_optimizer_steps,
        },
        "comparison": comparison,
        "artifacts": {
            "seed_adapter": {
                "path": str(_adapter_path(args.seed_adapter)),
                "sha256": sha256_file(_adapter_path(args.seed_adapter)),
            },
            "metal_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "metal_capture_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "mlx_adapter": {
                "path": str(_adapter_path(args.mlx_adapter)),
                "sha256": sha256_file(_adapter_path(args.mlx_adapter)),
            },
            "mlx_run_status": {
                "path": mlx_run["status_path"],
                "sha256": mlx_run["status_sha256"],
            },
            "mlx_run_report": {
                "path": mlx_run["report_path"],
                "sha256": mlx_run["report_sha256"],
            },
        },
    }
    write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output.expanduser().resolve()),
                **comparison["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
