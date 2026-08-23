#!/usr/bin/env python3
"""Qualify exact Gemma4 LoRA recovery after a real process interruption.

The qualification runs the same bound training trajectory three ways:

1. uninterrupted, with an epoch-boundary checkpoint enabled;
2. interrupted by SIGTERM immediately after the requested checkpoint becomes
   durable; and
3. resumed from that checkpoint in a fresh immutable output directory.

A PASS requires byte-identical final adapter tensors, matching trajectory
metrics after the recovery boundary, and strict-Metal evidence with no native
or interpreter fallback. The script is standard-library-only so the release
gate does not inherit a Python ML framework dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "antfly_gemma4_metal_resume_qualification/v2"
CHECKPOINT_TENSOR = "__trainer_state_v2"
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_FIELD_COUNT = 18
CHECKPOINT_FLOAT_COUNT = CHECKPOINT_FIELD_COUNT * 4
STRICT_METAL_ENV = {
    "TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR": "1",
    "TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR": "0",
    "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK": "0",
    "TERMITE_DEBUG_DEVICE_GRAD_NORM": "0",
}
EXPERIMENTAL_GGUF_QLORA_ENV = {
    "ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA": "1",
}
TRAJECTORY_FIELDS = (
    "examples_seen",
    "supervised_tokens_seen",
    "teacher_examples_seen",
    "teacher_supervised_tokens_seen",
    "mean_teacher_temperature",
    "average_loss",
    "mean_grad_norm",
    "optimizer_steps",
    "graph_executor_steps",
    "graph_executor_fallback_steps",
    "graph_executor_native_partitions",
    "graph_executor_unsupported_ops",
    "graph_executor_interpreter_fallbacks",
    "graph_executor_true_host_outputs",
    "metal_optimizer_steps",
)


class ContractError(RuntimeError):
    """The qualification could not establish the production contract."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where}: expected finite number")
    return result


def _regular_file(path: Path, where: str, *, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ContractError(f"{where}: not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ContractError(f"{where}: not executable: {resolved}")
    return resolved


def _directory(path: Path, where: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ContractError(f"{where}: not a directory: {resolved}")
    return resolved


def _model_artifact(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_dir():
        return resolved
    mode = resolved.stat().st_mode
    if stat.S_ISREG(mode) and resolved.suffix.lower() == ".gguf":
        return resolved
    raise ContractError(f"model artifact: expected a model directory or GGUF file: {resolved}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{where}: invalid JSON at {path}: {exc}") from exc
    return _mapping(value, where)


def _tree_snapshot(root: Path) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        info = path.lstat()
        snapshot.append(
            {
                "path": "." if path == root else path.relative_to(root).as_posix(),
                "kind": "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file",
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "inode": info.st_ino,
                "mode": stat.S_IMODE(info.st_mode),
            }
        )
    return snapshot


def _snapshot_digest(snapshot: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def inspect_training_checkpoint(path: Path) -> dict[str, Any]:
    """Read the bound progress tensor without importing safetensors/numpy."""

    checkpoint = _regular_file(path, "training checkpoint")
    file_size = checkpoint.stat().st_size
    with checkpoint.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ContractError("training checkpoint: truncated SafeTensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > 64 * 1024 * 1024 or 8 + header_size > file_size:
            raise ContractError("training checkpoint: invalid SafeTensors header length")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ContractError("training checkpoint: truncated SafeTensors header")
        try:
            header = _mapping(json.loads(raw_header.decode("utf-8").rstrip(" ")), "checkpoint header")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"training checkpoint: invalid SafeTensors header: {exc}") from exc
        descriptor = _mapping(header.get(CHECKPOINT_TENSOR), CHECKPOINT_TENSOR)
        if descriptor.get("dtype") != "F32" or descriptor.get("shape") != [CHECKPOINT_FLOAT_COUNT]:
            raise ContractError(f"{CHECKPOINT_TENSOR}: unexpected dtype or shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] - offsets[0] != CHECKPOINT_FLOAT_COUNT * 4
            or 8 + header_size + offsets[1] > file_size
        ):
            raise ContractError(f"{CHECKPOINT_TENSOR}: invalid data offsets")
        handle.seek(8 + header_size + offsets[0])
        raw_values = handle.read(CHECKPOINT_FLOAT_COUNT * 4)
    values = struct.unpack(f"<{CHECKPOINT_FLOAT_COUNT}f", raw_values)
    fields: list[int] = []
    for field_index in range(CHECKPOINT_FIELD_COUNT):
        field = 0
        for chunk_index in range(4):
            value = values[field_index * 4 + chunk_index]
            if not math.isfinite(value) or value < 0 or value > 65535 or value != math.floor(value):
                raise ContractError(f"{CHECKPOINT_TENSOR}: invalid encoded integer")
            field |= int(value) << (chunk_index * 16)
        fields.append(field)
    if fields[0] != CHECKPOINT_SCHEMA_VERSION:
        raise ContractError(f"{CHECKPOINT_TENSOR}: unsupported schema {fields[0]}")
    if fields[2] > fields[1] or fields[4] >= max(fields[5], 1):
        raise ContractError(f"{CHECKPOINT_TENSOR}: inconsistent counters")
    return {
        "schema_version": fields[0],
        "micro_batch_steps": fields[1],
        "optimizer_steps": fields[2],
        "stochastic_steps": fields[3],
        "accumulation_micro_batches": fields[4],
        "configured_accumulation_steps": fields[5],
        "trainer_seed": fields[6],
        "epoch_index": fields[7],
        "next_example_index": fields[8],
        "examples_seen": fields[9],
        "order_seed": fields[10],
        "order_cursor": fields[11],
        "rng_state": fields[12:16],
        "conditional_family_count": fields[16],
        "conditional_family_present_mask": fields[17],
        "size_bytes": file_size,
        "sha256": _sha256(checkpoint),
    }


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")


def _run_to_completion(
    command: Sequence[str],
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            result = subprocess.run(
                command,
                env=dict(env),
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError(f"training command exceeded {timeout_seconds:g}s") from exc
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise ContractError(
            f"training command failed with {result.returncode}; stderr tail:\n{_tail(stderr_path)}"
        )
    return {"returncode": result.returncode, "elapsed_seconds": elapsed}


def _run_and_interrupt(
    command: Sequence[str],
    env: Mapping[str, str],
    checkpoint_path: Path,
    expected_epoch: int,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, env=dict(env), stdout=stdout, stderr=stderr)
        state: dict[str, Any] | None = None
        try:
            while time.monotonic() - started < timeout_seconds:
                returncode = process.poll()
                if returncode is not None:
                    raise ContractError(
                        "interruption target exited before the requested checkpoint "
                        f"(returncode={returncode}); stderr tail:\n{_tail(stderr_path)}"
                    )
                if checkpoint_path.exists():
                    try:
                        candidate = inspect_training_checkpoint(checkpoint_path)
                    except ContractError:
                        candidate = None
                    if candidate is not None:
                        epoch = _integer(candidate["epoch_index"], "checkpoint epoch")
                        if epoch > expected_epoch:
                            raise ContractError(
                                f"checkpoint advanced past interruption boundary: {epoch} > {expected_epoch}"
                            )
                        if epoch == expected_epoch:
                            state = candidate
                            break
                time.sleep(poll_seconds)
            if state is None:
                raise ContractError(
                    f"checkpoint epoch {expected_epoch} was not observed within {timeout_seconds:g}s"
                )
            process.send_signal(signal.SIGTERM)
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=15)
            if returncode == 0:
                raise ContractError("interruption target completed successfully before SIGTERM took effect")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
    final_state = inspect_training_checkpoint(checkpoint_path)
    if final_state["sha256"] != state["sha256"]:
        raise ContractError("checkpoint changed after the observed interruption boundary")
    if final_state["accumulation_micro_batches"] != 0:
        raise ContractError("checkpoint is not at a gradient-accumulation boundary")
    if final_state["next_example_index"] != 0 or final_state["order_cursor"] != 0:
        raise ContractError("checkpoint is not at an epoch boundary")
    return (
        {
            "pid": process.pid,
            "returncode": returncode,
            "signal": -returncode if returncode < 0 else None,
            "elapsed_seconds": time.monotonic() - started,
        },
        final_state,
    )


def _training_report(output_dir: Path, where: str) -> Mapping[str, Any]:
    path = _regular_file(output_dir / "training_report.json", f"{where} training report")
    payload = _load_json(path, f"{where} training report")
    report = _mapping(payload.get("report"), f"{where}.report")
    if payload.get("task") != "gemma4_lora_train_eval":
        raise ContractError(f"{where}: unexpected training task")
    if report.get("trainer_kind") != "real_autodiff_causal_lm_v1":
        raise ContractError(f"{where}: production text autodiff trainer was not used")
    if report.get("backend_kind") != "metal":
        raise ContractError(f"{where}: backend is not Metal")
    return report


def _strict_metal_epoch(epoch: Mapping[str, Any], where: str) -> None:
    _integer(epoch.get("optimizer_steps"), f"{where}.optimizer_steps", 1)
    _integer(epoch.get("metal_optimizer_steps"), f"{where}.metal_optimizer_steps", 1)
    _integer(epoch.get("graph_executor_steps"), f"{where}.graph_executor_steps", 1)
    for field in (
        "graph_executor_fallback_steps",
        "graph_executor_native_partitions",
        "graph_executor_unsupported_ops",
        "graph_executor_interpreter_fallbacks",
        "graph_executor_true_host_outputs",
    ):
        if _integer(epoch.get(field), f"{where}.{field}") != 0:
            raise ContractError(f"{where}: strict-Metal violation in {field}")
    for field in ("average_loss", "mean_grad_norm"):
        _finite_number(epoch.get(field), f"{where}.{field}")


def _trajectory_view(epoch: Mapping[str, Any], where: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in TRAJECTORY_FIELDS:
        value = epoch.get(field)
        if field in ("mean_teacher_temperature", "average_loss", "mean_grad_norm"):
            result[field] = _finite_number(value, f"{where}.{field}")
        else:
            result[field] = _integer(value, f"{where}.{field}")
    return result


def _validate_outputs(
    uninterrupted_dir: Path,
    resumed_dir: Path,
    expected_resume_epoch: int,
    total_epochs: int,
    direct_gguf_model: bool,
) -> dict[str, Any]:
    uninterrupted = _training_report(uninterrupted_dir, "uninterrupted")
    resumed = _training_report(resumed_dir, "resumed")
    fingerprint = uninterrupted.get("run_fingerprint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ContractError("uninterrupted: invalid run fingerprint")
    if resumed.get("run_fingerprint_sha256") != fingerprint:
        raise ContractError("resumed trajectory has a different run fingerprint")
    if _integer(uninterrupted.get("epochs"), "uninterrupted.epochs", 1) != total_epochs:
        raise ContractError("uninterrupted report has the wrong epoch count")
    if _integer(resumed.get("epochs"), "resumed.epochs", 1) != total_epochs:
        raise ContractError("resumed report has the wrong epoch count")
    resume_contract = _mapping(resumed.get("checkpoint_resume"), "resumed.checkpoint_resume")
    if resume_contract.get("enabled") is not True:
        raise ContractError("resumed report does not attest checkpoint recovery")
    if _integer(resume_contract.get("start_epoch"), "resumed start_epoch") != expected_resume_epoch:
        raise ContractError("resumed report recovered from the wrong epoch")

    uninterrupted_history_raw = uninterrupted.get("epoch_history")
    resumed_history_raw = resumed.get("epoch_history")
    if not isinstance(uninterrupted_history_raw, list) or len(uninterrupted_history_raw) != total_epochs:
        raise ContractError("uninterrupted report has an incomplete epoch history")
    if not isinstance(resumed_history_raw, list) or len(resumed_history_raw) != total_epochs - expected_resume_epoch:
        raise ContractError("resumed report has an incomplete recovery suffix")
    uninterrupted_history = [
        _mapping(item, f"uninterrupted.epoch_history[{index}]")
        for index, item in enumerate(uninterrupted_history_raw)
    ]
    resumed_history = [
        _mapping(item, f"resumed.epoch_history[{index}]")
        for index, item in enumerate(resumed_history_raw)
    ]
    for index, epoch in enumerate(uninterrupted_history):
        _strict_metal_epoch(epoch, f"uninterrupted.epoch_history[{index}]")
    for index, epoch in enumerate(resumed_history):
        _strict_metal_epoch(epoch, f"resumed.epoch_history[{index}]")
    expected_suffix = [
        _trajectory_view(epoch, f"uninterrupted.epoch_history[{expected_resume_epoch + index}]")
        for index, epoch in enumerate(uninterrupted_history[expected_resume_epoch:])
    ]
    actual_suffix = [
        _trajectory_view(epoch, f"resumed.epoch_history[{index}]")
        for index, epoch in enumerate(resumed_history)
    ]
    if actual_suffix != expected_suffix:
        raise ContractError("post-resume optimizer trajectory differs from uninterrupted execution")

    uninterrupted_adapter = _regular_file(
        uninterrupted_dir / "adapter_model.safetensors", "uninterrupted adapter"
    )
    resumed_adapter = _regular_file(resumed_dir / "adapter_model.safetensors", "resumed adapter")
    uninterrupted_sha = _sha256(uninterrupted_adapter)
    resumed_sha = _sha256(resumed_adapter)
    if resumed_sha != uninterrupted_sha:
        raise ContractError("resumed final adapter is not byte-identical to uninterrupted output")
    for relative in (
        "adapter_config.json",
        "antfly_finetune_manifest.json",
        "run_manifest.json",
        "train_eval_report.json",
    ):
        _regular_file(uninterrupted_dir / relative, f"uninterrupted {relative}")
        _regular_file(resumed_dir / relative, f"resumed {relative}")
    for relative in ("tokenizer.json", "tokenizer_config.json"):
        uninterrupted_path = uninterrupted_dir / relative
        resumed_path = resumed_dir / relative
        if uninterrupted_path.exists() != resumed_path.exists():
            raise ContractError(f"{relative}: uninterrupted/resumed presence mismatch")
        if uninterrupted_path.exists():
            _regular_file(uninterrupted_path, f"uninterrupted {relative}")
            _regular_file(resumed_path, f"resumed {relative}")
        elif not direct_gguf_model:
            raise ContractError(f"{relative}: required for a model-directory run")
    return {
        "run_fingerprint_sha256": "sha256:" + fingerprint,
        "adapter_model_sha256": uninterrupted_sha,
        "adapter_model_size_bytes": uninterrupted_adapter.stat().st_size,
        "uninterrupted_epoch_history": [
            _trajectory_view(epoch, f"uninterrupted.epoch_history[{index}]")
            for index, epoch in enumerate(uninterrupted_history)
        ],
        "resumed_epoch_history": actual_suffix,
        "final_eval": resumed.get("after"),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ContractError(f"refusing to replace report: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    checkpoint_path: Path,
    resume: bool,
) -> list[str]:
    command = [
        str(args.binary),
        "inference",
        "finetune",
        "train",
        "gemma4-lora",
        "--model",
        str(args.model),
        "--adapter",
        str(args.adapter),
        "--train-prepared",
        str(args.train_prepared),
        "--eval-prepared",
        str(args.eval_prepared),
        "--out",
        str(output_dir),
        "--backend",
        "metal",
        "--trainer",
        "autodiff",
        "--lr",
        format(args.learning_rate, ".9g"),
        "--max-examples",
        str(args.max_examples),
        "--eval-max-examples",
        str(args.eval_max_examples),
        "--epochs",
        str(args.epochs),
        "--max-grad-norm",
        format(args.max_grad_norm, ".9g"),
        "--grad-accum",
        str(args.grad_accum),
        "--seed",
        str(args.seed),
        "--checkpoint-path",
        str(checkpoint_path),
        "--checkpoint-every-epochs",
        str(args.interrupt_after_epoch),
    ]
    if args.activation_checkpoint_interval:
        command.extend(
            ["--activation-checkpoint-interval", str(args.activation_checkpoint_interval)]
        )
    if resume:
        command.append("--resume")
    return command


def qualify(args: argparse.Namespace) -> Mapping[str, Any]:
    args.binary = _regular_file(args.binary, "antfly binary", executable=True)
    args.model = _model_artifact(args.model)
    args.adapter = _directory(args.adapter, "adapter directory")
    args.train_prepared = _regular_file(args.train_prepared, "prepared training data")
    args.eval_prepared = _regular_file(args.eval_prepared, "prepared evaluation data")
    if args.epochs < 2:
        raise ContractError("--epochs must be at least 2")
    if not 0 < args.interrupt_after_epoch < args.epochs:
        raise ContractError("--interrupt-after-epoch must be between 1 and epochs-1")
    if args.max_examples < 1 or args.eval_max_examples < 1 or args.grad_accum < 1:
        raise ContractError("example and accumulation counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ContractError("--learning-rate must be finite and positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm < 0:
        raise ContractError("--max-grad-norm must be finite and nonnegative")
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ContractError("timeouts must be positive")
    if args.model.is_file() and not args.experimental_gguf_qlora:
        raise ContractError("direct GGUF qualification requires --experimental-gguf-qlora")
    if args.experimental_gguf_qlora and not args.model.is_file():
        raise ContractError("--experimental-gguf-qlora requires a direct GGUF model file")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        raise ContractError(f"output directory already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    uninterrupted_dir = output_root / "uninterrupted"
    interrupted_dir = output_root / "interrupted-unpublished"
    resumed_dir = output_root / "resumed"
    uninterrupted_checkpoint = output_root / "uninterrupted-state.safetensors"
    interrupted_checkpoint = output_root / "interrupted-state.safetensors"

    immutable_roots = {"model": args.model, "adapter": args.adapter}
    snapshots_before = {name: _tree_snapshot(path) for name, path in immutable_roots.items()}
    input_evidence = {
        "binary": {"path": str(args.binary), "sha256": _sha256(args.binary)},
        "model": {
            "path": str(args.model),
            "kind": "gguf-file" if args.model.is_file() else "directory",
            "snapshot_sha256": _snapshot_digest(snapshots_before["model"]),
            "sha256": _sha256(args.model) if args.model.is_file() else None,
        },
        "adapter": {
            "path": str(args.adapter),
            "snapshot_sha256": _snapshot_digest(snapshots_before["adapter"]),
            "checkpoint_sha256": _sha256(
                _regular_file(args.adapter / "adapter_model.safetensors", "seed adapter checkpoint")
            ),
        },
        "train_prepared": {"path": str(args.train_prepared), "sha256": _sha256(args.train_prepared)},
        "eval_prepared": {"path": str(args.eval_prepared), "sha256": _sha256(args.eval_prepared)},
    }
    env = os.environ.copy()
    env.update(STRICT_METAL_ENV)
    effective_contract_env = dict(STRICT_METAL_ENV)
    if args.experimental_gguf_qlora:
        env.update(EXPERIMENTAL_GGUF_QLORA_ENV)
        effective_contract_env.update(EXPERIMENTAL_GGUF_QLORA_ENV)

    uninterrupted_command = _command(
        args,
        output_dir=uninterrupted_dir,
        checkpoint_path=uninterrupted_checkpoint,
        resume=False,
    )
    uninterrupted_run = _run_to_completion(
        uninterrupted_command,
        env,
        output_root / "uninterrupted.stdout.log",
        output_root / "uninterrupted.stderr.log",
        args.timeout_seconds,
    )
    uninterrupted_state = inspect_training_checkpoint(uninterrupted_checkpoint)
    if uninterrupted_state["epoch_index"] != args.epochs:
        raise ContractError("uninterrupted checkpoint did not reach the final epoch")

    interrupted_command = _command(
        args,
        output_dir=interrupted_dir,
        checkpoint_path=interrupted_checkpoint,
        resume=False,
    )
    interrupted_run, interrupted_state = _run_and_interrupt(
        interrupted_command,
        env,
        interrupted_checkpoint,
        args.interrupt_after_epoch,
        output_root / "interrupted.stdout.log",
        output_root / "interrupted.stderr.log",
        args.timeout_seconds,
        args.poll_seconds,
    )
    if interrupted_dir.exists():
        raise ContractError("interrupted command published its immutable output directory")

    resumed_command = _command(
        args,
        output_dir=resumed_dir,
        checkpoint_path=interrupted_checkpoint,
        resume=True,
    )
    resumed_run = _run_to_completion(
        resumed_command,
        env,
        output_root / "resumed.stdout.log",
        output_root / "resumed.stderr.log",
        args.timeout_seconds,
    )
    final_resumed_state = inspect_training_checkpoint(interrupted_checkpoint)
    if final_resumed_state["epoch_index"] != args.epochs:
        raise ContractError("resumed checkpoint did not reach the final epoch")
    parity = _validate_outputs(
        uninterrupted_dir,
        resumed_dir,
        args.interrupt_after_epoch,
        args.epochs,
        args.model.is_file(),
    )

    for name, root in immutable_roots.items():
        after = _tree_snapshot(root)
        if after != snapshots_before[name]:
            raise ContractError(f"immutable {name} input changed during qualification")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "contract": {
            "backend": "metal",
            "strict_metal_environment": effective_contract_env,
            "experimental_direct_gguf_qlora": args.experimental_gguf_qlora,
            "epochs": args.epochs,
            "interrupt_after_epoch": args.interrupt_after_epoch,
            "learning_rate": args.learning_rate,
            "max_examples": args.max_examples,
            "eval_max_examples": args.eval_max_examples,
            "grad_accum": args.grad_accum,
            "max_grad_norm": args.max_grad_norm,
            "activation_checkpoint_interval": args.activation_checkpoint_interval,
            "seed": args.seed,
        },
        "inputs": input_evidence,
        "commands": {
            "uninterrupted": uninterrupted_command,
            "interrupted": interrupted_command,
            "resumed": resumed_command,
        },
        "runs": {
            "uninterrupted": uninterrupted_run,
            "interrupted": interrupted_run,
            "resumed": resumed_run,
        },
        "checkpoints": {
            "uninterrupted_final": uninterrupted_state,
            "interrupted_boundary": interrupted_state,
            "resumed_final": final_resumed_state,
        },
        "parity": parity,
    }
    _atomic_write_json(output_root / "qualification_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-prepared", type=Path, required=True)
    parser.add_argument("--eval-prepared", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--interrupt-after-epoch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--eval-max-examples", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--activation-checkpoint-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument(
        "--experimental-gguf-qlora",
        action="store_true",
        help="explicitly admit a direct GGUF model and bind the experimental QLoRA environment",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = qualify(parse_args(argv))
    except (ContractError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: exact interrupted/resumed Gemma4 Metal trajectory; "
        f"adapter={report['parity']['adapter_model_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
