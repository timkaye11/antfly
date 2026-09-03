#!/usr/bin/env python3
"""Export an immutable Antfly Zig Gemma4 LoRA numerical-oracle trace.

The wrapper admits a clean, revision-bound Antfly binary and immutable local
model/prepared/adapter inputs, asks the typed Zig trainer for its raw capture,
then validates and repackages that capture into ``gemma4_oracle_trace/v1``.
It never downloads a model and never replaces an existing output directory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma4_oracle_contract import (
    ANTFLY_ADAPTER_KEY_FORMAT,
    ContractError,
    LOCK_PATH,
    TRACE_SCHEMA_VERSION,
    ZIG_ORACLE_PACKAGING_PACKAGES,
    build_evidence_ledger,
    canonicalize_adapter_tensor_name,
    hardware_fingerprint,
    inspect_adapter_artifact,
    load_json,
    load_lock,
    load_prepared_example,
    lock_digest,
    prefixed_sha256,
    validate_target_inventory,
    validate_trace,
    verify_model_directory,
    verify_prepared_source_dataset,
    write_json,
)


REQUEST_SCHEMA_VERSION = "antfly_gemma4_lora_zig_oracle_request/v1"
CAPTURE_SCHEMA_VERSION = "antfly_gemma4_lora_zig_oracle_capture/v1"
RUN_MANIFEST_SCHEMA_VERSION = "antfly.gemma4.finetune.run-manifest.v1"
ARTIFACT_FAMILY_VERSION = "gemma4_lora/v1alpha1"
SCRIPT_PATH = Path(__file__).resolve()
RUNNER_RELATIVE_PATH = "zig/pkg/inference/scripts/gemma4/export_gemma4_lora_zig_oracle.py"
STRICT_METAL_ENV = {
    "TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR": "1",
    "TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR": "0",
    "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK": "0",
    "TERMITE_DEBUG_DEVICE_GRAD_NORM": "0",
}
SANITIZED_ENV_PREFIXES = ("TERMITE_", "ANTFLY_GEMMA4_")
SANITIZED_ENV_NAMES = frozenset({"ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA"})
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_VERSION_LINE = re.compile(r"antfly inference v([^\r\n]+)")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
        raise ContractError(f"{where} must be a non-empty single-line string")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where} must be an integer >= {minimum}")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where} must be finite")
    return result


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], where: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ContractError(
            f"{where} keys differ (missing={sorted(expected_set - actual)}, "
            f"unknown={sorted(actual - expected_set)})"
        )


def parse_betas(value: str) -> tuple[float, float]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("betas must be BETA1,BETA2")
    try:
        result = (float(pieces[0]), float(pieces[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("betas must be numeric") from exc
    if not all(math.isfinite(item) and 0 <= item < 1 for item in result):
        raise argparse.ArgumentTypeError("betas must be finite and in [0,1)")
    return result


def _regular_file(path: Path, where: str, *, executable: bool = False) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ContractError(f"{where} cannot be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise ContractError(f"{where} is unavailable: {candidate}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ContractError(f"{where} must be a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ContractError(f"{where} is not executable: {resolved}")
    return resolved


def _tree_snapshot(path: Path, where: str) -> tuple[tuple[str, int, int, int, int], ...]:
    root = path.expanduser()
    if root.is_symlink():
        raise ContractError(f"{where} cannot be a symbolic link: {root}")
    resolved = root.resolve(strict=True)
    candidates = [resolved] if resolved.is_file() else sorted(
        resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()
    )
    rows: list[tuple[str, int, int, int, int]] = []
    for candidate in candidates:
        info = candidate.lstat()
        relative = candidate.name if resolved.is_file() else candidate.relative_to(resolved).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"{where} contains a symbolic link: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ContractError(f"{where} contains a non-regular entry: {relative}")
        rows.append((relative, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    if not rows:
        raise ContractError(f"{where} is empty: {resolved}")
    return tuple(rows)


def _source_identity(source_root: Path) -> tuple[Path, str]:
    root = source_root.expanduser().resolve(strict=True)
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    resolved_top = Path(top.stdout.strip()).resolve(strict=True) if top.returncode == 0 and top.stdout.strip() else None
    if resolved_top != root:
        raise ContractError("--source-root must be this wrapper's Git checkout root")
    try:
        relative = SCRIPT_PATH.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("Zig oracle wrapper is outside --source-root") from exc
    if relative != RUNNER_RELATIVE_PATH:
        raise ContractError("Zig oracle wrapper path differs from the admitted source path")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    revision = head.stdout.strip() if head.returncode == 0 else ""
    if _HEX40.fullmatch(revision) is None:
        raise ContractError(f"could not resolve a full Antfly source commit from {root}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if dirty.returncode != 0 or dirty.stdout:
        raise ContractError("Antfly Zig oracle source checkout must be clean, including untracked files")
    return root, revision


def _executable_version(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "inference", "version"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"could not query Antfly inference version: {exc}") from exc
    matches = [
        match.group(1)
        for line in (*result.stdout.splitlines(), *result.stderr.splitlines())
        if (match := _VERSION_LINE.fullmatch(line)) is not None
    ]
    if result.returncode != 0 or len(matches) != 1 or not matches[0] or len(matches[0]) > 256:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise ContractError(f"Antfly inference version output was missing or ambiguous: {detail}")
    return matches[0]


def _write_private_json(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())


def _terminate_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=10)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_child(command: Sequence[str], environment: Mapping[str, str], stdout: Path, stderr: Path, timeout: int) -> int:
    process: subprocess.Popen[bytes] | None = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                env=dict(environment),
                close_fds=True,
                start_new_session=True,
            )
            return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            _terminate_child(process)
        raise ContractError(f"Antfly Zig oracle timed out after {timeout} seconds") from exc
    except BaseException:
        if process is not None:
            _terminate_child(process)
        raise


def _oracle_environment(source: Mapping[str, str], backend: str) -> dict[str, str]:
    environment = {
        name: value
        for name, value in source.items()
        if name not in SANITIZED_ENV_NAMES
        and not any(name.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
    }
    if backend == "metal":
        environment.update(STRICT_METAL_ENV)
    environment.update(OFFLINE_ENV)
    return environment


def _tail(path: Path, limit: int = 8192) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            source.seek(max(0, source.tell() - limit))
            return source.read().decode(errors="replace")
    except OSError:
        return ""


def _verify_packaging_dependencies(lock: Mapping[str, Any]) -> dict[str, str]:
    expected_packages = _mapping(_mapping(lock["python_oracle"], "python_oracle")["packages"], "python_oracle.packages")
    expected = {name: str(expected_packages[name]) for name in ZIG_ORACLE_PACKAGING_PACKAGES}
    actual: dict[str, str] = {}
    for name, version in expected.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = "<missing>"
        if actual[name] != version:
            raise ContractError(f"Zig oracle packager requires {name}={version}, found {actual[name]}")
    return actual


def _host_identity(backend: str, executable_sha256: str, metal_device: str | None) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ContractError("Antfly Zig release-oracle traces require Darwin arm64")
    host = hardware_fingerprint()
    required = ("platform", "machine", "chip", "memory_bytes", "os_version", "os_build")
    missing = [field for field in required if field not in host]
    if missing:
        raise ContractError(f"could not resolve required Darwin hardware identity: {missing}")
    result = {field: host[field] for field in required}
    result.update({
        "backend": backend,
        "executable_sha256": executable_sha256,
        "git_dirty": False,
    })
    if backend == "metal":
        result["metal_device"] = _string(metal_device, "--metal-device")
    elif metal_device is not None:
        raise ContractError("--metal-device is valid only with --backend metal")
    return result


def _stable_probe_token_ids(target: int, vocab_size: int, predictor_position: int, seed: int) -> list[int]:
    if not 0 <= target < vocab_size:
        raise ContractError(f"supervised token {target} is outside vocabulary size {vocab_size}")
    candidates = [0, 1, 2, target, vocab_size - 1]
    state = (seed ^ (predictor_position * 0x9E3779B1)) & 0xFFFFFFFF
    for _ in range(4):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        candidates.append(state % vocab_size)
    return sorted(set(candidates))


def _model_vocab_size(model_dir: Path) -> int:
    config = _mapping(load_json(model_dir / "config.json"), "model config")
    text = _mapping(config.get("text_config", config), "model text_config")
    return _integer(text.get("vocab_size"), "model text_config.vocab_size", minimum=3)


def _validate_run_manifest(root: Path, expected_files: Sequence[str]) -> None:
    expected = set(expected_files)
    actual: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"Zig capture contains a non-regular entry: {path.name}")
        actual.add(path.name)
    if actual != expected | {"run_manifest.json"}:
        raise ContractError(
            f"Zig capture file set differs (missing={sorted((expected | {'run_manifest.json'}) - actual)}, "
            f"unknown={sorted(actual - (expected | {'run_manifest.json'}))})"
        )
    manifest = _mapping(load_json(root / "run_manifest.json"), "capture run manifest")
    _exact_keys(manifest, ("schema_version", "status", "artifact_family_version", "artifacts"), "capture run manifest")
    if (
        manifest["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["artifact_family_version"] != ARTIFACT_FAMILY_VERSION
    ):
        raise ContractError("Zig capture run manifest is not complete or uses the wrong schema")
    entries: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(_list(manifest["artifacts"], "capture run manifest.artifacts")):
        entry = _mapping(raw, f"capture run manifest.artifacts[{index}]")
        _exact_keys(entry, ("name", "sha256", "size_bytes"), f"capture run manifest.artifacts[{index}]")
        name = _string(entry["name"], f"capture run manifest.artifacts[{index}].name")
        if name in entries:
            raise ContractError(f"duplicate capture run-manifest entry: {name}")
        entries[name] = entry
    if set(entries) != expected:
        raise ContractError("Zig capture run manifest does not close over the expected file set")
    for name, entry in entries.items():
        path = root / name
        digest = entry["sha256"]
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ContractError(f"capture run manifest has a malformed digest for {name}")
        if digest != prefixed_sha256(path).removeprefix("sha256:"):
            raise ContractError(f"capture run manifest digest mismatch for {name}")
        if _integer(entry["size_bytes"], f"capture run manifest {name} size", minimum=1) != path.stat().st_size:
            raise ContractError(f"capture run manifest size mismatch for {name}")


def _artifact_file(root: Path, descriptor: Mapping[str, Any], expected_name: str, where: str) -> Path:
    _exact_keys(descriptor, ("path", "sha256", "size_bytes"), where)
    if descriptor["path"] != expected_name:
        raise ContractError(f"{where}.path must be {expected_name!r}")
    path = _regular_file(root / expected_name, where)
    if descriptor["sha256"] != prefixed_sha256(path):
        raise ContractError(f"{where} SHA-256 mismatch")
    if _integer(descriptor["size_bytes"], f"{where}.size_bytes", minimum=1) != path.stat().st_size:
        raise ContractError(f"{where} size mismatch")
    return path


def _safetensor_keys(path: Path) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ContractError("safetensors is required to validate a Zig oracle capture") from exc
    with safe_open(str(path), framework="np", device="cpu") as source:
        return set(source.keys())


def _decode_u64_chunks(values: Any, where: str) -> list[int]:
    import numpy as np

    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 1 or array.size % 4 != 0:
        raise ContractError(f"{where} must be a flat F32 array with four chunks per integer")
    result: list[int] = []
    for start in range(0, array.size, 4):
        value = 0
        for chunk_index, raw in enumerate(array[start : start + 4]):
            chunk = float(raw)
            if not math.isfinite(chunk) or chunk != math.floor(chunk) or not 0 <= chunk <= 65535:
                raise ContractError(f"{where} contains an invalid u16 chunk")
            value |= int(chunk) << (chunk_index * 16)
        result.append(value)
    return result


def _validate_checkpoint_contract(path: Path, targets: Sequence[Mapping[str, Any]], steps: int, seed: int) -> None:
    from safetensors import safe_open

    expected_keys = {"__trainer_counters", "__trainer_state_v2"}
    for target in targets:
        slot = _string(target["trainer_slot_name"], "capture target trainer_slot_name")
        expected_keys.update(f"{prefix}::{slot}" for prefix in ("weight", "adam_m", "adam_v", "adam_step", "grad_accum"))
    with safe_open(str(path), framework="np", device="cpu") as source:
        if set(source.keys()) != expected_keys:
            raise ContractError("trainer checkpoint tensor inventory differs from the closed oracle contract")
        counters = _decode_u64_chunks(source.get_tensor("__trainer_counters"), "__trainer_counters")
        if counters != [steps, steps]:
            raise ContractError(f"trainer checkpoint counters differ from requested steps: {counters}")
        state = _decode_u64_chunks(source.get_tensor("__trainer_state_v2"), "__trainer_state_v2")
        if len(state) != 18 or state[:7] != [2, steps, steps, steps, 0, 1, seed]:
            raise ContractError("trainer checkpoint state does not bind the requested completed trajectory")
        if state[7] != steps or state[8] != 0 or state[9] != steps or state[11] != 0:
            raise ContractError("trainer checkpoint dataset cursor is not at the requested epoch boundary")
        for target in targets:
            slot = str(target["trainer_slot_name"])
            shape = tuple(_integer(dim, "capture target shape", minimum=1) for dim in _list(target["shape"], "capture target shape"))
            for prefix in ("weight", "adam_m", "adam_v", "grad_accum"):
                tensor = source.get_tensor(f"{prefix}::{slot}")
                if tensor.dtype.name != "float32" or tuple(tensor.shape) != shape:
                    raise ContractError(f"checkpoint {prefix} tensor metadata differs for {slot}")
                if not bool((tensor == tensor).all()) or not bool(abs(tensor).max(initial=0) < float("inf")):
                    raise ContractError(f"checkpoint {prefix} tensor is non-finite for {slot}")
            step_tensor = source.get_tensor(f"adam_step::{slot}")
            if step_tensor.dtype.name != "float32" or tuple(step_tensor.shape) != (1,) or float(step_tensor[0]) != steps:
                raise ContractError(f"checkpoint Adam step differs for {slot}")
            if bool(source.get_tensor(f"grad_accum::{slot}").any()):
                raise ContractError(f"checkpoint retains a nonzero accumulated gradient for {slot}")


def validate_zig_capture(
    capture_dir: Path,
    request_path: Path,
    request: Mapping[str, Any],
    candidate_adapter_dir: Path,
    source_adapter: Mapping[str, Any],
    candidate_adapter: Mapping[str, Any],
    prepared: Mapping[str, Any],
    vocab_size: int,
) -> dict[str, Any]:
    """Validate the private Zig capture before it can become shared evidence."""

    root = capture_dir.expanduser().resolve(strict=True)
    if not root.is_dir() or capture_dir.is_symlink():
        raise ContractError("Zig oracle capture must be a real directory")
    _validate_run_manifest(
        root,
        ("capture.json", "raw_gradients.safetensors", "trainer_checkpoint.safetensors"),
    )
    payload = _mapping(load_json(root / "capture.json"), "Zig capture")
    _exact_keys(
        payload,
        ("schema_version", "request_sha256", "implementation", "bindings", "training", "result", "artifacts"),
        "Zig capture",
    )
    if payload["schema_version"] != CAPTURE_SCHEMA_VERSION:
        raise ContractError(f"unsupported Zig capture schema: {payload['schema_version']!r}")
    if payload["request_sha256"] != prefixed_sha256(request_path):
        raise ContractError("Zig capture request SHA-256 differs from the admitted request")
    for field in ("implementation", "bindings", "training"):
        if payload[field] != request[field]:
            raise ContractError(f"Zig capture changed request field {field}")

    training = _mapping(payload["training"], "Zig capture.training")
    steps = _integer(training.get("steps"), "Zig capture.training.steps", minimum=1)
    seed = _integer(training.get("seed"), "Zig capture.training.seed")
    result = _mapping(payload["result"], "Zig capture.result")
    _exact_keys(result, ("loss_history", "raw_gradient_norm", "supervised_tokens", "logit_probes", "targets", "execution"), "Zig capture.result")
    loss_history = [_finite(value, "Zig capture loss") for value in _list(result["loss_history"], "Zig capture.loss_history")]
    if len(loss_history) != steps:
        raise ContractError("Zig capture must contain one finite loss per requested step")
    raw_gradient_norm = _finite(result["raw_gradient_norm"], "Zig capture.raw_gradient_norm")
    if raw_gradient_norm <= 0:
        raise ContractError("Zig capture raw gradient norm must be positive")
    labels = _list(prepared["labels"], "prepared.labels")
    supervised_positions = [index for index in range(1, len(labels)) if labels[index] != -100]
    if result["supervised_tokens"] != len(supervised_positions):
        raise ContractError("Zig capture supervised-token count differs from the prepared row")

    probes = _list(result["logit_probes"], "Zig capture.logit_probes")
    if len(probes) != len(supervised_positions):
        raise ContractError("Zig capture must contain one logit probe per supervised causal position")
    for probe_index, (raw_probe, label_position) in enumerate(zip(probes, supervised_positions, strict=True)):
        probe = _mapping(raw_probe, f"Zig capture.logit_probes[{probe_index}]")
        _exact_keys(probe, ("predictor_position", "target_token_id", "token_ids", "values", "logsumexp"), f"Zig capture.logit_probes[{probe_index}]")
        predictor = label_position - 1
        target = labels[label_position]
        if probe["predictor_position"] != predictor or probe["target_token_id"] != target:
            raise ContractError("Zig capture logit probe is not aligned to the prepared causal target")
        token_ids = [_integer(value, "Zig capture probe token id") for value in _list(probe["token_ids"], "Zig capture probe token_ids")]
        if token_ids != _stable_probe_token_ids(target, vocab_size, predictor, seed):
            raise ContractError("Zig capture logit probe does not use the locked deterministic token projection")
        values = [_finite(value, "Zig capture probe value") for value in _list(probe["values"], "Zig capture probe values")]
        if len(values) != len(token_ids):
            raise ContractError("Zig capture logit probe values and token IDs differ in length")
        _finite(probe["logsumexp"], "Zig capture probe logsumexp")

    execution = _mapping(result["execution"], "Zig capture.execution")
    execution_fields = (
        "optimizer_steps", "micro_batch_steps", "metal_optimizer_steps", "graph_executor_steps",
        "graph_executor_fallback_steps", "graph_executor_native_partitions",
        "graph_executor_unsupported_ops", "graph_executor_interpreter_fallbacks",
        "graph_executor_true_host_outputs",
    )
    _exact_keys(execution, execution_fields, "Zig capture.execution")
    for field in execution_fields:
        _integer(execution[field], f"Zig capture.execution.{field}")
    if execution["optimizer_steps"] != steps or execution["micro_batch_steps"] != steps:
        raise ContractError("Zig capture execution counters differ from the requested trajectory")
    backend = _mapping(payload["implementation"], "Zig capture.implementation")["backend"]
    if backend == "metal":
        if (
            execution["metal_optimizer_steps"] != steps
            or execution["graph_executor_steps"] != steps
            or any(
                execution[field] != 0
                for field in (
                    "graph_executor_fallback_steps", "graph_executor_native_partitions",
                    "graph_executor_unsupported_ops", "graph_executor_interpreter_fallbacks",
                    "graph_executor_true_host_outputs",
                )
            )
        ):
            raise ContractError("Zig capture does not attest a fallback-free Metal trajectory")
    elif backend == "native":
        if execution["metal_optimizer_steps"] != 0:
            raise ContractError("native Zig capture unexpectedly reports Metal optimizer steps")
    else:
        raise ContractError(f"unsupported Zig capture backend: {backend!r}")

    targets = [_mapping(value, f"Zig capture.targets[{index}]") for index, value in enumerate(_list(result["targets"], "Zig capture.targets"))]
    target_fields = (
        "source_name", "trainer_slot_name", "shape", "gradient_storage_key",
        "checkpoint_weight_storage_key", "checkpoint_m_storage_key", "checkpoint_v_storage_key",
    )
    expected_count = _integer(_mapping(payload["bindings"], "Zig capture.bindings")["target_count"], "Zig capture target_count", minimum=1) * 2
    if len(targets) != expected_count:
        raise ContractError("Zig capture target count differs from the admitted adapter")
    source_tensors = _mapping(source_adapter["tensors"], "source adapter tensors")
    candidate_tensors = _mapping(candidate_adapter["tensors"], "candidate adapter tensors")
    if set(source_tensors) != set(candidate_tensors):
        raise ContractError("candidate adapter changed the initial canonical tensor inventory")
    identities: set[tuple[str, str]] = set()
    slots: set[str] = set()
    gradient_keys: set[str] = set()
    for index, target in enumerate(targets):
        _exact_keys(target, target_fields, f"Zig capture.targets[{index}]")
        source_name = _string(target["source_name"], f"Zig capture.targets[{index}].source_name")
        identity = canonicalize_adapter_tensor_name(source_name)
        if identity in identities or identity not in source_tensors:
            raise ContractError(f"Zig capture target identity is duplicate or absent from the adapter: {identity}")
        identities.add(identity)
        slot = _string(target["trainer_slot_name"], f"Zig capture.targets[{index}].trainer_slot_name")
        if slot in slots:
            raise ContractError(f"Zig capture reuses trainer slot {slot}")
        slots.add(slot)
        shape = tuple(_integer(dim, "Zig capture target shape", minimum=1) for dim in _list(target["shape"], "Zig capture target shape"))
        if len(shape) != 2 or shape != tuple(source_tensors[identity]["shape"]) or shape != tuple(candidate_tensors[identity]["shape"]):
            raise ContractError(f"Zig capture target shape differs from adapter metadata for {identity}")
        if source_tensors[identity]["dtype"] != "float32" or candidate_tensors[identity]["dtype"] != "float32":
            raise ContractError(f"Zig oracle adapters must use float32 LoRA tensors: {identity}")
        expected_storage = {
            "gradient_storage_key": f"gradient::{slot}",
            "checkpoint_weight_storage_key": f"weight::{slot}",
            "checkpoint_m_storage_key": f"adam_m::{slot}",
            "checkpoint_v_storage_key": f"adam_v::{slot}",
        }
        for field, expected_key in expected_storage.items():
            if target[field] != expected_key:
                raise ContractError(f"Zig capture target storage key is not canonical: {field}")
        gradient_keys.add(str(target["gradient_storage_key"]))
    if identities != set(source_tensors):
        raise ContractError("Zig capture does not cover the full adapter tensor inventory")

    artifacts = _mapping(payload["artifacts"], "Zig capture.artifacts")
    _exact_keys(artifacts, ("raw_gradients", "trainer_checkpoint", "candidate_adapter_model"), "Zig capture.artifacts")
    raw_path = _artifact_file(root, _mapping(artifacts["raw_gradients"], "raw_gradients artifact"), "raw_gradients.safetensors", "raw_gradients artifact")
    checkpoint_path = _artifact_file(root, _mapping(artifacts["trainer_checkpoint"], "trainer_checkpoint artifact"), "trainer_checkpoint.safetensors", "trainer_checkpoint artifact")
    candidate_descriptor = _mapping(artifacts["candidate_adapter_model"], "candidate_adapter_model artifact")
    _exact_keys(candidate_descriptor, ("sha256", "size_bytes"), "candidate_adapter_model artifact")
    candidate_checkpoint = _regular_file(candidate_adapter_dir / "adapter_model.safetensors", "candidate adapter checkpoint")
    if candidate_descriptor["sha256"] != prefixed_sha256(candidate_checkpoint):
        raise ContractError("Zig capture candidate-adapter SHA-256 mismatch")
    if _integer(candidate_descriptor["size_bytes"], "candidate adapter size", minimum=1) != candidate_checkpoint.stat().st_size:
        raise ContractError("Zig capture candidate-adapter size mismatch")
    if candidate_adapter["adapter_model_sha256"] != candidate_descriptor["sha256"]:
        raise ContractError("candidate adapter inspection and Zig capture digest disagree")
    if _safetensor_keys(raw_path) != gradient_keys:
        raise ContractError("raw-gradient Safetensors inventory differs from the capture target manifest")
    _validate_checkpoint_contract(checkpoint_path, targets, steps, seed)
    return {
        "payload": payload,
        "targets": targets,
        "raw_gradients_path": raw_path,
        "checkpoint_path": checkpoint_path,
        "loss_history": loss_history,
        "raw_gradient_norm": raw_gradient_norm,
        "probes": probes,
    }


def _tensor_f32(source: Any, key: str, shape: tuple[int, ...], where: str) -> Any:
    import numpy as np

    try:
        tensor = source.get_tensor(key)
    except Exception as exc:
        raise ContractError(f"{where} is missing tensor {key!r}: {exc}") from exc
    if tensor.dtype != np.float32 or tuple(tensor.shape) != shape:
        raise ContractError(f"{where} tensor {key!r} must be float32 with shape {shape}")
    if not bool(np.isfinite(tensor).all()):
        raise ContractError(f"{where} tensor {key!r} contains a non-finite value")
    return np.ascontiguousarray(tensor)


def _build_trace_tensor_store(
    destination: Path,
    capture: Mapping[str, Any],
    source_adapter: Mapping[str, Any],
    candidate_adapter: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], float]:
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    source_tensors = _mapping(source_adapter["tensors"], "source adapter tensors")
    candidate_tensors = _mapping(candidate_adapter["tensors"], "candidate adapter tensors")
    target_by_identity = {
        canonicalize_adapter_tensor_name(str(target["source_name"])): target
        for target in capture["targets"]
    }
    storage: dict[str, Any] = {}
    entries: dict[str, dict[str, Any]] = {}
    target_rows: list[dict[str, Any]] = []
    gradient_squares: list[float] = []
    steps = int(_mapping(capture["payload"]["training"], "capture training")["steps"])
    source_checkpoint = Path(str(source_adapter["checkpoint"]))
    candidate_checkpoint = Path(str(candidate_adapter["checkpoint"]))
    with (
        safe_open(str(source_checkpoint), framework="np", device="cpu") as initial_source,
        safe_open(str(candidate_checkpoint), framework="np", device="cpu") as updated_source,
        safe_open(str(capture["raw_gradients_path"]), framework="np", device="cpu") as gradient_source,
        safe_open(str(capture["checkpoint_path"]), framework="np", device="cpu") as checkpoint_source,
    ):
        for tensor_index, identity in enumerate(sorted(target_by_identity)):
            module, role = identity
            target = target_by_identity[identity]
            shape = tuple(int(dim) for dim in target["shape"])
            initial = _tensor_f32(
                initial_source,
                str(source_tensors[identity]["source_name"]),
                shape,
                "initial adapter",
            )
            updated = _tensor_f32(
                updated_source,
                str(candidate_tensors[identity]["source_name"]),
                shape,
                "candidate adapter",
            )
            gradient = _tensor_f32(
                gradient_source,
                str(target["gradient_storage_key"]),
                shape,
                "raw-gradient capture",
            )
            checkpoint_weight = _tensor_f32(
                checkpoint_source,
                str(target["checkpoint_weight_storage_key"]),
                shape,
                "trainer checkpoint",
            )
            optimizer_m = _tensor_f32(
                checkpoint_source,
                str(target["checkpoint_m_storage_key"]),
                shape,
                "trainer checkpoint",
            )
            optimizer_v = _tensor_f32(
                checkpoint_source,
                str(target["checkpoint_v_storage_key"]),
                shape,
                "trainer checkpoint",
            )
            if not np.array_equal(updated, checkpoint_weight):
                raise ContractError(f"candidate adapter and trainer checkpoint weights differ for {identity}")

            paired_b_zero = False
            if role == "lora_A" and steps == 1:
                paired = (module, "lora_B")
                paired_shape = tuple(source_tensors[paired]["shape"])
                paired_b = _tensor_f32(
                    initial_source,
                    str(source_tensors[paired]["source_name"]),
                    paired_shape,
                    "initial adapter",
                )
                paired_b_zero = not bool(np.count_nonzero(paired_b))
            expectation = "zero-by-zero-b-initialization" if paired_b_zero else "active"
            gradient_nonzero = bool(np.count_nonzero(gradient))
            if expectation == "active" and not gradient_nonzero:
                raise ContractError(f"active Zig target has an all-zero gradient: {identity}")
            if expectation != "active" and gradient_nonzero:
                raise ContractError(f"zero-initialization Zig gradient expectation failed: {identity}")
            gradient_squares.append(float(np.sum(gradient.astype(np.float64) ** 2, dtype=np.float64)))

            logical: dict[str, str] = {}
            for state_index, (state_name, tensor) in enumerate((
                ("initial", initial),
                ("gradient", gradient),
                ("updated", updated),
                ("optimizer_m", optimizer_m),
                ("optimizer_v", optimizer_v),
            )):
                logical_name = f"{module}:{role}:{state_name}"
                storage_key = f"tensor_{tensor_index:05d}_{state_index}"
                storage[storage_key] = tensor
                entries[logical_name] = {
                    "shape": list(shape),
                    "dtype": "float32",
                    "storage_key": storage_key,
                }
                logical[state_name] = logical_name
            target_rows.append({
                "canonical_name": module,
                "source_name": str(target["source_name"]),
                "role": role,
                "shape": list(shape),
                "gradient_expectation": expectation,
                "logical_tensors": logical,
            })
    save_file(storage, str(destination), metadata={"format": TRACE_SCHEMA_VERSION})
    grad_norm = math.sqrt(math.fsum(gradient_squares))
    if not math.isclose(grad_norm, float(capture["raw_gradient_norm"]), rel_tol=1e-9, abs_tol=1e-12):
        raise ContractError(
            "Zig capture gradient norm is not bound to raw_gradients.safetensors "
            f"(reported={capture['raw_gradient_norm']}, recomputed={grad_norm})"
        )
    return target_rows, entries, grad_norm


def _fsync_path(path: Path) -> None:
    flags = os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    paths = list(root.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ContractError(f"oracle evidence cannot contain a symlink: {path}")
        if path.is_file():
            _fsync_path(path)
    for directory in sorted((path for path in paths if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
        _fsync_path(directory)
    _fsync_path(root)


def _publish_staging(staging: Path, output: Path) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _fsync_tree(staging)
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise ContractError(f"refusing to replace existing Zig oracle output: {target}") from exc
    _fsync_path(target.parent)
    complete = staging / "COMPLETE.json"
    children = sorted((child for child in staging.iterdir() if child != complete), key=lambda child: child.name)
    for child in children:
        child.rename(target / child.name)
    _fsync_tree(target)
    complete.rename(target / complete.name)
    _fsync_path(target / complete.name)
    _fsync_path(target)
    _fsync_path(target.parent)


def _validate_optimizer_contract(args: argparse.Namespace, lock: Mapping[str, Any]) -> Mapping[str, Any]:
    trajectory = _mapping(lock["training_contract"], "training_contract")
    if args.steps not in trajectory["steps"]:
        raise ContractError(f"the locked trajectory admits exactly {trajectory['steps']} steps")
    supplied = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "betas": list(args.betas),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
    }
    expected = {key: trajectory[key] for key in supplied}
    if supplied != expected:
        raise ContractError(f"optimizer arguments differ from training_contract (expected={expected}, actual={supplied})")
    return trajectory


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def export(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = _regular_file(args.lock, "oracle lock")
    lock = load_lock(lock_path)
    packaging_dependencies = _verify_packaging_dependencies(lock)
    trajectory = _validate_optimizer_contract(args, lock)
    executable = _regular_file(args.antfly, "Antfly executable", executable=True)
    executable_sha256 = prefixed_sha256(executable)
    antfly_version = _executable_version(executable)
    source_root, source_revision = _source_identity(args.source_root)
    hardware = _host_identity(args.backend, executable_sha256, args.metal_device)

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ContractError(f"refusing to replace existing Zig oracle output: {output}")
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    adapter_dir = args.adapter.expanduser().resolve(strict=True)
    prepared_path = _regular_file(args.prepared, "prepared artifact")
    for immutable in (source_root, model_dir, adapter_dir, prepared_path, executable, lock_path):
        if _paths_overlap(output, immutable):
            raise ContractError(f"Zig oracle output overlaps immutable input: {immutable}")

    verified_model = verify_model_directory(lock, args.model_key, model_dir)
    vocab_size = _model_vocab_size(model_dir)
    prepared_summary, prepared = load_prepared_example(prepared_path, args.example_index)
    verified_source = verify_prepared_source_dataset(prepared_summary, args.source_dataset)
    source_dataset_path = Path(verified_source["path"])
    if _paths_overlap(output, source_dataset_path):
        raise ContractError("Zig oracle output overlaps the immutable prepared source dataset")

    source_adapter = inspect_adapter_artifact(adapter_dir, target_preset=args.target_preset)
    if source_adapter["key_layout"] != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError("the Zig oracle requires a provenance-bound Antfly internal-key adapter")
    provenance = _mapping(source_adapter.get("provenance"), "source adapter provenance")
    for field in ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256"):
        if provenance.get(field) != prepared_summary.get(field):
            raise ContractError(f"source adapter and prepared artifact disagree on {field}")
    semantics = _mapping(source_adapter["semantics"], "source adapter semantics")
    preset = _string(semantics.get("target_preset"), "source adapter target_preset")
    validate_target_inventory(lock, args.model_key, preset, semantics["target_modules"])
    rank = _integer(semantics["r"], "source adapter rank", minimum=1)
    alpha = _finite(semantics["lora_alpha"], "source adapter alpha")
    if alpha <= 0:
        raise ContractError("source adapter alpha must be positive")
    target_count = len({module for module, _role in source_adapter["tensors"]})
    if len(source_adapter["tensors"]) != target_count * 2:
        raise ContractError("source adapter does not contain one A/B pair per target")

    model_snapshot = _tree_snapshot(model_dir, "model directory")
    adapter_snapshot = _tree_snapshot(adapter_dir, "source adapter")
    prepared_snapshot = _tree_snapshot(prepared_path, "prepared artifact")
    source_dataset_snapshot = _tree_snapshot(source_dataset_path, "prepared source dataset")
    executable_info = executable.stat()
    executable_snapshot = (executable_info.st_dev, executable_info.st_ino, executable_info.st_size, executable_info.st_mtime_ns)

    implementation = {
        "version": antfly_version,
        "executable_sha256": executable_sha256,
        "source_revision": source_revision,
        "backend": args.backend,
        "metal_device": args.metal_device if args.backend == "metal" else None,
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "implementation": implementation,
        "bindings": {
            "oracle_lock_sha256": lock_digest(lock_path),
            "model_key": args.model_key,
            "model_revision": verified_model["revision"],
            "local_artifact_sha256": verified_model["local_artifact_sha256"],
            "base_model_sha256": prepared_summary["base_model_sha256"],
            "initial_adapter_sha256": source_adapter["adapter_model_sha256"],
            "train_prepared_sha256": prefixed_sha256(prepared_path),
            "source_dataset_sha256": prepared_summary["source_dataset_sha256"],
            "example_index": args.example_index,
            "target_preset": preset,
            "rank": rank,
            "alpha": alpha,
            "target_count": target_count,
        },
        "training": {
            "optimizer": "adamw",
            "seed": args.seed,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "betas": list(args.betas),
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "grad_accum_steps": int(trajectory["grad_accum_steps"]),
            "supervised_token_normalization": trajectory["supervised_token_normalization"],
            "dropout": 0.0,
            "use_cache": False,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.zig-oracle-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        request_path = temporary / "request.json"
        stdout_path = temporary / "stdout.log"
        stderr_path = temporary / "stderr.log"
        candidate_dir = temporary / "candidate-adapter"
        capture_dir = temporary / "zig-capture"
        _write_private_json(request_path, request)
        command = [
            str(executable), "inference", "finetune", "train", "gemma4-lora",
            "--model", str(model_dir),
            "--adapter", str(adapter_dir),
            "--train-prepared", str(prepared_path),
            "--eval-prepared", str(prepared_path),
            "--out", str(candidate_dir),
            "--backend", args.backend,
            "--trainer", "autodiff",
            "--lr", str(args.learning_rate),
            "--max-examples", "1",
            "--eval-max-examples", "1",
            "--epochs", str(args.steps),
            "--max-grad-norm", str(args.max_grad_norm),
            "--grad-accum", "1",
            "--seed", str(args.seed),
            "--oracle-request", str(request_path),
            "--oracle-capture-out", str(capture_dir),
        ]
        environment = _oracle_environment(os.environ, args.backend)
        returncode = _run_child(command, environment, stdout_path, stderr_path, args.timeout_seconds)
        if returncode != 0 or not candidate_dir.is_dir() or not capture_dir.is_dir():
            detail = _tail(stderr_path).strip()
            suffix = f"\nchild stderr tail:\n{detail}" if detail else ""
            raise ContractError(f"Antfly Zig oracle child failed with exit status {returncode}{suffix}")

        # Revalidate every source identity and mutable input boundary after the
        # child exits. A trace may never combine pre-run hashes with post-run
        # bytes from an artifact that changed concurrently.
        _, source_revision_after = _source_identity(source_root)
        if source_revision_after != source_revision:
            raise ContractError("Antfly source revision changed during the Zig oracle run")
        executable_info_after = executable.stat()
        executable_snapshot_after = (
            executable_info_after.st_dev,
            executable_info_after.st_ino,
            executable_info_after.st_size,
            executable_info_after.st_mtime_ns,
        )
        if executable_snapshot_after != executable_snapshot or prefixed_sha256(executable) != executable_sha256:
            raise ContractError("Antfly executable changed during the Zig oracle run")
        for path, expected, where in (
            (model_dir, model_snapshot, "model directory"),
            (adapter_dir, adapter_snapshot, "source adapter"),
            (prepared_path, prepared_snapshot, "prepared artifact"),
            (source_dataset_path, source_dataset_snapshot, "prepared source dataset"),
        ):
            if _tree_snapshot(path, where) != expected:
                raise ContractError(f"{where} changed during the Zig oracle run")
        if prefixed_sha256(prepared_path) != request["bindings"]["train_prepared_sha256"]:
            raise ContractError("prepared artifact digest changed during the Zig oracle run")
        if prefixed_sha256(adapter_dir / "adapter_model.safetensors") != request["bindings"]["initial_adapter_sha256"]:
            raise ContractError("source adapter digest changed during the Zig oracle run")
        verify_prepared_source_dataset(prepared_summary, args.source_dataset)

        candidate_adapter = inspect_adapter_artifact(candidate_dir, target_preset=preset)
        if (
            candidate_adapter["key_layout"] != ANTFLY_ADAPTER_KEY_FORMAT
            or candidate_adapter["inventory"] != source_adapter["inventory"]
            or candidate_adapter["semantics"] != source_adapter["semantics"]
        ):
            raise ContractError("candidate adapter changed the admitted LoRA contract")
        candidate_provenance = _mapping(candidate_adapter.get("provenance"), "candidate adapter provenance")
        for field in ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256"):
            if candidate_provenance.get(field) != prepared_summary.get(field):
                raise ContractError(f"candidate adapter and prepared artifact disagree on {field}")

        capture = validate_zig_capture(
            capture_dir,
            request_path,
            request,
            candidate_dir,
            source_adapter,
            candidate_adapter,
            prepared,
            vocab_size,
        )

        staging = temporary / "publication"
        staging.mkdir()
        tensor_path = staging / "trace.safetensors"
        target_rows, entries, grad_norm = _build_trace_tensor_store(
            tensor_path,
            capture,
            source_adapter,
            candidate_adapter,
        )
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "producer": {
                "name": f"antfly-zig-{args.backend}",
                "version": antfly_version,
                "source_revision": source_revision,
                "hardware": hardware,
            },
            "oracle_lock_sha256": lock_digest(lock_path),
            "model": {
                "key": args.model_key,
                "repo_id": verified_model["repo_id"],
                "revision": verified_model["revision"],
                "local_artifact_sha256": verified_model["local_artifact_sha256"],
            },
            "prepared": dict(prepared),
            "training": {
                "optimizer": "adamw",
                "seed": args.seed,
                "step": args.steps,
                "rank": rank,
                "alpha": alpha,
                "scale": alpha / rank,
                "target_preset": preset,
                "learning_rate": args.learning_rate,
                "betas": list(args.betas),
                "eps": args.eps,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "grad_accum_steps": 1,
                "supervised_token_normalization": "mean",
                "dropout": 0.0,
                "use_cache": False,
            },
            "metrics": {
                "loss": capture["loss_history"][-1],
                "loss_history": capture["loss_history"],
                "grad_norm": grad_norm,
                "supervised_tokens": len([value for value in prepared["labels"][1:] if value != -100]),
            },
            "logit_probes": capture["probes"],
            "target_tensors": target_rows,
            "tensor_store": {
                "format": "safetensors/v1",
                "path": tensor_path.name,
                "sha256": prefixed_sha256(tensor_path),
                "entries": entries,
            },
            "artifact": {
                "adapter_config_semantics": {
                    **candidate_adapter["semantics"],
                    "target_preset": preset,
                },
                "adapter_model_sha256": candidate_adapter["adapter_model_sha256"],
                "tensor_inventory": sorted(candidate_adapter["inventory"]),
                "key_layout": candidate_adapter["key_layout"],
                "policy_source": candidate_adapter["policy_source"],
            },
        }
        write_json(staging / "trace.json", trace)
        shutil.copy2(request_path, staging / "request.json")
        shutil.copy2(stdout_path, staging / "stdout.log")
        shutil.copy2(stderr_path, staging / "stderr.log")
        shutil.move(str(candidate_dir), staging / "candidate_adapter")
        shutil.move(str(capture_dir), staging / "producer_capture")
        write_json(staging / "packaging_environment.json", {
            "schema_version": "antfly_gemma4_lora_zig_oracle_packaging/v1",
            "python": platform.python_version(),
            "packages": packaging_dependencies,
            "command": command,
            "request_sha256": prefixed_sha256(staging / "request.json"),
            "source_root": str(source_root),
            "source_revision": source_revision,
            "executable_sha256": executable_sha256,
            "environment_contract": {
                "sanitized_prefixes": list(SANITIZED_ENV_PREFIXES),
                "sanitized_names": sorted(SANITIZED_ENV_NAMES),
                "offline": OFFLINE_ENV,
                "strict_metal": STRICT_METAL_ENV if args.backend == "metal" else {},
            },
        })
        write_json(staging / "COMPLETE.json", build_evidence_ledger(staging))
        validate_trace(staging / "trace.json", lock, lock_path=lock_path)
        _publish_staging(staging, output)

    validated = validate_trace(output / "trace.json", lock, lock_path=lock_path)
    return {
        "ok": True,
        "output_dir": str(output),
        "trace": str(validated.path),
        "trace_sha256": validated.trace_sha256,
        "evidence_manifest_sha256": validated.evidence_manifest_sha256,
        "model": args.model_key,
        "backend": args.backend,
        "target_preset": preset,
        "steps": args.steps,
        "loss": capture["loss_history"][-1],
        "grad_norm": validated.recomputed_grad_norm,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lock", type=Path, default=LOCK_PATH)
    result.add_argument("--antfly", type=Path, required=True, help="release-built Antfly executable")
    result.add_argument("--source-root", type=Path, required=True, help="clean source checkout used to build the executable")
    result.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--target-preset", choices=("peft-qv", "text-all-linear"))
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--source-dataset", type=Path, help="override the recorded v6 source path while preserving its digest")
    result.add_argument("--example-index", type=int, default=0)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--backend", choices=("native", "metal"), default="metal")
    result.add_argument("--metal-device", help="auditable Metal device label; required only for Metal")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--steps", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--betas", type=parse_betas, default=(0.9, 0.999))
    result.add_argument("--eps", type=float, default=1e-8)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    result.add_argument("--timeout-seconds", type=int, default=21_600)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.example_index < 0:
            raise ContractError("--example-index must be non-negative")
        if args.timeout_seconds <= 0:
            raise ContractError("--timeout-seconds must be positive")
        if args.backend == "metal" and not args.metal_device:
            raise ContractError("--metal-device is required with --backend metal")
        if args.backend == "native" and args.metal_device is not None:
            raise ContractError("--metal-device is valid only with --backend metal")
        print(json.dumps(export(args), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except ContractError as exc:
        print(f"Gemma4 Zig oracle error: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Gemma4 Zig oracle failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
