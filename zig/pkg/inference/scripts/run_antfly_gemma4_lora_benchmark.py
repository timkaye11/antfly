#!/usr/bin/env python3
"""Run one fresh-process Antfly Gemma4 LoRA same-Mac benchmark sample.

This runner is deliberately stricter than a generic wall-clock wrapper.  It
only publishes evidence when the typed Gemma4 trainer emits synchronized,
per-optimizer-step telemetry for the exact locked workload. Older binaries
fail closed with ``REQUIRED_ZIG_CHANGE`` instead of turning process time, log
lines, or epoch aggregates into invented step metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench_gemma4_lora_mlx_zig import (
    SYNC_POINT,
    TIMED_UNIT,
    benchmark_workload_sha256,
    canonical_target_inventory_sha256,
    expected_semantic_contract,
    validate_sample,
)
from gemma4_oracle_contract import (
    BENCHMARK_PRODUCER_RELATIVE_PATHS,
    BENCHMARK_SAMPLE_SCHEMA_VERSION,
    ContractError,
    LOCK_PATH,
    attest_benchmark_producer_source,
    hardware_fingerprint,
    load_json,
    load_lock,
    load_prepared_example,
    lock_digest,
    prefixed_sha256,
    verify_model_directory,
    verify_prepared_source_dataset,
)
from run_gemma4_lora_mlx_benchmark import (
    MEMORY_SAMPLER_INTERVAL_MS,
    ProcessMemoryMeasurement,
    capture_darwin_system_memory_snapshot,
    darwin_phys_footprint_bytes,
    darwin_system_memory_deltas,
    enforce_system_memory_gates,
    inspect_initial_adapter,
)


REQUEST_SCHEMA_VERSION = "antfly_gemma4_lora_benchmark_request/v2"
TELEMETRY_SCHEMA_VERSION = "antfly_gemma4_lora_benchmark_telemetry/v4"
PEAK_MEMORY_SOURCE = "darwin-proc-pid-rusage-v4-lifetime-max-phys-footprint"
COMMAND_DIGEST_DOMAIN = "antfly_gemma4_lora_benchmark_command/v1"
COLD_STEP_COUNT = 1
FIRST_STEADY_STEP_COUNT = 1
SCRIPT_PATH = Path(__file__).resolve()
ANTFLY_RUNNER_RELATIVE_PATH = "zig/pkg/inference/scripts/run_antfly_gemma4_lora_benchmark.py"
DIAGNOSTIC_SAMPLE_SCHEMA_VERSION = "antfly_gemma4_zig_diagnostic_sample/v2"
DIAGNOSTIC_SOURCE_SCHEMA_VERSION = "antfly_gemma4_diagnostic_producer_source/v1"
DIAGNOSTIC_RELEASE_BLOCKER = "diagnostic-mode-never-release-evidence"
MEASUREMENT_CONTROL = {
    "schema_version": "antfly_gemma4_benchmark_measurement_control/v1",
    "transport": "inherited-fd-byte-signals-with-ack",
    "signal_fd_environment": "ANTFLY_GEMMA4_BENCHMARK_CONTROL_FD",
    "ack_fd_environment": "ANTFLY_GEMMA4_BENCHMARK_ACK_FD",
    "before_measured_signal": "B",
    "before_measured_ack": "b",
    "after_measured_signal": "A",
    "after_measured_ack": "a",
}

PHASE_EVIDENCE_FIELDS = (
    "graph_build_ns",
    "runtime_input_ns",
    "train_step_ns",
    "compile_ns",
    "autodiff_ns",
    "execute_ns",
    "extract_ns",
    "optimizer_update_ns",
    "device_optimizer_ns",
    "total_ns",
    "metal_frame_wait_ns",
    "metal_frame_gpu_ns",
    "graph_executor_plan_build_ns",
    "graph_executor_buffer_plan_build_ns",
)

COMMAND_PLAN_EVIDENCE_FIELDS = (
    "graph_executor_partitions",
    "graph_executor_command_dispatches",
    "graph_executor_planned_dispatches",
    "graph_executor_runtime_region_dispatches",
    "graph_executor_runtime_region_active_regions",
    "graph_executor_runtime_region_covered_nodes",
    "graph_executor_runtime_region_elided_nodes",
    "graph_executor_runtime_region_plan_compiles",
    "graph_executor_runtime_region_plan_reuses",
    "graph_executor_plan_cache_hits",
    "graph_executor_plan_cache_misses",
    "metal_lora_backward_regions",
    "metal_low_rank_lora_backward_regions",
    "metal_rank_adapter_backward_regions",
    "metal_ffn_gelu_backward_regions",
    "metal_head_mlp_forward_regions",
    "metal_head_mlp_backward_regions",
    "metal_gemma4_bf16_gate_up_fused_calls",
    "metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls",
    "metal_linear_cce_forward_calls",
    "metal_linear_cce_backward_calls",
    "metal_linear_cce_forward_state_hits",
    "metal_linear_cce_forward_state_misses",
    "metal_linear_cce_peak_scratch_bytes",
    "metal_command_dot_general_dispatches",
    "metal_command_head_dot_dispatches",
    "metal_command_transpose_dispatches",
    "metal_command_gather_dispatches",
    "metal_command_reduce_dispatches",
    "metal_command_elementwise_dispatches",
    "metal_command_activation_dispatches",
    "metal_command_activation_backward_dispatches",
    "metal_command_other_dispatches",
    "metal_last_frame_compute_encoders",
    "metal_last_frame_blit_encoders",
    "metal_last_frame_planned_scopes",
    "metal_last_frame_planned_barriers",
    "metal_last_frame_planned_command_ops",
)

COMMAND_ATTRIBUTION_FIELDS = (
    "metal_command_dot_general_dispatches",
    "metal_command_head_dot_dispatches",
    "metal_command_transpose_dispatches",
    "metal_command_gather_dispatches",
    "metal_command_reduce_dispatches",
    "metal_command_elementwise_dispatches",
    "metal_command_activation_dispatches",
    "metal_command_activation_backward_dispatches",
    "metal_command_other_dispatches",
)

# Keep this actionable and co-located with the failure.  In particular, the
# required metric is an optimizer *window*, not StepProfile.total_ns from the
# last microbatch in a gradient-accumulation window.
REQUIRED_ZIG_CHANGE = (
    "add benchmark-only --benchmark-request and --benchmark-telemetry-out flags "
    "to the typed Gemma4 TrainOptions/CLI; validate the v2 request before model "
    "execution; expose each StepResult to a train-loop observer; time one complete "
    "optimizer window from before its first microbatch through the AdamW update "
    "and an explicit Metal queue/device synchronization; emit exactly one cold, "
    "one first-steady, three warmup, and twenty measured optimizer-window records with input and "
    "supervised-token totals plus strict-Metal fallback evidence; separately emit "
    "model/adapter load_ns, first-window graph compile_ns, and "
    "proc_pid_rusage(RUSAGE_INFO_V4).ri_lifetime_max_phys_footprint; report the "
    "binary's embedded build revision (not merely the request value); atomically "
    "publish telemetry only after all request bindings and step counts match"
)

STRICT_METAL_ENV = {
    "TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR": "1",
    "TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR": "0",
    "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK": "0",
    "TERMITE_DEBUG_DEVICE_GRAD_NORM": "0",
}

_HEX40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_INFERENCE_VERSION_LINE = re.compile(r"antfly inference v([^\r\n]+)")
_TERMITE_ENV_NAME = re.compile(r"TERMITE_[A-Z0-9_]+")


class BenchmarkInterfaceUnavailable(ContractError):
    """The child did not implement the evidence interface; never retry as wall time."""


@dataclass(frozen=True)
class PreparedBinding:
    case: dict[str, Any]
    workload_sha256: str
    input_tokens: int
    supervised_tokens: int
    artifact_sha256: str
    base_model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    snapshot: tuple[tuple[str, int, int, int, int], ...]


@dataclass(frozen=True)
class AdapterBinding:
    semantic_sha256: str
    tensor_count: int
    target_inventory_sha256: str
    canonical_modules: tuple[str, ...]
    rank: int
    alpha: float
    target_preset: str
    base_model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    snapshot: tuple[tuple[str, int, int, int, int], ...]


@dataclass(frozen=True)
class ChildRun:
    pid: int
    started_unix_ns: int
    returncode: int
    process_memory: ProcessMemoryMeasurement | None
    system_deltas: dict[str, int | float] | None
    measurement_error: str | None
    control_signals_seen: int


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], where: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise ContractError(
            f"{where}: fields differ (missing={sorted(wanted - actual)}, extra={sorted(actual - wanted)})"
        )


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def _regular_file(path: Path, where: str, *, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ContractError(f"{where} is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ContractError(f"{where} is not executable: {resolved}")
    return resolved


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ContractError(f"artifact is not a directory: {resolved}")
    rows: list[tuple[str, int, int, int, int]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()):
        info = path.lstat()
        relative = path.relative_to(resolved).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"artifact contains a symbolic link: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ContractError(f"artifact contains a non-regular entry: {relative}")
        rows.append((relative, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns))
    if not rows:
        raise ContractError(f"artifact directory is empty: {resolved}")
    return tuple(rows)


def _assert_snapshot_unchanged(root: Path, expected: tuple[tuple[str, int, int, int, int], ...], where: str) -> None:
    if _tree_snapshot(root) != expected:
        raise ContractError(f"{where} changed while the benchmark subprocess was running")


def source_identity(source_root: Path) -> tuple[Path, str]:
    root = source_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ContractError(f"Antfly source checkout is not a directory: {root}")
    top_level = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    resolved_top_level = (
        Path(top_level.stdout.strip()).resolve(strict=True)
        if top_level.returncode == 0 and top_level.stdout.strip()
        else None
    )
    if resolved_top_level != root:
        raise ContractError("Antfly --source-root must be the benchmark wrapper's Git checkout root")
    try:
        wrapper_relative = SCRIPT_PATH.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("Antfly benchmark wrapper is outside --source-root") from exc
    if wrapper_relative != ANTFLY_RUNNER_RELATIVE_PATH:
        raise ContractError("Antfly benchmark wrapper path differs from the admitted source path")
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
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if status_result.returncode != 0 or status_result.stdout:
        raise ContractError("Antfly benchmark source checkout must be clean, including untracked files")
    return root, revision


def _diagnostic_source_manifest_sha256(manifest_input: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        DIAGNOSTIC_SOURCE_SCHEMA_VERSION.encode("utf-8")
        + b"\0"
        + json.dumps(
            manifest_input,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def diagnostic_producer_source(source_root: Path) -> dict[str, Any]:
    """Content-bind a live checkout without claiming release cleanliness."""

    root = source_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ContractError(f"Antfly source checkout is not a directory: {root}")

    def git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
        try:
            return subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=True,
                capture_output=True,
                text=text,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", b"" if not text else "") or str(exc)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise ContractError(
                f"could not bind diagnostic Antfly benchmark source: {str(detail).strip()}"
            ) from exc

    resolved_top_level = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve(strict=True)
    if resolved_top_level != root:
        raise ContractError("Antfly --source-root must be the diagnostic wrapper's Git checkout root")
    try:
        relative_entrypoint = SCRIPT_PATH.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("diagnostic Antfly benchmark wrapper is outside --source-root") from exc
    if relative_entrypoint != ANTFLY_RUNNER_RELATIVE_PATH:
        raise ContractError("diagnostic Antfly benchmark wrapper path differs from the admitted source path")

    revision = git("rev-parse", "HEAD").stdout.strip()
    source_tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
    if _HEX40.fullmatch(revision) is None or _HEX40.fullmatch(source_tree) is None:
        raise ContractError("diagnostic Antfly source revision/tree is malformed")
    status = git("status", "--porcelain=v1", "--untracked-files=all", text=False).stdout

    files: list[dict[str, str]] = []
    for relative_path in BENCHMARK_PRODUCER_RELATIVE_PATHS:
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ContractError(
                f"diagnostic Antfly producer source must be a regular file: {relative_path}"
            )
        files.append({
            "relative_path": relative_path,
            "source_sha256": prefixed_sha256(path),
        })
    entrypoint_sha256 = next(
        item["source_sha256"]
        for item in files
        if item["relative_path"] == relative_entrypoint
    )
    manifest_input = {
        "relative_path": relative_entrypoint,
        "source_revision": revision,
        "source_tree": source_tree,
        "source_clean": not status,
        "working_tree_status_sha256": "sha256:" + hashlib.sha256(status).hexdigest(),
        "files": files,
    }
    return {
        "schema_version": DIAGNOSTIC_SOURCE_SCHEMA_VERSION,
        **manifest_input,
        "dirty_entry_count": len(status.splitlines()),
        "source_sha256": entrypoint_sha256,
        "manifest_sha256": _diagnostic_source_manifest_sha256(manifest_input),
    }


def executable_version(executable: Path) -> str:
    """Read the product version independently from source provenance."""
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
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise ContractError(f"could not query Antfly inference version: {detail}")
    # The inference CLI uses structured logging and currently writes its
    # version record to stderr, while older/fake runners used stdout. Admit
    # either transport but require one unambiguous canonical record overall.
    matches = [
        match.group(1)
        for line in (*result.stdout.splitlines(), *result.stderr.splitlines())
        if (match := _INFERENCE_VERSION_LINE.fullmatch(line)) is not None
    ]
    if len(matches) != 1 or not matches[0] or len(matches[0]) > 256:
        raise ContractError("Antfly inference version output was missing or ambiguous")
    return matches[0]


def prepared_binding(
    path: Path,
    *,
    sequence_length: int,
    grad_accum: int,
    example_index: int,
    source_dataset: Path | None,
) -> PreparedBinding:
    resolved = _regular_file(path, "prepared training artifact")
    summary, example = load_prepared_example(resolved, example_index)
    verify_prepared_source_dataset(summary, source_dataset)
    if summary.get("max_seq_len") != sequence_length:
        raise ContractError(
            f"prepared max_seq_len must equal the benchmark cell sequence length "
            f"({summary.get('max_seq_len')} != {sequence_length})"
        )

    input_ids = list(example["input_ids"])
    labels = list(example["labels"])
    if len(input_ids) != sequence_length or len(labels) != sequence_length:
        raise ContractError(
            "the selected prepared row length must exactly equal the benchmark sequence length; "
            "benchmark runners may not pad or truncate"
        )
    if any(token > 2**31 - 1 for token in input_ids) or any(
        label > 2**31 - 1 for label in labels if label != -100
    ):
        raise ContractError("prepared token IDs must fit the paired MLX int32 workload")
    input_rows = [list(input_ids) for _ in range(grad_accum)]
    label_rows = [list(labels) for _ in range(grad_accum)]
    mask_rows = [[1] * sequence_length for _ in range(grad_accum)]
    supervised_tokens = sum(label != -100 for label in labels[1:]) * grad_accum
    if supervised_tokens == 0:
        raise ContractError("the selected prepared row has no causal supervised labels")

    workload = benchmark_workload_sha256(input_rows, label_rows, mask_rows)
    artifact_sha = prefixed_sha256(resolved)
    file_info = resolved.stat()
    snapshot = ((resolved.name, file_info.st_dev, file_info.st_ino, file_info.st_size, file_info.st_mtime_ns),)
    return PreparedBinding(
        case={
            "schema_version": example["schema_version"],
            "artifact_sha256": artifact_sha,
            "example_index": example_index,
            "source_dataset_sha256": example["source_dataset_sha256"],
            "source_record_sha256": example["source_record_sha256"],
            "rendered_chat_sha256": example["rendered_chat_sha256"],
            "workload_sha256": workload,
        },
        workload_sha256=workload,
        input_tokens=sequence_length * grad_accum,
        supervised_tokens=supervised_tokens,
        artifact_sha256=artifact_sha,
        base_model_sha256=str(summary["base_model_sha256"]),
        tokenizer_sha256=str(summary["tokenizer_sha256"]),
        chat_template_sha256=str(summary["chat_template_sha256"]),
        snapshot=snapshot,
    )


def _assert_file_snapshot_unchanged(path: Path, snapshot: tuple[tuple[str, int, int, int, int], ...], where: str) -> None:
    resolved = path.expanduser().resolve(strict=True)
    info = resolved.stat()
    current = ((resolved.name, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),)
    if current != snapshot:
        raise ContractError(f"{where} changed while the benchmark subprocess was running")


def adapter_binding(
    path: Path,
    lock: Mapping[str, Any],
    model_key: str,
    target_preset: str,
    prepared_summary: Mapping[str, Any],
) -> AdapterBinding:
    root = path.expanduser().resolve(strict=True)
    artifact = inspect_initial_adapter(root, lock, model_key, target_preset, prepared_summary)
    config = artifact.semantics
    if config["policy_source"] != "antfly-finetune-manifest/v2":
        raise ContractError("Antfly benchmark requires a provenance-bound Antfly adapter manifest")
    gate = _mapping(lock["performance_gate"], "performance_gate")
    if config["r"] != gate["rank"] or float(config["lora_alpha"]) != float(gate["alpha"]):
        raise ContractError("adapter rank/alpha differ from the locked performance matrix")
    provenance = _mapping(config["provenance"], "adapter provenance")
    canonical_modules = tuple(sorted({module for module, _role in artifact.tensors}))
    return AdapterBinding(
        semantic_sha256=artifact.semantic_sha256,
        tensor_count=len(artifact.tensors),
        target_inventory_sha256=canonical_target_inventory_sha256(canonical_modules),
        canonical_modules=canonical_modules,
        rank=int(config["r"]),
        alpha=float(config["lora_alpha"]),
        target_preset=target_preset,
        base_model_sha256=str(provenance["base_model_sha256"]),
        tokenizer_sha256=str(provenance["tokenizer_sha256"]),
        chat_template_sha256=str(provenance["chat_template_sha256"]),
        snapshot=_tree_snapshot(root),
    )


def _host_identity(metal_device: str) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ContractError("Antfly same-Mac benchmark requires Darwin arm64")
    host = hardware_fingerprint()
    fields = ("platform", "machine", "chip", "memory_bytes", "os_version", "os_build")
    missing = [field for field in fields if field not in host]
    if missing:
        raise ContractError(f"could not resolve required Darwin hardware identity: {missing}")
    result = {field: host[field] for field in fields}
    result["metal_device"] = metal_device
    return result


def _canonical_json_sha256(domain: str, payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _write_private_json(path: Path, payload: Any) -> None:
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _tail(path: Path, limit: int = 8192) -> str:
    try:
        with path.open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - limit))
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class _ChildProcessMemorySampler:
    """Sample a child process without probing it after it has exited."""

    def __init__(
        self,
        pid: int,
        is_alive: Any,
        *,
        interval_ms: int = MEMORY_SAMPLER_INTERVAL_MS,
        probe: Any = None,
    ) -> None:
        self.pid = pid
        self.is_alive = is_alive
        self.interval_ms = interval_ms
        self.probe = darwin_phys_footprint_bytes if probe is None else probe
        self.samples: list[int] = []
        self.failure: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        value = self.probe(self.pid)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError("process-memory sampler produced an invalid physical footprint")
        self.samples.append(value)

    def _loop(self) -> None:
        interval = self.interval_ms / 1000.0
        deadline = time.monotonic() + interval
        while not self.stop_event.wait(max(0.0, deadline - time.monotonic())):
            try:
                self._sample()
            except BaseException as exc:
                if not self.is_alive():
                    return
                self.failure = exc
                self.stop_event.set()
                return
            deadline += interval

    def start(self) -> None:
        self._sample()
        self.thread = threading.Thread(target=self._loop, name="antfly-gemma4-memory-sampler", daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def stop(self) -> ProcessMemoryMeasurement:
        self.cancel()
        if self.failure is not None:
            raise ContractError(f"process-memory sampler failed: {self.failure}") from self.failure
        if len(self.samples) < 2:
            raise ContractError("process-memory sampler produced fewer than two samples")
        return ProcessMemoryMeasurement(max(self.samples), len(self.samples))


def _read_exact_control_byte(descriptor: int, where: str) -> bytes:
    while True:
        try:
            value = os.read(descriptor, 1)
            break
        except InterruptedError:
            continue
    if len(value) != 1:
        raise ContractError(f"benchmark control channel reached EOF before {where}")
    return value


def _write_exact_control_byte(descriptor: int, value: bytes, where: str) -> None:
    try:
        written = os.write(descriptor, value)
    except OSError as exc:
        raise ContractError(f"benchmark control channel failed while sending {where}: {exc}") from exc
    if written != 1:
        raise ContractError(f"benchmark control channel short-wrote {where}")


def _run_child(
    argv: Sequence[str],
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> ChildRun:
    started_unix_ns = time.time_ns()
    signal_read, signal_write = os.pipe()
    ack_read, ack_write = os.pipe()
    control_failure: list[BaseException] = []
    snapshots: list[Any] = []
    process_measurements: list[ProcessMemoryMeasurement] = []
    signals_seen: list[bytes] = []
    process: subprocess.Popen[bytes] | None = None
    sampler: _ChildProcessMemorySampler | None = None
    control_thread: threading.Thread | None = None
    child_environment = dict(env)
    child_environment[MEASUREMENT_CONTROL["signal_fd_environment"]] = str(signal_write)
    child_environment[MEASUREMENT_CONTROL["ack_fd_environment"]] = str(ack_read)

    def control_loop() -> None:
        nonlocal ack_write
        try:
            for expected, acknowledgement, before_measured in (
                (b"B", b"b", True),
                (b"A", b"a", False),
            ):
                observed = _read_exact_control_byte(signal_read, f"{expected.decode()} signal")
                if observed != expected:
                    raise ContractError(
                        f"benchmark control channel expected {expected!r}, received {observed!r}"
                    )
                signals_seen.append(observed)
                snapshots.append(
                    capture_darwin_system_memory_snapshot(before_measured=before_measured)
                )
                if not before_measured:
                    if sampler is None:
                        raise ContractError("process-memory sampler was not initialized")
                    process_measurements.append(sampler.stop())
                _write_exact_control_byte(ack_write, acknowledgement, f"{acknowledgement.decode()} ACK")
            while True:
                try:
                    trailing = os.read(signal_read, 1)
                    break
                except InterruptedError:
                    continue
            if trailing:
                raise ContractError("benchmark control channel emitted trailing bytes")
        except BaseException as exc:
            control_failure.append(exc)
            try:
                os.close(ack_write)
                ack_write = -1
            except OSError:
                pass

    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=child_environment,
                close_fds=True,
                pass_fds=(signal_write, ack_read),
                start_new_session=True,
            )
            os.close(signal_write)
            signal_write = -1
            os.close(ack_read)
            ack_read = -1
            sampler = _ChildProcessMemorySampler(process.pid, lambda: process.poll() is None)
            try:
                sampler.start()
            except BaseException as exc:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        try:
                            process.kill()
                        except OSError:
                            pass
                process.wait()
                detail = _tail(stderr_path)
                suffix = f"\nchild stderr tail:\n{detail}" if detail else ""
                raise ContractError(
                    f"process-memory sampler could not start before the child exited: {exc}{suffix}"
                ) from exc
            control_thread = threading.Thread(
                target=control_loop,
                name="antfly-gemma4-measurement-control",
                daemon=True,
            )
            control_thread.start()
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                finally:
                    process.wait()
                raise ContractError(f"Antfly benchmark subprocess exceeded {timeout} seconds") from exc

        sampler.cancel()
        control_thread.join(timeout=5)
        if control_thread.is_alive():
            control_failure.append(ContractError("benchmark control reader did not reach EOF"))
        measurement_error: str | None = None
        process_memory: ProcessMemoryMeasurement | None = None
        system_deltas: dict[str, int | float] | None = None
        try:
            # Preserve the original control-channel or Darwin-memory failure.
            # Reporting the derived absence of one stopped sampler first hides
            # the actionable cause (for example a failed after-window
            # memory_pressure query).
            if control_failure:
                raise control_failure[0]
            if len(process_measurements) != 1:
                raise ContractError(
                    "process-memory sampler did not stop exactly at the final measured-step boundary"
                )
            process_memory = process_measurements[0]
            if len(snapshots) != 2:
                raise ContractError("benchmark control channel did not bracket measured optimizer steps")
            system_deltas = darwin_system_memory_deltas(snapshots[0], snapshots[1])
        except BaseException as exc:
            measurement_error = str(exc)
        return ChildRun(
            pid=process.pid,
            started_unix_ns=started_unix_ns,
            returncode=returncode,
            process_memory=process_memory,
            system_deltas=system_deltas,
            measurement_error=measurement_error,
            control_signals_seen=len(signals_seen),
        )
    finally:
        if sampler is not None:
            sampler.cancel()
        for descriptor in (signal_read, signal_write, ack_read, ack_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _validate_telemetry(
    path: Path,
    *,
    request: Mapping[str, Any],
    request_sha256: str,
    command_sha256: str,
    pid: int,
) -> dict[str, Any]:
    telemetry = dict(_mapping(load_json(path), "Antfly benchmark telemetry"))
    _exact_keys(
        telemetry,
        (
            "schema_version", "producer", "bindings", "protocol", "runtime",
            "measurement_control", "timings", "memory",
        ),
        "telemetry",
    )
    if telemetry["schema_version"] != TELEMETRY_SCHEMA_VERSION:
        raise ContractError("unsupported Antfly benchmark telemetry schema")

    producer = _mapping(telemetry["producer"], "telemetry.producer")
    _exact_keys(
        producer,
        (
            "pid", "backend", "strict_metal_execution", "version", "metal_device", "executable_sha256",
            "source_revision", "request_sha256", "command_sha256",
        ),
        "telemetry.producer",
    )
    if _integer(producer["pid"], "telemetry.producer.pid", 1) != pid:
        raise ContractError("telemetry producer PID is not the fresh subprocess PID")
    if producer["backend"] != "metal" or producer["strict_metal_execution"] is not True:
        raise ContractError("telemetry did not attest strict Metal execution")
    expected_implementation = _mapping(request["implementation"], "request.implementation")
    _sha256_text = producer["executable_sha256"]
    if not isinstance(_sha256_text, str) or _SHA256.fullmatch(_sha256_text) is None:
        raise ContractError("telemetry producer executable SHA-256 is malformed")
    if not isinstance(producer["source_revision"], str) or _HEX40.fullmatch(producer["source_revision"]) is None:
        raise ContractError("telemetry producer source revision is malformed")
    for field in ("version", "metal_device", "executable_sha256", "source_revision"):
        if producer[field] != expected_implementation[field]:
            raise ContractError(f"telemetry producer {field} differs from the request")
    if producer["request_sha256"] != request_sha256 or producer["command_sha256"] != command_sha256:
        raise ContractError("telemetry request/command digest differs from the runner")
    for field in ("bindings", "protocol", "runtime", "measurement_control"):
        if telemetry[field] != request[field]:
            raise ContractError(f"telemetry {field} differs from the request")

    protocol = _mapping(request["protocol"], "request.protocol")
    timings = _mapping(telemetry["timings"], "telemetry.timings")
    _exact_keys(
        timings,
        (
            "load_ns", "cold_step_was_first_graph_execution",
            "cold_compile_and_step_ns", "cold_compile_ns",
            "first_steady_step_ns", "warmup_step_ns", "measured_step_ns",
            "optimizer_steps",
        ),
        "telemetry.timings",
    )
    _integer(timings["load_ns"], "telemetry.timings.load_ns")
    if timings["cold_step_was_first_graph_execution"] is not True:
        raise ContractError("telemetry did not prove the cold optimizer window was the first graph execution")
    cold_duration = _integer(
        timings["cold_compile_and_step_ns"],
        "telemetry.timings.cold_compile_and_step_ns",
        1,
    )
    cold_compile = _integer(timings["cold_compile_ns"], "telemetry.timings.cold_compile_ns", 1)
    if cold_compile > cold_duration:
        raise ContractError("telemetry cold compile duration exceeds its optimizer window")
    _integer(timings["first_steady_step_ns"], "telemetry.timings.first_steady_step_ns", 1)
    warmup_durations = timings["warmup_step_ns"]
    measured_durations = timings["measured_step_ns"]
    if not isinstance(warmup_durations, list) or len(warmup_durations) != protocol["warmup_steps"]:
        raise ContractError("telemetry must contain exactly three warmup durations")
    if not isinstance(measured_durations, list) or len(measured_durations) != protocol["measured_steps"]:
        raise ContractError("telemetry must contain exactly twenty measured durations")
    for index, value in enumerate(warmup_durations):
        _integer(value, f"telemetry.timings.warmup_step_ns[{index}]", 1)
    for index, value in enumerate(measured_durations):
        _integer(value, f"telemetry.timings.measured_step_ns[{index}]", 1)
    steps = timings["optimizer_steps"]
    if not isinstance(steps, list):
        raise ContractError("telemetry.timings.optimizer_steps must be an array")
    expected_count = (
        protocol["cold_optimizer_steps"]
        + protocol["first_steady_steps"]
        + protocol["warmup_steps"]
        + protocol["measured_steps"]
    )
    if len(steps) != expected_count:
        raise ContractError(f"telemetry contains {len(steps)} optimizer steps; expected {expected_count}")

    bindings = _mapping(request["bindings"], "request.bindings")
    expected_input = bindings["sequence_length"] * bindings["grad_accum"] * bindings["microbatch"]
    expected_supervised = bindings["supervised_tokens"]
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, f"telemetry step {index}")
        _exact_keys(
            step,
            (
                "index", "phase", "duration_ns", "input_tokens", "supervised_tokens",
                "optimizer_stepped", "explicit_device_sync", "strict_metal_evidence",
                "phase_evidence", "command_plan_evidence",
            ),
            f"telemetry step {index}",
        )
        if step["index"] != index:
            raise ContractError("telemetry optimizer-step indexes must be contiguous from zero")
        warmup_start = protocol["cold_optimizer_steps"] + protocol["first_steady_steps"]
        measured_start = warmup_start + protocol["warmup_steps"]
        expected_phase = (
            "cold" if index == 0 else
            "first" if index == 1 else
            "warmup" if index < measured_start else
            "measured"
        )
        if step["phase"] != expected_phase:
            raise ContractError(f"telemetry step {index} has the wrong phase")
        _integer(step["duration_ns"], f"telemetry step {index}.duration_ns", 1)
        if step["input_tokens"] != expected_input or step["supervised_tokens"] != expected_supervised:
            raise ContractError(f"telemetry step {index} token totals differ from the bound workload")
        if step["optimizer_stepped"] is not True or step["explicit_device_sync"] is not True:
            raise ContractError(f"telemetry step {index} is not a synchronized complete optimizer step")
        evidence = _mapping(step["strict_metal_evidence"], f"telemetry step {index}.strict_metal_evidence")
        _exact_keys(
            evidence,
            (
                "optimizer_backend", "metal_optimizer_steps", "graph_executor_steps",
                "graph_executor_fallback_steps", "native_partitions", "unsupported_ops",
                "interpreter_fallbacks", "runtime_region_fallbacks", "true_host_outputs",
                "host_gradient_tensors",
            ),
            f"telemetry step {index}.strict_metal_evidence",
        )
        for field in (
            "metal_optimizer_steps", "graph_executor_steps", "graph_executor_fallback_steps",
            "native_partitions", "unsupported_ops", "interpreter_fallbacks",
            "runtime_region_fallbacks", "true_host_outputs", "host_gradient_tensors",
        ):
            _integer(evidence[field], f"telemetry step {index}.strict_metal_evidence.{field}")
        if evidence["optimizer_backend"] != "metal" or evidence["metal_optimizer_steps"] != 1:
            raise ContractError(f"telemetry step {index} did not use one Metal optimizer update")
        if evidence["graph_executor_steps"] != bindings["grad_accum"]:
            raise ContractError(f"telemetry step {index} did not execute every accumulated microbatch")
        zero_fields = (
            "graph_executor_fallback_steps", "native_partitions", "unsupported_ops",
            "interpreter_fallbacks", "runtime_region_fallbacks", "true_host_outputs",
            "host_gradient_tensors",
        )
        if any(evidence[field] != 0 for field in zero_fields):
            raise ContractError(f"telemetry step {index} contains a forbidden strict-Metal fallback")

        phase_evidence = _mapping(step["phase_evidence"], f"telemetry step {index}.phase_evidence")
        _exact_keys(
            phase_evidence,
            PHASE_EVIDENCE_FIELDS,
            f"telemetry step {index}.phase_evidence",
        )
        for field in PHASE_EVIDENCE_FIELDS:
            _integer(phase_evidence[field], f"telemetry step {index}.phase_evidence.{field}")
        expected_compile_ns = cold_compile if index == 0 else 0
        if phase_evidence["compile_ns"] != expected_compile_ns:
            raise ContractError(f"telemetry step {index} phase compile duration differs from the summary")

        command_evidence = _mapping(
            step["command_plan_evidence"],
            f"telemetry step {index}.command_plan_evidence",
        )
        _exact_keys(
            command_evidence,
            COMMAND_PLAN_EVIDENCE_FIELDS,
            f"telemetry step {index}.command_plan_evidence",
        )
        for field in COMMAND_PLAN_EVIDENCE_FIELDS:
            _integer(command_evidence[field], f"telemetry step {index}.command_plan_evidence.{field}")
        if command_evidence["graph_executor_partitions"] < bindings["grad_accum"]:
            raise ContractError(f"telemetry step {index} command plan omitted graph partitions")
        command_dispatches = command_evidence["graph_executor_command_dispatches"]
        if command_dispatches == 0:
            raise ContractError(f"telemetry step {index} command plan omitted graph dispatches")
        if command_evidence["graph_executor_planned_dispatches"] > command_dispatches:
            raise ContractError(f"telemetry step {index} planned dispatches exceed graph dispatches")
        if sum(command_evidence[field] for field in COMMAND_ATTRIBUTION_FIELDS) > command_dispatches:
            raise ContractError(f"telemetry step {index} command attribution exceeds graph dispatches")
        cce_forward = command_evidence["metal_linear_cce_forward_calls"]
        cce_backward = command_evidence["metal_linear_cce_backward_calls"]
        cce_state_events = (
            command_evidence["metal_linear_cce_forward_state_hits"]
            + command_evidence["metal_linear_cce_forward_state_misses"]
        )
        if cce_state_events != cce_backward or cce_backward != cce_forward or command_evidence["metal_linear_cce_forward_state_misses"] != 0:
            raise ContractError(f"telemetry step {index} linear CCE route evidence is inconsistent")
        if (command_evidence["metal_linear_cce_peak_scratch_bytes"] != 0) != (cce_forward != 0):
            raise ContractError(f"telemetry step {index} linear CCE scratch evidence is inconsistent")
        cache_hits = command_evidence["graph_executor_plan_cache_hits"]
        cache_misses = command_evidence["graph_executor_plan_cache_misses"]
        if cache_hits + cache_misses != bindings["grad_accum"]:
            raise ContractError(f"telemetry step {index} command plan has the wrong cache lookup count")
        expected_cache_misses = 1 if index == 0 else 0
        if cache_misses != expected_cache_misses:
            raise ContractError(f"telemetry step {index} command plan has an unexpected cache miss count")
        plan_was_built = cache_misses != 0
        if ((phase_evidence["graph_executor_plan_build_ns"] != 0) != plan_was_built or
                (phase_evidence["graph_executor_buffer_plan_build_ns"] != 0) != plan_was_built):
            raise ContractError(f"telemetry step {index} plan-build phases differ from cache evidence")

    if steps[0]["duration_ns"] != timings["cold_compile_and_step_ns"]:
        raise ContractError("telemetry cold summary differs from the optimizer-step record")
    if steps[1]["duration_ns"] != timings["first_steady_step_ns"]:
        raise ContractError("telemetry first-steady summary differs from the optimizer-step record")
    if [step["duration_ns"] for step in steps[warmup_start:measured_start]] != warmup_durations:
        raise ContractError("telemetry warmup summary differs from the optimizer-step records")
    if [step["duration_ns"] for step in steps[measured_start:]] != measured_durations:
        raise ContractError("telemetry measured summary differs from the optimizer-step records")

    memory = _mapping(telemetry["memory"], "telemetry.memory")
    _exact_keys(memory, ("peak_bytes", "source"), "telemetry.memory")
    _integer(memory["peak_bytes"], "telemetry.memory.peak_bytes", 1)
    if memory["source"] != PEAK_MEMORY_SOURCE:
        raise ContractError("telemetry peak memory is not Darwin lifetime maximum physical footprint")
    return telemetry


def _atomic_publish_sample(output: Path, payload: Mapping[str, Any], lock: Mapping[str, Any], lock_path: Path) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ContractError(f"refusing to replace existing benchmark evidence: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.staging-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as staging:
            os.fchmod(staging.fileno(), 0o644)
            data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            staging.write(data)
            staging.flush()
            os.fsync(staging.fileno())
        validate_sample(temporary, lock, lock_path)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ContractError(f"refusing to replace existing benchmark evidence: {target}") from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_publish_diagnostic(output: Path, payload: Mapping[str, Any]) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ContractError(f"refusing to replace existing diagnostic output: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.staging-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as staging:
            os.fchmod(staging.fileno(), 0o644)
            data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            staging.write(data)
            staging.flush()
            os.fsync(staging.fileno())
        validate_diagnostic_payload(load_json(temporary))
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ContractError(f"refusing to replace existing diagnostic output: {target}") from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _case_payload(
    args: argparse.Namespace,
    model: Mapping[str, Any],
    adapter: AdapterBinding,
    train: PreparedBinding,
) -> dict[str, Any]:
    return {
        "model_key": args.model_key,
        "revision": model["revision"],
        "local_artifact_sha256": model["local_artifact_sha256"],
        "target_preset": adapter.target_preset,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "sequence_length": args.sequence_length,
        "grad_accum": args.grad_accum,
        "microbatch": 1,
        "prepared": train.case,
        "initial_adapter": {
            "schema_version": "antfly_gemma4_initial_adapter_semantics/v1",
            "semantic_sha256": adapter.semantic_sha256,
            "tensor_count": adapter.tensor_count,
            "tensor_dtype": "float32",
        },
        "target_inventory": {
            "schema_version": "antfly_gemma4_target_inventory/v1",
            "sha256": adapter.target_inventory_sha256,
            "module_count": len(adapter.canonical_modules),
            "canonical_modules": list(adapter.canonical_modules),
        },
    }


def assemble_sample(
    *,
    args: argparse.Namespace,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_revision: str,
    producer_source: Mapping[str, Any],
    antfly_version: str,
    executable_sha256: str,
    command_sha256: str,
    pid: int,
    started_unix_ns: int,
    hardware: Mapping[str, Any],
    model: Mapping[str, Any],
    adapter: AdapterBinding,
    train: PreparedBinding,
    protocol: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    child: ChildRun,
) -> dict[str, Any]:
    """Central sample assembly point; telemetry validation happens beforehand."""
    timings = _mapping(telemetry["timings"], "telemetry.timings")
    if child.process_memory is None or child.system_deltas is None or child.measurement_error is not None:
        raise ContractError(f"required process/system memory evidence is unavailable: {child.measurement_error}")
    case = _case_payload(args, model, adapter, train)
    return {
        "schema_version": BENCHMARK_SAMPLE_SCHEMA_VERSION,
        "framework": "antfly-zig-metal",
        "oracle_lock_sha256": lock_digest(lock_path),
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "repetition": args.repetition,
        "sequence_index": args.sequence_index,
        "implementation": {
            "command_sha256": command_sha256,
            "producer_source": dict(producer_source),
            "antfly": {
                "version": antfly_version,
                "source_revision": source_revision,
                "source_clean": producer_source.get("source_clean") is True,
                "executable_sha256": executable_sha256,
            },
        },
        "process": {"pid": pid, "started_unix_ns": started_unix_ns},
        "hardware": dict(hardware),
        "case": case,
        "semantic_contract": expected_semantic_contract(lock, case),
        "protocol": dict(protocol),
        "metrics": {
            "load_seconds": timings["load_ns"] / 1_000_000_000,
            "cold_compile_and_step_seconds": timings["cold_compile_and_step_ns"] / 1_000_000_000,
            "first_steady_step_seconds": timings["first_steady_step_ns"] / 1_000_000_000,
            "step_seconds": [duration / 1_000_000_000 for duration in timings["measured_step_ns"]],
            "input_tokens": train.input_tokens,
            "supervised_tokens": train.supervised_tokens,
            "execution_evidence": {
                "schema_version": "antfly_gemma4_zig_execution_evidence/v2",
                "optimizer_steps": [
                    {
                        "index": step["index"],
                        "phase": step["phase"],
                        "phase_evidence": dict(step["phase_evidence"]),
                        "command_plan_evidence": dict(step["command_plan_evidence"]),
                    }
                    for step in timings["optimizer_steps"]
                ],
            },
            "memory": {
                "process_peak_phys_footprint_bytes": child.process_memory.peak_phys_footprint_bytes,
                "sampler_interval_ms": MEMORY_SAMPLER_INTERVAL_MS,
                "sampler_sample_count": child.process_memory.sample_count,
                "framework_allocator_peak_bytes": None,
                "framework_allocator_peak_source": "antfly-metal-allocator-unavailable",
                "system_deltas": dict(child.system_deltas),
            },
        },
    }


def build_diagnostic_payload(
    sample: Mapping[str, Any],
    environment_overrides: Mapping[str, str],
) -> dict[str, Any]:
    payload = dict(sample)
    payload["schema_version"] = DIAGNOSTIC_SAMPLE_SCHEMA_VERSION
    payload["diagnostic"] = {
        "release_eligible": False,
        "release_gates_enforced": False,
        "release_blockers": [DIAGNOSTIC_RELEASE_BLOCKER],
        "publication_contract": f"never-valid-as-{BENCHMARK_SAMPLE_SCHEMA_VERSION}",
        "environment_overrides": dict(environment_overrides),
    }
    return payload


def validate_diagnostic_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    _exact_keys(
        payload,
        (
            "schema_version", "framework", "oracle_lock_sha256", "campaign_id", "run_id",
            "repetition", "sequence_index", "diagnostic", "implementation", "process",
            "hardware", "case", "semantic_contract", "protocol", "metrics",
        ),
        "Antfly diagnostic payload",
    )
    if payload["schema_version"] != DIAGNOSTIC_SAMPLE_SCHEMA_VERSION:
        raise ContractError("Antfly diagnostic payload uses the wrong schema")
    if payload["framework"] != "antfly-zig-metal":
        raise ContractError("Antfly diagnostic payload uses the wrong framework")
    diagnostic = _mapping(payload["diagnostic"], "Antfly diagnostic payload.diagnostic")
    _exact_keys(
        diagnostic,
        (
            "release_eligible", "release_gates_enforced", "release_blockers",
            "publication_contract", "environment_overrides",
        ),
        "Antfly diagnostic payload.diagnostic",
    )
    expected_diagnostic = {
        "release_eligible": False,
        "release_gates_enforced": False,
        "release_blockers": [DIAGNOSTIC_RELEASE_BLOCKER],
        "publication_contract": f"never-valid-as-{BENCHMARK_SAMPLE_SCHEMA_VERSION}",
    }
    if {key: diagnostic[key] for key in expected_diagnostic} != expected_diagnostic:
        raise ContractError("Antfly diagnostic payload release boundary drifted")
    _validate_diagnostic_environment_mapping(
        diagnostic["environment_overrides"],
        "Antfly diagnostic payload.diagnostic.environment_overrides",
    )

    implementation = _mapping(payload["implementation"], "Antfly diagnostic implementation")
    producer = _mapping(
        implementation.get("producer_source"),
        "Antfly diagnostic implementation.producer_source",
    )
    _exact_keys(
        producer,
        (
            "schema_version", "relative_path", "source_revision", "source_tree", "source_clean",
            "working_tree_status_sha256", "files", "dirty_entry_count", "source_sha256",
            "manifest_sha256",
        ),
        "Antfly diagnostic producer source",
    )
    if producer["schema_version"] != DIAGNOSTIC_SOURCE_SCHEMA_VERSION:
        raise ContractError("Antfly diagnostic payload omitted diagnostic source identity")
    if producer["relative_path"] != ANTFLY_RUNNER_RELATIVE_PATH:
        raise ContractError("Antfly diagnostic payload producer entrypoint differs")
    if not isinstance(producer["source_clean"], bool):
        raise ContractError("Antfly diagnostic producer source cleanliness is not boolean")
    if not isinstance(producer["dirty_entry_count"], int) or producer["dirty_entry_count"] < 0:
        raise ContractError("Antfly diagnostic producer dirty-entry count is invalid")
    if _SHA256.fullmatch(str(producer["working_tree_status_sha256"])) is None:
        raise ContractError("Antfly diagnostic producer status digest is malformed")
    antfly = _mapping(implementation.get("antfly"), "Antfly diagnostic implementation.antfly")
    if antfly.get("source_clean") is not producer["source_clean"]:
        raise ContractError("Antfly diagnostic implementation source cleanliness differs")

    manifest_input = {
        "relative_path": producer["relative_path"],
        "source_revision": producer["source_revision"],
        "source_tree": producer["source_tree"],
        "source_clean": producer["source_clean"],
        "working_tree_status_sha256": producer["working_tree_status_sha256"],
        "files": producer["files"],
    }
    if producer["manifest_sha256"] != _diagnostic_source_manifest_sha256(manifest_input):
        raise ContractError("Antfly diagnostic producer manifest digest differs")
    return payload


def _validate_diagnostic_environment_mapping(
    raw: Any,
    where: str,
) -> dict[str, str]:
    environment = _mapping(raw, where)
    result: dict[str, str] = {}
    for name, value in environment.items():
        if not isinstance(name, str) or _TERMITE_ENV_NAME.fullmatch(name) is None:
            raise ContractError(f"{where}: only TERMITE_[A-Z0-9_]+ names are allowed")
        if name in STRICT_METAL_ENV:
            raise ContractError(f"{where}: cannot replace locked strict-Metal variable {name}")
        if not isinstance(value, str) or "\x00" in value:
            raise ContractError(f"{where}.{name}: expected a NUL-free string")
        result[name] = value
    return dict(sorted(result.items()))


def diagnostic_environment_overrides(
    entries: list[str],
    *,
    diagnostic_only: bool,
) -> dict[str, str]:
    if entries and not diagnostic_only:
        raise ContractError("--diagnostic-env requires --diagnostic-only")
    parsed: dict[str, str] = {}
    for entry in entries:
        name, separator, value = entry.partition("=")
        if not separator:
            raise ContractError("--diagnostic-env must use KEY=VALUE syntax")
        if name in parsed:
            raise ContractError(f"--diagnostic-env repeats {name}")
        parsed[name] = value
    return _validate_diagnostic_environment_mapping(parsed, "--diagnostic-env")


def run(args: argparse.Namespace) -> dict[str, Any]:
    environment_overrides = diagnostic_environment_overrides(
        args.diagnostic_env,
        diagnostic_only=args.diagnostic_only,
    )
    lock_path = args.lock.expanduser().resolve(strict=True)
    lock = load_lock(lock_path)
    executable = _regular_file(args.antfly, "Antfly executable", executable=True)
    executable_sha = prefixed_sha256(executable)
    antfly_version = executable_version(executable)
    if args.diagnostic_only:
        source_root = args.source_root.expanduser().resolve(strict=True)
        producer_source = diagnostic_producer_source(source_root)
        source_revision = producer_source["source_revision"]
    else:
        source_root, source_revision = source_identity(args.source_root)
        producer_source = attest_benchmark_producer_source(
            SCRIPT_PATH,
            expected_entrypoint=ANTFLY_RUNNER_RELATIVE_PATH,
        )
        if producer_source["source_revision"] != source_revision:
            raise ContractError("Antfly wrapper provenance differs from the executable source checkout")

    gate = _mapping(lock["performance_gate"], "performance_gate")
    if args.sequence_length not in gate["primary_sequence_lengths"]:
        raise ContractError("sequence length is outside the locked performance matrix")
    if args.grad_accum not in gate["gradient_accumulation"]:
        raise ContractError("gradient accumulation is outside the locked performance matrix")
    if args.target_preset not in ("peft-qv", "text-all-linear"):
        raise ContractError("target preset is outside the locked performance matrix")

    model = verify_model_directory(lock, args.model_key, args.model_dir)
    model_snapshot = _tree_snapshot(Path(model["directory"]))
    train = prepared_binding(
        args.train_prepared,
        sequence_length=args.sequence_length,
        grad_accum=args.grad_accum,
        example_index=args.example_index,
        source_dataset=args.train_source_dataset,
    )
    eval_summary, _eval_example = load_prepared_example(args.eval_prepared.expanduser().resolve(strict=True), 0)
    verify_prepared_source_dataset(eval_summary, args.eval_source_dataset)
    eval_artifact_sha = prefixed_sha256(args.eval_prepared.expanduser().resolve(strict=True))
    eval_info = args.eval_prepared.expanduser().resolve(strict=True).stat()
    eval_snapshot = ((
        args.eval_prepared.expanduser().resolve(strict=True).name,
        eval_info.st_dev,
        eval_info.st_ino,
        eval_info.st_size,
        eval_info.st_mtime_ns,
    ),)
    adapter = adapter_binding(
        args.adapter_dir,
        lock,
        args.model_key,
        args.target_preset,
        {
            "base_model_sha256": train.base_model_sha256,
            "tokenizer_sha256": train.tokenizer_sha256,
            "chat_template_sha256": train.chat_template_sha256,
        },
    )

    provenance_triplet = (train.base_model_sha256, train.tokenizer_sha256, train.chat_template_sha256)
    if provenance_triplet != (adapter.base_model_sha256, adapter.tokenizer_sha256, adapter.chat_template_sha256):
        raise ContractError("training prepared artifact and adapter provenance differ")
    if provenance_triplet != (
        eval_summary["base_model_sha256"], eval_summary["tokenizer_sha256"], eval_summary["chat_template_sha256"],
    ):
        raise ContractError("training and evaluation prepared artifacts have different model provenance")
    if train.case["source_dataset_sha256"] == eval_summary["source_dataset_sha256"]:
        raise ContractError("benchmark train and evaluation prepared artifacts must come from disjoint sources")

    hardware = _host_identity(args.metal_device)
    protocol = {
        "fresh_process": True,
        "cold_optimizer_steps": 1,
        "cold_step_mutates_optimizer_state": True,
        "first_steady_steps": 1,
        "warmup_steps": int(lock["mlx_reference"]["warmup_steps"]),
        "measured_steps": int(lock["mlx_reference"]["measured_steps"]),
        "explicit_device_sync": True,
        "sync_point": SYNC_POINT,
        "timed_unit": TIMED_UNIT,
    }
    case = _case_payload(args, model, adapter, train)
    semantic_contract = expected_semantic_contract(lock, case)
    bindings = {
        "oracle_lock_sha256": lock_digest(lock_path),
        "model_key": args.model_key,
        "model_revision": model["revision"],
        "local_artifact_sha256": model["local_artifact_sha256"],
        "initial_adapter_semantic_sha256": adapter.semantic_sha256,
        "target_inventory_sha256": adapter.target_inventory_sha256,
        "target_count": len(adapter.canonical_modules),
        "semantic_contract_sha256": semantic_contract["sha256"],
        "train_prepared_sha256": train.artifact_sha256,
        "eval_prepared_sha256": eval_artifact_sha,
        "workload_sha256": train.workload_sha256,
        "prepared_example_index": args.example_index,
        "target_preset": adapter.target_preset,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "sequence_length": args.sequence_length,
        "grad_accum": args.grad_accum,
        "microbatch": 1,
        "supervised_tokens": train.supervised_tokens,
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "implementation": {
            "version": antfly_version,
            "executable_sha256": executable_sha,
            "source_revision": source_revision,
            "metal_device": args.metal_device,
        },
        "bindings": bindings,
        "protocol": protocol,
        "runtime": dict(lock["benchmark_contract"]["runtime"]),
        "measurement_control": dict(MEASUREMENT_CONTROL),
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ContractError(f"refusing to replace existing benchmark evidence: {output}")
    with tempfile.TemporaryDirectory(prefix=".antfly-gemma4-bench-", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        request_path = temporary / "request.json"
        telemetry_path = temporary / "telemetry.json"
        stdout_path = temporary / "stdout.log"
        stderr_path = temporary / "stderr.log"
        run_output = temporary / "train-output"
        _write_private_json(request_path, request)
        request_sha = prefixed_sha256(request_path)

        step_count = (
            protocol["cold_optimizer_steps"]
            + protocol["first_steady_steps"]
            + protocol["warmup_steps"]
            + protocol["measured_steps"]
        )
        training_contract = _mapping(lock["training_contract"], "training_contract")
        command = [
            str(executable), "inference", "finetune", "train", "gemma4-lora",
            "--model", str(Path(model["directory"])),
            "--adapter", str(args.adapter_dir.expanduser().resolve(strict=True)),
            "--train-prepared", str(args.train_prepared.expanduser().resolve(strict=True)),
            "--eval-prepared", str(args.eval_prepared.expanduser().resolve(strict=True)),
            "--out", str(run_output),
            "--backend", "metal",
            "--trainer", "autodiff",
            "--lr", str(training_contract["learning_rate"]),
            "--max-grad-norm", str(training_contract["max_grad_norm"]),
            "--seed", str(training_contract["seed"]),
            "--max-examples", str(args.grad_accum),
            "--eval-max-examples", "1",
            "--grad-accum", str(args.grad_accum),
            "--epochs", str(step_count),
            "--benchmark-request", str(request_path),
            "--benchmark-telemetry-out", str(telemetry_path),
        ]
        command_identity = {
                "argv": command,
                "strict_environment": STRICT_METAL_ENV,
                "measurement_control": MEASUREMENT_CONTROL,
                "request_sha256": request_sha,
                "initial_adapter_semantic_sha256": adapter.semantic_sha256,
                "train_prepared_sha256": train.artifact_sha256,
                "eval_prepared_sha256": eval_artifact_sha,
        }
        if environment_overrides:
            command_identity["diagnostic_environment_overrides"] = environment_overrides
        command_sha = _canonical_json_sha256(COMMAND_DIGEST_DOMAIN, command_identity)
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith("TERMITE_"):
                del environment[name]
        environment.update(STRICT_METAL_ENV)
        environment.update(environment_overrides)
        environment["ANTFLY_GEMMA4_BENCHMARK_COMMAND_SHA256"] = command_sha

        child = _run_child(
            command,
            environment,
            stdout_path,
            stderr_path,
            args.timeout_seconds,
        )
        if child.measurement_error is not None and child.control_signals_seen > 0:
            detail = _tail(stderr_path).strip()
            suffix = f"; child exit status {child.returncode}"
            if detail:
                suffix += f"\nchild stderr tail:\n{detail}"
            raise ContractError(
                f"benchmark measurement control failed closed: {child.measurement_error}{suffix}"
            )
        if child.returncode != 0 or not telemetry_path.is_file():
            detail = _tail(stderr_path).strip()
            suffix = f"\nchild stderr tail:\n{detail}" if detail else ""
            raise BenchmarkInterfaceUnavailable(
                "Antfly did not complete and publish the required benchmark telemetry; "
                "no sample was emitted. Required Zig change: " + REQUIRED_ZIG_CHANGE + suffix
            )
        telemetry = _validate_telemetry(
            telemetry_path,
            request=request,
            request_sha256=request_sha,
            command_sha256=command_sha,
            pid=child.pid,
        )
        if child.measurement_error is not None:
            raise ContractError(f"benchmark measurement control failed closed: {child.measurement_error}")
        assert child.system_deltas is not None
        if not args.diagnostic_only:
            enforce_system_memory_gates(lock, child.system_deltas)

        # Reject input or binary mutation during a long-running sample.  The
        # pinned model was content-verified before launch; stable inode/size/
        # timestamp snapshots avoid hashing multi-gigabyte weights twice.
        if prefixed_sha256(executable) != executable_sha:
            raise ContractError("Antfly executable changed during the benchmark")
        _assert_snapshot_unchanged(Path(model["directory"]), model_snapshot, "locked model artifact")
        _assert_snapshot_unchanged(args.adapter_dir, adapter.snapshot, "adapter artifact")
        _assert_file_snapshot_unchanged(args.train_prepared, train.snapshot, "training prepared artifact")
        _assert_file_snapshot_unchanged(args.eval_prepared, eval_snapshot, "evaluation prepared artifact")
        if args.diagnostic_only:
            if diagnostic_producer_source(source_root) != producer_source:
                raise ContractError("Antfly diagnostic producer source changed during the benchmark")
        else:
            _, source_revision_after = source_identity(source_root)
            if source_revision_after != source_revision:
                raise ContractError("Antfly source revision changed during the benchmark")

        sample = assemble_sample(
            args=args,
            lock=lock,
            lock_path=lock_path,
            source_revision=source_revision,
            producer_source=producer_source,
            antfly_version=antfly_version,
            executable_sha256=executable_sha,
            command_sha256=command_sha,
            pid=child.pid,
            started_unix_ns=child.started_unix_ns,
            hardware=hardware,
            model=model,
            adapter=adapter,
            train=train,
            protocol=protocol,
            telemetry=telemetry,
            child=child,
        )
        if args.diagnostic_only:
            diagnostic = build_diagnostic_payload(sample, environment_overrides)
            _atomic_publish_diagnostic(output, diagnostic)
            return diagnostic
        _atomic_publish_sample(output, sample, lock, lock_path)
        return sample


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lock", type=Path, default=LOCK_PATH)
    result.add_argument("--antfly", type=Path, required=True, help="installed Antfly executable")
    result.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="source checkout used to build the executable; must be clean unless --diagnostic-only is set",
    )
    result.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--train-prepared", type=Path, required=True)
    result.add_argument("--eval-prepared", type=Path, required=True)
    result.add_argument("--example-index", required=True, type=int)
    result.add_argument("--train-source-dataset", type=Path)
    result.add_argument("--eval-source-dataset", type=Path)
    result.add_argument("--target-preset", required=True, choices=("peft-qv", "text-all-linear"))
    result.add_argument("--sequence-length", required=True, type=int, choices=(128, 512, 2048))
    result.add_argument("--grad-accum", required=True, type=int, choices=(1, 4))
    result.add_argument("--campaign-id", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--repetition", required=True, type=int)
    result.add_argument("--sequence-index", required=True, type=int)
    result.add_argument("--metal-device", required=True)
    result.add_argument("--timeout-seconds", type=int, default=21_600)
    result.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Run the locked workload without publishing release evidence. The output uses a "
            "separate diagnostic schema that the comparison gate never accepts."
        ),
    )
    result.add_argument(
        "--diagnostic-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Diagnostic-only TERMITE_* child override for same-binary A/B runs; may be repeated. "
            "Overrides are recorded verbatim in the diagnostic artifact; do not pass secrets."
        ),
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.repetition < 0 or args.sequence_index < 0 or args.example_index < 0 or args.timeout_seconds <= 0:
            raise ContractError(
                "repetition/sequence-index/example-index must be non-negative and timeout must be positive"
            )
        run(args)
        print(str(args.output.expanduser().resolve()))
        return 0
    except BenchmarkInterfaceUnavailable as exc:
        print(f"Antfly Gemma4 benchmark unavailable: {exc}", file=sys.stderr)
        return 3
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"Antfly Gemma4 benchmark contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
