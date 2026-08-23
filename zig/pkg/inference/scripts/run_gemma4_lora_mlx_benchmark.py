#!/usr/bin/env python3
"""Run one locked, fresh-process MLX-LM Gemma4 LoRA benchmark sample.

This is deliberately a one-sample executable.  Campaign ordering and release
gates belong to ``bench_gemma4_lora_mlx_zig.py``.  The runner consumes exact
prepared token/label arrays, never tokenizes, never downloads, and publishes a
sample only after all locked inputs are rechecked for drift.

MLX imports are intentionally lazy so contract tests need only the standard
library.  A real run requires editable/importable checkouts at the exact MLX
and MLX-LM revisions pinned by ``gemma4_oracle.lock.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bench_gemma4_lora_mlx_zig import (
    INITIAL_ADAPTER_DOMAIN,
    MLX_RUNNER_RELATIVE_PATH,
    PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION,
    PRECISION_EVIDENCE_SCHEMA_VERSION,
    PRECISION_RUNNER_SCHEMA_VERSION,
    SYNC_POINT,
    TIMED_UNIT,
    benchmark_workload_sha256,
    canonical_moment_inventory_sha256,
    canonical_precision_sample_binding_sha256,
    canonical_target_inventory_sha256,
    canonical_tensor_inventory_sha256,
    expected_semantic_contract,
    validate_precision_evidence_payload,
    validate_sample,
)
from gemma4_oracle_contract import (
    BENCHMARK_PRODUCER_RELATIVE_PATHS,
    BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
    BENCHMARK_SAMPLE_SCHEMA_VERSION,
    ContractError,
    LOCK_PATH,
    MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
    attest_benchmark_producer_source,
    canonical_mlx_native_artifact_inventory_sha256,
    canonicalize_adapter_tensor_name,
    canonicalize_module_name,
    hardware_fingerprint,
    load_json,
    load_lock,
    load_prepared_example,
    lock_digest,
    prefixed_sha256,
    read_adapter_config,
    validate_target_inventory,
    verify_import_source,
    verify_model_directory,
    verify_packages,
    verify_prepared_source_dataset,
    verify_requirements_match_lock,
    verify_source_checkout,
)


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
MLX_REQUIREMENTS_PATH = SCRIPT_DIR / "requirements-gemma4-mlx-reference.txt"
PROCESS_STARTED_UNIX_NS = time.time_ns()
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
OFFLINE_ENVIRONMENT = {
    "DO_NOT_TRACK": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
PREPARED_CHAT_TEMPLATE_IDENTITY = b"antfly_gemma_chat/v1"
MEMORY_SAMPLER_INTERVAL_MS = 10
RUSAGE_INFO_V4 = 4
INTERNAL_WORKER_FLAG = "--_antfly-gemma4-mlx-worker-fd"
CONTROL_PROTOCOL_VERSION = "antfly_gemma4_mlx_benchmark_control/v2"
MAX_CONTROL_MESSAGE_BYTES = 4 * 1024 * 1024
CONTROL_IDLE_TIMEOUT_SECONDS = 4 * 60 * 60
DIAGNOSTIC_SAMPLE_SCHEMA_VERSION = "antfly_gemma4_mlx_diagnostic_sample/v1"
DIAGNOSTIC_SOURCE_SCHEMA_VERSION = "antfly_gemma4_diagnostic_producer_source/v1"
DIAGNOSTIC_RELEASE_BLOCKER = "benchmark-producer-source-is-not-clean-committed-release-evidence"
_PROCESS_CLAIMED = False
_DARWIN_PHYS_FOOTPRINT_PROBE: Callable[[int], int] | None = None


@dataclass(frozen=True)
class Workload:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    grad_accum: int
    input_tokens: int
    supervised_tokens: int
    digest: str


@dataclass(frozen=True)
class StepMeasurements:
    cold_compile_and_step_seconds: float
    first_steady_step_seconds: float
    step_seconds: tuple[float, ...]


@dataclass(frozen=True)
class AdapterTensor:
    source_name: str
    shape: tuple[int, int]
    data_sha256: str
    data_offset_start: int
    data_offset_end: int


@dataclass(frozen=True)
class AdapterArtifact:
    directory: Path
    checkpoint: Path
    checkpoint_sha256: str
    config_sha256: str
    semantic_sha256: str
    semantics: dict[str, Any]
    tensors: dict[tuple[str, str], AdapterTensor]
    bound_files: tuple[Path, ...]


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class DarwinSystemMemorySnapshot:
    page_size: int
    pageins: int
    pageouts: int
    swapins: int
    swapouts: int
    pressure_available_percent: float


@dataclass(frozen=True)
class ProcessMemoryMeasurement:
    peak_phys_footprint_bytes: int
    sample_count: int


@dataclass(frozen=True)
class Preflight:
    lock: dict[str, Any]
    model: dict[str, Any]
    prepared_summary: dict[str, Any]
    prepared: dict[str, Any]
    adapter: AdapterArtifact
    workload: Workload
    bound_files: tuple[FileIdentity, ...]


class PrecisionEvidenceRecorder:
    """Collect stable, closed observations made while tracing and executing one run."""

    _REQUIRED = (
        "base_model_storage",
        "lora_parameter_storage",
        "optimizer_moment_storage",
        "loss",
    )
    _GRADIENT_STAGES = ("raw", "accumulated", "clipped")

    def __init__(self) -> None:
        self._observations: dict[str, dict[str, Any]] = {}
        self._gradient_stages: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _copy(payload: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))

    def record(self, name: str, observation: Mapping[str, Any]) -> None:
        if name not in self._REQUIRED:
            raise ContractError(f"unsupported precision observation {name!r}")
        copied = self._copy(observation)
        previous = self._observations.get(name)
        if previous is not None and previous != copied:
            raise ContractError(f"precision observation {name} changed within one run")
        self._observations[name] = copied

    def record_gradient(self, stage: str, tensors: Sequence[Mapping[str, Any]]) -> None:
        if stage not in self._GRADIENT_STAGES:
            raise ContractError(f"unsupported gradient precision stage {stage!r}")
        copied_tensors = self._copy({"tensors": list(tensors)})["tensors"]
        observation = {
            "stage": stage,
            "inventory_sha256": canonical_tensor_inventory_sha256(copied_tensors),
            "tensors": copied_tensors,
        }
        previous = self._gradient_stages.get(stage)
        if previous is not None and previous != observation:
            raise ContractError(f"gradient precision observation {stage} changed within one run")
        self._gradient_stages[stage] = observation

    def finalize(self) -> dict[str, Any]:
        missing = [name for name in self._REQUIRED if name not in self._observations]
        missing.extend(
            f"gradient_storage.{stage}"
            for stage in self._GRADIENT_STAGES
            if stage not in self._gradient_stages
        )
        if missing:
            raise ContractError(f"precision evidence is incomplete: {missing}")
        lora_count = self._observations["lora_parameter_storage"]["tensor_count"]
        result = {name: self._copy(self._observations[name]) for name in self._REQUIRED}
        result["gradient_storage"] = {
            "evidence_kind": "compiled-gradient-tree-inventory",
            "dtype": "float32",
            "tensor_count": lora_count,
            "stages": [self._copy(self._gradient_stages[stage]) for stage in self._GRADIENT_STAGES],
        }
        return {
            name: result[name]
            for name in (
                "base_model_storage", "lora_parameter_storage", "gradient_storage",
                "optimizer_moment_storage", "loss",
            )
        }


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return result


def _campaign_identifier(value: str) -> str:
    if ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "identifier must be 1-128 ASCII letters, digits, '.', '_' or '-' and start alphanumeric"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit one pinned, fresh-process MLX-LM Gemma4 LoRA benchmark sample."
    )
    parser.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        required=True,
        type=Path,
        help="Local F32 PEFT or Antfly seed adapter whose exact A/B values initialize MLX.",
    )
    parser.add_argument("--example-index", required=True, type=_nonnegative_int)
    parser.add_argument("--target-preset", required=True, choices=("peft-qv", "text-all-linear"))
    parser.add_argument("--sequence-length", required=True, type=_positive_int)
    parser.add_argument("--grad-accum", required=True, type=_positive_int)
    parser.add_argument("--campaign-id", required=True, type=_campaign_identifier)
    parser.add_argument("--run-id", required=True, type=_campaign_identifier)
    parser.add_argument("--repetition", required=True, type=_nonnegative_int)
    parser.add_argument("--sequence-index", required=True, type=_nonnegative_int)
    parser.add_argument("--mlx-source", required=True, type=Path)
    parser.add_argument("--mlx-lm-source", required=True, type=Path)
    parser.add_argument(
        "--mlx-build-attestation",
        required=True,
        type=Path,
        help="Strict local attestation binding the loaded MLX native runtime artifacts.",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Run the locked workload without publishing release evidence. The output uses a "
            "separate diagnostic schema and is never accepted by the comparison gate."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def claim_fresh_process() -> None:
    global _PROCESS_CLAIMED
    if _PROCESS_CLAIMED:
        raise ContractError("one runner process may publish only one benchmark sample")
    _PROCESS_CLAIMED = True


def force_offline_environment() -> None:
    for name in tuple(os.environ):
        if name.startswith("DYLD_"):
            del os.environ[name]
    for name, value in OFFLINE_ENVIRONMENT.items():
        os.environ[name] = value
    # The worker calls this before importing either source tree.  Set the
    # interpreter flag as well because changing only the environment after
    # process startup does not update Python's bytecode policy.
    sys.dont_write_bytecode = True


def _load_darwin_phys_footprint_probe() -> Callable[[int], int]:
    """Bind the Darwin V4 rusage ABI lazily, after platform admission."""
    import ctypes

    class RusageInfoV4(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            *(
                (name, ctypes.c_uint64)
                for name in (
                    "ri_user_time",
                    "ri_system_time",
                    "ri_pkg_idle_wkups",
                    "ri_interrupt_wkups",
                    "ri_pageins",
                    "ri_wired_size",
                    "ri_resident_size",
                    "ri_phys_footprint",
                    "ri_proc_start_abstime",
                    "ri_proc_exit_abstime",
                    "ri_child_user_time",
                    "ri_child_system_time",
                    "ri_child_pkg_idle_wkups",
                    "ri_child_interrupt_wkups",
                    "ri_child_pageins",
                    "ri_child_elapsed_abstime",
                    "ri_diskio_bytesread",
                    "ri_diskio_byteswritten",
                    "ri_cpu_time_qos_default",
                    "ri_cpu_time_qos_maintenance",
                    "ri_cpu_time_qos_background",
                    "ri_cpu_time_qos_utility",
                    "ri_cpu_time_qos_legacy",
                    "ri_cpu_time_qos_user_initiated",
                    "ri_cpu_time_qos_user_interactive",
                    "ri_billed_system_time",
                    "ri_serviced_system_time",
                    "ri_logical_writes",
                    "ri_lifetime_max_phys_footprint",
                    "ri_instructions",
                    "ri_cycles",
                    "ri_billed_energy",
                    "ri_serviced_energy",
                    "ri_interval_max_phys_footprint",
                    "ri_runnable_time",
                )
            ),
        ]

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pid_rusage = libproc.proc_pid_rusage
    except (OSError, AttributeError) as exc:
        raise ContractError(f"Darwin proc_pid_rusage is unavailable: {exc}") from exc
    proc_pid_rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    proc_pid_rusage.restype = ctypes.c_int

    def probe(pid: int) -> int:
        info = RusageInfoV4()
        result = proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info))
        if result != 0:
            error_number = ctypes.get_errno() or result
            raise ContractError(
                f"proc_pid_rusage(RUSAGE_INFO_V4) failed for pid {pid}: "
                f"errno={error_number}"
            )
        footprint = int(info.ri_phys_footprint)
        if footprint <= 0:
            raise ContractError("proc_pid_rusage reported a non-positive physical footprint")
        return footprint

    return probe


def darwin_phys_footprint_bytes(pid: int) -> int:
    global _DARWIN_PHYS_FOOTPRINT_PROBE
    if platform.system() != "Darwin":
        raise ContractError("physical-footprint sampling requires Darwin")
    if _DARWIN_PHYS_FOOTPRINT_PROBE is None:
        _DARWIN_PHYS_FOOTPRINT_PROBE = _load_darwin_phys_footprint_probe()
    return _DARWIN_PHYS_FOOTPRINT_PROBE(pid)


class DarwinProcessMemorySampler:
    """Sample this runner's current physical footprint on a fixed cadence."""

    def __init__(
        self,
        *,
        pid: int | None = None,
        interval_ms: int = MEMORY_SAMPLER_INTERVAL_MS,
        probe: Callable[[int], int] = darwin_phys_footprint_bytes,
    ) -> None:
        if interval_ms <= 0:
            raise ContractError("process-memory sampler interval must be positive")
        self.interval_ms = interval_ms
        self.pid = os.getpid() if pid is None else pid
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ContractError("process-memory sampler pid must be positive")
        self._probe = probe
        self._samples: list[int] = []
        self._failure: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        value = self._probe(self.pid)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError("process-memory sampler produced an invalid physical footprint")
        self._samples.append(value)

    def _sample_loop(self) -> None:
        interval_seconds = self.interval_ms / 1000.0
        deadline = time.monotonic() + interval_seconds
        try:
            while not self._stop.wait(max(0.0, deadline - time.monotonic())):
                self._sample()
                deadline += interval_seconds
        except BaseException as exc:
            self._failure = exc
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise ContractError("process-memory sampler may only be started once")
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gemma4-mlx-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def stop(self) -> ProcessMemoryMeasurement:
        if self._thread is None:
            raise ContractError("process-memory sampler was not started")
        self._stop.set()
        self._thread.join()
        if self._failure is not None:
            raise ContractError(f"process-memory sampler failed: {self._failure}") from self._failure
        self._sample()
        if len(self._samples) < 2:
            raise ContractError("process-memory sampler produced fewer than two samples")
        return ProcessMemoryMeasurement(max(self._samples), len(self._samples))


def _run_memory_tool(argv: Sequence[str], where: str) -> str:
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"could not execute {where}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise ContractError(f"{where} failed: {detail}")
    return result.stdout


def parse_vm_stat(output: str) -> tuple[int, dict[str, int]]:
    first_line, separator, remainder = output.partition("\n")
    if not separator:
        raise ContractError("vm_stat output is truncated")
    page_size_match = re.fullmatch(
        r"Mach Virtual Memory Statistics: \(page size of ([0-9]+) bytes\)",
        first_line.strip(),
    )
    if page_size_match is None:
        raise ContractError("vm_stat did not report an unambiguous page size")
    page_size = int(page_size_match.group(1))
    if page_size <= 0:
        raise ContractError("vm_stat reported a non-positive page size")
    wanted = {"Pageins", "Pageouts", "Swapins", "Swapouts"}
    counters: dict[str, int] = {}
    for raw_line in remainder.splitlines():
        match = re.fullmatch(r"([A-Za-z]+):\s+([0-9]+)\.", raw_line.strip())
        if match is None or match.group(1) not in wanted:
            continue
        name = match.group(1)
        if name in counters:
            raise ContractError(f"vm_stat repeated counter {name}")
        counters[name] = int(match.group(2))
    missing = sorted(wanted - set(counters))
    if missing:
        raise ContractError(f"vm_stat omitted required counters: {missing}")
    return page_size, counters


def parse_memory_pressure_available_percent(output: str) -> float:
    matches = re.findall(
        r"^System-wide memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%\s*$",
        output,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ContractError("memory_pressure -Q did not report one available-memory percentage")
    result = float(matches[0])
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ContractError("memory_pressure -Q reported an invalid available-memory percentage")
    return result


def capture_darwin_system_memory_snapshot(*, before_measured: bool) -> DarwinSystemMemorySnapshot:
    """Bracket measured steps tightly while collecting both Darwin sources."""
    if platform.system() != "Darwin":
        raise ContractError("system-memory snapshots require Darwin")
    if before_measured:
        pressure_output = _run_memory_tool(("/usr/bin/memory_pressure", "-Q"), "memory_pressure -Q")
        vm_output = _run_memory_tool(("/usr/bin/vm_stat",), "vm_stat")
    else:
        # Read the monotonic VM counters immediately after the final measured
        # device sync, before memory_pressure itself can perturb them.
        vm_output = _run_memory_tool(("/usr/bin/vm_stat",), "vm_stat")
        pressure_output = _run_memory_tool(("/usr/bin/memory_pressure", "-Q"), "memory_pressure -Q")
    page_size, counters = parse_vm_stat(vm_output)
    return DarwinSystemMemorySnapshot(
        page_size=page_size,
        pageins=counters["Pageins"],
        pageouts=counters["Pageouts"],
        swapins=counters["Swapins"],
        swapouts=counters["Swapouts"],
        pressure_available_percent=parse_memory_pressure_available_percent(pressure_output),
    )


def darwin_system_memory_deltas(
    before: DarwinSystemMemorySnapshot,
    after: DarwinSystemMemorySnapshot,
) -> dict[str, int | float]:
    if before.page_size != after.page_size:
        raise ContractError("vm_stat page size changed during measured optimizer steps")
    result: dict[str, int | float] = {}
    for field, output_name in (
        ("swapins", "swapins_bytes"),
        ("swapouts", "swapouts_bytes"),
        ("pageins", "pageins_bytes"),
        ("pageouts", "pageouts_bytes"),
    ):
        delta_pages = getattr(after, field) - getattr(before, field)
        if delta_pages < 0:
            raise ContractError(f"Darwin {field} counter regressed during measured optimizer steps")
        result[output_name] = delta_pages * before.page_size
    pressure_delta = after.pressure_available_percent - before.pressure_available_percent
    if not math.isfinite(pressure_delta) or not -100.0 <= pressure_delta <= 100.0:
        raise ContractError("available-memory pressure delta is invalid")
    result["pressure_available_percent_delta"] = pressure_delta
    return result


def enforce_system_memory_gates(
    lock: Mapping[str, Any],
    deltas: Mapping[str, int | float],
) -> None:
    memory_contract = lock["benchmark_contract"]["memory"]
    for field in ("swapins", "swapouts", "pageins", "pageouts"):
        metric = f"{field}_bytes"
        measured = deltas[metric]
        maximum = memory_contract[f"maximum_{field}_bytes"]
        if measured > maximum:
            raise ContractError(
                f"measured optimizer steps incurred {field} ({measured} bytes > {maximum})"
            )
    if (
        deltas["pressure_available_percent_delta"]
        < memory_contract["minimum_pressure_available_percent_delta"]
    ):
        raise ContractError("available-memory pressure degraded beyond the locked benchmark gate")


class JsonControlChannel:
    """Closed, newline-framed JSON protocol over one inherited Unix socket."""

    def __init__(
        self,
        channel: socket.socket,
        *,
        receive_timeout_seconds: float | None = None,
    ) -> None:
        if receive_timeout_seconds is not None and (
            not math.isfinite(receive_timeout_seconds) or receive_timeout_seconds <= 0
        ):
            raise ContractError("control channel timeout must be finite and positive")
        self._channel = channel
        self._receive_timeout_seconds = receive_timeout_seconds
        self._channel.settimeout(receive_timeout_seconds)
        self._buffer = bytearray()

    def send(self, payload: Mapping[str, Any]) -> None:
        try:
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as exc:
            raise ContractError(f"control message is not canonical JSON: {exc}") from exc
        if len(encoded) > MAX_CONTROL_MESSAGE_BYTES:
            raise ContractError("control message exceeds the fixed size limit")
        try:
            self._channel.sendall(encoded)
        except OSError as exc:
            raise ContractError(f"benchmark control channel write failed: {exc}") from exc

    def receive(self) -> dict[str, Any]:
        while b"\n" not in self._buffer:
            if len(self._buffer) >= MAX_CONTROL_MESSAGE_BYTES:
                raise ContractError("control message exceeds the fixed size limit")
            try:
                chunk = self._channel.recv(min(64 * 1024, MAX_CONTROL_MESSAGE_BYTES - len(self._buffer)))
            except socket.timeout as exc:
                timeout = self._receive_timeout_seconds
                description = "the configured deadline" if timeout is None else f"{timeout:g} seconds"
                raise ContractError(
                    f"benchmark control channel was idle for {description}"
                ) from exc
            except OSError as exc:
                raise ContractError(f"benchmark control channel read failed: {exc}") from exc
            if not chunk:
                raise ContractError("benchmark worker closed the control channel unexpectedly")
            self._buffer.extend(chunk)
        raw, separator, remainder = self._buffer.partition(b"\n")
        assert separator
        self._buffer = bytearray(remainder)
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_number,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ContractError(f"benchmark control message is invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ContractError("benchmark control message must be an object")
        if payload.get("schema_version") != CONTROL_PROTOCOL_VERSION:
            raise ContractError("benchmark control protocol version differs")
        return payload


class WorkerPhaseReporter:
    def __init__(self, channel: JsonControlChannel) -> None:
        self._channel = channel

    def barrier(self, phase: str) -> None:
        self._channel.send(
            {
                "schema_version": CONTROL_PROTOCOL_VERSION,
                "kind": "phase",
                "phase": phase,
            }
        )
        acknowledgement = self._channel.receive()
        expected = {
            "schema_version": CONTROL_PROTOCOL_VERSION,
            "kind": "ack",
            "phase": phase,
        }
        if acknowledgement != expected:
            raise ContractError(f"benchmark coordinator sent an invalid {phase} acknowledgement")


def _zig_hash_bytes(hasher: Any, value: bytes | str) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    hasher.update(struct.pack("<Q", len(encoded)))
    hasher.update(encoded)


def _zig_hash_file(hasher: Any, role: str, path: Path) -> None:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ContractError(f"could not stat provenance file {path}: {exc}") from exc
    _zig_hash_bytes(hasher, role)
    _zig_hash_bytes(hasher, path.name)
    hasher.update(struct.pack("<Q", stat.st_size))
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise ContractError(f"could not hash provenance file {path}: {exc}") from exc


def _checkpoint_paths(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        payload = load_json(index)
        if not isinstance(payload, Mapping):
            raise ContractError("model.safetensors.index.json must be an object")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ContractError("model.safetensors.index.json has no weight_map")
        relative_names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or not name for name in relative_names):
            raise ContractError("model.safetensors.index.json contains an invalid shard name")
        result: list[Path] = []
        for name in relative_names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError("model.safetensors.index.json contains an unsafe shard path")
            result.append((model_dir / relative).resolve())
        return result
    checkpoint = model_dir / "model.safetensors"
    if checkpoint.is_file():
        return [checkpoint.resolve()]
    raise ContractError("locked MLX model must expose model.safetensors or its shard index")


def zig_model_provenance(model_dir: Path) -> dict[str, str]:
    """Mirror the Zig Gemma4 base/tokenizer/chat provenance domains."""
    root = model_dir.expanduser().resolve()
    base = hashlib.sha256()
    _zig_hash_bytes(base, "gemma4_base_model/v1")
    config = root / "config.json"
    if config.is_file():
        _zig_hash_file(base, "config", config)
    else:
        _zig_hash_bytes(base, "config_absent")
    for checkpoint in sorted(_checkpoint_paths(root), key=lambda path: (path.name, str(path))):
        _zig_hash_file(base, "safetensors", checkpoint)
    base_digest = base.hexdigest()

    tokenizer = hashlib.sha256()
    _zig_hash_bytes(tokenizer, "gemma4_tokenizer/v1")
    tokenizer_count = 0
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        path = root / name
        if not path.is_file():
            continue
        _zig_hash_file(tokenizer, "tokenizer_asset", path)
        tokenizer_count += 1
    if tokenizer_count == 0:
        _zig_hash_bytes(tokenizer, base_digest)
    return {
        "base_model_sha256": base_digest,
        "tokenizer_sha256": tokenizer.hexdigest(),
        "chat_template_sha256": hashlib.sha256(PREPARED_CHAT_TEMPLATE_IDENTITY).hexdigest(),
    }


def require_prepared_model_binding(summary: Mapping[str, Any], model_dir: Path) -> None:
    actual = zig_model_provenance(model_dir)
    for field, digest in actual.items():
        if summary.get(field) != digest:
            raise ContractError(f"prepared {field} does not match the locked local model")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    raise ContractError(f"non-finite JSON number: {value}")


def _load_strict_json_file(path: Path, where: str) -> Any:
    try:
        encoded = path.read_text(encoding="utf-8")
        return json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"invalid {where}: {exc}") from exc


def _read_f32_tensor_digest_and_finiteness(
    checkpoint: Path,
    data_start: int,
    start: int,
    end: int,
    tensor_name: str,
) -> str:
    hasher = hashlib.sha256()
    remaining = end - start
    try:
        with checkpoint.open("rb") as source:
            source.seek(data_start + start)
            while remaining:
                chunk = source.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise ContractError(f"adapter tensor payload is truncated: {tensor_name}")
                if len(chunk) % 4:
                    raise ContractError(f"adapter tensor payload is not aligned F32: {tensor_name}")
                hasher.update(chunk)
                for (value,) in struct.iter_unpack("<f", chunk):
                    if not math.isfinite(value):
                        raise ContractError(f"adapter tensor contains non-finite data: {tensor_name}")
                remaining -= len(chunk)
    except OSError as exc:
        raise ContractError(f"could not read adapter tensor {tensor_name}: {exc}") from exc
    return "sha256:" + hasher.hexdigest()


def canonical_initial_adapter_file_sha256(
    checkpoint: Path,
    data_start: int,
    tensors: Mapping[tuple[str, str], AdapterTensor],
) -> str:
    """Stream the validator's canonical F32 semantic digest from Safetensors."""
    rows = sorted(tensors.items(), key=lambda item: item[0])
    if not rows:
        raise ContractError("initial adapter semantic digest requires tensors")
    hasher = hashlib.sha256(INITIAL_ADAPTER_DOMAIN)
    hasher.update(struct.pack("<Q", len(rows)))
    try:
        with checkpoint.open("rb") as source:
            for (module, role), descriptor in rows:
                for text_value in (module, role, "float32"):
                    encoded = text_value.encode("utf-8")
                    hasher.update(struct.pack("<Q", len(encoded)))
                    hasher.update(encoded)
                hasher.update(struct.pack("<Q", len(descriptor.shape)))
                for dimension in descriptor.shape:
                    hasher.update(struct.pack("<Q", dimension))
                value_bytes = descriptor.data_offset_end - descriptor.data_offset_start
                hasher.update(struct.pack("<Q", value_bytes))
                source.seek(data_start + descriptor.data_offset_start)
                remaining = value_bytes
                while remaining:
                    chunk = source.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ContractError(
                            f"initial adapter payload drifted while hashing {descriptor.source_name}"
                        )
                    hasher.update(chunk)
                    remaining -= len(chunk)
    except OSError as exc:
        raise ContractError(f"could not hash initial adapter semantics: {exc}") from exc
    return "sha256:" + hasher.hexdigest()


def inspect_initial_adapter(
    adapter_dir: Path,
    lock: Mapping[str, Any],
    model_key: str,
    target_preset: str,
    prepared_summary: Mapping[str, Any],
) -> AdapterArtifact:
    """Inspect F32 Safetensors without importing NumPy, Safetensors, or MLX."""
    root = adapter_dir.expanduser().absolute()
    if root.is_symlink():
        raise ContractError("initial adapter directory may not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise ContractError(f"initial adapter directory does not exist: {root}")
    config_path = root / "adapter_config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ContractError("initial adapter config must be a regular non-symlink file")
    raw_config = _load_strict_json_file(config_path, "initial adapter config")
    manifest_path = root / "antfly_finetune_manifest.json"
    if manifest_path.exists():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ContractError("initial adapter manifest must be a regular non-symlink file")
        _load_strict_json_file(manifest_path, "initial adapter manifest")
    semantics = read_adapter_config(root, target_preset=target_preset)
    gate = lock["performance_gate"]
    if semantics["r"] != gate["rank"] or float(semantics["lora_alpha"]) != float(gate["alpha"]):
        raise ContractError("initial adapter rank/alpha differ from the locked performance matrix")
    if semantics["target_preset"] != target_preset:
        raise ContractError("initial adapter target preset differs from the benchmark cell")
    provenance = semantics.get("provenance")
    if provenance is not None:
        for field in ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256"):
            if provenance.get(field) != prepared_summary.get(field):
                raise ContractError(f"initial adapter {field} differs from the prepared/model identity")

    if not isinstance(raw_config, Mapping) or raw_config.get("inference_mode") is not False:
        raise ContractError("initial adapter must explicitly set inference_mode=false")
    if raw_config.get("fan_in_fan_out", False) is not False:
        raise ContractError("initial adapter fan_in_fan_out must be false for Gemma4 linear layers")
    if raw_config.get("bias", "none") != "none":
        raise ContractError("initial adapter may not train or serialize non-LoRA bias parameters")
    checkpoint = root / "adapter_model.safetensors"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ContractError("initial adapter checkpoint must be a regular non-symlink file")
    try:
        with checkpoint.open("rb") as source:
            raw_length = source.read(8)
            if len(raw_length) != 8:
                raise ContractError("adapter Safetensors header length is truncated")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length == 0 or header_length > 128 * 1024 * 1024:
                raise ContractError("adapter Safetensors header length is invalid")
            raw_header = source.read(header_length)
            if len(raw_header) != header_length:
                raise ContractError("adapter Safetensors header is truncated")
    except OSError as exc:
        raise ContractError(f"could not read initial adapter checkpoint: {exc}") from exc
    try:
        header = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"invalid adapter Safetensors header: {exc}") from exc
    if not isinstance(header, Mapping):
        raise ContractError("adapter Safetensors header must be an object")

    data_start = 8 + header_length
    payload_size = checkpoint.stat().st_size - data_start
    tensors: dict[tuple[str, str], AdapterTensor] = {}
    ranges: list[tuple[int, int, str]] = []
    layouts: set[str] = set()
    for source_name, raw_descriptor in header.items():
        if source_name == "__metadata__":
            if (
                not isinstance(raw_descriptor, Mapping)
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in raw_descriptor.items())
            ):
                raise ContractError("Safetensors __metadata__ must map strings to strings")
            continue
        if not isinstance(raw_descriptor, Mapping) or set(raw_descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ContractError(f"adapter tensor descriptor is not closed: {source_name}")
        if raw_descriptor["dtype"] != "F32":
            raise ContractError(f"initial adapter tensor must be F32: {source_name}")
        shape = raw_descriptor["shape"]
        offsets = raw_descriptor["data_offsets"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ContractError(f"initial adapter tensor must have a positive rank-2 shape: {source_name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise ContractError(f"initial adapter tensor offsets are malformed: {source_name}")
        start, end = offsets
        expected_bytes = shape[0] * shape[1] * 4
        if start < 0 or end <= start or end - start != expected_bytes or end > payload_size:
            raise ContractError(f"initial adapter tensor byte range is invalid: {source_name}")
        identity = canonicalize_adapter_tensor_name(source_name)
        if identity in tensors:
            raise ContractError(f"duplicate canonical initial adapter tensor: {identity}")
        role = identity[1]
        if role == "lora_A" and shape[0] != gate["rank"]:
            raise ContractError(f"initial LoRA A rank mismatch: {source_name}")
        if role == "lora_B" and shape[1] != gate["rank"]:
            raise ContractError(f"initial LoRA B rank mismatch: {source_name}")
        layouts.add("antfly" if ".weight.lora_" in source_name else "stock-peft")
        tensors[identity] = AdapterTensor(
            source_name=source_name,
            shape=(shape[0], shape[1]),
            data_sha256=_read_f32_tensor_digest_and_finiteness(
                checkpoint, data_start, start, end, source_name
            ),
            data_offset_start=start,
            data_offset_end=end,
        )
        ranges.append((start, end, source_name))
    if not tensors or len(layouts) != 1:
        raise ContractError("initial adapter must contain one consistent tensor key layout")
    ordered_ranges = sorted(ranges)
    expected_start = 0
    for start, end, name in ordered_ranges:
        if start != expected_start:
            raise ContractError(f"initial adapter has a gap or overlap before tensor {name}")
        expected_start = end
    if expected_start != payload_size:
        raise ContractError("initial adapter has uncommitted trailing tensor payload bytes")
    manifest_backed = semantics["policy_source"] == "antfly-finetune-manifest/v2"
    if (manifest_backed and layouts != {"antfly"}) or (not manifest_backed and layouts != {"stock-peft"}):
        raise ContractError("initial adapter tensor key layout conflicts with its policy source")

    modules: dict[str, set[str]] = {}
    for module, role in tensors:
        modules.setdefault(module, set()).add(role)
    if any(roles != {"lora_A", "lora_B"} for roles in modules.values()):
        raise ContractError("initial adapter must contain one A/B pair for every target module")
    configured_targets = semantics["target_modules"]
    unmatched = [
        module
        for module in modules
        if not any(module == target or module.endswith("." + target) for target in configured_targets)
    ]
    unused = [
        target
        for target in configured_targets
        if not any(module == target or module.endswith("." + target) for module in modules)
    ]
    if unmatched or unused:
        raise ContractError(
            f"initial adapter config/tensor target mismatch (unmatched={unmatched}, unused={unused})"
        )
    validate_target_inventory(lock, model_key, target_preset, modules)
    semantics = dict(semantics)
    semantics["target_modules"] = sorted(modules)
    bound_files = [config_path, checkpoint]
    if manifest_path.exists():
        bound_files.append(manifest_path)
    semantic_sha256 = canonical_initial_adapter_file_sha256(checkpoint, data_start, tensors)
    return AdapterArtifact(
        directory=root,
        checkpoint=checkpoint,
        checkpoint_sha256=prefixed_sha256(checkpoint),
        config_sha256=prefixed_sha256(config_path),
        semantic_sha256=semantic_sha256,
        semantics=semantics,
        tensors=tensors,
        bound_files=tuple(bound_files),
    )


def build_workload(example: Mapping[str, Any], sequence_length: int, grad_accum: int) -> Workload:
    input_ids = tuple(example["input_ids"])
    labels = tuple(example["labels"])
    if len(input_ids) != sequence_length or len(labels) != sequence_length:
        raise ContractError(
            "the selected prepared row length must exactly equal --sequence-length; "
            "benchmark runners may not pad or truncate"
        )
    if any(token > 2**31 - 1 for token in input_ids) or any(
        label > 2**31 - 1 for label in labels if label != -100
    ):
        raise ContractError("prepared token IDs must fit MLX int32 without conversion")
    supervised_per_microstep = sum(label != -100 for label in labels[1:])
    if supervised_per_microstep <= 0:
        raise ContractError("the selected prepared row has no causal supervised labels")
    attention_mask = (1,) * sequence_length
    input_rows = [input_ids] * grad_accum
    label_rows = [labels] * grad_accum
    mask_rows = [attention_mask] * grad_accum
    return Workload(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        grad_accum=grad_accum,
        input_tokens=sequence_length * grad_accum,
        supervised_tokens=supervised_per_microstep * grad_accum,
        digest=benchmark_workload_sha256(input_rows, label_rows, mask_rows),
    )


def _load_surface_paths(lock: Mapping[str, Any], model_key: str, model_dir: Path) -> list[Path]:
    root = model_dir.expanduser().resolve()
    locked_files = lock["models"][model_key]["files"]
    paths = [root / relative for relative in locked_files]
    locked_checkpoints = {
        path for path in paths if path.name.startswith("model") and path.suffix == ".safetensors"
    }
    actual_checkpoints = {path.absolute() for path in root.glob("model*.safetensors") if path.is_file()}
    if actual_checkpoints != locked_checkpoints:
        raise ContractError("MLX model load surface contains an unlocked or missing model*.safetensors file")
    generation_config = root / "generation_config.json"
    if generation_config.exists() and "generation_config.json" not in locked_files:
        raise ContractError("MLX model load surface contains an unlocked generation_config.json")
    config = load_json(root / "config.json")
    if not isinstance(config, Mapping) or config.get("model_type") != "gemma4":
        raise ContractError("locked MLX benchmark model config must declare model_type=gemma4")
    custom_model = config.get("model_file")
    if custom_model is not None:
        if not isinstance(custom_model, str) or custom_model not in locked_files:
            raise ContractError("MLX model config references an unlocked custom model_file")
    for path in paths:
        if path.is_symlink():
            raise ContractError(f"locked model artifacts may not be symlinks: {path}")
    return paths


def capture_file_identities(paths: Sequence[Path]) -> tuple[FileIdentity, ...]:
    identities: list[FileIdentity] = []
    normalized: set[Path] = set()
    for path in paths:
        unresolved = path.expanduser().absolute()
        if unresolved.is_symlink():
            raise ContractError(f"bound input must not be a symlink: {unresolved}")
        normalized.add(unresolved.resolve())
    for raw_path in sorted(normalized, key=str):
        try:
            stat = raw_path.stat()
        except OSError as exc:
            raise ContractError(f"could not stat bound input {raw_path}: {exc}") from exc
        if not raw_path.is_file() or raw_path.is_symlink():
            raise ContractError(f"bound input must be a regular non-symlink file: {raw_path}")
        identities.append(
            FileIdentity(
                path=raw_path,
                device=stat.st_dev,
                inode=stat.st_ino,
                size=stat.st_size,
                modified_ns=stat.st_mtime_ns,
                changed_ns=stat.st_ctime_ns,
            )
        )
    return tuple(identities)


def require_files_unchanged(identities: Sequence[FileIdentity]) -> None:
    for expected in identities:
        current = capture_file_identities((expected.path,))[0]
        if current != expected:
            raise ContractError(f"bound benchmark input drifted during execution: {expected.path}")


def preflight(args: argparse.Namespace) -> Preflight:
    lock = load_lock(LOCK_PATH)
    gate = lock["performance_gate"]
    if args.sequence_length not in gate["primary_sequence_lengths"]:
        raise ContractError("sequence length is outside the locked performance matrix")
    if args.grad_accum not in gate["gradient_accumulation"]:
        raise ContractError("gradient accumulation is outside the locked performance matrix")
    if args.target_preset not in lock["target_presets"]:
        raise ContractError("target preset is outside the locked performance matrix")

    model = verify_model_directory(lock, args.model_key, args.model_dir)
    load_paths = _load_surface_paths(lock, args.model_key, args.model_dir)
    summary, prepared = load_prepared_example(args.prepared, args.example_index)
    verify_prepared_source_dataset(summary, args.source_dataset)
    require_prepared_model_binding(summary, args.model_dir)
    adapter = inspect_initial_adapter(
        args.adapter,
        lock,
        args.model_key,
        args.target_preset,
        summary,
    )
    workload = build_workload(prepared, args.sequence_length, args.grad_accum)
    bound_files = capture_file_identities(
        (
            *load_paths,
            *adapter.bound_files,
            args.prepared,
            args.source_dataset,
            args.mlx_build_attestation,
            LOCK_PATH,
            MLX_REQUIREMENTS_PATH,
            SCRIPT_PATH,
            Path(sys.executable).resolve(),
        )
    )
    return Preflight(
        lock=lock,
        model=model,
        prepared_summary=summary,
        prepared=prepared,
        adapter=adapter,
        workload=workload,
        bound_files=bound_files,
    )


def verify_mlx_native_build_before_import(
    args: argparse.Namespace,
    lock: Mapping[str, Any],
    mlx_checkout: Mapping[str, str],
    mlx_lm_checkout: Mapping[str, str],
) -> dict[str, Any]:
    """Admit the complete build bundle before any MLX native code executes."""
    from build_and_attest_gemma4_mlx import (
        ignored_untracked_files,
        verify_attestation_bundle,
    )

    source_root = Path(mlx_checkout["path"]).resolve(strict=True)
    mlx_lm_root = Path(mlx_lm_checkout["path"]).resolve(strict=True)
    ignored_mlx_lm = ignored_untracked_files(mlx_lm_root)
    if ignored_mlx_lm:
        raise ContractError(
            "pinned MLX-LM checkout contains ignored residue before import "
            f"(unexpected={[str(path) for path in ignored_mlx_lm]})"
        )
    return verify_attestation_bundle(args.mlx_build_attestation, source_root, lock)


def canonicalize_loaded_dyld_image_path(text: str, index: int) -> Path:
    """Canonicalize a dyld image while tolerating sealed shared-cache paths."""
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ContractError(f"dyld loaded-image path {index} is not absolute")
    absolute = Path(os.path.normpath(str(path)))
    try:
        return absolute.resolve(strict=True)
    except FileNotFoundError:
        # Recent macOS releases may report system frameworks by their logical
        # path even when the image exists only inside the sealed dyld shared
        # cache.  Keep that normalized identity; the required MLX/JACCL images
        # are ordinary files and are compared against strict resolved paths.
        return absolute


def loaded_dyld_image_paths() -> tuple[Path, ...]:
    """Return the native images actually loaded into this Darwin worker."""
    import ctypes

    dyld = ctypes.CDLL(None)
    image_count = dyld._dyld_image_count
    image_count.argtypes = ()
    image_count.restype = ctypes.c_uint32
    image_name = dyld._dyld_get_image_name
    image_name.argtypes = (ctypes.c_uint32,)
    image_name.restype = ctypes.c_char_p
    count = int(image_count())
    if count <= 0 or count > 100_000:
        raise ContractError(f"dyld exposed an invalid loaded-image count: {count}")
    paths: list[Path] = []
    for index in range(count):
        raw = image_name(index)
        if raw is None:
            raise ContractError(f"dyld omitted loaded-image path {index}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError(f"dyld loaded-image path {index} is not UTF-8") from exc
        paths.append(canonicalize_loaded_dyld_image_path(text, index))
    return tuple(paths)


def verify_mlx_native_runtime(
    args: argparse.Namespace,
    lock: Mapping[str, Any],
    mlx_checkout: Mapping[str, str],
    mx: Any,
    preverified_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the loaded extension, runtime dylibs, Metal library, and build attestation."""
    native_contract = lock["mlx_reference"]["native_runtime"]
    if getattr(mx, "__name__", None) != native_contract["extension_module"]:
        raise ContractError("loaded MLX core module differs from the locked native extension module")
    try:
        unresolved_core = Path(inspect.getfile(mx)).expanduser().absolute()
    except (TypeError, OSError) as exc:
        raise ContractError(f"could not locate the loaded MLX core extension: {exc}") from exc
    if unresolved_core.is_symlink():
        raise ContractError("loaded MLX core extension may not be a symbolic link")
    core_path = unresolved_core.resolve(strict=True)
    if not core_path.is_file() or not core_path.name.startswith("core.") or core_path.suffix != ".so":
        raise ContractError(f"loaded mlx.core is not the expected native extension: {core_path}")
    package_root = core_path.parent
    library_root = package_root / "lib"
    if library_root.is_symlink() or not library_root.is_dir():
        raise ContractError("loaded MLX package lib directory must be a regular non-symlink directory")
    expected_paths = {
        "jaccl-runtime-dylib": library_root / "libjaccl.dylib",
        "metal-library": library_root / "mlx.metallib",
        "python-extension": core_path,
        "runtime-dylib": library_root / "libmlx.dylib",
    }
    relevant_paths = {
        path.absolute()
        for path in package_root.glob("*.so")
        if path.is_file()
    }
    if library_root.is_dir():
        relevant_paths.update(
            path.absolute()
            for path in library_root.rglob("*")
            if path.is_file() and path.suffix in (".dylib", ".metallib")
        )
    expected_unresolved = {path.absolute() for path in expected_paths.values() if path.exists()}
    if relevant_paths != expected_unresolved or len(expected_unresolved) != len(expected_paths):
        missing = sorted(str(path) for path in expected_paths.values() if not path.is_file())
        extra = sorted(str(path) for path in relevant_paths - expected_unresolved)
        raise ContractError(
            f"MLX native runtime surface differs from the closed four-artifact inventory "
            f"(missing={missing}, extra={extra})"
        )

    artifacts: list[dict[str, Any]] = []
    bound_native_paths: list[Path] = []
    for role in native_contract["artifact_roles"]:
        unresolved = expected_paths[role]
        if unresolved.is_symlink() or not unresolved.is_file():
            raise ContractError(f"MLX native {role} must be a regular non-symlink file")
        path = unresolved.resolve(strict=True)
        info = path.stat()
        artifacts.append(
            {
                "role": role,
                "relative_path": path.relative_to(package_root).as_posix(),
                "size_bytes": info.st_size,
                "sha256": prefixed_sha256(path),
            }
        )
        bound_native_paths.append(path)
    inventory_sha256 = canonical_mlx_native_artifact_inventory_sha256(artifacts)
    loaded_images = set(loaded_dyld_image_paths())
    required_loaded_images = {
        core_path,
        expected_paths["jaccl-runtime-dylib"].resolve(strict=True),
        expected_paths["runtime-dylib"].resolve(strict=True),
    }
    if not required_loaded_images.issubset(loaded_images):
        missing_loaded = sorted(str(path) for path in required_loaded_images - loaded_images)
        raise ContractError(
            "loaded MLX Mach-O images differ from the attested extension/runtime dylib "
            f"(missing={missing_loaded})"
        )
    loaded_libmlx = sorted(path for path in loaded_images if path.name == "libmlx.dylib")
    if loaded_libmlx != [expected_paths["runtime-dylib"].resolve(strict=True)]:
        raise ContractError(
            "worker loaded an unbound libmlx.dylib image "
            f"(loaded={[str(path) for path in loaded_libmlx]})"
        )
    loaded_libjaccl = sorted(path for path in loaded_images if path.name == "libjaccl.dylib")
    if loaded_libjaccl != [expected_paths["jaccl-runtime-dylib"].resolve(strict=True)]:
        raise ContractError(
            "worker loaded an unbound libjaccl.dylib image "
            f"(loaded={[str(path) for path in loaded_libjaccl]})"
        )

    attestation_unresolved = args.mlx_build_attestation.expanduser().absolute()
    if attestation_unresolved.is_symlink() or not attestation_unresolved.is_file():
        raise ContractError("MLX build attestation must be a regular non-symlink file")
    attestation_path = attestation_unresolved.resolve(strict=True)
    source_root = Path(mlx_checkout["path"]).resolve(strict=True)
    if not attestation_path.is_relative_to(source_root):
        raise ContractError("MLX build attestation must reside inside the pinned MLX checkout")
    raw_attestation = _load_strict_json_file(attestation_path, "MLX native build attestation")
    if not isinstance(raw_attestation, Mapping):
        raise ContractError("MLX native build attestation must be an object")
    expected_attestation_keys = {
        "schema_version",
        "source_revision",
        "source_clean",
        "native_artifact_inventory_sha256",
        "build_command_sha256",
        "precision_policy_sha256",
    }
    if set(raw_attestation) != expected_attestation_keys:
        raise ContractError(
            "MLX native build attestation fields differ "
            f"(missing={sorted(expected_attestation_keys - set(raw_attestation))}, "
            f"extra={sorted(set(raw_attestation) - expected_attestation_keys)})"
        )
    expected_values = {
        "schema_version": native_contract["build_attestation_schema_version"],
        "source_revision": mlx_checkout["revision"],
        "source_clean": True,
        "native_artifact_inventory_sha256": inventory_sha256,
        "precision_policy_sha256": native_contract["precision_policy_sha256"],
    }
    for field, expected in expected_values.items():
        if raw_attestation[field] != expected:
            raise ContractError(f"MLX native build attestation {field} differs from runtime/lock")
    build_command_sha256 = raw_attestation["build_command_sha256"]
    if not isinstance(build_command_sha256, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", build_command_sha256
    ) is None:
        raise ContractError("MLX native build attestation build_command_sha256 is malformed")

    if preverified_bundle["native_artifact_inventory_sha256"] != inventory_sha256:
        raise ContractError("MLX native build receipt inventory differs from loaded runtime")
    if preverified_bundle["attestation_sha256"] != prefixed_sha256(attestation_path):
        raise ContractError("MLX native build attestation drifted after pre-import admission")
    bundle_bound_paths = tuple(Path(path) for path in preverified_bundle["bound_paths"])

    attestation = dict(raw_attestation)
    attestation.update(
        {
            "path": str(attestation_path),
            "sha256": prefixed_sha256(attestation_path),
        }
    )
    return {
        "native_artifact_inventory": {
            "schema_version": MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
            "sha256": inventory_sha256,
            "loaded_core_path": str(core_path),
            "artifacts": artifacts,
        },
        "build_attestation": attestation,
        "bound_file_identities": capture_file_identities(
            (*bound_native_paths, *bundle_bound_paths)
        ),
    }


def verify_mlx_environment(args: argparse.Namespace, lock: Mapping[str, Any]) -> dict[str, Any]:
    reference = lock["mlx_reference"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != reference["python"]:
        raise ContractError(
            f"MLX benchmark requires Python {reference['python']}, found {actual_python}"
        )
    mlx_checkout = verify_source_checkout(
        args.mlx_source,
        reference["source_revisions"]["mlx"],
        source_name="MLX",
    )
    mlx_lm_checkout = verify_source_checkout(
        args.mlx_lm_source,
        reference["source_revisions"]["mlx-lm"],
        source_name="MLX-LM",
    )
    verify_requirements_match_lock(lock, "mlx_reference", MLX_REQUIREMENTS_PATH)
    versions = verify_packages(lock, "mlx_reference")

    # Source roots precede site-packages.  The imported modules are checked
    # again below, so an incompatible source layout fails rather than falling
    # back to an ambient wheel.
    mlx_root = Path(mlx_checkout["path"])
    mlx_lm_root = Path(mlx_lm_checkout["path"])
    sys.dont_write_bytecode = True
    preverified_bundle = verify_mlx_native_build_before_import(
        args,
        lock,
        mlx_checkout,
        mlx_lm_checkout,
    )
    sys.path[:0] = [str(mlx_lm_root), str(mlx_root / "python"), str(mlx_root)]
    try:
        import mlx.core as mx
        import mlx.utils as mlx_utils
        import mlx_lm
    except ImportError as exc:
        raise ContractError(f"could not import the pinned MLX reference environment: {exc}") from exc
    verify_import_source(mlx_utils, mlx_root, source_name="MLX Python support")
    verify_import_source(mlx_lm, mlx_lm_root, source_name="MLX-LM")
    native_runtime = verify_mlx_native_runtime(
        args,
        lock,
        mlx_checkout,
        mx,
        preverified_bundle,
    )
    return {
        "mx": mx,
        "versions": versions,
        "mlx_checkout": mlx_checkout,
        "mlx_lm_checkout": mlx_lm_checkout,
        "mlx_utils_module": mlx_utils,
        "mlx_lm_module": mlx_lm,
        **native_runtime,
    }


def target_module_names(model: Any, lock: Mapping[str, Any], model_key: str, preset: str) -> list[str]:
    suffixes = tuple(lock["target_presets"][preset])
    selected: list[tuple[str, str]] = []
    for name, _module in model.named_modules():
        if not name:
            continue
        canonical = canonicalize_module_name(name)
        if any(canonical == suffix or canonical.endswith("." + suffix) for suffix in suffixes):
            selected.append((name, canonical))
    if not selected:
        raise ContractError("MLX-LM model exposed no modules for the locked target preset")
    canonical_names = [canonical for _name, canonical in selected]
    validate_target_inventory(lock, model_key, preset, canonical_names)
    return sorted(name for name, _canonical in selected)


def _tensor_inventory(
    flattened: Sequence[tuple[str, Any]],
    mx: Any,
    *,
    expected_dtype: Any,
    dtype_name: str,
    where: str,
) -> list[dict[str, Any]]:
    rows = list(flattened)
    if not rows:
        raise ContractError(f"{where} exposed no tensors")
    names = [name for name, _value in rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise ContractError(f"{where} exposed an invalid tensor name")
    if len(names) != len(set(names)):
        raise ContractError(f"{where} exposed duplicate tensor names")
    tensors: list[dict[str, Any]] = []
    for name, value in rows:
        if value.dtype != expected_dtype:
            raise ContractError(f"{where} tensor {name} is not {dtype_name}")
        try:
            shape = [int(dim) for dim in value.shape]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractError(f"{where} tensor {name} has no auditable shape") from exc
        if any(dim <= 0 for dim in shape):
            raise ContractError(f"{where} tensor {name} has a non-positive dimension")
        tensors.append({"name": name, "dtype": dtype_name, "shape": shape})
    return sorted(tensors, key=lambda tensor: tensor["name"])


def _inventory_observation(
    tensors: Sequence[Mapping[str, Any]],
    *,
    evidence_kind: str,
    dtype: str,
) -> dict[str, Any]:
    rows = [dict(tensor) for tensor in tensors]
    return {
        "evidence_kind": evidence_kind,
        "dtype": dtype,
        "tensor_count": len(rows),
        "inventory_sha256": canonical_tensor_inventory_sha256(rows),
        "tensors": rows,
    }


def require_exact_trainables(
    model: Any,
    target_names: Sequence[str],
    mx: Any,
) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    flattened = list(tree_flatten(model.trainable_parameters()))
    trainables = dict(flattened)
    expected = {
        f"{target}.{suffix}"
        for target in target_names
        for suffix in ("lora_a", "lora_b")
    }
    if set(trainables) != expected:
        missing = sorted(expected - set(trainables))
        unknown = sorted(set(trainables) - expected)
        raise ContractError(
            f"MLX-LM trainable inventory drift (missing={missing[:8]}, unknown={unknown[:8]})"
        )
    tensors = _tensor_inventory(
        flattened,
        mx,
        expected_dtype=mx.float32,
        dtype_name="float32",
        where="MLX-LM LoRA parameters",
    )
    return _inventory_observation(
        tensors,
        evidence_kind="materialized-trainable-parameter-inventory",
        dtype="float32",
    )


def require_f32_gradient_inventory(
    gradients: Any,
    expected_tensors: Sequence[Mapping[str, Any]],
    mx: Any,
    *,
    where: str,
    tree_flatten_fn: Callable[[Any], Sequence[tuple[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Prove the logical gradient tree has exact F32 LoRA names and shapes."""
    if tree_flatten_fn is None:
        from mlx.utils import tree_flatten as tree_flatten_fn

    tensors = _tensor_inventory(
        list(tree_flatten_fn(gradients)),
        mx,
        expected_dtype=mx.float32,
        dtype_name="float32",
        where=where,
    )
    expected = [dict(tensor) for tensor in expected_tensors]
    if tensors != expected:
        raise ContractError(f"{where} inventory differs from the exact LoRA trainables")
    return tensors


def require_f32_optimizer_state(
    model: Any,
    optimizer: Any,
    mx: Any,
    *,
    tree_flatten_fn: Callable[[Any], Sequence[tuple[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Audit AdamW's materialized moments after the cold mutating step."""
    if tree_flatten_fn is None:
        from mlx.utils import tree_flatten as tree_flatten_fn

    trainables = _tensor_inventory(
        list(tree_flatten_fn(model.trainable_parameters())),
        mx,
        expected_dtype=mx.float32,
        dtype_name="float32",
        where="MLX-LM LoRA parameters",
    )
    trainable_by_name = {tensor["name"]: tensor for tensor in trainables}
    state = list(tree_flatten_fn(optimizer.state))
    if not state:
        raise ContractError("MLX AdamW state was not materialized by the cold optimizer step")
    names = [name for name, _value in state]
    if len(names) != len(set(names)):
        raise ContractError("MLX AdamW state contains duplicate flattened names")
    values = dict(state)
    expected_moment_names = {
        f"{parameter_name}.{role}"
        for parameter_name in trainable_by_name
        for role in ("m", "v")
    }
    expected_state_names = {"step", "learning_rate", *expected_moment_names}
    if set(values) != expected_state_names:
        missing = sorted(expected_state_names - set(values))
        unknown = sorted(set(values) - expected_state_names)
        raise ContractError(
            f"MLX AdamW state inventory drift (missing={missing[:8]}, unknown={unknown[:8]})"
        )
    if values["step"].dtype != mx.uint64 or values["step"].size != 1:
        raise ContractError("MLX AdamW step state is not one uint64 scalar")
    if values["learning_rate"].dtype != mx.float32 or values["learning_rate"].size != 1:
        raise ContractError("MLX AdamW learning-rate state is not one float32 scalar")
    moments: list[dict[str, Any]] = []
    for name in sorted(expected_moment_names):
        parameter_name, role = name.rsplit(".", 1)
        value = values[name]
        if value.dtype != mx.float32:
            raise ContractError(f"MLX AdamW state {name} is not float32")
        try:
            shape = [int(dim) for dim in value.shape]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractError(f"MLX AdamW state {name} has no auditable shape") from exc
        if shape != trainable_by_name[parameter_name]["shape"]:
            raise ContractError(f"MLX AdamW state {name} shape differs from {parameter_name}")
        moments.append({
            "name": name,
            "parameter_name": parameter_name,
            "role": role,
            "dtype": "float32",
            "shape": shape,
        })
    return {
        "evidence_kind": "materialized-post-cold-adamw-moment-inventory",
        "dtype": "float32",
        "parameter_count": len(trainables),
        "moment_tensor_count": len(moments),
        "inventory_sha256": canonical_moment_inventory_sha256(moments),
        "moments": moments,
    }


def load_exact_initial_adapter(
    model: Any,
    target_names: Sequence[str],
    artifact: AdapterArtifact,
    mx: Any,
) -> None:
    """Translate PEFT/Antfly [rank,in]/[out,rank] tensors to MLX exactly."""
    from mlx.utils import tree_flatten, tree_unflatten

    try:
        loaded = mx.load(str(artifact.checkpoint))
    except Exception as exc:
        raise ContractError(f"MLX could not load the initial adapter Safetensors: {exc}") from exc
    if set(loaded) != {tensor.source_name for tensor in artifact.tensors.values()}:
        raise ContractError("MLX-loaded initial adapter tensor keys differ from the inspected header")
    mlx_targets = {canonicalize_module_name(name): name for name in target_names}
    if len(mlx_targets) != len(target_names) or set(mlx_targets) != {
        module for module, _role in artifact.tensors
    }:
        raise ContractError("MLX target modules do not exactly match the canonical initial adapter modules")

    updates: list[tuple[str, Any]] = []
    expected_values: dict[str, Any] = {}
    finite_checks: list[tuple[str, Any]] = []
    for identity, descriptor in artifact.tensors.items():
        module, role = identity
        source = loaded[descriptor.source_name]
        if source.dtype != mx.float32 or tuple(source.shape) != descriptor.shape:
            raise ContractError(f"MLX changed initial adapter dtype/shape while loading {descriptor.source_name}")
        finite_checks.append((descriptor.source_name, mx.isfinite(source).all()))
        mlx_suffix = "lora_a" if role == "lora_A" else "lora_b"
        destination = f"{mlx_targets[module]}.{mlx_suffix}"
        translated = source.T
        updates.append((destination, translated))
        expected_values[destination] = translated
    mx.eval(*(check for _name, check in finite_checks))
    nonfinite = [name for name, check in finite_checks if not bool(check.item())]
    if nonfinite:
        raise ContractError(f"MLX-loaded initial adapter contains non-finite tensors: {nonfinite[:8]}")

    try:
        model.update(tree_unflatten(updates), strict=True)
    except Exception as exc:
        raise ContractError(f"could not install translated initial adapter tensors into MLX: {exc}") from exc
    mx.eval(model.trainable_parameters())
    actual = dict(tree_flatten(model.trainable_parameters()))
    equality_checks = [
        (name, mx.array_equal(actual[name], expected))
        for name, expected in expected_values.items()
        if name in actual
    ]
    if len(equality_checks) != len(expected_values):
        raise ContractError("MLX trainable inventory is missing a translated initial adapter tensor")
    mx.eval(*(check for _name, check in equality_checks))
    changed = [name for name, check in equality_checks if not bool(check.item())]
    if changed:
        raise ContractError(f"MLX initial adapter value translation was not exact: {changed[:8]}")


def require_bf16_base_model(model: Any, mx: Any) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    parameters = list(tree_flatten(model.parameters()))
    tensors = _tensor_inventory(
        parameters,
        mx,
        expected_dtype=mx.bfloat16,
        dtype_name="bfloat16",
        where="MLX-LM base model parameters",
    )
    return _inventory_observation(
        tensors,
        evidence_kind="materialized-parameter-inventory",
        dtype="bfloat16",
    )


def make_optimizer_step(
    model: Any,
    optimizer: Any,
    mx: Any,
    nn: Any,
    optim: Any,
    input_ids: Any,
    labels: Any,
    grad_accum: int,
    max_grad_norm: float,
    precision_recorder: PrecisionEvidenceRecorder,
) -> Callable[[], Any]:
    from mlx.utils import tree_flatten, tree_map

    expected_gradient_tensors = _tensor_inventory(
        list(tree_flatten(model.trainable_parameters())),
        mx,
        expected_dtype=mx.float32,
        dtype_name="float32",
        where="MLX-LM LoRA parameters",
    )

    def loss_fn(current_model: Any, tokens: Any, target_labels: Any) -> Any:
        # Process the complete prepared row, matching Antfly's input-position
        # total, then apply the explicit one-token causal label shift.
        logits = current_model(tokens)
        shifted_logits = logits[:, :-1, :].astype(mx.float32)
        shifted_labels = target_labels[:, 1:]
        mask = shifted_labels != -100
        safe_labels = mx.where(mask, shifted_labels, mx.zeros_like(shifted_labels))
        losses = nn.losses.cross_entropy(shifted_logits, safe_labels)
        supervised = mask.sum()
        reduction_input = (losses * mask).astype(mx.float32)
        if reduction_input.dtype != mx.float32:
            raise ContractError("MLX supervised loss reduction input is not float32")
        loss = reduction_input.sum() / supervised
        if loss.dtype != mx.float32:
            raise ContractError("MLX supervised loss tensor is not float32")
        precision_recorder.record(
            "loss",
            {
                "evidence_kind": "evaluated-training-loss-graph",
                "loss_tensor_dtype": "float32",
                "reduction_input_dtype": "float32",
            },
        )
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    def optimizer_step(tokens: Any, target_labels: Any) -> Any:
        accumulated = None
        loss_total = None
        for _ in range(grad_accum):
            loss, gradients = loss_and_grad(model, tokens, target_labels)
            raw_tensors = require_f32_gradient_inventory(
                gradients, expected_gradient_tensors, mx, where="MLX raw gradients",
            )
            precision_recorder.record_gradient("raw", raw_tensors)
            loss_total = loss if loss_total is None else loss_total + loss
            accumulated = (
                gradients
                if accumulated is None
                else tree_map(lambda left, right: left + right, accumulated, gradients)
            )
        averaged = tree_map(lambda gradient: gradient / grad_accum, accumulated)
        accumulated_tensors = require_f32_gradient_inventory(
            averaged, expected_gradient_tensors, mx, where="MLX accumulated gradients",
        )
        precision_recorder.record_gradient("accumulated", accumulated_tensors)
        clipped, _raw_norm = optim.clip_grad_norm(averaged, max_grad_norm)
        clipped_tensors = require_f32_gradient_inventory(
            clipped, expected_gradient_tensors, mx, where="MLX clipped gradients",
        )
        precision_recorder.record_gradient("clipped", clipped_tensors)
        optimizer.update(model, clipped)
        return loss_total / grad_accum

    compiled = mx.compile(optimizer_step, inputs=state, outputs=state)

    def execute() -> Any:
        loss = compiled(input_ids, labels)
        mx.eval(loss, model.state, optimizer.state)
        return loss

    return execute


def synchronized_step_seconds(
    synchronize: Callable[[], None],
    execute: Callable[[], Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[float, Any]:
    synchronize()
    started = clock()
    result = execute()
    synchronize()
    duration = clock() - started
    if not math.isfinite(duration) or duration <= 0:
        raise ContractError("synchronized optimizer-step duration must be finite and positive")
    return duration, result


def measure_steps(
    synchronize: Callable[[], None],
    execute: Callable[[], Any],
    warmup_steps: int,
    measured_steps: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
    result_is_finite: Callable[[Any], bool] | None = None,
    after_cold: Callable[[Any], None] | None = None,
    before_measured: Callable[[], None] | None = None,
    after_measured: Callable[[], None] | None = None,
) -> StepMeasurements:
    durations: list[float] = []

    def run_one() -> tuple[float, Any]:
        duration, result = synchronized_step_seconds(synchronize, execute, clock=clock)
        if result_is_finite is not None and not result_is_finite(result):
            raise ContractError("MLX-LM optimizer step produced a non-finite loss")
        return duration, result

    # MLX compiles lazily.  The compile observation therefore includes the
    # first complete optimizer step; the separately reported first step is the
    # first execution of the compiled graph.
    cold_compile_and_step_seconds, cold_result = run_one()
    if after_cold is not None:
        after_cold(cold_result)
    first_steady_step_seconds, _first_result = run_one()
    for _ in range(warmup_steps):
        run_one()
    if before_measured is not None:
        before_measured()
    try:
        for _ in range(measured_steps):
            duration, _result = run_one()
            durations.append(duration)
    finally:
        if after_measured is not None:
            after_measured()
    return StepMeasurements(
        cold_compile_and_step_seconds,
        first_steady_step_seconds,
        tuple(durations),
    )


def expected_unused_shared_kv_parameter_names(config: Mapping[str, Any]) -> set[str]:
    """Return the exact legacy checkpoint tensors unused by Gemma4 KV sharing."""
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise ContractError("locked Gemma4 config omitted text_config")
    num_layers = text_config.get("num_hidden_layers")
    num_shared = text_config.get("num_kv_shared_layers")
    layer_types = text_config.get("layer_types")
    attention_k_eq_v = text_config.get("attention_k_eq_v", False)
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
        or isinstance(num_shared, bool)
        or not isinstance(num_shared, int)
        or num_shared < 0
        or num_shared >= num_layers
        or not isinstance(layer_types, list)
        or len(layer_types) != num_layers
        or any(layer_type not in ("sliding_attention", "full_attention") for layer_type in layer_types)
        or not isinstance(attention_k_eq_v, bool)
    ):
        raise ContractError("locked Gemma4 KV-sharing config is malformed")
    first_shared = num_layers - num_shared
    result: set[str] = set()
    for layer_index in range(first_shared, num_layers):
        prefix = f"language_model.model.layers.{layer_index}.self_attn"
        result.add(f"{prefix}.k_norm.weight")
        result.add(f"{prefix}.k_proj.weight")
        if not (attention_k_eq_v and layer_types[layer_index] == "full_attention"):
            result.add(f"{prefix}.v_proj.weight")
    return result


def validate_mlx_gemma4_checkpoint_coverage(
    config: Mapping[str, Any],
    model_parameter_names: set[str],
    checkpoint_parameter_names: set[str],
) -> set[str]:
    """Admit only the closed legacy shared-KV surplus; never admit missing weights."""
    if not model_parameter_names or not checkpoint_parameter_names:
        raise ContractError("MLX Gemma4 model/checkpoint parameter inventory is empty")
    missing = model_parameter_names - checkpoint_parameter_names
    extras = checkpoint_parameter_names - model_parameter_names
    allowed_extras = expected_unused_shared_kv_parameter_names(config)
    if missing:
        raise ContractError(
            "locked Gemma4 checkpoint is missing MLX model parameters "
            f"(missing={sorted(missing)[:16]})"
        )
    unexpected = extras - allowed_extras
    if unexpected:
        raise ContractError(
            "locked Gemma4 checkpoint surplus differs from the exact unused shared-KV inventory "
            f"(unexpected={sorted(unexpected)[:16]})"
        )
    return extras


def load_locked_mlx_gemma4(
    model_dir: Path,
    mx: Any,
    *,
    load_config_fn: Callable[[Path], dict[str, Any]],
    get_model_classes_fn: Callable[..., tuple[Any, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Load the unquantized text model with exact coverage, dtype, and shape checks."""
    from mlx.utils import tree_flatten

    config = load_config_fn(model_dir)
    if config.get("model_type") != "gemma4" or "quantization" in config or "quantization_config" in config:
        raise ContractError("MLX-LM benchmark requires an unquantized Gemma4 model")
    try:
        model_class, model_args_class = get_model_classes_fn(config=config)
        model = model_class(model_args_class.from_dict(config))
    except Exception as exc:
        raise ContractError(f"pinned MLX-LM could not instantiate locked Gemma4: {exc}") from exc

    weights: dict[str, Any] = {}
    for checkpoint in _checkpoint_paths(model_dir):
        try:
            shard = mx.load(str(checkpoint))
        except Exception as exc:
            raise ContractError(f"MLX could not load locked Gemma4 checkpoint {checkpoint.name}: {exc}") from exc
        duplicates = set(weights).intersection(shard)
        if duplicates:
            raise ContractError(
                f"locked Gemma4 checkpoint shards duplicate parameters: {sorted(duplicates)[:16]}"
            )
        weights.update(shard)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    if not isinstance(weights, dict) or any(not isinstance(name, str) for name in weights):
        raise ContractError("pinned MLX-LM Gemma4 sanitizer returned an invalid weight inventory")

    current = dict(tree_flatten(model.parameters()))
    ignored = validate_mlx_gemma4_checkpoint_coverage(config, set(current), set(weights))
    admitted = [(name, weights[name]) for name in current]
    wrong_dtype = [name for name, value in admitted if value.dtype != mx.bfloat16]
    if wrong_dtype:
        raise ContractError(
            f"locked Gemma4 checkpoint parameters are not BF16: {wrong_dtype[:16]}"
        )
    try:
        # Filtering is safe only after the exact ignored set has been admitted.
        # strict=True still checks every used name and shape.
        model.load_weights(admitted, strict=True)
    except Exception as exc:
        raise ContractError(f"locked Gemma4 checkpoint parameter shapes differ: {exc}") from exc
    if set(weights) - {name for name, _value in admitted} != ignored:
        raise ContractError("locked Gemma4 ignored parameter inventory drifted during load")
    return model, config


def run_mlx(
    args: argparse.Namespace,
    preflight_result: Preflight,
    precision_recorder: PrecisionEvidenceRecorder,
    *,
    before_measured: Callable[[], None] | None = None,
    after_measured: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = verify_mlx_environment(args, preflight_result.lock)
    mx = environment["mx"]
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ContractError("MLX-LM release benchmarks require Darwin arm64")
    if not mx.metal.is_available():
        raise ContractError("MLX Metal is unavailable")
    mx.set_default_device(mx.gpu)

    try:
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import _get_classes, load_config
    except ImportError as exc:
        raise ContractError(f"pinned MLX-LM training API is unavailable: {exc}") from exc

    benchmark_contract = preflight_result.lock["benchmark_contract"]
    optimizer_contract = benchmark_contract["optimizer"]
    gate = preflight_result.lock["performance_gate"]
    mx.random.seed(benchmark_contract["determinism"]["seed"])
    load_started = time.perf_counter()
    try:
        model, config = load_locked_mlx_gemma4(
            Path(preflight_result.model["directory"]),
            mx,
            load_config_fn=load_config,
            get_model_classes_fn=_get_classes,
        )
    except Exception as exc:
        raise ContractError(f"pinned MLX-LM could not load the locked local model: {exc}") from exc
    mx.eval(model.parameters())
    mx.synchronize()
    precision_recorder.record("base_model_storage", require_bf16_base_model(model, mx))

    model.freeze()
    targets = target_module_names(
        model,
        preflight_result.lock,
        args.model_key,
        args.target_preset,
    )
    linear_to_lora_layers(
        model,
        -1,
        {
            "rank": gate["rank"],
            "scale": float(gate["alpha"]) / gate["rank"],
            "dropout": 0.0,
            "keys": set(targets),
        },
    )
    require_exact_trainables(model, targets, mx)
    load_exact_initial_adapter(model, targets, preflight_result.adapter, mx)
    precision_recorder.record(
        "lora_parameter_storage",
        require_exact_trainables(model, targets, mx),
    )
    model.train()
    mx.eval(model.state)
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started
    if not math.isfinite(load_seconds) or load_seconds < 0:
        raise ContractError("MLX-LM load duration is invalid")

    optimizer = optim.AdamW(
        learning_rate=optimizer_contract["learning_rate"],
        betas=tuple(optimizer_contract["betas"]),
        eps=optimizer_contract["eps"],
        weight_decay=optimizer_contract["weight_decay"],
        bias_correction=optimizer_contract["bias_correction"],
    )
    input_ids = mx.array([preflight_result.workload.input_ids], dtype=mx.int32)
    labels = mx.array([preflight_result.workload.labels], dtype=mx.int32)
    execute = make_optimizer_step(
        model,
        optimizer,
        mx,
        nn,
        optim,
        input_ids,
        labels,
        preflight_result.workload.grad_accum,
        optimizer_contract["gradient_clip_max_norm"],
        precision_recorder,
    )
    reference = preflight_result.lock["mlx_reference"]

    def result_is_finite(loss: Any) -> bool:
        if loss.dtype != mx.float32:
            raise ContractError("MLX optimizer-step loss result is not float32")
        return math.isfinite(float(loss.item()))

    measurements = measure_steps(
        mx.synchronize,
        execute,
        reference["warmup_steps"],
        reference["measured_steps"],
        result_is_finite=result_is_finite,
        after_cold=lambda _loss: precision_recorder.record(
            "optimizer_moment_storage",
            require_f32_optimizer_state(model, optimizer, mx),
        ),
        before_measured=before_measured,
        after_measured=after_measured,
    )
    mx.synchronize()
    peak_memory = int(mx.get_peak_memory())
    if peak_memory <= 0:
        raise ContractError("MLX did not report a positive allocator peak-memory value")
    return environment, {
        "load_seconds": load_seconds,
        "cold_compile_and_step_seconds": measurements.cold_compile_and_step_seconds,
        "first_steady_step_seconds": measurements.first_steady_step_seconds,
        "step_seconds": list(measurements.step_seconds),
        "input_tokens": preflight_result.workload.input_tokens,
        "supervised_tokens": preflight_result.workload.supervised_tokens,
        "memory": {
            "framework_allocator_peak_bytes": peak_memory,
            "framework_allocator_peak_source": "mlx-metal-get-peak-memory",
        },
    }


def command_digest(argv: Sequence[str]) -> str:
    descriptor = {
        "schema_version": "antfly_gemma4_mlx_benchmark_command/v2",
        "argv": [str(Path(sys.executable).resolve()), str(SCRIPT_PATH), *argv],
        "process_topology": "coordinator-samples-one-fresh-child-worker",
        "control_transport": "inherited-bidirectional-unix-socket",
        "control_protocol": CONTROL_PROTOCOL_VERSION,
        "control_idle_timeout_seconds": CONTROL_IDLE_TIMEOUT_SECONDS,
        "phase_barriers": [
            "worker-start-signal-parent-pre-snapshot-ack",
            "worker-final-sync-end-signal-parent-post-snapshot-ack",
            "parent-result-ack-before-worker-exit",
        ],
        "process_memory_sampler": "parent-proc-pid-rusage-v4-10ms",
        "mlx_wired_limit_policy": "default-no-runner-mutation",
        "offline_environment": OFFLINE_ENVIRONMENT,
        "runner_sha256": prefixed_sha256(SCRIPT_PATH),
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def runner_source_attestation() -> dict[str, Any]:
    """Bind this runner and its complete local dependency closure to one clean commit."""
    source = attest_benchmark_producer_source(
        SCRIPT_PATH,
        expected_entrypoint=MLX_RUNNER_RELATIVE_PATH,
    )
    if source["schema_version"] != BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION:
        raise ContractError("MLX benchmark producer source schema drifted")
    return source


def diagnostic_runner_source() -> dict[str, Any]:
    """Content-bind a live producer checkout without claiming release cleanliness."""

    def git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
        try:
            return subprocess.run(
                ("git", "-C", str(SCRIPT_PATH.parent), *arguments),
                check=True,
                capture_output=True,
                text=text,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", b"" if not text else "") or str(exc)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise ContractError(
                f"could not bind diagnostic benchmark producer source: {str(detail).strip()}"
            ) from exc

    root = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve(strict=True)
    try:
        relative_entrypoint = SCRIPT_PATH.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("diagnostic benchmark producer is outside its Git checkout") from exc
    if relative_entrypoint != MLX_RUNNER_RELATIVE_PATH:
        raise ContractError("diagnostic benchmark producer entrypoint differs")
    revision = git("rev-parse", "HEAD").stdout.strip()
    source_tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or re.fullmatch(
        r"[0-9a-f]{40}", source_tree
    ) is None:
        raise ContractError("diagnostic benchmark producer revision/tree is malformed")
    status = git("status", "--porcelain=v1", "--untracked-files=all", text=False).stdout
    files: list[dict[str, str]] = []
    for relative_path in BENCHMARK_PRODUCER_RELATIVE_PATHS:
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ContractError(
                f"diagnostic benchmark producer source must be a regular file: {relative_path}"
            )
        files.append(
            {
                "relative_path": relative_path,
                "source_sha256": prefixed_sha256(path),
            }
        )
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
    manifest_sha256 = "sha256:" + hashlib.sha256(
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
    return {
        "schema_version": DIAGNOSTIC_SOURCE_SCHEMA_VERSION,
        **manifest_input,
        "dirty_entry_count": len(status.splitlines()),
        "source_sha256": entrypoint_sha256,
        "manifest_sha256": manifest_sha256,
    }


def build_precision_evidence(
    sample: Mapping[str, Any],
    observations: Mapping[str, Any],
    runner: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one closed precision artifact from audited runtime observations."""
    implementation = sample["implementation"]
    mlx = implementation["mlx"]
    attestation = mlx["build_attestation"]
    precision = lock["benchmark_contract"]["precision"]
    evidence = {
        "schema_version": PRECISION_EVIDENCE_SCHEMA_VERSION,
        "framework": "mlx-lm",
        "oracle_lock_sha256": sample["oracle_lock_sha256"],
        "sample_binding": {
            "campaign_id": sample["campaign_id"],
            "run_id": sample["run_id"],
            "repetition": sample["repetition"],
            "sequence_index": sample["sequence_index"],
            "command_sha256": implementation["command_sha256"],
            "semantic_contract_sha256": sample["semantic_contract"]["sha256"],
            "sample_payload_sha256": canonical_precision_sample_binding_sha256(sample),
        },
        "runner": dict(runner),
        "native_runtime": {
            "mlx_source_revision": mlx["source_revision"],
            "mlx_lm_source_revision": implementation["mlx_lm"]["source_revision"],
            "native_artifact_inventory_sha256": mlx["native_artifact_inventory"]["sha256"],
            "build_attestation_sha256": attestation["sha256"],
            "build_command_sha256": attestation["build_command_sha256"],
            "precision_policy_sha256": attestation["precision_policy_sha256"],
        },
        "verified": dict(precision["verified"]),
        "not_asserted": list(precision["not_asserted"]),
        "comparison_policy": precision["comparison_policy"],
        "observations": dict(observations),
    }
    return validate_precision_evidence_payload(evidence, sample, lock)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def precision_evidence_output_path(sample_output: Path) -> Path:
    requested = sample_output.expanduser().absolute()
    return requested.with_name(requested.name + ".precision.json")


def precision_evidence_reference(
    sample_output: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_path = precision_evidence_output_path(sample_output)
    return {
        "schema_version": PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION,
        "relative_path": evidence_path.name,
        "artifact_sha256": "sha256:" + hashlib.sha256(canonical_json_bytes(evidence)).hexdigest(),
    }


def build_sample_payload(
    args: argparse.Namespace,
    preflight_result: Preflight,
    environment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    hardware: Mapping[str, Any],
    argv: Sequence[str],
    producer_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Central sample assembly seam; execution code is schema-independent."""
    required_hardware = ("platform", "machine", "chip", "memory_bytes", "os_version", "os_build")
    missing = [field for field in required_hardware if not hardware.get(field)]
    if missing:
        raise ContractError(f"could not establish benchmark hardware identity: {missing}")
    model_lock = preflight_result.lock["models"][args.model_key]
    prepared = preflight_result.prepared
    canonical_modules = sorted({module for module, _role in preflight_result.adapter.tensors})
    validate_target_inventory(
        preflight_result.lock,
        args.model_key,
        args.target_preset,
        canonical_modules,
    )
    case = {
        "model_key": args.model_key,
        "revision": model_lock["revision"],
        "local_artifact_sha256": preflight_result.model["local_artifact_sha256"],
        "target_preset": args.target_preset,
        "rank": preflight_result.lock["performance_gate"]["rank"],
        "alpha": float(preflight_result.lock["performance_gate"]["alpha"]),
        "sequence_length": args.sequence_length,
        "grad_accum": args.grad_accum,
        "microbatch": preflight_result.lock["performance_gate"]["microbatch"],
        "prepared": {
            "schema_version": prepared["schema_version"],
            "artifact_sha256": prepared["artifact_sha256"],
            "example_index": prepared["example_index"],
            "source_dataset_sha256": prepared["source_dataset_sha256"],
            "source_record_sha256": prepared["source_record_sha256"],
            "rendered_chat_sha256": prepared["rendered_chat_sha256"],
            "workload_sha256": preflight_result.workload.digest,
        },
        "initial_adapter": {
            "schema_version": "antfly_gemma4_initial_adapter_semantics/v1",
            "semantic_sha256": preflight_result.adapter.semantic_sha256,
            "tensor_count": len(preflight_result.adapter.tensors),
            "tensor_dtype": "float32",
        },
        "target_inventory": {
            "schema_version": "antfly_gemma4_target_inventory/v1",
            "sha256": canonical_target_inventory_sha256(canonical_modules),
            "module_count": len(canonical_modules),
            "canonical_modules": canonical_modules,
        },
    }
    executable = Path(sys.executable).resolve()
    payload = {
        "schema_version": BENCHMARK_SAMPLE_SCHEMA_VERSION,
        "framework": "mlx-lm",
        "oracle_lock_sha256": lock_digest(LOCK_PATH),
        "campaign_id": args.campaign_id,
        "run_id": args.run_id,
        "repetition": args.repetition,
        "sequence_index": args.sequence_index,
        "implementation": {
            "command_sha256": command_digest(argv),
            "producer_source": dict(producer_source),
            "mlx": {
                "version": environment["versions"]["mlx"],
                "source_revision": environment["mlx_checkout"]["revision"],
                "source_clean": True,
                "native_artifact_inventory": environment["native_artifact_inventory"],
                "build_attestation": environment["build_attestation"],
            },
            "mlx_lm": {
                "version": environment["versions"]["mlx-lm"],
                "source_revision": environment["mlx_lm_checkout"]["revision"],
                "source_clean": True,
            },
            "python": {
                "version": platform.python_version(),
                "executable": str(executable),
                "executable_sha256": prefixed_sha256(executable),
            },
        },
        "process": {"pid": os.getpid(), "started_unix_ns": PROCESS_STARTED_UNIX_NS},
        "hardware": {
            "platform": hardware["platform"],
            "machine": hardware["machine"],
            "chip": hardware["chip"],
            "memory_bytes": hardware["memory_bytes"],
            "os_version": hardware["os_version"],
            "os_build": hardware["os_build"],
            # Apple Silicon exposes one integrated Metal device.  Using the
            # sysctl chip brand is stable and reproducible by the Zig runner.
            "metal_device": hardware["chip"],
        },
        "case": case,
        "semantic_contract": expected_semantic_contract(preflight_result.lock, case),
        "protocol": {
            "fresh_process": True,
            "cold_optimizer_steps": 1,
            "cold_step_mutates_optimizer_state": True,
            "first_steady_steps": 1,
            "warmup_steps": preflight_result.lock["mlx_reference"]["warmup_steps"],
            "measured_steps": preflight_result.lock["mlx_reference"]["measured_steps"],
            "explicit_device_sync": True,
            "sync_point": SYNC_POINT,
            "timed_unit": TIMED_UNIT,
        },
        "metrics": dict(metrics),
    }
    return payload


def build_diagnostic_payload(
    args: argparse.Namespace,
    preflight_result: Preflight,
    environment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    hardware: Mapping[str, Any],
    argv: Sequence[str],
    producer_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a conspicuously non-release payload around the real execution result."""
    payload = build_sample_payload(
        args,
        preflight_result,
        environment,
        metrics,
        hardware,
        argv,
        producer_source,
    )
    payload["schema_version"] = DIAGNOSTIC_SAMPLE_SCHEMA_VERSION
    payload["diagnostic"] = {
        "release_eligible": False,
        "release_gates_enforced": False,
        "release_blockers": [DIAGNOSTIC_RELEASE_BLOCKER],
        "publication_contract": "never-valid-as-antfly_gemma4_lora_benchmark_sample/v5",
    }
    return payload


def validate_diagnostic_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Reject diagnostics that could be mistaken for release benchmark evidence."""
    payload = dict(raw)
    expected_keys = {
        "schema_version",
        "framework",
        "oracle_lock_sha256",
        "campaign_id",
        "run_id",
        "repetition",
        "sequence_index",
        "diagnostic",
        "implementation",
        "process",
        "hardware",
        "case",
        "semantic_contract",
        "protocol",
        "metrics",
        "precision_observations",
    }
    if set(payload) != expected_keys:
        raise ContractError("MLX diagnostic payload has incomplete or unexpected fields")
    if payload["schema_version"] != DIAGNOSTIC_SAMPLE_SCHEMA_VERSION:
        raise ContractError("MLX diagnostic payload uses the wrong schema")
    if payload["framework"] != "mlx-lm":
        raise ContractError("MLX diagnostic payload uses the wrong framework")
    expected_diagnostic = {
        "release_eligible": False,
        "release_gates_enforced": False,
        "release_blockers": [DIAGNOSTIC_RELEASE_BLOCKER],
        "publication_contract": "never-valid-as-antfly_gemma4_lora_benchmark_sample/v5",
    }
    if payload["diagnostic"] != expected_diagnostic:
        raise ContractError("MLX diagnostic payload release boundary drifted")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ContractError("MLX diagnostic payload omitted implementation identity")
    source = implementation.get("producer_source")
    if not isinstance(source, Mapping) or source.get("schema_version") != DIAGNOSTIC_SOURCE_SCHEMA_VERSION:
        raise ContractError("MLX diagnostic payload omitted diagnostic source identity")
    if source.get("relative_path") != MLX_RUNNER_RELATIVE_PATH:
        raise ContractError("MLX diagnostic payload producer entrypoint differs")
    observations = payload.get("precision_observations")
    if not isinstance(observations, Mapping) or set(observations) != {
        "base_model_storage",
        "lora_parameter_storage",
        "gradient_storage",
        "optimizer_moment_storage",
        "loss",
    }:
        raise ContractError("MLX diagnostic payload precision observations are incomplete")
    metrics = payload.get("metrics")
    memory = metrics.get("memory") if isinstance(metrics, Mapping) else None
    if not isinstance(memory, Mapping) or set(memory) != {
        "framework_allocator_peak_bytes",
        "framework_allocator_peak_source",
        "process_peak_phys_footprint_bytes",
        "sampler_interval_ms",
        "sampler_sample_count",
        "system_deltas",
    }:
        raise ContractError("MLX diagnostic payload memory measurements are incomplete")
    return payload


def atomic_publish_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    validator: Callable[[Path], None] | None = None,
) -> Path:
    """Fsync and atomically publish canonical JSON without replacement."""
    requested = path.expanduser().absolute()
    try:
        requested.parent.mkdir(parents=True, exist_ok=True)
        parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"could not prepare evidence directory: {exc}") from exc
    target = parent / requested.name
    if target.exists() or target.is_symlink():
        raise ContractError(f"refusing to replace existing evidence: {target}")
    encoded = canonical_json_bytes(payload)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent
        )
    except OSError as exc:
        raise ContractError(f"could not create temporary evidence file: {exc}") from exc
    temporary = Path(temporary_name)
    try:
        try:
            with os.fdopen(descriptor, "wb") as output:
                os.fchmod(output.fileno(), 0o644)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
        except OSError as exc:
            raise ContractError(f"could not durably write temporary evidence: {exc}") from exc
        if validator is not None:
            validator(temporary)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContractError(f"refusing to replace existing evidence: {target}") from exc
        except OSError as exc:
            raise ContractError(f"could not atomically publish benchmark evidence: {exc}") from exc
        try:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            try:
                target.unlink()
            except OSError as cleanup_exc:
                raise ContractError(
                    f"evidence directory fsync failed ({exc}) and cleanup failed ({cleanup_exc})"
                ) from cleanup_exc
            raise ContractError(f"evidence directory fsync failed: {exc}") from exc
        return target
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def postflight(args: argparse.Namespace, preflight_result: Preflight, environment: Mapping[str, Any]) -> None:
    require_files_unchanged(preflight_result.bound_files)
    require_files_unchanged(environment["bound_file_identities"])
    if prefixed_sha256(args.prepared.resolve()) != preflight_result.prepared["artifact_sha256"]:
        raise ContractError("prepared artifact drifted during benchmark execution")
    if prefixed_sha256(preflight_result.adapter.checkpoint) != preflight_result.adapter.checkpoint_sha256:
        raise ContractError("initial adapter checkpoint drifted during benchmark execution")
    if prefixed_sha256(preflight_result.adapter.directory / "adapter_config.json") != preflight_result.adapter.config_sha256:
        raise ContractError("initial adapter config drifted during benchmark execution")
    verify_prepared_source_dataset(preflight_result.prepared_summary, args.source_dataset)
    reference = preflight_result.lock["mlx_reference"]
    verify_source_checkout(args.mlx_source, reference["source_revisions"]["mlx"], source_name="MLX")
    verify_source_checkout(args.mlx_lm_source, reference["source_revisions"]["mlx-lm"], source_name="MLX-LM")
    preverified_bundle = verify_mlx_native_build_before_import(
        args,
        preflight_result.lock,
        environment["mlx_checkout"],
        environment["mlx_lm_checkout"],
    )
    native_runtime = verify_mlx_native_runtime(
        args,
        preflight_result.lock,
        environment["mlx_checkout"],
        environment["mx"],
        preverified_bundle,
    )
    for field in ("native_artifact_inventory", "build_attestation"):
        if native_runtime[field] != environment[field]:
            raise ContractError(f"MLX {field} drifted during benchmark execution")
    if verify_packages(preflight_result.lock, "mlx_reference") != environment["versions"]:
        raise ContractError("MLX package environment drifted during benchmark execution")


def build_worker_payload(argv: Sequence[str], reporter: WorkerPhaseReporter) -> dict[str, Any]:
    claim_fresh_process()
    force_offline_environment()
    args = build_parser().parse_args(list(argv))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ContractError("MLX-LM release benchmarks require Darwin arm64")
    prepared = preflight(args)
    runner = diagnostic_runner_source() if args.diagnostic_only else runner_source_attestation()
    precision_recorder = PrecisionEvidenceRecorder()
    environment, metrics = run_mlx(
        args,
        prepared,
        precision_recorder,
        before_measured=lambda: reporter.barrier("measured-optimizer-steps-start"),
        after_measured=lambda: reporter.barrier("measured-optimizer-steps-end"),
    )
    payload_builder = build_diagnostic_payload if args.diagnostic_only else build_sample_payload
    payload = payload_builder(
        args, prepared, environment, metrics, hardware_fingerprint(), argv, runner
    )
    postflight(args, prepared, environment)
    return {
        "sample": payload,
        "precision_observations": precision_recorder.finalize(),
        "runner": runner,
    }


def worker_process_main(argv: Sequence[str], control_fd: int) -> int:
    try:
        raw_channel = socket.socket(fileno=control_fd)
    except OSError as exc:
        print(f"error: could not adopt benchmark control fd: {exc}", file=sys.stderr)
        return 2
    channel = JsonControlChannel(
        raw_channel,
        receive_timeout_seconds=CONTROL_IDLE_TIMEOUT_SECONDS,
    )
    try:
        payload = build_worker_payload(argv, WorkerPhaseReporter(channel))
        channel.send(
            {
                "schema_version": CONTROL_PROTOCOL_VERSION,
                "kind": "result",
                "payload": payload,
            }
        )
        if channel.receive() != {
            "schema_version": CONTROL_PROTOCOL_VERSION,
            "kind": "ack",
            "phase": "result",
        }:
            raise ContractError("benchmark coordinator sent an invalid result acknowledgement")
        return 0
    except ContractError as exc:
        try:
            channel.send(
                {
                    "schema_version": CONTROL_PROTOCOL_VERSION,
                    "kind": "error",
                    "message": str(exc),
                }
            )
        except ContractError:
            pass
        return 2
    except Exception as exc:
        try:
            channel.send(
                {
                    "schema_version": CONTROL_PROTOCOL_VERSION,
                    "kind": "error",
                    "message": f"unexpected {type(exc).__name__}: {exc}",
                }
            )
        except ContractError:
            pass
        return 3
    finally:
        raw_channel.close()


def _terminate_worker(worker: subprocess.Popen[Any]) -> None:
    if worker.poll() is not None:
        return
    worker.terminate()
    try:
        worker.wait(timeout=10)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()


def _coordinator_result(
    channel: JsonControlChannel,
    sampler: DarwinProcessMemorySampler,
    lock: Mapping[str, Any],
    *,
    enforce_release_memory_gates: bool = True,
) -> tuple[dict[str, Any], ProcessMemoryMeasurement, dict[str, int | float]]:
    expected_phase = "measured-optimizer-steps-start"
    before: DarwinSystemMemorySnapshot | None = None
    after: DarwinSystemMemorySnapshot | None = None
    while True:
        message = channel.receive()
        kind = message.get("kind")
        if kind == "error":
            if set(message) != {"schema_version", "kind", "message"}:
                raise ContractError("benchmark worker emitted a malformed error message")
            detail = message["message"]
            if not isinstance(detail, str) or not detail:
                raise ContractError("benchmark worker emitted an empty error message")
            raise ContractError(f"MLX benchmark worker failed: {detail}")
        if kind == "phase":
            if set(message) != {"schema_version", "kind", "phase"}:
                raise ContractError("benchmark worker emitted a malformed phase message")
            phase = message["phase"]
            if phase != expected_phase:
                raise ContractError(
                    f"benchmark worker phase order drifted ({phase!r} != {expected_phase!r})"
                )
            if phase == "measured-optimizer-steps-start":
                before = capture_darwin_system_memory_snapshot(before_measured=True)
                expected_phase = "measured-optimizer-steps-end"
            else:
                after = capture_darwin_system_memory_snapshot(before_measured=False)
                expected_phase = "result"
            channel.send(
                {
                    "schema_version": CONTROL_PROTOCOL_VERSION,
                    "kind": "ack",
                    "phase": phase,
                }
            )
            continue
        if kind != "result" or set(message) != {"schema_version", "kind", "payload"}:
            raise ContractError("benchmark worker emitted an unexpected control message")
        if expected_phase != "result" or before is None or after is None:
            raise ContractError("benchmark worker returned without bracketing measured optimizer steps")
        payload = message["payload"]
        if not isinstance(payload, dict):
            raise ContractError("benchmark worker result payload must be an object")
        process_memory = sampler.stop()
        system_deltas = darwin_system_memory_deltas(before, after)
        if enforce_release_memory_gates:
            enforce_system_memory_gates(lock, system_deltas)
        channel.send(
            {
                "schema_version": CONTROL_PROTOCOL_VERSION,
                "kind": "ack",
                "phase": "result",
            }
        )
        return payload, process_memory, system_deltas


def run(argv: Sequence[str]) -> Path:
    claim_fresh_process()
    force_offline_environment()
    args = build_parser().parse_args(list(argv))
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ContractError("MLX-LM release benchmarks require Darwin arm64")
    output_path = args.output.expanduser().absolute()
    evidence_path = precision_evidence_output_path(args.output)
    targets = (output_path,) if args.diagnostic_only else (output_path, evidence_path)
    for target in targets:
        if target.exists() or target.is_symlink():
            raise ContractError(f"refusing to replace existing evidence: {target}")
    lock = load_lock(LOCK_PATH)
    if lock["benchmark_contract"]["memory"]["sampler_interval_ms"] != MEMORY_SAMPLER_INTERVAL_MS:
        raise ContractError("process-memory sampler interval differs from the lock")

    parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    command = [
        str(Path(sys.executable).resolve()),
        str(SCRIPT_PATH),
        INTERNAL_WORKER_FLAG,
        str(child_socket.fileno()),
        *argv,
    ]
    try:
        worker = subprocess.Popen(
            command,
            env=dict(os.environ),
            pass_fds=(child_socket.fileno(),),
            close_fds=True,
        )
    except OSError as exc:
        parent_socket.close()
        child_socket.close()
        raise ContractError(f"could not start fresh MLX benchmark worker: {exc}") from exc
    child_socket.close()
    sampler = DarwinProcessMemorySampler(
        pid=worker.pid,
        interval_ms=MEMORY_SAMPLER_INTERVAL_MS,
    )
    sampler_started = False
    try:
        sampler.start()
        sampler_started = True
        bundle, process_memory, system_deltas = _coordinator_result(
            JsonControlChannel(
                parent_socket,
                receive_timeout_seconds=CONTROL_IDLE_TIMEOUT_SECONDS,
            ),
            sampler,
            lock,
            enforce_release_memory_gates=not args.diagnostic_only,
        )
        sampler_started = False
        try:
            return_code = worker.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise ContractError("MLX benchmark worker did not exit after returning its result") from exc
        if return_code != 0:
            raise ContractError(f"MLX benchmark worker exited with status {return_code} after its result")
    except BaseException:
        if sampler_started:
            sampler.cancel()
        _terminate_worker(worker)
        raise
    finally:
        parent_socket.close()

    if not isinstance(bundle, dict) or set(bundle) != {"sample", "precision_observations", "runner"}:
        raise ContractError("MLX benchmark worker emitted an invalid evidence bundle")
    payload = bundle["sample"]
    observations = bundle["precision_observations"]
    runner = bundle["runner"]
    if not isinstance(payload, dict) or not isinstance(observations, dict) or not isinstance(runner, dict):
        raise ContractError("MLX benchmark worker evidence bundle values must be objects")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ContractError("MLX benchmark worker omitted metrics")
    memory = metrics.get("memory")
    if not isinstance(memory, dict) or set(memory) != {
        "framework_allocator_peak_bytes",
        "framework_allocator_peak_source",
    }:
        raise ContractError("MLX benchmark worker emitted an invalid partial memory metric")
    memory.update(
        {
            "process_peak_phys_footprint_bytes": process_memory.peak_phys_footprint_bytes,
            "sampler_interval_ms": MEMORY_SAMPLER_INTERVAL_MS,
            "sampler_sample_count": process_memory.sample_count,
            "system_deltas": system_deltas,
        }
    )
    if args.diagnostic_only:
        payload["precision_observations"] = observations
        validate_diagnostic_payload(payload)
        return atomic_publish_json(
            args.output,
            payload,
            validator=lambda temporary: validate_diagnostic_payload(load_json(temporary)),
        )
    evidence = build_precision_evidence(payload, observations, runner, lock)
    payload["precision_evidence"] = precision_evidence_reference(args.output, evidence)
    # Publish and durably fsync the content-addressed evidence first.  A later
    # sample-publication failure can leave only an unreferenced evidence file;
    # it can never leave a published sample whose evidence was never durable.
    atomic_publish_json(
        evidence_path,
        evidence,
        validator=lambda temporary: validate_precision_evidence_payload(
            load_json(temporary), payload, lock, where=str(temporary),
        ),
    )
    return atomic_publish_json(
        args.output,
        payload,
        validator=lambda temporary: validate_sample(
            temporary,
            lock,
            LOCK_PATH,
            precision_evidence_path=evidence_path,
        ),
    )


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == INTERNAL_WORKER_FLAG:
        if len(argv) < 2 or re.fullmatch(r"[0-9]+", argv[1]) is None:
            print("error: malformed internal benchmark worker fd", file=sys.stderr)
            return 2
        return worker_process_main(argv[2:], int(argv[1]))
    if INTERNAL_WORKER_FLAG in argv:
        print("error: internal benchmark worker flag must be first", file=sys.stderr)
        return 2
    try:
        output = run(argv)
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
