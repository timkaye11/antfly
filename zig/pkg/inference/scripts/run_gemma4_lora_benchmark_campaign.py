#!/usr/bin/env python3
"""Run a closed, serial Antfly-versus-MLX Gemma4 LoRA campaign.

The framework runners emit one sample each.  This orchestrator is the sole
owner of campaign/run identifiers and sequence metadata: a plan contains only
runner argv prefixes, and this process appends the reserved identity and output
arguments.  Successful runs are recorded in launch order and a ``COMPLETE.json``
ledger is published atomically without replacing prior evidence.

Only the Python standard library is required.  Model/framework dependencies
are loaded by the child runners, never by this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gemma4_oracle_contract import ContractError, load_json, prefixed_sha256, require_exact_keys


PLAN_SCHEMA_VERSION = "antfly_gemma4_lora_benchmark_campaign_plan/v1"
MANIFEST_SCHEMA_VERSION = "antfly_gemma4_lora_benchmark_campaign_manifest/v1"
ORCHESTRATOR_SCHEMA_VERSION = "antfly_gemma4_lora_benchmark_campaign_orchestrator/v1"
ORCHESTRATOR_RELATIVE_PATH = "zig/pkg/inference/scripts/run_gemma4_lora_benchmark_campaign.py"
ARGV_DIGEST_DOMAIN = b"antfly_gemma4_lora_benchmark_campaign_argv/v1\0"
FRAMEWORKS = ("antfly-zig-metal", "mlx-lm")
RUNNER_NAMES = {
    "antfly-zig-metal": "run_antfly_gemma4_lora_benchmark.py",
    "mlx-lm": "run_gemma4_lora_mlx_benchmark.py",
}
RESERVED_OPTIONS = (
    "--campaign-id",
    "--run-id",
    "--repetition",
    "--sequence-index",
    "--output",
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SCRIPT_PATH = Path(__file__).resolve()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}: expected non-empty string")
    return value


def _identifier(value: Any, where: str) -> str:
    result = _string(value, where)
    if ID_PATTERN.fullmatch(result) is None:
        raise ContractError(f"{where}: invalid campaign identifier")
    return result


def _sha256(value: Any, where: str) -> str:
    result = _string(value, where)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ContractError(f"{where}: expected sha256:<64 lowercase hex characters>")
    return result


def canonical_argv_sha256(argv: Sequence[str]) -> str:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ContractError("runner argv must be a non-empty array")
    normalized: list[str] = []
    for index, argument in enumerate(argv):
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise ContractError(f"runner argv[{index}] must be a non-empty string without NUL")
        normalized.append(argument)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(ARGV_DIGEST_DOMAIN + encoded).hexdigest()


def _reserved_option_present(argv: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(option + "=") for argument in argv)


def _option_value(argv: Sequence[str], option: str, where: str) -> str:
    values: list[str] = []
    for index, argument in enumerate(argv):
        if argument == option:
            if index + 1 >= len(argv):
                raise ContractError(f"{where}: {option} has no value")
            values.append(argv[index + 1])
        elif argument.startswith(option + "="):
            values.append(argument[len(option) + 1 :])
    if len(values) != 1 or not values[0]:
        raise ContractError(f"{where}: expected exactly one {option}")
    return values[0]


def _normalized_relative_path(value: Any, where: str) -> str:
    text = _string(value, where)
    path = Path(text)
    if path.is_absolute() or text != path.as_posix() or any(part in ("", ".", "..") for part in path.parts):
        raise ContractError(f"{where}: expected a normalized relative path")
    return text


def _regular_non_symlink(path: Path, where: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.is_symlink():
        raise ContractError(f"{where}: symbolic links are forbidden")
    try:
        info = absolute.stat()
    except OSError as exc:
        raise ContractError(f"{where}: could not stat {absolute}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ContractError(f"{where}: expected a regular file")
    return absolute.resolve(strict=True)


def _executable_path(path: Path, where: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        info = absolute.stat()
    except OSError as exc:
        raise ContractError(f"{where}: could not resolve executable {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(absolute, os.X_OK):
        raise ContractError(f"{where}: expected an executable regular file")
    # Preserve a virtual-environment launcher symlink: replacing it with the
    # base interpreter can change Python's prefix and imported package tree.
    return absolute


def _evidence_file(root: Path, relative_path: str, where: str) -> Path:
    candidate = root.joinpath(*Path(relative_path).parts)
    current = root
    for part in Path(relative_path).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{where}: symbolic-link path components are forbidden")
    resolved = _regular_non_symlink(candidate, where)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ContractError(f"{where}: path escapes the campaign directory") from exc
    return resolved


def _resolve_command(raw: Any, framework: str, where: str) -> list[str]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ContractError(f"{where}: expected [python, runner, ...] argv")
    argv = list(raw)
    canonical_argv_sha256(argv)
    if any(_reserved_option_present(argv, option) for option in RESERVED_OPTIONS):
        raise ContractError(f"{where}: campaign identity/output options are reserved to the orchestrator")
    executable_text = argv[0]
    executable_candidate = Path(executable_text).expanduser()
    if executable_candidate.is_absolute():
        executable = _executable_path(executable_candidate, f"{where}[0]")
    else:
        located = shutil.which(executable_text)
        if located is None:
            raise ContractError(f"{where}[0]: executable was not found")
        executable = _executable_path(Path(located), f"{where}[0]")
    runner_candidate = Path(argv[1]).expanduser()
    if not runner_candidate.is_absolute():
        raise ContractError(f"{where}[1]: runner path must be absolute")
    runner = _regular_non_symlink(runner_candidate, f"{where}[1]")
    if runner.name != RUNNER_NAMES[framework]:
        raise ContractError(f"{where}[1]: expected {RUNNER_NAMES[framework]}")
    argv[0] = str(executable)
    argv[1] = str(runner)
    return argv


def load_plan(path: Path) -> tuple[dict[str, Any], str]:
    plan_path = _regular_non_symlink(path, "campaign plan")
    digest = prefixed_sha256(plan_path)
    plan = dict(_mapping(load_json(plan_path), "campaign plan"))
    require_exact_keys(plan, ("schema_version", "repetitions", "cells"), where="campaign plan")
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ContractError("campaign plan: unsupported schema version")
    repetitions = _integer(plan["repetitions"], "campaign plan.repetitions", 1)
    raw_cells = plan["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ContractError("campaign plan.cells must be a non-empty array")
    cells: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_cell in enumerate(raw_cells):
        cell = _mapping(raw_cell, f"campaign plan.cells[{index}]")
        require_exact_keys(cell, ("cell_id", "commands"), where=f"campaign plan.cells[{index}]")
        cell_id = _identifier(cell["cell_id"], f"campaign plan.cells[{index}].cell_id")
        if cell_id in seen_ids:
            raise ContractError(f"campaign plan: duplicate cell_id {cell_id!r}")
        seen_ids.add(cell_id)
        commands = _mapping(cell["commands"], f"campaign plan.cells[{index}].commands")
        require_exact_keys(commands, FRAMEWORKS, where=f"campaign plan.cells[{index}].commands")
        cells.append({
            "cell_id": cell_id,
            "commands": {
                framework: _resolve_command(
                    commands[framework], framework, f"campaign plan.cells[{index}].commands.{framework}"
                )
                for framework in FRAMEWORKS
            },
        })
    return {"schema_version": PLAN_SCHEMA_VERSION, "repetitions": repetitions, "cells": cells}, digest


def _sample_identity(path: Path, expected: Mapping[str, Any]) -> None:
    resolved = _regular_non_symlink(path, "runner sample")
    payload = _mapping(load_json(resolved), "runner sample")
    for field in ("campaign_id", "run_id", "repetition", "sequence_index", "framework", "process"):
        if field not in payload:
            raise ContractError(f"runner sample: missing orchestrator-owned field {field}")
    for field in ("campaign_id", "run_id", "repetition", "sequence_index", "framework"):
        if payload[field] != expected[field]:
            raise ContractError(f"runner sample: {field} differs from the orchestrated run")


def _atomic_publish(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ContractError(f"refusing to replace existing campaign ledger: {target}")
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{target.name}.staging-", dir=target.parent)
    staging = Path(staging_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o644)
            output.write((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(staging, target)
        except FileExistsError as exc:
            raise ContractError(f"refusing to replace existing campaign ledger: {target}") from exc
        published = True
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        staging.unlink(missing_ok=True)


def _default_run_process(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), check=False, stdin=subprocess.DEVNULL).returncode


def execute_campaign(
    plan_path: Path,
    output_directory: Path,
    *,
    run_process: Callable[[Sequence[str]], int] = _default_run_process,
    clock: Callable[[], int] = time.time_ns,
    token: Callable[[int], str] = secrets.token_hex,
) -> Path:
    """Execute a plan serially and return the newly published COMPLETE ledger."""
    plan, plan_digest = load_plan(plan_path)
    output_root = output_directory.expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    sample_root = output_root / "samples"
    if sample_root.is_symlink():
        raise ContractError("campaign sample directory must not be a symbolic link")
    sample_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "COMPLETE.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ContractError(f"refusing to replace existing campaign ledger: {manifest_path}")

    campaign_id = _identifier("gemma4-" + token(16), "generated campaign_id")
    created_unix_ns = _integer(clock(), "campaign creation timestamp", 1)
    planned_count = len(plan["cells"]) * plan["repetitions"] * len(FRAMEWORKS)
    for sequence_index in range(planned_count):
        target = sample_root / f"{sequence_index:04d}.json"
        if target.exists() or target.is_symlink():
            raise ContractError(f"refusing to replace existing benchmark evidence: {target}")

    runs: list[dict[str, Any]] = []
    generated_run_ids: set[str] = set()
    sequence_index = 0
    for cell in plan["cells"]:
        for repetition in range(plan["repetitions"]):
            order = FRAMEWORKS if repetition % 2 == 0 else tuple(reversed(FRAMEWORKS))
            for framework in order:
                run_id = _identifier(
                    f"{campaign_id}.run-{sequence_index:04d}-{token(8)}", "generated run_id"
                )
                if run_id in generated_run_ids:
                    raise ContractError("campaign run identifier generator produced a collision")
                generated_run_ids.add(run_id)
                sample_path = sample_root / f"{sequence_index:04d}.json"
                argv = [
                    *cell["commands"][framework],
                    "--campaign-id", campaign_id,
                    "--run-id", run_id,
                    "--repetition", str(repetition),
                    "--sequence-index", str(sequence_index),
                    "--output", str(sample_path),
                ]
                started_unix_ns = _integer(clock(), "run start timestamp", 1)
                if runs and started_unix_ns < runs[-1]["completed_unix_ns"]:
                    raise ContractError("campaign clock moved backwards across serial runs")
                returncode = run_process(argv)
                completed_unix_ns = _integer(clock(), "run completion timestamp", 1)
                if completed_unix_ns <= started_unix_ns:
                    raise ContractError("campaign run completion must be after its start")
                if isinstance(returncode, bool) or not isinstance(returncode, int):
                    raise ContractError("campaign runner returned a non-integer exit status")
                if returncode != 0:
                    raise ContractError(
                        f"campaign run {run_id} ({framework}) failed with exit status {returncode}; COMPLETE was not published"
                    )
                expected = {
                    "campaign_id": campaign_id,
                    "run_id": run_id,
                    "repetition": repetition,
                    "sequence_index": sequence_index,
                    "framework": framework,
                }
                _sample_identity(sample_path, expected)
                runs.append({
                    "cell_id": cell["cell_id"],
                    "framework": framework,
                    "repetition": repetition,
                    "sequence_index": sequence_index,
                    "run_id": run_id,
                    "argv": argv,
                    "argv_sha256": canonical_argv_sha256(argv),
                    "started_unix_ns": started_unix_ns,
                    "completed_unix_ns": completed_unix_ns,
                    "sample": {
                        "relative_path": sample_path.relative_to(output_root).as_posix(),
                        "sha256": prefixed_sha256(sample_path),
                        "size_bytes": sample_path.stat().st_size,
                    },
                })
                sequence_index += 1

    if prefixed_sha256(_regular_non_symlink(plan_path, "campaign plan")) != plan_digest:
        raise ContractError("campaign plan changed during execution; COMPLETE was not published")
    for run in runs:
        artifact = run["sample"]
        sample_path = _evidence_file(output_root, artifact["relative_path"], "final campaign sample")
        if sample_path.stat().st_size != artifact["size_bytes"] or prefixed_sha256(sample_path) != artifact["sha256"]:
            raise ContractError("a campaign sample changed before COMPLETE publication")
    completed_unix_ns = _integer(clock(), "campaign completion timestamp", 1)
    if not runs or completed_unix_ns < runs[-1]["completed_unix_ns"]:
        raise ContractError("campaign completion timestamp precedes its final run")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "campaign_id": campaign_id,
        "created_unix_ns": created_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "plan": {
            "sha256": plan_digest,
            "cell_count": len(plan["cells"]),
            "repetitions_per_cell": plan["repetitions"],
        },
        "orchestrator": {
            "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
            "relative_path": ORCHESTRATOR_RELATIVE_PATH,
            "source_sha256": prefixed_sha256(SCRIPT_PATH),
        },
        "run_count": len(runs),
        "runs": runs,
    }
    _atomic_publish(manifest_path, manifest)
    return manifest_path


def verify_complete_campaign_manifest(
    manifest_path: Path,
    samples: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Verify a COMPLETE ledger and its exact set of final sample artifacts."""
    path = _regular_non_symlink(manifest_path, "campaign manifest")
    root = path.parent.resolve(strict=True)
    if path.name != "COMPLETE.json":
        raise ContractError("campaign manifest must be the COMPLETE.json commit marker")
    manifest = dict(_mapping(load_json(path), "campaign manifest"))
    require_exact_keys(
        manifest,
        (
            "schema_version", "status", "campaign_id", "created_unix_ns",
            "completed_unix_ns", "plan", "orchestrator", "run_count", "runs",
        ),
        where="campaign manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION or manifest["status"] != "complete":
        raise ContractError("campaign manifest is not a supported COMPLETE ledger")
    campaign_id = _identifier(manifest["campaign_id"], "campaign manifest.campaign_id")
    created = _integer(manifest["created_unix_ns"], "campaign manifest.created_unix_ns", 1)
    completed = _integer(manifest["completed_unix_ns"], "campaign manifest.completed_unix_ns", 1)
    if completed < created:
        raise ContractError("campaign manifest completion precedes creation")

    plan = _mapping(manifest["plan"], "campaign manifest.plan")
    require_exact_keys(plan, ("sha256", "cell_count", "repetitions_per_cell"), where="campaign manifest.plan")
    _sha256(plan["sha256"], "campaign manifest.plan.sha256")
    cell_count = _integer(plan["cell_count"], "campaign manifest.plan.cell_count", 1)
    repetitions = _integer(plan["repetitions_per_cell"], "campaign manifest.plan.repetitions_per_cell", 1)

    orchestrator = _mapping(manifest["orchestrator"], "campaign manifest.orchestrator")
    require_exact_keys(
        orchestrator, ("schema_version", "relative_path", "source_sha256"),
        where="campaign manifest.orchestrator",
    )
    if orchestrator["schema_version"] != ORCHESTRATOR_SCHEMA_VERSION:
        raise ContractError("campaign manifest: unsupported orchestrator schema")
    if orchestrator["relative_path"] != ORCHESTRATOR_RELATIVE_PATH:
        raise ContractError("campaign manifest: orchestrator path differs from the admitted source")
    orchestrator_source_sha256 = _sha256(
        orchestrator["source_sha256"], "campaign manifest.orchestrator.source_sha256"
    )
    if orchestrator_source_sha256 != prefixed_sha256(SCRIPT_PATH):
        raise ContractError("campaign manifest: orchestrator source hash differs from the validating source")

    raw_runs = manifest["runs"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ContractError("campaign manifest.runs must be a non-empty array")
    run_count = _integer(manifest["run_count"], "campaign manifest.run_count", 1)
    expected_run_count = cell_count * repetitions * len(FRAMEWORKS)
    if run_count != len(raw_runs) or run_count != expected_run_count:
        raise ContractError("campaign manifest run count is not closed over every cell/repetition/framework")

    supplied: dict[Path, Mapping[str, Any]] = {}
    for sample_path, payload in samples:
        resolved = _regular_non_symlink(sample_path, "comparison sample")
        if resolved in supplied:
            raise ContractError("comparison supplied the same sample artifact more than once")
        supplied[resolved] = _mapping(payload, "comparison sample")

    ledger_paths: set[Path] = set()
    run_ids: set[str] = set()
    cell_order: list[str] = []
    last_cell: str | None = None
    seen_cells: set[str] = set()
    command_prefixes: dict[tuple[str, str], list[str]] = {}
    normalized_runs: list[Mapping[str, Any]] = []
    for index, raw_run in enumerate(raw_runs):
        run = _mapping(raw_run, f"campaign manifest.runs[{index}]")
        require_exact_keys(
            run,
            (
                "cell_id", "framework", "repetition", "sequence_index", "run_id",
                "argv", "argv_sha256", "started_unix_ns", "completed_unix_ns", "sample",
            ),
            where=f"campaign manifest.runs[{index}]",
        )
        cell_id = _identifier(run["cell_id"], f"campaign manifest.runs[{index}].cell_id")
        if cell_id != last_cell:
            if cell_id in seen_cells:
                raise ContractError("campaign manifest cell runs are not contiguous")
            seen_cells.add(cell_id)
            cell_order.append(cell_id)
            last_cell = cell_id
        framework = run["framework"]
        if framework not in FRAMEWORKS:
            raise ContractError(f"campaign manifest.runs[{index}]: unsupported framework")
        repetition = _integer(run["repetition"], f"campaign manifest.runs[{index}].repetition")
        sequence_index = _integer(run["sequence_index"], f"campaign manifest.runs[{index}].sequence_index")
        if sequence_index != index:
            raise ContractError("campaign manifest sequence_index values must match immutable launch order")
        run_id = _identifier(run["run_id"], f"campaign manifest.runs[{index}].run_id")
        if run_id in run_ids:
            raise ContractError("campaign manifest run_id values must be unique")
        run_ids.add(run_id)
        argv = run["argv"]
        if not isinstance(argv, list) or len(argv) < 12:
            raise ContractError(f"campaign manifest.runs[{index}].argv must contain the runner and owned suffix")
        argv_digest = canonical_argv_sha256(argv)
        if run["argv_sha256"] != argv_digest:
            raise ContractError(f"campaign manifest.runs[{index}]: argv digest mismatch")
        if not Path(argv[0]).is_absolute() or not Path(argv[1]).is_absolute():
            raise ContractError(f"campaign manifest.runs[{index}]: executable and runner paths must be absolute")
        if Path(argv[1]).name != RUNNER_NAMES[framework]:
            raise ContractError(f"campaign manifest.runs[{index}]: framework is relabeled against runner argv")
        expected_flags = {
            "--campaign-id": campaign_id,
            "--run-id": run_id,
            "--repetition": str(repetition),
            "--sequence-index": str(sequence_index),
        }
        for option, expected_value in expected_flags.items():
            if _option_value(argv, option, f"campaign manifest.runs[{index}].argv") != expected_value:
                raise ContractError(f"campaign manifest.runs[{index}]: {option} differs from ledger identity")
        expected_suffix = [
            "--campaign-id", campaign_id,
            "--run-id", run_id,
            "--repetition", str(repetition),
            "--sequence-index", str(sequence_index),
            "--output", _option_value(argv, "--output", f"campaign manifest.runs[{index}].argv"),
        ]
        if argv[-len(expected_suffix) :] != expected_suffix:
            raise ContractError("campaign manifest runner argv does not end in the orchestrator-owned identity suffix")
        prefix_key = (cell_id, framework)
        previous_prefix = command_prefixes.setdefault(prefix_key, argv[: -len(expected_suffix)])
        if argv[: -len(expected_suffix)] != previous_prefix:
            raise ContractError(f"campaign manifest cell {cell_id!r} changed runner argv across repetitions")
        started = _integer(run["started_unix_ns"], f"campaign manifest.runs[{index}].started_unix_ns", 1)
        finished = _integer(run["completed_unix_ns"], f"campaign manifest.runs[{index}].completed_unix_ns", 1)
        if finished <= started:
            raise ContractError("campaign manifest contains a non-positive run interval")
        if index == 0 and started < created:
            raise ContractError("campaign manifest first run precedes campaign creation")
        if normalized_runs and started < normalized_runs[-1]["completed_unix_ns"]:
            raise ContractError("campaign manifest contains overlapping benchmark runs")

        artifact = _mapping(run["sample"], f"campaign manifest.runs[{index}].sample")
        require_exact_keys(artifact, ("relative_path", "sha256", "size_bytes"), where=f"campaign manifest.runs[{index}].sample")
        relative_path = _normalized_relative_path(
            artifact["relative_path"], f"campaign manifest.runs[{index}].sample.relative_path"
        )
        sample_path = _evidence_file(root, relative_path, f"campaign manifest.runs[{index}].sample")
        if sample_path in ledger_paths:
            raise ContractError("campaign manifest lists one sample artifact more than once")
        ledger_paths.add(sample_path)
        sample_digest = _sha256(artifact["sha256"], f"campaign manifest.runs[{index}].sample.sha256")
        sample_size = _integer(
            artifact["size_bytes"], f"campaign manifest.runs[{index}].sample.size_bytes", 1
        )
        if sample_path.stat().st_size != sample_size or prefixed_sha256(sample_path) != sample_digest:
            raise ContractError("campaign manifest sample artifact was replaced or modified")
        if _option_value(argv, "--output", f"campaign manifest.runs[{index}].argv") != str(sample_path):
            raise ContractError("campaign manifest runner output path differs from the recorded sample")
        payload = supplied.get(sample_path)
        if payload is None:
            raise ContractError("campaign manifest sample membership differs from comparison inputs")
        disk_payload = _mapping(load_json(sample_path), "campaign manifest sample")
        encoded_payload = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
        encoded_disk_payload = json.dumps(
            disk_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
        if encoded_payload != encoded_disk_payload:
            raise ContractError("comparison sample payload differs from its recorded artifact")
        payload = disk_payload
        implementation = _mapping(payload.get("implementation"), "comparison sample.implementation")
        producer_source = _mapping(
            implementation.get("producer_source"), "comparison sample.implementation.producer_source"
        )
        producer_files = producer_source.get("files")
        if not isinstance(producer_files, list):
            raise ContractError("comparison sample producer source files must be an array")
        orchestrator_sources = [
            item
            for item in producer_files
            if isinstance(item, Mapping) and item.get("relative_path") == ORCHESTRATOR_RELATIVE_PATH
        ]
        if (
            len(orchestrator_sources) != 1
            or orchestrator_sources[0].get("source_sha256") != orchestrator_source_sha256
        ):
            raise ContractError("campaign manifest orchestrator differs from the sample producer source closure")
        expected_identity = {
            "campaign_id": campaign_id,
            "run_id": run_id,
            "framework": framework,
            "repetition": repetition,
            "sequence_index": sequence_index,
        }
        for field, expected_value in expected_identity.items():
            if payload.get(field) != expected_value:
                raise ContractError(f"campaign manifest sample {field} differs from launch evidence")
        process = _mapping(payload.get("process"), "comparison sample.process")
        process_started = _integer(process.get("started_unix_ns"), "comparison sample.process.started_unix_ns", 1)
        if not started <= process_started <= finished:
            raise ContractError("campaign manifest sample process start is outside its launch interval")
        normalized_runs.append(run)

    if len(cell_order) != cell_count:
        raise ContractError("campaign manifest cell count differs from its launch groups")
    if completed < normalized_runs[-1]["completed_unix_ns"]:
        raise ContractError("campaign manifest completion precedes its final run")
    if ledger_paths != set(supplied):
        raise ContractError("campaign manifest sample membership differs from comparison inputs")
    for cell_id in cell_order:
        cell_runs = [run for run in normalized_runs if run["cell_id"] == cell_id]
        if len(cell_runs) != repetitions * 2:
            raise ContractError(f"campaign manifest cell {cell_id!r} is incomplete")
        for repetition in range(repetitions):
            pair = [run for run in cell_runs if run["repetition"] == repetition]
            expected_order = list(FRAMEWORKS if repetition % 2 == 0 else tuple(reversed(FRAMEWORKS)))
            if len(pair) != 2 or [run["framework"] for run in pair] != expected_order:
                raise ContractError(
                    f"campaign manifest cell {cell_id!r} repetition {repetition} is grouped, relabeled, or not alternating"
                )
            if pair[1]["sequence_index"] != pair[0]["sequence_index"] + 1:
                raise ContractError(f"campaign manifest cell {cell_id!r} repetition {repetition} is not adjacent")
            pair_payloads = [
                supplied[_evidence_file(root, run["sample"]["relative_path"], "campaign pair sample")]
                for run in pair
            ]
            case_encodings = {
                json.dumps(
                    _mapping(payload.get("case"), "comparison sample.case"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                for payload in pair_payloads
            }
            if len(case_encodings) != 1:
                raise ContractError(f"campaign manifest cell {cell_id!r} paired different benchmark cases")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = execute_campaign(args.plan, args.output_dir)
        print(str(manifest))
        return 0
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"Gemma4 benchmark campaign error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
