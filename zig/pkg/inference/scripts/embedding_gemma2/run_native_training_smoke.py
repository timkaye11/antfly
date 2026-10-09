#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run two native steps uninterrupted and as stop/resume, then compare state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_safetensors(path: Path) -> dict[str, tuple[tuple[int, ...], bytes]]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated safetensors: {path}")
    header_size = int.from_bytes(raw[:8], "little")
    if header_size <= 0 or 8 + header_size > len(raw):
        raise ValueError(f"invalid safetensors header: {path}")
    header = json.loads(raw[8 : 8 + header_size])
    data = memoryview(raw)[8 + header_size :]
    result: dict[str, tuple[tuple[int, ...], bytes]] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if metadata.get("dtype") != "F32":
            raise ValueError(f"{path}: {name} is not F32")
        begin, end = metadata["data_offsets"]
        shape = tuple(int(value) for value in metadata["shape"])
        if begin < 0 or end < begin or end > len(data):
            raise ValueError(f"{path}: invalid offsets for {name}")
        payload = bytes(data[begin:end])
        elements = 1
        for dimension in shape:
            elements *= dimension
        if len(payload) != elements * 4:
            raise ValueError(f"{path}: invalid shape for {name}")
        result[name] = (shape, payload)
    return result


def canonical_tensor_digest(tensors: dict[str, tuple[tuple[int, ...], bytes]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        shape, payload = tensors[name]
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def descendant_pids(pid: int) -> set[int]:
    result = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = (Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except (FileNotFoundError, PermissionError):
            continue
        for child_text in children:
            child = int(child_text)
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def gpu_memory_mib(pid: int) -> int:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return 0
    total = 0
    owned = descendant_pids(pid)
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in owned:
            total += int(fields[1])
    return total


def run_native(binary: Path, config: Path, extra: list[str], log_path: Path) -> dict[str, Any]:
    # This harness consumes the dedicated `train-embeddinggemma2` artifact;
    # the multi-call application routes to the same entrypoint separately.
    command = [str(binary), str(config), "--trace", *extra]
    started = time.monotonic()
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    peak_mib = 0
    lines: list[str] = []

    def monitor() -> None:
        nonlocal peak_mib
        while process.poll() is None:
            peak_mib = max(peak_mib, gpu_memory_mib(process.pid))
            time.sleep(0.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
    returncode = process.wait()
    thread.join(timeout=2)
    log_path.write_text("".join(lines))
    if returncode != 0:
        raise RuntimeError(f"native training failed ({returncode}); see {log_path}")
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "event" in value:
            events.append(value)
    return {
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "peak_process_gpu_mib": peak_mib,
        "events": events,
        "log": str(log_path),
    }


def write_config(source: dict[str, Any], path: Path, output: Path) -> None:
    value = dict(source)
    value["output_dir"] = str(output.resolve())
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def checkpoint_summary(directory: Path) -> dict[str, Any]:
    tensor_path = directory / "training.safetensors"
    state_path = directory / "state.json"
    tensors = read_safetensors(tensor_path)
    state = json.loads(state_path.read_text())
    return {
        "resume": {"next_step_state_sha256": canonical_tensor_digest(tensors)},
        "provenance": {
            key: state[key]
            for key in (
                "job_fingerprint",
                "base_sha256",
                "dataset_sha256",
                "tokenizer_sha256",
                "preprocessing_sha256",
                "seed",
                "shuffle_identity",
                "total_steps",
            )
        },
        "optimizer_step": state["optimizer_step"],
        "generation": state["generation"],
        "tensor_count": len(tensors),
        "safetensors_sha256": sha256(tensor_path),
    }


def compare_initial(trace_path: Path, reference_path: Path) -> dict[str, Any]:
    trace = read_safetensors(trace_path)
    reference = read_safetensors(reference_path)
    initial = {
        name.removeprefix("initial_adapter."): value
        for name, value in trace.items()
        if name.startswith("initial_adapter.")
    }
    expected = {
        name.removeprefix("adapter.").removeprefix("base_model.model."): value
        for name, value in reference.items()
        if ".lora_A.weight" in name or ".lora_B.weight" in name
    }
    missing = sorted(set(expected) - set(initial))
    extra = sorted(set(initial) - set(expected))
    mismatched = sorted(name for name in set(initial) & set(expected) if initial[name] != expected[name])
    return {
        "pass": len(initial) == 338 and not missing and not extra and not mismatched,
        "count": len(initial),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "reference_sha256": sha256(reference_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.binary.is_file() or not args.binary.is_absolute():
        raise ValueError("--binary must be an absolute executable path")
    if args.work_dir.exists():
        raise FileExistsError(f"work directory must be new: {args.work_dir}")
    args.work_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    outputs = {name: args.work_dir / name for name in ("uninterrupted", "stopped", "resumed")}
    configs = {name: args.work_dir / f"{name}.json" for name in outputs}
    for name in outputs:
        write_config(config, configs[name], outputs[name])

    runs = {}
    runs["uninterrupted"] = run_native(args.binary, configs["uninterrupted"], [], args.work_dir / "uninterrupted.log")
    runs["stopped"] = run_native(args.binary, configs["stopped"], ["--stop-after-steps", "1"], args.work_dir / "stopped.log")
    first_checkpoint = outputs["stopped"] / "checkpoint-00000001"
    runs["resumed"] = run_native(args.binary, configs["resumed"], ["--resume-from", str(first_checkpoint)], args.work_dir / "resumed.log")

    uninterrupted = checkpoint_summary(outputs["uninterrupted"] / "checkpoint-00000002")
    resumed = checkpoint_summary(outputs["resumed"] / "checkpoint-00000002")
    uninterrupted_path = args.work_dir / "uninterrupted-summary.json"
    resumed_path = args.work_dir / "resumed-summary.json"
    uninterrupted_path.write_text(json.dumps(uninterrupted, indent=2, sort_keys=True) + "\n")
    resumed_path.write_text(json.dumps(resumed, indent=2, sort_keys=True) + "\n")
    state_equal = uninterrupted["resume"] == resumed["resume"]
    provenance_equal = uninterrupted["provenance"] == resumed["provenance"]
    initial = None
    if args.initial_adapter:
        initial = compare_initial(outputs["uninterrupted"] / "trace-step-0.safetensors", args.initial_adapter)
    report = {
        "schema": "antfly.embedding_gemma2.native_training_smoke.v1",
        "pass": state_equal and provenance_equal and (initial is None or initial["pass"]),
        "binary": str(args.binary),
        "binary_sha256": sha256(args.binary),
        "config_sha256": sha256(args.config),
        "resume": {
            "schema": "antfly.embedding_gemma2.finetune_resume.v1",
            "pass": state_equal and provenance_equal,
            "next_step_state_sha256": uninterrupted["resume"]["next_step_state_sha256"],
            "uninterrupted_next_step_sha256": uninterrupted["resume"]["next_step_state_sha256"],
            "resumed_next_step_sha256": resumed["resume"]["next_step_state_sha256"],
            "provenance_equal": provenance_equal,
            "uninterrupted_summary": str(uninterrupted_path),
            "resumed_summary": str(resumed_path),
        },
        "initial_adapter": initial,
        "runs": runs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": report["pass"], "report": str(args.report)}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
