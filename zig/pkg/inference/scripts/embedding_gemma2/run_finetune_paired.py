#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Collect fresh, alternating resident native/PyTorch training-step pairs.

The manifest contains `cells`: [{"id": ..., "job": "/absolute/job.json",
"initial_adapter": "/absolute/adapter_model.safetensors"}, ...]. Seven cells
and 100 measured pairs are required for production qualification. Smaller
runs exercise the measurement protocol and are explicitly unqualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from finetune_reference import BASE_WEIGHT_SHA256, PreparedDataset, batch_indices, file_sha256
from finetune_contract import validate_performance_cells, validate_performance_workload
from run_native_training_smoke import descendant_pids, gpu_memory_mib


def host_peak_bytes(pid: int) -> int:
    """Conservative external high-water bound for the command process tree."""
    total = 0
    for child in descendant_pids(pid):
        try:
            status = Path(f"/proc/{child}/status").read_text()
        except (FileNotFoundError, PermissionError):
            continue
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                total += int(line.split()[1]) * 1024
                break
    return total


class Worker:
    def __init__(self, command: list[str], directory: Path, environment: dict[str, str]):
        self.directory = directory
        self.control = directory / "control"
        self.control.mkdir(parents=True)
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.metadata: dict[str, Any] | None = None
        self.peak_device_bytes = 0
        self.peak_host_bytes = 0
        self.process = subprocess.Popen(
            [*command, "--paired-control-dir", str(self.control)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=environment, start_new_session=True,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.reader.start()
        self.monitor.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        with self.process.stdout, (self.directory / "worker.log").open("w") as log:
            for line in self.process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("event"):
                    self.events.put(event)
        self.events.put(None)

    def _monitor(self) -> None:
        while self.process.poll() is None:
            self.peak_host_bytes = max(self.peak_host_bytes, host_peak_bytes(self.process.pid))
            self.peak_device_bytes = max(self.peak_device_bytes, gpu_memory_mib(self.process.pid) * 1024 * 1024)
            time.sleep(0.1)

    def wait_event(self, kind: str, step: int, deadline: float) -> dict[str, Any]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.directory.name}: timed out waiting for {kind} {step}")
            try:
                event = self.events.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"{self.directory.name}: no {kind} {step}; see worker.log") from error
            if event is None:
                raise RuntimeError(f"{self.directory.name}: exited before {kind} {step}; see worker.log")
            if event["event"] == "identity":
                self.metadata = event
            elif event["event"] in ("ready", "step"):
                if event["event"] != kind or event.get("step") != step:
                    raise ValueError(f"unexpected paired protocol event: {event}")
                return event

    def grant(self, prefix: str, step: int) -> None:
        # Exclusive creation makes duplicate/out-of-order grants visible.
        with (self.control / f"{prefix}-{step:08d}").open("xb"):
            pass

    def finish(self, deadline: float) -> None:
        code = self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
        self.reader.join(timeout=2)
        self.monitor.join(timeout=2)
        if code:
            raise RuntimeError(f"{self.directory.name}: exit {code}; see worker.log")

    def close(self) -> None:
        if self.process.poll() is None:
            # The native supervisor checkpoints on the first signal; a lost
            # controller must still have a finite cleanup deadline.
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)


def checked_job(path: Path, count: int) -> dict[str, Any]:
    job = json.loads(path.read_text())
    defaults = {"version": 1, "backend": "cuda", "logical_batch_size": 64,
                "microbatch_size": 2, "query_max_tokens": 128, "document_max_tokens": 512,
                "max_negatives": 4, "seed": 42, "epochs": 1, "temperature": 0.05}
    for key, value in defaults.items():
        job.setdefault(key, value)
    lora = {"rank": 16, "alpha": 32, "dropout": 0, "targets": "encoder_and_embedding_projection"}
    lora.update(job.get("lora", {}))
    job["lora"] = lora
    optimizer = {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.999,
                 "epsilon": 1e-8, "weight_decay": 0.01, "max_grad_norm": 1,
                 "warmup_fraction": 0.05}
    # The current reference's optimizer/temperature contract is fixed. Refuse
    # mismatched math rather than silently accepting an incomparable workload.
    if any(job.get("optimizer", {}).get(key, value) != value for key, value in optimizer.items()):
        raise ValueError("paired collector requires the canonical reference optimizer")
    if job["temperature"] != 0.05 or job["backend"] != "cuda" or job.get("device_id", 0) != 0:
        raise ValueError("paired collector requires canonical temperature and CUDA device zero")
    if lora["dropout"] != 0 or lora["targets"] != "encoder_and_embedding_projection":
        raise ValueError("paired collector requires the canonical target/dropout policy")
    prepared = Path(job["prepared_dir"])
    manifest = json.loads((prepared / "manifest.json").read_text())
    for key in ("query_max_tokens", "document_max_tokens", "max_negatives"):
        if job[key] != manifest[key]:
            raise ValueError(f"prepared/job {key} mismatch")
    total = math.ceil(manifest["train_records"] / job["logical_batch_size"]) * job["epochs"]
    if total < count:
        raise ValueError(f"job has {total} steps, but warmup and measurement require {count}")
    job["checkpoint_every_steps"] = total + 1
    return job


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(("finetune_reference.py", "finetune_contract.py", "run_finetune_paired.py")):
        digest.update(relative.encode() + b"\0")
        digest.update(bytes.fromhex(file_sha256(root / relative)))
    return digest.hexdigest()


def observed_shapes(job: dict[str, Any], warmup: int, iterations: int) -> dict[str, Any]:
    """Describe authenticated active lengths rather than relabeling caps."""
    rows = PreparedDataset(Path(job["prepared_dir"]))
    queries, documents, query_counts, candidate_counts = [], [], [], []
    micro = job["microbatch_size"]
    try:
        for step in range(warmup, warmup + iterations):
            batch = [rows[index] for index in batch_indices(len(rows), job["seed"], step, job["logical_batch_size"])]
            q_lengths = [len(record["query_ids"]) for record in batch]
            candidates = {}
            for record in batch:
                document = record["positive"]
                candidates.setdefault(document["content_sha256"], document)
            for record in batch:
                for document in record["negatives"][:job["max_negatives"]]:
                    candidates.setdefault(document["content_sha256"], document)
            d_lengths = [len(document["input_ids"]) for document in candidates.values()]
            queries.extend(max(q_lengths[start:start + micro]) for start in range(0, len(q_lengths), micro))
            documents.extend(max(d_lengths[start:start + micro]) for start in range(0, len(d_lengths), micro))
            query_counts.append(len(q_lengths))
            candidate_counts.append(len(d_lengths))
    finally:
        rows.close()
    def summary(values):
        return {"minimum": min(values), "maximum": max(values), "mean": sum(values) / len(values), "microbatches": len(values)}
    return {"active_query_tokens": summary(queries), "active_document_tokens": summary(documents),
            "observed_query_counts": query_counts, "observed_candidate_counts": candidate_counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--native-source-manifest", type=Path, required=True,
                        help="immutable source manifest recorded when the native binary was built")
    parser.add_argument("--reference-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reference-attention", choices=("sdpa", "eager"), default="sdpa",
                        help="Explicit PyTorch comparison mode; numerical oracle traces use eager separately")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100 or not 1 <= args.warmup <= 100 or not 1 <= args.timeout_seconds <= 7 * 86400:
        parser.error("invalid bounded measurement limits")
    manifest = json.loads(args.manifest.read_text())
    cells = manifest["cells"]
    ids = [cell["id"] for cell in cells]
    if not 1 <= len(cells) <= 7 or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in ids):
        parser.error("manifest requires one to seven unique safe cell IDs")
    qualified_shape = len(cells) == 7 and args.iterations == 100
    jobs = [checked_job(Path(cell["job"]), args.warmup + args.iterations) for cell in cells]
    if qualified_shape:
        validate_performance_cells(jobs)
    workloads = [observed_shapes(job, args.warmup, args.iterations) for job in jobs]
    if qualified_shape:
        for job, shapes in zip(jobs, workloads):
            validate_performance_workload({**shapes, "batch_size": job["logical_batch_size"],
                                           "negatives": job["max_negatives"],
                                           "query_tokens": job["query_max_tokens"],
                                           "document_tokens": job["document_max_tokens"]}, args.iterations)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    binary_digest = file_sha256(args.native_binary)
    native_source_digest = file_sha256(args.native_source_manifest)
    native_source_manifest = json.loads(args.native_source_manifest.read_text())
    if native_source_manifest.get("binary_sha256") != binary_digest:
        raise ValueError("as-built source manifest does not identify the measured native binary")
    reference_digest = source_digest(scripts)
    source_copy = args.output_dir / "native-source-manifest.json"
    source_copy.write_bytes(args.native_source_manifest.read_bytes())
    if file_sha256(source_copy) != native_source_digest:
        raise ValueError("as-built source manifest changed while copying")
    collected: dict[str, dict[str, Any]] = {"native": {}, "reference": {}}
    software: dict[str, Any] = {}
    for cell, job, shapes in zip(cells, jobs, workloads):
        directory = args.output_dir / cell["id"]
        directory.mkdir()
        job["output_dir"] = str(directory / "native-run")
        job["timeout_seconds"] = args.timeout_seconds
        config_path = directory / "job.json"
        config_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        prepared_sha = file_sha256(Path(job["prepared_dir"]) / "manifest.json")
        initial = Path(cell["initial_adapter"]).resolve()
        initial_digest = file_sha256(initial)
        initialization = json.loads((initial.parent / "initialization_report.json").read_text())
        expected_initialization = {"base_weight_sha256": BASE_WEIGHT_SHA256, "seed": job["seed"],
                                   "rank": job["lora"]["rank"], "alpha": job["lora"]["alpha"],
                                   "tensor_count": 338, "adapter_sha256": initial_digest}
        if any(initialization.get(key) != value for key, value in expected_initialization.items()):
            raise ValueError("shared initialization does not match the native job")
        prepared_manifest = json.loads((Path(job["prepared_dir"]) / "manifest.json").read_text())
        total = math.ceil(prepared_manifest["train_records"] / job["logical_batch_size"]) * job["epochs"]
        count = args.warmup + args.iterations
        reference_output = directory / "reference.json"
        native_command = [str(args.native_binary.resolve()), str(config_path), "--stop-after-steps", str(count)]
        # Keep the virtualenv executable path: resolving its symlink changes
        # Python's prefix/site-packages and can silently launch the system env.
        reference_command = [str(args.reference_python.absolute()), str(scripts / "finetune_reference.py"),
            "--model", job["model_dir"], "--prepared-dir", job["prepared_dir"],
            "--initial-adapter", str(initial), "--output", str(reference_output),
            "--steps", str(count), "--total-steps", str(total), "--precision", "bf16", "--device", "cuda",
            "--attention", args.reference_attention,
            "--logical-batch-size", str(job["logical_batch_size"]), "--microbatch-size", str(job["microbatch_size"]),
            "--max-negatives", str(job["max_negatives"]), "--rank", str(job["lora"]["rank"]),
            "--alpha", str(job["lora"]["alpha"]), "--seed", str(job["seed"])]
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        workers: dict[str, Worker] = {}
        timings: dict[str, list[float]] = {"native": [], "reference": []}
        samples: list[dict[str, Any]] = []
        deadline = time.monotonic() + args.timeout_seconds
        try:
            workers["native"] = Worker(native_command, directory / "native", environment)
            workers["reference"] = Worker(reference_command, directory / "reference", environment)
            for step in range(count):
                for worker in workers.values():
                    worker.wait_event("ready", step, deadline)
                order = ("native", "reference") if step % 2 == 0 else ("reference", "native")
                events = {}
                for label in order:
                    workers[label].grant("grant", step)
                    event = workers[label].wait_event("step", step, deadline)
                    if event.get("trace_mode") or not math.isfinite(event["step_ms"]) or event["step_ms"] <= 0 or not math.isfinite(event["loss"]):
                        raise ValueError(f"invalid timed step: {event}")
                    if step >= args.warmup:
                        offset = step - args.warmup
                        if event.get("queries") != shapes["observed_query_counts"][offset] or event.get("candidates") != shapes["observed_candidate_counts"][offset]:
                            raise ValueError("timed query/candidate counts differ from the authenticated workload")
                    events[label] = event
                    if step >= args.warmup:
                        timings[label].append(event["step_ms"])
                for worker in workers.values():
                    worker.grant("continue", step)
                samples.append({"step": step, "warmup": step < args.warmup, "order": order, "events": events})
                print(json.dumps({"cell": cell["id"], "pair": step, "warmup": step < args.warmup,
                                  "native_ms": events["native"]["step_ms"], "reference_ms": events["reference"]["step_ms"]}), flush=True)
            for worker in workers.values():
                worker.finish(deadline)
            native_identity = workers["native"].metadata
            reference_report = json.loads(reference_output.read_text())
            reference_provenance = reference_report["provenance"]
            if not native_identity or native_identity["prepared_manifest_sha256"] != prepared_sha or native_identity["base_weight_sha256"] != BASE_WEIGHT_SHA256:
                raise ValueError("native measured identity mismatch")
            if reference_provenance["prepared_manifest_sha256"] != prepared_sha or reference_provenance["base_weight_sha256"] != BASE_WEIGHT_SHA256:
                raise ValueError("reference measured identity mismatch")
            expected_math = {
                "precision": "bf16", "padding_policy": "per-micro-active-v1",
                "compute_profile": "bf16-edges-fp32-masters-v1",
                "logical_batch_size": job["logical_batch_size"], "microbatch_size": job["microbatch_size"],
                "max_negatives": job["max_negatives"], "rank": job["lora"]["rank"],
                "alpha": job["lora"]["alpha"], "seed": job["seed"],
            }
            if any(native_identity.get(key) != value or reference_provenance.get(key) != value
                   for key, value in expected_math.items()):
                raise ValueError("native/reference measured training math differs from the admitted job")
            if reference_provenance.get("preprocessing_sha256") != prepared_manifest["preprocessing_sha256"]:
                raise ValueError("reference measured preprocessing identity mismatch")
            if reference_provenance.get("attention") != args.reference_attention or reference_provenance.get("loaded_attention") != args.reference_attention:
                raise ValueError("reference did not load the requested attention implementation")
            if native_identity.get("attention") != "eager":
                raise ValueError("native measured attention identity is missing or unsupported")
            common = {**shapes, "batch_size": job["logical_batch_size"], "microbatch_size": job["microbatch_size"],
                      "padding_policy": expected_math["padding_policy"], "compute_profile": expected_math["compute_profile"],
                      "native_attention": "eager", "reference_attention": args.reference_attention,
                      "query_tokens": job["query_max_tokens"], "document_tokens": job["document_max_tokens"],
                      "negatives": job["max_negatives"], "rank": job["lora"]["rank"], "alpha": job["lora"]["alpha"],
                      "dataset_sha256": prepared_sha, "job_fingerprint": native_identity["job_fingerprint"],
                      "initial_adapter_sha256": initial_digest, "trace_mode": False, "memory_measurement": "external_peak",
                      "host_measurement": "process-tree VmHWM upper bound", "device_measurement": "sampled NVML via nvidia-smi",
                      "memory_sample_interval_seconds": 0.1, "warmup_steps": args.warmup,
                      "pairing": "resident-alternating-two-barriers-v1"}
            for label, worker in workers.items():
                if worker.peak_device_bytes <= 0 or worker.peak_host_bytes <= 0:
                    raise ValueError("external memory measurement was unavailable")
                collected[label][cell["id"]] = {**common, "step_ms": timings[label],
                    "peak_device_bytes": worker.peak_device_bytes, "peak_host_bytes": worker.peak_host_bytes}
            software[cell["id"]] = reference_provenance
            (directory / "pairs.json").write_text(json.dumps(samples, indent=2) + "\n")
        finally:
            for worker in workers.values():
                worker.close()
    if file_sha256(args.native_binary) != binary_digest or source_digest(scripts) != reference_digest:
        raise ValueError("measured code changed during collection")
    for label in collected:
        report = {"schema": "antfly.embedding_gemma2.finetune_samples.v1", "cell_contract": ids,
                  "cells": collected[label], "qualification_shape": qualified_shape,
                  "provenance": {"hardware": hardware, "software": software,
                      "source_sha256": native_source_digest if label == "native" else reference_digest,
                      "binary_sha256": binary_digest if label == "native" else None,
                      "base_weight_sha256": BASE_WEIGHT_SHA256,
                      "dataset_sha256": {name: value["dataset_sha256"] for name, value in collected[label].items()},
                      "job_fingerprint": {name: value["job_fingerprint"] for name, value in collected[label].items()}}}
        (args.output_dir / f"{label}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"collected": True, "qualification_shape": qualified_shape, "production_qualified": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
