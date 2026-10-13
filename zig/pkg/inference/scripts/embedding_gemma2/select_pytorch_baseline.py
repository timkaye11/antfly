#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select the fastest parity-valid PyTorch CUDA reference configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

from contract import cosine, render_text, validate_oracle
from pytorch_worker import ATTENTION_MODES, COMPILE_MODES

MIN_COSINE = 0.999
DEFAULT_ATTENTION_MODES = ("eager", "sdpa")


def post(url: str, body: dict, timeout: float) -> tuple[dict, float]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read(64 * 1024).decode('utf-8', errors='replace')
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return payload, time.perf_counter() - started


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def readiness_line(process: subprocess.Popen, timeout: float) -> str:
    if process.stdout is None:
        return ""
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        pending = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError(f"worker did not become ready within {timeout:.0f}s")
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                return pending.decode('utf-8')
            pending.extend(chunk)
            if len(pending) > 64 * 1024:
                raise ValueError('worker readiness line exceeds 64 KiB')
            newline = pending.find(b'\n')
            if newline >= 0:
                return pending[:newline + 1].decode('utf-8')
    finally:
        selector.close()


def failure_status(message: str, timed_out: bool = False) -> tuple[str, bool | None]:
    lower = message.lower()
    if timed_out:
        return "inconclusive_timeout", None
    if "does not support" in lower or "unsupported attention" in lower:
        return "unsupported", False
    if "outofmemory" in lower or "out of memory" in lower or "out of resource" in lower or "hardware limit" in lower:
        return "failed_hardware", True
    return "failed_runtime", True


def selection_complete(results: list[dict], *, include_flex: bool = False) -> bool:
    attention_modes = ATTENTION_MODES if include_flex else DEFAULT_ATTENTION_MODES
    expected = {(attention, mode) for attention in attention_modes for mode in COMPILE_MODES}
    conclusive = {"qualified", "failed_parity", "unsupported", "failed_hardware"}
    return (len(results) == len(expected)
            and {(row['attention'], row['compile']) for row in results} == expected
            and all(row.get('status') in conclusive for row in results))


def write_checkpoint(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def expanded_tokens(processor, text: str, task_type: str) -> int:
    encoded = processor(text=[render_text(text, task_type)], padding=False, truncation=False, return_tensors="pt")
    return int(encoded["attention_mask"].sum())


def exact_length_text(processor, target: int, task_type: str, seed: str = "") -> str:
    if target < expanded_tokens(processor, seed, task_type):
        raise ValueError(f"target {target} is shorter than the task prefix and special tokens")
    low, high = 0, target
    while low <= high:
        count = (low + high) // 2
        text = seed + (" ant" * count)
        actual = expanded_tokens(processor, text, task_type)
        if actual == target:
            return text
        if actual < target:
            low = count + 1
        else:
            high = count - 1
    raise ValueError(f"could not materialize exactly {target} expanded tokens")


def exact_cell_inputs(processor, corpus: str, batch_size: int, target: int, task_type: str) -> list[str]:
    prose = "Ant colonies coordinate through local chemical signals while workers adapt routes to changing conditions. "
    code = "fn cosine(a: []const f32, b: []const f32) f32 { var sum: f32 = 0; for (a, b) |x, y| sum += x * y; return sum; }\n"

    def realistic_exact(length: int, kind: str) -> str:
        unit = code if kind == "source_code" and length >= 128 else prose if kind != "source_code" else "fn"
        low, high = 0, max(1, length)
        while low < high:
            middle = (low + high + 1) // 2
            if expanded_tokens(processor, unit * middle, task_type) <= length:
                low = middle
            else:
                high = middle - 1
        seed = unit * low
        return exact_length_text(processor, length, task_type, seed)

    if corpus != "heterogeneous_padding":
        text = realistic_exact(target, corpus)
        return [text] * batch_size
    minimum = expanded_tokens(processor, "", task_type)
    targets = (target, max(minimum, target // 4), max(minimum, target // 16), max(minimum, target // 64))
    return [realistic_exact(targets[index % len(targets)], "natural_text") for index in range(batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--corpus", choices=("natural_text", "source_code", "heterogeneous_padding"), default="natural_text")
    parser.add_argument("--case", choices=("text", "image", "audio", "mixed_ordered"), default="text")
    parser.add_argument("--task-type", default="RETRIEVAL_DOCUMENT")
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 8), default=1)
    parser.add_argument("--include-flex-attention", action="store_true",
                        help="Include optional FlexAttention diagnostic configurations")
    parser.add_argument("--candidate-url", required=True, help="native endpoint used to validate the exact timed cell")
    parser.add_argument("--candidate-model", default="embeddinggemma-2", help="model name served by the native endpoint")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("iterations must be at least two")

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    validate_oracle(oracle)
    text_cases = [case for case in oracle["cases"] if all(part["type"] == "text" for part in case["parts"])]
    if args.case == "text":
        from transformers import EmbeddingGemma2Processor
        processor = EmbeddingGemma2Processor.from_pretrained(args.model, local_files_only=True)
        cell_inputs = exact_cell_inputs(processor, args.corpus, args.batch_size, args.tokens, args.task_type)
    else:
        from qualify_endpoint import request_input
        media_case = next(case for case in oracle["cases"] if case["id"] == args.case)
        cell_inputs = [request_input(media_case) for _ in range(args.batch_size)]
        args.task_type = media_case["task_type"]
    cell_body = {"model": args.candidate_model, "input": cell_inputs, "task_type": args.task_type}
    candidate_payload, _ = post(args.candidate_url, cell_body, args.timeout)
    if candidate_payload.get("backend") != "cuda":
        raise RuntimeError(f"candidate endpoint did not prove CUDA backend: {candidate_payload.get('backend')!r}")
    candidate_vectors = [[float(value) for value in row["embedding"]] for row in sorted(candidate_payload["data"], key=lambda row: row["index"])]
    worker_script = Path(__file__).with_name("pytorch_worker.py")
    url = f"http://127.0.0.1:{args.port}/v1/embeddings"
    results = []
    attention_modes = ATTENTION_MODES if args.include_flex_attention else DEFAULT_ATTENTION_MODES
    for attention in attention_modes:
        for compile_mode in COMPILE_MODES:
            print(json.dumps({'event': 'configuration_begin', 'attention': attention, 'compile': compile_mode}), file=sys.stderr, flush=True)
            command = [
                str(args.python), str(worker_script), "--model", str(args.model), "--port", str(args.port),
                "--attention", attention, "--compile", compile_mode, "--warmup", str(args.warmup),
                "--torch-threads", str(args.torch_threads),
            ]
            with tempfile.NamedTemporaryFile(prefix=f"embeddinggemma2-{attention}-{compile_mode}-", suffix=".log", delete=False) as stderr_file:
                stderr_path = Path(stderr_file.name)
                process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=stderr_file, start_new_session=True)
            try:
                line = readiness_line(process, args.timeout)
                if not line:
                    stderr = stderr_path.read_text(errors="replace")
                    status, supported = failure_status(stderr)
                    results.append({"attention": attention, "compile": compile_mode, "status": status, "supported": supported, "error": stderr[-4000:] or "worker exited without readiness output", "stderr_log": str(stderr_path)})
                    continue
                ready = json.loads(line)
                similarities = []
                for case in text_cases:
                    body = {"input": case["parts"][0]["text"], "task_type": case["task_type"]}
                    payload, _ = post(url, body, args.timeout)
                    similarities.append(cosine(payload["data"][0]["embedding"], case["embeddings"]["768"]))
                valid = min(similarities) >= MIN_COSINE
                cell_payload, _ = post(url, cell_body, args.timeout)
                cell_vectors = [[float(value) for value in row["embedding"]] for row in sorted(cell_payload["data"], key=lambda row: row["index"])]
                finite = len(cell_vectors) == args.batch_size and all(math.isfinite(value) for vector in cell_vectors for value in vector)
                paired_cosines = [cosine(vector, candidate) for vector, candidate in zip(cell_vectors, candidate_vectors)] if candidate_vectors is not None and len(candidate_vectors) == len(cell_vectors) else []
                cell_parity_valid = finite and bool(paired_cosines) and min(paired_cosines) >= MIN_COSINE
                timings = []
                if valid and cell_parity_valid:
                    for _ in range(args.warmup):
                        post(url, cell_body, args.timeout)
                    for _ in range(args.iterations):
                        _, elapsed = post(url, cell_body, args.timeout)
                        timings.append(elapsed)
                results.append({
                    "attention": attention,
                    "compile": compile_mode,
                    "supported": True,
                    "status": "qualified" if valid and cell_parity_valid else "failed_parity",
                    "parity_valid": valid,
                    "minimum_text_oracle_cosine": min(similarities),
                    "paired_cell_finite": finite,
                    "paired_cell_minimum_cosine": min(paired_cosines) if paired_cosines else None,
                    "paired_cell_parity_valid": cell_parity_valid,
                    "latency_ms": {"p50": percentile(timings, 0.5) * 1000, "p95": percentile(timings, 0.95) * 1000} if timings else None,
                    "runtime": cell_payload.get("runtime", ready.get("runtime")),
                    "warmup": ready.get("warmup"),
                    "stderr_log": str(stderr_path),
                })
            except Exception as exc:
                excerpt = str(exc)
                if stderr_path.exists():
                    excerpt = (excerpt + "\n" + stderr_path.read_text(errors="replace")[-4000:]).strip()
                status, supported = failure_status(excerpt, isinstance(exc, TimeoutError))
                results.append({"attention": attention, "compile": compile_mode, "status": status, "supported": supported, "error": excerpt, "stderr_log": str(stderr_path)})
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                print(json.dumps({'event': 'configuration_end', **results[-1]}), file=sys.stderr, flush=True)
                if args.report:
                    write_checkpoint(args.report, {'schema': 'antfly.embedding_gemma2.pytorch_baseline_selection.v1',
                        'pass': False, 'complete': False, 'configurations': results,
                        'attention_modes': list(attention_modes),
                        'include_flex_attention': args.include_flex_attention})

    valid = [row for row in results if row.get("parity_valid") and row.get("paired_cell_parity_valid") and row.get("latency_ms")]
    selected = min(valid, key=lambda row: row["latency_ms"]["p50"]) if valid else None
    complete = selection_complete(results, include_flex=args.include_flex_attention)
    report = {"schema": "antfly.embedding_gemma2.pytorch_baseline_selection.v1", "pass": selected is not None and complete, "complete": complete, "minimum_cosine": MIN_COSINE, "workload": {"case": args.case, "batch_size": args.batch_size, "expanded_tokens": args.tokens if args.case == "text" else None, "corpus": args.corpus if args.case == "text" else None, "task_type": args.task_type, "cell_warmup": args.warmup}, "candidate_url": args.candidate_url, "candidate_model": args.candidate_model, "candidate_backend": candidate_payload.get("backend"), "selected": selected, "configurations": results}
    report.update(attention_modes=list(attention_modes), include_flex_attention=args.include_flex_attention)
    if args.report:
        write_checkpoint(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['pass'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
