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

"""Paired/interleaved Antfly-versus-PyTorch HTTP benchmark with CI gates."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
import urllib.request

from qualify_endpoint import request_input
from oracle import cases
from select_pytorch_baseline import exact_cell_inputs

SCHEMA = "antfly.embedding_gemma2.paired_benchmark.v1"
MIN_THROUGHPUT_RATIO_CI95 = 0.90
MAX_P95_LATENCY_RATIO = 1.10


def post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    if not payload.get("data"):
        raise RuntimeError(f"endpoint returned no embeddings: {url}")
    return payload


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def bootstrap_ratio(candidate: list[float], reference: list[float], samples: int, seed: int) -> tuple[float, float]:
    if len(candidate) != len(reference):
        raise ValueError("paired bootstrap requires equal sample counts")
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        indexes = [rng.randrange(len(candidate)) for _ in candidate]
        c = statistics.mean(candidate[index] for index in indexes)
        r = statistics.mean(reference[index] for index in indexes)
        ratios.append(r / c)  # inverse latency is throughput
    ratios.sort()
    return ratios[int(samples * 0.025)], ratios[min(samples - 1, int(samples * 0.975))]


def corpus_inputs(corpus: str, batch_size: int, tokens: int) -> list[str]:
    if corpus == "natural_text":
        text = " ".join(["ant"] * tokens)
        return [text] * batch_size
    if corpus == "source_code":
        unit = "fn cosine(a: []f32, b: []f32) f32 { return dot(a, b); }"
        text = "\n".join([unit] * max(1, tokens // 20))
        return [text] * batch_size
    if corpus == "heterogeneous_padding":
        # Keep the maximum row fixed while spreading rows over short, medium,
        # and long lengths to expose padding-sensitive batch regressions.
        fractions = (1, 4, 16, 64)
        return [" ".join(["ant"] * max(1, tokens // fractions[index % len(fractions)])) for index in range(batch_size)]
    raise ValueError(f"unsupported corpus: {corpus}")


def evaluate_gates(ci_low: float, p95_ratio: float) -> bool:
    return ci_low >= MIN_THROUGHPUT_RATIO_CI95 and p95_ratio <= MAX_P95_LATENCY_RATIO


def vectors(payload: dict) -> list[list[float]]:
    rows = [[float(value) for value in row["embedding"]] for row in sorted(payload["data"], key=lambda row: row["index"])]
    if not rows or any(not row or not all(math.isfinite(value) for value in row) for row in rows):
        raise RuntimeError("endpoint returned empty or non-finite embeddings")
    return rows


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise RuntimeError(f"embedding width mismatch: {len(a)} != {len(b)}")
    aa, bb = sum(x*x for x in a), sum(x*x for x in b)
    if aa <= 0 or bb <= 0:
        raise RuntimeError("endpoint returned a zero-norm embedding")
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(aa*bb)


def write_report(path: Path, report: dict) -> None:
    """Atomically retain the latest complete benchmark checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antfly-url", required=True)
    parser.add_argument("--pytorch-url", required=True)
    parser.add_argument("--model", default="google/embeddinggemma-2:bf16-safetensors-bundle-v1")
    parser.add_argument("--processor-dir", type=Path, help="pinned local processor; required for exact text token counts")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--corpus", choices=("natural_text", "source_code", "heterogeneous_padding"), default="natural_text")
    parser.add_argument("--case", choices=("text", "image", "audio", "mixed_ordered"), default="text", help="official deterministic fixture; text uses --corpus")
    parser.add_argument("--task-type", default="RETRIEVAL_DOCUMENT")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--exploratory", action="store_true", help="allow fewer than 100 measured iterations")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.tokens < 1 or args.iterations < 2:
        parser.error("batch size and tokens must be positive; iterations must be at least two")
    if args.iterations < 100 and not args.exploratory:
        parser.error("qualification requires at least 100 iterations (use --exploratory for diagnostics)")
    if args.case == "text":
        if args.processor_dir is None:
            parser.error("--processor-dir is required for exact text cells")
        from transformers import EmbeddingGemma2Processor
        processor = EmbeddingGemma2Processor.from_pretrained(args.processor_dir, local_files_only=True)
        inputs = exact_cell_inputs(processor, args.corpus, args.batch_size, args.tokens, args.task_type)
        from select_pytorch_baseline import expanded_tokens
        expected_token_lengths = [expanded_tokens(processor, value, args.task_type) for value in inputs]
    else:
        case = next(item for item in cases() if item["id"] == args.case)
        inputs = [request_input(case) for _ in range(args.batch_size)]
        args.task_type = case["task_type"]
        expected_token_lengths = None
    body = {"model": args.model, "input": inputs, "task_type": args.task_type}
    candidate_check = post(args.antfly_url, body, args.timeout)
    reference_check = post(args.pytorch_url, body, args.timeout)
    if candidate_check.get("backend") != "cuda":
        raise RuntimeError(f"Antfly endpoint did not prove CUDA backend: {candidate_check.get('backend')!r}")
    if expected_token_lengths is not None:
        actual_tokens = candidate_check.get("usage", {}).get("prompt_tokens")
        if actual_tokens != sum(expected_token_lengths):
            raise RuntimeError(f"native prompt token usage {actual_tokens!r} != expected {sum(expected_token_lengths)}")
    candidate_vectors, reference_vectors = vectors(candidate_check), vectors(reference_check)
    if len(candidate_vectors) != args.batch_size or len(reference_vectors) != args.batch_size:
        raise RuntimeError("endpoint embedding count does not match requested batch size")
    similarities = [cosine(a, b) for a, b in zip(candidate_vectors, reference_vectors)]
    if min(similarities) < 0.999:
        raise RuntimeError(f"paired-cell cosine below 0.999: {min(similarities):.9f}")
    runtime = {}
    for _ in range(args.warmup):
        runtime["antfly"] = post(args.antfly_url, body, args.timeout).get("runtime")
        runtime["pytorch"] = post(args.pytorch_url, body, args.timeout).get("runtime")
    timings = {"antfly": [], "pytorch": []}
    endpoints = {"antfly": args.antfly_url, "pytorch": args.pytorch_url}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(args.iterations):
        order = ("antfly", "pytorch") if iteration % 2 == 0 else ("pytorch", "antfly")
        for name in order:
            started = time.perf_counter()
            payload = post(endpoints[name], body, args.timeout)
            runtime[name] = payload.get("runtime", runtime.get(name))
            timings[name].append(time.perf_counter() - started)
        if args.report:
            write_report(args.report, {
                "schema": SCHEMA, "status": "running", "completed_pairs": iteration + 1,
                "requested_pairs": args.iterations,
                "workload": {"case": args.case, "batch_size": args.batch_size,
                    "expanded_tokens": args.tokens if args.case == "text" else None,
                    "expected_token_lengths": expected_token_lengths,
                    "corpus": args.corpus if args.case == "text" else None,
                    "task_type": args.task_type},
                "correctness": {"minimum_cosine": min(similarities),
                    "embedding_width": len(candidate_vectors[0]),
                    "backend": candidate_check.get("backend"),
                    "usage": candidate_check.get("usage")},
                "runtime": runtime,
                "raw_latency_ms": {name: [value * 1000 for value in values] for name, values in timings.items()},
            })
    ci_low, ci_high = bootstrap_ratio(timings["antfly"], timings["pytorch"], args.bootstrap_samples, args.seed)
    p95_ratio = percentile(timings["antfly"], 0.95) / percentile(timings["pytorch"], 0.95)
    report = {
        "schema": SCHEMA,
        "pass": evaluate_gates(ci_low, p95_ratio),
        "workload": {"case": args.case, "batch_size": args.batch_size, "expanded_tokens": args.tokens if args.case == "text" else None, "expected_token_lengths": expected_token_lengths, "corpus": args.corpus if args.case == "text" else None, "task_type": args.task_type, "warmup": args.warmup, "iterations": args.iterations},
        "runtime": runtime,
        "correctness": {"minimum_cosine": min(similarities), "embedding_width": len(candidate_vectors[0]), "backend": candidate_check.get("backend"), "usage": candidate_check.get("usage")},
        "throughput_ratio_ci95": [ci_low, ci_high],
        "p95_latency_ratio": p95_ratio,
        "latency_ms": {name: {"p50": percentile(values, 0.5) * 1000, "p95": percentile(values, 0.95) * 1000, "p99": percentile(values, 0.99) * 1000} for name, values in timings.items()},
        "raw_latency_ms": {name: [value * 1000 for value in values] for name, values in timings.items()},
        "gates": {"throughput_ratio_lower_95": MIN_THROUGHPUT_RATIO_CI95, "p95_latency_ratio_max": MAX_P95_LATENCY_RATIO},
    }
    if args.report:
        write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
