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

"""Standalone CUDA boundary smoke for large heterogeneous batches."""

import argparse
import json
import math
from pathlib import Path
import urllib.request
import urllib.error

from contract import cosine
from select_pytorch_baseline import exact_cell_inputs, expanded_tokens

SCHEMA = "antfly.embedding_gemma2.large_batch.v1"


def post(url, body, timeout):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_vectors(payload, expected_count):
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise RuntimeError(f"expected {expected_count} embedding rows, received {len(data) if isinstance(data, list) else 'invalid data'}")
    indices = [row.get("index") for row in data]
    if sorted(indices) != list(range(expected_count)) or len(set(indices)) != expected_count:
        raise RuntimeError(f"embedding indices are not exactly 0..{expected_count - 1}: {indices}")
    vectors = []
    for row in sorted(data, key=lambda item: item["index"]):
        vector = row.get("embedding")
        if not isinstance(vector, list) or len(vector) != 768:
            raise RuntimeError(f"embedding {row['index']} width is {len(vector) if isinstance(vector, list) else 'invalid'}, expected 768")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise RuntimeError(f"embedding {row['index']} contains a non-finite or non-numeric value")
        vectors.append(vector)
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--processor-dir", type=Path, required=True); parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from transformers import EmbeddingGemma2Processor
    processor = EmbeddingGemma2Processor.from_pretrained(args.processor_dir, local_files_only=True)
    rows = []
    write_report(args.report, {"schema": SCHEMA, "status": "running", "completed_cases": 0, "rows": rows})
    for batch_size in (8, 32):
        try:
            inputs = exact_cell_inputs(processor, "heterogeneous_padding", batch_size, 8192, "RETRIEVAL_DOCUMENT")
            lengths = [expanded_tokens(processor, value, "RETRIEVAL_DOCUMENT") for value in inputs]
            body = {"model": args.model, "input": inputs, "task_type": "RETRIEVAL_DOCUMENT"}
            batched = post(args.url, body, args.timeout)
            vectors = checked_vectors(batched, batch_size)
            singles = [checked_vectors(post(args.url, {**body, "input": value}, args.timeout), 1)[0] for value in inputs]
            similarities = [cosine(left, right) for left, right in zip(vectors, singles)]
            norms = [math.sqrt(sum(value*value for value in vector)) for vector in vectors]
            rows.append({"batch_size": batch_size, "token_lengths": lengths, "usage": batched.get("usage"),
                         "backend": batched.get("backend"), "embedding_rows": len(vectors), "embedding_width": 768,
                         "minimum_batch_single_cosine": min(similarities),
                         "maximum_norm_error": max(abs(norm-1) for norm in norms),
                         "pass": batched.get("backend") == "cuda" and batched.get("usage", {}).get("prompt_tokens") == sum(lengths) and min(similarities) >= .9999 and max(abs(norm-1) for norm in norms) <= 1e-3})
            write_report(args.report, {"schema": SCHEMA, "status": "running", "completed_cases": len(rows), "rows": rows})
        except Exception as exc:
            report = {"schema": SCHEMA, "status": "failed", "pass": False, "failed_batch_size": batch_size,
                      "error_type": type(exc).__name__, "error": str(exc), "rows": rows}
            write_report(args.report, report)
            print(json.dumps(report, sort_keys=True))
            return 1
    report = {"schema": SCHEMA, "status": "complete", "pass": all(row["pass"] for row in rows), "rows": rows}
    write_report(args.report, report); print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
