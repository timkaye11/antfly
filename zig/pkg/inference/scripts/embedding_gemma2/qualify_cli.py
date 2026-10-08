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

"""Gate the local Antfly CLI against a pinned EmbeddingGemma 2 oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from contract import TRAINED_DIMENSIONS, cosine, truncate_and_normalize, validate_oracle
from qualify_endpoint import request_input

MIN_COSINE = 0.999


def run_cli(command: list[str], timeout: float) -> dict:
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-4000:] or completed.stdout.strip()[-4000:]
        raise RuntimeError(f"CLI failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    validate_oracle(oracle)
    gates = []
    with tempfile.TemporaryDirectory(prefix="embeddinggemma2-cli-") as directory:
        directory = Path(directory)
        for case in oracle["cases"]:
            body = {"input": request_input(case), "task_type": case["task_type"]}
            request_path = directory / f"{case['id']}.json"
            request_path.write_text(json.dumps(body), encoding="utf-8")
            base_command = [str(args.binary), "embed", str(args.model_dir), "--backend", args.backend, "--content-json", str(request_path)]
            response = run_cli(base_command, args.timeout)
            vectors = response.get("embeddings", [])
            vector = [float(value) for value in vectors[0]] if len(vectors) == 1 else []
            expected = case["embeddings"]["768"]
            similarity = cosine(vector, expected) if len(vector) == len(expected) else -1.0
            norm = math.sqrt(sum(value * value for value in vector))
            gates.extend((
                {"gate": "oracle_cosine", "case": case["id"], "pass": similarity >= MIN_COSINE, "value": similarity, "threshold": MIN_COSINE},
                {"gate": "unit_norm", "case": case["id"], "pass": abs(norm - 1.0) <= 1e-3, "value": norm},
                {"gate": "backend_provenance", "case": case["id"], "pass": response.get("backend") == args.backend, "value": response.get("backend")},
                {"gate": "expanded_token_usage", "case": case["id"], "pass": response.get("usage", {}).get("prompt_tokens") == case["expanded_tokens"], "value": response.get("usage", {}).get("prompt_tokens"), "expected": case["expanded_tokens"]},
            ))
            for dimensions in TRAINED_DIMENSIONS[:-1]:
                reduced_response = run_cli(base_command + ["--dimensions", str(dimensions)], args.timeout)
                reduced_rows = reduced_response.get("embeddings", [])
                reduced = [float(value) for value in reduced_rows[0]] if len(reduced_rows) == 1 else []
                expected_reduced = case["embeddings"][str(dimensions)]
                oracle_similarity = cosine(reduced, expected_reduced) if len(reduced) == dimensions else -1.0
                self_similarity = cosine(reduced, truncate_and_normalize(vector, dimensions)) if len(reduced) == dimensions else -1.0
                gates.extend((
                    {"gate": "mrl_oracle_cosine", "case": f"{case['id']}:{dimensions}", "pass": oracle_similarity >= MIN_COSINE, "value": oracle_similarity, "threshold": MIN_COSINE},
                    {"gate": "truncate_renormalize", "case": f"{case['id']}:{dimensions}", "pass": self_similarity >= 0.99999, "value": self_similarity, "threshold": 0.99999},
                ))

    report = {
        "schema": "antfly.embedding_gemma2.cli_qualification.v1",
        "pass": all(gate["pass"] for gate in gates),
        "backend": args.backend,
        "gates": gates,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
