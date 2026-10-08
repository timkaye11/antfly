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

"""Gate a running Antfly EmbeddingGemma 2 endpoint against an oracle."""

from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import math
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse

from contract import TRAINED_DIMENSIONS, cosine, truncate_and_normalize, validate_oracle
from oracle import fixture_audio, fixture_image, float32_wav

SCHEMA = "antfly.embedding_gemma2.cuda_qualification.v1"
MIN_COSINE = 0.999
BATCH_MIN_COSINE = 0.9999
MRL_MIN_COSINE = 0.99999


def data_uris() -> tuple[str, str]:
    image_buffer = io.BytesIO()
    fixture_image().save(image_buffer, format="PNG")
    return (
        "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode(),
        "data:audio/wav;base64," + base64.b64encode(float32_wav(fixture_audio())).decode(),
    )


def request_input(case: dict) -> str | dict:
    if len(case["parts"]) == 1 and case["parts"][0]["type"] == "text":
        return case["parts"][0]["text"]
    image_uri, audio_uri = data_uris()
    content = []
    for part in case["parts"]:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        elif part["type"] == "audio":
            content.append({"type": "media", "mime_type": "audio/wav", "data": audio_uri.split(",", 1)[1]})
    return {"content": content}


def post(url: str, body: dict, timeout: float) -> tuple[list[list[float]], dict]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    rows = sorted(payload["data"], key=lambda row: row["index"])
    return [[float(value) for value in row["embedding"]] for row in rows], payload


def expect_http_error(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    raise RuntimeError("request unexpectedly succeeded")


def cancel_request(url: str, body: dict, timeout: float) -> None:
    parsed = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    payload = json.dumps(body).encode()
    connection.request("POST", parsed.path, payload, {"Content-Type": "application/json", "Content-Length": str(len(payload))})
    connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="google/embeddinggemma-2:bf16-safetensors-bundle-v1")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--exercise-cancellation", action="store_true")
    args = parser.parse_args(argv)
    if args.backend != "cuda":
        parser.error("production qualification requires --backend cuda")
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    validate_oracle(oracle)
    gates = []
    full_vectors = {}
    for case in oracle["cases"]:
        body = {"model": args.model, "input": request_input(case), "task_type": case["task_type"]}
        vectors, response = post(args.url, body, args.timeout)
        vector = vectors[0]
        full_vectors[case["id"]] = vector
        expected = case["embeddings"]["768"]
        similarity = cosine(vector, expected) if len(vector) == len(expected) else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        gates.extend([
            {"gate": "oracle_cosine", "case": case["id"], "pass": similarity >= MIN_COSINE, "value": similarity, "threshold": MIN_COSINE},
            {"gate": "unit_norm", "case": case["id"], "pass": abs(norm - 1.0) <= 1e-3, "value": norm},
            {"gate": "backend_provenance", "case": case["id"], "pass": response.get("backend") == "cuda", "value": response.get("backend")},
        ])
        for dimensions in TRAINED_DIMENSIONS[:-1]:
            reduced, _ = post(args.url, {**body, "dimensions": dimensions}, args.timeout)
            expected_reduced = case["embeddings"][str(dimensions)]
            oracle_similarity = cosine(reduced[0], expected_reduced) if len(reduced[0]) == dimensions else -1.0
            self_similarity = cosine(reduced[0], truncate_and_normalize(vector, dimensions)) if len(reduced[0]) == dimensions else -1.0
            gates.extend([
                {"gate": "mrl_oracle_cosine", "case": f"{case['id']}:{dimensions}", "pass": oracle_similarity >= MIN_COSINE, "value": oracle_similarity},
                {"gate": "truncate_renormalize", "case": f"{case['id']}:{dimensions}", "pass": self_similarity >= MRL_MIN_COSINE, "value": self_similarity},
            ])

    text_cases = [case for case in oracle["cases"] if all(part["type"] == "text" for part in case["parts"])]
    for case in text_cases:
        text = case["parts"][0]["text"]
        batch_body = {"model": args.model, "input": [text, text], "task_type": case["task_type"]}
        batch, _ = post(args.url, batch_body, args.timeout)
        for index, vector in enumerate(batch):
            similarity = cosine(vector, full_vectors[case["id"]])
            gates.append({"gate": "batch_single_cosine", "case": f"{case['id']}:{index}", "pass": similarity >= BATCH_MIN_COSINE, "value": similarity})

    multimodal_cases = [case for case in oracle["cases"] if any(part["type"] != "text" for part in case["parts"])]
    for case in multimodal_cases:
        grouped = request_input(case)
        batch_body = {"model": args.model, "input": [grouped, grouped], "task_type": case["task_type"]}
        batch, _ = post(args.url, batch_body, args.timeout)
        for index, vector in enumerate(batch):
            similarity = cosine(vector, full_vectors[case["id"]])
            gates.append({"gate": "mixed_batch_single_cosine", "case": f"{case['id']}:{index}", "pass": similarity >= BATCH_MIN_COSINE, "value": similarity})

    negative_bodies = (
        ("oversize", {"model": args.model, "input": " ".join(["ant"] * 8193)}, "INPUT_TOO_LONG"),
        # Fail-fast top-level parse failures use the endpoint's generic request
        # contract. Per-item mode reports the narrower INVALID_MEDIA code.
        ("malformed_media", {"model": args.model, "input": {"content": [{"type": "media", "mime_type": "audio/wav", "data": "%%%"}]}}, "INVALID_REQUEST"),
        ("unknown_task", {"model": args.model, "input": "ants", "task_type": "CUSTOM"}, "INVALID_REQUEST"),
        ("dimensions_zero", {"model": args.model, "input": "ants", "dimensions": 0}, "INVALID_REQUEST"),
        ("dimensions_769", {"model": args.model, "input": "ants", "dimensions": 769}, "INVALID_REQUEST"),
    )
    for case_id, body, expected_code in negative_bodies:
        status, response = expect_http_error(args.url, body, args.timeout)
        gates.append({"gate": "negative_contract", "case": case_id, "pass": status == 400 and response.get("error") == expected_code, "status": status, "value": response.get("error"), "expected": expected_code})

    if args.exercise_cancellation:
        cancel_request(args.url, {"model": args.model, "input": " ".join(["ant"] * 4096)}, args.timeout)
        reused, response = post(args.url, {"model": args.model, "input": "session reuse after cancellation", "task_type": "RETRIEVAL_DOCUMENT"}, args.timeout)
        finite = len(reused) == 1 and all(math.isfinite(value) for value in reused[0])
        gates.append({"gate": "cancel_then_reuse", "case": "resident_session", "pass": finite and response.get("backend") == "cuda", "backend": response.get("backend")})

    report = {"schema": SCHEMA, "pass": all(row["pass"] for row in gates), "model": args.model, "backend": args.backend, "oracle_runtime": oracle["runtime"], "gates": gates}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
