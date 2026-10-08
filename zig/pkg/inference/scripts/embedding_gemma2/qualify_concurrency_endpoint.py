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

"""Check concurrent resident requests and indexed partial failures on CUDA."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from contract import cosine
from oracle import cases
from qualify_endpoint import request_input


def post(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def valid_vector(vector: list[float]) -> bool:
    return len(vector) == 768 and all(math.isfinite(value) for value in vector) and abs(math.sqrt(sum(value * value for value in vector)) - 1) <= 1e-3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.rounds < 1:
        parser.error("workers and rounds must be positive")

    bodies = [{"model": args.model, "input": request_input(case), "task_type": case["task_type"]} for case in cases()]
    expected = []
    for body in bodies:
        status, payload = post(args.url, body, args.timeout)
        if status != 200 or payload.get("backend") != "cuda":
            raise RuntimeError(f"baseline request failed: {status} {payload}")
        vector = payload["data"][0]["embedding"]
        if not valid_vector(vector):
            raise RuntimeError("baseline request returned an invalid embedding")
        expected.append(vector)

    def run(index: int) -> dict:
        case_index = index % len(bodies)
        deadline = time.monotonic() + args.timeout
        retries = 0
        retry_codes = {}
        while True:
            status, payload = post(args.url, bodies[case_index], args.timeout)
            if status not in (429, 503) or time.monotonic() >= deadline:
                break
            retries += 1
            code = str(payload.get("error", "unknown"))
            retry_codes[code] = retry_codes.get(code, 0) + 1
            # Honor the server's capacity backoff instead of flooding the
            # admission path while a long media request holds the session.
            time.sleep(min(1.0, max(0.01, float(payload.get("retry_after_ms", 50)) / 1000)))
        rows = payload.get("data", [])
        vector = rows[0].get("embedding", []) if len(rows) == 1 else []
        similarity = cosine(vector, expected[case_index]) if valid_vector(vector) else -1
        return {"gate": "concurrent_same_session", "request": index, "case": cases()[case_index]["id"],
                "pass": status == 200 and payload.get("backend") == "cuda" and similarity >= .9999,
                "status": status, "cosine": similarity, "capacity_retries": retries, "retry_codes": retry_codes}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        gates = list(pool.map(run, range(args.rounds * len(bodies))))

    bad_image = {"type": "media", "mime_type": "image/png", "data": base64.b64encode(b"not a png").decode()}
    bad_audio = {"type": "media", "mime_type": "audio/wav", "data": base64.b64encode(b"not a wav").decode()}
    # Two corrupt images in the same group must produce one indexed failure.
    inputs = ["first valid sibling", {"content": [bad_image, bad_image]},
              {"content": [bad_audio]}, "missing media <|image|>", "last valid sibling"]
    status, partial = post(args.url, {"model": args.model, "input": inputs, "error_policy": "per_item"}, args.timeout)
    data = sorted(partial.get("data", []), key=lambda row: row["index"])
    errors = sorted(partial.get("errors", []), key=lambda row: row["index"])
    gates.append({"gate": "partial_sibling_preservation", "pass": status == 200 and [row["index"] for row in data] == [0, 4]
                  and all(valid_vector(row.get("embedding", [])) for row in data) and partial.get("backend") == "cuda",
                  "status": status, "success_indexes": [row["index"] for row in data],
                  "request_error": partial.get("error"), "message": partial.get("message")})
    gates.append({"gate": "partial_error_deduplication", "pass": [(row["index"], row["code"], row["status"]) for row in errors]
                  == [(1, "INVALID_IMAGE", 400), (2, "INVALID_AUDIO", 400), (3, "INVALID_MEDIA_LAYOUT", 400)],
                  "errors": errors, "summary": partial.get("summary")})
    report = {"schema": "antfly.embedding_gemma2.concurrency.v1", "pass": all(row["pass"] for row in gates),
              "workers": args.workers, "rounds": args.rounds, "gates": gates}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
