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

"""Exercise EmbeddingGemma 2 HTTP rejection and cancellation contracts."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import socket
import struct
import time
import urllib.error
import urllib.parse
import urllib.request

from select_pytorch_baseline import exact_length_text

SCHEMA = "antfly.embedding_gemma2.endpoint_contract.v1"


def request(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        json.dumps(body, allow_nan=False).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def float_wav_with_nan() -> bytes:
    """Return a valid mono IEEE-float WAV whose only sample is nonfinite."""
    sample = struct.pack("<f", math.nan)
    fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(sample)) + sample
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def media_input(mime_type: str, payload: bytes) -> dict:
    return {
        "content": [{
            "type": "media",
            "mime_type": mime_type,
            "data": base64.b64encode(payload).decode(),
        }]
    }


def abort_in_flight(url: str, body: dict, delay: float, timeout: float) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("cancellation qualification requires an http:// endpoint")
    payload = json.dumps(body, allow_nan=False).encode()
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    header = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=timeout) as stream:
        stream.sendall(header + payload)
        time.sleep(delay)
        # An RST makes disconnect observation deterministic; a graceful FIN can
        # allow the server to finish and write a response into the socket buffer.
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def gate(case: str, status: int, payload: dict, expected_status: int, expected_code: str) -> dict:
    actual_code = payload.get("error")
    return {
        "gate": "negative_contract",
        "case": case,
        "pass": status == expected_status and actual_code == expected_code,
        "status": status,
        "expected_status": expected_status,
        "value": actual_code,
        "expected": expected_code,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="google/embeddinggemma-2:bf16-safetensors-bundle-v1")
    parser.add_argument("--processor-dir", type=Path, required=True,
                        help="Pinned Hugging Face processor used to materialize exact 8192/8193-token requests")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cancel-delay", type=float, default=0.05)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    accepted_text = exact_length_text(processor, 8192, "RETRIEVAL_DOCUMENT")
    rejected_text = exact_length_text(processor, 8193, "RETRIEVAL_DOCUMENT")
    base = {"model": args.model, "task_type": "RETRIEVAL_DOCUMENT"}
    gates: list[dict] = []

    status, payload = request(args.url, {**base, "input": accepted_text}, args.timeout)
    prompt_tokens = payload.get("usage", {}).get("prompt_tokens")
    rows = payload.get("data", [])
    accepted_finite = (
        status == 200
        and prompt_tokens == 8192
        and len(rows) == 1
        and len(rows[0].get("embedding", [])) == 768
        and all(math.isfinite(float(value)) for value in rows[0]["embedding"])
    )
    gates.append({"gate": "expanded_token_boundary", "case": "8192", "pass": accepted_finite,
                  "status": status, "prompt_tokens": prompt_tokens})

    negative_cases = [
        ("expanded_8193", {**base, "input": rejected_text}, 400, "INPUT_TOO_LONG"),
        ("dimensions_zero", {**base, "input": "ants", "dimensions": 0}, 400, "INVALID_REQUEST"),
        ("dimensions_769", {**base, "input": "ants", "dimensions": 769}, 400, "INVALID_REQUEST"),
        ("unknown_task", {"model": args.model, "input": "ants", "task_type": "CUSTOM"}, 400, "INVALID_REQUEST"),
        ("text_only_image_placeholder", {**base, "input": "ants <|image|>"}, 400, "INVALID_MEDIA_LAYOUT"),
        ("text_only_audio_placeholder", {**base, "input": "ants <|audio|>"}, 400, "INVALID_MEDIA_LAYOUT"),
        ("text_only_video_placeholder", {**base, "input": "ants <|video|>"}, 400, "INVALID_MEDIA_LAYOUT"),
        ("empty_content", {**base, "input": {"content": []}}, 400, "INVALID_REQUEST"),
        ("nested_content", {**base, "input": {"content": [[{"type": "text", "text": "ants"}]]}}, 400, "INVALID_REQUEST"),
        ("malformed_image", {**base, "input": media_input("image/png", b"not a png")}, 400, "INVALID_IMAGE"),
        ("malformed_audio", {**base, "input": media_input("audio/wav", b"not a wav")}, 400, "INVALID_AUDIO"),
        ("nonfinite_pcm", {**base, "input": media_input("audio/wav", float_wav_with_nan())}, 400, "INVALID_AUDIO"),
    ]
    for case_id, body, expected_status, expected_code in negative_cases:
        status, payload = request(args.url, body, args.timeout)
        gates.append(gate(case_id, status, payload, expected_status, expected_code))

    abort_in_flight(args.url, {**base, "input": accepted_text}, args.cancel_delay, args.timeout)
    # The reset is asynchronous at the server. Reuse must eventually succeed
    # on the same resident model without a restart or corrupt output.
    reuse_status = 0
    reuse_payload: dict = {}
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        reuse_status, reuse_payload = request(args.url, {**base, "input": "session reuse after cancellation"}, args.timeout)
        if reuse_status == 200:
            break
        if reuse_status not in (429, 503):
            break
        time.sleep(0.05)
    reuse_rows = reuse_payload.get("data", [])
    vector = reuse_rows[0].get("embedding", []) if len(reuse_rows) == 1 else []
    norm = math.sqrt(sum(float(value) ** 2 for value in vector)) if vector else 0.0
    gates.append({
        "gate": "cancel_then_reuse",
        "case": "resident_session",
        "pass": reuse_status == 200 and len(vector) == 768
        and all(math.isfinite(float(value)) for value in vector)
        and abs(norm - 1.0) <= 1e-3
        and reuse_payload.get("backend") == "cuda",
        "status": reuse_status,
        "backend": reuse_payload.get("backend"),
        "norm": norm,
    })

    report = {"schema": SCHEMA, "pass": all(row["pass"] for row in gates), "model": args.model, "gates": gates}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
