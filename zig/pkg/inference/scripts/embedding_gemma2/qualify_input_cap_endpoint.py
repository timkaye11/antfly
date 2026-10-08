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

"""Check a resident model whose manifest narrows the expanded input-token cap.

Serve a separate verified model bundle with the manifest capability
inference.limits.max_input_tokens_per_item=512 before running this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from qualify_contract_endpoint import request
from qualify_endpoint import data_uris
from select_pytorch_baseline import exact_length_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="embeddinggemma-2-limit512")
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_dir, local_files_only=True)
    image_uri, _ = data_uris()
    image = {"type": "image_url", "image_url": {"url": image_uri}}
    cases = (
        ("text512", exact_length_text(processor, 512, "RETRIEVAL_DOCUMENT"), 200, 512),
        ("text513", exact_length_text(processor, 513, "RETRIEVAL_DOCUMENT"), 400, None),
        ("image270", {"content": [image]}, 200, 270),
        ("images538", {"content": [image, image]}, 400, None),
    )
    gates = []
    for case, content, expected_status, expected_tokens in cases:
        status, payload = request(args.url, {
            "model": args.model,
            "input": content,
            "task_type": "RETRIEVAL_DOCUMENT",
        }, args.timeout)
        passed = status == expected_status
        if expected_status == 200:
            rows = payload.get("data", [])
            vector = rows[0].get("embedding", []) if len(rows) == 1 else []
            norm = math.sqrt(sum(value * value for value in vector))
            passed = (
                passed and payload.get("backend") == "cuda"
                and payload.get("usage", {}).get("prompt_tokens") == expected_tokens
                and len(vector) == 768 and all(math.isfinite(value) for value in vector)
                and abs(norm - 1) <= 1e-3
            )
        else:
            passed = passed and payload.get("error") == "INPUT_TOO_LONG"
        gates.append({
            "case": case, "pass": passed, "status": status,
            "expected_status": expected_status, "error": payload.get("error"),
            "usage": payload.get("usage"), "backend": payload.get("backend"),
        })
    report = {
        "schema": "antfly.embedding_gemma2.input_cap.v1",
        "pass": all(gate["pass"] for gate in gates),
        "max_input_tokens_per_item": 512, "gates": gates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
