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

"""Generate the production EmbeddingGemma 2 qualification workload matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import MAX_TOKENS, PROMPTS, TRAINED_DIMENSIONS

SCHEMA = "antfly.embedding_gemma2.workloads.v1"
LENGTHS = (16, 128, 512, 2048, 8192)
BATCHES = (1, 8, 32)


def cells() -> list[dict]:
    rows: list[dict] = []
    for batch in BATCHES:
        for length in LENGTHS:
            for corpus in ("natural_text", "source_code", "heterogeneous_padding"):
                rows.append({
                    "id": f"text-{corpus}-b{batch}-s{length}",
                    "lane": "paired_http",
                    "modality": "text",
                    "corpus": corpus,
                    "batch_size": batch,
                    "target_expanded_tokens": length,
                    "task_type": "CODE_RETRIEVAL" if corpus == "source_code" else "RETRIEVAL_DOCUMENT",
                    "dimensions": 768,
                })
    for width, height in ((32, 32), (128, 96), (896, 896), (1536, 256), (256, 1536)):
        rows.append({"id": f"image-{width}x{height}", "lane": "quality", "modality": "image", "width": width, "height": height, "task_type": "RETRIEVAL_DOCUMENT"})
    for seconds, rate in ((0.1, 16_000), (1, 16_000), (10, 16_000), (31, 16_000), (1, 8_000), (1, 44_100), (10, 48_000)):
        label = str(seconds).replace(".", "p")
        rows.append({"id": f"audio-{label}s-{rate}hz", "lane": "quality", "modality": "audio", "seconds": seconds, "sample_rate": rate, "channels": 1, "task_type": "RETRIEVAL_DOCUMENT"})
    rows.extend((
        {"id": "mixed-text-image-audio", "lane": "quality", "modality": "mixed", "order": ["text", "image", "audio"]},
        {"id": "mixed-audio-text-image", "lane": "quality", "modality": "mixed", "order": ["audio", "text", "image"]},
        {"id": "mixed-image-image-text", "lane": "quality", "modality": "mixed", "order": ["image", "image", "text"]},
        {"id": "mixed-audio-audio-text", "lane": "quality", "modality": "mixed", "order": ["audio", "audio", "text"]},
    ))
    for task_type in PROMPTS:
        rows.append({"id": f"task-{task_type.lower()}", "lane": "quality", "modality": "text", "task_type": task_type, "target_expanded_tokens": 128})
    for dimensions in TRAINED_DIMENSIONS:
        rows.append({"id": f"mrl-{dimensions}", "lane": "quality", "modality": "text", "task_type": "RETRIEVAL_QUERY", "target_expanded_tokens": 128, "dimensions": dimensions})
    return rows


def negative_cases() -> list[dict]:
    return [
        {"id": "expanded-8193", "expect": "reject", "reason": "expanded_token_limit", "target_expanded_tokens": MAX_TOKENS + 1},
        {"id": "all-padding-mask", "expect": "reject", "reason": "empty_effective_sequence"},
        {"id": "malformed-image", "expect": "reject", "reason": "media_decode"},
        {"id": "malformed-audio", "expect": "reject", "reason": "media_decode"},
        {"id": "non-finite-result", "expect": "reject", "reason": "finite_embedding_contract"},
        {"id": "cancel-then-reuse-session", "expect": "subsequent_success", "reason": "resident_session_cleanup"},
        {"id": "unknown-task-prefix", "expect": "reject", "reason": "task_type_allowlist"},
        {"id": "dimensions-zero", "expect": "reject", "reason": "invalid_dimensions"},
        {"id": "dimensions-769", "expect": "reject", "reason": "invalid_dimensions"},
    ]


def manifest() -> dict:
    return {
        "schema": SCHEMA,
        "contract": {
            "max_expanded_tokens": MAX_TOKENS,
            "batch_sizes": list(BATCHES),
            "sequence_lengths": list(LENGTHS),
            "task_types": list(PROMPTS),
            "trained_dimensions": list(TRAINED_DIMENSIONS),
        },
        "cells": cells(),
        "negative_cases": negative_cases(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
