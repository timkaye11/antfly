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

"""Shared, model-free EmbeddingGemma 2 qualification contract."""

from __future__ import annotations

import math
from typing import Any

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
WEIGHT_BYTES = 1_488_915_288
WEIGHT_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
MAX_TOKENS = 8_192
FULL_DIMENSIONS = 768
TRAINED_DIMENSIONS = (128, 256, 512, 768)
PROMPTS = {
    "RETRIEVAL_DOCUMENT": "title: none | text: ",
    "RETRIEVAL_QUERY": "task: search result | query: ",
    "CODE_RETRIEVAL": "task: code retrieval | query: ",
    "CLASSIFICATION": "task: classification | query: ",
    "CLUSTERING": "task: clustering | query: ",
    "QUESTION_ANSWERING": "task: question answering | query: ",
    "FACT_CHECKING": "task: fact checking | query: ",
    "SEMANTIC_SIMILARITY": "task: sentence similarity | query: ",
}


class ContractError(ValueError):
    pass


def render_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> str:
    try:
        prefix = PROMPTS[task_type]
    except KeyError as exc:
        raise ContractError(f"unsupported task type: {task_type}") from exc
    return prefix + text


def l2_normalize(values: list[float]) -> list[float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ContractError("embedding must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ContractError("embedding has zero norm")
    return [value / norm for value in values]


def truncate_and_normalize(values: list[float], dimensions: int) -> list[float]:
    if dimensions < 1 or dimensions > len(values):
        raise ContractError("dimensions outside the embedding width")
    return l2_normalize(values[:dimensions])


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ContractError("cosine requires equal non-empty vectors")
    return sum(a * b for a, b in zip(l2_normalize(left), l2_normalize(right)))


def validate_oracle(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "antfly.embedding_gemma2.oracle.v1":
        raise ContractError("unexpected oracle schema")
    if payload.get("model", {}).get("revision") != REVISION:
        raise ContractError("oracle model revision is not pinned")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("oracle has no cases")
    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ContractError("oracle case ids must be unique strings")
        ids.add(case_id)
        if case.get("expanded_tokens", MAX_TOKENS + 1) > MAX_TOKENS:
            raise ContractError(f"{case_id}: expanded token limit exceeded")
        embeddings = case.get("embeddings", {})
        for dimensions in TRAINED_DIMENSIONS:
            vector = embeddings.get(str(dimensions))
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise ContractError(f"{case_id}: invalid {dimensions}-dimension vector")
            norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
                raise ContractError(f"{case_id}: vector is not unit normalized")
