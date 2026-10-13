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

"""Shared, model-free contracts for EmbeddingGemma 2 fine-tune qualification."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable

MRL_DIMENSIONS = (128, 256, 512, 768)
QUALITY_SEEDS = (17, 29, 43)
MAX_TRAIN_TOKENS = 512
MIN_THROUGHPUT_RATIO = 0.90
MAX_P95_RATIO = 1.10
MAX_DEVICE_BYTES = 20 * 1024**3
MAX_HOST_BYTES = 8 * 1024**3
BASE_WEIGHT_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
REQUIRED_QUALITY_CELLS = frozenset({
    ("msmarco", None, "target"),
    ("codesearchnet", "python", "target"),
    ("codesearchnet", "go", "target"),
    ("scifact", None, "retention"),
    ("nfcorpus", None, "retention"),
})


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_artifacts(*, checkpoint: Path, model: Path, prepared_manifest: Path,
                                 training_source_manifest: Path, seed: int) -> dict[str, str]:
    """Bind exported model bytes to their checkpoint, data, seed, and sources."""
    for label, path in (("checkpoint", checkpoint), ("model", model),
                        ("prepared manifest", prepared_manifest),
                        ("training source manifest", training_source_manifest)):
        if not path.is_file():
            raise ValueError(f"candidate {label} does not exist: {path}")
    checkpoint_sha = file_sha256(checkpoint)
    model_sha = file_sha256(model)
    prepared_sha = file_sha256(prepared_manifest)
    training_source_sha = file_sha256(training_source_manifest)
    if load_json(prepared_manifest).get("source_manifest_sha256") != training_source_sha:
        raise ValueError("prepared manifest is not bound to the supplied training-source manifest")
    provenance_path = model.parent / "antfly_finetune_provenance.json"
    provenance = load_json(provenance_path)
    if (provenance.get("schema") != "antfly.embedding_gemma2.finetune_export.v1"
            or provenance.get("base_weight_sha256") != BASE_WEIGHT_SHA256
            or provenance.get("merged_weight_sha256") != model_sha):
        raise ValueError("candidate export provenance does not bind the pinned merged model bytes")
    training = provenance.get("training")
    if (not isinstance(training, dict)
            or training.get("schema") != "antfly.embedding_gemma2.native_training.v1"
            or training.get("seed") != seed
            or training.get("dataset_sha256") != prepared_sha
            or training.get("checkpoint_weight_sha256") != checkpoint_sha
            or training.get("public_source_manifest_sha256") != training_source_sha):
        raise ValueError("candidate export training provenance differs from its seed/data/checkpoint/source artifacts")
    return {"candidate_checkpoint_sha256": checkpoint_sha,
            "candidate_model_sha256": model_sha,
            "candidate_bundle_manifest_sha256": file_sha256(provenance_path)}


def validate_official_corpus(value: object) -> dict:
    """Validate a full-corpus/query/qrels identity without inventing counts."""
    if not isinstance(value, dict):
        raise ValueError("official corpus metadata must be an object")
    required = {"source_repo", "source_revision", "corpus_config", "corpus_split",
                "queries_config", "queries_split", "qrels_source_repo", "qrels_source_revision",
                "qrels_config", "qrels_split", "corpus_sha256", "queries_sha256", "qrels_sha256",
                "corpus_count", "query_count"}
    if set(value) != required:
        raise ValueError(f"official corpus metadata fields differ: {sorted(set(value) ^ required)}")
    if (not isinstance(value["source_repo"], str) or "/" not in value["source_repo"]
            or any(not isinstance(value[name], str) or not value[name] for name in
                   ("corpus_config", "corpus_split", "queries_config", "queries_split",
                    "qrels_source_repo", "qrels_config", "qrels_split"))):
        raise ValueError("official corpus source identity is invalid")
    if "/" not in value["qrels_source_repo"]:
        raise ValueError("official qrels source identity is invalid")
    if (not isinstance(value["source_revision"], str)
            or len(value["source_revision"]) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in value["source_revision"])):
        raise ValueError("official source revision must be an immutable lowercase digest")
    if (len(value["qrels_source_revision"]) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in value["qrels_source_revision"])):
        raise ValueError("official qrels source revision must be an immutable lowercase digest")
    for name in ("corpus_sha256", "queries_sha256", "qrels_sha256"):
        if not valid_sha256(value[name]):
            raise ValueError("official corpus artifacts require lowercase SHA256 identities")
    for name in ("corpus_count", "query_count"):
        if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 1:
            raise ValueError("official corpus counts must be positive integers")
    return value


def validate_official_source_manifest(manifest: dict) -> dict[tuple[str, str | None, str], dict]:
    """Authenticate the exact five reviewed full-corpus inventory records.

    This validates structure and immutable identities. The promotion gate also
    compares repo/config/revision/split against its reviewed source allowlist.
    """
    if manifest.get("schema") != "antfly.embedding_gemma2.official_retrieval_sources.v1":
        raise ValueError("unsupported official public-source manifest schema")
    inventories = manifest.get("official_inventories")
    if not isinstance(inventories, list):
        raise ValueError("public-source manifest lacks official query inventories")
    by_key = {}
    for inventory in inventories:
        if not isinstance(inventory, dict):
            raise ValueError("official query inventory must be an object")
        key = (inventory.get("dataset"), inventory.get("language"), inventory.get("role"))
        if key in by_key:
            raise ValueError("duplicate official query inventory")
        if key not in REQUIRED_QUALITY_CELLS:
            raise ValueError("unreviewed official query inventory")
        if set(inventory) != {"dataset", "language", "role", "official_corpus"}:
            raise ValueError("official query inventory fields differ")
        by_key[key] = validate_official_corpus(inventory["official_corpus"])
    if set(by_key) != REQUIRED_QUALITY_CELLS:
        raise ValueError("public-source manifest lacks the exact five official inventories")
    return by_key

# (logical batch, microbatch, query cap, document cap, negatives, rank, alpha)
# This set is deliberately exact: seven arbitrary tiny workloads cannot be
# substituted for the reviewed scaling and hard-negative matrix.
PERFORMANCE_CELL_FIELDS = ("logical_batch_size", "microbatch_size", "query_max_tokens",
                           "document_max_tokens", "max_negatives", "rank", "alpha")
PERFORMANCE_CELLS = frozenset({
    (16, 2, 32, 128, 0, 16, 32.0),
    (32, 2, 32, 128, 0, 16, 32.0),
    (64, 2, 32, 128, 0, 16, 32.0),
    (32, 2, 128, 512, 0, 16, 32.0),
    (64, 2, 128, 512, 0, 16, 32.0),
    (32, 2, 128, 512, 1, 16, 32.0),
    (64, 2, 128, 512, 4, 16, 32.0),
})


def performance_cell_tuple(value: dict) -> tuple[int, int, int, int, int, int, float]:
    """Normalize a job or collected-cell mapping to the reviewed tuple."""
    lora = value.get("lora", {})
    aliases = (
        value.get("logical_batch_size", value.get("batch_size")),
        value.get("microbatch_size"),
        value.get("query_max_tokens", value.get("query_tokens")),
        value.get("document_max_tokens", value.get("document_tokens")),
        value.get("max_negatives", value.get("negatives")),
        value.get("rank", lora.get("rank")),
        value.get("alpha", lora.get("alpha")),
    )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in aliases):
        raise ValueError("performance cell fields must be numeric")
    normalized = tuple(int(item) for item in aliases[:6]) + (float(aliases[6]),)
    if any(float(original) != normalized[index] for index, original in enumerate(aliases)):
        raise ValueError("performance cell integer fields must be integral")
    return normalized


def validate_performance_cells(values: Iterable[dict]) -> tuple[tuple[int, int, int, int, int, int, float], ...]:
    normalized = tuple(performance_cell_tuple(value) for value in values)
    if len(normalized) != len(PERFORMANCE_CELLS) or len(set(normalized)) != len(normalized):
        raise ValueError("performance qualification requires seven unique reviewed cells")
    if set(normalized) != PERFORMANCE_CELLS:
        missing = sorted(PERFORMANCE_CELLS - set(normalized))
        extra = sorted(set(normalized) - PERFORMANCE_CELLS)
        raise ValueError(f"performance cell contract differs: missing={missing}, extra={extra}")
    return normalized


def validate_performance_workload(cell: dict, expected_observations: int) -> None:
    if (not isinstance(expected_observations, int) or expected_observations < 1
            or len(cell["observed_query_counts"]) != expected_observations
            or any(count != cell["batch_size"] for count in cell["observed_query_counts"])):
        raise ValueError("production timing cells require full observed logical batches")
    minimum_candidates = cell["batch_size"] * (1 + cell["negatives"])
    if (len(cell["observed_candidate_counts"]) != expected_observations
            or any(not isinstance(count, int) or count < minimum_candidates
                   for count in cell["observed_candidate_counts"])):
        raise ValueError("observed candidate counts do not fill the configured negative workload")
    for key, cap in (("active_query_tokens", cell["query_tokens"]),
                     ("active_document_tokens", cell["document_tokens"])):
        bounds = cell[key]
        if (set(bounds) != {"minimum", "maximum", "mean", "microbatches"}
                or not all(isinstance(bounds[name], (int, float)) and math.isfinite(bounds[name])
                           for name in ("minimum", "maximum", "mean"))
                or not 1 <= bounds["minimum"] <= bounds["mean"] <= bounds["maximum"] <= cap
                or bounds["mean"] < .9 * cap
                or not isinstance(bounds["microbatches"], int)
                or bounds["microbatches"] < expected_observations):
            raise ValueError("observed active token ranges do not fill the configured token workload")


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def normalized_content_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty sample")
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def paired_ratio_ci(candidate_ms: list[float], reference_ms: list[float], *, samples: int = 5000, seed: int = 20261008) -> tuple[float, float]:
    if len(candidate_ms) != len(reference_ms) or len(candidate_ms) < 2:
        raise ValueError("paired timing arrays must have equal length >= 2")
    if not all(math.isfinite(v) and v > 0 for v in candidate_ms + reference_ms):
        raise ValueError("timings must be finite and positive")
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        indexes = [rng.randrange(len(candidate_ms)) for _ in candidate_ms]
        ratios.append(sum(reference_ms[i] for i in indexes) / sum(candidate_ms[i] for i in indexes))
    ratios.sort()
    return ratios[int(0.025 * samples)], ratios[min(samples - 1, int(0.975 * samples))]


def validate_dataset_manifest(manifest: dict) -> dict:
    required = {"schema", "sources", "splits", "miner"}
    if not required.issubset(manifest):
        raise ValueError(f"dataset manifest missing {sorted(required - set(manifest))}")
    if manifest["schema"] != "antfly.embedding_gemma2.finetune_dataset.v1":
        raise ValueError("unsupported dataset manifest schema")
    train = set(manifest["splits"].get("train_content_sha256", []))
    validation = set(manifest["splits"].get("validation_content_sha256", []))
    heldout = set(manifest["splits"].get("heldout_content_sha256", []))
    if not train or not validation or not heldout:
        raise ValueError("train, validation, and heldout hashes are required")
    overlap = (train & validation) | (train & heldout) | (validation & heldout)
    if overlap:
        raise ValueError(f"content leakage detected ({len(overlap)} hashes)")
    for digest in train | validation | heldout:
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("split content identities must be lowercase SHA256 digests")
    corpus_ids = set(manifest["splits"].get("train_corpus_ids", []))
    mined_ids = set(manifest["miner"].get("corpus_ids", []))
    if not corpus_ids or not all(isinstance(value, str) and value for value in corpus_ids | mined_ids):
        raise ValueError("training corpus IDs must be nonempty strings")
    if not mined_ids.issubset(corpus_ids):
        raise ValueError("hard-negative miner references non-training corpus IDs")
    policy = manifest["miner"].get("policy")
    supported_policies = ("frozen-base-cosine-train-only-v1", "deterministic-train-only-cross-group-random-v1")
    legacy_bounded = policy == "deterministic-train-only-v1"
    if policy not in supported_policies and not legacy_bounded:
        raise ValueError("unsupported hard-negative mining policy")
    frozen_model = manifest["miner"].get("frozen_base_model")
    if legacy_bounded and (manifest.get("fixture_scope") != "bounded-functional-not-official-benchmark" or frozen_model is not None):
        raise ValueError("legacy deterministic mining is admitted only for bounded fixtures without a frozen-model claim")
    if policy == "frozen-base-cosine-train-only-v1" and not isinstance(frozen_model, str):
        raise ValueError("frozen-base mining requires its model identity")
    if policy != "frozen-base-cosine-train-only-v1" and frozen_model is not None:
        raise ValueError("random mining cannot claim a frozen-base model")
    raw_positive_pairs = manifest["splits"].get("train_positive_pairs", [])
    raw_negative_pairs = manifest["miner"].get("negative_pairs", [])
    for pair in [*raw_positive_pairs, *raw_negative_pairs]:
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(value, str) and value for value in pair):
            raise ValueError("positive and negative pairs must contain two nonempty string IDs")
    positive_pairs = {tuple(pair) for pair in raw_positive_pairs}
    negative_pairs = {tuple(pair) for pair in raw_negative_pairs}
    false_negatives = positive_pairs & negative_pairs
    if false_negatives:
        raise ValueError(f"known positives used as negatives ({len(false_negatives)} pairs)")
    if not isinstance(manifest["sources"], list) or not manifest["sources"]:
        raise ValueError("at least one pinned dataset source is required")
    source_identities = set()
    for source in manifest["sources"]:
        for key in ("dataset", "revision", "config", "license", "snapshot_sha256"):
            if not source.get(key):
                raise ValueError(f"dataset source missing {key}")
        if (not isinstance(source["revision"], str) or len(source["revision"]) not in (40, 64)
                or any(c not in "0123456789abcdef" for c in source["revision"])):
            raise ValueError("dataset source revision must be an immutable lowercase digest")
        if (not isinstance(source["snapshot_sha256"], str) or len(source["snapshot_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in source["snapshot_sha256"])):
            raise ValueError("dataset snapshot identity must be a lowercase SHA256 digest")
        identity = (source["dataset"], source["revision"], source["config"])
        if identity in source_identities:
            raise ValueError("duplicate dataset source identity")
        source_identities.add(identity)
    return {"train": len(train), "validation": len(validation), "heldout": len(heldout),
            "miner_corpus": len(mined_ids), "miner_policy": policy,
            "legacy_bounded_policy": legacy_bounded}


def retrieval_metrics(ranked_ids: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    if not relevant:
        raise ValueError("query has no relevant documents")
    top = ranked_ids[:k]
    reciprocal = next((1.0 / (i + 1) for i, doc_id in enumerate(top) if doc_id in relevant), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(top) if doc_id in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    recall_100 = len(set(ranked_ids[:100]) & relevant) / len(relevant)
    return {"mrr_at_10": reciprocal, "ndcg_at_10": dcg / ideal, "recall_at_100": recall_100}


def mean_metrics(per_query: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(per_query)
    if not rows:
        raise ValueError("no query metrics")
    return {name: statistics.fmean(row[name] for row in rows) for name in ("mrr_at_10", "ndcg_at_10", "recall_at_100")}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
