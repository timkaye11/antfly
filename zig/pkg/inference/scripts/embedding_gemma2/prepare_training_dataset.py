#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build a bounded, pinned public EG2 retrieval fixture.

This intentionally produces a functional qualification fixture, not a full
MS MARCO, CodeSearchNet, SciFact, or NFCorpus benchmark snapshot. Hugging Face
datasets are opened in streaming mode and every source revision is immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from finetune_contract import canonical_hash, normalized_content_hash, validate_dataset_manifest

SOURCES = {
    "msmarco": ("sentence-transformers/msmarco-msmarco-distilbert-base-tas-b", "e63dd8397db956cdde78e5e18c0d734585874db2", "triplet", "ms-marco-research-terms"),
    "codesearchnet": ("code-search-net/code_search_net", "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555", None, "various-upstream-repository-licenses"),
    "scifact": ("BeIR/scifact", "b3b5335604bf5ee3c4447671af975ea25143d4f5", None, "cc-by-sa-4.0"),
    "scifact_qrels": ("BeIR/scifact-qrels", "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029", None, "cc-by-sa-4.0"),
    "nfcorpus": ("BeIR/nfcorpus", "b5026a0e96e8a7ac4f95f482a596389289d46269", None, "cc-by-sa-4.0"),
    "nfcorpus_qrels": ("BeIR/nfcorpus-qrels", "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65", None, "cc-by-sa-4.0"),
}


def take(dataset, count: int) -> list[dict]:
    result = []
    for row in dataset:
        result.append(dict(row))
        if len(result) == count:
            break
    return result


def document(identifier: str, text: str, title: str = "") -> dict:
    text = text.strip()
    if not identifier or not text:
        raise ValueError("public source has an empty document ID or body")
    value = {"id": identifier, "text": text}
    if title.strip():
        value["title"] = title.strip()
    return value


def msmarco_records(rows: list[dict], split: str = "train") -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        prefix = f"msmarco:{index}"
        records.append({
            "query": row["query"].strip(), "query_id": prefix, "group_id": prefix,
            "positive": document(prefix + ":positive", row["positive"]),
            "negatives": [document(prefix + ":negative", row["negative"])],
            "known_positive_ids": [prefix + ":positive"], "task": "retrieval", "split": split,
        })
    return records


def codesearch_records(rows: list[dict], language: str, split: str) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        query = (row.get("func_documentation_string") or row.get("docstring") or "").strip()
        code = (row.get("whole_func_string") or row.get("code") or row.get("func_code_string") or "").strip()
        if not query or not code:
            continue
        source_id = str(row.get("url") or row.get("func_name") or index)
        identifier = f"codesearchnet:{language}:{split}:{hashlib.sha256(source_id.encode()).hexdigest()[:20]}"
        repository = str(row.get("repository_name") or row.get("repo") or source_id)
        group = f"codesearchnet:{language}:repo:{hashlib.sha256(repository.encode()).hexdigest()[:20]}"
        records.append({"query": query, "query_id": identifier, "group_id": group,
                        "positive": document(identifier + ":code", code), "negatives": [],
                        "known_positive_ids": [identifier + ":code"], "task": "code_retrieval", "split": split})
    return records


def diverse_prefix(records: list[dict], cap: int) -> list[dict]:
    selected, seen_groups = [], set()
    for record in records:
        if record["group_id"] in seen_groups:
            continue
        selected.append(record); seen_groups.add(record["group_id"])
        if len(selected) == cap:
            return selected
    for record in records:
        if record not in selected:
            selected.append(record)
            if len(selected) == cap:
                break
    return selected


def add_train_only_negatives(records: list[dict], seed: int) -> None:
    train = [record for record in records if record["split"] == "train"]
    corpus = sorted(((record, record["positive"]) for record in train), key=lambda value: value[1]["id"])
    rng = random.Random(seed)
    offset = rng.randrange(len(corpus)) if corpus else 0
    for row_index, record in enumerate(train):
        if record["negatives"]:
            continue
        for delta in range(1, len(corpus) + 1):
            candidate_record, candidate = corpus[(row_index + offset + delta) % len(corpus)]
            if candidate_record["group_id"] != record["group_id"] and candidate["id"] not in record["known_positive_ids"] and normalized_content_hash(candidate["text"]) != normalized_content_hash(record["positive"]["text"]):
                record["negatives"] = [candidate]
                break
        if not record["negatives"]:
            raise ValueError("training record has no admissible hard negative")


def add_frozen_base_negatives(records: list[dict], model_dir: str, batch_size: int) -> None:
    """Mine the highest-cosine admissible train positive with a frozen local model."""
    from sentence_transformers import SentenceTransformer
    train = [record for record in records if record["split"] == "train"]
    candidates = sorted(train, key=lambda record: record["positive"]["id"])
    if len(candidates) < 2:
        raise ValueError("frozen-base mining needs at least two train records")
    model = SentenceTransformer(model_dir, local_files_only=True)
    queries = model.encode([record["query"] for record in train], batch_size=batch_size,
                           convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    documents = model.encode([record["positive"]["text"] for record in candidates], batch_size=batch_size,
                             convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    for row_index, record in enumerate(train):
        if record["negatives"]:  # Preserve source-provided MS MARCO mined negatives.
            continue
        ranked = sorted(((float(queries[row_index] @ documents[index]), candidate)
                         for index, candidate in enumerate(candidates)),
                        key=lambda value: (-value[0], value[1]["positive"]["id"]))
        for _, candidate_record in ranked:
            candidate = candidate_record["positive"]
            if (candidate_record["group_id"] != record["group_id"] and
                    candidate["id"] not in record["known_positive_ids"] and
                    normalized_content_hash(candidate["text"]) != normalized_content_hash(record["positive"]["text"])):
                record["negatives"] = [candidate]
                break
        if not record["negatives"]:
            raise ValueError("training record has no admissible frozen-base hard negative")


def field(row: dict, *names: str):
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"missing one of fields {names}")


def beir_records(corpus_rows: list[dict], query_rows: list[dict], qrel_rows: list[dict], dataset: str, split: str, cap: int) -> list[dict]:
    corpus = {str(field(row, "_id", "id", "doc_id", "corpus-id")): row for row in corpus_rows}
    queries = {str(field(row, "_id", "id", "query_id", "query-id")): row for row in query_rows}
    relevant: dict[str, list[str]] = {}
    for row in qrel_rows:
        score = float(row.get("score", row.get("relevance", 1)))
        if score <= 0:
            continue
        query_id = str(field(row, "query-id", "query_id", "qid"))
        document_id = str(field(row, "corpus-id", "corpus_id", "doc_id"))
        relevant.setdefault(query_id, []).append(document_id)
    records = []
    for query_id in sorted(relevant):
        ids = sorted(set(relevant[query_id]))
        if query_id not in queries or not ids or ids[0] not in corpus:
            continue
        query = queries[query_id]
        positive = corpus[ids[0]]
        record_id = f"beir:{dataset}:{split}:{query_id}"
        records.append({"query": str(field(query, "text", "query")).strip(), "query_id": record_id,
                        "group_id": record_id,
                        "positive": document(f"beir:{dataset}:doc:{ids[0]}", str(positive.get("text", "")), str(positive.get("title", ""))),
                        "negatives": [], "known_positive_ids": [f"beir:{dataset}:doc:{value}" for value in ids],
                        "task": "retrieval", "split": split})
        if len(records) == cap:
            break
    return records


def decontaminate(records: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    priority = {"train": 0, "validation": 1, "test": 2}
    accepted, claimed, claimed_groups, same_split_queries = [], {}, {}, {}
    for record in sorted(records, key=lambda value: (-priority[value["split"]], value["query_id"])):
        split = record["split"]
        group = record["group_id"]
        if group in claimed_groups and claimed_groups[group] != split:
            continue
        hashes = {normalized_content_hash(record["query"]), normalized_content_hash(record["positive"]["text"])}
        if any(digest in claimed and claimed[digest] != split for digest in hashes):
            continue
        query_key = (split, normalized_content_hash(record["query"]))
        if query_key in same_split_queries:
            retained = same_split_queries[query_key]
            retained["known_positive_ids"] = sorted(set(retained["known_positive_ids"] + record["known_positive_ids"] + [record["positive"]["id"]]))
            continue
        for digest in hashes:
            claimed[digest] = split
        claimed_groups[group] = split
        accepted.append(record)
        same_split_queries[query_key] = record
    accepted.sort(key=lambda value: (priority[value["split"]], value["query_id"]))
    split_hashes = {name: [] for name in priority}
    for record in accepted:
        split_hashes[record["split"]].extend(sorted({normalized_content_hash(record["query"]), normalized_content_hash(record["positive"]["text"])}))
    return accepted, split_hashes


def source_entry(name: str, selected: list[dict], config_override: str | None = None) -> dict:
    dataset, revision, config, license_name = SOURCES[name]
    return {"dataset": dataset, "revision": revision, "config": config_override or config or "default", "license": license_name,
            "snapshot_sha256": canonical_hash(selected)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument("--validation-cap", type=int, default=100)
    parser.add_argument("--test-cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frozen-base-model", help="local frozen SentenceTransformer used to mine code negatives")
    parser.add_argument("--mining-batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.train_cap <= 10_000 or not 1 <= args.validation_cap <= 1000 or not 1 <= args.test_cap <= 1000:
        parser.error("fixture caps exceed the bounded contract")
    if not 1 <= args.mining_batch_size <= 1024:
        parser.error("mining batch size must be in [1,1024]")
    from datasets import load_dataset

    selected: dict[str, list[dict]] = {}
    msm_repo, msm_rev, msm_config, _ = SOURCES["msmarco"]
    selected["msmarco"] = take(load_dataset(msm_repo, msm_config, split="train", revision=msm_rev, streaming=True), args.train_cap // 2)
    records = msmarco_records(selected["msmarco"])
    for language in ("python", "go"):
        repo, revision, _, _ = SOURCES["codesearchnet"]
        for split, cap in (("train", args.train_cap // 4), ("validation", args.validation_cap), ("test", args.test_cap)):
            key = f"codesearchnet_{language}_{split}"
            selected[key] = take(load_dataset(repo, language, split=split, revision=revision, streaming=True), cap * 20)
            records.extend(diverse_prefix(codesearch_records(selected[key], language, split), cap))
    for dataset in ("scifact", "nfcorpus"):
        repo, revision, _, _ = SOURCES[dataset]
        corpus_rows = list(load_dataset(repo, "corpus", split="corpus", revision=revision, streaming=True))
        query_rows = list(load_dataset(repo, "queries", split="queries", revision=revision, streaming=True))
        qrepo, qrevision, _, _ = SOURCES[dataset + "_qrels"]
        qsplit = "test"
        qrels = list(load_dataset(qrepo, split=qsplit, revision=qrevision, streaming=True))
        selected[dataset] = corpus_rows + query_rows
        selected[dataset + "_qrels"] = qrels
        records.extend(beir_records(corpus_rows, query_rows, qrels, dataset, "test", args.test_cap))
    records, split_hashes = decontaminate(records)
    if args.frozen_base_model:
        add_frozen_base_negatives(records, args.frozen_base_model, args.mining_batch_size)
    else:
        add_train_only_negatives(records, args.seed)
    train = [record for record in records if record["split"] == "train"]
    sources = [source_entry("msmarco", selected["msmarco"])]
    for language in ("python", "go"):
        sources.append(source_entry("codesearchnet", [row for split in ("train", "validation", "test") for row in selected[f"codesearchnet_{language}_{split}"]], language))
    for name in ("scifact", "scifact_qrels", "nfcorpus", "nfcorpus_qrels"):
        sources.append(source_entry(name, selected[name]))
    manifest = {"schema": "antfly.embedding_gemma2.finetune_dataset.v1",
                "fixture_scope": "bounded-functional-not-official-benchmark", "seed": args.seed,
                "sources": sources,
                "splits": {"train_content_sha256": sorted(set(split_hashes["train"])),
                           "validation_content_sha256": sorted(set(split_hashes["validation"])),
                           "heldout_content_sha256": sorted(set(split_hashes["test"])),
                           "train_corpus_ids": sorted({doc["id"] for row in train for doc in [row["positive"], *row["negatives"]]}),
                           "train_positive_pairs": sorted([[row["query_id"], row["positive"]["id"]] for row in train])},
                "miner": {"policy": ("frozen-base-cosine-train-only-v1" if args.frozen_base_model else "deterministic-train-only-cross-group-random-v1"),
                          "seed": args.seed, "frozen_base_model": args.frozen_base_model,
                          "corpus_ids": sorted({doc["id"] for row in train for doc in [row["positive"], *row["negatives"]]}),
                          "negative_pairs": sorted([[row["query_id"], doc["id"]] for row in train for doc in row["negatives"]])}}
    validate_dataset_manifest(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "retrieval.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "train": len(train), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
