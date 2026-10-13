#!/usr/bin/env python3
"""Produce exact, bounded-memory full-corpus retrieval quality evidence.

The evaluator intentionally makes one complete corpus pass for every query
chunk.  This costs extra endpoint work, but bounds resident vectors to two
query matrices and two corpus chunks instead of retaining a multi-million-row
embedding matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sqlite3
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Iterator

from finetune_contract import (BASE_WEIGHT_SHA256 as BASE_WEIGHT_SHA256, MRL_DIMENSIONS, QUALITY_SEEDS, REQUIRED_QUALITY_CELLS,
                               file_sha256, load_json, validate_candidate_artifacts, validate_official_corpus,
                               validate_official_source_manifest)
from contract import PROMPTS
from qualify_finetune_quality import validate_reviewed_inventory

try:  # NumPy makes the full official corpus practical; tests and small fixtures need no dependency.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in the minimal repository test environment
    _np = None

METRICS = ("mrr_at_10", "ndcg_at_10", "recall_at_100")
CELL_KEYS = REQUIRED_QUALITY_CELLS


def jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            yield value


def batches(values: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def record_id(row: dict) -> str:
    value = row.get("id", row.get("_id"))
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value):
        raise ValueError("corpus and query records require nonempty id or _id")
    return str(value)


def query_text(row: dict) -> str:
    value = row.get("text", row.get("query"))
    if not isinstance(value, str) or not value:
        raise ValueError("query record requires nonempty text or query")
    return value


def document_text(row: dict) -> str:
    text = row.get("text", row.get("contents"))
    if not isinstance(text, str) or not text:
        raise ValueError("corpus record requires nonempty text or contents")
    title = row.get("title")
    if title is None or title == "":
        return text
    if not isinstance(title, str):
        raise ValueError("document title must be a string")
    return f"title: {title} | text: {text}"


def document_input(row: dict) -> str | dict:
    # A title belongs in the group metadata. Putting it in the body makes the
    # serving frontend prepend a second document prompt.
    document_text(row)  # Validate body and optional title.
    text = row.get("text", row.get("contents"))
    title = row.get("title")
    return {"title": title, "content": [{"type": "text", "text": text}]} if title else text


class TextInputLimiter:
    """Apply identical bounded text preprocessing to base and adapted models."""

    def __init__(self, tokenizer, query_cap=128, document_cap=512):
        self.tokenizer = tokenizer
        self.query_cap, self.document_cap = query_cap, document_cap
        self.truncated = 0

    def length(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False)) + 2

    def __call__(self, value, task):
        title = value.get("title") if isinstance(value, dict) else None
        text = value["content"][0]["text"] if isinstance(value, dict) else value
        cap = self.document_cap if task == "RETRIEVAL_DOCUMENT" else self.query_cap
        prefix = f"title: {title} | text: " if title else PROMPTS[task]
        changed = False
        # Retain the document prompt even when an unusually long title alone
        # exceeds the cap. All corpus rows remain in the ranking universe.
        while self.length(prefix) > cap:
            if not title:
                raise ValueError("token cap is shorter than the required prompt")
            title = title[:len(title) // 2]
            prefix = f"title: {title or 'none'} | text: "
            changed = True
        if self.length(prefix + text) > cap:
            encoded = self.tokenizer(prefix + text, add_special_tokens=False, return_offsets_mapping=True)
            end = encoded['offset_mapping'][cap - 3][1]
            text = text[:max(0, end - len(prefix))]
            # Retokenization at the new boundary can alter a final token.
            while self.length(prefix + text) > cap:
                text = text[:-1]
            changed = True
        self.truncated += changed
        if title:
            return {"title": title, "content": [{"type": "text", "text": text}]}
        return text


def truncate_normalize(vectors: list[list[float]], dimension: int) -> list[list[float]]:
    result = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != 768:
            raise ValueError("endpoint must return full 768-vectors")
        row = vector[:dimension]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in row):
            raise ValueError("endpoint returned non-finite embeddings")
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0:
            raise ValueError("cannot normalize a zero embedding")
        result.append([value / norm for value in row])
    return result


def similarities(queries: list[list[float]], documents: list[list[float]]) -> list[list[float]]:
    if queries and documents and len(queries[0]) != len(documents[0]):
        raise ValueError("query and document embedding dimensions differ")
    if _np is not None:
        # Pin float64 to match the dependency-free path, including near ties.
        return (_np.asarray(queries, dtype=_np.float64) @
                _np.asarray(documents, dtype=_np.float64).T).tolist()
    return [[sum(a * b for a, b in zip(query, document)) for document in documents]
            for query in queries]


def embedding_identity(directory: Path, *, weights_digest: str | None = None) -> str:
    """Match the native identity of the exact bundle and embedding recipe."""
    digest = hashlib.sha256(b"embeddinggemma2-f32-mean-v1")
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
             "config_sentence_transformers.json", "1_Pooling/config.json", "weights")
    for name in names:
        value = weights_digest if name == "weights" and weights_digest else file_sha256(
            directory / ("model.safetensors" if name == "weights" else name))
        digest.update(name.encode() + b"\0" + bytes.fromhex(value))
    return digest.hexdigest()


class HttpEmbedder:
    def __init__(self, url: str, model: str, timeout: float, *, limiter=None, batch_size=32,
                 expected_identity: str | None = None):
        self.url, self.model, self.timeout = url, model, timeout
        self.provenance: set[str] = set()
        self.limiter, self.batch_size = limiter, batch_size
        self.expected_identity = expected_identity
        if not 1 <= batch_size <= 32:
            raise ValueError("embedding batch size must be between 1 and 32")

    def __call__(self, texts: list[str | dict], task_type: str) -> list[list[float]]:
        vectors = []
        for chunk in batches(texts, self.batch_size):
            values = [self.limiter(value, task_type) for value in chunk] if self.limiter else chunk
            if any(isinstance(value, dict) for value in values):
                values = [value if isinstance(value, dict) else {"content": [{"type": "text", "text": value}]} for value in values]
            vectors.extend(self._request(values, task_type))
        return vectors

    def _request(self, texts, task_type):
        # Omitting `dimensions` is deliberate: MRL is performed locally from
        # the endpoint's full 768-vector result.
        fields = {"model": self.model, "input": texts, "task_type": task_type}
        if self.expected_identity is not None:
            fields["model_identity"] = self.expected_identity
        body = json.dumps(fields).encode()
        request = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if payload.get("backend") != "cuda":
            raise ValueError("full-corpus evidence requires endpoint backend=cuda")
        if self.expected_identity is not None and payload.get("model_identity") != self.expected_identity:
            raise ValueError("quality endpoint model identity differs from the authenticated bundle")
        self.provenance.add(payload["backend"])
        rows = sorted(payload.get("data", []), key=lambda row: row.get("index", -1))
        if [row.get("index") for row in rows] != list(range(len(texts))):
            raise ValueError("endpoint response indexes do not cover the request")
        return [row.get("embedding", []) for row in rows]


class TopK:
    """Exact deterministic top-k, ordering score descending then ID ascending."""
    def __init__(self, count: int, k: int = 100):
        self.k = k
        self.rows: list[list[tuple[float, str]]] = [[] for _ in range(count)]

    def add(self, scores: list[list[float]], ids: list[str]) -> None:
        if len(scores) != len(self.rows) or any(len(row) != len(ids) for row in scores):
            raise ValueError("score matrix shape differs from query/document IDs")
        for query_index, row in enumerate(scores):
            candidates = self.rows[query_index] + [(float(score), ids[i]) for i, score in enumerate(row)]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            self.rows[query_index] = candidates[:self.k]

    def ids(self, index: int) -> list[str]:
        return [doc_id for _, doc_id in self.rows[index]]


def load_qrels(path: Path, wanted: set[str]) -> dict[str, dict[str, float]]:
    result = {query_id: {} for query_id in wanted}
    for row in jsonl(path):
        query_id = str(row.get("query-id", row.get("query_id", row.get("qid", ""))))
        if query_id not in wanted:
            continue
        doc_id = str(row.get("corpus-id", row.get("corpus_id", row.get("doc_id", row.get("pid", "")))))
        relevance = row.get("score", row.get("relevance", 1))
        if (not doc_id or isinstance(relevance, bool) or not isinstance(relevance, (int, float))
                or not math.isfinite(relevance) or relevance < 0):
            raise ValueError("qrel requires doc ID and nonnegative numeric relevance")
        result[query_id][doc_id] = max(float(relevance), result[query_id].get(doc_id, 0.0))
    if any(not any(score > 0 for score in grades.values()) for grades in result.values()):
        raise ValueError("every official query must have at least one positive qrel")
    return result


def metrics(ranked: list[str], grades: dict[str, float]) -> dict[str, float]:
    positive = {doc_id for doc_id, grade in grades.items() if grade > 0}
    reciprocal = next((1.0 / (rank + 1) for rank, doc_id in enumerate(ranked[:10]) if doc_id in positive), 0.0)
    dcg = sum((2.0 ** grades.get(doc_id, 0.0) - 1.0) / math.log2(rank + 2)
              for rank, doc_id in enumerate(ranked[:10]))
    ideal_grades = sorted((grade for grade in grades.values() if grade > 0), reverse=True)[:10]
    ideal = sum((2.0 ** grade - 1.0) / math.log2(rank + 2)
                for rank, grade in enumerate(ideal_grades))
    return {"mrr_at_10": reciprocal, "ndcg_at_10": dcg / ideal,
            "recall_at_100": len(set(ranked[:100]) & positive) / len(positive)}


def count_unique(path: Path) -> int:
    # An 8.8-million-ID Python set would exceed the intended memory bound.
    with tempfile.TemporaryDirectory(prefix="eg2-corpus-inventory-") as directory:
        with sqlite3.connect(str(Path(directory) / "ids.sqlite")) as db:
            db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            count = 0
            for rows in batches(jsonl(path), 4096):
                try:
                    db.executemany("INSERT INTO ids VALUES (?)", [(record_id(row),) for row in rows])
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"duplicate ID in {path}") from error
                count += len(rows)
            if not count:
                raise ValueError(f"{path} is empty")
            return count


def evaluate(*, corpus: Path, queries: Path, qrels: Path, base_embed: Callable,
             candidate_embed: Callable, dimension: int, query_chunk_size: int,
             corpus_chunk_size: int, query_task="RETRIEVAL_QUERY") -> tuple[list[str], dict, dict, int]:
    query_ids: list[str] = []
    base_arrays = {name: [] for name in METRICS}
    candidate_arrays = {name: [] for name in METRICS}
    passes = 0
    for query_rows in batches(jsonl(queries), query_chunk_size):
        ids = [record_id(row) for row in query_rows]
        if len(ids) != len(set(ids)) or set(ids) & set(query_ids):
            raise ValueError("query IDs must be globally unique")
        texts = [query_text(row) for row in query_rows]
        grades = load_qrels(qrels, set(ids))
        base_queries = truncate_normalize(base_embed(texts, query_task), dimension)
        candidate_queries = truncate_normalize(candidate_embed(texts, query_task), dimension)
        base_top, candidate_top = TopK(len(ids)), TopK(len(ids))
        seen_docs: set[str] = set()
        for corpus_rows in batches(jsonl(corpus), corpus_chunk_size):
            doc_ids = [record_id(row) for row in corpus_rows]
            if len(doc_ids) != len(set(doc_ids)) or seen_docs.intersection(doc_ids):
                raise ValueError("corpus IDs must be globally unique")
            seen_docs.update(doc_ids)
            doc_texts = [document_input(row) for row in corpus_rows]
            base_docs = truncate_normalize(base_embed(doc_texts, "RETRIEVAL_DOCUMENT"), dimension)
            candidate_docs = truncate_normalize(candidate_embed(doc_texts, "RETRIEVAL_DOCUMENT"), dimension)
            base_top.add(similarities(base_queries, base_docs), doc_ids)
            candidate_top.add(similarities(candidate_queries, candidate_docs), doc_ids)
        missing = {doc_id for query_id in ids for doc_id, grade in grades[query_id].items()
                   if grade > 0 and doc_id not in seen_docs}
        if missing:
            raise ValueError(f"positive qrels reference {len(missing)} documents absent from the full corpus")
        passes += 1
        for index, query_id in enumerate(ids):
            for name, value in metrics(base_top.ids(index), grades[query_id]).items():
                base_arrays[name].append(value)
            for name, value in metrics(candidate_top.ids(index), grades[query_id]).items():
                candidate_arrays[name].append(value)
        query_ids.extend(ids)
    return query_ids, base_arrays, candidate_arrays, passes


def artifact(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def command_evaluate(args: argparse.Namespace) -> int:
    key = (args.dataset, args.language, args.role)
    if key not in CELL_KEYS or args.dimension not in MRL_DIMENSIONS or args.seed not in QUALITY_SEEDS:
        raise ValueError("cell is outside the reviewed five-role/four-dimension/three-seed matrix")
    corpus_count, query_count = count_unique(args.corpus), count_unique(args.queries)
    if (corpus_count, query_count) != (args.expected_corpus_count, args.expected_query_count):
        raise ValueError("actual corpus/query counts differ from the pinned contract")
    inventories = validate_official_source_manifest(load_json(args.source_manifest))
    validate_reviewed_inventory(inventories)
    official = inventories[key]
    actual = {"corpus_sha256": file_sha256(args.corpus), "queries_sha256": file_sha256(args.queries),
              "qrels_sha256": file_sha256(args.qrels), "corpus_count": corpus_count,
              "query_count": query_count}
    if any(official[name] != value for name, value in actual.items()):
        raise ValueError("local full-corpus artifacts differ from the authenticated official inventory")
    # Hash every evidence artifact before beginning the potentially expensive
    # repeated corpus passes.
    evidence_artifacts = {name: artifact(getattr(args, name)) for name in
                          ("bundle_manifest", "checkpoint", "prepared_manifest", "training_source_manifest", "source_manifest")}
    model=args.bundle_manifest.parent/"model.safetensors"
    candidate_artifacts=validate_candidate_artifacts(checkpoint=args.checkpoint,model=model,
        prepared_manifest=args.prepared_manifest,training_source_manifest=args.training_source_manifest,seed=args.seed)
    evidence_artifacts["model"] = artifact(model)
    if (candidate_artifacts["candidate_bundle_manifest_sha256"]!=evidence_artifacts["bundle_manifest"]["sha256"]
            or candidate_artifacts["candidate_model_sha256"]!=evidence_artifacts["model"]["sha256"]):
        raise ValueError("candidate export artifacts differ from their provenance")
    from transformers import AutoTokenizer
    from finetune_reference import TOKENIZER_SHA256
    if file_sha256(args.processor_dir / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("quality preprocessing tokenizer differs from the pinned contract")
    tokenizer = AutoTokenizer.from_pretrained(args.processor_dir, local_files_only=True)
    if file_sha256(args.processor_dir / "model.safetensors") != BASE_WEIGHT_SHA256:
        raise ValueError("quality base endpoint bundle differs from the pinned weights")
    base = HttpEmbedder(args.base_url, args.base_model, args.timeout, limiter=TextInputLimiter(tokenizer),
                        expected_identity=embedding_identity(args.processor_dir, weights_digest=BASE_WEIGHT_SHA256))
    candidate = HttpEmbedder(args.candidate_url, args.candidate_model, args.timeout, limiter=TextInputLimiter(tokenizer),
                             expected_identity=embedding_identity(model.parent, weights_digest=evidence_artifacts['model']['sha256']))
    ids, base_values, candidate_values, passes = evaluate(
        corpus=args.corpus, queries=args.queries, qrels=args.qrels, base_embed=base,
        candidate_embed=candidate, dimension=args.dimension,
        query_chunk_size=args.query_chunk_size, corpus_chunk_size=args.corpus_chunk_size,
        query_task="CODE_RETRIEVAL" if args.dataset == "codesearchnet" else "RETRIEVAL_QUERY")
    if len(ids) != query_count or base.provenance != {"cuda"} or candidate.provenance != {"cuda"}:
        raise ValueError("incomplete queries or missing CUDA endpoint provenance")
    cell = {"schema": "antfly.embedding_gemma2.finetune_quality_cell.v1",
            "qualification_scope": "official-full-corpus", "dataset": args.dataset,
            "language": args.language, "role": args.role, "dimension": args.dimension,
            "seed": args.seed, "base_query_ids": ids, "candidate_query_ids": ids,
            "base_per_query": base_values, "candidate_per_query": candidate_values,
            "base_mean": {name: statistics.fmean(base_values[name]) for name in METRICS},
            "candidate_mean": {name: statistics.fmean(candidate_values[name]) for name in METRICS},
            "official_corpus": official,
            "execution": {"base_url": args.base_url, "candidate_url": args.candidate_url,
                          "base_model": args.base_model, "candidate_model": args.candidate_model,
                          "base_identity": base.expected_identity, "candidate_identity": candidate.expected_identity,
                          "backend": "cuda", "query_chunk_size": args.query_chunk_size,
                          "corpus_chunk_size": args.corpus_chunk_size, "corpus_passes": passes,
                          "query_task": "CODE_RETRIEVAL" if args.dataset == "codesearchnet" else "RETRIEVAL_QUERY",
                          "text_preprocessing": "retain-rendered-text-prefix-with-bos-eos-v1",
                          "query_cap": 128, "document_cap": 512,
                          "full_vector_dimension": 768, "mrl": "truncate-then-l2-normalize"},
            "artifacts": evidence_artifacts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    return 0


def command_merge(args: argparse.Namespace) -> int:
    cells = [load_json(path) for path in args.cells]
    checkpoints = seed_paths(args.candidate_checkpoint, "candidate checkpoint")
    models = seed_paths(args.candidate_model, "candidate model")
    supplied_candidates = {seed: validate_candidate_artifacts(checkpoint=checkpoints[seed],model=models[seed],
        prepared_manifest=args.prepared_manifest,training_source_manifest=args.training_source_manifest,seed=seed)
        for seed in QUALITY_SEEDS}
    supplied_shared = {"prepared_manifest": file_sha256(args.prepared_manifest),
                       "training_source_manifest": file_sha256(args.training_source_manifest),
                       "source_manifest": file_sha256(args.source_manifest)}
    expected = {(d, l, r, dim, seed) for d, l, r in CELL_KEYS
                for dim in MRL_DIMENSIONS for seed in QUALITY_SEEDS}
    observed = {(c.get("dataset"), c.get("language"), c.get("role"), c.get("dimension"), c.get("seed")) for c in cells}
    if len(cells) != 60 or len(observed) != 60 or observed != expected:
        raise ValueError("merge requires exactly 60 unique reviewed cells")
    if any(c.get("qualification_scope") != "official-full-corpus" for c in cells):
        raise ValueError("bounded or synthetic cells cannot claim official evidence")
    source_inventories = validate_official_source_manifest(load_json(args.source_manifest))
    validate_reviewed_inventory(source_inventories)
    contracts = {}
    for cell in cells:
        key = (cell["dataset"], cell.get("language"), cell["role"])
        official = validate_official_corpus(cell.get("official_corpus"))
        if official != source_inventories[key]:
            raise ValueError("cell differs from authenticated official source inventory")
        if key in contracts and contracts[key] != official:
            raise ValueError("official corpus contract differs across dimension or seed")
        contracts[key] = official
        query_ids = cell.get("candidate_query_ids")
        if (query_ids != cell.get("base_query_ids") or not isinstance(query_ids, list)
                or len(query_ids) != official["query_count"] or len(set(query_ids)) != len(query_ids)
                or not all(isinstance(value, str) and value for value in query_ids)):
            raise ValueError("cell does not contain exact paired official query coverage")
        for side in ("base_per_query", "candidate_per_query"):
            values = cell.get(side, {})
            if any(not isinstance(values.get(metric), list) or len(values[metric]) != len(query_ids)
                   or any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                          for value in values[metric]) for metric in METRICS):
                raise ValueError("cell per-query metrics are incomplete or invalid")
        if cell.get("execution", {}).get("backend") != "cuda":
            raise ValueError("cell lacks CUDA endpoint provenance")
        recorded = cell.get("artifacts", {})
        candidate=supplied_candidates[cell["seed"]]
        expected_artifacts = {**supplied_shared,
            "checkpoint":candidate["candidate_checkpoint_sha256"],
            "model":candidate["candidate_model_sha256"],
            "bundle_manifest":candidate["candidate_bundle_manifest_sha256"]}
        if any(recorded.get(name, {}).get("sha256") != digest
               for name, digest in expected_artifacts.items()):
            raise ValueError("cell artifact identities differ from merge inputs")
    provenance = {"candidates_by_seed": {str(seed): {
                      **supplied_candidates[seed]}
                      for seed in QUALITY_SEEDS},
                  "base_weight_sha256": args.base_weight_sha256,
                  "prepared_manifest_sha256": supplied_shared["prepared_manifest"],
                  "training_source_manifest_sha256": supplied_shared["training_source_manifest"],
                  "public_source_manifest_sha256": supplied_shared["source_manifest"],
                  "public_sources": load_json(args.source_manifest).get("sources", [])}
    output = {"schema": "antfly.embedding_gemma2.finetune_quality_input.v1",
              "qualification_scope": "official-full-corpus", "seeds": list(QUALITY_SEEDS),
              "cell_contract": [{"dataset": key[0], "language": key[1], "role": key[2],
                                 "fixture_scope": "official-full-corpus", "official_corpus": value}
                                for key, value in sorted(contracts.items(), key=lambda item: str(item[0]))],
              "cells": [{**{key: value for key, value in cell.items()
                             if key not in ("schema", "qualification_scope", "official_corpus", "execution", "artifacts",
                                            "base_mean", "candidate_mean")},
                         **supplied_candidates[cell["seed"]]}
                        for cell in cells], "provenance": provenance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


def seed_paths(values: list[str], label: str) -> dict[int, Path]:
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text.isdigit() or not path_text:
            raise ValueError(f"{label} must use SEED=PATH")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate {label} seed {seed}")
        result[seed] = Path(path_text)
    if set(result) != set(QUALITY_SEEDS):
        raise ValueError(f"{label} requires exactly seeds {QUALITY_SEEDS}")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("evaluate")
    for name in ("corpus", "queries", "qrels", "bundle-manifest", "checkpoint", "prepared-manifest", "training-source-manifest", "source-manifest", "processor-dir", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--dataset", required=True); run.add_argument("--language")
    run.add_argument("--role", choices=("target", "retention"), required=True)
    run.add_argument("--dimension", type=int, required=True); run.add_argument("--seed", type=int, required=True)
    run.add_argument("--expected-corpus-count", type=int, required=True)
    run.add_argument("--expected-query-count", type=int, required=True)
    run.add_argument("--base-url", required=True); run.add_argument("--candidate-url", required=True)
    run.add_argument("--base-model", required=True); run.add_argument("--candidate-model", required=True)
    run.add_argument("--query-chunk-size", type=int, default=32)
    run.add_argument("--corpus-chunk-size", type=int, default=256); run.add_argument("--timeout", type=float, default=300)
    run.set_defaults(function=command_evaluate)
    merge = commands.add_parser("merge")
    merge.add_argument("cells", nargs="+", type=Path)
    for name in ("prepared-manifest", "training-source-manifest", "source-manifest", "output"):
        merge.add_argument("--" + name, type=Path, required=True)
    merge.add_argument("--candidate-checkpoint", action="append", required=True, metavar="SEED=PATH")
    merge.add_argument("--candidate-model", action="append", required=True, metavar="SEED=PATH")
    merge.add_argument("--base-weight-sha256", required=True)
    merge.set_defaults(function=command_merge)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "query_chunk_size", 1) < 1 or getattr(args, "corpus_chunk_size", 1) < 1:
        raise ValueError("chunk sizes must be positive")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
