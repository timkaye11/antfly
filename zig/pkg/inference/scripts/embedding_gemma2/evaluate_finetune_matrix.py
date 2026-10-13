#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate all seeds and MRL dimensions in one exact full-corpus pass.

Query matrices and top-100 rankings remain resident; document embeddings are
discarded after each bounded chunk. No corpus sampling or vector quantization
is used. The model manifest supplies a base endpoint and the three exported
candidate endpoints, checkpoints, and bundle manifests.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from evaluate_finetune_full_corpus import (
    METRICS, HttpEmbedder, TextInputLimiter, artifact, batches, count_unique,
    document_input, embedding_identity, jsonl, load_qrels, metrics, query_text, record_id,
)
from finetune_contract import (
    BASE_WEIGHT_SHA256, MRL_DIMENSIONS, QUALITY_SEEDS, file_sha256, load_json,
    validate_candidate_artifacts, validate_official_source_manifest,
)
from qualify_finetune_quality import validate_reviewed_inventory


class IntegerDocIds:
    """Lossless lexical keys for the pinned corpora's decimal document IDs."""

    def __init__(self):
        self.prefix = None

    def encode(self, value):
        match = re.fullmatch(r"(.*?)([0-9]+)", value)
        if match is None or len(match.group(2)) > 15:
            raise ValueError(f"document ID cannot use the reviewed lexical key: {value!r}")
        prefix, digits = match.groups()
        if self.prefix is None:
            self.prefix = prefix
        if prefix != self.prefix:
            raise ValueError("document IDs have inconsistent prefixes")
        # Zero is the terminator; digits use 1..10. Padding to fifteen
        # nibbles preserves lexical ordering and leading zeroes in 60 bits.
        key = 0
        for digit in digits:
            key = (key << 4) | (int(digit) + 1)
        return key << (4 * (15 - len(digits)))

    def decode(self, key):
        digits = []
        for shift in range(56, -1, -4):
            nibble = (key >> shift) & 15
            if nibble == 0:
                break
            if nibble > 10:
                raise ValueError("invalid lexical document key")
            digits.append(str(nibble - 1))
        if not digits or self.prefix is None:
            raise ValueError("empty lexical document key")
        return self.prefix + "".join(digits)


class TensorTopK:
    """Float64 scores, deterministic score-descending/ID-ascending ties."""

    def __init__(self, count, device, k=100):
        import torch
        self.k = k
        self.scores = torch.empty((count, 0), dtype=torch.float64, device=device)
        self.keys = torch.empty((count, 0), dtype=torch.int64, device=device)

    def add(self, scores, keys):
        import torch
        if scores.ndim != 2 or scores.shape[0] != self.scores.shape[0] or scores.shape[1] != len(keys):
            raise ValueError("ranking score shape differs from query/document IDs")
        if scores.dtype != torch.float64 or not bool(torch.isfinite(scores).all()):
            raise ValueError("ranking requires finite float64 scores")
        values = torch.cat((self.scores, scores), dim=1)
        ids = torch.cat((self.keys, keys.expand(scores.shape[0], -1)), dim=1)
        by_id = torch.argsort(ids, dim=1, stable=True)
        values, ids = values.gather(1, by_id), ids.gather(1, by_id)
        by_score = torch.argsort(values, dim=1, descending=True, stable=True)[:, :self.k]
        self.scores, self.keys = values.gather(1, by_score), ids.gather(1, by_score)

    def rows(self, codec):
        return [[codec.decode(key) for key in row] for row in self.keys.cpu().tolist()]


def tensor_vectors(values, device):
    import torch
    vectors = torch.tensor(values, dtype=torch.float64, device=device)
    if vectors.ndim != 2 or vectors.shape[1] != 768 or not bool(torch.isfinite(vectors).all()):
        raise ValueError("embedding endpoint must return finite full 768-vectors")
    if bool((vectors.norm(dim=1) == 0).any()):
        raise ValueError("embedding endpoint returned a zero vector")
    return vectors


def normalized(values, dimension):
    import torch
    vectors = values[:, :dimension]
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    if bool((norms == 0).any()):
        raise ValueError("MRL truncation produced a zero vector")
    return vectors / norms


def evaluate_matrix(*, corpus, queries, qrels, embedders, dimensions=MRL_DIMENSIONS,
                    query_task="RETRIEVAL_QUERY", query_chunk_size=32,
                    corpus_chunk_size=512, device="cuda", progress=None):
    import torch
    if query_chunk_size < 1 or corpus_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    query_rows = list(jsonl(queries))
    query_ids = [record_id(row) for row in query_rows]
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be nonempty and globally unique")
    grades = load_qrels(qrels, set(query_ids))
    wanted_docs = {doc for row in grades.values() for doc, grade in row.items() if grade > 0}
    matrices, rankings = {}, {}
    for model, embed in embedders.items():
        full = torch.empty((len(query_ids), 768), dtype=torch.float64, device=device)
        offset = 0
        for rows in batches(query_rows, query_chunk_size):
            values = tensor_vectors(embed([query_text(row) for row in rows], query_task), device)
            if len(values) != len(rows):
                raise ValueError("query endpoint omitted embeddings")
            full[offset:offset + len(rows)] = values
            offset += len(rows)
        for dimension in dimensions:
            matrices[model, dimension] = normalized(full, dimension)
            rankings[model, dimension] = TensorTopK(len(query_ids), device)
    codec, observed = IntegerDocIds(), 0
    for rows in batches(jsonl(corpus), corpus_chunk_size):
        ids = [record_id(row) for row in rows]
        wanted_docs.difference_update(ids)
        keys = torch.tensor([codec.encode(value) for value in ids], dtype=torch.int64, device=device)
        inputs = [document_input(row) for row in rows]
        for model, embed in embedders.items():
            values = tensor_vectors(embed(inputs, "RETRIEVAL_DOCUMENT"), device)
            if len(values) != len(rows):
                raise ValueError("document endpoint omitted embeddings")
            for dimension in dimensions:
                scores = matrices[model, dimension] @ normalized(values, dimension).T
                rankings[model, dimension].add(scores, keys)
        observed += len(rows)
        if progress:
            progress(observed)
    if not observed or wanted_docs:
        raise ValueError("positive qrels reference documents absent from the full corpus")
    result = {}
    for key, top in rankings.items():
        arrays = {metric: [] for metric in METRICS}
        for query_id, ranked in zip(query_ids, top.rows(codec), strict=True):
            for name, value in metrics(ranked, grades[query_id]).items():
                arrays[name].append(value)
        result[key] = arrays
    return query_ids, result, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "queries", "qrels", "models", "prepared-manifest",
                 "training-source-manifest", "source-manifest", "processor-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--language")
    parser.add_argument("--role", choices=("target", "retention"), required=True)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--corpus-chunk-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    inventories = validate_official_source_manifest(load_json(args.source_manifest))
    validate_reviewed_inventory(inventories)
    official = inventories[args.dataset, args.language, args.role]
    actual = {"corpus_count": count_unique(args.corpus), "query_count": count_unique(args.queries),
              **{name + "_sha256": file_sha256(getattr(args, name)) for name in ("corpus", "queries", "qrels")}}
    if any(official[name] != value for name, value in actual.items()):
        raise ValueError("local full-corpus artifacts differ from the pinned inventory")
    models = load_json(args.models)
    if set(models) != {"base", *map(str, QUALITY_SEEDS)}:
        raise ValueError("model manifest requires the base and exact three seeds")
    shared = {name: artifact(getattr(args, name)) for name in
              ("prepared_manifest", "training_source_manifest", "source_manifest")}
    seed_artifacts = {}
    for seed in QUALITY_SEEDS:
        model = models[str(seed)]
        bundle = Path(model['bundle_manifest'])
        weight = bundle.parent / "model.safetensors"
        checkpoint = Path(model['checkpoint'])
        validated = validate_candidate_artifacts(checkpoint=checkpoint, model=weight,
            prepared_manifest=args.prepared_manifest, training_source_manifest=args.training_source_manifest, seed=seed)
        artifacts = {**shared, "checkpoint": artifact(checkpoint), "model": artifact(weight), "bundle_manifest": artifact(bundle)}
        if any(validated['candidate_' + name + '_sha256'] != artifacts[name]['sha256'] for name in ("checkpoint", "model")):
            raise ValueError("candidate artifacts differ from export provenance")
        if validated['candidate_bundle_manifest_sha256'] != artifacts['bundle_manifest']['sha256']:
            raise ValueError("candidate bundle differs from export provenance")
        seed_artifacts[seed] = artifacts
    from transformers import AutoTokenizer
    from finetune_reference import TOKENIZER_SHA256
    if file_sha256(args.processor_dir / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("quality tokenizer differs from the pinned contract")
    tokenizer = AutoTokenizer.from_pretrained(args.processor_dir, local_files_only=True)
    if file_sha256(args.processor_dir / "model.safetensors") != BASE_WEIGHT_SHA256:
        raise ValueError("quality base endpoint bundle differs from the pinned weights")
    identities = {'base': embedding_identity(args.processor_dir, weights_digest=BASE_WEIGHT_SHA256)}
    for seed in QUALITY_SEEDS:
        identities[str(seed)] = embedding_identity(Path(models[str(seed)]['bundle_manifest']).parent,
                                                   weights_digest=seed_artifacts[seed]['model']['sha256'])
    embedders = {key: HttpEmbedder(value['url'], value['model'], args.timeout,
                                 limiter=TextInputLimiter(tokenizer), expected_identity=identities[key])
                 for key, value in models.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_task = "CODE_RETRIEVAL" if args.dataset == "codesearchnet" else "RETRIEVAL_QUERY"

    def progress(count):
        row = {"dataset": args.dataset, "language": args.language,
               "corpus_rows": count, "corpus_count": official['corpus_count']}
        print(json.dumps(row), file=sys.stderr, flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps(row) + "\n")

    ids, values, count = evaluate_matrix(corpus=args.corpus, queries=args.queries, qrels=args.qrels,
        embedders=embedders, query_task=query_task, query_chunk_size=args.query_chunk_size,
        corpus_chunk_size=args.corpus_chunk_size, progress=progress)
    if count != official['corpus_count'] or len(ids) != official['query_count']:
        raise ValueError("full-corpus evaluation coverage is incomplete")
    if any(embed.provenance != {"cuda"} for embed in embedders.values()):
        raise ValueError("quality evaluation lacks CUDA endpoint provenance")
    if any(file_sha256(getattr(args, name)) != actual[name + '_sha256'] for name in ("corpus", "queries", "qrels")):
        raise ValueError("official input artifacts changed during evaluation")
    for seed in QUALITY_SEEDS:
        for dimension in MRL_DIMENSIONS:
            base, candidate = values['base', dimension], values[str(seed), dimension]
            cell = {"schema": "antfly.embedding_gemma2.finetune_quality_cell.v1",
                "qualification_scope": "official-full-corpus", "dataset": args.dataset,
                "language": args.language, "role": args.role, "dimension": dimension, "seed": seed,
                "base_query_ids": ids, "candidate_query_ids": ids, "base_per_query": base,
                "candidate_per_query": candidate, "official_corpus": official,
                "base_mean": {name: statistics.fmean(base[name]) for name in METRICS},
                "candidate_mean": {name: statistics.fmean(candidate[name]) for name in METRICS},
                "artifacts": seed_artifacts[seed], "execution": {
                    "backend": "cuda", "scoring_device": "cuda", "scoring_dtype": "float64",
                    "corpus_passes": 1, "corpus_chunk_size": args.corpus_chunk_size,
                    "query_chunk_size": args.query_chunk_size, "query_task": query_task,
                    "full_vector_dimension": 768, "mrl": "truncate-then-l2-normalize",
                    "text_preprocessing": "retain-rendered-text-prefix-with-bos-eos-v1",
                    "query_cap": 128, "document_cap": 512,
                    "endpoints": models,
                    "endpoint_identities": identities,
                    "truncated_inputs": {key: embed.limiter.truncated for key, embed in embedders.items()},
                }}
            path = args.output_dir / f"seed-{seed}-dimension-{dimension}.json"
            with path.open("x") as output:
                output.write(json.dumps(cell, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
