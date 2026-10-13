#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Materialize all five immutable official retrieval inventories without sampling.

BEIR evaluation queries include every query with a positive qrel in the reviewed
evaluation split. CodeSearchNet uses every test function as a corpus document and
every nonempty test docstring as a query with its source function as the positive.
Source file bytes are verified against the pinned Hub tree before conversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path

from finetune_contract import canonical_hash, file_sha256, validate_official_source_manifest
from qualify_finetune_quality import OFFICIAL_SOURCE_CONTRACTS, validate_reviewed_inventory


def row_id(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError("source IDs must be nonempty strings or integers")
    return str(value)


def qrel(row: dict) -> dict:
    score = row.get("score")
    if isinstance(score, bool):
        raise ValueError("qrel grades must be finite and nonnegative")
    score = float(score)
    if not math.isfinite(score) or score < 0:
        raise ValueError("qrel grades must be finite and nonnegative")
    return {"query_id": row_id(row.get("query-id", row.get("query_id"))),
            "corpus_id": row_id(row.get("corpus-id", row.get("corpus_id"))), "score": score}


def write_row(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def write_unique_rows(path: Path, rows, *, positive_documents: set[str] | None = None) -> int:
    """Use a disk index so millions of corpus IDs cannot consume host RAM."""
    index_path = path.with_suffix(".ids.sqlite")
    count = 0
    index = sqlite3.connect(index_path)
    try:
        index.execute("PRAGMA journal_mode=OFF")
        index.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                identifier = row_id(row.get("id"))
                try:
                    index.execute("INSERT INTO ids VALUES (?)", (identifier,))
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"duplicate source ID {identifier!r}") from error
                if positive_documents is not None:
                    positive_documents.discard(identifier)
                write_row(stream, row)
                count += 1
                if count % 100_000 == 0:
                    print(json.dumps({"artifact": path.parent.name + "/" + path.name, "rows": count}), flush=True)
        if count == 0:
            raise ValueError("official artifact is empty")
        if positive_documents:
            raise ValueError("positive qrels reference documents absent from the official corpus")
        return count
    finally:
        index.close()
        index_path.unlink(missing_ok=True)


def verify_source(path: Path, metadata) -> str:
    digest = file_sha256(path)
    lfs = metadata.lfs
    if lfs is not None:
        expected = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
        if digest != expected:
            raise ValueError(f"downloaded source digest differs from pinned tree: {metadata.rfilename}")
    else:
        git_digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                git_digest.update(chunk)
        if git_digest.hexdigest() != metadata.blob_id:
            raise ValueError(f"downloaded source blob differs from pinned tree: {metadata.rfilename}")
    return digest


class Sources:
    def __init__(self, cache: Path):
        from huggingface_hub import HfApi
        self.api, self.cache = HfApi(), cache
        self.infos, self.sources, self.artifacts = {}, {}, {}

    def files(self, repo: str, revision: str, config: str, split: str) -> list[Path]:
        from huggingface_hub import hf_hub_download
        key = (repo, revision)
        if key not in self.infos:
            info = self.api.dataset_info(repo, revision=revision, files_metadata=True, token=False)
            if info.sha != revision:
                raise ValueError("Hub source revision differs from the reviewed immutable pin")
            self.infos[key] = info
        info = self.infos[key]
        files = sorted((item for item in info.siblings if
                        item.rfilename.startswith(f"{config}/{split}-") and item.rfilename.endswith(".parquet")
                        or config == "default" and (item.rfilename == f"{split}.tsv"
                            or item.rfilename.startswith(f"data/{split}-") and item.rfilename.endswith(".parquet"))),
                       key=lambda item: item.rfilename)
        if not files:
            raise ValueError(f"reviewed source split is missing: {repo}/{config}/{split}")
        paths, identities = [], []
        for item in files:
            artifact_key = (repo, revision, item.rfilename)
            if artifact_key not in self.artifacts:
                path = Path(hf_hub_download(repo, item.rfilename, repo_type="dataset", revision=revision,
                                           cache_dir=self.cache, token=False))
                self.artifacts[artifact_key] = {"path": path, "repo": repo, "revision": revision,
                                               "filename": item.rfilename, "sha256": verify_source(path, item),
                                               "bytes": path.stat().st_size}
            identity = self.artifacts[artifact_key]
            paths.append(identity["path"])
            identities.append({name: value for name, value in identity.items() if name != "path"})
        license_value = info.card_data.to_dict().get("license", "unspecified") if info.card_data else "unspecified"
        license_text = ",".join(license_value) if isinstance(license_value, list) else str(license_value)
        self.sources[(repo, revision, config)] = {"source_repo": repo, "source_revision": revision,
                                                "config": config, "license": license_text,
                                                "snapshot_sha256": canonical_hash(identities)}
        return paths


def records(paths: list[Path]):
    import pyarrow.parquet as parquet
    for path in paths:
        if path.suffix == ".tsv":
            with path.open(encoding="utf-8", newline="") as stream:
                yield from csv.DictReader(stream, delimiter="\t")
        else:
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=8192):
                yield from batch.to_pylist()


def materialize_beir(directory: Path, sources: Sources, contract: dict) -> tuple[int, int]:
    label_paths = sources.files(contract["qrels_source_repo"], contract["qrels_source_revision"],
                                contract["qrels_config"], contract["qrels_split"])
    query_ids, positive_ids = set(), set()
    with (directory / "qrels.jsonl").open("x", encoding="utf-8") as stream:
        for raw in records(label_paths):
            row = qrel(raw)
            write_row(stream, row)
            if row["score"] > 0:
                query_ids.add(row["query_id"])
                positive_ids.add(row["corpus_id"])
    corpus_paths = sources.files(contract["source_repo"], contract["source_revision"], contract["corpus_config"], contract["corpus_split"])
    corpus = ({"id": row_id(row["_id"]), "title": row.get("title", ""), "text": row["text"]} for row in records(corpus_paths))
    corpus_count = write_unique_rows(directory / "corpus.jsonl", corpus, positive_documents=positive_ids)
    query_paths = sources.files(contract["source_repo"], contract["source_revision"], contract["queries_config"], contract["queries_split"])
    queries = ({"id": row_id(row["_id"]), "text": row["text"]} for row in records(query_paths) if row_id(row["_id"]) in query_ids)
    query_count = write_unique_rows(directory / "queries.jsonl", queries)
    if query_count != len(query_ids):
        raise ValueError("official query artifact does not cover every positively judged evaluation query")
    return corpus_count, query_count


def materialize_code(directory: Path, sources: Sources, contract: dict, language: str) -> tuple[int, int]:
    paths = sources.files(contract["source_repo"], contract["source_revision"], language, "test")
    corpus_count, query_count = 0, 0
    with (directory / "corpus.jsonl").open("x", encoding="utf-8") as corpus, \
            (directory / "queries.jsonl").open("x", encoding="utf-8") as queries, \
            (directory / "qrels.jsonl").open("x", encoding="utf-8") as labels:
        for index, row in enumerate(records(paths)):
            identifier = f"codesearchnet:{language}:test:{index:08d}"
            code, query = row.get("func_code_string"), row.get("func_documentation_string")
            if not isinstance(code, str) or not code:
                raise ValueError("official CodeSearchNet test function has no code")
            write_row(corpus, {"id": identifier, "text": code})
            corpus_count += 1
            if isinstance(query, str) and query.strip():
                write_row(queries, {"id": identifier, "text": query})
                write_row(labels, {"query_id": identifier, "corpus_id": identifier, "score": 1})
                query_count += 1
    if not corpus_count or not query_count:
        raise ValueError("official CodeSearchNet test split is empty")
    return corpus_count, query_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("official output directory must be new")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".official-retrieval-", dir=args.output_dir.parent))
    sources = Sources(args.hub_cache or args.output_dir.parent / "hub-public-data-cache")
    inventories = []
    try:
        # Materialize the small inventories first; MS MARCO remains complete.
        for key in sorted(OFFICIAL_SOURCE_CONTRACTS, key=lambda value: (value[0] == "msmarco", str(value))):
            dataset, language, role = key
            contract = OFFICIAL_SOURCE_CONTRACTS[key]
            directory = staging / (dataset + ("-" + language if language else ""))
            directory.mkdir()
            counts = (materialize_code(directory, sources, contract, language) if language else
                      materialize_beir(directory, sources, contract))
            official = {**contract, "corpus_count": counts[0], "query_count": counts[1],
                        **{name + "_sha256": file_sha256(directory / (name + ".jsonl")) for name in ("corpus", "queries", "qrels")}}
            inventories.append({"dataset": dataset, "language": language, "role": role, "official_corpus": official})
            print(json.dumps({"dataset": dataset, "language": language, "corpus_count": counts[0], "query_count": counts[1]}), flush=True)
        manifest = {"schema": "antfly.embedding_gemma2.official_retrieval_sources.v1",
                    "query_selection_policy": "all-positive-qrels-queries;all-nonempty-test-code-docstrings",
                    "official_inventories": inventories, "sources": list(sources.sources.values()),
                    "source_files": [{name: value for name, value in item.items() if name != "path"} for item in sources.artifacts.values()]}
        validate_reviewed_inventory(validate_official_source_manifest(manifest))
        (staging / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if args.output_dir.exists():
            raise FileExistsError("official output directory appeared during materialization")
        staging.rename(args.output_dir)
        print(json.dumps({"output": str(args.output_dir), "complete": True, "inventories": len(inventories)}), flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
