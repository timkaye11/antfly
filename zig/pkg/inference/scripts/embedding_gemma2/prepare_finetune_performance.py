#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepare controlled active-length fixtures for the seven training speed cells.

These fixtures repeat public text to fill the reviewed token caps and assign
distinct document bodies to every positive/negative. They are performance data;
they must never be used as training-quality or official retrieval evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from finetune_contract import file_sha256

CELLS = (
    (16, 32, 128, 0), (32, 32, 128, 0), (64, 32, 128, 0),
    (32, 128, 512, 0), (64, 128, 512, 0),
    (32, 128, 512, 1), (64, 128, 512, 4),
)


def train_source_rows(source: Path) -> list[dict]:
    # Unicode line/paragraph separators may appear inside valid JSON strings.
    # File iteration splits only the actual JSONL record delimiter.
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [row for row in rows if row.get("split", "train") == "train"]


def prepare(source: Path, job_template: Path, native_prepare: Path, initial_adapter: Path,
            output: Path, records: int) -> Path:
    rows = train_source_rows(source)
    if not rows or records < 7040 or records % 64:
        raise ValueError("performance fixtures need train sources and at least 110 full 64-query batches")
    template = json.loads(job_template.read_text())
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "controlled-performance.jsonl"
    with raw.open("w") as stream:
        for index in range(records):
            row = rows[index % len(rows)]
            query = f"Performance query {index}: {row['query'][:512]} " + "retrieval code text document search " * 64

            def document(offset: int) -> dict:
                identifier = index * 5 + offset
                text = rows[identifier % len(rows)]["positive"]["text"][:1024]
                return {"id": f"perf:{identifier}", "text": f"Performance document {identifier}: {text} "
                        + "vector retrieval document inference kernel benchmark data " * 100}

            positive = document(0)
            value = {"query": query, "query_id": f"perf:q:{index}", "group_id": f"perf:g:{index}",
                     "positive": positive, "negatives": [document(offset) for offset in range(1, 5)],
                     "known_positive_ids": [positive["id"]], "task": row["task"], "split": "train"}
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    raw_by_negatives = {4: raw}
    for negatives in (0, 1):
        reduced = output / f"controlled-performance-n{negatives}.jsonl"
        with raw.open(encoding="utf-8") as source_stream, reduced.open("w") as destination:
            for line in source_stream:
                value = json.loads(line)
                value["negatives"] = value["negatives"][:negatives]
                destination.write(json.dumps(value, ensure_ascii=False) + "\n")
        raw_by_negatives[negatives] = reduced
    prepared = {}
    for query_cap, document_cap, negatives in sorted({cell[1:] for cell in CELLS}):
        name = f"q{query_cap}-d{document_cap}-n{negatives}"
        directory = output / ("prepared-" + name)
        log = output / ("prepare-" + name + ".log")
        command = [str(native_prepare.resolve()), template["model_dir"], str(raw_by_negatives[negatives].resolve()), str(directory.resolve()),
                   "--query-max-tokens", str(query_cap), "--document-max-tokens", str(document_cap),
                   "--max-negatives", str(negatives), "--source-repo", "antfly-controlled-performance"]
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=stream, check=True)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["train_records"] != records or manifest["records"] != records:
            raise ValueError("native preparation lost controlled performance records")
        prepared[query_cap, document_cap, negatives] = directory.resolve()
        print(f"Prepared {name}: {records} rows", flush=True)
    cells = []
    for batch, query_cap, document_cap, negatives in CELLS:
        identifier = f"b{batch}-q{query_cap}-d{document_cap}-n{negatives}"
        job = {**template, "logical_batch_size": batch, "microbatch_size": 2,
               "query_max_tokens": query_cap, "document_max_tokens": document_cap,
               "max_negatives": negatives, "epochs": 1,
               "prepared_dir": str(prepared[query_cap, document_cap, negatives]),
               "output_dir": str((output / ("unused-run-" + identifier)).resolve())}
        job_path = output / (identifier + ".json")
        job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
        cells.append({"id": identifier, "job": str(job_path.resolve()),
                      "initial_adapter": str(initial_adapter.resolve())})
    manifest = output / "performance-cells.json"
    manifest.write_text(json.dumps({"cells": cells}, indent=2) + "\n")
    provenance = {"scope": "controlled-repeated-public-text-performance",
                  "production_quality_eligible": False, "records": records,
                  "public_source_sha256": file_sha256(source), "raw_sha256": file_sha256(raw),
                  "raw_by_negatives": {str(count): {"path": str(path.resolve()), "sha256": file_sha256(path)}
                                       for count, path in raw_by_negatives.items()},
                  "generator_sha256": file_sha256(Path(__file__)),
                  "native_prepare_sha256": file_sha256(native_prepare),
                  "length_policy": "append fixed words; native preparation clips to reviewed caps",
                  "candidate_policy": "five distinct document bodies per query; no cross-query deduplication"}
    (output / "performance-fixture-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "job-template", "native-prepare", "initial-adapter", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--records", type=int, default=8192)
    args = parser.parse_args()
    manifest = prepare(args.source, args.job_template, args.native_prepare, args.initial_adapter,
                       args.output_dir, args.records)
    print(manifest)


if __name__ == "__main__":
    main()
