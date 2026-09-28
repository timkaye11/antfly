#!/usr/bin/env python3
"""Deterministically reorder a Gemma chat training JSONL without changing records.

The SFT trainer consumes prepared examples in file order. Balanced selection in
source order can leave long runs of one label. This preprocessing step assigns
an order from the seed and source IDs, independent of labels or input order.
Run it on the training split before the public dataset prepare command. It
does not select examples, change supervision, or shuffle once per epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOMAIN = "gemma4-sft-order/v1"


class DatasetOrderError(ValueError):
    """The input cannot be reordered without an ambiguous record identity."""


def ordered_records(payload: bytes, seed: int) -> tuple[bytes, list[str]]:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise DatasetOrderError("seed must be an unsigned 64-bit integer")
    records: list[tuple[bytes, str, bytes]] = []
    seen: set[str] = set()
    # Split only JSONL line feeds; Unicode separators inside strings are data.
    lines = payload.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    for number, raw in enumerate(lines, 1):
        if not raw.strip():
            raise DatasetOrderError(f"line {number}: empty record")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise DatasetOrderError(f"line {number}: invalid UTF-8 JSON") from exc
        if not isinstance(row, dict) or row.get("schema") != "gemma_chat/v1":
            raise DatasetOrderError(f"line {number}: expected gemma_chat/v1")
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise DatasetOrderError(f"line {number}: nonempty string id required")
        if source_id in seen:
            raise DatasetOrderError(f"line {number}: duplicate id {source_id!r}")
        seen.add(source_id)
        key = hashlib.sha256(f"{DOMAIN}:{seed}:{source_id}".encode()).digest()
        records.append((key, source_id, raw))
    if not records:
        raise DatasetOrderError("training dataset is empty")
    # ID is an explicit tie-breaker, so order never depends on input position.
    records.sort(key=lambda item: (item[0], item[1]))
    return b"".join(raw + b"\n" for _, _, raw in records), [
        sid for _, sid, _ in records
    ]


def materialize(dataset: Path, output: Path, seed: int) -> dict:
    source = dataset.resolve(strict=True)
    payload = source.read_bytes()
    reordered, ids = ordered_records(payload, seed)
    manifest = {
        "schema_version": "antfly_gemma4_sft_dataset_order/v1",
        "algorithm": f"ascending SHA256 of UTF8 {DOMAIN}:<seed>:<id>; id tie-breaker",
        "seed": seed,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "dataset_path": "train.jsonl",
        "dataset_sha256": hashlib.sha256(reordered).hexdigest(),
        "examples": len(ids),
        "ordered_source_ids": ids,
        "record_policy": "preserve every JSON record byte; terminate each record with LF",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "one fixed training order; no tokenization, selection, or label changes",
    }
    # Exclusive creation protects existing campaigns, including symlink aliases.
    # Validate everything before creating any output; retain partial artifacts
    # on I/O failure instead of removing a directory another writer may use.
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train.jsonl").open("xb") as stream:
        stream.write(reordered)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="training gemma_chat/v1 JSONL"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new immutable output directory"
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        manifest = materialize(args.dataset, args.output, args.seed)
    except (DatasetOrderError, OSError) as exc:
        parser.exit(1, f"SFT dataset order failed: {exc}\n")
    print(
        json.dumps(
            {key: manifest[key] for key in ("examples", "seed", "dataset_sha256")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
