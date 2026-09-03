#!/usr/bin/env python3
"""Generate deterministic, group-disjoint Gemma4 chat/tool contract data.

The checked-in case specification covers plain/system/multi-turn chat,
single/parallel tool calls, Unicode normalization hazards, escaped JSON, and
511/512/513-character boundary payloads.  Exact token-boundary fixtures are
created later by the pinned tokenizer and must be compared to Antfly's
prepared arrays; character length is not presented as token length here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from gemma4_oracle_contract import ContractError, load_json, prefixed_sha256, require_exact_keys, write_json


DEFAULT_SPEC = (
    Path(__file__).parent.parent.parent
    / "testdata"
    / "gemma4_oracle"
    / "contract_cases.json"
)
SPLITS = ("train", "eval", "test")
CODE_WORDS = ("amber", "cobalt", "indigo", "saffron", "violet", "zircon")
CITIES = ("Accra", "Lima", "Osaka", "Reykjavik", "Tunis", "Zürich")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest_atom(seed: int, split: str, index: int, case_name: str) -> str:
    payload = f"antfly-gemma4-oracle-fixture/v1\0{seed}\0{split}\0{index}\0{case_name}".encode()
    return hashlib.sha256(payload).hexdigest()


def substitute(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for needle, replacement in replacements.items():
            result = result.replace("{{" + needle + "}}", replacement)
        if "{{" in result or "}}" in result:
            raise ContractError(f"unresolved fixture placeholder: {result!r}")
        return result
    if isinstance(value, list):
        return [substitute(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, replacements) for key, item in value.items()}
    return value


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise ContractError("fixture case spec must be an object")
    require_exact_keys(payload, ("schema_version", "cases"), where="fixture case spec")
    if payload["schema_version"] != "antfly_gemma4_contract_cases/v1":
        raise ContractError("unsupported fixture case schema")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ContractError("fixture case spec must contain cases")
    cases: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_case in enumerate(payload["cases"]):
        if not isinstance(raw_case, Mapping):
            raise ContractError(f"cases[{index}] must be an object")
        allowed = {"name", "messages", "tools"}
        unknown = set(raw_case) - allowed
        missing = {"name", "messages"} - set(raw_case)
        if unknown or missing:
            raise ContractError(f"cases[{index}] field mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})")
        name = raw_case["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ContractError(f"cases[{index}] has an invalid or duplicate name")
        names.add(name)
        messages = raw_case["messages"]
        if not isinstance(messages, list) or not messages:
            raise ContractError(f"cases[{index}] must contain messages")
        if not any(isinstance(message, Mapping) and message.get("role") == "assistant" for message in messages):
            raise ContractError(f"cases[{index}] has no assistant supervision")
        cases.append(dict(raw_case))
    return cases


def generate_rows(spec_path: Path, seed: int, counts: Mapping[str, int]) -> tuple[bytes, dict[str, Any]]:
    cases = load_cases(spec_path)
    rows: list[dict[str, Any]] = []
    row_hashes: list[str] = []
    group_ids: set[str] = set()
    for split in SPLITS:
        count = counts[split]
        if count <= 0:
            raise ContractError(f"{split} count must be positive")
        # A split-specific rotation ensures all case families appear without
        # sharing an example or group identity across partitions.
        rotation = int(hashlib.sha256(f"{seed}:{split}".encode()).hexdigest()[:8], 16) % len(cases)
        for index in range(count):
            case = cases[(rotation + index) % len(cases)]
            atom = digest_atom(seed, split, index, case["name"])
            nonce = atom[:12]
            boundary_length = (511, 512, 513)[index % 3]
            replacements = {
                "nonce": nonce,
                "code_word": CODE_WORDS[int(atom[12:20], 16) % len(CODE_WORDS)],
                "city": CITIES[int(atom[20:28], 16) % len(CITIES)],
                "boundary_payload": "x" * boundary_length,
            }
            group_id = f"gemma4-oracle-{split}-{atom[28:44]}"
            if group_id in group_ids:
                raise ContractError("deterministic group id collision")
            group_ids.add(group_id)
            row: dict[str, Any] = {
                "schema": "gemma_chat/v1",
                "id": f"{split}-{index:06d}-{case['name']}",
                "split": split,
                "messages": substitute(case["messages"], replacements),
                "metadata": {
                    "policy_version": "gemma4-oracle-contract/v1",
                    "source": "generated-contract-fixture",
                    "group_id": group_id,
                },
            }
            if "tools" in case:
                row["tools"] = substitute(case["tools"], replacements)
            rows.append(row)
            row_hashes.append(hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest())
    lines = [canonical_json(row) for row in rows]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    split_groups = {
        split: sorted(row["metadata"]["group_id"] for row in rows if row["split"] == split)
        for split in SPLITS
    }
    if any(set(split_groups[left]) & set(split_groups[right]) for left in SPLITS for right in SPLITS if left < right):
        raise ContractError("generated split groups overlap")
    manifest = {
        "schema_version": "antfly_gemma4_oracle_fixture_manifest/v1",
        "generator_policy": "sha256-derived-no-random-runtime-state",
        "seed": seed,
        "counts": dict(counts),
        "case_spec_sha256": prefixed_sha256(spec_path.resolve()),
        "dataset_sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "row_count": len(rows),
        "row_sha256": row_hashes,
        "split_group_sha256": {
            split: "sha256:" + hashlib.sha256("\n".join(split_groups[split]).encode()).hexdigest()
            for split in SPLITS
        },
    }
    return data, manifest


def write_exclusive(path: Path, data: bytes) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ContractError(f"refusing to replace existing fixture: {target}") from exc
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--manifest", type=Path, help="default: OUTPUT.manifest.json")
    result.add_argument("--seed", type=int, default=20260810)
    result.add_argument("--train-count", type=int, default=32)
    result.add_argument("--eval-count", type=int, default=8)
    result.add_argument("--test-count", type=int, default=8)
    result.add_argument("--check", action="store_true", help="verify existing output and manifest without writing")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    try:
        data, manifest = generate_rows(
            args.spec.resolve(),
            args.seed,
            {"train": args.train_count, "eval": args.eval_count, "test": args.test_count},
        )
        expected_manifest = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if args.check:
            if args.output.read_bytes() != data:
                raise ContractError(f"generated fixture drifted: {args.output}")
            if manifest_path.read_bytes() != expected_manifest:
                raise ContractError(f"generated fixture manifest drifted: {manifest_path}")
        else:
            write_exclusive(args.output, data)
            write_json(manifest_path, manifest)
        print(json.dumps({
            "ok": True,
            "check": args.check,
            "output": str(args.output.resolve()),
            "manifest": str(manifest_path.resolve()),
            "dataset_sha256": manifest["dataset_sha256"],
            "row_count": manifest["row_count"],
        }, sort_keys=True))
        return 0
    except (ContractError, OSError) as exc:
        print(f"Gemma4 fixture generation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
