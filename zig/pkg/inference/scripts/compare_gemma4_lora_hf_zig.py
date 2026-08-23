#!/usr/bin/env python3
"""Validate and compare pinned Gemma4 LoRA oracle evidence.

This command never downloads a model.  ``compare`` consumes two complete
versioned traces (normally HF/PEFT and Antfly Zig), proves their discrete
contracts are identical, then applies the named numerical tolerance profile.
``compare-adapters`` separately checks PEFT/Antfly adapter inventories and
payload values rather than relying on serializer byte equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gemma4_oracle_contract import (
    ContractError,
    LOCK_PATH,
    compare_traces,
    inspect_adapter_artifact,
    load_lock,
    load_prepared_example,
    lock_digest,
    validate_evidence_ledger,
    validate_trace,
    validate_target_inventory,
    vector_metrics,
    verify_model_directory,
    verify_prepared_source_dataset,
    verify_requirements_match_lock,
    write_json,
)


def emit(payload: Any, output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    else:
        write_json(output, payload)


def command_validate_lock(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    script_dir = args.lock.resolve().parent
    oracle_requirements = script_dir / "requirements-gemma4-oracle.txt"
    mlx_requirements = script_dir / "requirements-gemma4-mlx-reference.txt"
    verify_requirements_match_lock(lock, "python_oracle", oracle_requirements)
    verify_requirements_match_lock(lock, "mlx_reference", mlx_requirements)
    return {
        "schema_version": lock["schema_version"],
        "lock": str(args.lock.resolve()),
        "sha256": lock_digest(args.lock),
        "models": {
            key: {"repo_id": value["repo_id"], "revision": value["revision"]}
            for key, value in lock["models"].items()
        },
        "requirements": {
            "python_oracle": str(oracle_requirements),
            "mlx_reference": str(mlx_requirements),
        },
        "ok": True,
    }


def command_validate_model(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    return {"ok": True, **verify_model_directory(lock, args.model_key, args.model_dir)}


def command_validate_prepared(args: argparse.Namespace) -> dict[str, Any]:
    summary, prepared = load_prepared_example(args.prepared, args.example_index)
    source = verify_prepared_source_dataset(summary, args.source_dataset)
    return {
        "ok": True,
        "prepared": str(args.prepared.resolve()),
        "selected": prepared,
        "source_dataset": source,
        "supervised_tokens": sum(label != -100 for label in prepared["labels"][1:]),
    }


def command_validate_trace(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    trace = validate_trace(args.trace, lock, lock_path=args.lock)
    return {
        "ok": True,
        "trace": str(trace.path),
        "trace_sha256": trace.trace_sha256,
        "evidence_manifest_sha256": trace.evidence_manifest_sha256,
        "producer": trace.payload["producer"],
        "model": trace.payload["model"],
        "target_tensor_count": len(trace.payload["target_tensors"]),
        "supervised_tokens": trace.payload["metrics"]["supervised_tokens"],
        "recomputed_grad_norm": trace.recomputed_grad_norm,
    }


def command_compare(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    reference = validate_trace(args.reference, lock, lock_path=args.lock)
    candidate = validate_trace(args.candidate, lock, lock_path=args.lock)
    return compare_traces(reference, candidate, lock, args.profile)


def command_inspect_adapter(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    artifact = inspect_adapter_artifact(args.adapter, target_preset=args.target_preset)
    validate_target_inventory(
        lock,
        args.model_key,
        artifact["semantics"]["target_preset"],
        artifact["semantics"]["target_modules"],
    )
    return {
        "ok": True,
        "directory": artifact["directory"],
        "adapter_model_sha256": artifact["adapter_model_sha256"],
        "semantics": artifact["semantics"],
        "configured_target_modules": artifact["configured_target_modules"],
        "policy_source": artifact["policy_source"],
        "provenance": artifact["provenance"],
        "key_layout": artifact["key_layout"],
        "direct_stock_peft_load_compatibility": "unproven; run the real-model load gate",
        "inventory": artifact["inventory"],
        "tensors": {
            f"{module}:{role}": {
                "source_name": tensor["source_name"],
                "shape": tensor["shape"],
                "dtype": tensor["dtype"],
            }
            for (module, role), tensor in sorted(artifact["tensors"].items())
        },
    }


def command_compare_adapters(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    if args.profile not in lock["tolerance_profiles"]:
        raise ContractError(f"unknown tolerance profile: {args.profile}")
    tolerance = lock["tolerance_profiles"][args.profile]
    reference_root = args.reference.expanduser().resolve().parent
    candidate_root = args.candidate.expanduser().resolve().parent
    reference_publication_sha, reference_files = validate_evidence_ledger(reference_root / "trace.json")
    candidate_publication_sha, candidate_files = validate_evidence_ledger(candidate_root / "trace.json")
    for label, adapter_dir, files in (
        ("reference", args.reference.expanduser().resolve(), reference_files),
        ("candidate", args.candidate.expanduser().resolve(), candidate_files),
    ):
        prefix = adapter_dir.name + "/"
        if not any(path.startswith(prefix) for path in files):
            raise ContractError(f"{label} adapter is not committed by its COMPLETE.json ledger")
    reference = inspect_adapter_artifact(args.reference, target_preset=args.reference_target_preset)
    candidate = inspect_adapter_artifact(args.candidate, target_preset=args.candidate_target_preset)
    validate_target_inventory(
        lock,
        args.model_key,
        reference["semantics"]["target_preset"],
        reference["semantics"]["target_modules"],
    )
    validate_target_inventory(
        lock,
        args.model_key,
        candidate["semantics"]["target_preset"],
        candidate["semantics"]["target_modules"],
    )
    failures: list[str] = []
    semantics_equal = reference["semantics"] == candidate["semantics"]
    if not semantics_equal:
        failures.append("adapter config semantics differ")
    inventory_equal = reference["inventory"] == candidate["inventory"]
    if not inventory_equal:
        failures.append("adapter tensor inventories differ")
    tensor_rows = []
    identities = sorted(set(reference["tensors"]) & set(candidate["tensors"]))
    for identity in identities:
        left = reference["tensors"][identity]
        right = candidate["tensors"][identity]
        if left["shape"] != right["shape"]:
            failures.append(f"shape differs for {identity[0]}:{identity[1]}")
            continue
        metrics = vector_metrics(left["values"], right["values"])
        ok = (
            metrics.max_abs <= tolerance["state_max_abs"]
            and metrics.rel_l2 <= tolerance["state_rel_l2"]
            and metrics.cosine >= tolerance["state_cosine_min"]
        )
        tensor_rows.append({
            "canonical_name": identity[0],
            "role": identity[1],
            "shape": left["shape"],
            "rel_l2": metrics.rel_l2,
            "cosine": metrics.cosine,
            "max_abs": metrics.max_abs,
            "max_abs_limit": tolerance["state_max_abs"],
            "ok": ok,
        })
        if not ok:
            failures.append(f"payload differs for {identity[0]}:{identity[1]}")
    return {
        "schema_version": "antfly_gemma4_adapter_comparison/v1",
        "ok": not failures,
        "profile": args.profile,
        "reference": {
            "directory": reference["directory"],
            "adapter_model_sha256": reference["adapter_model_sha256"],
            "publication_manifest_sha256": reference_publication_sha,
        },
        "candidate": {
            "directory": candidate["directory"],
            "adapter_model_sha256": candidate["adapter_model_sha256"],
            "publication_manifest_sha256": candidate_publication_sha,
        },
        "byte_hashes_equal": reference["adapter_model_sha256"] == candidate["adapter_model_sha256"],
        "reference_key_layout": reference["key_layout"],
        "candidate_key_layout": candidate["key_layout"],
        "canonical_name_normalization_applied": reference["key_layout"] != candidate["key_layout"],
        "direct_bidirectional_interoperability_proven": False,
        "semantics_equal": semantics_equal,
        "inventory_equal": inventory_equal,
        "tensors": tensor_rows,
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--lock", type=Path, default=LOCK_PATH)
    root.add_argument("--output", type=Path, help="write a new JSON result instead of stdout")
    commands = root.add_subparsers(dest="command", required=True)

    validate_lock = commands.add_parser("validate-lock", help="validate lock and requirements synchronization")
    validate_lock.set_defaults(handler=command_validate_lock)

    validate_model = commands.add_parser("validate-model", help="hash and validate a local pinned model snapshot")
    validate_model.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    validate_model.add_argument("--model-dir", type=Path, required=True)
    validate_model.set_defaults(handler=command_validate_model)

    validate_prepared = commands.add_parser("validate-prepared", help="validate one exact text-only prepared example")
    validate_prepared.add_argument("--prepared", type=Path, required=True)
    validate_prepared.add_argument(
        "--source-dataset",
        type=Path,
        help="override the recorded v6 source path while verifying its exact digest",
    )
    validate_prepared.add_argument("--example-index", type=int, default=0)
    validate_prepared.set_defaults(handler=command_validate_prepared)

    validate_one_trace = commands.add_parser(
        "validate-trace",
        help="validate a COMPLETE.json-committed release numerical trace",
    )
    validate_one_trace.add_argument("--trace", type=Path, required=True)
    validate_one_trace.set_defaults(handler=command_validate_trace)

    compare = commands.add_parser("compare", help="compare HF/PEFT and Zig traces")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument(
        "--profile",
        default="hf-zig-bf16",
        choices=("tiny-f32", "native-metal-bf16", "hf-zig-bf16", "resume"),
    )
    compare.set_defaults(handler=command_compare)

    inspect_adapter = commands.add_parser("inspect-adapter", help="validate one PEFT-format adapter")
    inspect_adapter.add_argument("--adapter", type=Path, required=True)
    inspect_adapter.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    inspect_adapter.add_argument("--target-preset", choices=("peft-qv", "text-all-linear"), help="required policy for a stock PEFT adapter without an Antfly manifest")
    inspect_adapter.set_defaults(handler=command_inspect_adapter)

    compare_adapters = commands.add_parser(
        "compare-adapters",
        help="compare adapters committed by COMPLETE.json release publications",
    )
    compare_adapters.add_argument("--reference", type=Path, required=True)
    compare_adapters.add_argument("--candidate", type=Path, required=True)
    compare_adapters.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    compare_adapters.add_argument(
        "--profile",
        default="hf-zig-bf16",
        choices=("tiny-f32", "native-metal-bf16", "hf-zig-bf16", "resume"),
    )
    compare_adapters.add_argument("--reference-target-preset", choices=("peft-qv", "text-all-linear"))
    compare_adapters.add_argument("--candidate-target-preset", choices=("peft-qv", "text-all-linear"))
    compare_adapters.set_defaults(handler=command_compare_adapters)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = args.handler(args)
        emit(result, args.output)
        return 0 if result.get("ok", True) else 1
    except ContractError as exc:
        print(f"Gemma4 oracle contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
