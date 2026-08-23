#!/usr/bin/env python3
"""Evaluate Gemma4 DPO adapters under one pinned MLX oracle.

This command never tokenizes, downloads, or optimizes. It consumes the exact
token case emitted by ``materialize_gemma4_dpo_hf_parity.py``, scores every
candidate against one base-model reference, and reports pairwise behavioral
differences without cross-runtime numerical contamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_gemma4_dpo_mlx_benchmark as dpo  # noqa: E402


RESULT_SCHEMA_VERSION = "antfly_gemma4_dpo_mlx_adapter_evaluation/v2"


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path


def parse_candidate(value: str) -> Candidate:
    label, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", label):
        raise argparse.ArgumentTypeError(
            "candidate must be label=/path with a lowercase stable label"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("candidate path must be non-empty")
    return Candidate(label=label, path=Path(raw_path))


def compare_evaluations(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_rows = left.get("rows")
    right_rows = right.get("rows")
    if not isinstance(left_rows, list) or not isinstance(right_rows, list) or not left_rows:
        raise dpo.DpoBenchmarkContractError("candidate evaluation rows are malformed")
    if len(left_rows) != len(right_rows):
        raise dpo.DpoBenchmarkContractError("candidate evaluation row counts differ")
    loss_deltas = []
    reward_margin_deltas = []
    policy_chosen_deltas = []
    policy_rejected_deltas = []
    decision_agreements = []
    for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows)):
        if left_row.get("index") != index or right_row.get("index") != index:
            raise dpo.DpoBenchmarkContractError("candidate evaluation row order drifted")
        loss_deltas.append(abs(float(left_row["loss"]) - float(right_row["loss"])))
        reward_margin_deltas.append(
            abs(float(left_row["reward_margin"]) - float(right_row["reward_margin"]))
        )
        policy_chosen_deltas.append(
            abs(
                float(left_row["policy_chosen_logp"])
                - float(right_row["policy_chosen_logp"])
            )
        )
        policy_rejected_deltas.append(
            abs(
                float(left_row["policy_rejected_logp"])
                - float(right_row["policy_rejected_logp"])
            )
        )
        decision_agreements.append(
            bool(left_row["preferred"]) == bool(right_row["preferred"])
        )

    right_mean_loss = float(right["mean_loss"])
    right_mean_reward_margin = float(right["mean_reward_margin"])
    return {
        "examples": len(left_rows),
        "preference_decision_agreement": sum(decision_agreements) / len(decision_agreements),
        "accuracy_abs_delta": abs(float(left["accuracy"]) - float(right["accuracy"])),
        "mean_loss_abs_delta": abs(float(left["mean_loss"]) - right_mean_loss),
        "mean_loss_ratio": (
            float(left["mean_loss"]) / right_mean_loss
            if right_mean_loss != 0.0
            else None
        ),
        "mean_reward_margin_abs_delta": abs(
            float(left["mean_reward_margin"]) - right_mean_reward_margin
        ),
        "mean_reward_margin_ratio": (
            float(left["mean_reward_margin"]) / right_mean_reward_margin
            if right_mean_reward_margin != 0.0
            else None
        ),
        "row_loss_mae": sum(loss_deltas) / len(loss_deltas),
        "row_loss_max_abs": max(loss_deltas),
        "row_reward_margin_mae": sum(reward_margin_deltas) / len(reward_margin_deltas),
        "policy_chosen_logp_mae": sum(policy_chosen_deltas) / len(policy_chosen_deltas),
        "policy_rejected_logp_mae": sum(policy_rejected_deltas) / len(policy_rejected_deltas),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = dpo.load_case(args.case)
    if case.schema_version != dpo.DATASET_CASE_SCHEMA_VERSION or case.dataset is None:
        raise dpo.DpoBenchmarkContractError(
            "adapter evaluation requires a provenance-bound dataset case"
        )
    if len(args.candidate) < 2:
        raise dpo.DpoBenchmarkContractError("adapter evaluation requires at least two candidates")
    labels = [candidate.label for candidate in args.candidate]
    if len(set(labels)) != len(labels):
        raise dpo.DpoBenchmarkContractError("candidate labels must be unique")
    candidate_paths = [
        dpo.require_regular_nonsymlink(
            candidate.path,
            f"candidate {candidate.label}",
        )
        for candidate in args.candidate
    ]
    if len(set(candidate_paths)) != len(candidate_paths):
        raise dpo.DpoBenchmarkContractError("candidate paths must be unique")

    lock = dpo.locked.load_lock(args.lock)
    if case.model_key not in dpo.MODEL_KEYS:
        raise dpo.DpoBenchmarkContractError(
            "model_key is outside the Gemma4 DPO matrix"
        )
    if case.target_preset not in lock["target_presets"]:
        raise dpo.DpoBenchmarkContractError(
            "target_preset is outside the locked Gemma4 DPO matrix"
        )
    model_dir = args.model_dir.expanduser().resolve()
    try:
        locked_model = dpo.locked.verify_model_directory(
            lock, case.model_key, model_dir
        )
    except dpo.locked.ContractError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"model directory differs from the locked {case.model_key} artifact: {exc}"
        ) from exc
    mlx_contract = lock["mlx_reference"]
    dpo.locked.force_offline_environment()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != mlx_contract["python"]:
        raise dpo.DpoBenchmarkContractError(
            f"MLX evaluation requires Python {mlx_contract['python']}, found {actual_python}"
        )
    revisions = mlx_contract["source_revisions"]
    mlx_checkout = dpo.require_source_checkout(
        args.mlx_source_root, revisions["mlx"], "MLX"
    )
    mlx_lm_checkout = dpo.require_source_checkout(
        args.mlx_lm_source_root, revisions["mlx-lm"], "MLX-LM"
    )
    mlx_revision = mlx_checkout["revision"]
    mlx_lm_revision = mlx_lm_checkout["revision"]
    if platform.system() != mlx_contract["required_platform"]:
        raise dpo.DpoBenchmarkContractError("MLX evaluation must run on the locked platform")
    if platform.machine() != mlx_contract["required_machine"]:
        raise dpo.DpoBenchmarkContractError("MLX evaluation must run on the locked machine")

    try:
        dpo.locked.verify_requirements_match_lock(
            lock,
            "mlx_reference",
            dpo.locked.MLX_REQUIREMENTS_PATH,
        )
        package_versions = dpo.locked.verify_packages(lock, "mlx_reference")
        preverified_native_bundle = dpo.locked.verify_mlx_native_build_before_import(
            args,
            lock,
            mlx_checkout,
            mlx_lm_checkout,
        )
    except dpo.locked.ContractError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"could not attest the pinned MLX native environment: {exc}"
        ) from exc

    mlx_root = Path(mlx_checkout["path"]).resolve()
    mlx_lm_root = Path(mlx_lm_checkout["path"]).resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(mlx_root / "python"), str(mlx_root)]

    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"could not import the pinned MLX source environment: {exc}"
        ) from exc

    core_path = Path(mx.__file__ or "").resolve()
    if not dpo._path_is_within(core_path, mlx_root):
        raise dpo.DpoBenchmarkContractError(
            f"imported MLX is outside the attested checkout: {core_path}"
        )
    try:
        native_runtime = dpo.locked.verify_mlx_native_runtime(
            args,
            lock,
            mlx_checkout,
            mx,
            preverified_native_bundle,
        )
    except dpo.locked.ContractError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"loaded MLX native runtime differs from its attestation: {exc}"
        ) from exc
    dpo.install_mlx_lm_source_namespace(args.mlx_lm_source_root)
    try:
        from mlx_lm.models import gemma4 as mlx_gemma4
        from mlx_lm.tuner.lora import LoRALinear
        from mlx.utils import tree_unflatten
    except ImportError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"could not import the pinned MLX-LM source environment: {exc}"
        ) from exc

    gemma4_source_path = Path(mlx_gemma4.__file__ or "").resolve()
    lora_source_path = Path(sys.modules[LoRALinear.__module__].__file__ or "").resolve()
    for label, source_path in (
        ("MLX-LM Gemma4", gemma4_source_path),
        ("MLX-LM LoRA", lora_source_path),
    ):
        if not dpo._path_is_within(source_path, mlx_lm_root):
            raise dpo.DpoBenchmarkContractError(
                f"imported {label} is outside the attested checkout: {source_path}"
            )

    contract_dir = args.adapter_contract_dir.expanduser().resolve()
    try:
        manifest = json.loads(
            (contract_dir / "antfly_finetune_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise dpo.DpoBenchmarkContractError(
            f"could not load adapter contract manifest: {exc}"
        ) from exc
    binding_fields = (
        "base_model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
    )
    try:
        prepared_summary = {key: manifest[key] for key in binding_fields}
    except (KeyError, TypeError) as exc:
        raise dpo.DpoBenchmarkContractError(
            "adapter contract manifest is missing model binding"
        ) from exc
    base_model_provenance = dpo.locked.zig_model_provenance(model_dir)
    if prepared_summary != base_model_provenance:
        raise dpo.DpoBenchmarkContractError(
            "adapter contract model binding differs from the evaluation model"
        )
    adapter_contract = dpo.locked.inspect_initial_adapter(
        contract_dir,
        lock,
        case.model_key,
        case.target_preset,
        prepared_summary,
    )

    bound_input_paths = [
        *(model_dir / relative for relative in locked_model["files"]),
        *adapter_contract.bound_files,
        *candidate_paths,
        case.source_path,
        args.lock.expanduser().resolve(),
        dpo.locked.MLX_REQUIREMENTS_PATH,
        SCRIPT_PATH,
        dpo.SCRIPT_PATH,
        dpo.locked.SCRIPT_PATH,
        Path(sys.executable).resolve(),
        gemma4_source_path,
        lora_source_path,
    ]
    bound_input_paths.append(
        (
            case.source_path.parent
            / Path(case.dataset["materialized_jsonl"])
        ).resolve()
    )
    try:
        bound_input_identities = dpo.locked.capture_file_identities(
            bound_input_paths
        )
    except dpo.locked.ContractError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"could not bind immutable DPO evaluation inputs: {exc}"
        ) from exc

    padded_examples = []
    for example in case.examples:
        chosen_ids, chosen_labels = dpo.padded_sequence(
            example.prompt_token_ids, example.chosen_token_ids, case.sequence_length
        )
        rejected_ids, rejected_labels = dpo.padded_sequence(
            example.prompt_token_ids, example.rejected_token_ids, case.sequence_length
        )
        padded_examples.append(
            (chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        )

    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    sampler = dpo.locked.DarwinProcessMemorySampler()
    sampler.start()
    sampler_active = True
    started = time.perf_counter()
    try:
        model, _config = dpo.locked.load_locked_mlx_gemma4(
            model_dir,
            mx,
            load_config_fn=lambda path: json.loads(
                (path / "config.json").read_text(encoding="utf-8")
            ),
            get_model_classes_fn=lambda **_kwargs: (
                mlx_gemma4.Model,
                mlx_gemma4.ModelArgs,
            ),
        )
        mx.eval(model.parameters())
        mx.synchronize()
        model.freeze()
        base_inventory = dpo.locked.require_bf16_base_model(model, mx)

        chosen = [mx.array([item[0]], dtype=mx.int32) for item in padded_examples]
        chosen_y = [mx.array([item[1]], dtype=mx.int32) for item in padded_examples]
        rejected = [mx.array([item[2]], dtype=mx.int32) for item in padded_examples]
        rejected_y = [mx.array([item[3]], dtype=mx.int32) for item in padded_examples]

        def sequence_logp(current_model: Any, tokens: Any, labels: Any) -> Any:
            logits = current_model(tokens)[:, :-1, :].astype(mx.float32)
            shifted = labels[:, 1:]
            mask = shifted != -100
            safe = mx.where(mask, shifted, mx.zeros_like(shifted))
            losses = nn.losses.cross_entropy(logits, safe)
            return -(losses * mask).sum()

        reference_started = time.perf_counter()
        ref_chosen = [
            sequence_logp(model, tokens, labels)
            for tokens, labels in zip(chosen, chosen_y)
        ]
        ref_rejected = [
            sequence_logp(model, tokens, labels)
            for tokens, labels in zip(rejected, rejected_y)
        ]
        mx.eval(*ref_chosen, *ref_rejected)
        mx.synchronize()
        reference_seconds = time.perf_counter() - reference_started
        reference_chosen = [float(value.item()) for value in ref_chosen]
        reference_rejected = [float(value.item()) for value in ref_rejected]

        targets = dpo.locked.target_module_names(
            model, lock, case.model_key, case.target_preset
        )
        target_set = set(targets)
        updates = []
        for name, module in model.named_modules():
            if name not in target_set:
                continue
            if not isinstance(module, nn.Linear):
                raise dpo.DpoBenchmarkContractError(f"non-linear LoRA target: {name}")
            updates.append(
                (
                    name,
                    LoRALinear.from_base(module, r=16, scale=2.0, dropout=0.0),
                )
            )
        if {name for name, _module in updates} != target_set:
            raise dpo.DpoBenchmarkContractError("incomplete LoRA target conversion")
        model.update_modules(tree_unflatten(updates))
        trainable_inventory = dpo.locked.require_exact_trainables(
            model, targets, mx
        )
        model.eval()

        evaluations: dict[str, Any] = {}
        for candidate, candidate_path in zip(args.candidate, candidate_paths):
            candidate_sha256 = "sha256:" + hashlib.sha256(
                candidate_path.read_bytes()
            ).hexdigest()
            artifact = replace(
                adapter_contract,
                checkpoint=candidate_path,
                checkpoint_sha256=candidate_sha256,
            )
            try:
                dpo.locked.load_exact_initial_adapter(model, targets, artifact, mx)
            except dpo.locked.ContractError as exc:
                raise dpo.DpoBenchmarkContractError(
                    f"could not load candidate {candidate.label}: {exc}"
                ) from exc
            policy_chosen = [
                sequence_logp(model, tokens, labels)
                for tokens, labels in zip(chosen, chosen_y)
            ]
            policy_rejected = [
                sequence_logp(model, tokens, labels)
                for tokens, labels in zip(rejected, rejected_y)
            ]
            mx.eval(*policy_chosen, *policy_rejected)
            mx.synchronize()
            evaluations[candidate.label] = {
                "path": str(candidate_path),
                "sha256": candidate_sha256,
                **dpo.dpo_evaluation_metrics(
                    [float(value.item()) for value in policy_chosen],
                    [float(value.item()) for value in policy_rejected],
                    reference_chosen,
                    reference_rejected,
                    case.beta,
                ),
            }
        elapsed_seconds = time.perf_counter() - started
        memory = sampler.stop()
        sampler_active = False
    finally:
        if sampler_active:
            sampler.stop()

    pairwise = {}
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            pairwise[f"{left_label}_vs_{right_label}"] = compare_evaluations(
                evaluations[left_label], evaluations[right_label]
            )

    try:
        dpo.locked.require_files_unchanged(bound_input_identities)
        dpo.locked.require_files_unchanged(
            native_runtime["bound_file_identities"]
        )
        post_model = dpo.locked.verify_model_directory(
            lock, case.model_key, model_dir
        )
        if post_model != locked_model:
            raise dpo.locked.ContractError(
                "locked model identity drifted during evaluation"
            )
        if dpo.locked.verify_packages(lock, "mlx_reference") != package_versions:
            raise dpo.locked.ContractError(
                "MLX package environment drifted during evaluation"
            )
        post_mlx_checkout = dpo.locked.verify_source_checkout(
            args.mlx_source_root,
            revisions["mlx"],
            source_name="MLX",
        )
        post_mlx_lm_checkout = dpo.locked.verify_source_checkout(
            args.mlx_lm_source_root,
            revisions["mlx-lm"],
            source_name="MLX-LM",
        )
        postverified_native_bundle = dpo.locked.verify_mlx_native_build_before_import(
            args,
            lock,
            post_mlx_checkout,
            post_mlx_lm_checkout,
        )
        post_native_runtime = dpo.locked.verify_mlx_native_runtime(
            args,
            lock,
            post_mlx_checkout,
            mx,
            postverified_native_bundle,
        )
        for field in ("native_artifact_inventory", "build_attestation"):
            if post_native_runtime[field] != native_runtime[field]:
                raise dpo.locked.ContractError(
                    f"MLX {field} drifted during DPO evaluation"
                )
    except dpo.locked.ContractError as exc:
        raise dpo.DpoBenchmarkContractError(
            f"DPO evaluation input postflight failed: {exc}"
        ) from exc

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "framework": "mlx-lm",
        "algorithm": "dpo-adapter-evaluation",
        "model_key": case.model_key,
        "sequence_length": case.sequence_length,
        "beta": case.beta,
        "case": {
            "path": str(case.source_path),
            "semantic_sha256": case.semantic_sha256,
            "examples": len(case.examples),
        },
        "dataset": case.dataset,
        "reference_chosen_logps": reference_chosen,
        "reference_rejected_logps": reference_rejected,
        "reference_precompute_seconds": reference_seconds,
        "evaluations": evaluations,
        "pairwise": pairwise,
        "elapsed_seconds": elapsed_seconds,
        "peak_phys_footprint_bytes": memory.peak_phys_footprint_bytes,
        "mlx_allocator_peak_bytes": int(mx.get_peak_memory()),
        "base_inventory_sha256": base_inventory["inventory_sha256"],
        "trainable_inventory_sha256": trainable_inventory["inventory_sha256"],
        "adapter_contract_semantic_sha256": adapter_contract.semantic_sha256,
        "base_model_provenance": base_model_provenance,
        "mlx_revision": mlx_revision,
        "mlx_lm_revision": mlx_lm_revision,
        "locked_package_versions": package_versions,
        "python_version": actual_python,
        "locked_model": locked_model,
        "mlx_native_runtime": {
            "native_artifact_inventory": native_runtime[
                "native_artifact_inventory"
            ],
            "build_attestation": native_runtime["build_attestation"],
        },
        "mlx_core_path": str(core_path),
        "mlx_lm_gemma4_path": str(gemma4_source_path),
        "mlx_lm_lora_path": str(lora_source_path),
        "runner_sha256": "sha256:"
        + hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-contract-dir", type=Path, required=True)
    result.add_argument("--mlx-source-root", type=Path, required=True)
    result.add_argument("--mlx-lm-source-root", type=Path, required=True)
    result.add_argument(
        "--mlx-build-attestation",
        type=Path,
        required=True,
        help="strict local attestation binding the loaded MLX native runtime",
    )
    result.add_argument("--case", type=Path, required=True)
    result.add_argument("--lock", type=Path, default=dpo.locked.LOCK_PATH)
    result.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="repeat label=/path for each adapter Safetensors candidate",
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        dpo.write_json_exclusive(args.output, payload)
    except (dpo.DpoBenchmarkContractError, dpo.locked.ContractError) as exc:
        print(f"Gemma4 DPO MLX adapter evaluation contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
