#!/usr/bin/env python3
"""Export a pinned HF/PEFT Gemma4 LoRA numerical trace.

The exporter is intentionally a plain, eager PyTorch training step rather than
TRL or an optimized trainer.  It loads only locally verified model and adapter
artifacts, feeds Antfly's exact prepared token arrays, computes causal
cross-entropy explicitly, and captures LoRA initial values, raw gradients,
AdamW state, post-update values, and supervised-position logit probes.

No model is downloaded.  Run only inside the exact environment in
``requirements-gemma4-oracle.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from gemma4_oracle_contract import (
    ANTFLY_ADAPTER_KEY_FORMAT,
    ContractError,
    LOCK_PATH,
    STOCK_PEFT_KEY_FORMAT,
    TRACE_SCHEMA_VERSION,
    antfly_to_stock_peft_tensor_name,
    build_evidence_ledger,
    canonicalize_adapter_tensor_name,
    hardware_fingerprint,
    inspect_adapter_artifact,
    load_lock,
    load_prepared_example,
    lock_digest,
    prefixed_sha256,
    vector_metrics,
    validate_evidence_ledger,
    validate_target_inventory,
    verify_model_directory,
    verify_packages,
    verify_prepared_source_dataset,
    verify_python_runtime,
    verify_import_source,
    verify_source_checkout,
    write_json,
)


def parse_betas(value: str) -> tuple[float, float]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("betas must be BETA1,BETA2")
    try:
        result = (float(pieces[0]), float(pieces[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("betas must be numeric") from exc
    if not all(math.isfinite(item) and 0 <= item < 1 for item in result):
        raise argparse.ArgumentTypeError("betas must be finite and in [0,1)")
    return result


def validate_target_preset(
    lock: Mapping[str, Any],
    model_key: str,
    semantics: Mapping[str, Any],
) -> str:
    preset = semantics.get("target_preset")
    if preset not in lock["target_presets"]:
        raise ContractError("adapter_config.json must record peft-qv or text-all-linear")
    validate_target_inventory(lock, model_key, str(preset), semantics["target_modules"])
    return str(preset)


def parameter_inventory(model: Any) -> dict[tuple[str, str], tuple[str, Any]]:
    result: dict[tuple[str, str], tuple[str, Any]] = {}
    unexpected_trainables: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_A" not in name and ".lora_B" not in name:
            unexpected_trainables.append(name)
            continue
        identity = canonicalize_adapter_tensor_name(name)
        if identity in result:
            raise ContractError(f"PEFT model has duplicate canonical parameter {identity}")
        result[identity] = (name, parameter)
    if unexpected_trainables:
        raise ContractError(f"non-LoRA parameters are trainable: {unexpected_trainables}")
    if not result:
        raise ContractError("PEFT model exposed no trainable LoRA parameters")
    modules: dict[str, set[str]] = {}
    for module, role in result:
        modules.setdefault(module, set()).add(role)
    if any(roles != {"lora_A", "lora_B"} for roles in modules.values()):
        raise ContractError("PEFT model does not expose a complete A/B pair for every target")
    return result


def tensor_values(tensor: Any) -> list[float]:
    values = [float(value) for value in tensor.detach().float().cpu().reshape(-1).tolist()]
    if any(not math.isfinite(value) for value in values):
        raise ContractError("captured tensor contains a non-finite value")
    return values


def stable_probe_token_ids(target: int, vocab_size: int, predictor_position: int, seed: int) -> list[int]:
    if not 0 <= target < vocab_size:
        raise ContractError(f"supervised token {target} is outside vocabulary size {vocab_size}")
    candidates = [0, 1, 2, target, vocab_size - 1]
    state = (seed ^ (predictor_position * 0x9E3779B1)) & 0xFFFFFFFF
    for _ in range(4):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        candidates.append(state % vocab_size)
    return sorted(set(candidates))


def logit_probes(logits: Any, labels: list[int], seed: int) -> list[dict[str, Any]]:
    import torch

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ContractError(f"expected logits [1,sequence,vocab], found {tuple(logits.shape)}")
    vocab_size = int(logits.shape[-1])
    result: list[dict[str, Any]] = []
    detached = logits.detach().float().cpu()[0]
    for label_position in range(1, len(labels)):
        target = labels[label_position]
        if target == -100:
            continue
        predictor = label_position - 1
        row = detached[predictor]
        token_ids = stable_probe_token_ids(target, vocab_size, predictor, seed)
        result.append({
            "predictor_position": predictor,
            "target_token_id": target,
            "token_ids": token_ids,
            "values": [float(row[token_id].item()) for token_id in token_ids],
            "logsumexp": float(torch.logsumexp(row, dim=-1).item()),
        })
    return result


def explicit_causal_loss(logits: Any, labels: Any) -> tuple[Any, int]:
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    supervised = int((shifted_labels != -100).sum().item())
    if supervised <= 0:
        raise ContractError("selected prepared example has no causal supervised tokens")
    total = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return total / supervised, supervised


def translate_antfly_adapter_to_stock_peft(
    source_dir: Path,
    destination_dir: Path,
    source_artifact: Mapping[str, Any],
    target_preset: str,
) -> dict[str, Any]:
    """Materialize the one-way, verified key translation required by PEFT.

    The source checkpoint remains untouched. The temporary result contains a
    stock ``adapter_config.json`` plus a Safetensors file whose keys omit
    Antfly's frozen ``.weight`` segment and carry PEFT's wrapper prefix.
    """
    if source_artifact["key_layout"] != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError("only an Antfly internal-key adapter may be translated")
    if source_artifact["provenance"].get("tensor_key_format") != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError("Antfly adapter sidecar does not authorize internal-key translation")

    from safetensors import safe_open
    from safetensors.torch import save_file

    destination_dir.mkdir()
    shutil.copy2(source_dir.resolve() / "adapter_config.json", destination_dir / "adapter_config.json")
    source_checkpoint = source_dir.resolve() / "adapter_model.safetensors"
    translated_tensors: dict[str, Any] = {}
    key_map: dict[str, str] = {}
    with safe_open(str(source_checkpoint), framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for source_name in source.keys():
            destination_name = antfly_to_stock_peft_tensor_name(source_name)
            if destination_name in translated_tensors:
                raise ContractError(f"Antfly key translation collision: {destination_name}")
            translated_tensors[destination_name] = source.get_tensor(source_name).contiguous()
            key_map[source_name] = destination_name
    save_file(
        translated_tensors,
        str(destination_dir / "adapter_model.safetensors"),
        metadata=metadata,
    )
    write_json(destination_dir / "antfly_oracle_translation.json", {
        "schema_version": "antfly_to_stock_peft_translation/v1",
        "source_adapter_model_sha256": source_artifact["adapter_model_sha256"],
        "source_tensor_key_format": ANTFLY_ADAPTER_KEY_FORMAT,
        "destination_tensor_key_format": STOCK_PEFT_KEY_FORMAT,
        "target_preset": target_preset,
        "key_map": key_map,
        "reverse_interoperability_proven": False,
    })
    translated = inspect_adapter_artifact(destination_dir, target_preset=target_preset)
    if translated["key_layout"] != STOCK_PEFT_KEY_FORMAT:
        raise ContractError("translated adapter did not produce stock PEFT keys")
    if source_artifact["inventory"] != translated["inventory"]:
        raise ContractError("translated adapter changed the canonical tensor inventory")
    for identity in source_artifact["tensors"]:
        left = source_artifact["tensors"][identity]
        right = translated["tensors"][identity]
        if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
            raise ContractError(f"translated adapter changed tensor metadata for {identity}")
        if vector_metrics(left["values"], right["values"]).max_abs != 0:
            raise ContractError(f"translated adapter changed tensor values for {identity}")
    return translated


def load_model(args: argparse.Namespace, lock: Mapping[str, Any], adapter_dir: Path) -> tuple[Any, Any, Any]:
    # Force offline behavior before importing/loading Hugging Face code.  This
    # turns a missing local file into an error instead of a network side effect.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import peft
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM

    verify_import_source(transformers, args.transformers_source, source_name="Transformers")
    verify_import_source(peft, args.peft_source, source_name="PEFT")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")

    base = AutoModelForMultimodalLM.from_pretrained(
        str(args.model_dir.resolve()),
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=lock["python_oracle"]["execution"]["attention_implementation"],
    )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir.resolve()),
        is_trainable=True,
        local_files_only=True,
        autocast_adapter_dtype=True,
    )
    model.config.use_cache = False
    model.to(args.device)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            raise ContractError(f"nonzero dropout module remained in parity model: p={module.p}")
    return torch, transformers, model


def fsync_path(path: Path) -> None:
    flags = os.O_RDONLY | (getattr(os, "O_DIRECTORY", 0) if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    paths = list(root.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ContractError(f"oracle evidence cannot contain a symlink: {path}")
        if path.is_file():
            fsync_path(path)
    directories = [path for path in paths if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        fsync_path(directory)
    fsync_path(root)


def publish_staging(staging: Path, output: Path) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    validate_evidence_ledger(staging / "trace.json")
    fsync_tree(staging)
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise ContractError(f"refusing to replace existing oracle output: {target}") from exc
    fsync_path(target.parent)
    # The directory reservation is no-replace. Readers treat COMPLETE.json as
    # the commit marker and must ignore an interrupted directory without it.
    complete = staging / "COMPLETE.json"
    children = sorted((child for child in staging.iterdir() if child != complete), key=lambda child: child.name)
    try:
        for child in children:
            child.rename(target / child.name)
        fsync_tree(target)
        complete.rename(target / complete.name)
        fsync_path(target / complete.name)
        fsync_path(target)
        fsync_path(target.parent)
    except BaseException:
        # Preserve the incomplete evidence for diagnosis; it cannot be
        # mistaken for a complete run because the marker is published last.
        raise


def export(args: argparse.Namespace) -> dict[str, Any]:
    lock = load_lock(args.lock)
    verify_python_runtime(lock)
    package_versions = verify_packages(lock, "python_oracle")
    verify_source_checkout(
        args.transformers_source,
        lock["python_oracle"]["source_revisions"]["transformers"],
        source_name="Transformers",
    )
    verify_source_checkout(
        args.peft_source,
        lock["python_oracle"]["source_revisions"]["peft"],
        source_name="PEFT",
    )
    # A clean source checkout must remain clean while imported by the oracle.
    sys.dont_write_bytecode = True
    trajectory = lock["training_contract"]
    if args.steps not in trajectory["steps"]:
        raise ContractError(f"the locked trajectory admits exactly {trajectory['steps']} steps")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ContractError("learning rate must be finite and positive")
    if args.eps <= 0 or not math.isfinite(args.eps):
        raise ContractError("AdamW epsilon must be finite and positive")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ContractError("weight decay must be finite and non-negative")
    if args.max_grad_norm < 0 or not math.isfinite(args.max_grad_norm):
        raise ContractError("max grad norm must be finite and non-negative")
    supplied_trajectory = {
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "betas": list(args.betas),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
    }
    expected_trajectory = {key: trajectory[key] for key in supplied_trajectory}
    if supplied_trajectory != expected_trajectory:
        raise ContractError(
            f"optimizer arguments differ from training_contract "
            f"(expected={expected_trajectory}, actual={supplied_trajectory})"
        )
    execution = lock["python_oracle"]["execution"]
    if args.device != execution["device"] or args.dtype != execution["dtype"]:
        raise ContractError(
            f"release oracle requires {execution['device']}/{execution['dtype']}, "
            f"found {args.device}/{args.dtype}"
        )

    verified_model = verify_model_directory(lock, args.model_key, args.model_dir)
    prepared_summary, prepared = load_prepared_example(args.prepared, args.example_index)
    verify_prepared_source_dataset(prepared_summary, args.source_dataset)
    source_adapter = inspect_adapter_artifact(args.adapter, target_preset=args.target_preset)
    preset = validate_target_preset(lock, args.model_key, source_adapter["semantics"])
    rank = source_adapter["semantics"]["r"]
    alpha = float(source_adapter["semantics"]["lora_alpha"])

    translation_applied = source_adapter["key_layout"] == ANTFLY_ADAPTER_KEY_FORMAT
    if translation_applied:
        with tempfile.TemporaryDirectory(prefix="antfly-gemma4-peft-translation-") as translated_tmp:
            translated_dir = Path(translated_tmp) / "adapter"
            translated_adapter = translate_antfly_adapter_to_stock_peft(
                args.adapter,
                translated_dir,
                source_adapter,
                preset,
            )
            torch, transformers, model = load_model(args, lock, translated_dir)
        hf_load_key_layout = translated_adapter["key_layout"]
    else:
        if source_adapter["key_layout"] != STOCK_PEFT_KEY_FORMAT:
            raise ContractError(f"unsupported HF load key layout: {source_adapter['key_layout']}")
        torch, transformers, model = load_model(args, lock, args.adapter)
        hf_load_key_layout = source_adapter["key_layout"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA was selected but torch.cuda.is_available() is false")
    parameters = parameter_inventory(model)
    if set(parameters) != set(source_adapter["tensors"]):
        missing = sorted(set(source_adapter["tensors"]) - set(parameters))
        extra = sorted(set(parameters) - set(source_adapter["tensors"]))
        raise ContractError(f"loaded PEFT target inventory differs (missing={missing}, extra={extra})")
    for identity, (_, parameter) in parameters.items():
        if parameter.dtype != torch.float32:
            raise ContractError(f"{identity}: LoRA parameter must remain float32, found {parameter.dtype}")
        loaded = tensor_values(parameter)
        source = source_adapter["tensors"][identity]["values"]
        if vector_metrics(source, loaded).max_abs != 0:
            raise ContractError(f"PEFT changed adapter values while loading {identity}")

    input_ids = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=args.device)
    labels = torch.tensor([prepared["labels"]], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in parameters.values()],
        lr=args.learning_rate,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    initial = {identity: parameter.detach().float().cpu().clone() for identity, (_, parameter) in parameters.items()}
    final_raw_gradients: dict[tuple[str, str], Any] = {}
    loss_history: list[float] = []
    final_grad_norm = 0.0
    final_probes: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if not hasattr(outputs, "logits"):
            raise ContractError("Gemma4 HF model did not return logits")
        loss, supervised = explicit_causal_loss(outputs.logits, labels)
        if not torch.isfinite(loss):
            raise ContractError(f"step {step}: non-finite loss")
        loss.backward()
        grad_sq = 0.0
        final_raw_gradients = {}
        for identity, (_, parameter) in parameters.items():
            if parameter.grad is None:
                raise ContractError(f"step {step}: missing gradient for {identity}")
            gradient = parameter.grad.detach().float().cpu().clone()
            if not torch.isfinite(gradient).all():
                raise ContractError(f"step {step}: non-finite gradient for {identity}")
            final_raw_gradients[identity] = gradient
            grad_sq = math.fsum((grad_sq, float(torch.sum(gradient.double().square()).item())))
        final_grad_norm = math.sqrt(grad_sq)
        if final_grad_norm == 0:
            raise ContractError(f"step {step}: all adapter gradients are zero")
        final_probes = logit_probes(outputs.logits, prepared["labels"], args.seed)
        loss_history.append(float(loss.detach().float().cpu().item()))
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in parameters.values()],
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
        optimizer.step()

    host = hardware_fingerprint()
    host["torch_device"] = args.device
    host["dtype"] = args.dtype
    if args.device == "cuda":
        host["cuda_device"] = torch.cuda.get_device_name(torch.device(args.device))
        host["cuda_runtime"] = torch.version.cuda

    output = args.output_dir.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as tmp:
        staging = Path(tmp)
        reference_adapter_dir = staging / "reference_adapter"
        model.save_pretrained(reference_adapter_dir, safe_serialization=True)
        saved_adapter = inspect_adapter_artifact(reference_adapter_dir, target_preset=preset)
        if set(saved_adapter["tensors"]) != set(parameters):
            raise ContractError("saved PEFT adapter inventory changed after the optimizer step")

        storage_tensors: dict[str, Any] = {}
        entries: dict[str, dict[str, Any]] = {}
        target_rows: list[dict[str, Any]] = []
        inventory: list[str] = []
        for tensor_index, identity in enumerate(sorted(parameters)):
            module, role = identity
            source_name, parameter = parameters[identity]
            states = optimizer.state.get(parameter)
            if states is None or "exp_avg" not in states or "exp_avg_sq" not in states:
                raise ContractError(f"optimizer state is missing for {identity}")
            values_by_state = {
                "initial": initial[identity],
                "gradient": final_raw_gradients[identity],
                "updated": parameter.detach().float().cpu(),
                "optimizer_m": states["exp_avg"].detach().float().cpu(),
                "optimizer_v": states["exp_avg_sq"].detach().float().cpu(),
            }
            logical: dict[str, str] = {}
            for state_index, (state_name, tensor) in enumerate(values_by_state.items()):
                logical_name = f"{module}:{role}:{state_name}"
                storage_key = f"tensor_{tensor_index:05d}_{state_index}"
                contiguous = tensor.contiguous()
                storage_tensors[storage_key] = contiguous
                entries[logical_name] = {
                    "shape": list(contiguous.shape),
                    "dtype": "float32",
                    "storage_key": storage_key,
                }
                logical[state_name] = logical_name
            b_initial_zero = False
            if role == "lora_A" and args.steps == 1:
                b_initial = initial.get((module, "lora_B"))
                b_initial_zero = b_initial is not None and bool(torch.count_nonzero(b_initial).item() == 0)
            gradient_expectation = "zero-by-zero-b-initialization" if b_initial_zero else "active"
            gradient_nonzero = bool(torch.count_nonzero(final_raw_gradients[identity]).item() != 0)
            if gradient_expectation == "active" and not gradient_nonzero:
                raise ContractError(f"active target has an all-zero gradient: {identity}")
            if gradient_expectation != "active" and gradient_nonzero:
                raise ContractError(f"zero-initialization expectation failed: {identity}")
            target_rows.append({
                "canonical_name": module,
                "source_name": source_name,
                "role": role,
                "shape": list(parameter.shape),
                "gradient_expectation": gradient_expectation,
                "logical_tensors": logical,
            })
            inventory.append(f"{module}:{role}")

        from safetensors.torch import save_file

        tensor_path = staging / "trace.safetensors"
        save_file(storage_tensors, str(tensor_path), metadata={"format": "antfly_gemma4_lora_trace/v1"})
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "producer": {
                "name": "hf-peft",
                "version": ";".join(f"{name}={version}" for name, version in sorted(package_versions.items())),
                "source_revision": lock["python_oracle"]["source_revisions"]["transformers"],
                "hardware": host,
            },
            "oracle_lock_sha256": lock_digest(args.lock),
            "model": {
                "key": args.model_key,
                "repo_id": verified_model["repo_id"],
                "revision": verified_model["revision"],
                "local_artifact_sha256": verified_model["local_artifact_sha256"],
            },
            "prepared": prepared,
            "training": {
                "optimizer": "adamw",
                "seed": args.seed,
                "step": args.steps,
                "rank": rank,
                "alpha": alpha,
                "scale": alpha / rank,
                "target_preset": preset,
                "learning_rate": args.learning_rate,
                "betas": list(args.betas),
                "eps": args.eps,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "grad_accum_steps": 1,
                "supervised_token_normalization": "mean",
                "dropout": 0.0,
                "use_cache": False,
            },
            "metrics": {
                "loss": loss_history[-1],
                "loss_history": loss_history,
                "grad_norm": final_grad_norm,
                "supervised_tokens": supervised,
            },
            "logit_probes": final_probes,
            "target_tensors": target_rows,
            "tensor_store": {
                "format": "safetensors/v1",
                "path": tensor_path.name,
                "sha256": prefixed_sha256(tensor_path),
                "entries": entries,
            },
            "artifact": {
                "adapter_config_semantics": {
                    **saved_adapter["semantics"],
                    "target_preset": preset,
                },
                "adapter_model_sha256": saved_adapter["adapter_model_sha256"],
                "tensor_inventory": sorted(inventory),
                "key_layout": saved_adapter["key_layout"],
                "policy_source": saved_adapter["policy_source"],
            },
        }
        trace_path = staging / "trace.json"
        write_json(trace_path, trace)
        write_json(staging / "COMPLETE.json", build_evidence_ledger(staging))
        del model
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        publish_staging(staging, output)
    return {
        "ok": True,
        "output_dir": str(output),
        "trace": str(output / "trace.json"),
        "model": args.model_key,
        "target_preset": preset,
        "steps": args.steps,
        "source_adapter_key_layout": source_adapter["key_layout"],
        "hf_load_adapter_key_layout": hf_load_key_layout,
        "antfly_to_stock_peft_translation_applied": translation_applied,
        "direct_bidirectional_interoperability_proven": False,
        "loss": loss_history[-1],
        "supervised_tokens": supervised,
        "transformers": transformers.__version__,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lock", type=Path, default=LOCK_PATH)
    result.add_argument("--model-key", required=True, choices=("gemma-4-E2B-it", "gemma-4-E4B-it"))
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--transformers-source", type=Path, required=True, help="clean checkout at the locked Transformers commit and active import source")
    result.add_argument("--peft-source", type=Path, required=True, help="clean checkout at the locked PEFT commit and active import source")
    result.add_argument("--target-preset", choices=("peft-qv", "text-all-linear"), help="required for a stock PEFT adapter without an Antfly manifest")
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--source-dataset", type=Path, help="override the recorded source dataset path while verifying its v6 digest")
    result.add_argument("--example-index", type=int, default=0)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--steps", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--betas", type=parse_betas, default=(0.9, 0.999))
    result.add_argument("--eps", type=float, default=1e-8)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--max-grad-norm", type=float, default=1.0)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = export(args)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except ContractError as exc:
        print(f"Gemma4 HF oracle error: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as exc:
        # Model/framework errors are contract failures. Do not silently retry
        # with another attention implementation, dtype, device, or model class.
        print(f"Gemma4 HF oracle failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
