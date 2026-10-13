#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independently verify a native export against its checkpoint and PyTorch/PEFT."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from finetune_reference import BASE_WEIGHT_SHA256, file_sha256, target_module_names

SIDECARS = (
    "config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
    "modules.json", "sentence_bert_config.json", "config_sentence_transformers.json",
    "1_Pooling/config.json", "2_Normalize/config.json",
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def verify(base: Path, checkpoint: Path, bundle: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model

    torch.set_num_threads(1)
    state = json.loads((checkpoint / "state.json").read_text())
    provenance = json.loads((bundle / "antfly_finetune_provenance.json").read_text())
    manifest = json.loads((bundle / "model_manifest.json").read_text())
    adapter_config = json.loads((bundle / "adapter_config.json").read_text())
    base_weights = artifact(base / "model.safetensors")
    checkpoint_weights = artifact(checkpoint / "training.safetensors")
    merged_weights = artifact(bundle / "model.safetensors")
    if base_weights["sha256"] != BASE_WEIGHT_SHA256 or state["base_sha256"] != BASE_WEIGHT_SHA256:
        raise ValueError("base model does not match the official pinned weights")
    if checkpoint_weights["sha256"] != state["tensor_file_sha256"]:
        raise ValueError("checkpoint tensor digest mismatch")
    if provenance["merged_weight_sha256"] != merged_weights["sha256"]:
        raise ValueError("merged model digest mismatch")
    if provenance["base_weight_sha256"] != BASE_WEIGHT_SHA256:
        raise ValueError("export base digest mismatch")
    sidecars = {}
    for name in SIDECARS:
        copied = artifact(bundle / name)
        if copied["sha256"] != file_sha256(base / name):
            raise ValueError(f"exported processor/tokenizer sidecar changed: {name}")
        sidecars[name] = copied
    if provenance.get("base_model") != "google/embeddinggemma-2" or provenance.get("base_revision") != "914f7f89142e33e77833254d9c9b90c3cef7303b":
        raise ValueError("export does not pin the reviewed upstream model revision")
    training = provenance["training"]
    if training["checkpoint_weight_sha256"] != checkpoint_weights["sha256"]:
        raise ValueError("export refers to a different checkpoint")
    for name in ("job_fingerprint", "dataset_sha256", "optimizer_step", "generation", "epoch", "cursor", "total_steps", "seed"):
        if training[name] != state[name]:
            raise ValueError(f"export checkpoint provenance mismatch: {name}")
    if (
        manifest.get("inputs") != ["text"]
        or manifest.get("media_enabled") is not False
        or provenance.get("inputs") != ["text"]
    ):
        raise ValueError("adapted bundle must advertise text-only serving")
    if provenance["rank"] != adapter_config["r"] or provenance["alpha"] != adapter_config["lora_alpha"]:
        raise ValueError("adapter configuration disagrees with export provenance")
    rank = adapter_config["r"]
    scale = adapter_config["lora_alpha"] / rank
    if rank not in (8, 16, 32) or not 0 < scale < float("inf"):
        raise ValueError("unsupported LoRA layout")
    config = EmbeddingGemma2Config.from_pretrained(base, local_files_only=True)
    with torch.device("meta"):
        topology = EmbeddingGemma2Model(config)
    targets = target_module_names(topology)
    del topology
    if len(targets) != 169 or len(adapter_config["target_modules"]) != 169 or set(adapter_config["target_modules"]) != set(targets):
        raise ValueError("adapter target inventory differs from the official text topology")
    expected_adapter_names = {
        f"base_model.model.{name}.lora_{kind}.weight"
        for name in targets
        for kind in ("A", "B")
    }
    rows: list[dict[str, Any]] = []
    untouched = merged_elements = 0
    with (
        safe_open(str(base / "model.safetensors"), framework="pt", device="cpu") as original,
        safe_open(str(bundle / "model.safetensors"), framework="pt", device="cpu") as merged,
        safe_open(str(checkpoint / "training.safetensors"), framework="pt", device="cpu") as trained,
        safe_open(str(bundle / "adapter_model.safetensors"), framework="pt", device="cpu") as adapters,
    ):
        if set(original.keys()) != set(merged.keys()) or set(adapters.keys()) != expected_adapter_names:
            raise ValueError("exported tensor inventory mismatch")
        expected_state_names = {
            name.removeprefix("base_model.model.") + suffix
            for name in expected_adapter_names
            for suffix in ("", ".exp_avg", ".exp_avg_sq")
        }
        if set(trained.keys()) != expected_state_names:
            raise ValueError("checkpoint must contain all 1,014 master and Adam tensors")
        for name in expected_adapter_names:
            source = trained.get_tensor(name.removeprefix("base_model.model."))
            candidate = adapters.get_tensor(name)
            if source.dtype != torch.float32 or candidate.dtype != torch.float32 or not torch.equal(source, candidate):
                raise ValueError(f"exported FP32 adapter differs from its checkpoint: {name}")
            for suffix in ("", ".exp_avg", ".exp_avg_sq"):
                value = trained.get_tensor(name.removeprefix("base_model.model.") + suffix)
                if value.dtype != torch.float32 or value.shape != source.shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid checkpoint master or Adam tensor: {name}{suffix}")
                if suffix == ".exp_avg_sq" and (value < 0).any():
                    raise ValueError(f"negative Adam second moment: {name}")
        target_weights = {name + ".weight" for name in targets}
        for name in original.keys():
            source, candidate = original.get_tensor(name), merged.get_tensor(name)
            if source.dtype != candidate.dtype or source.shape != candidate.shape:
                raise ValueError(f"exported base shape or dtype changed: {name}")
            if name not in target_weights:
                if not torch.equal(source.reshape(-1).view(torch.uint8), candidate.reshape(-1).view(torch.uint8)):
                    raise ValueError(f"untouched base tensor changed: {name}")
                untouched += 1
                continue
            stem = name.removesuffix(".weight")
            a = trained.get_tensor(stem + ".lora_A.weight")
            b = trained.get_tensor(stem + ".lora_B.weight")
            if source.dtype != torch.bfloat16 or a.shape != (rank, source.shape[1]) or b.shape != (source.shape[0], rank):
                raise ValueError(f"invalid merged target layout: {name}")
            expected = (source.float() + scale * (b @ a)).to(torch.bfloat16)
            different = int(torch.count_nonzero(expected.view(torch.int16) != candidate.view(torch.int16)))
            rows.append({"tensor": name, "elements": source.numel(), "different_bf16_bits": different})
            merged_elements += source.numel()
    # Reload through the actual upstream PEFT loader with a real CPU base.
    # This checks interoperability as well as the file-level adapter comparison.
    model = EmbeddingGemma2Model.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16)
    reloaded = PeftModel.from_pretrained(model, bundle, is_trainable=False, local_files_only=True)
    reloaded_names: set[str] = set()
    with safe_open(str(bundle / "adapter_model.safetensors"), framework="pt", device="cpu") as adapters:
        for name, parameter in reloaded.named_parameters():
            canonical = name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
                ".lora_B.default.weight", ".lora_B.weight"
            )
            if canonical not in expected_adapter_names:
                continue
            expected = adapters.get_tensor(canonical)
            if parameter.dtype != torch.float32 or not torch.equal(parameter.detach().cpu(), expected):
                raise ValueError(f"PEFT reload differs from its exported FP32 adapter: {canonical}")
            reloaded_names.add(canonical)
    if reloaded_names != expected_adapter_names:
        raise ValueError("PEFT reload did not expose all 338 adapter tensors")
    return {
        "schema": "antfly.embedding_gemma2.finetune_export_verification.v1",
        "pass": len(rows) == 169 and all(row["different_bf16_bits"] == 0 for row in rows),
        "merged_targets": len(rows),
        "merged_elements": merged_elements,
        "adapter_tensors": len(expected_adapter_names),
        "byte_identical_untouched_tensors": untouched,
        "peft_reload_exact": True,
        "rows": rows,
        "artifacts": {
            "base": base_weights,
            "checkpoint": checkpoint_weights,
            "checkpoint_state": artifact(checkpoint / "state.json"),
            "merged": merged_weights,
            "adapters": artifact(bundle / "adapter_model.safetensors"),
            "provenance": artifact(bundle / "antfly_finetune_provenance.json"),
            "verifier": artifact(Path(__file__)),
            "sidecars": sidecars,
        },
        "reference": {"torch": torch.__version__, "peft": version("peft"),
                      "transformers": version("transformers"), "merge": "CPU FP32 B@A, scale, add, BF16 RNE"},
        "qualification_scope": "export-functional",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "checkpoint", "bundle", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.base, args.checkpoint, args.bundle)
    except Exception as exc:
        report = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": report["pass"], "report": str(args.report)}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
