#!/usr/bin/env python3
"""Exercise Antfly's installed Gemma 4 PEFT export through stock PEFT.

This is a dependency-pinned integration smoke, not a Gemma 4 numerical oracle.
It builds a tiny local causal-LM fixture with Gemma-compatible module names,
uses the public Antfly CLI for bootstrap and export, loads the result through
``PeftModel.from_pretrained``, and verifies a stock PEFT save/reload roundtrip.
No model or package is downloaded by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from gemma4_oracle_contract import (
    ContractError,
    inspect_adapter_artifact,
    load_lock,
    prefixed_sha256,
    verify_packages,
    verify_python_runtime,
    verify_requirements_match_lock,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LOCK_PATH = SCRIPT_DIR / "gemma4_oracle.lock.json"
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements-gemma4-oracle.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antfly", type=Path, required=True, help="ReleaseFast Antfly executable")
    parser.add_argument("--work-dir", type=Path, required=True, help="new directory for durable smoke artifacts")
    return parser.parse_args()


def run_antfly(executable: Path, arguments: list[str]) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    result = subprocess.run(
        [str(executable), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if result.returncode != 0:
        raise ContractError(
            f"Antfly command failed ({result.returncode}): {' '.join(arguments)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Antfly command did not emit one JSON object: {arguments}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"Antfly command emitted a non-object JSON value: {arguments}")
    return payload


def require_same_adapter(left: dict[str, Any], right: dict[str, Any]) -> None:
    if left["inventory"] != right["inventory"]:
        raise ContractError("stock PEFT save changed the canonical adapter inventory")
    if left["semantics"]["r"] != right["semantics"]["r"]:
        raise ContractError("stock PEFT save changed adapter rank")
    if left["semantics"]["lora_alpha"] != right["semantics"]["lora_alpha"]:
        raise ContractError("stock PEFT save changed adapter alpha")
    for identity, source in left["tensors"].items():
        destination = right["tensors"].get(identity)
        if destination is None:
            raise ContractError(f"stock PEFT save omitted adapter tensor {identity}")
        if source["shape"] != destination["shape"] or source["dtype"] != destination["dtype"]:
            raise ContractError(f"stock PEFT save changed tensor metadata for {identity}")
        if source["values"] != destination["values"]:
            raise ContractError(f"stock PEFT save changed tensor values for {identity}")


def main() -> int:
    args = parse_args()
    executable = args.antfly.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if not executable.is_file():
        raise ContractError(f"Antfly executable is missing: {executable}")
    if work_dir.exists():
        raise ContractError(f"refusing to replace work directory: {work_dir}")
    work_dir.mkdir(parents=True)

    lock = load_lock(LOCK_PATH)
    verify_requirements_match_lock(lock, "python_oracle", REQUIREMENTS_PATH)
    python_version = verify_python_runtime(lock)
    packages = verify_packages(lock, "python_oracle")

    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import torch
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    base_dir = work_dir / "tiny-base"
    internal_dir = work_dir / "antfly-adapter"
    export_dir = work_dir / "peft-export"
    peft_resaved_dir = work_dir / "peft-resaved"

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    created = LlamaForCausalLM(config).eval()
    created.save_pretrained(base_dir, safe_serialization=True)
    del created

    bootstrap = run_antfly(executable, [
        "inference", "finetune", "adapter", "bootstrap", "gemma4",
        "--model", str(base_dir), "--out", str(internal_dir),
        "--rank", "2", "--alpha", "4", "--target-preset", "peft-qv",
    ])
    exported = run_antfly(executable, [
        "inference", "finetune", "adapter", "export", "gemma4-peft",
        "--model", str(base_dir), "--adapter", str(internal_dir), "--out", str(export_dir),
    ])
    export_artifact = inspect_adapter_artifact(export_dir)
    if export_artifact["key_layout"] != "stock-peft/v1":
        raise ContractError("Antfly export did not use the stock PEFT tensor-key layout")
    if export_artifact["policy_source"] != "antfly-peft-export/v1":
        raise ContractError("Antfly export provenance sidecar was not recognized")

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    base = LlamaForCausalLM.from_pretrained(base_dir, local_files_only=True).eval()
    with torch.no_grad():
        base_logits = base(input_ids=input_ids).logits.detach().clone()
    adapted = PeftModel.from_pretrained(base, export_dir, local_files_only=True).eval()
    with torch.no_grad():
        adapted_logits = adapted(input_ids=input_ids).logits.detach()
    load_max_abs = float((adapted_logits - base_logits).abs().max().item())
    if load_max_abs != 0.0:
        raise ContractError(f"zero-delta PEFT load changed logits: max_abs={load_max_abs}")

    adapted.save_pretrained(peft_resaved_dir, safe_serialization=True)
    resaved_artifact = inspect_adapter_artifact(peft_resaved_dir, target_preset="peft-qv")
    require_same_adapter(export_artifact, resaved_artifact)

    fresh_base = LlamaForCausalLM.from_pretrained(base_dir, local_files_only=True).eval()
    reloaded = PeftModel.from_pretrained(fresh_base, peft_resaved_dir, local_files_only=True).eval()
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=input_ids).logits.detach()
    roundtrip_max_abs = float((reloaded_logits - adapted_logits).abs().max().item())
    if roundtrip_max_abs != 0.0:
        raise ContractError(f"stock PEFT save/reload changed logits: max_abs={roundtrip_max_abs}")

    report = {
        "schema_version": "antfly_gemma4_peft_export_roundtrip/v1",
        "status": "pass",
        "scope": "tiny-local-causal-lm-structural-smoke-not-gemma4-numerical-proof",
        "antfly": {
            "path": str(executable),
            "sha256": prefixed_sha256(executable),
        },
        "python": python_version,
        "packages": packages,
        "bootstrap": bootstrap,
        "export": exported,
        "export_adapter_model_sha256": prefixed_sha256(export_dir / "adapter_model.safetensors"),
        "stock_resaved_adapter_model_sha256": prefixed_sha256(peft_resaved_dir / "adapter_model.safetensors"),
        "tensor_count": len(export_artifact["inventory"]),
        "peft_load_max_abs_logit_difference": load_max_abs,
        "peft_save_reload_max_abs_logit_difference": roundtrip_max_abs,
    }
    write_json(work_dir / "roundtrip_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
