#!/usr/bin/env python3
"""Inspect a local DFlash draft checkpoint without loading tensor payloads."""

import argparse
import json
import os
import struct
import sys
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safetensors_header(path: Path):
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = f.read(header_len)
        if len(header) != header_len:
            raise ValueError(f"{path} has a truncated safetensors header")
    return json.loads(header.decode("utf-8"))


def find_safetensors(root: Path):
    if root.is_file() and root.suffix == ".safetensors":
        return [root]
    return sorted(root.glob("*.safetensors"))


def classify(name: str):
    if name == "fc.weight" or "feature" in name or "context" in name:
        return "feature_fusion"
    if "self_attn.k_proj.weight" in name or "self_attn.v_proj.weight" in name or "self_attn.k_norm.weight" in name:
        return "kv_injection"
    if name.startswith("layers.") or ".layers." in name:
        return "draft_transformer"
    if name in {"norm.weight", "hidden_norm.weight", "lm_head.weight"}:
        return "draft_head"
    return "other"


def tensor_summary(headers):
    out = {}
    for header in headers:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            kind = classify(name)
            out.setdefault(kind, []).append(
                {
                    "name": name,
                    "dtype": meta.get("dtype"),
                    "shape": meta.get("shape"),
                }
            )
    for tensors in out.values():
        tensors.sort(key=lambda item: item["name"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Local DFlash checkpoint directory or .safetensors file")
    args = parser.parse_args()

    root = Path(args.checkpoint).expanduser()
    if not root.exists():
        print(f"checkpoint path does not exist: {root}", file=sys.stderr)
        return 2

    config_path = root / "config.json" if root.is_dir() else root.with_name("config.json")
    config = read_json(config_path) if config_path.exists() else {}
    safetensors = find_safetensors(root)
    headers = [safetensors_header(path) for path in safetensors]

    dflash_config = config.get("dflash_config", {})
    summary = {
        "path": os.fspath(root),
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "block_size": config.get("block_size") or dflash_config.get("block_size"),
        "mask_token_id": dflash_config.get("mask_token_id") or config.get("mask_token_id"),
        "target_layer_ids": dflash_config.get("target_layer_ids")
        or config.get("target_feature_layers")
        or config.get("selected_target_feature_layers"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "sliding_window": config.get("sliding_window"),
        "safetensors_files": [os.fspath(path) for path in safetensors],
        "tensors": tensor_summary(headers),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
