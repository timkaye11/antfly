#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture compact official audio-tower tensors for native differential tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle import fixture_audio


def tensor(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = tensor(item)
            if found is not None:
                return found
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    return None


def stats(value):
    value = value.detach().float().cpu().contiguous()
    flat = value.flatten()
    return {"shape": list(value.shape), "min": float(flat.min()), "max": float(flat.max()),
            "mean": float(flat.mean()), "l2": float(flat.norm()), "first16": flat[:16].tolist()}


def capture(model_dir: Path, precision: str, device: str):
    import torch
    from transformers import EmbeddingGemma2Model, EmbeddingGemma2Processor

    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    processor = EmbeddingGemma2Processor.from_pretrained(model_dir, local_files_only=True)
    encoded = processor(text=["<|audio|>"], audio=[[fixture_audio()]], return_tensors="pt")
    model = EmbeddingGemma2Model.from_pretrained(model_dir, local_files_only=True, dtype=dtype).to(device).eval()
    captured = {"frontend_features": encoded["input_features"].detach().clone()}
    modules = {
        "subsample_projection": model.audio_tower.subsample_conv_projection,
        "layer0_output": model.audio_tower.layers[0],
        "layer11_output": model.audio_tower.layers[11],
        "tower_output_projection": model.audio_tower.output_proj,
        "projected_audio_embeddings": model.embed_audio,
    }
    hooks = []
    for name, module in modules.items():
        hooks.append(module.register_forward_hook(lambda _m, _i, output, name=name: captured.__setitem__(name, tensor(output).detach().clone())))
    with torch.inference_mode():
        model(**{key: value.to(device) for key, value in encoded.items()})
    for hook in hooks:
        hook.remove()
    return captured, {name: stats(value) for name, value in captured.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    captured, summary = capture(args.model, args.precision, args.device)
    import numpy as np
    npz = args.output_prefix.with_suffix(".npz")
    metadata = args.output_prefix.with_suffix(".json")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **{name: value.float().cpu().numpy() for name, value in captured.items()})
    metadata.write_text(json.dumps({"precision": args.precision, "tensors": summary}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"npz": str(npz), "metadata": str(metadata), "tensors": list(captured)}))


if __name__ == "__main__":
    main()
