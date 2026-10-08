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

"""Capture compact official BF16 document intermediates for native differential tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from contract import render_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import EmbeddingGemma2Model, EmbeddingGemma2Processor
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import apply_rotary_pos_emb

    processor = EmbeddingGemma2Processor.from_pretrained(args.model, local_files_only=True)
    model = EmbeddingGemma2Model.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    language = model.language_model
    layer = language.layers[0]
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def output_hook(name: str, transform=lambda value: value):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = transform(value).detach().float().cpu()
        return hook

    hooks.append(language.embed_tokens.register_forward_hook(output_hook("input_scaled")))
    hooks.append(language.ple.register_forward_hook(output_hook("ple0", lambda value: value[:, :, 0, :])))
    hooks.append(layer.input_layernorm.register_forward_hook(output_hook("l0_input_norm")))
    hooks.append(layer.self_attn.q_norm.register_forward_hook(output_hook("l0_q_norm")))
    hooks.append(layer.self_attn.k_norm.register_forward_hook(output_hook("l0_k_norm")))
    hooks.append(layer.self_attn.v_norm.register_forward_hook(output_hook("l0_v_norm")))
    hooks.append(layer.self_attn.register_forward_hook(output_hook("l0_attention")))
    hooks.append(layer.post_attention_layernorm.register_forward_hook(output_hook("l0_post_attention_norm")))
    hooks.append(layer.ple_block.register_forward_pre_hook(lambda _module, inputs: captured.__setitem__("l0_ffn_residual", inputs[0].detach().float().cpu())))
    hooks.append(layer.ple_block.register_forward_hook(output_hook("l0_output")))

    text = render_text("Ant colonies coordinate through local signals.", "RETRIEVAL_DOCUMENT")
    encoded = processor(text=[text], padding=True, truncation=False, return_tensors="pt")
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = F.normalize((output * mask).sum(1) / mask.sum(1), p=2, dim=-1)
    for hook in hooks:
        hook.remove()

    captured["l0_attn_residual"] = captured["input_scaled"] + captured["l0_post_attention_norm"]
    with torch.inference_mode():
        sequence = captured["input_scaled"].shape[1]
        hidden_device = captured["input_scaled"].to("cuda", dtype=torch.bfloat16)
        position_ids = torch.arange(sequence, device="cuda").unsqueeze(0)
        layer_type = language.config.layer_types[0]
        cos, sin = language.rotary_emb(hidden_device, position_ids, layer_type)
        q_device = captured.pop("l0_q_norm").to("cuda", dtype=torch.bfloat16)
        k_device = captured.pop("l0_k_norm").to("cuda", dtype=torch.bfloat16)
        q_rope, k_rope = apply_rotary_pos_emb(q_device, k_device, cos, sin)
        captured["l0_q_rope"] = q_rope.float().cpu()
        captured["l0_k_rope"] = k_rope.float().cpu()
    captured.pop("l0_post_attention_norm")
    captured["pooled"] = pooled.cpu()

    ordered_names = (
        "input_scaled", "ple0", "l0_input_norm", "l0_q_rope", "l0_k_rope", "l0_v_norm",
        "l0_attention", "l0_attn_residual", "l0_ffn_residual", "l0_output", "pooled",
    )
    arrays = {name: captured[name].numpy() for name in ordered_names}
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.npz, **arrays)
    stats = {}
    for name, array in arrays.items():
        flat = array.reshape(-1).astype(np.float64)
        stats[name] = {
            "shape": list(array.shape),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "l2": math.sqrt(float(np.dot(flat, flat))),
            "first16": flat[:16].tolist(),
        }
    args.stats.write_text(json.dumps({"precision": "bfloat16", "attention": "eager", "input_ids": encoded["input_ids"][0].cpu().tolist(), "tensors": stats}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"npz": str(args.npz), "stats": str(args.stats), "tensors": len(arrays), "sequence": int(encoded["attention_mask"].sum())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
