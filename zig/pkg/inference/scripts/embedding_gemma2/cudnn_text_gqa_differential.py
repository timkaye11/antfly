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

"""Compare the direct cuDNN GQA proof output with PyTorch SDPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_bf16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} BF16 values, got {raw.size}")
    # Preserve each BF16 bit pattern when widening to float32.
    widened = raw.astype(np.uint32) << 16
    return torch.from_numpy(widened.view(np.float32).reshape(shape))


def cosine_float64(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.astype(np.float64, copy=False).ravel()
    candidate = candidate.astype(np.float64, copy=False).ravel()
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("reference and candidate must both be finite")
    reference_norm = np.linalg.norm(reference)
    candidate_norm = np.linalg.norm(candidate)
    if reference_norm == 0.0 or candidate_norm == 0.0:
        return 1.0 if reference_norm == candidate_norm else 0.0
    return float(np.dot(reference, candidate) / (reference_norm * candidate_norm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    q_shape = (args.batch, args.sequence, args.query_heads, args.head_dim)
    kv_shape = (args.batch, args.sequence, args.kv_heads, args.head_dim)
    q = read_bf16(args.directory / "cudnn_text_gqa_q.bf16", q_shape).permute(0, 2, 1, 3)
    k = read_bf16(args.directory / "cudnn_text_gqa_k.bf16", kv_shape).permute(0, 2, 1, 3)
    v = read_bf16(args.directory / "cudnn_text_gqa_v.bf16", kv_shape).permute(0, 2, 1, 3)
    candidate = read_bf16(
        args.directory / "cudnn_text_gqa_out.bf16", q_shape
    ).permute(0, 2, 1, 3)

    q = q.to(device=args.device, dtype=torch.bfloat16)
    k = k.to(device=args.device, dtype=torch.bfloat16)
    v = v.to(device=args.device, dtype=torch.bfloat16)
    reference = F.scaled_dot_product_attention(q, k, v, scale=args.scale, enable_gqa=True)
    reference_np = reference.float().cpu().numpy()
    candidate_np = candidate.numpy()
    delta = np.abs(reference_np.astype(np.float64) - candidate_np.astype(np.float64))
    result = {
        "finite": bool(np.isfinite(reference_np).all() and np.isfinite(candidate_np).all()),
        "cosine_float64": cosine_float64(reference_np, candidate_np),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
