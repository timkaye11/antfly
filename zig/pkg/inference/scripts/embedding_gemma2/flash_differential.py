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

"""Numerical/timing gate for the production EG2 CUDA attention routes.

Build first with build_flash_differential.sh. This script uses ctypes and the
active Torch CUDA stream and compiles the canonical kernel source directly.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, default=Path("/tmp/antfly-eg2-flash-differential.so"))
    p.add_argument("--lengths", default="1,15,16,17,129,513,8192")
    p.add_argument("--batches", default="1,2")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--qk-scales", default="0,0.1,1.0")
    p.add_argument("--skip-reference-above", type=int, default=8192)
    p.add_argument("--json", type=Path)
    p.add_argument("--lt", action="store_true", help="compare bounded cuBLASLt attention")
    return p.parse_args()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().flatten(), b.float().flatten()
    an, bn = float(torch.linalg.vector_norm(af)), float(torch.linalg.vector_norm(bf))
    if not math.isfinite(an) or not math.isfinite(bn):
        return float("nan")
    if an == 0.0 or bn == 0.0:
        return 1.0 if an == 0.0 and bn == 0.0 else 0.0
    value = float(torch.dot(af, bf) / (an * bn))
    return value if math.isfinite(value) else float("nan")


def call(lib, output, q, k, v, mask, radius: int, flash: bool) -> None:
    b, s, h, d = q.shape
    kh = k.shape[2]
    stream = torch.cuda.current_stream().cuda_stream
    status = lib.antfly_eg2_attention_differential_launch(
        ctypes.c_void_p(output.data_ptr()), ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(mask.data_ptr()), b, s, h, kh, d, radius,
        int(flash), ctypes.c_void_p(stream))
    if status:
        raise RuntimeError(f"CUDA launch failed with status {status}")


def reference(q, k, v, valid, radius: int) -> torch.Tensor:
    b, s, h, d = q.shape
    kh = k.shape[2]
    repeat = h // kh
    kt = k.repeat_interleave(repeat, dim=2).transpose(1, 2)
    vt = v.repeat_interleave(repeat, dim=2).transpose(1, 2)
    qt = q.transpose(1, 2)
    key_mask = valid[:, None, None, :].bool()
    if radius:
        pos = torch.arange(s, device=q.device)
        local = (pos[:, None] - pos[None, :]).abs() <= radius
        attn_mask = key_mask & local[None, None, :, :]
    else:
        attn_mask = key_mask
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask, scale=1.0)
    out = out.transpose(1, 2).contiguous()
    return out.masked_fill(~valid[:, :, None, None].bool(), 0).to(torch.bfloat16)


def timed(lib, output, q, k, v, mask, radius, flash, warmup, iterations) -> float:
    for _ in range(warmup):
        call(lib, output, q, k, v, mask, radius, flash)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        call(lib, output, q, k, v, mask, radius, flash)
    end.record(); end.synchronize()
    return begin.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    lib = ctypes.CDLL(str(args.library))
    fn = lib.antfly_eg2_attention_differential_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 7 + [ctypes.c_void_p]
    lengths = [int(x) for x in args.lengths.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    qk_scales = [float(x) for x in args.qk_scales.split(",")]
    torch.manual_seed(0xE622)
    rows = []
    for d, kh, radius, kind in ((256, 2, 512, "local"), (512, 1, 0, "global")):
        for b in batches:
            for s in lengths:
              for qk_scale in qk_scales:
                q = torch.randn((b, s, 4, d), device="cuda", dtype=torch.bfloat16) * qk_scale
                k = torch.randn((b, s, kh, d), device="cuda", dtype=torch.bfloat16) * qk_scale
                v = torch.randn((b, s, kh, d), device="cuda", dtype=torch.bfloat16)
                valid = torch.ones((b, s), device="cuda", dtype=torch.int64)
                if s > 1 and b > 1:
                    valid[1, math.ceil(s * 0.75):] = 0
                warp, flash = torch.empty_like(q), torch.empty_like(q)
                call(lib, warp, q, k, v, valid, radius, False)
                candidate = 2 if args.lt else 1
                call(lib, flash, q, k, v, valid, radius, candidate)
                torch.cuda.synchronize()
                ref = reference(q, k, v, valid, radius) if s <= args.skip_reference_above else None
                row = {
                    "kind": kind, "batch": b, "sequence": s, "head_dim": d,
                    "qk_scale": qk_scale,
                    "flash_warp_cosine": cosine(flash, warp),
                    "flash_warp_max_abs": float((flash.float() - warp.float()).abs().max()),
                    "warp_ms": timed(lib, warp, q, k, v, valid, radius, False, args.warmup, args.iterations),
                    "flash_ms": timed(lib, flash, q, k, v, valid, radius, candidate, args.warmup, args.iterations),
                }
                if ref is not None:
                    row.update(warp_reference_cosine=cosine(warp, ref),
                               flash_reference_cosine=cosine(flash, ref))
                row["passes_numerical_gate"] = (
                    math.isfinite(row["flash_warp_cosine"])
                    and row["flash_warp_cosine"] >= 0.999
                    and (ref is None or (
                        math.isfinite(row["flash_reference_cosine"])
                        and row["flash_reference_cosine"] >= 0.999
                    ))
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.json:
        args.json.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    if not all(r["passes_numerical_gate"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
