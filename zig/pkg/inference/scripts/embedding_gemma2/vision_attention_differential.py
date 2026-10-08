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

"""Numerical and timing gate for the production QT1024 EG2 vision attention."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("/tmp/vision_lt_q1024_scaled.so"))
    parser.add_argument("--report", type=Path, default=Path("/tmp/vision_lt_q1024_scaled.json"))
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    return parser.parse_args()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.float().flatten(), b.float().flatten()
    an, bn = float(af.norm()), float(bf.norm())
    if not math.isfinite(an) or not math.isfinite(bn):
        return float("nan")
    if an == 0.0 or bn == 0.0:
        return 1.0 if an == 0.0 and bn == 0.0 else 0.0
    value = float(torch.dot(af, bf) / (an * bn))
    return value if math.isfinite(value) else float("nan")


def load_library(path: Path):
    lib = ctypes.CDLL(str(path))
    fn = lib.antfly_eg2_attention_differential_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint] * 7 + [ctypes.c_void_p]
    return lib


def launch(lib, output, q, k, v, mask) -> None:
    batch, sequence, heads, head_dim = q.shape
    status = lib.antfly_eg2_attention_differential_launch(
        *(ctypes.c_void_p(t.data_ptr()) for t in (output, q, k, v, mask)),
        batch, sequence, heads, k.shape[2], head_dim, 0, 2,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
    )
    if status:
        raise RuntimeError(f"CUDA launch failed with status {status}")


def reference(q_scaled_bf16, k_bf16, v_bf16, valid) -> torch.Tensor:
    q = q_scaled_bf16.transpose(1, 2).float()
    k = k_bf16.transpose(1, 2).float()
    v = v_bf16.transpose(1, 2).float()
    scores = torch.matmul(q, k.transpose(-1, -2)) * 0.125
    scores = scores.masked_fill(~valid[:, None, None, :].bool(), -torch.inf)
    probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    output = torch.matmul(probabilities.float(), v).transpose(1, 2).contiguous()
    return output.masked_fill(~valid[:, :, None, None].bool(), 0)


def timed(lib, output, q, k, v, mask, iterations: int) -> float:
    for _ in range(3):
        launch(lib, output, q, k, v, mask)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        launch(lib, output, q, k, v, mask)
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    lib = load_library(args.library)
    torch.manual_seed(6214)
    rows = []
    for batch, sequence in ((1, 17), (1, 513), (1, 2394), (2, 2394)):
        for scale in (0.0, 0.1, 1.0):
            q0 = torch.randn((batch, sequence, 12, 64), device="cuda", dtype=torch.float32) * scale
            q = (q0 * 8).to(torch.bfloat16)
            k = (torch.randn_like(q0) * scale).to(torch.bfloat16)
            v = torch.randn_like(q0).to(torch.bfloat16)
            valid = torch.ones((batch, sequence), device="cuda", dtype=torch.int64)
            if batch == 2:
                valid[1, math.ceil(sequence * 0.75):] = 0
            output = torch.empty_like(q0)
            launch(lib, output, q, k, v, valid)
            torch.cuda.synchronize()
            expected = reference(q, k, v, valid)
            row = {
                "batch": batch,
                "sequence": sequence,
                "qk_scale": scale,
                "cosine": cosine(output, expected),
                "max_abs": float((output - expected).abs().max()),
                "latency_ms": timed(lib, output, q, k, v, valid, 20 if sequence < 1000 else 5),
            }
            row["passes"] = math.isfinite(row["cosine"]) and row["cosine"] >= args.minimum_cosine
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"minimum_cosine": args.minimum_cosine, "cases": rows, "pass": all(r["passes"] for r in rows)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
