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

"""Build and qualify the direct-C cuDNN inclusive local-band GQA proof."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def read_bf16(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} BF16 values, got {raw.size}")
    return torch.from_numpy(
        (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    )


def cosine_float64(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = reference.astype(np.float64, copy=False).ravel()
    b = candidate.astype(np.float64, copy=False).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("reference and candidate must both be finite")
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0.0 or bn == 0.0:
        return 1.0 if an == bn else 0.0
    return float(np.dot(a, b) / (an * bn))


def build(args: argparse.Namespace) -> None:
    subprocess.run(
        [
            args.nvcc,
            "-O2",
            "-std=c++17",
            f"-I{args.cudnn_include}",
            str(args.source),
            f"-L{args.cudnn_lib}",
            "-l:libcudnn.so.9",
            "-o",
            str(args.binary),
        ],
        check=True,
    )


def reference_chunks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sequence: int,
    chunk_rows: int,
) -> np.ndarray:
    positions = torch.arange(sequence, device=q.device)
    chunks: list[np.ndarray] = []
    for begin in range(0, sequence, chunk_rows):
        end = min(begin + chunk_rows, sequence)
        query_positions = positions[begin:end]
        mask = (query_positions[:, None] - positions[None, :]).abs() <= 512
        output = F.scaled_dot_product_attention(
            q[:, :, begin:end, :], k, v, attn_mask=mask, scale=1.0, enable_gqa=True
        )
        chunks.append(output.float().cpu().numpy())
    return np.concatenate(chunks, axis=2)


def run_case(
    args: argparse.Namespace, batch: int, sequence: int, amplitude: float
) -> dict[str, object]:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cudnn_lib}:{args.cuda_lib}:" + environment.get("LD_LIBRARY_PATH", "")
    )
    process = subprocess.run(
        [str(args.binary), str(sequence), str(batch), str(amplitude)],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    match = re.search(r"workspace=(\d+) avg_ms=([0-9.eE+-]+)", process.stdout)
    if match is None:
        raise RuntimeError(f"unexpected proof output: {process.stdout}")

    q_shape = (batch, sequence, 4, 256)
    kv_shape = (batch, sequence, 2, 256)
    directory = Path("/tmp")
    q = read_bf16(directory / "cudnn_text_gqa_q.bf16", q_shape).permute(0, 2, 1, 3)
    k = read_bf16(directory / "cudnn_text_gqa_k.bf16", kv_shape).permute(0, 2, 1, 3)
    v = read_bf16(directory / "cudnn_text_gqa_v.bf16", kv_shape).permute(0, 2, 1, 3)
    candidate = (
        read_bf16(directory / "cudnn_text_gqa_out.bf16", q_shape)
        .permute(0, 2, 1, 3)
        .numpy()
    )
    reference = reference_chunks(
        q.to(args.device, dtype=torch.bfloat16),
        k.to(args.device, dtype=torch.bfloat16),
        v.to(args.device, dtype=torch.bfloat16),
        sequence,
        args.chunk_rows,
    )
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "batch": batch,
        "sequence": sequence,
        "amplitude": amplitude,
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
        "cosine_float64": cosine_float64(reference, candidate),
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "workspace_bytes": int(match.group(1)),
        "average_ms": float(match.group(2)),
    }


def run_boundary_case(args: argparse.Namespace) -> dict[str, object]:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cudnn_lib}:{args.cuda_lib}:" + environment.get("LD_LIBRARY_PATH", "")
    )
    subprocess.run(
        [str(args.binary), "515", "1", "0", "boundary"],
        check=True,
        capture_output=True,
        env=environment,
    )
    output = read_bf16(Path("/tmp/cudnn_text_gqa_out.bf16"), (1, 515, 4, 256)).numpy()
    # All logits are zero. Each sentinel contributes exactly one value to its
    # channel, so excluded boundary positions must remain exact zero.
    checks = {
        "q0_includes_k512": float(output[0, 0, 0, 2]),
        "q0_excludes_k513": float(output[0, 0, 0, 3]),
        "q513_excludes_k0": float(output[0, 513, 0, 0]),
        "q513_includes_k1": float(output[0, 513, 0, 1]),
        "q514_excludes_k1": float(output[0, 514, 0, 1]),
        "q514_includes_k512": float(output[0, 514, 0, 2]),
    }
    expected_513 = float(torch.tensor(1.0 / 513.0, dtype=torch.bfloat16).float())
    expected_514 = float(torch.tensor(1.0 / 514.0, dtype=torch.bfloat16).float())
    passed = (
        checks["q0_includes_k512"] == expected_513
        and checks["q0_excludes_k513"] == 0.0
        and checks["q513_excludes_k0"] == 0.0
        and checks["q513_includes_k1"] == expected_514
        and checks["q514_excludes_k1"] == 0.0
        and checks["q514_includes_k512"] == expected_513
    )
    return {
        "sequence": 515,
        "expected_1_over_513_bf16": expected_513,
        "expected_1_over_514_bf16": expected_514,
        "values": checks,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--source", type=Path, default=here / "cudnn_text_local_band_backend_proof.c"
    )
    parser.add_argument(
        "--binary", type=Path, default=Path("/tmp/cudnn_text_local_band_backend")
    )
    parser.add_argument("--nvcc", default="/usr/local/cuda-13.2/bin/nvcc")
    parser.add_argument(
        "--cudnn-include",
        type=Path,
        default=Path(
            "/tmp/antfly-gliner25-family/venv/lib/python3.11/site-packages/nvidia/cudnn/include"
        ),
    )
    parser.add_argument(
        "--cudnn-lib",
        type=Path,
        default=Path("/tmp/antfly-embeddinggemma2/cudnn-native-runtime-9.24.0.43/lib"),
    )
    parser.add_argument(
        "--cuda-lib", type=Path, default=Path("/usr/local/cuda-13.2/lib64")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if not args.no_build:
        build(args)

    cases = [
        run_case(args, batch, sequence, amplitude)
        for batch, sequence in ((32, 2048), (1, 8192))
        for amplitude in (0.0, 0.1, 1.0)
    ]
    boundary = run_boundary_case(args)
    result = {
        "device": "NVIDIA L4 SM89",
        "cudnn_version": 92400,
        "mask": "abs(query_position - key_position) <= 512",
        "query_heads": 4,
        "kv_heads": 2,
        "head_dim": 256,
        "scale": 1.0,
        "reference": "torch.nn.functional.scaled_dot_product_attention BF16, chunked query rows",
        "cases": cases,
        "strict_boundary": boundary,
        "minimum_cosine": min(case["cosine_float64"] for case in cases),
        "all_finite": all(case["finite"] for case in cases),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(encoded)
    print(encoded, end="")
    if (
        not result["all_finite"]
        or result["minimum_cosine"] < 0.999
        or not boundary["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
