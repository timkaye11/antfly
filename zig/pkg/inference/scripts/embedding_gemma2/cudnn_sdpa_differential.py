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

"""Compare the direct cuDNN C-backend proof with PyTorch SDPA on exact BF16 inputs."""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load(path: str, batch: int, sequence: int):
    raw = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(raw).view(torch.bfloat16).cuda().reshape(batch, sequence, 12, 64).permute(0, 2, 1, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for sequence in (17, 513, 2394):
        for amplitude in (0.0, 0.1, 1.0):
            run = subprocess.check_output([str(args.proof), str(sequence), "1", str(amplitude)], text=True).strip()
            q, k, v = (load(f"/tmp/cudnn_sdpa_{name}.bf16", 1, sequence) for name in ("q", "k", "v"))
            with torch.no_grad():
                reference = F.scaled_dot_product_attention(q, k, v, scale=0.125).permute(0, 2, 1, 3).float().cpu().flatten()
            raw = np.fromfile("/tmp/cudnn_sdpa_out.bf16", dtype=np.uint16)
            candidate = torch.from_numpy(raw).view(torch.bfloat16).float()
            error = (reference - candidate).abs()
            reference64 = reference.numpy().astype(np.float64, copy=False)
            candidate64 = candidate.numpy().astype(np.float64, copy=False)
            reference_norm = np.linalg.norm(reference64)
            candidate_norm = np.linalg.norm(candidate64)
            if reference_norm == 0.0 or candidate_norm == 0.0:
                cosine = 1.0 if reference_norm == candidate_norm else 0.0
            else:
                cosine = float(np.dot(reference64, candidate64) / (reference_norm * candidate_norm))
            finite = bool(np.isfinite(reference64).all() and np.isfinite(candidate64).all())
            rows.append({"sequence": sequence, "amplitude": amplitude, "cosine": cosine,
                         "max_abs": error.max().item(), "mean_abs": error.mean().item(),
                         "finite": finite, "run": run})
    report = {"schema": "antfly.embedding_gemma2.cudnn_sdpa_differential.v1",
              "pass": all(row["finite"] and row["cosine"] >= 0.999 for row in rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
