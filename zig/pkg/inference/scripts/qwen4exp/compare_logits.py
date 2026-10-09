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

"""Compare teacher-forced native and pinned FP32-reference logits (NumPy)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--tokens", required=True, help="Comma-separated teacher-forced IDs"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokens = [int(token) for token in args.tokens.split(",")]
    if not tokens or any(token < 0 or token >= 248320 for token in tokens):
        parser.error("tokens must be nonempty and within the artifact vocabulary")
    arrays = []
    digests = []
    for path in (args.native, args.reference):
        raw = path.read_bytes()
        if len(raw) != len(tokens) * 248320 * 4:
            parser.error(f"{path}: expected one complete vocabulary per input token")
        values = (
            np.frombuffer(raw, dtype="<f4")
            .reshape(len(tokens), 248320)
            .astype(np.float64)
        )
        if not np.isfinite(values).all():
            parser.error(f"{path}: nonfinite logits")
        arrays.append(values)
        digests.append(hashlib.sha256(raw).hexdigest())
    positions = []
    for position, (native, reference) in enumerate(zip(*arrays)):
        denom = max(float(np.linalg.norm(reference)), 1e-30)
        relative_l2 = float(np.linalg.norm(native - reference) / denom)
        top1 = int(native.argmax())
        reference_top1 = int(reference.argmax())
        positions.append(
            {
                "position": position,
                "input_token": tokens[position],
                "native_top1": top1,
                "reference_top1": reference_top1,
                "relative_l2": relative_l2,
                "max_abs": float(np.max(np.abs(native - reference))),
                "cosine": float(
                    native
                    @ reference
                    / max(float(np.linalg.norm(native)) * denom, 1e-30)
                ),
                "passed": top1 == reference_top1 and relative_l2 <= 0.002,
            }
        )
    receipt = {
        "reference": "pinned llama.cpp graph with FP32 activation adapter; see reference.cpp",
        "source_manifest": json.loads(
            Path(__file__).with_name("artifact.json").read_text()
        ),
        "native_sha256": digests[0],
        "reference_sha256": digests[1],
        "relative_l2_limit": 0.002,
        "positions": positions,
        "passed": all(position["passed"] for position in positions),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "positions": positions}, indent=2))
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
