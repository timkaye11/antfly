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

"""Serial CPU/GPU dequantization A/B with optional full-vocabulary parity."""

import argparse
import array
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark import CASES, swapouts


def compare_logits(first, second):
    left = sorted(first.glob("logits-pos-*.f32"))
    assert left and [x.name for x in left] == sorted(
        x.name for x in second.glob("logits-pos-*.f32")
    )
    comparisons = []
    for path in left:
        x, y = array.array("f"), array.array("f")
        x.frombytes(path.read_bytes())
        y.frombytes((second / path.name).read_bytes())
        if sys.byteorder != "little":
            x.byteswap()
            y.byteswap()
        assert len(x) == len(y) == 248320
        assert all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y)
        error = [float(a) - b for a, b in zip(x, y)]
        relative = math.sqrt(
            sum(v * v for v in error) / max(sum(float(v) * v for v in x), 1e-30)
        )
        same_argmax = max(range(len(x)), key=x.__getitem__) == max(
            range(len(y)), key=y.__getitem__
        )
        comparisons.append(
            {
                "file": path.name,
                "max_abs": max(map(abs, error)),
                "relative_l2": relative,
                "argmax_equal": same_argmax,
            }
        )
        assert relative < 0.001 and same_argmax, comparisons[-1]
    return comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=CASES, default="chat")
    p.add_argument(
        "--order",
        nargs="+",
        choices=("cpu", "gpu"),
        default=["cpu", "gpu", "gpu", "cpu"],
    )
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--capture", action="store_true")
    a = p.parse_args()
    if not 2 <= a.tokens <= 64 or not {"cpu", "gpu"}.issubset(a.order):
        p.error("use 2..64 tokens and include both CPU and GPU variants")
    a.output.mkdir(parents=True, exist_ok=True)
    report = {
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "case": a.case,
        "profile": a.profile,
        "capture": a.capture,
        "cache_note": "Serial fresh processes; OS/device caches uncontrolled.",
        "cases": [],
    }
    baseline = None
    baseline_dir = None
    for index, variant in enumerate(a.order):
        dest = a.output / f"{index}-{variant}"
        dest.mkdir()  # Never silently overwrite evidence.
        receipt = dest / "receipt.json"
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTFLY_QWEN")}
        overrides = {}
        if variant == "cpu":
            overrides["ANTFLY_QWEN35_CPU_DEQUANT"] = "1"
        if a.profile:
            overrides["ANTFLY_QWEN35_PROFILE"] = "1"
        if a.capture:
            overrides.update(
                ANTFLY_QWEN35_CAPTURE_DIR=str(dest.resolve()),
                ANTFLY_QWEN35_CAPTURE_LOGITS_ONLY="1",
            )
        env.update(overrides)
        cmd = [
            str(a.binary.resolve()),
            str(a.model),
            CASES[a.case],
            "--experimental-qwen35-streamed",
            "--backend",
            "metal",
            "--speculation-policy",
            "off",
            "--context-size",
            "4096",
            "--max-tokens",
            str(a.tokens),
            "--disable-thinking",
            "--json-timing",
            str(receipt.resolve()),
        ]
        before = swapouts()
        started = time.monotonic()
        with (dest / "stdout").open("w") as out, (dest / "stderr").open("w") as err:
            result = subprocess.run(
                cmd, env=env, stdout=out, stderr=err, timeout=1800, check=False
            )
        row = {
            "variant": variant,
            "command": cmd,
            "env_overrides": overrides,
            "seconds": time.monotonic() - started,
            "swapouts_delta": swapouts() - before,
            "exit_code": result.returncode,
        }
        if receipt.exists():
            row["receipt"] = json.loads(receipt.read_text())
        report["cases"].append(row)
        (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(index, variant, row["seconds"], row["swapouts_delta"], flush=True)
        assert result.returncode == 0 and row["swapouts_delta"] == 0, row
        actual = row["receipt"]
        assert actual["cpu_weight_dequantization"] == (variant == "cpu")
        if variant == "gpu":
            assert actual["cpu_dequant_bytes"] == 0
        if baseline is None:
            baseline, baseline_dir = actual, dest
        else:
            assert baseline["prompt_token_ids"] == actual["prompt_token_ids"]
            assert baseline["generated_token_ids"] == actual["generated_token_ids"]
            if a.capture:
                row["logits"] = compare_logits(baseline_dir, dest)
    report["passed"] = True
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
