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

"""Serial 27B chunk and prompt-lookup qualification with full-vocabulary captures."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from benchmark import CASES, swapouts
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("chunks", "lookup"), required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    t = Tokenizer.from_file(str(a.model.parent / "tokenizer.json"))
    if a.mode == "chunks":
        ids = t.encode(CASES["document"] * 4, add_special_tokens=False).ids[:1024]
        prompt = t.decode(ids, skip_special_tokens=False)
        assert len(t.encode(prompt, add_special_tokens=False).ids) == 1024
        variants = [("chunk512", 512, 0), ("chunk1024", 1024, 0)]
        max_tokens = 2
    else:
        prompt = "x" * 1024
        variants = [("control", 512, 0), ("lookup", 512, 4)]
        max_tokens = 8
    report = {
        "mode": a.mode,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "prompt": prompt,
        "cases": [],
    }
    for name, chunk, lookup in variants:
        dest = a.output / name
        dest.mkdir(exist_ok=True)
        receipt = dest / "receipt.json"
        if receipt.exists():
            raise FileExistsError(receipt)
        cmd = [
            str(a.binary.resolve()),
            str(a.model),
            prompt,
            "--experimental-qwen35-streamed",
            "--backend",
            "metal",
            "--speculation-policy",
            "force" if lookup else "off",
            "--raw-prompt",
            "--ignore-eos",
            "--max-tokens",
            str(max_tokens),
            "--prefill-chunk-size",
            str(chunk),
            "--json-timing",
            str(receipt),
        ]
        if lookup:
            cmd += ["--prompt-lookup", str(lookup)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTFLY_QWEN")}
        if a.mode == "chunks":
            env["ANTFLY_QWEN35_CAPTURE_DIR"] = str(dest)
            env["ANTFLY_QWEN35_CAPTURE_LOGITS_ONLY"] = "1"
        before = swapouts()
        started = time.monotonic()
        with (dest / "stdout").open("w") as out, (dest / "stderr").open("w") as err:
            result = subprocess.run(
                cmd, stdout=out, stderr=err, env=env, timeout=1800, check=False
            )
        row = {
            "name": name,
            "command": cmd,
            "seconds": time.monotonic() - started,
            "swapouts_delta": swapouts() - before,
            "exit_code": result.returncode,
        }
        if receipt.exists():
            row["receipt"] = json.loads(receipt.read_text())
        report["cases"].append(row)
        (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, row["seconds"], row["swapouts_delta"], flush=True)
        assert result.returncode == 0 and row["swapouts_delta"] == 0
    first, second = [r["receipt"] for r in report["cases"]]
    assert first["prompt_token_ids"] == second["prompt_token_ids"]
    assert first["generated_token_ids"] == second["generated_token_ids"]
    report["tokens_equal"] = True
    if a.mode == "chunks":
        comparisons = []
        for pos in (1024, 1025):
            x = np.fromfile(
                a.output / "chunk512" / f"logits-pos-{pos}.f32", dtype="<f4"
            )
            y = np.fromfile(
                a.output / "chunk1024" / f"logits-pos-{pos}.f32", dtype="<f4"
            )
            assert (
                x.size == y.size == 248320
                and np.isfinite(x).all()
                and np.isfinite(y).all()
            )
            rel = float(
                np.linalg.norm(x.astype(float) - y) / np.linalg.norm(x.astype(float))
            )
            comparisons.append(
                {
                    "position": pos,
                    "max_abs": float(np.max(np.abs(x - y))),
                    "relative_l2": rel,
                    "argmax_equal": int(x.argmax()) == int(y.argmax()),
                }
            )
            assert rel < 0.001 and x.argmax() == y.argmax()
        report["logits"] = comparisons
    else:
        assert second["lookup_stats"]["accepted"] > 0
    report["passed"] = True
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
