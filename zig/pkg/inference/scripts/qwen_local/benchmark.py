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

"""Serial same-prompt benchmark for the two experimental local Qwen runners."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from chat import process_budget

CASES = {
    "chat": "Explain why the sky is blue in two short sentences, then give one everyday example of light scattering.",
    "code": "Write a Python function that merges two sorted lists without sorting the result. Explain its time and space complexity and handle empty inputs.",
    "document": "Summarize the decisions and unresolved questions in these meeting notes in three bullets. "
    + "The team agreed to keep the service on the current hardware, measure latency before changing cache sizes, and preserve output correctness. The budget is fixed. Ownership of the next benchmark and the rollout date remain undecided. "
    * 8,
}


def swapouts():
    return int(
        next(
            x
            for x in subprocess.check_output(["vm_stat"], text=True).splitlines()
            if x.startswith("Swapouts:")
        )
        .split(":")[1]
        .strip()
        .rstrip(".")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", choices=("27b", "flash"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--chunk", type=int)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--cache-mb", type=int, default=1536)
    p.add_argument("--combined-budget-mb", type=int)
    p.add_argument("--batch-linear", action="store_true")
    p.add_argument("--prefill-mps", action="store_true")
    a = p.parse_args()
    try:
        budget = process_budget(a.family, a.combined_budget_mb)
    except ValueError as error:
        p.error(str(error))
    if a.family != "flash" and (a.batch_linear or a.prefill_mps or a.trace):
        p.error("batch-linear, prefill-mps, and route traces require family flash")
    if not 1 <= a.rounds <= 10 or not 2 <= a.tokens <= 64:
        p.error("rounds must be 1..10 and tokens 2..64")
    a.output.mkdir(parents=True, exist_ok=True)
    report = {
        "family": a.family,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "model": str(a.model),
        "cases": [],
        "cache_note": "Serial fresh processes; OS/device caches uncontrolled.",
    }
    for round_id in range(a.rounds):
        for name, prompt in CASES.items():
            dest = a.output / f"{round_id}-{name}"
            dest.mkdir(exist_ok=True)
            receipt = dest / "receipt.json"
            if receipt.exists():
                raise FileExistsError(receipt)
            command = [
                str(a.binary.resolve()),
                str(a.model),
                prompt,
                "--experimental-qwen35-streamed"
                if a.family == "27b"
                else "--experimental-qwen4exp",
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
                str(receipt),
            ]
            command += ["--combined-budget-mb", str(budget)]
            if a.family == "flash":
                command += ["--moe-cache-mb", str(a.cache_mb)]
            if a.stream:
                command += ["--stream"]
            if a.chunk is not None:
                command += ["--prefill-chunk-size", str(a.chunk)]
            env = os.environ.copy()
            for key in list(env):
                if key.startswith(("ANTFLY_QWEN35_", "ANTFLY_QWEN4EXP_")):
                    env.pop(key)
            if a.batch_linear:
                env["ANTFLY_QWEN4EXP_BATCH_LINEAR"] = "1"
            if a.prefill_mps:
                env["ANTFLY_QWEN4EXP_PREFILL_MPS"] = "1"
            if a.trace and a.family == "flash":
                env["ANTFLY_QWEN4EXP_ROUTE_TRACE"] = str(dest / "routes.u16")
            before = swapouts()
            start = time.monotonic()
            with (dest / "stdout").open("w") as out, (dest / "stderr").open("w") as err:
                result = subprocess.run(
                    command, stdout=out, stderr=err, env=env, timeout=2400, check=False
                )
            row = {
                "round": round_id,
                "case": name,
                "prompt": prompt,
                "command": command,
                "exit_code": result.returncode,
                "seconds": time.monotonic() - start,
                "swapouts_delta": swapouts() - before,
                "receipt_path": str(receipt),
            }
            if receipt.exists():
                row["receipt"] = json.loads(receipt.read_text())
            report["cases"].append(row)
            (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, round_id, row["seconds"], row["swapouts_delta"], flush=True)
            if result.returncode or row["swapouts_delta"]:
                raise RuntimeError(
                    "case failed or system swapouts increased; inspect report"
                )


if __name__ == "__main__":
    main()
