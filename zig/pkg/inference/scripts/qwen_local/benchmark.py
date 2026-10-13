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
import platform
import subprocess
import time
from pathlib import Path

from chat import process_budget
from evidence import (
    clean_swap_delta,
    read_receipt,
    swap_snapshot,
    validate_receipt,
    validate_repeat,
)
from machine_profile import (
    FLAGS,
    MAX_CONTEXT,
    VERIFY_MODES,
    budget_arguments,
    explicit_mb,
    physical_memory_bytes,
)

CASES = {
    "chat": "Explain why the sky is blue in two short sentences, then give one everyday example of light scattering.",
    "code": "Write a Python function that merges two sorted lists without sorting the result. Explain its time and space complexity and handle empty inputs.",
    "document": "Summarize the decisions and unresolved questions in these meeting notes in three bullets. "
    + "The team agreed to keep the service on the current hardware, measure latency before changing cache sizes, and preserve output correctness. The budget is fixed. Ownership of the next benchmark and the rollout date remain undecided. "
    * 8,
}


def swapouts():
    # Retained for older experiment harnesses; this benchmark checks both counters.
    return swap_snapshot()["swapouts"]


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
    p.add_argument(
        "--cache-mb",
        type=int,
        help="Flash expert cache in MiB (default: derived by the runner)",
    )
    p.add_argument(
        "--combined-budget-mb",
        type=int,
        help="Process cap in MiB (default: derived by the runner)",
    )
    p.add_argument("--context", type=int, default=MAX_CONTEXT)
    p.add_argument(
        "--verify",
        choices=VERIFY_MODES,
        default="stamp",
        help="stamp hashes once per host and install; full hashes in every process",
    )
    p.add_argument("--timeout", type=float, default=2400)
    p.add_argument("--batch-linear", action="store_true")
    p.add_argument("--prefill-mps", action="store_true")
    a = p.parse_args()
    try:
        budget = process_budget(a.family, a.combined_budget_mb)
        cache = explicit_mb("expert cache", a.cache_mb)
    except ValueError as error:
        p.error(str(error))
    if a.family != "flash" and (
        a.batch_linear or a.prefill_mps or a.trace or cache is not None
    ):
        p.error(
            "batch-linear, prefill-mps, route traces and cache-mb require family flash"
        )
    if not 1 <= a.context <= MAX_CONTEXT:
        p.error(f"context must be 1..{MAX_CONTEXT}")
    if not 1 <= a.rounds <= 10 or not 2 <= a.tokens < a.context:
        p.error("rounds must be 1..10 and tokens 2..context-1")
    a.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 2,
        "benchmark_checks_passed": False,
        "family": a.family,
        "rounds_requested": a.rounds,
        "repeated_greedy_check": a.rounds > 1,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "model": str(a.model),
        "cases": [],
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "physical_memory_bytes": physical_memory_bytes(),
        },
        "requested": {
            "combined_budget_mb": budget,
            "cache_mb": cache,
            "context": a.context,
            "verify": a.verify,
        },
        "scope": "Local timing and repeated greedy-output checks; not independent parity or production qualification.",
        "cache_note": "Serial fresh processes; OS/device caches uncontrolled.",
    }
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    previous = {}
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
                FLAGS[a.family],
                "--backend",
                "metal",
                "--speculation-policy",
                "off",
                "--context-size",
                str(a.context),
                "--max-tokens",
                str(a.tokens),
                "--disable-thinking",
                "--json-timing",
                str(receipt),
            ]
            command += budget_arguments(a.family, cache, budget, a.verify)
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
            before = swap_snapshot()
            start = time.monotonic()
            row = {
                "round": round_id,
                "case": name,
                "prompt": prompt,
                "command": command,
                "checks_passed": False,
                "swap_before": before,
                "receipt_path": str(receipt),
            }
            try:
                with (
                    (dest / "stdout").open("w") as out,
                    (dest / "stderr").open("w") as err,
                ):
                    result = subprocess.run(
                        command,
                        stdout=out,
                        stderr=err,
                        env=env,
                        timeout=a.timeout,
                        check=False,
                    )
                row["exit_code"] = result.returncode
                row["swap_after"] = swap_snapshot()
                for key in ("swapins", "swapouts"):
                    row[f"{key}_delta"] = row["swap_after"][key] - before[key]
                if result.returncode:
                    raise RuntimeError(f"case exited with status {result.returncode}")
                # A zero exit status without a complete valid receipt is a failure.
                row["receipt"] = read_receipt(receipt)
                row["metrics"] = validate_receipt(
                    row["receipt"],
                    a.family,
                    budget,
                    a.tokens,
                    context=a.context,
                    verify=a.verify,
                    cache_mb=cache,
                )
                clean_swap_delta(before, row["swap_after"])
                if name in previous:
                    validate_repeat(previous[name], row["receipt"])
                previous[name] = row["receipt"]
                row["checks_passed"] = True
            except BaseException as error:
                # Preserve evidence even when the operator interrupts a run.
                row["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                row["seconds"] = time.monotonic() - start
                report["cases"].append(row)
                (a.output / "summary.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(
                    name,
                    round_id,
                    row["seconds"],
                    row.get("error", "checks passed"),
                    flush=True,
                )
    report["benchmark_checks_passed"] = True
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
