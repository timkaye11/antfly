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

"""Hardware checks for streamed local sessions, prefix reuse, and state reset."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from benchmark import swapouts
from chat import Session, command, process_budget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", choices=("27b", "flash"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-mb", type=int, default=1536)
    p.add_argument("--mode", choices=("both", "raw", "chat"), default="both")
    p.add_argument("--combined-budget-mb", type=int)
    a = p.parse_args()
    try:
        budget = process_budget(a.family, a.combined_budget_mb)
    except ValueError as error:
        p.error(str(error))
    a.output.mkdir(parents=True, exist_ok=True)
    report_path = a.output / "summary.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTFLY_QWEN")}
    report = {
        "family": a.family,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "combined_budget_mb": budget,
        "environment_note": "Inherited ANTFLY_QWEN toggles removed",
        "commands": {},
        "cases": {},
    }
    before = swapouts()
    for raw in (True, False) if a.mode == "both" else (a.mode == "raw",):
        mode = "raw" if raw else "chat"
        with (a.output / f"{mode}.stderr").open("w") as err:
            started = time.monotonic()
            cmd = command(
                a.binary, a.model, a.family, 2, a.cache_mb, raw, combined_mb=budget
            )
            report["commands"][mode] = cmd
            session = Session(cmd, stderr=err, env=env)
            report[f"{mode}_load_seconds"] = time.monotonic() - started
            try:

                def run(name, request, session=session, mode=mode):
                    events = session.request(request)
                    report["cases"][f"{mode}_{name}"] = events
                    (a.output / "summary.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(
                        mode,
                        name,
                        events[-1].get("reused_tokens"),
                        events[-1].get("prefill_ms"),
                        flush=True,
                    )
                    return events

                first = run(
                    "first",
                    {
                        "prompt": "The capital of France is"
                        if raw
                        else "Reply with only the word hello."
                    },
                )
                assert first[-1]["event"] == "done"
                assert first[-1]["reused_tokens"] == 0
                prompt = (
                    "The capital of France is"
                    + first[-1]["text"]
                    + " The capital of Italy is"
                    if raw
                    else "Reply with only the word goodbye."
                )
                warm = run("warm", {"prompt": prompt})
                assert warm[-1]["event"] == "done"
                assert warm[-1]["reused_tokens"] > 0
                for result in (first, warm):
                    assert (
                        "".join(e["text"] for e in result if e["event"] == "delta")
                        == result[-1]["text"]
                    )
                    assert result[-1]["position"] <= 4096
                    limit = budget * 1024**2
                    assert result[-1]["process_memory"]["peak_footprint_bytes"] < limit
                if raw:
                    cold = run("reset_parity", {"reset": True, "prompt": prompt})
                    assert cold[-1]["reused_tokens"] == 0
                    assert (
                        cold[-1]["generated_token_ids"]
                        == warm[-1]["generated_token_ids"]
                    )
                    mismatch = run("mismatch", {"prompt": "The capital of France is"})
                    assert mismatch[-1]["reused_tokens"] == 0
                    assert (
                        mismatch[-1]["generated_token_ids"]
                        == first[-1]["generated_token_ids"]
                    )
                invalid = run("invalid", {"prompt": "ignored", "max_tokens": 0})
                assert invalid[-1]["event"] == "error" and not invalid[-1]["fatal"]
                reset = run("reset", {"reset": True})
                assert reset[-1]["event"] == "reset"
                again = run(
                    "after_reset",
                    {
                        "prompt": "The capital of France is"
                        if raw
                        else "Reply with only the word hello."
                    },
                )
                assert (
                    again[-1]["generated_token_ids"] == first[-1]["generated_token_ids"]
                )
            finally:
                session.close()
    report["swapouts_delta"] = swapouts() - before
    report["passed"] = report["swapouts_delta"] == 0
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"]


if __name__ == "__main__":
    main()
