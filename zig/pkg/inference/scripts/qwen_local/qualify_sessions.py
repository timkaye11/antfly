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

from chat import Session, command, process_budget
from evidence import swap_snapshot
from machine_profile import MIB, explicit_mb


def main():
    if not __debug__:
        raise RuntimeError("qualification requires Python assertions")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", choices=("27b", "flash"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--cache-mb",
        type=int,
        help="Flash expert cache in MiB (default: derived by the runner)",
    )
    p.add_argument("--mode", choices=("both", "raw", "chat"), default="both")
    p.add_argument(
        "--combined-budget-mb",
        type=int,
        help="Process cap in MiB (default: derived by the runner)",
    )
    p.add_argument("--soak-turns", type=int, default=8)
    a = p.parse_args()
    try:
        budget = process_budget(a.family, a.combined_budget_mb)
        cache = explicit_mb("expert cache", a.cache_mb)
    except ValueError as error:
        p.error(str(error))
    if not 4 <= a.soak_turns <= 100:
        p.error("soak-turns must be 4..100")
    a.output.mkdir(parents=True, exist_ok=False)
    report_path = a.output / "summary.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTFLY_QWEN")}
    report = {
        "passed": False,
        "family": a.family,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "combined_budget_mb": budget,
        "cache_mb": cache,
        "environment_note": "Inherited ANTFLY_QWEN toggles removed",
        "commands": {},
        "cases": {},
    }
    before = swap_snapshot()
    report["swap_before"] = before
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for raw in (True, False) if a.mode == "both" else (a.mode == "raw",):
        mode = "raw" if raw else "chat"
        with (a.output / f"{mode}.stderr").open("w") as err:
            started = time.monotonic()
            # Qualification always hashes every artifact byte in this process.
            cmd = command(
                a.binary,
                a.model,
                a.family,
                2,
                cache,
                raw,
                combined_mb=budget,
                verify="full",
            )
            report["commands"][mode] = cmd
            session = Session(cmd, stderr=err, env=env)
            report[f"{mode}_load_seconds"] = time.monotonic() - started
            report[f"{mode}_identity"] = session.ready
            if session.ready.get("hash_verified") is not True:
                session.close()
                raise AssertionError("session did not verify model artifact")
            # The runner reports the cap it enforces; an explicit request
            # must come back unchanged.
            limit = session.ready.get("process_cap_bytes")
            context = session.ready.get("context_size")
            if type(limit) is not int or limit <= 0 or type(context) is not int:
                session.close()
                raise AssertionError("session did not report its admitted limits")
            if budget is not None and limit != budget * MIB:
                session.close()
                raise AssertionError("runner enforced a different process cap")
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
                    assert result[-1]["position"] <= context
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
                for name, bad in (
                    ("unknown_field", {"unexpected": True}),
                    ("wrong_type", {"prompt": 7}),
                    ("empty", {"prompt": ""}),
                    ("too_many_tokens", {"prompt": "x", "max_tokens": context + 1}),
                    ("context_overflow", {"prompt": " x" * context}),
                ):
                    invalid = run(name, bad)
                    assert invalid[-1]["event"] == "error" and not invalid[-1]["fatal"]
                session.process.stdin.write(b'{"prompt":\n')
                session.process.stdin.flush()
                malformed = session.read_event(time.monotonic() + 15)
                report["cases"][f"{mode}_malformed_json"] = [malformed]
                assert malformed["event"] == "error" and not malformed["fatal"]
                # Stopping a reply must keep the session, the loaded model and
                # deterministic output: the next turn matches the first one.
                session.send(
                    {
                        "reset": True,
                        "prompt": "The capital of France is"
                        if raw
                        else "Count from one to one hundred in words.",
                        "max_tokens": 48,
                    }
                )
                early = session.collect(until=lambda event: event["event"] == "delta")
                assert early[-1]["event"] == "delta", early[-1]
                session.cancel()
                stopped = session.collect()
                report["cases"][f"{mode}_cancel"] = stopped
                done = stopped[-1]
                assert done["event"] == "done", done
                assert done["finish_reason"] == "cancelled", done
                assert 1 <= len(done["generated_token_ids"]) < 48
                assert (
                    "".join(e["text"] for e in stopped if e["event"] == "delta")
                    == done["text"]
                )
                resumed = run(
                    "after_cancel",
                    {
                        "reset": True,
                        "prompt": "The capital of France is"
                        if raw
                        else "Reply with only the word hello.",
                    },
                )
                assert resumed[-1]["event"] == "done"
                assert (
                    resumed[-1]["generated_token_ids"]
                    == first[-1]["generated_token_ids"]
                )
                # A cancel with nothing to stop is ignored without a reply,
                # so it cannot desynchronise the next request.
                session.cancel()
                unicode_result = run(
                    "unicode",
                    {"reset": True, "prompt": "Say café 東京 🌍", "max_tokens": 1},
                )
                assert unicode_result[-1]["event"] == "done"
                assert (
                    "".join(e["text"] for e in unicode_result if e["event"] == "delta")
                    == unicode_result[-1]["text"]
                )
                footprints = []
                for turn in range(a.soak_turns):
                    result = run(
                        f"soak_{turn}",
                        {
                            "reset": True,
                            "prompt": "The capital of France is"
                            if raw
                            else "Reply with only the word hello.",
                        },
                    )
                    done = result[-1]
                    assert done["event"] == "done" and done["reused_tokens"] == 0
                    assert (
                        done["generated_token_ids"] == first[-1]["generated_token_ids"]
                    )
                    assert (
                        "".join(e["text"] for e in result if e["event"] == "delta")
                        == done["text"]
                    )
                    memory = done["process_memory"]
                    assert (
                        memory["available"]
                        and 0 < memory["peak_footprint_bytes"] <= limit
                    )
                    footprints.append(memory["footprint_bytes"])
                # Compare a fixed repeated workload after warmup, allowing bounded
                # allocator/cache settling, not unbounded growth with turn count.
                growth = max(footprints[2:]) - min(footprints[2:])
                assert growth <= 128 * 1024**2, ("session footprint growth", growth)
                report[f"{mode}_soak"] = {
                    "turns": a.soak_turns,
                    "footprints": footprints,
                    "post_warmup_range_bytes": growth,
                    "limit_bytes": 128 * 1024**2,
                }

            finally:
                session.close()
    report["swap_after"] = swap_snapshot()
    report["swap_delta"] = {
        key: report["swap_after"][key] - before[key] for key in before
    }
    report["functional_checks_passed"] = True
    report["passed"] = all(value == 0 for value in report["swap_delta"].values())
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"]


if __name__ == "__main__":
    main()
