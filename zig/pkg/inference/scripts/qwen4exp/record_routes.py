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

"""Record Qwen4Exp expert routes for the hit-rate-versus-cache-size replay.

Runs chat, code and document prompts as separate one-shot processes and one
multi-turn chat session, each with ANTFLY_QWEN4EXP_ROUTE_TRACE set, and writes
a manifest for `cache_replay.py --manifest`. Routes do not depend on the cache
size used while recording: any admitted cache yields the same trace. Run
serially, with no other model process or build on the host.
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "qwen_local"))

from chat import Session, command
from evidence import read_receipt, swap_snapshot
from machine_profile import (
    FLAGS,
    MAX_CONTEXT,
    VERIFY_MODES,
    budget_arguments,
    dry_run,
    explicit_mb,
    physical_memory_bytes,
)

NOTES = (
    "The storage team met on Tuesday to review the migration of the search cluster. "
    "Three shards were moved to the new hosts without incident, but the fourth stalled "
    "when the replica fell behind and the leader refused to transfer ownership. The "
    "on-call engineer paused the migration, confirmed that no writes were lost, and "
    "opened an incident. Latency for the affected tenant rose from forty to ninety "
    "milliseconds at the ninety-fifth percentile for about twenty minutes. The team "
    "agreed that the transfer should wait for the replica to catch up rather than rely "
    "on a fixed timeout, and that the dashboard should show replication lag per shard. "
    "Nobody owns the dashboard change yet. The budget for additional hosts is fixed "
    "until the next quarter, so the remaining shards will be moved one at a time during "
    "the low-traffic window. A follow-up review is planned, but its date depends on "
    "whether the lag metric can be exported before the next migration attempt. "
)
PROMPTS = {
    "chat": (
        "Explain in detail how a refrigerator moves heat from its cold interior to the "
        "warmer room. Cover the refrigerant cycle step by step, what the compressor, "
        "condenser, expansion valve and evaporator each do, why the back of the "
        "appliance feels warm, and three practical ways to make one run more "
        "efficiently. Write at least six paragraphs."
    ),
    "code": (
        "Write a complete, well-commented Python module that implements an LRU cache "
        "with a fixed byte budget rather than a fixed entry count. Include a class with "
        "get, put and evict methods, per-entry size accounting, hit and miss counters, "
        "and a small command-line demo. Then write unit tests for eviction order, "
        "oversized entries and counter accuracy, and explain the time complexity of "
        "every method."
    ),
    "document": (
        "Read the following incident notes, which repeat several times in the log. "
        + NOTES * 6
        + "Write a thorough incident report from these notes with these sections: "
        "summary, timeline, customer impact, root cause, decisions made, open questions, "
        "and recommended follow-up work with an owner role for each item."
    ),
}
SESSION_TURNS = (
    (
        "I am planning a three-day hiking trip in the mountains in early autumn. What "
        "should I think about when choosing a route? Answer in several paragraphs."
    ),
    (
        "Suppose the forecast shows rain on the second day. How should I change the plan "
        "and what extra gear matters most? Be specific."
    ),
    (
        "Now write a day-by-day packing and food list for two people based on what we "
        "discussed, including approximate weights."
    ),
    (
        "One of us has a mild knee injury. Rewrite the plan with shorter days and explain "
        "each change you made."
    ),
    (
        "Summarize the final plan as a checklist I can print, then list the three biggest "
        "risks and how we are mitigating each."
    ),
)


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="first Flash shard")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--workloads",
        nargs="+",
        choices=(*PROMPTS, "session"),
        default=[*PROMPTS, "session"],
    )
    p.add_argument("--tokens", type=int, default=640, help="one-shot output limit")
    p.add_argument("--turn-tokens", type=int, default=160, help="session output limit")
    p.add_argument("--cache-mb", type=int, help="default: derived by the runner")
    p.add_argument("--combined-budget-mb", type=int, help="default: derived")
    p.add_argument("--verify", choices=VERIFY_MODES, default="stamp")
    p.add_argument("--timeout", type=float, default=7200)
    a = p.parse_args()
    try:
        cache = explicit_mb("expert cache", a.cache_mb)
        budget = explicit_mb("process budget", a.combined_budget_mb)
    except ValueError as error:
        p.error(str(error))
    if not 2 <= a.tokens < MAX_CONTEXT or not 2 <= a.turn_tokens < MAX_CONTEXT:
        p.error(f"token limits must be 2..{MAX_CONTEXT - 1}")
    a.output.mkdir(parents=True, exist_ok=False)
    plan = dry_run(a.binary, a.model, "flash", cache, budget)
    manifest = {
        "schema": 1,
        "purpose": "Qwen4Exp route traces for cache_replay.py; not timing evidence",
        "binary_sha256": digest(a.binary),
        "model": str(a.model),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "physical_memory_bytes": physical_memory_bytes(),
        },
        "plan": plan,
        "requested": {
            "cache_mb": cache,
            "combined_budget_mb": budget,
            "verify": a.verify,
        },
        "swap_before": swap_snapshot(),
        "workloads": [],
    }

    def save():
        manifest["swap_after"] = swap_snapshot()
        (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTFLY_QWEN")}
    for name in a.workloads:
        trace = a.output / f"{name}.u16"
        env["ANTFLY_QWEN4EXP_ROUTE_TRACE"] = str(trace)
        started = time.monotonic()
        row = {"name": name, "trace": trace.name}
        if name == "session":
            cmd = command(
                a.binary,
                a.model,
                "flash",
                a.turn_tokens,
                cache,
                combined_mb=budget,
                verify=a.verify,
            )
            turns = []
            with (a.output / "session.stderr").open("w") as err:
                session = Session(cmd, stderr=err, env=env)
                try:
                    for prompt in SESSION_TURNS:
                        done = session.request({"prompt": prompt})[-1]
                        if done["event"] != "done":
                            raise RuntimeError(f"session turn failed: {done}")
                        turns.append(
                            {
                                key: done[key]
                                for key in (
                                    "prompt_tokens",
                                    "reused_tokens",
                                    "prefilled_tokens",
                                    "position",
                                    "finish_reason",
                                    "prefill_ms",
                                    "decode_ms",
                                )
                            }
                            | {"generated_tokens": len(done["generated_token_ids"])}
                        )
                        print(name, len(turns), turns[-1], flush=True)
                        if done["position"] + a.turn_tokens + 512 > MAX_CONTEXT:
                            break
                finally:
                    # The runner writes the trace when the session ends.
                    session.close()
            for earlier, later in itertools.pairwise(turns):
                if later["reused_tokens"] != earlier["position"]:
                    raise RuntimeError(
                        "session did not extend one sequence; its trace was overwritten"
                    )
            log = {
                "turns": turns,
                "expert_strides": plan["expert_strides"],
                "prefill_chunk": session.ready["prefill_chunk"],
                "prefill_read_group": 10,
                "identity": session.ready,
                "command": cmd,
            }
            (a.output / "session.json").write_text(json.dumps(log, indent=2) + "\n")
            row |= {
                "session": "session.json",
                "decode_tokens": sum(
                    t["position"] - t["reused_tokens"] - t["prefilled_tokens"]
                    for t in turns
                ),
            }
        else:
            receipt = a.output / f"{name}.json"
            cmd = [
                str(a.binary.resolve()),
                str(a.model),
                PROMPTS[name],
                FLAGS["flash"],
                "--backend",
                "metal",
                "--speculation-policy",
                "off",
                "--context-size",
                str(MAX_CONTEXT),
                "--max-tokens",
                str(a.tokens),
                "--disable-thinking",
                "--json-timing",
                str(receipt),
            ] + budget_arguments("flash", cache, budget, a.verify)
            with (
                (a.output / f"{name}.stdout").open("w") as out,
                (a.output / f"{name}.stderr").open("w") as err,
            ):
                code = subprocess.run(
                    cmd, stdout=out, stderr=err, env=env, timeout=a.timeout, check=False
                ).returncode
            if code:
                raise RuntimeError(f"{name} exited with status {code}")
            data = read_receipt(receipt)
            if data.get("finish_reason") not in ("stop", "length"):
                raise RuntimeError(
                    f"{name} did not finish: {data.get('finish_reason')}"
                )
            row |= {
                "receipt": receipt.name,
                "command": cmd,
                "prompt_tokens": data["prompt_tokens"],
                "decode_tokens": data["decode_steps"],
                "finish_reason": data["finish_reason"],
                "decode_ms": data["decode_ms"],
                "decode_expert_io": data["decode_expert_io"],
            }
        row["seconds"] = time.monotonic() - started
        row["trace_bytes"] = trace.stat().st_size
        manifest["workloads"].append(row)
        save()
        print(
            name,
            row["decode_tokens"],
            "decode tokens",
            f"{row['seconds']:.0f}s",
            flush=True,
        )
    short = [w["name"] for w in manifest["workloads"] if w["decode_tokens"] < 500]
    if short:
        print(
            "fewer than 500 decoded tokens:",
            ", ".join(short),
            "(the model stopped early; raise --tokens or extend the prompt)",
        )


if __name__ == "__main__":
    main()
