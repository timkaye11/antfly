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

"""Matched, serial Qwen4Exp measurements. No model downloads or cache purges.

Each sample starts a fresh process with an empty application expert cache.
The OS cache is uncontrolled; the native expert reader requests F_NOCACHE.
Use --profile for diagnostics only, and separate normal runs for throughput.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(*args):
    p = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"status": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def host():
    return {
        "time_ns": time.time_ns(),
        "vm": command("vm_stat"),
        "swap": command("sysctl", "vm.swapusage"),
        "power": command("pmset", "-g", "batt"),
    }


def swap_events(snapshot):
    text = snapshot["vm"]["stdout"]
    return {
        name: int(re.search(rf"^{name}:\s+(\d+)", text, re.MULTILINE).group(1))
        for name in ("Swapins", "Swapouts")
        if re.search(rf"^{name}:\s+(\d+)", text, re.MULTILINE)
    }


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (pos - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        action="append",
        required=True,
        help="NAME=/absolute/executable; repeat for matched comparisons",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="JSON list: name, prompt, max_tokens (optional)",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory; existing directories are rejected",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument(
        "--cache-mb",
        type=int,
        help="Expert cache in MiB (default: derived by the runner's machine profile)",
    )
    parser.add_argument(
        "--combined-budget-mb",
        type=int,
        help="Process cap in MiB (default: derived by the runner's machine profile)",
    )
    parser.add_argument(
        "--verify",
        choices=("full", "stamp"),
        default="stamp",
        help="stamp hashes the artifact once per host instead of in every sample",
    )
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--variant-env",
        action="append",
        default=[],
        help="NAME:KEY=VALUE, e.g. serial:ANTFLY_QWEN4EXP_DISABLE_IO_OVERLAP=1",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 2:
        parser.error("rounds must be positive and max-tokens at least two")
    binaries = dict(item.split("=", 1) for item in args.binary)
    variants = {}
    for item in args.variant_env:
        name, setting = item.split(":", 1)
        key, value = setting.split("=", 1)
        if name not in binaries or not key.startswith("ANTFLY_QWEN4EXP_"):
            parser.error(
                "variant environment must target a named binary and a Qwen4Exp control"
            )
        variants.setdefault(name, {})[key] = value
    cases = json.loads(args.cases.read_text())
    if not cases or len({c["name"] for c in cases}) != len(cases):
        parser.error("cases must be nonempty and have unique names")
    args.out.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": 1,
        "machine": platform.platform(),
        "hardware": command("sysctl", "hw.memsize", "machdep.cpu.brand_string"),
        "git": command("git", "rev-parse", "HEAD"),
        "dirty": command("git", "status", "--short"),
        "binaries": {
            k: {"path": str(Path(v).resolve()), "sha256": digest(v)}
            for k, v in binaries.items()
        },
        "model": str(args.model.resolve()),
        "first_shard_sha256": digest(args.model),
        "artifact_pin": json.loads(
            Path(__file__).with_name("artifact.json").read_text()
        ),
        "cases_sha256": digest(args.cases),
        "settings": vars(args),
        "observer_source_sha256": {
            str(p): digest(p) for p in Path(__file__).parent.glob("*.py")
        },
        "cache_state": "fresh process and empty expert slots; OS cache not purged",
        "timing": "decode_steps excludes first token computed during prefill; loading is separate",
        "samples": [],
    }

    def save():
        (args.out / "report.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n"
        )

    save()
    for round_no in range(args.rounds):
        order = list(binaries)
        if round_no % 2:
            order.reverse()
        for case_no, case in enumerate(cases):
            for name in order:
                prefix = (
                    args.out / f"r{round_no}-c{case_no}-{list(binaries).index(name)}"
                )
                receipt = prefix.with_suffix(".json")
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith("ANTFLY_QWEN4EXP_")
                }
                if args.profile:
                    env["ANTFLY_QWEN4EXP_PROFILE"] = "1"
                env.update(variants.get(name, {}))
                argv = [
                    str(Path(binaries[name]).resolve()),
                    str(args.model.resolve()),
                    case["prompt"],
                    "--experimental-qwen4exp",
                    "--backend",
                    "metal",
                    "--speculation-policy",
                    "off",
                    "--disable-thinking",
                    "--context-size",
                    str(args.context),
                    "--verify",
                    args.verify,
                    "--max-tokens",
                    str(case.get("max_tokens", args.max_tokens)),
                    "--json-timing",
                    str(receipt.resolve()),
                    "--print-token-ids",
                ]
                # Unset budgets are derived by the runner and read back from
                # the receipt (process_cap_bytes, expert_cache_bytes).
                if args.cache_mb is not None:
                    argv += ["--moe-cache-mb", str(args.cache_mb)]
                if args.combined_budget_mb is not None:
                    argv += ["--combined-budget-mb", str(args.combined_budget_mb)]
                before = host()
                started = time.monotonic()
                sample = {
                    "variant": name,
                    "case": case["name"],
                    "round": round_no,
                    "argv": argv,
                    "environment": {
                        k: v for k, v in env.items() if k.startswith("ANTFLY_QWEN4EXP_")
                    },
                    "before": before,
                }
                with prefix.with_suffix(".log").open("w") as log:
                    try:
                        result = subprocess.run(
                            argv,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                            check=False,
                        )
                        sample["returncode"] = result.returncode
                    except subprocess.TimeoutExpired:
                        sample["error"] = "timeout; subprocess killed and joined"
                sample.update(wall_seconds=time.monotonic() - started, after=host())
                swap_before, swap_after = (
                    swap_events(before),
                    swap_events(sample["after"]),
                )
                sample["swap_event_deltas"] = {
                    k: swap_after[k] - v
                    for k, v in swap_before.items()
                    if k in swap_after
                }
                sample["clean_swap_observation"] = len(
                    sample["swap_event_deltas"]
                ) == 2 and all(v == 0 for v in sample["swap_event_deltas"].values())
                if sample.get("returncode") == 0 and receipt.exists():
                    data = json.loads(receipt.read_text())
                    sample["receipt"] = data
                    sample["decode_tps"] = (
                        data["decode_steps"] * 1000 / data["decode_ms"]
                        if data["decode_steps"]
                        else None
                    )
                    sample["prefill_tps"] = (
                        data["prompt_tokens"] * 1000 / data["prefill_ms"]
                    )
                    sample["output_sha256"] = hashlib.sha256(
                        json.dumps(data["generated_token_ids"]).encode()
                    ).hexdigest()
                    times = data.get("decode_token_ms", [])
                    if times:
                        sample["token_p50_ms"] = percentile(times, 0.5)
                        sample["token_p95_ms"] = percentile(times, 0.95)
                provenance["samples"].append(sample)
                save()
                print(
                    json.dumps(
                        {
                            k: sample.get(k)
                            for k in (
                                "variant",
                                "case",
                                "round",
                                "returncode",
                                "decode_tps",
                                "prefill_tps",
                                "error",
                            )
                        }
                    ),
                    flush=True,
                )
                if "receipt" not in sample:
                    raise SystemExit(f"Failed sample retained at {prefix}; stopping")
    provenance["summary"] = []
    for case in cases:
        for name in binaries:
            samples = [
                s
                for s in provenance["samples"]
                if s["variant"] == name and s["case"] == case["name"]
            ]
            rates = [s["decode_tps"] for s in samples if s["decode_tps"] is not None]
            provenance["summary"].append(
                {
                    "variant": name,
                    "case": case["name"],
                    "median_decode_tps": statistics.median(rates) if rates else None,
                    "median_prefill_tps": statistics.median(
                        s["prefill_tps"] for s in samples
                    ),
                    "output_hashes": sorted({s["output_sha256"] for s in samples}),
                    "all_clean_swap_observations": all(
                        s["clean_swap_observation"] for s in samples
                    ),
                }
            )
    save()
    print(json.dumps(provenance["summary"], indent=2))


if __name__ == "__main__":
    main()
