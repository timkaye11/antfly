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

"""Fail-closed evidence checks for local, non-speculative Qwen benchmarks.

Passing these checks establishes usable timing evidence, not model quality,
physical disk bandwidth, independent numerical parity, or production readiness.
"""

import json
import math
import re
import subprocess

from machine_profile import MAX_CONTEXT, MIB, VERIFY_MODES


def read_receipt(path):
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def swap_snapshot():
    return parse_swap_snapshot(subprocess.check_output(["vm_stat"], text=True))


def parse_swap_snapshot(text):
    result = {}
    for name in ("Swapins", "Swapouts"):
        match = re.search(rf"^{name}:\s+(\d+)\.\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing vm_stat counter: {name}")
        result[name.lower()] = int(match[1])
    return result


def clean_swap_delta(before, after):
    delta = {key: after[key] - before[key] for key in ("swapins", "swapouts")}
    if any(value != 0 for value in delta.values()):
        raise ValueError(f"system swap counters changed: {delta}")
    return delta


def validate_receipt(
    receipt,
    family,
    budget_mb,
    token_limit,
    context=MAX_CONTEXT,
    verify="full",
    cache_mb=None,
):
    """Check one receipt against what the harness asked for.

    `budget_mb` and `cache_mb` are the explicit overrides passed to the runner,
    or None when it derived them; the admitted values are then read from the
    receipt. `verify` is the weakest artifact verification the caller accepts.
    """

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def integer(value):
        return type(value) is int and value >= 0

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    require(isinstance(receipt, dict), "receipt must be an object")
    require(
        receipt.get("architecture") == {"27b": "qwen35", "flash": "qwen4exp"}[family],
        "wrong receipt architecture",
    )
    require(receipt.get("backend") == "metal", "wrong receipt backend")
    reason = receipt.get("finish_reason")
    require(reason in ("stop", "length"), "generation did not finish successfully")
    require(
        receipt.get("failure_stage") is None, "receipt contains an execution failure"
    )
    require(verify in VERIFY_MODES, "unknown verification mode")
    require(receipt.get("context_size") == context, "unexpected context size")
    prompt, generated = (
        receipt.get("prompt_token_ids"),
        receipt.get("generated_token_ids"),
    )
    for name, ids in (("prompt", prompt), ("generated", generated)):
        require(isinstance(ids, list) and len(ids) > 0, f"missing {name} token IDs")
        require(
            all(integer(x) and x < 248320 for x in ids), f"invalid {name} token IDs"
        )
    require(
        2 <= len(generated) <= token_limit,
        "need at least two generated tokens to benchmark decode",
    )
    require(
        len(prompt) + token_limit <= context, "prompt exceeds requested context budget"
    )
    require(
        type(receipt.get("generated_tokens")) is int
        and receipt["generated_tokens"] == len(generated),
        "generated token count mismatch",
    )
    require(
        reason != "length" or len(generated) == token_limit,
        "short output with length finish reason",
    )
    # The first output is predicted by prefill. EOS detection advances once more
    # than a length-limited run, without adding EOS to generated_token_ids.
    steps = len(generated) - (reason == "length")
    require(
        type(receipt.get("decode_steps")) is int and receipt["decode_steps"] == steps,
        "decode step count mismatch",
    )
    times = receipt.get("decode_token_ms")
    require(
        isinstance(times, list)
        and len(times) == steps
        and all(positive(x) for x in times),
        "invalid decode step timings",
    )
    for key in ("load_ms", "prefill_ms", "decode_ms", "first_token_ready_ms"):
        require(positive(receipt.get(key)), f"missing or invalid {key}")
    require(
        sum(times) <= receipt["decode_ms"] + 1, "decode steps exceed total decode time"
    )
    require(
        abs(
            receipt["first_token_ready_ms"] - receipt["load_ms"] - receipt["prefill_ms"]
        )
        < 1,
        "first-token timing mismatch",
    )
    # Receipts name the cap the runner actually enforced. An explicit request
    # must come back unchanged; a derived one must say that it was derived.
    reported = receipt.get("process_cap_bytes", receipt.get("cap_bytes"))
    if budget_mb is None:
        require(
            integer(reported) and reported > 0, "missing derived process cap in receipt"
        )
        require(
            receipt.get("process_cap_source") == "derived",
            "process cap was not derived by the runner",
        )
        cap = reported
    else:
        cap = budget_mb * MIB
        require(
            reported is None or reported == cap,
            "runner enforced a different process cap than requested",
        )
        require(
            receipt.get("process_cap_source") in (None, "explicit"),
            "explicit process cap reported as derived",
        )
    admitted = receipt.get("admitted_bytes")
    require(integer(admitted) and 0 < admitted <= cap, "invalid memory admission")
    memory = receipt.get("process_memory")
    require(
        isinstance(memory, dict) and memory.get("available") is True,
        "process memory measurement unavailable",
    )
    peak = memory.get("peak_footprint_bytes")
    require(
        integer(peak) and 0 < peak <= cap,
        "missing peak footprint or memory cap exceeded",
    )
    hashed = receipt.get("hash_verified") is True
    stamped = receipt.get("stamp_verified") is True
    require(not (hashed and stamped), "ambiguous artifact verification")
    if verify == "full":
        require(hashed, "artifact hash not verified")
        require(
            receipt.get("verification_mode") in (None, "full"),
            "artifact verified in a weaker mode than required",
        )
    else:
        # A stamp-mode run still hashes when no valid stamp exists, and says so.
        require(hashed or stamped, "artifact neither hashed nor stamp verified")
        require(
            hashed or receipt.get("verification_mode") == "stamp",
            "stamp accepted outside stamp mode",
        )
    cache_bytes = None
    if family == "27b":
        require(receipt.get("prompt_lookup") == 0, "unexpected speculative decoding")
        require(
            receipt.get("position") == len(prompt) + steps, "decoder position mismatch"
        )
    else:
        require(
            receipt.get("prompt_tokens") == len(prompt), "prompt token count mismatch"
        )
        require(
            receipt.get("sidecars_hash_verified") is True,
            "Flash tokenizer and generation sidecars not verified",
        )
        # The cache is reported as allocated, never as requested.
        cache_bytes = receipt.get("expert_cache_bytes")
        if cache_bytes is not None or cache_mb is not None:
            counts, strides = (
                receipt.get("expert_slot_counts"),
                receipt.get("expert_strides"),
            )
            require(
                isinstance(counts, list)
                and isinstance(strides, list)
                and len(counts) == len(strides) == 48
                and all(integer(x) and x >= 10 for x in counts)
                and all(integer(x) and x > 0 for x in strides),
                "invalid expert slot inventory",
            )
            require(
                integer(cache_bytes)
                and cache_bytes == sum(a * b for a, b in zip(counts, strides)),
                "expert cache bytes do not match the slot inventory",
            )
            require(cache_bytes < admitted, "expert cache exceeds admitted memory")
            if cache_mb is None:
                require(
                    receipt.get("expert_cache_source") == "derived",
                    "expert cache was not derived by the runner",
                )
            else:
                require(
                    receipt.get("expert_cache_source") == "explicit"
                    and cache_bytes <= cache_mb * MIB,
                    "runner allocated a different expert cache than requested",
                )
    return {
        "process_cap_bytes": cap,
        "expert_cache_bytes": cache_bytes,
        "verification": "hash" if hashed else "stamp",
        "effective_decode_tokens_per_second": (len(generated) - 1)
        * 1000
        / receipt["decode_ms"],
        "executed_decode_steps": steps,
        "peak_footprint_bytes": peak,
        "artifact_hash_verified_by_runtime": receipt.get("hash_verified") is True,
    }


def validate_repeat(previous, current):
    for key in ("prompt_token_ids", "generated_token_ids", "finish_reason"):
        if previous[key] != current[key]:
            raise ValueError(f"repeated greedy run changed {key}")
