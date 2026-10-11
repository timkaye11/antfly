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

"""Serial local qualification. Explicit opt-in to the long-running 4K run."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np


def compare(a, b):
    a = np.fromfile(a, dtype="<f4")
    b = np.fromfile(b, dtype="<f4")
    if a.shape != b.shape or not np.isfinite(a).all() or (not np.isfinite(b).all()):
        raise AssertionError("shape or finite-value mismatch")
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
    result = {
        "max_abs": float(abs(a - b).max()),
        "relative_l2": rel,
        "argmax_equal": int(a.argmax()) == int(b.argmax()),
    }
    if rel > 0.002 or not result["argmax_equal"]:
        raise AssertionError(result)
    return result


def swapouts():
    text = subprocess.check_output(["vm_stat"], text=True)
    return int(
        next(line for line in text.splitlines() if line.startswith("Swapouts:"))
        .split(":")[1]
        .strip()
        .rstrip(".")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--long", action="store_true")
    p.add_argument(
        "--chunk-parity",
        action="store_true",
        help="compare 512-token prefill at chunk sizes 128 and 512, then decode",
    )
    p.add_argument("--reference", type=Path)
    p.add_argument("--wide-reference", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    base = [
        str(a.binary.resolve()),
        str(a.model),
        "The capital of France is",
        "--experimental-qwen35-streamed",
        "--backend",
        "metal",
        "--speculation-policy",
        "off",
        "--max-tokens",
        "2",
        "--print-token-ids",
    ]
    summary = {}

    def run(
        name,
        prompt=None,
        extra=(),
        env=None,
        cancel=False,
        expected=0,
        expected_reason=None,
    ):
        dest = a.output / name
        dest.mkdir(exist_ok=True)
        receipt_path = dest / "receipt.json"
        receipt_path.unlink(missing_ok=True)
        command = base.copy()
        if prompt is not None:
            command[2] = prompt
        command += ["--json-timing", str(dest / "receipt.json"), *extra]
        variables = os.environ.copy()
        variables.update(env or {})
        variables.pop("ANTFLY_QWEN35_CAPTURE_STATE", None)
        if name in {
            "batched",
            "chunk1",
            "synchronous",
            "wide",
            "chunk128-wide",
            "chunk512-wide",
        }:
            variables["ANTFLY_QWEN35_CAPTURE_DIR"] = str(dest)
        else:
            variables.pop("ANTFLY_QWEN35_CAPTURE_DIR", None)
        before = swapouts()
        start = time.monotonic()
        with (dest / "stdout").open("w") as out, (dest / "stderr").open("w") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, env=variables)
            try:
                sent = False
                while process.poll() is None:
                    time.sleep(0.25)
                    if (
                        cancel
                        and (not sent)
                        and ("prefilling" in (dest / "stderr").read_text())
                    ):
                        time.sleep(1)
                        process.send_signal(signal.SIGINT)
                        sent = True
                    if time.monotonic() - start > 3600:
                        raise TimeoutError(name)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        code = process.returncode
        receipt = (
            json.loads((dest / "receipt.json").read_text())
            if (dest / "receipt.json").exists()
            else {
                "finish_reason": (dest / "stderr").read_text().strip(),
                "process_memory": {},
            }
        )
        if expected == 0 and code != 0 or (expected != 0 and code == 0):
            raise AssertionError((name, code, receipt))
        if cancel and receipt["finish_reason"] != "Cancelled":
            raise AssertionError(receipt)
        if expected_reason and expected_reason not in receipt["finish_reason"]:
            raise AssertionError((name, receipt))
        peak = receipt["process_memory"].get("peak_footprint_bytes")
        if peak is not None and peak > 4 * 1024**3:
            raise AssertionError(("memory cap", peak))
        result = {
            "exit_code": code,
            "seconds": time.monotonic() - start,
            "swapouts_delta": swapouts() - before,
            "peak_footprint_bytes": peak,
            "receipt": str(dest / "receipt.json"),
        }
        summary[name] = result
        (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, json.dumps(result), flush=True)
        return (dest, receipt)

    if a.wide_reference:
        wide, _ = run(
            "wide", prompt=" x" * 128, extra=("--raw-prompt", "--max-tokens", "1")
        )
        summary["wide_reference_logits"] = compare(
            a.wide_reference / "logits-0.f32", wide / "logits-pos-128.f32"
        )
        layer_errors = []
        for layer in range(64):
            name = f"layer-{layer:02}-pos-128.f32"
            layer_errors.append(compare(a.wide_reference / name, wide / name))
        summary["wide_reference_layers"] = {
            "count": 64,
            "max_relative_l2": max(x["relative_l2"] for x in layer_errors),
            "max_abs": max(x["max_abs"] for x in layer_errors),
        }
    batched, _ = run("batched", extra=("--raw-prompt",))
    if a.reference:
        summary["reference_prefill"] = compare(
            a.reference / "logits-0.f32", batched / "logits-pos-5.f32"
        )
        summary["reference_decode"] = compare(
            a.reference / "logits-1.f32", batched / "logits-pos-6.f32"
        )
    single, _ = run("chunk1", extra=("--raw-prompt", "--prefill-chunk-size", "1"))
    summary["chunk_parity"] = compare(
        batched / "logits-pos-5.f32", single / "logits-pos-5.f32"
    )
    summary["chunk_decode_parity"] = compare(
        batched / "logits-pos-6.f32", single / "logits-pos-6.f32"
    )
    sync, _ = run(
        "synchronous",
        extra=("--raw-prompt",),
        env={"ANTFLY_QWEN35_DISABLE_IO_OVERLAP": "1"},
    )
    summary["io_parity"] = compare(
        batched / "logits-pos-6.f32", sync / "logits-pos-6.f32"
    )
    run(
        "chat",
        prompt="Reply with only the word hello.",
        extra=("--stream", "--max-tokens", "8"),
    )
    run("cancel", prompt=" x" * 130, extra=("--raw-prompt",), cancel=True, expected=1)
    run(
        "low-budget",
        extra=("--raw-prompt", "--combined-budget-mb", "2048"),
        expected=1,
        expected_reason="Qwen35MemoryBudgetExceeded",
    )
    run(
        "context-overflow",
        extra=("--raw-prompt", "--context-size", "4097"),
        expected=1,
        expected_reason="UnsupportedQwen35Context",
    )
    if a.chunk_parity:
        small, _ = run(
            "chunk128-wide",
            prompt=" x" * 512,
            extra=("--raw-prompt", "--prefill-chunk-size", "128"),
        )
        large, _ = run(
            "chunk512-wide",
            prompt=" x" * 512,
            extra=("--raw-prompt", "--prefill-chunk-size", "512"),
        )
        summary["wide_chunk_parity"] = {
            str(position): compare(
                small / f"logits-pos-{position}.f32",
                large / f"logits-pos-{position}.f32",
            )
            for position in (512, 513)
        }
    if a.long:
        _, receipt = run(
            "context4096", prompt=" x" * 4094, extra=("--raw-prompt", "--ignore-eos")
        )
        if (
            len(receipt["prompt_token_ids"]) != 4094
            or receipt["position"] != 4095
            or receipt["generated_tokens"] != 2
        ):
            raise AssertionError("did not exercise a 4096-token sequence")
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
