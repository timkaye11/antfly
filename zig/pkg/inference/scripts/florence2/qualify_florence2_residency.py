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

"""Compare Florence Metal vision optimizations with their individual rollbacks.

Run under scripts/benchmark_resources.py. Checks exact reader output parity,
zero encoder tensor readbacks, malformed-input recovery and concurrent reads.
The same candidate executable is used for both paths, in alternating order.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen3vl"))
import qualify_qwen3vl_residency as shared


def image_url(path):
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return {
        "url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    }


def check_encoder(records):
    if not records:
        raise ValueError("missing Florence encoder counters")
    for record in records:
        if record["tensor_download_bytes"] or record["tensor_device_readbacks"]:
            raise ValueError(f"encoder payload left Metal: {record}")


def parse_encoder(log):
    return [
        {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        for line in log.splitlines()
        if "florence_metal_phase:" in line
    ]


ROLLBACK_FLAGS = {
    "vision-tail": "ANTFLY_INFERENCE_FLORENCE_DISABLE_DEVICE_VISION_TAIL",
    "conv3x3": "TERMITE_FLORENCE2_METAL_DISABLE_CONV3X3",
}


def run(args):
    shared.MODEL = args.model
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "qwen3vl/fixtures/ocr"
    cases = [
        {
            "id": name,
            "body": {
                "images": [image_url(fixtures / f"{name}.png")],
                "prompt": "<OCR>",
                "max_tokens": 512,
            },
        }
        for name in ("sparse", "portrait", "landscape_table")
    ]
    cases.append(
        {
            "id": "batch",
            "body": {
                "images": [
                    image_url(fixtures / "sparse.png"),
                    image_url(fixtures / "portrait.png"),
                ],
                "prompt": "<OCR>",
                "max_tokens": 512,
            },
        }
    )
    if args.photo:
        cases.append(
            {
                "id": "photo",
                "body": {
                    "images": [image_url(args.photo)],
                    "prompt": "<MORE_DETAILED_CAPTION>",
                    "max_tokens": 256,
                },
            }
        )
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
    for case in cases:
        case["route"] = "read"
    report = {
        "schema": "antfly.florence2.residency.v1",
        "pass": False,
        "model": args.model,
        "comparison": args.comparison,
        "recovery_checked": not args.skip_recovery,
        "binary_sha256": shared.sha256_file(args.binary),
        "runs": [],
        "outputs": {},
        "model_files": {
            p.name: shared.sha256_file(p)
            for p in sorted(args.model_dir.iterdir())
            if p.is_file()
        },
        "request_sha256": {
            c["id"]: hashlib.sha256(
                json.dumps(c["body"], sort_keys=True).encode()
            ).hexdigest()
            for c in cases
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for round_index in range(args.rounds):
            for label in (
                ("rollback", "resident")
                if round_index % 2 == 0
                else ("resident", "rollback")
            ):
                if not shared.port_is_free(args.port):
                    raise RuntimeError("qualification port is occupied")
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith(("TERMITE_", "ANTFLY_INFERENCE_"))
                }
                flags = {
                    "ANTFLY_INFERENCE_READ_PROFILE": "1",
                    "ANTFLY_INFERENCE_REQUIRED_BACKEND": "metal",
                    "TERMITE_FLORENCE2_METAL_STRICT_RESIDENT": "1",
                }
                if label == "rollback":
                    flags[ROLLBACK_FLAGS[args.comparison]] = "1"
                if args.comparison == "conv3x3" and label == "resident":
                    flags["TERMITE_FLORENCE2_METAL_REQUIRE_CONV3X3"] = "1"
                env.update(flags)
                command = [
                    str(args.binary.resolve()),
                    "run",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--models-dir",
                    str(args.model_dir.parent.parent.resolve()),
                    "--max-loaded-models",
                    "1",
                    "--max-concurrent-requests",
                    "32",
                    "--process-memory-budget-mb",
                    "8192",
                    "--host-budget-mb",
                    "4096",
                    "--backend-budget-mb",
                    "4096",
                    "--combined-budget-mb",
                    "8192",
                    "--scratch-budget-mb",
                    "4096",
                    "--kernel-jit-mode",
                    "off",
                ]
                log_path = args.output.with_name(
                    f"{args.output.stem}-{round_index}-{label}.log"
                )
                row = {
                    "round": round_index,
                    "profile": label,
                    "environment": flags,
                    "samples": [],
                    "log": str(log_path),
                }
                report["runs"].append(row)
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        shared.wait_ready(process, args.port)
                        for case in cases:
                            shared.request(args.port, "read", case["body"])
                        for iteration in range(args.iters):
                            for case in cases:
                                offset = log_path.stat().st_size
                                elapsed, response = shared.request(
                                    args.port, "read", case["body"]
                                )
                                if len(response["data"]) != len(case["body"]["images"]):
                                    raise ValueError("missing image response")
                                expected = report["outputs"].setdefault(
                                    case["id"], response
                                )
                                if response != expected:
                                    raise ValueError(
                                        f"output/token mismatch: {label}/{case['id']}"
                                    )
                                sample_log = log_path.read_bytes()[offset:].decode(
                                    errors="replace"
                                )
                                counters = parse_encoder(sample_log)
                                row["samples"].append(
                                    {
                                        "case": case["id"],
                                        "iteration": iteration,
                                        "ms": elapsed,
                                        "encoder": counters,
                                        "vision_ms": [
                                            float(v)
                                            for v in re.findall(
                                                r"florence-profile phase=vision_total elapsed_ms=([0-9.]+)",
                                                sample_log,
                                            )
                                        ],
                                        "downsample_ms": [
                                            float(v)
                                            for v in re.findall(
                                                r"florence-profile phase=vision_stage_conv_embed stage=[123] elapsed_ms=([0-9.]+)",
                                                sample_log,
                                            )
                                        ],
                                    }
                                )
                                if label == "resident" or args.comparison == "conv3x3":
                                    check_encoder(counters)
                                shared.write_json_atomic(args.output, report)
                                print(
                                    f"{round_index} {label} {case['id']} {elapsed:.1f} ms",
                                    flush=True,
                                )
                        if label == "resident" and not args.skip_recovery:
                            row["recovery"] = shared.recovery_checks(
                                args.port, cases[0], report["outputs"][cases[0]["id"]]
                            )
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        until = time.monotonic() + 5
                        while (
                            not shared.port_is_free(args.port)
                            and time.monotonic() < until
                        ):
                            time.sleep(0.1)
                        if not shared.port_is_free(args.port):
                            raise RuntimeError("server worker did not shut down")
                if re.search(
                    r"panic:|segmentation fault|Metal.*command.*error",
                    log_path.read_text(),
                    re.I,
                ):
                    raise ValueError("runtime failure in server log")
        report["latency"] = {}
        for case in cases:
            medians = {
                label: statistics.median(
                    s["ms"]
                    for r in report["runs"]
                    if r["profile"] == label
                    for s in r["samples"]
                    if s["case"] == case["id"]
                )
                for label in ("rollback", "resident")
            }
            report["latency"][case["id"]] = {
                "median_ms": medians,
                "speedup": medians["rollback"] / medians["resident"],
            }
            if medians["resident"] > medians["rollback"] * 1.05:
                raise ValueError(f"latency regression exceeds 5%: {case['id']}")
        report["pass"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shared.write_json_atomic(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", default="antflydb/Florence-2-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photo", type=Path)
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--case",
        action="append",
        choices=("sparse", "portrait", "landscape_table", "batch", "photo"),
    )
    parser.add_argument(
        "--comparison",
        choices=tuple(ROLLBACK_FLAGS),
        default="vision-tail",
        help="optimization to compare with its rollback; other optimizations stay enabled",
    )
    parser.add_argument(
        "--skip-recovery", action="store_true", help="diagnostic probes only"
    )
    args = parser.parse_args()
    if (
        not 1 <= args.rounds <= 10
        or not 1 <= args.iters <= 100
        or not 1024 <= args.port <= 65535
    ):
        parser.error("invalid run bounds")
    if args.case and "photo" in args.case and not args.photo:
        parser.error("photo case requires --photo")
    result = run(args)
    print(
        json.dumps(
            {k: result[k] for k in ("pass", "error", "latency") if k in result},
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
