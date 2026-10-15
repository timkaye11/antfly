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

"""Serial AB/BA HTTP qualification for Qwen3-VL Metal residency.

Run under scripts/benchmark_resources.py so server workers and client share
the monitored process group. The baseline and candidate use identical inputs
and environment; only their executable differs. For a same-binary rollback
comparison, --baseline-rollback disables the three residency improvements.
Evidence is preserved on failure. This is a scoped local gate, not a claim
about untested models, deployment hardware, or sustained production traffic.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from benchmark_qwen3vl_ocr_endpoint import load_cases
from qualify_qwen3vl_metal import (
    sha256_file,
    validate_managed_bundle,
    write_json_atomic,
)

MODEL = "Qwen/Qwen3-VL-2B-Instruct-GGUF:q4-k-m-bundle-v1"
ROLLBACK = {
    "TERMITE_QWEN3VL_DISABLE_RESIDENT_FEATURES": "1",
    "TERMITE_METAL_DISABLE_RESIDENT_PREFILL_KV_SHADOW": "1",
    "TERMITE_METAL_DISABLE_ACTIVE_PAGED_SLOT_ATTENTION_DEFERRED": "1",
}


def canonical(response: dict, route: str) -> dict:
    if response.get("model") != MODEL:
        raise ValueError("response model mismatch")
    if route == "read":
        data = response["data"]
        if not data or any(not item.get("text") for item in data):
            raise ValueError("empty reader response")
        return {"data": data, "usage": response["usage"]}
    choices = response["choices"]
    if not choices or any(c["finish_reason"] != "stop" for c in choices):
        raise ValueError("generation did not finish naturally")
    return {"choices": choices, "usage": response["usage"]}


def request(port: int, route: str, body: dict) -> tuple[float, dict]:
    encoded = json.dumps({"model": MODEL, **body}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/ai/v1/{route}",
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = canonical(json.load(response), route)
    except urllib.error.HTTPError as exc:
        # Preserve status/type for admission probes while retaining the server's
        # reason in failure evidence instead of only "Payload Too Large".
        exc.msg = exc.read(4096).decode(errors="replace")
        raise
    if (
        route == "read"
        and len(body.get("images", [])) == 1
        and result["usage"]["completion_tokens"] >= body.get("max_tokens", 512)
    ):
        raise ValueError("reader exhausted its token bound; completion is unproven")
    return (time.monotonic() - started) * 1000, result


def cases(fixture: Path) -> list[dict]:
    loaded = load_cases(fixture)
    selected = [
        dict(
            id=c["id"],
            route="read",
            body=c["request"],
            expected_text=c["fixture"]["source_text"],
        )
        for c in loaded
        if c["id"] in {"sparse", "portrait", "landscape_table", "screenshot"}
    ]
    images = [selected[0]["body"]["images"][0], selected[1]["body"]["images"][0]]
    content = [{"type": "text", "text": "Describe each image briefly, in order."}]
    content += [{"type": "image_url", "image_url": item} for item in images]
    selected.append(
        dict(
            id="two_images",
            route="generate",
            body={
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 256,
                "temperature": 0,
            },
        )
    )
    selected.append(
        dict(
            id="text_only",
            route="generate",
            body={
                "messages": [{"role": "user", "content": "Reply with exactly: Ready"}],
                "max_tokens": 32,
                "temperature": 0,
            },
        )
    )
    return selected


def check_ocr_text(case: dict, response: dict) -> None:
    if "expected_text" in case:
        data = response["data"]
        if len(data) != 1 or data[0]["text"].split() != case["expected_text"].split():
            raise ValueError(f"OCR content differs from fixture: {case['id']}")


def check_latency(latency: dict, max_regression_percent: float) -> None:
    for name, result in latency.items():
        medians = result["median_ms"]
        if medians["candidate"] > medians["baseline"] * (
            1 + max_regression_percent / 100
        ):
            raise ValueError(
                f"latency regression exceeds {max_regression_percent}%: {name}"
            )


def phases(log: str) -> list[dict]:
    records = []
    for line in log.splitlines():
        if "qwen3vl_metal_phase:" in line:
            values = dict(re.findall(r"(\w+)=([\w.]+)", line))
            records.append(
                {k: (v if k == "phase" else int(v)) for k, v in values.items()}
            )
    return records


def check_residency(records: list[dict], *, prefill_logit_rows: int = 1) -> None:
    if prefill_logit_rows < 1:
        raise ValueError("invalid prefill logit row count")
    for phase in ("multimodal_prepare", "prefill", "decode"):
        matching = [r for r in records if r["phase"] == phase]
        if not matching:
            raise ValueError(f"missing {phase} residency counters")
        # Image prefill reads only the final vocabulary row. The existing
        # text-only path reads all prompt logit rows in one tensor; bound that
        # explicitly using the pinned 2B vocabulary and measured prompt size.
        # Both paths permit only ONE access, never per-layer K/V readbacks.
        limit = 151936 * 4 * prefill_logit_rows if phase == "prefill" else 0
        access_limit = 1 if phase == "prefill" else 0
        if any(
            r["tensor_download_bytes"] > limit
            or r["tensor_device_readbacks"] > access_limit
            for r in matching
        ):
            raise ValueError(f"unexpected tensor readback in {phase}")


def check_decode_frames(records: list[dict], completion_tokens: int) -> None:
    decode = [r for r in records if r["phase"] == "decode"]
    if not decode or sum(r["frame_submits"] for r in decode) > completion_tokens + 2:
        raise ValueError("decode exceeded one frame per token plus completion overhead")


def wait_ready(process: subprocess.Popen, port: int) -> None:
    until = time.monotonic() + 60
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError("server exited before readiness")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/readyz", timeout=1
            ) as r:
                if r.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise TimeoutError("server readiness deadline")


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def parse_stream(lines, expected_text: str) -> dict:
    parts, finish, done = [], None, False
    received = 0
    for raw in lines:
        received += len(raw)
        if received > 1024 * 1024:
            raise ValueError("stream exceeded bounded response size")
        line = raw.decode("utf-8").strip()
        if line == "event: error":
            raise ValueError("server reported a stream error")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise ValueError("server reported a stream error")
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("unexpected stream choice")
            parts.append(choice.get("delta", {}).get("content") or "")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done or finish != "stop" or "".join(parts) != expected_text:
        raise ValueError("stream was incomplete or differed from non-streaming output")
    return {"complete": True, "exact_text": True, "chunks": len(parts)}


def stream_check(port: int, case: dict, expected: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/ai/v1/generate",
        data=json.dumps({"model": MODEL, **case["body"], "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return parse_stream(response, expected["choices"][0]["message"]["content"])


def recovery_checks(port: int, case: dict, expected: dict) -> dict:
    # Admission must reject malformed media and leave the model usable.
    try:
        request(port, "read", {"images": [{"url": "data:image/png;base64,AAAA"}]})
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 422):
            raise
        malformed = exc.code
    else:
        raise ValueError("malformed image was accepted")

    def read_one(_):
        try:
            return request(port, case["route"], case["body"])[1]
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503):
                raise
            return {"busy_status": exc.code}

    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(read_one, range(2)))
    if not any(value == expected for value in concurrent):
        raise ValueError("neither concurrent request completed")
    if any(value != expected and "busy_status" not in value for value in concurrent):
        raise ValueError("concurrent output mismatch")

    # Disconnect after sending a valid request, then check recovery. This
    # proves recovery, not the exact point at which cancellation was observed.
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "POST",
        "/ai/v1/read",
        json.dumps({"model": MODEL, **case["body"]}),
        {"Content-Type": "application/json"},
    )
    time.sleep(0.1)
    connection.close()
    for attempt in range(20):
        value = read_one(None)
        if value == expected:
            return {
                "malformed_status": malformed,
                "concurrent": concurrent,
                "post_disconnect_recovery": True,
            }
        time.sleep(0.25)
    raise ValueError("server did not recover after client disconnect")


def run(args) -> dict:
    selected = cases(args.fixture)
    if args.case_names:
        selected = [case for case in selected if case["id"] in args.case_names]
    report = {
        "schema": "antfly.qwen3vl.residency_qualification.v1",
        "pass": False,
        "model": validate_managed_bundle(args.model_dir),
        "runs": [],
        "outputs": {},
        "fixture_sha256": sha256_file(args.fixture),
        "harness_sha256": sha256_file(Path(__file__)),
        "cases": [case["id"] for case in selected],
        "binaries": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (
                ("baseline", args.baseline),
                ("candidate", args.candidate),
            )
        },
        "rounds": args.rounds,
        "iterations": args.iters,
        "max_regression_percent": args.max_regression_percent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for round_index in range(args.rounds):
            for label in (
                ("baseline", "candidate")
                if round_index % 2 == 0
                else ("candidate", "baseline")
            ):
                binary = args.baseline if label == "baseline" else args.candidate
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith(("TERMITE_", "ANTFLY_INFERENCE_"))
                }
                flags = {
                    "TERMITE_QWEN3VL_PROFILE": "1",
                    "TERMITE_SERVER_GENERATE_TIMING": "1",
                    "ANTFLY_INFERENCE_REQUIRED_BACKEND": "metal",
                }
                if label == "baseline" and args.baseline_rollback:
                    flags.update(ROLLBACK)
                env.update(flags)
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
                # This setting is weighted admission capacity, not the number
                # of simultaneously executing models. Four units can exhaust
                # the pre-decode transport reservation before image admission.
                command = [
                    str(binary.resolve()),
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
                    "2048",
                    "--kernel-jit-mode",
                    "off",
                ]
                row["command"] = command
                if not port_is_free(args.port):
                    raise RuntimeError("qualification port is already in use")
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        wait_ready(process, args.port)
                        row["warmup_ms"], _ = request(
                            args.port, selected[0]["route"], selected[0]["body"]
                        )
                        for iteration in range(args.iters):
                            for case in selected:
                                log_offset = log_path.stat().st_size
                                elapsed, response = request(
                                    args.port, case["route"], case["body"]
                                )
                                check_ocr_text(case, response)
                                request_phases = phases(
                                    log_path.read_bytes()[log_offset:].decode(
                                        errors="replace"
                                    )
                                )
                                expected = report["outputs"].setdefault(
                                    case["id"], response
                                )
                                if response != expected:
                                    raise ValueError(
                                        f"output/token mismatch: {label}/{case['id']}"
                                    )
                                row["samples"].append(
                                    {
                                        "case": case["id"],
                                        "iteration": iteration,
                                        "ms": elapsed,
                                        "completion_tokens": response["usage"][
                                            "completion_tokens"
                                        ],
                                        "phases": request_phases,
                                    }
                                )
                                if label == "candidate":
                                    check_decode_frames(
                                        request_phases,
                                        response["usage"]["completion_tokens"],
                                    )
                                    check_residency(
                                        request_phases,
                                        prefill_logit_rows=(
                                            response["usage"]["prompt_tokens"]
                                            if case["id"] == "text_only"
                                            else 1
                                        ),
                                    )
                                write_json_atomic(args.output, report)
                                print(
                                    f"{round_index} {label} {case['id']} {elapsed:.1f} ms",
                                    flush=True,
                                )
                        if label == "candidate":
                            # Backend counters are shared: qualify only serial
                            # samples, before the deliberate concurrency probe.
                            reader = next(
                                (case for case in selected if case["route"] == "read"),
                                None,
                            )
                            if reader:
                                row["recovery"] = recovery_checks(
                                    args.port, reader, report["outputs"][reader["id"]]
                                )
                            row["streaming"] = {
                                case["id"]: stream_check(
                                    args.port, case, report["outputs"][case["id"]]
                                )
                                for case in selected
                                if case["route"] == "generate"
                            }
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        until = time.monotonic() + 5
                        while not port_is_free(args.port) and time.monotonic() < until:
                            time.sleep(0.1)
                        if not port_is_free(args.port):
                            raise RuntimeError("server worker did not shut down")
                content = log_path.read_text()
                if re.search(
                    r"panic:|segmentation fault|Metal.*command.*error", content, re.I
                ):
                    raise ValueError("server log contains a runtime failure")
                row["phases"] = phases(content)
        report["latency"] = {}
        for case in selected:
            medians = {
                label: statistics.median(
                    s["ms"]
                    for r in report["runs"]
                    if r["profile"] == label
                    for s in r["samples"]
                    if s["case"] == case["id"]
                )
                for label in ("baseline", "candidate")
            }
            report["latency"][case["id"]] = {
                "median_ms": medians,
                "speedup": medians["baseline"] / medians["candidate"],
            }
        check_latency(report["latency"], args.max_regression_percent)
        report["pass"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write_json_atomic(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-rollback", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).parent / "fixtures/ocr/fixture.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-regression-percent", type=float, default=5)
    parser.add_argument(
        "--case",
        dest="case_names",
        action="append",
        choices=(
            "sparse",
            "portrait",
            "landscape_table",
            "screenshot",
            "two_images",
            "text_only",
        ),
        help="restrict a diagnostic probe; omit for the full qualification matrix",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.rounds <= 10
        or not 1 <= args.iters <= 100
        or not 1024 <= args.port <= 65535
    ):
        parser.error("invalid run bounds")
    if not 0 <= args.max_regression_percent <= 100:
        parser.error("invalid regression tolerance")
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
