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

"""Serial, supervised CLI cancellation and session transport qualification.

Never mutates or disconnects model files. Run separately from builds/inference.
"""

import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path

from chat import Session, command
from evidence import swap_snapshot


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def wait_marker(process, path, marker, timeout=600):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if marker in path.read_text(errors="replace"):
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"process exited before {marker}: {path.read_text()[-2000:]}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"waiting for {marker}")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def write_with_deadline(process, payload, deadline):
    """A non-reading child must not trap the supervisor in a pipe write."""
    fd = process.stdin.fileno()
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            offset = 0
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("session request write")
                try:
                    offset += os.write(fd, payload[offset:])
                except BlockingIOError:
                    continue
                except BrokenPipeError:
                    return
    finally:
        os.set_blocking(fd, blocking)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("27b", "flash"), required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--phases",
        nargs="+",
        choices=(
            "verification",
            "prefill",
            "decode",
            "idle",
            "oversized",
            "broken-pipe",
            "stream-broken-pipe",
            "receipt-write",
        ),
        default=[
            "verification",
            "prefill",
            "decode",
            "idle",
            "oversized",
            "broken-pipe",
            "stream-broken-pipe",
            "receipt-write",
        ],
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTFLY_QWEN")}
    report = {
        "passed": False,
        "family": a.family,
        "binary_sha256": hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        "cases": [],
        "swap_before": swap_snapshot(),
    }
    target = a.output / "summary.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    for phase in a.phases:
        row = {"phase": phase, "passed": False}
        started = time.monotonic()
        try:
            cmd = command(a.binary, a.model, a.family, 32, 1536, True)
            row["command"] = cmd.copy()
            stderr = a.output / f"{phase}.stderr"
            with stderr.open("w") as err:
                if phase == "receipt-write":
                    cmd.remove("--session")
                    cmd[2] = "The capital of France is"
                    cmd[cmd.index("--max-tokens") + 1] = "1"
                    # A directory is never a valid timing file. This leaves all
                    # installed files unchanged and exercises the real writer.
                    cmd += ["--json-timing", str(a.output)]
                    row["command"] = cmd
                    with (a.output / f"{phase}.stdout").open("w") as out:
                        process = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
                        try:
                            row["exit_code"] = process.wait(timeout=600)
                            require(
                                row["exit_code"] > 0, "receipt write error was ignored"
                            )
                            require(
                                (a.output / f"{phase}.stdout").read_text().strip(),
                                "failed before completing generation",
                            )
                            require(
                                "receipt" in stderr.read_text().lower(),
                                "missing receipt write diagnostic",
                            )
                        finally:
                            stop(process)
                elif phase == "stream-broken-pipe":
                    cmd.remove("--session")
                    cmd[2] = "The capital of France is"
                    receipt = a.output / f"{phase}.json"
                    cmd += ["--stream", "--json-timing", str(receipt)]
                    row["command"] = cmd
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=err, env=env
                    )
                    process.stdout.close()
                    try:
                        row["exit_code"] = process.wait(timeout=600)
                        row["receipt"] = json.loads(receipt.read_text())
                        require(
                            row["exit_code"] > 0,
                            "unhandled streaming write failure",
                        )
                        require(
                            row["receipt"].get("finish_reason") == "OutputWriteFailed"
                            and row["receipt"].get("failure_stage") == "decode",
                            "streaming failure left an invalid success receipt",
                        )
                    finally:
                        stop(process)
                elif phase in ("verification", "prefill", "decode"):
                    cmd.remove("--session")
                    cmd[2] = "The capital of France is"
                    receipt = a.output / f"{phase}.json"
                    cmd += ["--ignore-eos", "--json-timing", str(receipt)]
                    row["command"] = cmd
                    with (a.output / f"{phase}.stdout").open("w") as out:
                        process = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
                        try:
                            marker = {
                                "verification": "verifying",
                                "prefill": "prefilling",
                                "decode": "decoding",
                            }[phase]
                            wait_marker(process, stderr, marker)
                            time.sleep(0.15)
                            tick = time.monotonic()
                            process.send_signal(signal.SIGINT)
                            code = process.wait(timeout=15)
                            row["cancellation_seconds"] = time.monotonic() - tick
                            row["exit_code"] = code
                            row["receipt"] = json.loads(receipt.read_text())
                            require(
                                code > 0,
                                "did not cooperatively fail",
                            )
                            require(
                                row["receipt"]["finish_reason"] == "Cancelled",
                                "wrong cancellation reason",
                            )
                            require(
                                row["receipt"]["failure_stage"]
                                == (
                                    "artifact_verification"
                                    if phase == "verification"
                                    else phase
                                ),
                                "wrong failure stage",
                            )
                        finally:
                            stop(process)
                else:
                    session = Session(cmd, stderr=err, env=env)
                    try:
                        require(
                            session.ready.get("hash_verified") is True,
                            "artifact not verified",
                        )
                        tick = time.monotonic()
                        deadline = tick + 15
                        if phase == "idle":
                            session.process.send_signal(signal.SIGTERM)
                        elif phase == "oversized":
                            write_with_deadline(
                                session.process,
                                (
                                    json.dumps({"prompt": "x" * (1024 * 1024 + 1)})
                                    + "\n"
                                ).encode(),
                                deadline,
                            )
                        else:
                            # Closing every read endpoint must produce a handled write
                            # failure, not SIGPIPE termination or a stuck GPU owner.
                            session.selector.unregister(session.process.stdout)
                            session.process.stdout.close()
                            write_with_deadline(
                                session.process, b'{"reset":true}\n', deadline
                            )
                        if phase != "broken-pipe":
                            row["event"] = session.read_event(deadline)
                            require(
                                row["event"].get("fatal") is True,
                                "expected fatal error event",
                            )
                            require(
                                row["event"]["error_name"]
                                == (
                                    "Cancelled"
                                    if phase == "idle"
                                    else "SessionRequestTooLarge"
                                ),
                                "wrong protocol error",
                            )
                        row["exit_code"] = session.process.wait(
                            timeout=max(0, deadline - time.monotonic())
                        )
                        row["exit_seconds"] = time.monotonic() - tick
                        require(row["exit_seconds"] <= 15, "session exit deadline")
                        require(
                            row["exit_code"] > 0,
                            "unhandled termination",
                        )
                        if phase == "broken-pipe":
                            require(
                                "OutputWriteFailed" in stderr.read_text(),
                                "missing broken-pipe diagnostic",
                            )
                    finally:
                        # On a timed-out write the input pipe can still be full.
                        # Reap the child before close() attempts its quit message.
                        stop(session.process)
                        session.close()
            row["passed"] = True
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["seconds"] = time.monotonic() - started
            report["cases"].append(row)
            report["swap_after"] = swap_snapshot()
            report["swap_delta"] = {
                k: report["swap_after"][k] - report["swap_before"][k]
                for k in report["swap_before"]
            }
            target.write_text(json.dumps(report, indent=2) + "\n")
            print(a.family, phase, row, flush=True)
    # Cancellation/transport correctness is distinct from performance qualification.
    report["passed"] = all(row["passed"] for row in report["cases"])
    report["clean_swap"] = all(value == 0 for value in report["swap_delta"].values())
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
