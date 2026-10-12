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

"""Interactive terminal client for bounded, persistent local Qwen JSONL sessions."""

import argparse
import json
import os
import selectors
import subprocess
import time
from pathlib import Path


class Session:
    def __init__(self, command, stderr=None, env=None):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
        )
        self.buffer = b""
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.read_event(time.monotonic() + 600)
            if self.ready.get("event") != "ready":
                raise RuntimeError(self.ready)
        except BaseException:
            self.close()
            raise

    def read_event(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("session response")
            data = os.read(self.process.stdout.fileno(), 8192)
            if not data:
                raise RuntimeError(f"session exited: {self.process.poll()}")
            self.buffer += data
            if len(self.buffer) > 16 * 1024 * 1024:
                raise RuntimeError("oversized session event")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def request(self, request, on_text=None):
        started = time.monotonic()
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()
        events = []
        while True:
            event = self.read_event(started + 2400)
            event["observed_seconds"] = time.monotonic() - started
            events.append(event)
            if event["event"] == "delta" and on_text:
                on_text(event["text"])
            if event["event"] in {"done", "error", "reset"}:
                return events

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b'{"quit":true}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        finally:
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except BrokenPipeError:
                    pass


def process_budget(family, requested=None):
    budget = (
        requested if requested is not None else (9216 if family == "flash" else 4096)
    )
    maximum = 10240 if family == "flash" else 4096
    if not 1 <= budget <= maximum:
        raise ValueError(f"{family} process budget must be 1..{maximum} MiB")
    return budget


def command(
    binary, model, family, max_tokens=64, cache_mb=1536, raw=False, combined_mb=None
):
    result = [
        str(Path(binary).resolve()),
        str(model),
        "-",
        "--experimental-qwen35-streamed"
        if family == "27b"
        else "--experimental-qwen4exp",
        "--backend",
        "metal",
        "--speculation-policy",
        "off",
        "--session",
        "--disable-thinking",
        "--max-tokens",
        str(max_tokens),
    ]
    result += ["--combined-budget-mb", str(process_budget(family, combined_mb))]
    if family == "flash":
        result += ["--moe-cache-mb", str(cache_mb)]
    if raw:
        result += ["--raw-prompt"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", choices=("27b", "flash"), required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--cache-mb", type=int, default=1536)
    p.add_argument(
        "--combined-budget-mb",
        type=int,
        help="Process cap in MiB (Flash default 9216, maximum 10240; 27B maximum 4096)",
    )
    a = p.parse_args()
    try:
        budget = process_budget(a.family, a.combined_budget_mb)
    except ValueError as error:
        p.error(str(error))
    session = Session(
        command(
            a.binary, a.model, a.family, a.max_tokens, a.cache_mb, combined_mb=budget
        )
    )
    print("Ready. /reset starts a new conversation; /quit exits.")
    try:
        while True:
            prompt = input("You: ")
            if prompt == "/quit":
                break
            request = {"reset": True} if prompt == "/reset" else {"prompt": prompt}
            events = session.request(
                request, lambda text: print(text, end="", flush=True)
            )
            print()
            last = events[-1]
            if last["event"] == "done":
                print(
                    f"[reused {last['reused_tokens']} tokens; prefilled {last['prefilled_tokens']}; {last['prefill_ms'] / 1000:.2f}s prefill]"
                )
            else:
                print(last)
            if last.get("fatal"):
                break
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        session.close()


if __name__ == "__main__":
    main()
