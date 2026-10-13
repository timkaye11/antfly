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

from machine_profile import FLAGS, VERIFY_MODES, budget_arguments, explicit_mb


class Session:
    def __init__(self, command, stderr=None, env=None, detach=False):
        # A detached child does not receive the terminal's Ctrl-C, so the
        # client can turn it into a request cancel instead of losing the model.
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            start_new_session=detach,
        )
        self.buffer = b""
        self.started = None
        self.events = []
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

    def send(self, request):
        self.started = time.monotonic()
        self.events = []
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()

    def cancel(self):
        """Stop the request in flight. Its own `done` event is the reply."""
        self.process.stdin.write(b'{"cancel":true}\n')
        self.process.stdin.flush()

    def collect(self, on_text=None, until=None):
        """Read events for the request in flight; safe to resume if interrupted."""
        while True:
            event = self.read_event(self.started + 2400)
            event["observed_seconds"] = time.monotonic() - self.started
            self.events.append(event)
            if event["event"] == "delta" and on_text:
                on_text(event["text"])
            if event["event"] in {"done", "error", "reset"} or (
                until is not None and until(event)
            ):
                return self.events

    def request(self, request, on_text=None):
        self.send(request)
        return self.collect(on_text)

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
    """An explicit process cap in MiB, or None to let the runner derive one."""
    return explicit_mb(f"{family} process budget", requested)


def command(
    binary,
    model,
    family,
    max_tokens=64,
    cache_mb=None,
    raw=False,
    combined_mb=None,
    verify=None,
):
    result = [
        str(Path(binary).resolve()),
        str(model),
        "-",
        FLAGS[family],
        "--backend",
        "metal",
        "--speculation-policy",
        "off",
        "--session",
        "--disable-thinking",
        "--max-tokens",
        str(max_tokens),
    ]
    result += budget_arguments(
        family,
        cache_mb if family == "flash" else None,
        process_budget(family, combined_mb),
        verify,
    )
    if raw:
        result += ["--raw-prompt"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--family", choices=("27b", "flash"), required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--cache-mb",
        type=int,
        help="Flash expert cache in MiB (default: derived from the machine profile)",
    )
    p.add_argument(
        "--combined-budget-mb",
        type=int,
        help="Process cap in MiB (default: derived from the machine profile)",
    )
    p.add_argument(
        "--verify",
        choices=VERIFY_MODES,
        default="stamp",
        help="stamp reuses an earlier full hash while the files are unchanged",
    )
    a = p.parse_args()
    try:
        budget = process_budget(a.family, a.combined_budget_mb)
        cache = explicit_mb("expert cache", a.cache_mb)
    except ValueError as error:
        p.error(str(error))
    session = Session(
        command(
            a.binary,
            a.model,
            a.family,
            a.max_tokens,
            cache,
            combined_mb=budget,
            verify=a.verify,
        ),
        detach=True,
    )
    ready = session.ready
    print(
        f"Ready ({ready.get('verification_mode')} verification, "
        f"{ready.get('process_cap_bytes', 0) / 1024**3:.2f} GiB process cap). "
        "/reset starts a new conversation; /quit exits; Ctrl-C stops a reply."
    )
    try:
        while True:
            prompt = input("You: ")
            if prompt == "/quit":
                break
            request = {"reset": True} if prompt == "/reset" else {"prompt": prompt}
            session.send(request)
            while True:
                try:
                    events = session.collect(
                        lambda text: print(text, end="", flush=True)
                    )
                    break
                except KeyboardInterrupt:
                    # Keep the loaded model: stop this reply, not the session.
                    session.cancel()
            print()
            last = events[-1]
            if last["event"] == "done":
                state = "; conversation state reset" if last.get("state_reset") else ""
                print(
                    f"[{last['finish_reason']}; reused {last['reused_tokens']} tokens; prefilled {last['prefilled_tokens']}; {last['prefill_ms'] / 1000:.2f}s prefill{state}]"
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
