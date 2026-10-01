#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Reproduce the uncached Magnitude Gemma4 E2B streaming workload.

Owns only the server it starts. An overall POSIX deadline bounds the entire
request, including streams that continue to send heartbeat lines. Diagnostics
and timed runs must use separate output directories and fresh server processes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

WARMUP = ("warmup", 33, 64, "Write a detailed guide to brewing coffee at home, covering equipment, grind size, water temperature, ratios, and troubleshooting.")


def cases():
    long_prompt = (
        "Reference notes:\n"
        + "".join(
            f"Batch {i}: beans were washed, dried carefully, roasted to medium, "
            "and brewed using filtered water at 93 degrees Celsius. "
            "The tasting notes were citrus, caramel, and chocolate.\n"
            for i in range(1, 101)
        )
        + "\nWrite a detailed analytical report about these batches and a "
        "comprehensive plan for improving consistency. Use at least 1000 words."
    )
    return [
        ("short-256", 27, 256, "Write an extensive practical guide to brewing pour-over coffee. Explain every step in detail."),
        ("short-512-a", 27, 512, "Write a comprehensive guide to designing a reliable local inference service, with detailed explanations and examples."),
        ("short-512-b", 33, 512, "Write a long, detailed science-fiction story about a botanist growing coffee on Mars. Include dialogue and vivid descriptions."),
        ("short-1024", 38, 1024, "Write a comprehensive textbook chapter explaining how transformer language models work. Explain each component in detail and continue for at least 2000 words."),
        ("long-input-512", 4031, 512, long_prompt),
    ]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def request_deadline(seconds):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("This Metal benchmark requires POSIX request deadlines")
    def expired(_signum, _frame):
        raise TimeoutError(f"overall request deadline exceeded ({seconds}s)")
    previous = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, previous)


def timing_result(start, first, last, end, count):
    decode = last - first if first is not None and last is not None else None
    return {
        "ttft_s": first - start if first is not None else None,
        "total_s": end - start,
        "decode_s": decode,
        "decode_tokens_per_s": (count - 1) / decode if count and decode and decode > 0 else None,
        "end_to_end_tokens_per_s": count / (end - start) if count and end > start else None,
    }


@contextmanager
def benchmark_response(req, timeout):
    # Keep HTTP error bodies inside the same overall deadline as the stream.
    # Bound the diagnostic body as well so an error cannot consume unbounded
    # memory or escape receipt/owned-server cleanup if its read fails.
    with request_deadline(timeout):
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            with error:
                body = error.read(4096).decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {body}") from error
        with response:
            yield response


def request(url, model, case, output, args):
    label, expected_input, limit, prompt = case
    if args.max_tokens is not None and label != "warmup":
        limit = args.max_tokens
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": limit, "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "chat_template_kwargs": {"enable_thinking": False},
        "backend": "metal", "prompt_cache": False,
        "stream": True, "stream_options": {"include_usage": True},
    }
    (output / "request.json").write_text(json.dumps(payload, indent=2))
    result = {"case": label, "expected_prompt_tokens": expected_input, "output_limit": limit}
    first = last = None
    usage = finish = None
    chunks = reasoning_chars = 0
    text = []
    events = []
    event_type = "message"
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with benchmark_response(req, args.request_timeout) as response:
            for raw in response:
                if raw.startswith(b"event: "):
                    event_type = raw[7:].strip().decode()
                    continue
                if not raw.strip():
                    event_type = "message"
                    continue
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if not data:
                    continue
                if data == b"[DONE]":
                    break
                if event_type == "error":
                    raise RuntimeError(data.decode(errors="replace"))
                try:
                    event = json.loads(data)
                except ValueError:
                    result["invalid_event_preview"] = data[:240].decode(errors="replace")
                    raise
                events.append({"elapsed_s": time.perf_counter() - start, "event": event})
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    reasoning_chars += len(delta.get("reasoning_content") or "")
                    content = delta.get("content")
                    if content:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
                        text.append(content)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except (TimeoutError, OSError, ValueError, RuntimeError) as error:
        result["error"] = str(error)
    finally:
        result.update(timing_result(start, first, last, time.perf_counter(), usage.get("completion_tokens") if usage else None))
        result.update(usage=usage, finish_reason=finish, content_chunks=chunks, reasoning_chars=reasoning_chars)
        result["output_sha256"] = hashlib.sha256("".join(text).encode()).hexdigest()
        (output / "output.txt").write_text("".join(text))
        (output / "events.json").write_text(json.dumps(events))
        if "error" not in result:
            if not usage or usage.get("prompt_tokens") != expected_input or usage.get("completion_tokens") != limit:
                result["error"] = "prompt/output token counts differ from benchmark contract"
            elif finish != "length" or reasoning_chars:
                result["error"] = "expected length finish with reasoning disabled"
        (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def command_output(*command):
    return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()


def system_state():
    return {
        "hardware": command_output("sysctl", "hw.model", "hw.memsize", "machdep.cpu.brand_string", "hw.logicalcpu", "vm.swapusage"),
        "memory_pressure": command_output("memory_pressure"),
        "thermal_limits": command_output("pmset", "-g", "therm"),
        "time": time.time(),
    }


def physical_footprint(pid):
    """Darwin rusage_info_v2: includes shared Metal allocations missed by RSS."""
    if sys.platform != "darwin":
        return None
    class RusageInfoV2(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in (
                "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
                "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
                "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
                "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
            )
        ]
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return None
    probe = library.proc_pid_rusage
    probe.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    probe.restype = ctypes.c_int
    info = RusageInfoV2()
    return info.phys_footprint if probe(pid, 2, ctypes.byref(info)) == 0 else None


def process_tree_pids(root):
    rows = [tuple(map(int, row.split())) for row in command_output("ps", "-axo", "pid,ppid").splitlines()[1:]]
    owned = {root}
    for _ in range(8):
        grown = owned | {pid for pid, parent in rows if parent in owned}
        if grown == owned:
            break
        owned = grown
    return owned


def parse_server_counters(log):
    counters = []
    for line in log.splitlines():
        if re.search(r"(?:generate_timing_ms|metal_\w+|decoder_\w+):", line):
            values = dict(re.findall(r"([\w]+)=([\w.+-]+)", line))
            counters.append({"label": line.split(":", 1)[0], "values": values})
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/gemma4-e2b-qat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=[c[0] for c in cases()], action="append")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--request-gap", type=float, default=1)
    parser.add_argument("--max-footprint-gib", type=float, default=10)
    parser.add_argument("--prefill-chunk-size", type=int, choices=(256, 512, 1024, 2048), default=2048)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--server-arg", action="append", default=[], help="Explicit server CLI argument; use = for values starting with --")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.request_timeout <= 0 or args.request_gap < 0 or args.max_footprint_gib <= 0 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("invalid rounds, token limit, deadline, or request gap")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = {"models_dir": str(args.models_dir.resolve()), "ml_dir": str(output / "ml"), "max_loaded_models": 1,
              "prompt_cache": {"enabled": False}, "admission": {"inference": {"max_concurrent_requests": 1}},
              "generation_batching": {"max_idle_prefill_chunk_size": args.prefill_chunk_size}}
    config_path = output / "server-config.json"
    config_path.write_text(json.dumps(config, indent=2))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TERMITE_", "ANTFLY_INFERENCE_METAL_"))}
    requested_env = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not key.startswith(("TERMITE_", "ANTFLY_")):
            parser.error("--env requires a TERMITE_ or ANTFLY_ KEY=VALUE")
        requested_env[key] = value
    if args.diagnostic:
        requested_env.update(TERMITE_SERVER_GENERATE_TIMING="1", TERMITE_DEBUG_METAL_TIMING="1", TERMITE_METAL_STAGE_TIMING="1")
    elif any("TIMING" in k or "TRACE" in k for k, v in requested_env.items() if v != "0"):
        parser.error("tracing/timing environment requires --diagnostic")
    env.update(requested_env)
    binary = args.binary.resolve()
    model_file = args.models_dir / "generators" / args.model / "model.gguf"
    cmd = [str(binary), "run", "--host", "127.0.0.1", "--port", str(port), "--models-dir", str(args.models_dir.resolve()), "--config", str(config_path)]
    cmd.extend(args.server_arg)
    receipt = {"schema": "antfly.magnitude_gemma4_http.v1", "diagnostic": args.diagnostic,
               "binary": str(binary), "binary_sha256": file_sha256(binary),
               "model_file": str(model_file.resolve()), "model_sha256": file_sha256(model_file),
               "source_sha": command_output("git", "rev-parse", "HEAD"), "source_status": command_output("git", "status", "--porcelain"),
               "command": cmd, "environment": requested_env, "config": config, "before": system_state(), "results": []}
    log_path = output / "server.log"
    log = log_path.open("w")
    server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    monitor_stop = threading.Event()
    receipt["physical_footprint_samples"] = []
    def monitor_memory():
        while not monitor_stop.wait(2):
            try:
                values = [physical_footprint(pid) for pid in process_tree_pids(server.pid)]
                measured = [value for value in values if value is not None]
                if measured:
                    footprint = sum(measured)
                    receipt["physical_footprint_samples"].append({"time": time.time(), "bytes": footprint})
                    if footprint > args.max_footprint_gib * 1024 ** 3:
                        receipt["watchdog_error"] = "process tree physical footprint exceeded benchmark limit"
                        os.killpg(server.pid, signal.SIGTERM)
                        return
            except (OSError, subprocess.SubprocessError):
                return
    memory_thread = threading.Thread(target=monitor_memory, daemon=True)
    memory_thread.start()
    try:
        ready_by = time.monotonic() + 30
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                if time.monotonic() >= ready_by:
                    raise TimeoutError("server readiness deadline exceeded")
                time.sleep(0.2)
        workload = [c for c in cases() if not args.case or c[0] in args.case]
        for case in [WARMUP] + workload * args.rounds:
            index = len(receipt["results"])
            case_output = output / f"{index:03d}-{case[0]}"
            case_output.mkdir()
            result = request(f"http://127.0.0.1:{port}/ai/v1/generate", args.model, case, case_output, args)
            receipt["results"].append(result)
            (output / "results.json").write_text(json.dumps(receipt, indent=2))
            print(json.dumps(result), flush=True)
            if result.get("error"):
                raise RuntimeError(result["error"])
            time.sleep(args.request_gap)
        # footprint includes shared Metal allocations, unlike ps RSS. This is
        # post-workload evidence, not a claim to have measured peak footprint.
        if Path("/usr/bin/footprint").exists():
            subprocess.run(["/usr/bin/footprint", "-p", str(server.pid), "-j", str(output / "footprint.json")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except (RuntimeError, TimeoutError, OSError) as error:
        receipt["error"] = str(error)
    finally:
        monitor_stop.set()
        memory_thread.join(timeout=3)
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if server.poll() is None:
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        log.close()
        receipt["server_counters"] = parse_server_counters(log_path.read_text())
        receipt["max_physical_footprint_bytes"] = max((s["bytes"] for s in receipt["physical_footprint_samples"]), default=None)
        receipt["after"] = system_state()
        (output / "results.json").write_text(json.dumps(receipt, indent=2))
    return 1 if receipt.get("error") or receipt.get("watchdog_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
