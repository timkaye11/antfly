#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
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
"""Measure warm serving TTFT for the Gemma 4 26B-A4B compact model.

This drives the real ``antfly-inference`` HTTP server (no CLI shortcut) through
a cold / replay / strict-extension / second-prompt / non-streaming sequence and
reports time-to-first-token plus prompt-cache accounting. It is stdlib-only.

Compact Metal prompt reuse defaults OFF for correctness: logical prompt-cache
blocks do not yet retain the per-sequence device KV slots after a request ends.
The default gate therefore proves deterministic cold serving, zero compact KV
reuse, and the configured process-footprint ceiling. Passing
``--experimental-compact-prompt-cache-reuse`` enables both the node cache and
the explicitly unsafe runtime validation seam; that mode requires positive
cache-hit evidence and exact repeated-prompt output before it can qualify.

Warm-boot behaviour under test: the A4B model is preloaded before the first
request. Experimental reuse additionally provisions its cache pool at boot so
the first keyed request does not hide provisioning in measured TTFT.

TTFT definition: the server emits an initial ``role`` delta before generation,
so wall-clock time-to-first-token is measured to the first SSE data event that
carries a non-empty ``choices[0].delta.content`` (the first generated token),
not the role preamble.
"""

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from run_guarded_a4b import read_phys_footprint


# ---------------------------------------------------------------------------
# Deterministic prompt construction
# ---------------------------------------------------------------------------

_CORPUS = [
    "The maintenance crew logged the reactor coolant pressure at every shift.",
    "Auroral substorms brightened the northern sky above the survey station.",
    "A ledger of grain shipments crossed the river delta before the monsoon.",
    "The cartographer traced each tributary with a fine sable brush and ink.",
    "Migrating cranes rested on the salt flats where the old canal once ran.",
    "Engineers rebalanced the turbine blades after the third harmonic warning.",
    "The archivist catalogued brittle letters describing a forgotten treaty.",
    "Solar panels tilted to follow the low winter arc across the tundra.",
    "The apprentice measured tolerances twice before cutting the brass gear.",
    "Fog settled over the harbor while the lightkeeper trimmed the wick.",
    "A quiet algorithm sorted the census rolls by district and by trade.",
    "The botanist pressed each fern between sheets of unbleached paper.",
    "Seismographs traced faint tremors beneath the basalt plateau at dawn.",
    "The courier memorized the mountain passes to avoid the flooded road.",
    "Weavers dyed the wool with lichen gathered from the granite outcrops.",
    "The observatory logged a slow drift in the binary star's eclipse timing.",
]


def build_prompt(target_words: int, seed_offset: int = 0) -> str:
    """Return a deterministic prose block of roughly ``target_words`` words."""
    parts = []
    count = 0
    i = seed_offset
    para = []
    while count < target_words:
        sentence = _CORPUS[i % len(_CORPUS)]
        para.append(f"[{i:04d}] {sentence}")
        count += len(sentence.split()) + 1
        i += 1
        if len(para) == 5:
            parts.append(" ".join(para))
            para = []
    if para:
        parts.append(" ".join(para))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class BusyError(RuntimeError):
    """The server rejected the request for transient resource pressure (503).

    Admission runs before any token streams, so a fresh retry produces a clean
    TTFT for the succeeding attempt.
    """


def call_with_retry(fn, attempts=6, backoff=6.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except BusyError as exc:
            last = exc
            print(f"  [busy] {exc}; retry {i + 1}/{attempts} in {backoff:.0f}s", flush=True)
            time.sleep(backoff)
    raise last


def _delta_content(obj):
    try:
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("content")
    except (AttributeError, IndexError, TypeError):
        return None


def stream_generate(host, port, body, timeout):
    """POST a streaming generate request; return timing + cache accounting.

    TTFT is the wall time to the first SSE data event carrying non-empty
    ``delta.content``. ``first_event_ms`` is the time to the first data event of
    any kind (the role preamble) for transparency.
    """
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = json.dumps(body).encode("utf-8")
    t0 = time.perf_counter()
    conn.request(
        "POST", "/ai/v1/generate", body=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        data = resp.read()
        conn.close()
        msg = f"HTTP {resp.status}: {data[:600].decode('utf-8', 'replace')}"
        if resp.status == 503:
            raise BusyError(msg)
        raise RuntimeError(msg)

    ttft_ms = None
    first_event_ms = None
    total_ms = None
    last_usage = None
    events = 0
    content_pieces = []
    buf = b""
    while True:
        part = resp.read1(4096)
        now = time.perf_counter()
        if not part:
            break
        buf += part
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:"):].strip()
            if data == b"[DONE]":
                total_ms = (now - t0) * 1000.0
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            events += 1
            if first_event_ms is None:
                first_event_ms = (now - t0) * 1000.0
            content = _delta_content(obj)
            if content:
                if ttft_ms is None:
                    ttft_ms = (now - t0) * 1000.0
                content_pieces.append(content)
            usage = obj.get("usage")
            if usage:
                last_usage = usage
        else:
            continue
        break
    if total_ms is None:
        total_ms = (time.perf_counter() - t0) * 1000.0
    conn.close()

    usage = last_usage or {}
    return {
        "ttft_ms": ttft_ms,
        "first_event_ms": first_event_ms,
        "total_ms": total_ms,
        "events": events,
        "text": "".join(content_pieces),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_prompt_tokens": usage.get("cached_prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def nonstream_generate(host, port, body, timeout):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = json.dumps(body).encode("utf-8")
    t0 = time.perf_counter()
    conn.request(
        "POST", "/ai/v1/generate", body=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    raw = resp.read()
    total_ms = (time.perf_counter() - t0) * 1000.0
    conn.close()
    if resp.status != 200:
        msg = f"HTTP {resp.status}: {raw[:600].decode('utf-8', 'replace')}"
        if resp.status == 503:
            raise BusyError(msg)
        raise RuntimeError(msg)
    obj = json.loads(raw)
    usage = obj.get("usage") or {}
    choices = obj.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    text = message.get("content") if isinstance(message, dict) else choice.get("text")
    return {
        "total_ms": total_ms,
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_prompt_tokens": usage.get("cached_prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def http_get(host, port, path, timeout=5):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def scrape_prompt_cache_metrics(host, port):
    try:
        status, body = http_get(host, port, "/ai/v1/metrics", timeout=10)
    except OSError:
        return {}
    if status != 200:
        return {}
    wanted = {
        "antfly_inference_prompt_cache_block_hash_hits_total": "block_hash_hits",
        "antfly_inference_prompt_cache_block_hash_misses_total": "block_hash_misses",
        "antfly_inference_prompt_cache_cached_tokens": "cached_tokens",
        "antfly_inference_prompt_cache_live_bytes": "live_bytes",
        "antfly_inference_prompt_cache_block_hash_cached_blocks": "cached_blocks",
        "antfly_inference_prompt_cache_live_entries": "live_entries",
    }
    out = {}
    for raw in body.decode("utf-8", "replace").splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        fields = raw.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        if name in wanted:
            try:
                out[wanted[name]] = int(float(fields[1]))
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def wait_for_health(proc, host, port, deadline_s):
    start = time.time()
    while time.time() - start < deadline_s:
        if proc.poll() is not None:
            return False
        try:
            status, _ = http_get(host, port, "/healthz", timeout=3)
            if status == 200:
                return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


def acquire_lock(lock_dir, retry_sleep=30, max_wait=3600):
    start = time.time()
    while True:
        try:
            os.mkdir(lock_dir)
            return True
        except FileExistsError:
            if time.time() - start > max_wait:
                raise TimeoutError(f"could not acquire {lock_dir} within {max_wait}s")
            print(f"[lock] {lock_dir} held; waiting {retry_sleep}s...", flush=True)
            time.sleep(retry_sleep)


def release_lock(lock_dir):
    try:
        os.rmdir(lock_dir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.normpath(os.path.join(here, "..", "zig-out", "bin", "antfly-inference"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--budget-mb", type=int, default=6144)
    ap.add_argument(
        "--device-routing",
        choices=("off", "partial", "full", "auto"),
        default="off",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-phys-footprint-bytes",
        type=int,
        default=None,
        help="server footprint gate; defaults to --budget-mb expressed in MiB",
    )
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--binary", default=default_bin)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--prompt-words", type=int, default=850)
    ap.add_argument("--cache-mb", type=int, default=512)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument(
        "--experimental-compact-prompt-cache-reuse",
        action="store_true",
        help=(
            "enable the compact Metal device-KV reuse experiment; default is the "
            "correct cold-serving gate"
        ),
    )
    ap.add_argument(
        "--max-ttft-ms",
        type=float,
        default=0.0,
        help="optional TTFT ceiling; 0 disables the performance gate",
    )
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--ready-timeout", type=float, default=900.0)
    ap.add_argument("--keep-server", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--lock-dir", default="/tmp/a4b-run-lock")
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args()
    if args.budget_mb < 2048 or args.seed < 0 or args.max_ttft_ms < 0:
        ap.error(
            "--budget-mb must be at least 2048; --seed and --max-ttft-ms "
            "must be non-negative"
        )
    if args.max_phys_footprint_bytes is None:
        args.max_phys_footprint_bytes = args.budget_mb * 1024 * 1024

    model_dir = os.path.abspath(args.model_dir)
    models_dir = os.path.dirname(model_dir)
    model_name = os.path.basename(model_dir)
    if not os.path.isdir(model_dir):
        print(f"error: model dir not found: {model_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.binary):
        print(f"error: server binary not found: {args.binary}", file=sys.stderr)
        return 2

    config = {
        "models_dir": models_dir,
        "max_loaded_models": 1,
        "allow_unknown_models": False,
        "max_concurrent_requests": 32,
        "preload": [{
            "kind": "generator",
            "name": model_name,
            "backend": "metal",
            "memory_profile": "compact_2gbs",
            "memory_budget_mb": args.budget_mb,
            "device_routing": args.device_routing,
        }],
        "prompt_cache": {
            "enabled": args.experimental_compact_prompt_cache_reuse,
            "mode": "block_hash",
            "max_bytes_mb": args.cache_mb,
            "min_tokens": args.min_tokens,
            "ttl_ms": 300000,
        },
    }

    lock_held = False
    if not args.no_lock:
        acquire_lock(args.lock_dir)
        lock_held = True
        print(f"[lock] acquired {args.lock_dir}", flush=True)

    cfg_fd, cfg_path = tempfile.mkstemp(prefix="a4b-serving-", suffix=".json")
    with os.fdopen(cfg_fd, "w") as fh:
        json.dump(config, fh, indent=2)

    server_log_path = args.server_log or os.path.join(
        tempfile.gettempdir(), f"a4b-serving-{args.port}.log")
    log_fh = open(server_log_path, "w")

    env = dict(os.environ)
    env["TERMITE_SERVER_GENERATE_TIMING"] = "1"
    if args.experimental_compact_prompt_cache_reuse:
        env["TERMITE_COMPACT_PROMPT_CACHE_REUSE"] = "1"

    proc = None
    result = {"config_path": cfg_path, "server_log": server_log_path, "model_dir": model_dir,
              "budget_mb": args.budget_mb, "device_routing": args.device_routing,
              "seed": args.seed, "max_phys_footprint_bytes": args.max_phys_footprint_bytes,
              "prompt_cache_mode": (
                  "experimental_compact_reuse"
                  if args.experimental_compact_prompt_cache_reuse
                  else "cold_correctness"
              ),
              "peak_phys_footprint_bytes": 0, "cases": {}}
    try:
        print(f"[server] starting {args.binary} run --config {cfg_path} "
              f"--host {args.host} --port {args.port}", flush=True)
        proc = subprocess.Popen(
            [args.binary, "run", "--config", cfg_path, "--host", args.host,
             "--port", str(args.port)],
            stdout=log_fh, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        if not wait_for_health(proc, args.host, args.port, args.ready_timeout):
            tail = _tail_file(server_log_path, 60)
            raise RuntimeError(f"server did not become healthy; log tail:\n{tail}")
        print(
            "[server] healthy (warm complete, prompt cache "
            + ("experimental" if args.experimental_compact_prompt_cache_reuse else "disabled")
            + ")",
            flush=True,
        )

        def sample_server_footprint():
            sampled = read_phys_footprint(proc.pid)
            if sampled is not None:
                result["peak_phys_footprint_bytes"] = max(
                    result["peak_phys_footprint_bytes"], sampled[0], sampled[1]
                )

        sample_server_footprint()

        metrics_boot = scrape_prompt_cache_metrics(args.host, args.port)
        result["metrics_boot"] = metrics_boot

        p1 = build_prompt(args.prompt_words, seed_offset=0)
        p2 = build_prompt(args.prompt_words, seed_offset=7)
        ext_para = (
            "\n\nFollow-up notes appended for the strict-extension probe: "
            + build_prompt(80, seed_offset=3)
            + "\n\nQuestion: In one sentence, summarize the operational theme above."
        )
        ext_para2 = (
            "\n\nSecond follow-up appended after interleaving: "
            + build_prompt(80, seed_offset=11)
            + "\n\nQuestion: Name one recurring subject across these notes."
        )

        def gen_body(prompt, key, stream):
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "seed": args.seed,
                "stream": stream,
            }
            if args.experimental_compact_prompt_cache_reuse:
                body["prompt_cache_key"] = key
            return body

        k1 = "measure-k1"
        k2 = "measure-k2"

        def stream_case(prompt, key):
            return call_with_retry(
                lambda: stream_generate(args.host, args.port, gen_body(prompt, key, True), args.request_timeout))

        def nonstream_case(prompt, key):
            return call_with_retry(
                lambda: nonstream_generate(args.host, args.port, gen_body(prompt, key, False), args.request_timeout))

        # (a) COLD
        print("[case] cold (K1, fresh prompt)...", flush=True)
        cold = stream_case(p1, k1)
        sample_server_footprint()
        result["cases"]["cold"] = cold

        # (b) REPLAY (identical prompt, same key)
        print("[case] replay (K1, identical prompt)...", flush=True)
        replay = stream_case(p1, k1)
        sample_server_footprint()
        result["cases"]["replay"] = replay

        # (c) STRICT EXTENSION (same key, prompt + appended paragraph + question)
        print("[case] strict-extension (K1, prompt + appendix)...", flush=True)
        ext = stream_case(p1 + ext_para, k1)
        sample_server_footprint()
        result["cases"]["strict_extension"] = ext

        # (d) SECOND KEY (different prompt) then K1 extension again (survives interleaving)
        print("[case] second-key (K2, different prompt)...", flush=True)
        second = stream_case(p2, k2)
        sample_server_footprint()
        result["cases"]["second_key"] = second

        print("[case] strict-extension-after-interleave (K1)...", flush=True)
        ext2 = stream_case(p1 + ext_para2, k1)
        sample_server_footprint()
        result["cases"]["strict_extension_interleaved"] = ext2

        # (e) NON-STREAMING replay asserting usage.cached_prompt_tokens in the body
        print("[case] non-streaming replay (K1)...", flush=True)
        nonstream = nonstream_case(p1, k1)
        sample_server_footprint()
        result["cases"]["nonstream_replay"] = nonstream

        result["metrics_final"] = scrape_prompt_cache_metrics(args.host, args.port)
        result["generate_timing_ms"] = _extract_generate_timing(server_log_path)

        _assert_and_report(result, args)
        if result["peak_phys_footprint_bytes"] == 0:
            raise RuntimeError("server phys_footprint telemetry unavailable")
        if result["peak_phys_footprint_bytes"] > args.max_phys_footprint_bytes:
            raise RuntimeError(
                f"server phys_footprint {result['peak_phys_footprint_bytes']} exceeds "
                f"{args.max_phys_footprint_bytes}"
            )

    finally:
        if proc is not None and not args.keep_server:
            print("[server] stopping...", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        log_fh.close()
        if not args.keep_server:
            try:
                os.remove(cfg_path)
            except OSError:
                pass
        if lock_held and not args.keep_server:
            release_lock(args.lock_dir)
            print(f"[lock] released {args.lock_dir}", flush=True)
        elif lock_held:
            print(f"[lock] holding {args.lock_dir} (server kept alive)", flush=True)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[out] wrote {args.json_out}", flush=True)

    return 0 if result.get("passed") else 1


def _tail_file(path, n):
    try:
        with open(path, "r", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no log)"


def _extract_generate_timing(path):
    lines = []
    try:
        with open(path, "r", errors="replace") as fh:
            for raw in fh:
                if "generate_timing_ms:" in raw:
                    lines.append(raw.strip())
    except OSError:
        pass
    return lines


def _fmt(v, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.0f}{suffix}"
    return f"{v}{suffix}"


def _assert_and_report(result, args):
    cases = result["cases"]

    print("\n=== Serving TTFT summary ===")
    header = f"{'case':<32}{'ttft_ms':>10}{'total_ms':>10}{'cached_tok':>12}{'prompt_tok':>12}{'gen_tok':>9}"
    print(header)
    print("-" * len(header))
    order = ["cold", "replay", "strict_extension", "second_key",
             "strict_extension_interleaved", "nonstream_replay"]
    for name in order:
        c = cases.get(name)
        if not c:
            continue
        ttft = c.get("ttft_ms")
        if name == "nonstream_replay":
            ttft = None  # non-streaming has no first-token event
        print(f"{name:<32}{_fmt(ttft):>10}{_fmt(c.get('total_ms')):>10}"
              f"{_fmt(c.get('cached_prompt_tokens')):>12}{_fmt(c.get('prompt_tokens')):>12}"
              f"{_fmt(c.get('completion_tokens')):>9}")

    mb = result.get("metrics_boot", {})
    mf = result.get("metrics_final", {})
    print("\n=== Prompt cache ledger evidence (antfly_inference_prompt_cache_*) ===")
    print(f"  block_hash_hits : boot={mb.get('block_hash_hits')} final={mf.get('block_hash_hits')}")
    print(f"  cached_tokens   : boot={mb.get('cached_tokens')} final={mf.get('cached_tokens')}")
    print(f"  cached_blocks   : boot={mb.get('cached_blocks')} final={mf.get('cached_blocks')}")
    print(f"  live_bytes (prompt_cache_kv retained): boot={mb.get('live_bytes')} final={mf.get('live_bytes')}")

    # ---- assertions / gate ----
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))

    cold = cases["cold"]
    replay = cases["replay"]
    ext = cases["strict_extension"]
    second = cases["second_key"]
    ext2 = cases["strict_extension_interleaved"]
    nonstream = cases["nonstream_replay"]

    for name in order[:-1]:
        check(f"{name} produced a first token", cases[name].get("ttft_ms") is not None)

    experimental = args.experimental_compact_prompt_cache_reuse
    if experimental:
        check("replay cached_prompt_tokens > 0",
              (replay.get("cached_prompt_tokens") or 0) > 0,
              f"={replay.get('cached_prompt_tokens')}")
        # Replay should reuse most of the prompt (prompt - one page .. prompt).
        pt = replay.get("prompt_tokens") or 0
        ct = replay.get("cached_prompt_tokens") or 0
        check("replay reused >= prompt-32 tokens", pt > 0 and ct >= max(0, pt - 32),
              f"cached={ct} prompt={pt}")
        check("strict-extension cached_prompt_tokens > 0",
              (ext.get("cached_prompt_tokens") or 0) > 0,
              f"={ext.get('cached_prompt_tokens')}")
        check("second-key isolation: fresh key not fully cached",
              second.get("cached_prompt_tokens") is not None
              and second.get("prompt_tokens") is not None
              and (second["cached_prompt_tokens"] == 0
                   or second["cached_prompt_tokens"] < second["prompt_tokens"]),
              f"cached={second.get('cached_prompt_tokens')} prompt={second.get('prompt_tokens')}")
        check("K1 survives interleaving (extension still caches)",
              (ext2.get("cached_prompt_tokens") or 0) > 0,
              f"={ext2.get('cached_prompt_tokens')}")
        check("non-streaming usage.cached_prompt_tokens > 0",
              (nonstream.get("cached_prompt_tokens") or 0) > 0,
              f"={nonstream.get('cached_prompt_tokens')}")
        check("experimental replay output matches cold output",
              bool(cold.get("text")) and cold.get("text") == replay.get("text"))
        check("prompt-cache hit metric increased",
              mb.get("block_hash_hits") is not None
              and mf.get("block_hash_hits") is not None
              and mf["block_hash_hits"] > mb["block_hash_hits"])
        gate_case_name = "strict-extension"
        gate_ttft = ext.get("ttft_ms")
    else:
        for name in order:
            check(f"{name} reports zero cached prompt tokens",
                  cases[name].get("cached_prompt_tokens") == 0,
                  f"={cases[name].get('cached_prompt_tokens')}")
        check("cold replay output is deterministic",
              bool(cold.get("text")) and cold.get("text") == replay.get("text"))
        check("streaming and non-streaming output agree",
              bool(cold.get("text")) and cold.get("text") == nonstream.get("text"))
        check("cold serving retains zero prompt-cache bytes",
              mb.get("live_bytes") == 0 and mf.get("live_bytes") == 0,
              f"boot={mb.get('live_bytes')} final={mf.get('live_bytes')}")
        check("cold serving records no prompt-cache hits",
              mb.get("block_hash_hits") is not None
              and mf.get("block_hash_hits") == mb.get("block_hash_hits"),
              f"boot={mb.get('block_hash_hits')} final={mf.get('block_hash_hits')}")
        gate_case_name = "replay-cold"
        gate_ttft = replay.get("ttft_ms")

    if args.max_ttft_ms > 0:
        gate_ok = gate_ttft is not None and gate_ttft <= args.max_ttft_ms
        check(f"GATE {gate_case_name} TTFT <= {args.max_ttft_ms:.0f} ms", gate_ok,
              f"={_fmt(gate_ttft)} ms")
    else:
        gate_ok = True

    print("\n=== Checks ===")
    all_ok = True
    for name, ok, detail in checks:
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    result["passed"] = all_ok
    result["gate_case"] = gate_case_name
    result["gate_ttft_ms"] = gate_ttft
    result["gate_limit_ms"] = args.max_ttft_ms
    result["gate_pass"] = gate_ok

    if args.max_ttft_ms > 0 and not gate_ok:
        print("\n=== GATE MISS: attribution from server generate_timing_ms ===")
        timing = result.get("generate_timing_ms", [])
        for line in timing[-6:]:
            print("  " + line)
        if not timing:
            print("  (no generate_timing_ms lines captured; is TERMITE_SERVER_GENERATE_TIMING set?)")

    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
