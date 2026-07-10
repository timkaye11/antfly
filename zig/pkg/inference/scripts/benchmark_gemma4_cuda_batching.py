#!/usr/bin/env python3
"""Gate CUDA continuous batching against the serialized server baseline."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shlex
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((pct / 100.0) * len(ordered) + 0.999999) - 1))
    return ordered[index]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def choose_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def post_json(url: str, body: dict, timeout: float = 600.0) -> tuple[float, dict]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    elapsed_ms = (time.monotonic() - started) * 1000.0
    return elapsed_ms, json.loads(raw)


def response_fingerprint(response: dict) -> str:
    choice = response["choices"][0]
    content = choice.get("message", {}).get("content", choice.get("text", ""))
    completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
    return hashlib.sha256(f"{completion_tokens}\0{content}".encode()).hexdigest()


def response_summary(response: dict) -> dict:
    choice = response["choices"][0]
    return {
        "completion_tokens": int(response.get("usage", {}).get("completion_tokens", 0)),
        "finish_reason": choice.get("finish_reason"),
        "content": choice.get("message", {}).get("content", choice.get("text", "")),
    }


class Server:
    def __init__(self, args: argparse.Namespace, mode: str, output_dir: pathlib.Path):
        self.args = args
        self.mode = mode
        self.output_dir = output_dir
        self.port = choose_port()
        self.process: subprocess.Popen | None = None
        self.log = None

    def __enter__(self) -> "Server":
        config_path = self.output_dir / f"server-{self.mode}.json"
        config_path.write_text(
            json.dumps(
                {
                    "models_dir": str(self.args.models_dir),
                    "max_concurrent_requests": 32,
                    "generation_batching": {
                        "mode": self.mode,
                        "max_step_items": self.args.max_step_items,
                        "max_step_query_tokens": 512,
                        "max_decode_wait_us": self.args.decode_wait_us,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        self.log = (self.output_dir / f"server-{self.mode}.log").open("wb")
        env = os.environ.copy()
        env.setdefault("ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE", "1")
        env.pop("ANTFLY_INFERENCE_DISABLE_CONTINUOUS_BATCHING", None)
        command = [
            str(self.args.antfly_bin),
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--models-dir",
            str(self.args.models_dir),
            "--config",
            str(config_path),
        ]
        if self.args.server_prefix:
            command = shlex.split(self.args.server_prefix) + command
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        health = f"http://127.0.0.1:{self.port}/healthz"
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server {self.mode} exited with {self.process.returncode}")
            try:
                with urllib.request.urlopen(health, timeout=1):
                    return self
            except Exception:
                time.sleep(0.25)
        raise RuntimeError(f"server {self.mode} did not become ready")

    def __exit__(self, *_: object) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()

    @property
    def generate_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/ai/v1/generate"

    @property
    def metrics_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/metrics"


def run_wave(server: Server, body: dict, concurrency: int) -> dict:
    barrier = threading.Barrier(concurrency)

    def request() -> tuple[float, dict]:
        barrier.wait()
        return post_json(server.generate_url, body)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request) for _ in range(concurrency)]
        results = [future.result() for future in futures]
    wall_ms = (time.monotonic() - started) * 1000.0
    latencies = [elapsed for elapsed, _ in results]
    tokens = [int(response.get("usage", {}).get("completion_tokens", 0)) for _, response in results]
    return {
        "wall_ms": wall_ms,
        "latencies_ms": latencies,
        "completion_tokens": tokens,
        "aggregate_tok_s": sum(tokens) * 1000.0 / wall_ms,
        "fingerprints": [response_fingerprint(response) for _, response in results],
        "responses": [response_summary(response) for _, response in results],
    }


def measure_mode(server: Server, body: dict, concurrencies: list[int], warmups: int, repeats: int) -> dict:
    for _ in range(warmups):
        run_wave(server, body, 1)
    measured = {}
    for concurrency in concurrencies:
        waves = [run_wave(server, body, concurrency) for _ in range(repeats)]
        measured[str(concurrency)] = {
            "aggregate_tok_s": stats([wave["aggregate_tok_s"] for wave in waves]),
            "wall_ms": stats([wave["wall_ms"] for wave in waves]),
            "request_latency_ms": stats([value for wave in waves for value in wave["latencies_ms"]]),
            "fingerprints": sorted({fingerprint for wave in waves for fingerprint in wave["fingerprints"]}),
            "waves": waves,
        }
    try:
        with urllib.request.urlopen(server.metrics_url, timeout=5) as response:
            metrics = response.read().decode("utf-8", errors="replace")
    except Exception:
        metrics = ""
    return {"measurements": measured, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    repo = pathlib.Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--antfly-bin", type=pathlib.Path, default=repo / "zig/pkg/inference/zig-out/bin/antfly-inference")
    parser.add_argument("--model", type=pathlib.Path, default=repo / ".models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf")
    parser.add_argument("--models-dir", type=pathlib.Path, default=repo / ".models")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("/tmp/antfly-gemma4-cuda-batching"))
    parser.add_argument("--prompt", default="Write one sentence about ants.")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--cache-dtype", default="f32")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-wait-us", type=int, default=1000)
    parser.add_argument("--max-step-items", type=int, default=2)
    parser.add_argument("--min-c4-speedup", type=float, default=2.0)
    parser.add_argument("--max-c1-p95-ratio", type=float, default=1.05)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--server-prefix", default=os.environ.get("ANTFLY_BATCH_SERVER_PREFIX", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.antfly_bin.exists() or not args.model.exists():
        raise SystemExit("antfly binary or model does not exist")
    if 1 not in args.concurrency or 4 not in args.concurrency:
        raise SystemExit("--concurrency must include 1 and 4 for acceptance gates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "model": str(args.model.resolve()),
        "backend": "cuda",
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.tokens,
        "temperature": 0,
        "stream": False,
        "cache_dtype": args.cache_dtype,
    }

    with Server(args, "off", args.output_dir) as server:
        baseline = measure_mode(server, body, [1], args.warmups, args.repeats)
    with Server(args, "on", args.output_dir) as server:
        batched = measure_mode(server, body, args.concurrency, args.warmups, args.repeats)

    baseline_c1 = baseline["measurements"]["1"]
    batched_c1 = batched["measurements"]["1"]
    batched_c4 = batched["measurements"]["4"]
    speedup = batched_c4["aggregate_tok_s"]["median"] / baseline_c1["aggregate_tok_s"]["median"]
    c1_p95_ratio = batched_c1["request_latency_ms"]["p95"] / baseline_c1["request_latency_ms"]["p95"]
    expected_fingerprints = baseline_c1["fingerprints"]
    exact = len(expected_fingerprints) == 1 and all(
        measurement["fingerprints"] == expected_fingerprints
        for measurement in batched["measurements"].values()
    )
    summary = {
        "config": {
            "model": str(args.model.resolve()),
            "tokens": args.tokens,
            "cache_dtype": args.cache_dtype,
            "concurrency": args.concurrency,
            "decode_wait_us": args.decode_wait_us,
            "max_step_items": args.max_step_items,
        },
        "baseline": baseline,
        "batched": batched,
        "acceptance": {
            "c4_aggregate_speedup": speedup,
            "min_c4_aggregate_speedup": args.min_c4_speedup,
            "c1_p95_latency_ratio": c1_p95_ratio,
            "max_c1_p95_latency_ratio": args.max_c1_p95_ratio,
            "exact_response_fingerprints": exact,
            "passed": speedup >= args.min_c4_speedup and c1_p95_ratio <= args.max_c1_p95_ratio and exact,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"batching_gate c4_speedup={speedup:.3f} c1_p95_ratio={c1_p95_ratio:.3f} "
        f"exact={str(exact).lower()} output={args.output_dir / 'summary.json'}"
    )
    if not summary["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
