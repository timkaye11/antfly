#!/usr/bin/env python3
"""Paired Gemma 4 26B-A4B benchmark: Antfly compact Metal vs TurboFieldfare.

Memory telemetry is phys_footprint-first: each engine process is spawned
directly (no /usr/bin/time wrapper) and sampled every ~50ms via
``proc_pid_rusage(RUSAGE_INFO_V4)`` from libproc, recording the peak
``ri_phys_footprint`` observed plus the last readable
``ri_lifetime_max_phys_footprint`` before the child is reaped.  The primary
memory gate is ``--max-phys-footprint-bytes``; max RSS (from ``os.wait4``
rusage, reported in bytes on macOS) is retained as secondary telemetry with an
optional ``--max-rss-bytes`` gate (0 = report-only).  On platforms without
``proc_pid_rusage`` the footprint records 0 and the run fails closed unless
``--allow-missing-footprint`` is passed.  A ``--max-decode-cv`` gate (default
0.03) fails the run when either engine's decode throughput coefficient of
variation exceeds it: CV <= 3% or no claim. Every measured run must also
produce the same normalized generated text across and within engines; output
divergence is rejected before throughput is summarized.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


ANTFLY_INNER_RE = re.compile(
    r"^generate_timing_ms:.*\bprefill=(?P<prefill>\d+)\s+decode=(?P<decode>\d+).*$",
    re.MULTILINE,
)
ANTFLY_TOTAL_RE = re.compile(
    r"^timing_ms:\s+.*\bload_model=(?P<load>\d+).*\btotal=(?P<total>\d+)\s*$",
    re.MULTILINE,
)
ANTFLY_RATE_RE = re.compile(r"^decode_tok_per_s=(?P<rate>[0-9.]+)\s*$", re.MULTILINE)
ANTFLY_PROMPT_IDS_RE = re.compile(r"^prompt_token_ids:(?P<ids>(?:\s+\d+)*)\s*$", re.MULTILINE)
ANTFLY_TOKEN_IDS_RE = re.compile(r"^token_ids:(?P<ids>(?:\s+\d+)*)\s*$", re.MULTILINE)
TURBO_FOOTER_RE = re.compile(
    r"\[stop=.*?\s+prefill=(?P<prefill>\d+)tok\s+new=(?P<new>\d+)tok"
    r"\s+decode=(?P<decode>[0-9.]+)s\s+tok/s=(?P<rate>[0-9.]+)\]"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# Legacy /usr/bin/time -l parsers.  Live runs no longer wrap engines in
# /usr/bin/time (wall time is a monotonic clock around the child and RSS comes
# from wait4 rusage), but the parse functions still accept optional time-style
# stderr so recorded logs and unit tests remain parseable.
TIME_REAL_RE = re.compile(r"^\s*(?P<seconds>[0-9.]+)\s+real\b", re.MULTILINE)
TIME_RSS_RE = re.compile(r"^\s*(?P<bytes>\d+)\s+maximum resident set size\s*$", re.MULTILINE)

# Matches the compact contract's resident ceiling at the 2048 MiB budget
# floor (CompactInferenceConfig.default_resident_ceiling_bytes). Runs
# qualifying a larger --memory-budget-mb should pass the matching ceiling via
# --max-phys-footprint-bytes.
DEFAULT_MAX_PHYS_FOOTPRINT_BYTES = 2048 * 1024 * 1024
DEFAULT_MAX_DECODE_CV = 0.03
FOOTPRINT_POLL_SECONDS = 0.05
RUSAGE_INFO_V4 = 4


class BenchmarkError(RuntimeError):
    pass


class RUsageInfoV4(ctypes.Structure):
    """struct rusage_info_v4 from <sys/resource.h> (macOS).

    Layout is ri_uuid[16] followed by 35 uint64 counters; the full field list
    keeps ctypes offsets exact (ri_phys_footprint at 72,
    ri_lifetime_max_phys_footprint at 240, sizeof == 296).
    """

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time",
            "ri_serviced_system_time",
            "ri_logical_writes",
            "ri_lifetime_max_phys_footprint",
            "ri_instructions",
            "ri_cycles",
            "ri_billed_energy",
            "ri_serviced_energy",
            "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


def _load_libproc() -> ctypes.CDLL | None:
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libproc.proc_pid_rusage.restype = ctypes.c_int
        libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        return libproc
    except (OSError, AttributeError):
        return None


_LIBPROC = _load_libproc()


def read_phys_footprint(pid: int) -> tuple[int, int] | None:
    """Return (ri_phys_footprint, ri_lifetime_max_phys_footprint) bytes or None."""
    if _LIBPROC is None:
        return None
    info = RUsageInfoV4()
    if _LIBPROC.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint), int(info.ri_lifetime_max_phys_footprint)


@dataclass(frozen=True)
class Sample:
    engine: str
    iteration: int
    prompt_tokens: int
    output_tokens: int
    decode_ms: float
    decode_tok_per_s: float
    ttft_proxy_ms: float
    wall_ms: float
    max_rss_bytes: int
    peak_phys_footprint_bytes: int
    output_sha256: str
    log_path: str


def _ids(match: re.Match[str] | None, label: str) -> list[int]:
    if match is None:
        raise BenchmarkError(f"missing {label}")
    return [int(value) for value in match.group("ids").split()]


def _normalized_generated_text(text: str) -> str:
    """Canonicalize transport-only differences without changing content."""
    without_ansi = ANSI_ESCAPE_RE.sub("", text)
    normalized_newlines = without_ansi.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines).strip()


def _output_hash(text: str) -> str:
    return hashlib.sha256(_normalized_generated_text(text).encode("utf-8")).hexdigest()


def _antfly_generated_text(stdout: str) -> str:
    prompt_ids = ANTFLY_PROMPT_IDS_RE.search(stdout)
    token_ids = ANTFLY_TOKEN_IDS_RE.search(stdout)
    if prompt_ids is None or token_ids is None or token_ids.start() < prompt_ids.end():
        raise BenchmarkError("cannot isolate Antfly generated text from token diagnostics")
    return stdout[prompt_ids.end():token_ids.start()]


def _turbo_generated_text(stdout: str) -> str:
    # Turbo normally writes its footer to stderr, but accept stdout footers in
    # recorded logs while hashing only the user-visible generated text.
    return TURBO_FOOTER_RE.sub("", stdout)


def _wall_and_rss(
    time_stderr: str, wall_ms: float | None, max_rss_bytes: int | None, label: str
) -> tuple[float, int]:
    """Use harness measurements when given, else fall back to /usr/bin/time -l text."""
    if wall_ms is None:
        real = TIME_REAL_RE.search(time_stderr)
        if real is None:
            raise BenchmarkError(f"missing {label} wall time (no harness value or time -l output)")
        wall_ms = float(real.group("seconds")) * 1000.0
    if max_rss_bytes is None:
        rss = TIME_RSS_RE.search(time_stderr)
        if rss is None:
            raise BenchmarkError(f"missing {label} max RSS (no harness value or time -l output)")
        max_rss_bytes = int(rss.group("bytes"))
    return wall_ms, max_rss_bytes


def parse_antfly(
    stdout: str,
    stderr: str,
    iteration: int,
    log_path: Path,
    wall_ms: float | None = None,
    max_rss_bytes: int | None = None,
    peak_phys_footprint_bytes: int = 0,
) -> Sample:
    combined = stdout + "\n" + stderr
    inner = ANTFLY_INNER_RE.search(combined)
    total = ANTFLY_TOTAL_RE.search(combined)
    rate = ANTFLY_RATE_RE.search(combined)
    if None in (inner, total, rate):
        raise BenchmarkError("incomplete Antfly timing output")
    wall_ms, max_rss_bytes = _wall_and_rss(stderr, wall_ms, max_rss_bytes, "Antfly")
    prompt_ids = _ids(ANTFLY_PROMPT_IDS_RE.search(combined), "Antfly prompt token ids")
    token_ids = _ids(ANTFLY_TOKEN_IDS_RE.search(combined), "Antfly output token ids")
    decode_ms = float(inner.group("decode"))
    total_ms = float(total.group("total"))
    output_hash = _output_hash(_antfly_generated_text(stdout))
    return Sample(
        engine="antfly",
        iteration=iteration,
        prompt_tokens=len(prompt_ids),
        output_tokens=len(token_ids),
        decode_ms=decode_ms,
        decode_tok_per_s=float(rate.group("rate")),
        # Use the same fresh-process wall-minus-decode proxy as Turbo. The
        # engine-reported total is retained only as a parser completeness gate.
        ttft_proxy_ms=max(0.0, wall_ms - decode_ms),
        wall_ms=wall_ms,
        max_rss_bytes=max_rss_bytes,
        peak_phys_footprint_bytes=peak_phys_footprint_bytes,
        output_sha256=output_hash,
        log_path=str(log_path),
    )


def parse_turbo(
    stdout: str,
    stderr: str,
    iteration: int,
    log_path: Path,
    wall_ms: float | None = None,
    max_rss_bytes: int | None = None,
    peak_phys_footprint_bytes: int = 0,
) -> Sample:
    combined = stdout + "\n" + stderr
    footer = TURBO_FOOTER_RE.search(combined)
    if footer is None:
        raise BenchmarkError("incomplete TurboFieldfare footer output")
    wall_ms, max_rss_bytes = _wall_and_rss(stderr, wall_ms, max_rss_bytes, "TurboFieldfare")
    decode_ms = float(footer.group("decode")) * 1000.0
    return Sample(
        engine="turbo",
        iteration=iteration,
        prompt_tokens=int(footer.group("prefill")),
        output_tokens=int(footer.group("new")),
        decode_ms=decode_ms,
        decode_tok_per_s=float(footer.group("rate")),
        # The public CLI exposes decode time but not TTFT. Fresh-process wall
        # minus decode is the comparable load+prefill+first-token proxy.
        ttft_proxy_ms=max(0.0, wall_ms - decode_ms),
        wall_ms=wall_ms,
        max_rss_bytes=max_rss_bytes,
        peak_phys_footprint_bytes=peak_phys_footprint_bytes,
        output_sha256=_output_hash(_turbo_generated_text(stdout)),
        log_path=str(log_path),
    )


def run_sample(
    engine: str,
    iteration: int,
    command: list[str],
    out_dir: Path,
    timeout: float,
) -> Sample:
    log_path = out_dir / f"{iteration:02d}-{engine}.log"
    started = time.monotonic()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        peak_footprint = 0
        lifetime_max_footprint = 0
        while True:
            # Sample before the reap check so the footprint stays readable; the
            # observed ri_lifetime_max_phys_footprint is monotonic and carries
            # the process-lifetime peak up to the final pre-reap sample.
            footprint = read_phys_footprint(process.pid)
            if footprint is not None:
                peak_footprint = max(peak_footprint, footprint[0])
                lifetime_max_footprint = max(lifetime_max_footprint, footprint[1])
            reaped_pid, status, rusage = os.wait4(process.pid, os.WNOHANG)
            if reaped_pid == process.pid:
                break
            if time.monotonic() - started > timeout:
                process.kill()
                os.wait4(process.pid, 0)
                process.returncode = -9
                raise BenchmarkError(
                    f"{engine} timed out after {timeout:.0f}s: {' '.join(command)}"
                )
            time.sleep(FOOTPRINT_POLL_SECONDS)
        wall_ms = (time.monotonic() - started) * 1000.0
        if os.WIFSIGNALED(status):
            process.returncode = -os.WTERMSIG(status)
        else:
            process.returncode = os.WEXITSTATUS(status)
        # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
        max_rss_bytes = int(rusage.ru_maxrss)
        if sys.platform != "darwin":
            max_rss_bytes *= 1024
        stdout_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr_file.seek(0)
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    peak_phys_footprint_bytes = max(peak_footprint, lifetime_max_footprint)
    log_path.write_text(
        f"$ {' '.join(command)}\n"
        f"exit_code={process.returncode} wall_seconds={wall_ms / 1000.0:.6f} "
        f"max_rss_bytes={max_rss_bytes} "
        f"peak_phys_footprint_bytes={peak_phys_footprint_bytes} "
        f"(sampled_peak={peak_footprint} lifetime_max={lifetime_max_footprint})\n"
        "--- stdout ---\n"
        f"{stdout}\n"
        "--- stderr ---\n"
        f"{stderr}\n"
    )
    if process.returncode != 0:
        raise BenchmarkError(f"{engine} exited {process.returncode}; see {log_path}")
    parser = parse_antfly if engine == "antfly" else parse_turbo
    return parser(
        stdout,
        stderr,
        iteration,
        log_path,
        wall_ms=wall_ms,
        max_rss_bytes=max_rss_bytes,
        peak_phys_footprint_bytes=peak_phys_footprint_bytes,
    )


def _median(samples: list[Sample], field: str) -> float:
    return statistics.median(float(getattr(sample, field)) for sample in samples)


def _cv(samples: list[Sample], field: str) -> float:
    values = [float(getattr(sample, field)) for sample in samples]
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0


def summarize(
    samples: list[Sample],
    requested_tokens: int,
    *,
    max_phys_footprint_bytes: int = DEFAULT_MAX_PHYS_FOOTPRINT_BYTES,
    antfly_max_phys_footprint_bytes: int | None = None,
    turbo_max_phys_footprint_bytes: int | None = None,
    max_rss_bytes: int = 0,
    max_decode_cv: float = DEFAULT_MAX_DECODE_CV,
    allow_missing_footprint: bool = False,
) -> dict:
    by_engine = {
        engine: [sample for sample in samples if sample.engine == engine]
        for engine in ("antfly", "turbo")
    }
    if not all(by_engine.values()):
        raise BenchmarkError("both engines require at least one measured sample")
    prompt_counts = {sample.prompt_tokens for sample in samples}
    if len(prompt_counts) != 1:
        raise BenchmarkError(f"prompt token counts differ across engines: {sorted(prompt_counts)}")
    output_hashes = {
        engine: {sample.output_sha256 for sample in engine_samples}
        for engine, engine_samples in by_engine.items()
    }
    for engine, hashes in output_hashes.items():
        if len(hashes) != 1:
            raise BenchmarkError(
                f"{engine} generated nondeterministic outputs across measured runs: {sorted(hashes)}"
            )
    antfly_output = next(iter(output_hashes["antfly"]))
    turbo_output = next(iter(output_hashes["turbo"]))
    if antfly_output != turbo_output:
        raise BenchmarkError(
            "generated outputs differ across engines; throughput is not a matched-workload comparison "
            f"(antfly={antfly_output}, turbo={turbo_output})"
        )
    footprint_notes: set[str] = set()
    for sample in samples:
        if sample.output_tokens != requested_tokens:
            raise BenchmarkError(
                f"{sample.engine} emitted {sample.output_tokens}, expected {requested_tokens}"
            )
        if sample.peak_phys_footprint_bytes > 0:
            engine_ceiling = (
                antfly_max_phys_footprint_bytes
                if sample.engine == "antfly" and antfly_max_phys_footprint_bytes is not None
                else turbo_max_phys_footprint_bytes
                if sample.engine == "turbo" and turbo_max_phys_footprint_bytes is not None
                else max_phys_footprint_bytes
            )
            if sample.peak_phys_footprint_bytes > engine_ceiling:
                raise BenchmarkError(
                    f"{sample.engine} phys_footprint {sample.peak_phys_footprint_bytes} "
                    f"exceeds {engine_ceiling}"
                )
        elif not allow_missing_footprint:
            raise BenchmarkError(
                f"{sample.engine} iteration {sample.iteration} has no phys_footprint "
                "telemetry (proc_pid_rusage unavailable?); pass --allow-missing-footprint "
                "for RSS-only reporting"
            )
        else:
            footprint_notes.add(
                f"{sample.engine}: phys_footprint unavailable; RSS-only telemetry"
            )
        if max_rss_bytes > 0 and sample.max_rss_bytes > max_rss_bytes:
            raise BenchmarkError(
                f"{sample.engine} RSS {sample.max_rss_bytes} exceeds {max_rss_bytes}"
            )

    engines: dict[str, dict] = {}
    for engine, engine_samples in by_engine.items():
        engines[engine] = {
            "samples": len(engine_samples),
            "median_decode_tok_per_s": _median(engine_samples, "decode_tok_per_s"),
            "decode_tok_per_s_cv": _cv(engine_samples, "decode_tok_per_s"),
            "median_decode_ms": _median(engine_samples, "decode_ms"),
            "median_ttft_proxy_ms": _median(engine_samples, "ttft_proxy_ms"),
            "median_wall_ms": _median(engine_samples, "wall_ms"),
            "peak_phys_footprint_bytes": max(
                sample.peak_phys_footprint_bytes for sample in engine_samples
            ),
            "peak_rss_bytes": max(sample.max_rss_bytes for sample in engine_samples),
            "output_sha256": sorted({sample.output_sha256 for sample in engine_samples}),
        }
        cv = engines[engine]["decode_tok_per_s_cv"]
        if max_decode_cv > 0 and cv > max_decode_cv:
            raise BenchmarkError(
                f"{engine} decode_tok_per_s CV {cv:.4f} exceeds --max-decode-cv "
                f"{max_decode_cv:.4f}: unstable run, no claim"
            )
    turbo_rate = engines["turbo"]["median_decode_tok_per_s"]
    engines["antfly"]["decode_ratio_vs_turbo"] = (
        engines["antfly"]["median_decode_tok_per_s"] / turbo_rate
        if turbo_rate
        else math.inf
    )
    engines["antfly"]["ttft_proxy_ratio_vs_turbo"] = (
        engines["antfly"]["median_ttft_proxy_ms"]
        / engines["turbo"]["median_ttft_proxy_ms"]
        if engines["turbo"]["median_ttft_proxy_ms"]
        else math.inf
    )
    return {
        "contract": {
            "prompt_tokens": prompt_counts.pop(),
            "output_tokens": requested_tokens,
            "memory_gate_primary": "peak_phys_footprint_bytes (proc_pid_rusage RUSAGE_INFO_V4)",
            "max_phys_footprint_bytes": max_phys_footprint_bytes,
            "antfly_max_phys_footprint_bytes": antfly_max_phys_footprint_bytes,
            "turbo_max_phys_footprint_bytes": turbo_max_phys_footprint_bytes,
            "max_rss_bytes": max_rss_bytes,
            "rss_role": "secondary telemetry" if max_rss_bytes == 0 else "secondary gate",
            "max_decode_cv": max_decode_cv,
            "footprint_notes": sorted(footprint_notes),
            "ttft_definition": "fresh-process wall time minus reported decode time",
            "matched_output_sha256": antfly_output,
            "output_contract": "NFC text, normalized newlines, surrounding whitespace removed",
        },
        "engines": engines,
        "samples": [asdict(sample) for sample in samples],
    }


def evaluate_performance_gate(
    summary: dict,
    min_antfly_decode_ratio: float,
    require_antfly_win: bool,
) -> dict:
    """Return a serializable, explicit pass/fail record for speed claims."""
    antfly = summary["engines"]["antfly"]
    turbo = summary["engines"]["turbo"]
    failures = []
    if antfly["decode_ratio_vs_turbo"] < min_antfly_decode_ratio:
        failures.append(
            f"Antfly decode ratio {antfly['decode_ratio_vs_turbo']:.3f} is below "
            f"{min_antfly_decode_ratio:.3f}"
        )
    if require_antfly_win and (
        antfly["decode_ratio_vs_turbo"] <= 1.0
        or antfly["median_ttft_proxy_ms"] >= turbo["median_ttft_proxy_ms"]
    ):
        failures.append("Antfly did not beat TurboFieldfare on decode and TTFT proxy")
    return {
        "passed": not failures,
        "min_antfly_decode_ratio": min_antfly_decode_ratio,
        "require_antfly_win": require_antfly_win,
        "failures": failures,
    }


def command_lines(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "antfly": [
            args.antfly_bin,
            "generate",
            args.antfly_model,
            args.prompt,
            "--backend",
            "metal",
            "--memory-profile",
            "2gbs",
            "--memory-budget-mb",
            str(args.antfly_memory_budget_mb),
            "--compact-device-routing",
            args.antfly_device_routing,
            "--raw-prompt",
            "--max-tokens",
            str(args.tokens),
            "--temperature",
            "0",
            "--seed",
            str(args.seed),
            "--ignore-eos",
            "--print-token-ids",
            "--print-prompt-token-ids",
            "--print-timing",
        ],
        "turbo": [
            args.turbo_bin,
            "--model",
            args.turbo_model,
            "--prompt",
            args.prompt,
            "--max-new",
            str(args.tokens),
            "--temperature",
            "0",
            "--seed",
            str(args.seed),
        ],
    }


def existing_path(value: str, label: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise BenchmarkError(f"missing {label}: {path}")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antfly-bin", required=True)
    parser.add_argument("--antfly-model", required=True)
    parser.add_argument("--turbo-bin", required=True)
    parser.add_argument("--turbo-model", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lane",
        choices=("compact-2g", "full-residency"),
        default="compact-2g",
        help="named benchmark contract recorded in summary.json",
    )
    parser.add_argument(
        "--antfly-memory-budget-mb",
        type=int,
        default=None,
        help="defaults to 2048 for compact-2g and 16384 for full-residency",
    )
    parser.add_argument(
        "--antfly-device-routing",
        choices=("off", "partial", "full", "auto"),
        default=None,
        help="defaults to off for compact-2g and full for full-residency",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-phys-footprint-bytes",
        type=int,
        default=DEFAULT_MAX_PHYS_FOOTPRINT_BYTES,
        help="primary memory ceiling on sampled phys_footprint",
    )
    parser.add_argument(
        "--max-rss-bytes",
        type=int,
        default=0,
        help="secondary RSS ceiling; 0 records RSS without gating",
    )
    parser.add_argument(
        "--antfly-max-phys-footprint-bytes",
        type=int,
        default=None,
        help="Antfly-specific primary memory ceiling (overrides the shared ceiling)",
    )
    parser.add_argument(
        "--turbo-max-phys-footprint-bytes",
        type=int,
        default=None,
        help="Turbo-specific primary memory ceiling (overrides the shared ceiling)",
    )
    parser.add_argument(
        "--allow-missing-footprint",
        action="store_true",
        help="permit runs without proc_pid_rusage telemetry (RSS-only reporting)",
    )
    parser.add_argument(
        "--max-decode-cv",
        type=float,
        default=DEFAULT_MAX_DECODE_CV,
        help="fail when either engine's decode tok/s CV exceeds this (<=0 disables)",
    )
    parser.add_argument(
        "--min-antfly-decode-ratio",
        type=float,
        default=0.95,
        help="fail when Antfly is more than 5%% slower by median decode throughput",
    )
    parser.add_argument("--require-antfly-win", action="store_true")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)
    if args.tokens < 2 or args.warmups < 0 or args.repeats < 2:
        parser.error("--tokens >= 2, --warmups >= 0, and --repeats >= 2 are required")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.antfly_memory_budget_mb is None:
        args.antfly_memory_budget_mb = 2048 if args.lane == "compact-2g" else 16384
    if args.antfly_device_routing is None:
        args.antfly_device_routing = "off" if args.lane == "compact-2g" else "full"
    if args.lane == "full-residency" and args.antfly_device_routing != "full":
        parser.error("full-residency lane requires --antfly-device-routing full")
    if args.antfly_memory_budget_mb < 2048:
        parser.error("--antfly-memory-budget-mb must be at least 2048")
    if args.antfly_max_phys_footprint_bytes is None:
        args.antfly_max_phys_footprint_bytes = args.antfly_memory_budget_mb * 1024 * 1024
    try:
        args.antfly_bin = existing_path(args.antfly_bin, "Antfly binary")
        args.antfly_model = existing_path(args.antfly_model, "Antfly GGUF")
        args.turbo_bin = existing_path(args.turbo_bin, "TurboFieldfare binary")
        args.turbo_model = existing_path(args.turbo_model, "TurboFieldfare .gturbo model")
        out_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else Path("/tmp") / time.strftime("antfly-a4b-turbo-%Y%m%dT%H%M%SZ", time.gmtime())
        )
        out_dir.mkdir(parents=True, exist_ok=False)
        commands = command_lines(args)

        for warmup in range(args.warmups):
            for engine in ("antfly", "turbo"):
                run_sample(engine, -(warmup + 1), commands[engine], out_dir, args.timeout)

        samples: list[Sample] = []
        for iteration in range(args.repeats):
            order = ("antfly", "turbo") if iteration % 2 == 0 else ("turbo", "antfly")
            for engine in order:
                samples.append(
                    run_sample(engine, iteration, commands[engine], out_dir, args.timeout)
                )
        summary = summarize(
            samples,
            args.tokens,
            max_phys_footprint_bytes=args.max_phys_footprint_bytes,
            antfly_max_phys_footprint_bytes=args.antfly_max_phys_footprint_bytes,
            turbo_max_phys_footprint_bytes=args.turbo_max_phys_footprint_bytes,
            max_rss_bytes=args.max_rss_bytes,
            max_decode_cv=args.max_decode_cv,
            allow_missing_footprint=args.allow_missing_footprint,
        )
        summary["commands"] = commands
        summary["lane"] = args.lane
        summary["seed"] = args.seed
        antfly = summary["engines"]["antfly"]
        turbo = summary["engines"]["turbo"]
        performance_gate = evaluate_performance_gate(
            summary,
            args.min_antfly_decode_ratio,
            args.require_antfly_win,
        )
        summary["performance_gate"] = performance_gate
        summary["passed"] = performance_gate["passed"]
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(
            f"antfly={antfly['median_decode_tok_per_s']:.3f} tok/s "
            f"turbo={turbo['median_decode_tok_per_s']:.3f} tok/s "
            f"ratio={antfly['decode_ratio_vs_turbo']:.3f} "
            f"antfly_ttft_proxy={antfly['median_ttft_proxy_ms']:.0f}ms "
            f"turbo_ttft_proxy={turbo['median_ttft_proxy_ms']:.0f}ms "
            f"antfly_footprint_mb={antfly['peak_phys_footprint_bytes'] / 1e6:.0f} "
            f"turbo_footprint_mb={turbo['peak_phys_footprint_bytes'] / 1e6:.0f} "
            f"summary={summary_path}"
        )
        if performance_gate["failures"]:
            raise BenchmarkError("; ".join(performance_gate["failures"]))
        return 0
    except (BenchmarkError, subprocess.TimeoutExpired) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
