#!/usr/bin/env python3
"""External memory watchdog for gemma4 A4B compact runs on small machines.

Launches the given command, samples the child's phys_footprint (via
``proc_pid_rusage(RUSAGE_INFO_V4)`` from libproc) and the system
memorystatus level every 100 ms, and SIGKILLs the child if:

  - the child's phys_footprint exceeds ``--kill-gb`` (default 2.20 GB), or
  - available system memory falls below ~25% (``kern.memorystatus_level``,
    the signal macOS memory pressure acts on -- free-page counts lie), or
  - the hard timeout expires.

Prints a footprint/timing summary either way and mirrors the child's output
to a log file. Stdlib-only; macOS-only (elsewhere the footprint sampler is
disabled and only the timeout guard applies).

Usage:
    scripts/run_guarded_a4b.py [--kill-gb GB] [--log PATH] TIMEOUT_S CMD...

Example:
    scripts/run_guarded_a4b.py --kill-gb 3.2 600 \
        zig-out/bin/antfly-inference generate <model> "Hi" --max-tokens 64
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

RUSAGE_INFO_V4 = 4
MEMORYSTATUS_KILL_LEVEL = 25  # percent of memory available
POLL_SECONDS = 0.1


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


def memorystatus_level() -> int:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(out.stdout.strip())
    except Exception:
        return 100


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kill-gb",
        type=float,
        default=2.20,
        help="SIGKILL the child when its phys_footprint exceeds this many GB (default: 2.20)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/tmp/guarded-a4b-out.log"),
        help="file receiving the child's combined stdout/stderr (default: /tmp/guarded-a4b-out.log)",
    )
    parser.add_argument("timeout_s", type=float, help="hard wall-clock timeout in seconds")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to launch (everything after the timeout)",
    )
    args = parser.parse_args(argv)
    if not args.command:
        parser.error("missing command to run")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    footprint_kill_bytes = int(args.kill_gb * 1_000_000_000)
    started = time.monotonic()
    with args.log.open("w") as sink:
        child = subprocess.Popen(args.command, stdout=sink, stderr=subprocess.STDOUT)
        peak = 0
        breach = None
        while True:
            rc = child.poll()
            if rc is not None:
                break
            sampled = read_phys_footprint(child.pid)
            footprint = sampled[0] if sampled else 0
            peak = max(peak, footprint, sampled[1] if sampled else 0)
            if footprint > footprint_kill_bytes:
                breach = f"footprint {footprint / 1e9:.2f} GB > {args.kill_gb:.2f} GB"
            elif memorystatus_level() < MEMORYSTATUS_KILL_LEVEL:
                breach = f"memorystatus_level below {MEMORYSTATUS_KILL_LEVEL}%"
            elif time.monotonic() - started > args.timeout_s:
                breach = f"timeout {args.timeout_s}s"
            if breach:
                child.kill()
                child.wait()
                rc = -9
                break
            time.sleep(POLL_SECONDS)
    elapsed = time.monotonic() - started
    print(
        f"watchdog: rc={rc} elapsed={elapsed:.1f}s "
        f"peak_phys_footprint={peak / 1e9:.3f}GB breach={breach}"
    )
    print(f"watchdog: child output in {args.log}")
    return 0 if rc == 0 and breach is None else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
