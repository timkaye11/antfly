#!/usr/bin/env python3
"""Fail-closed memory watchdog for Gemma4 A4B benchmark processes.

The command runs in its own process group. On a footprint, memory-pressure,
timeout, or signal breach the watchdog terminates the complete group, waits a
short grace period, and then kills any survivors. Memory units are explicit:
``--kill-bytes`` is exact, ``--kill-gib`` is binary GiB, and the deprecated
``--kill-gb`` compatibility option is decimal GB.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


RUSAGE_INFO_V4 = 4
MEMORYSTATUS_KILL_LEVEL = 25
POLL_SECONDS = 0.1
DEFAULT_KILL_GIB = 2.20
DEFAULT_LOG = Path("/tmp/guarded-a4b-out.log")


class RUsageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
            "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy",
            "ri_interval_max_phys_footprint", "ri_runnable_time",
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
    if _LIBPROC is None:
        return None
    info = RUsageInfoV4()
    if _LIBPROC.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint), int(info.ri_lifetime_max_phys_footprint)


def memorystatus_level() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0:
            return None
        level = int(result.stdout.strip())
        return level if 0 <= level <= 100 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    units = parser.add_mutually_exclusive_group()
    units.add_argument("--kill-bytes", type=int, help="exact phys_footprint ceiling")
    units.add_argument("--kill-gib", type=float, help="phys_footprint ceiling in binary GiB")
    units.add_argument(
        "--kill-gb",
        type=float,
        help="deprecated: phys_footprint ceiling in decimal GB",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--terminate-grace-s", type=float, default=2.0)
    parser.add_argument(
        "--allow-unknown-memory-pressure",
        action="store_true",
        help="continue when kern.memorystatus_level cannot be read",
    )
    parser.add_argument("timeout_s", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command to run")
    if args.timeout_s <= 0:
        parser.error("timeout_s must be positive")
    if args.terminate_grace_s < 0:
        parser.error("--terminate-grace-s must be non-negative")
    if args.kill_bytes is not None:
        threshold = args.kill_bytes
    elif args.kill_gib is not None:
        threshold = int(args.kill_gib * (1024**3))
    elif args.kill_gb is not None:
        threshold = int(args.kill_gb * 1_000_000_000)
    else:
        threshold = int(DEFAULT_KILL_GIB * (1024**3))
    if threshold <= 0:
        parser.error("memory ceiling must be positive")
    if not args.log.parent.is_dir():
        parser.error(f"log directory does not exist: {args.log.parent}")
    args.kill_threshold_bytes = threshold
    return args


def terminate_process_group(child: subprocess.Popen[bytes], grace_s: float) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    peak = 0
    breach: str | None = None
    received_signal: int | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    child: subprocess.Popen[bytes] | None = None
    rc = 1
    try:
        with args.log.open("wb") as sink:
            child = subprocess.Popen(
                args.command,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                polled = child.poll()
                if polled is not None:
                    rc = polled
                    break
                sampled = read_phys_footprint(child.pid)
                if sampled is not None:
                    peak = max(peak, sampled[0], sampled[1])
                    if sampled[0] > args.kill_threshold_bytes:
                        breach = (
                            f"footprint {sampled[0]} bytes exceeds "
                            f"{args.kill_threshold_bytes} bytes"
                        )
                pressure = memorystatus_level()
                if breach is None and pressure is None and not args.allow_unknown_memory_pressure:
                    breach = "memorystatus_level unavailable"
                elif breach is None and pressure is not None and pressure < MEMORYSTATUS_KILL_LEVEL:
                    breach = f"memorystatus_level {pressure}% below {MEMORYSTATUS_KILL_LEVEL}%"
                if breach is None and time.monotonic() - started > args.timeout_s:
                    breach = f"timeout {args.timeout_s}s"
                if breach is None and received_signal is not None:
                    breach = f"watchdog received {signal.Signals(received_signal).name}"
                if breach is not None:
                    terminate_process_group(child, args.terminate_grace_s)
                    rc = child.returncode if child.returncode is not None else -signal.SIGKILL
                    break
                time.sleep(POLL_SECONDS)
    finally:
        if child is not None and child.poll() is None:
            terminate_process_group(child, args.terminate_grace_s)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    elapsed = time.monotonic() - started
    print(
        f"watchdog: rc={rc} elapsed={elapsed:.1f}s "
        f"peak_phys_footprint_bytes={peak} "
        f"ceiling_bytes={args.kill_threshold_bytes} breach={breach}"
    )
    print(f"watchdog: child output in {args.log}")
    return 0 if rc == 0 and breach is None else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
