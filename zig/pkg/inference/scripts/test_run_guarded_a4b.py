#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("run_guarded_a4b.py")
SPEC = importlib.util.spec_from_file_location("run_guarded_a4b", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class GuardedA4BTest(unittest.TestCase):
    def test_memory_units_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "out.log"
            exact = MOD.parse_args(["--kill-bytes", "17", "--log", str(log), "1", "true"])
            binary = MOD.parse_args(["--kill-gib", "2", "--log", str(log), "1", "true"])
            decimal = MOD.parse_args(["--kill-gb", "2", "--log", str(log), "1", "true"])
        self.assertEqual(17, exact.kill_threshold_bytes)
        self.assertEqual(2 * 1024**3, binary.kill_threshold_bytes)
        self.assertEqual(2_000_000_000, decimal.kill_threshold_bytes)

    def test_invalid_limits_fail_at_parse_time(self) -> None:
        with self.assertRaises(SystemExit):
            MOD.parse_args(["--kill-bytes", "0", "1", "true"])
        with self.assertRaises(SystemExit):
            MOD.parse_args(["1"])

    def test_unknown_memory_pressure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            MOD, "memorystatus_level", return_value=None
        ), mock.patch.object(MOD, "read_phys_footprint", return_value=(1, 1)):
            rc = MOD.main(
                [
                    "--kill-bytes", "1024", "--terminate-grace-s", "2",
                    "--log", str(Path(temp_dir) / "out.log"), "10",
                    sys.executable, "-c", "import time; time.sleep(10)",
                ]
            )
        self.assertEqual(1, rc)

    def test_allow_unknown_pressure_runs_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            MOD, "memorystatus_level", return_value=None
        ), mock.patch.object(MOD, "read_phys_footprint", return_value=(1, 1)):
            rc = MOD.main(
                [
                    "--kill-bytes", "1024", "--allow-unknown-memory-pressure",
                    "--log", str(Path(temp_dir) / "out.log"), "10",
                    sys.executable, "-c", "print('ok')",
                ]
            )
        self.assertEqual(0, rc)


if __name__ == "__main__":
    unittest.main()
