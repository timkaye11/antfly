#!/usr/bin/env python3

import contextlib
import ctypes
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_gemma4_a4b_turbo.py")
SPEC = importlib.util.spec_from_file_location("benchmark_gemma4_a4b_turbo", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


ANTFLY_STDOUT = """\
prompt_token_ids: 2 9259
generate_timing_ms: prompt_format=0 tokenize=1 runtime_prepare=0 prefill=300 decode=700 text_decode=0 total=1001
token_ids: 1 2 3 4
timing_ms: load_model=2000 prompt_prep=1 scheduler=0 backend_setup=4 decode_setup=0 generate=1002 total=3007
decode_tok_per_s=5.714
"""


def _sample(
    engine: str = "antfly",
    iteration: int = 0,
    rate: float = 5.7,
    rss: int = 1_900_000_000,
    footprint: int = 1_900_000_000,
    output_sha256: str = "x",
) -> "MOD.Sample":
    return MOD.Sample(
        engine,
        iteration,
        2,
        4,
        700.0,
        rate,
        2300.0,
        3000.0,
        rss,
        footprint,
        output_sha256,
        f"{engine}-{iteration}.log",
    )


class BenchmarkGemma4A4BTurboTest(unittest.TestCase):
    def test_parses_antfly_and_turbo_samples_from_legacy_time_stderr(self) -> None:
        # Live runs no longer use /usr/bin/time -l, but time-style stderr must
        # remain parseable when harness measurements are not supplied.
        time_stderr = """\
        3.10 real         1.00 user         0.20 sys
          1900000000  maximum resident set size
"""
        antfly = MOD.parse_antfly(ANTFLY_STDOUT, time_stderr, 0, Path("a.log"))
        self.assertEqual(2, antfly.prompt_tokens)
        self.assertEqual(4, antfly.output_tokens)
        self.assertEqual(2307.0, antfly.ttft_proxy_ms)
        self.assertEqual(3100.0, antfly.wall_ms)
        self.assertEqual(1_900_000_000, antfly.max_rss_bytes)
        self.assertEqual(0, antfly.peak_phys_footprint_bytes)

        turbo_stderr = """\
[stop=length prefill=2tok new=4tok decode=0.80s tok/s=5.000]
        2.90 real         1.00 user         0.20 sys
          1800000000  maximum resident set size
"""
        turbo = MOD.parse_turbo("text", turbo_stderr, 0, Path("t.log"))
        self.assertEqual(2, turbo.prompt_tokens)
        self.assertEqual(4, turbo.output_tokens)
        self.assertEqual(2100.0, turbo.ttft_proxy_ms)

    def test_parses_with_harness_measurements_and_no_time_output(self) -> None:
        antfly = MOD.parse_antfly(
            ANTFLY_STDOUT,
            "engine noise, no time -l lines\n",
            1,
            Path("a.log"),
            wall_ms=3100.0,
            max_rss_bytes=1_850_000_000,
            peak_phys_footprint_bytes=2_050_000_000,
        )
        self.assertEqual(3100.0, antfly.wall_ms)
        self.assertEqual(1_850_000_000, antfly.max_rss_bytes)
        self.assertEqual(2_050_000_000, antfly.peak_phys_footprint_bytes)
        self.assertEqual(2307.0, antfly.ttft_proxy_ms)

        turbo = MOD.parse_turbo(
            "[stop=length prefill=2tok new=4tok decode=0.80s tok/s=5.000]",
            "",
            1,
            Path("t.log"),
            wall_ms=2900.0,
            max_rss_bytes=1_800_000_000,
            peak_phys_footprint_bytes=1_950_000_000,
        )
        self.assertEqual(2100.0, turbo.ttft_proxy_ms)
        self.assertEqual(1_950_000_000, turbo.peak_phys_footprint_bytes)

    def test_missing_wall_or_rss_without_harness_values_fails(self) -> None:
        with self.assertRaises(MOD.BenchmarkError):
            MOD.parse_antfly(ANTFLY_STDOUT, "no time output here\n", 0, Path("a.log"))
        with self.assertRaises(MOD.BenchmarkError):
            MOD.parse_turbo(
                "[stop=length prefill=2tok new=4tok decode=0.80s tok/s=5.000]",
                "no time output here\n",
                0,
                Path("t.log"),
                wall_ms=2900.0,
            )

    def test_summary_enforces_token_and_footprint_contract(self) -> None:
        samples = [_sample("antfly", rate=5.7), _sample("turbo", rate=5.0)]
        summary = MOD.summarize(samples, 4)
        self.assertAlmostEqual(1.14, summary["engines"]["antfly"]["decode_ratio_vs_turbo"])
        self.assertEqual(
            1_900_000_000, summary["engines"]["antfly"]["peak_phys_footprint_bytes"]
        )
        self.assertEqual(
            MOD.DEFAULT_MAX_PHYS_FOOTPRINT_BYTES,
            summary["contract"]["max_phys_footprint_bytes"],
        )
        with self.assertRaises(MOD.BenchmarkError):
            MOD.summarize(samples, 5)

        over_ceiling = [
            _sample("antfly", footprint=2_200_000_000),
            _sample("turbo", rate=5.0),
        ]
        with self.assertRaisesRegex(MOD.BenchmarkError, "phys_footprint"):
            MOD.summarize(over_ceiling, 4)
        # A custom ceiling admits the same sample.
        summary = MOD.summarize(over_ceiling, 4, max_phys_footprint_bytes=2_300_000_000)
        self.assertEqual(
            2_200_000_000, summary["engines"]["antfly"]["peak_phys_footprint_bytes"]
        )

    def test_missing_footprint_requires_allow_flag_and_records_rss_only_note(self) -> None:
        samples = [
            _sample("antfly", footprint=0),
            _sample("turbo", rate=5.0, footprint=0),
        ]
        with self.assertRaisesRegex(MOD.BenchmarkError, "allow-missing-footprint"):
            MOD.summarize(samples, 4)
        summary = MOD.summarize(samples, 4, allow_missing_footprint=True)
        notes = summary["contract"]["footprint_notes"]
        self.assertEqual(2, len(notes))
        self.assertTrue(all("RSS-only" in note for note in notes))
        self.assertEqual(0, summary["engines"]["antfly"]["peak_phys_footprint_bytes"])

    def test_rss_is_secondary_and_gates_only_when_requested(self) -> None:
        samples = [
            _sample("antfly", rss=2_100_000_000),
            _sample("turbo", rate=5.0, rss=1_800_000_000),
        ]
        # Default --max-rss-bytes 0 reports without gating.
        summary = MOD.summarize(samples, 4)
        self.assertEqual(2_100_000_000, summary["engines"]["antfly"]["peak_rss_bytes"])
        self.assertEqual("secondary telemetry", summary["contract"]["rss_role"])
        with self.assertRaisesRegex(MOD.BenchmarkError, "RSS"):
            MOD.summarize(samples, 4, max_rss_bytes=2_000_000_000)

    def test_decode_cv_gate(self) -> None:
        samples = [
            _sample("antfly", iteration=0, rate=5.0),
            _sample("antfly", iteration=1, rate=8.0),
            _sample("turbo", iteration=0, rate=5.0),
            _sample("turbo", iteration=1, rate=5.0),
        ]
        with self.assertRaisesRegex(MOD.BenchmarkError, "max-decode-cv"):
            MOD.summarize(samples, 4)
        summary = MOD.summarize(samples, 4, max_decode_cv=1.0)
        self.assertGreater(summary["engines"]["antfly"]["decode_tok_per_s_cv"], 0.03)
        # <= 0 disables the gate.
        summary = MOD.summarize(samples, 4, max_decode_cv=0.0)
        self.assertIn("engines", summary)

    def test_rusage_info_v4_struct_layout(self) -> None:
        self.assertEqual(4, MOD.RUSAGE_INFO_V4)
        self.assertEqual(296, ctypes.sizeof(MOD.RUsageInfoV4))
        self.assertEqual(72, MOD.RUsageInfoV4.ri_phys_footprint.offset)
        self.assertEqual(240, MOD.RUsageInfoV4.ri_lifetime_max_phys_footprint.offset)

    @unittest.skipUnless(sys.platform == "darwin", "proc_pid_rusage is macOS-only")
    def test_read_phys_footprint_of_this_process(self) -> None:
        result = MOD.read_phys_footprint(os.getpid())
        self.assertIsNotNone(result)
        phys_footprint, lifetime_max = result
        self.assertGreater(phys_footprint, 0)
        self.assertGreaterEqual(lifetime_max, phys_footprint)

    def test_run_sample_measures_stub_engine_without_time_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            stub = out_dir / "engine.sh"
            stub.write_text(
                "#!/bin/sh\n"
                + "".join(f"echo '{line}'\n" for line in ANTFLY_STDOUT.splitlines())
                + "sleep 0.12\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
            sample = MOD.run_sample("antfly", 0, ["/bin/sh", str(stub)], out_dir, 30.0)
            self.assertEqual(4, sample.output_tokens)
            self.assertEqual(5.714, sample.decode_tok_per_s)
            self.assertGreaterEqual(sample.wall_ms, 100.0)
            self.assertGreater(sample.max_rss_bytes, 0)
            if sys.platform == "darwin":
                self.assertGreater(sample.peak_phys_footprint_bytes, 0)
            log_text = Path(sample.log_path).read_text(encoding="utf-8")
            self.assertIn("peak_phys_footprint_bytes=", log_text)
            self.assertNotIn("/usr/bin/time", log_text)

    def test_run_sample_fails_closed_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            stub = out_dir / "broken.sh"
            stub.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
            stub.chmod(0o755)
            with self.assertRaisesRegex(MOD.BenchmarkError, "exited 3"):
                MOD.run_sample("antfly", 0, ["/bin/sh", str(stub)], out_dir, 30.0)

    def test_cli_fails_closed_when_turbo_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir) / "antfly"
            existing.write_text("")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = MOD.main(
                    [
                        "--antfly-bin",
                        str(existing),
                        "--antfly-model",
                        str(existing),
                        "--turbo-bin",
                        str(Path(temp_dir) / "missing"),
                        "--turbo-model",
                        str(Path(temp_dir) / "missing-model"),
                    ]
                )
            self.assertEqual(1, rc)
            self.assertIn("missing TurboFieldfare binary", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
