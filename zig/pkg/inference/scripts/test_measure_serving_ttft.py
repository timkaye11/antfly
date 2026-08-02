#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("measure_serving_ttft.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("measure_serving_ttft", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _case(*, text="same output", cached=0, ttft=100.0, prompt=100):
    return {
        "text": text,
        "ttft_ms": ttft,
        "total_ms": 200.0,
        "cached_prompt_tokens": cached,
        "prompt_tokens": prompt,
        "completion_tokens": 4,
    }


def _result(*, experimental=False):
    cached = 96 if experimental else 0
    return {
        "cases": {
            "cold": _case(cached=0),
            "replay": _case(cached=cached),
            "strict_extension": _case(cached=cached, prompt=120),
            "second_key": _case(cached=0, text="second output"),
            "strict_extension_interleaved": _case(cached=cached, prompt=120),
            "nonstream_replay": _case(cached=cached, ttft=None),
        },
        "metrics_boot": {"block_hash_hits": 0, "live_bytes": 0},
        "metrics_final": {
            "block_hash_hits": 4 if experimental else 0,
            "live_bytes": 4096 if experimental else 0,
        },
        "generate_timing_ms": [],
    }


class MeasureServingTtftTest(unittest.TestCase):
    def _run(self, result, *, experimental=False, max_ttft_ms=0.0):
        args = SimpleNamespace(
            experimental_compact_prompt_cache_reuse=experimental,
            max_ttft_ms=max_ttft_ms,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            MOD._assert_and_report(result, args)

    def test_default_cold_gate_requires_zero_reuse_and_deterministic_output(self):
        result = _result(experimental=False)
        self._run(result)
        self.assertTrue(result["passed"])
        self.assertEqual("replay-cold", result["gate_case"])

        result = _result(experimental=False)
        result["cases"]["nonstream_replay"]["text"] = "different"
        self._run(result)
        self.assertFalse(result["passed"])

        result = _result(experimental=False)
        result["cases"]["replay"]["cached_prompt_tokens"] = None
        self._run(result)
        self.assertFalse(result["passed"])

    def test_experimental_gate_requires_cache_evidence(self):
        result = _result(experimental=True)
        self._run(result, experimental=True, max_ttft_ms=150.0)
        self.assertTrue(result["passed"])
        self.assertEqual("strict-extension", result["gate_case"])

        result = _result(experimental=True)
        result["cases"]["replay"]["cached_prompt_tokens"] = 0
        self._run(result, experimental=True)
        self.assertFalse(result["passed"])

        result = _result(experimental=True)
        result["cases"]["second_key"]["cached_prompt_tokens"] = None
        self._run(result, experimental=True)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
