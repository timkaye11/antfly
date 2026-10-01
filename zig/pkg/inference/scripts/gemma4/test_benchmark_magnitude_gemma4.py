#!/usr/bin/env python3
"""Contract tests for client timing and the overall streaming deadline."""
import json
import io
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_magnitude_gemma4 as bench


class HeartbeatStream:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        while True:
            time.sleep(0.01)
            yield b": heartbeat\n"


class BenchmarkContractTests(unittest.TestCase):
    def test_http_error_body_cannot_escape_deadline_and_receipt_survives(self):
        class DrippingErrorBody(io.BytesIO):
            def read(self, *_):
                while True:
                    time.sleep(0.01)

        body = DrippingErrorBody()
        error = bench.urllib.error.HTTPError("http://localhost/generate", 503, "busy", {}, body)
        args = SimpleNamespace(max_tokens=None, temperature=0.8, top_p=0.95, top_k=0, request_timeout=0.05)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.urllib.request, "urlopen", side_effect=error):
            output = Path(directory)
            start = time.monotonic()
            result = bench.request("http://localhost/generate", "model", bench.cases()[0], output, args)
            self.assertLess(time.monotonic() - start, 1)
            self.assertIn("overall request deadline", result["error"])
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertTrue(body.closed)

    def test_http_error_body_read_failure_is_recorded(self):
        class BrokenErrorBody(io.BytesIO):
            def read(self, *_):
                raise OSError("error body disconnected")

        body = BrokenErrorBody()
        error = bench.urllib.error.HTTPError("http://localhost/generate", 503, "busy", {}, body)
        args = SimpleNamespace(max_tokens=None, temperature=0.8, top_p=0.95, top_k=0, request_timeout=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.urllib.request, "urlopen", side_effect=error):
            output = Path(directory)
            result = bench.request("http://localhost/generate", "model", bench.cases()[0], output, args)
            self.assertEqual(result["error"], "error body disconnected")
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertTrue(body.closed)

    def test_http_error_body_is_bounded_and_decoded_safely(self):
        body = io.BytesIO(b"busy\xff" + b"x" * 8192)
        error = bench.urllib.error.HTTPError("http://localhost/generate", 503, "busy", {}, body)
        args = SimpleNamespace(max_tokens=None, temperature=0.8, top_p=0.95, top_k=0, request_timeout=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.urllib.request, "urlopen", side_effect=error):
            result = bench.request("http://localhost/generate", "model", bench.cases()[0], Path(directory), args)
            self.assertTrue(result["error"].startswith("HTTP 503: busy\ufffd"))
            self.assertEqual(len(result["error"]), len("HTTP 503: ") + 4096)
            self.assertTrue(body.closed)

    def test_heartbeat_cannot_extend_request_deadline_and_partial_receipt_survives(self):
        args = SimpleNamespace(max_tokens=None, temperature=0.8, top_p=0.95, top_k=0, request_timeout=0.05)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.urllib.request, "urlopen", return_value=HeartbeatStream()):
            output = Path(directory)
            start = time.monotonic()
            result = bench.request("http://localhost/generate", "model", bench.cases()[0], output, args)
            self.assertLess(time.monotonic() - start, 1)
            self.assertIn("overall request deadline", result["error"])
            self.assertIsNone(result["decode_tokens_per_s"])
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertEqual(json.loads((output / "events.json").read_text()), [])

    def test_rate_excludes_first_token_and_uses_last_content_timestamp(self):
        result = bench.timing_result(10, 12, 16, 17, 9)
        self.assertEqual(result["ttft_s"], 2)
        self.assertEqual(result["total_s"], 7)
        self.assertEqual(result["decode_tokens_per_s"], 2)
        self.assertIsNone(bench.timing_result(10, None, None, 11, None)["decode_tokens_per_s"])

    def test_truncated_stream_cannot_be_reported_as_a_successful_benchmark(self):
        class Truncated(HeartbeatStream):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n'
        args = SimpleNamespace(max_tokens=None, temperature=0.8, top_p=0.95, top_k=0, request_timeout=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.urllib.request, "urlopen", return_value=Truncated()):
            result = bench.request("http://localhost/generate", "model", bench.cases()[0], Path(directory), args)
            self.assertIn("token counts", result["error"])
            self.assertIsNone(result["decode_tokens_per_s"])

    def test_named_sse_error_reports_admission_reason_without_parsing_it_as_json(self):
        class ErrorStream(HeartbeatStream):
            def __iter__(self):
                yield b"event: error\n"
                yield b"data: insufficient inference capacity is currently available\n"
        args = SimpleNamespace(max_tokens=None, temperature=0.8, top_p=0.95, top_k=0, request_timeout=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.urllib.request, "urlopen", return_value=ErrorStream()):
            result = bench.request("http://localhost/generate", "model", bench.cases()[0], Path(directory), args)
            self.assertEqual(result["error"], "insufficient inference capacity is currently available")


if __name__ == "__main__":
    unittest.main()
