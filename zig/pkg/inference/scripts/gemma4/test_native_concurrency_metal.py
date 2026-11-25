from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest import mock

import qualify_native_concurrency as concurrency


class Response:
    status = 503

    def read(self, _limit):
        return json.dumps({"error": "MODEL_RESOURCE_BUSY"}).encode()


class Connection:
    instances = []

    def __init__(self, *args, **kwargs):
        self.request_args = None
        self.closed = False
        self.instances.append(self)

    def request(self, *args):
        self.request_args = args

    def getresponse(self):
        return Response()

    def close(self):
        self.closed = True


class MetalConcurrencyContractTests(unittest.TestCase):
    def test_prepare_competitor_propagates_requested_encoding(self):
        args = SimpleNamespace(port=1234, encoding="rotated4",
                               checkpoint=Path("unused"))
        with mock.patch.object(concurrency.http.client, "HTTPConnection", Connection):
            result = concurrency.competing_request(args, {"model": "m"}, "prepare")
        request = Connection.instances[-1].request_args
        self.assertEqual(request[1], "/ai/v1/generate/context/checkpoint")
        self.assertEqual(request[3]["X-Antfly-KV-Encoding"], "rotated4")
        self.assertEqual(result["status"], 503)
        self.assertTrue(Connection.instances[-1].closed)

    def test_admission_metrics_are_strictly_typed(self):
        row = concurrency.parse_admission_metrics(
            "antfly_admission_inference_capacity_units 48\n"
            "antfly_admission_inference_in_flight_units 1\n"
            "antfly_admission_inference_available_units 47\n"
            "antfly_admission_inference_in_flight_requests 1\n"
        )
        self.assertEqual(row["capacity_units"], 48)
        with self.assertRaises(AssertionError):
            concurrency.parse_admission_metrics(
                "antfly_admission_inference_capacity_units 48\n"
                "antfly_admission_inference_in_flight_units 1\n"
            )

    def test_metal_competitor_uses_a_valid_native_prepare_route(self):
        args = SimpleNamespace(port=1234, encoding="fp8", backend="metal",
                               server_kind="metal-test", qualification_minimum_savings=1.5)
        body = {"model": "m", "max_tokens": 16}
        with mock.patch.object(concurrency.http.client, "HTTPConnection", Connection):
            result = concurrency.competing_request(args, body, "prepare")
        request = Connection.instances[-1].request_args
        self.assertEqual(json.loads(request[2]), dict(body, mode="eager"))
        self.assertNotIn("mode", body)
        self.assertEqual(request[3], concurrency.prepare_headers(args))
        self.assertEqual(result["status"], 503)

    def test_held_upload_fixture_selects_metal_eager_without_mutating_ordinary_request(self):
        original = {"model": "m", "cache_dtype": "f16",
                    "context_management": {"mode": "reject"}, "max_tokens": 16}
        expected = {"model": "m", "max_tokens": 16, "mode": "eager"}
        self.assertEqual(concurrency.native_fixture_body(SimpleNamespace(backend="metal"), original),
                         expected)
        self.assertEqual(original["cache_dtype"], "f16")
        self.assertIn("context_management", original)
        self.assertNotIn("mode", original)
        expected.pop("mode")
        self.assertEqual(concurrency.native_fixture_body(SimpleNamespace(backend="cuda"), original),
                         expected)


if __name__ == "__main__":
    unittest.main()
