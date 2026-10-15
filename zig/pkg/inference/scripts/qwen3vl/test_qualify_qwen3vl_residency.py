#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
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

import io
import json
import unittest
from unittest.mock import patch
import qualify_qwen3vl_residency as gate


class ResidencyGateTests(unittest.TestCase):
    def records(self):
        return [
            dict(
                phase=name,
                tensor_download_bytes=0,
                tensor_device_readbacks=0,
                frame_submits=1,
            )
            for name in ("multimodal_prepare", "prefill", "decode")
        ]

    def test_missing_or_large_readbacks_fail(self):
        records = self.records()
        gate.check_residency(records)
        with self.assertRaisesRegex(ValueError, "missing"):
            gate.check_residency(records[:2])
        for index in range(3):
            records = self.records()
            records[index]["tensor_download_bytes"] = 2 * 1024 * 1024
            with self.assertRaisesRegex(ValueError, "readback"):
                gate.check_residency(records)

    def test_shared_memory_host_accesses_still_fail(self):
        records = self.records()
        records[0]["tensor_device_readbacks"] = 1
        with self.assertRaisesRegex(ValueError, "readback"):
            gate.check_residency(records)

    def test_text_logits_are_bounded_separately_from_kv_readbacks(self):
        records = self.records()
        records[1]["tensor_download_bytes"] = 13 * 151936 * 4
        records[1]["tensor_device_readbacks"] = 1
        gate.check_residency(records, prefill_logit_rows=13)
        with self.assertRaisesRegex(ValueError, "readback"):
            gate.check_residency(records)
        records[1]["tensor_device_readbacks"] = 2
        with self.assertRaisesRegex(ValueError, "readback"):
            gate.check_residency(records, prefill_logit_rows=13)

    def test_per_layer_decode_synchronization_fails(self):
        records = self.records()
        gate.check_decode_frames(records, 1)
        records[-1]["frame_submits"] = 29
        with self.assertRaisesRegex(ValueError, "frame"):
            gate.check_decode_frames(records, 1)

    def test_truncated_generation_fails(self):
        with self.assertRaisesRegex(ValueError, "naturally"):
            gate.canonical(
                {"model": gate.MODEL, "choices": [{"finish_reason": "length"}]},
                "generate",
            )

    def test_ocr_ignores_spacing_but_rejects_missing_content(self):
        case = {"id": "sparse", "expected_text": "ANTFLY\nOrder 1042"}
        gate.check_ocr_text(case, {"data": [{"text": "ANTFLY  Order\n1042"}]})
        with self.assertRaisesRegex(ValueError, "OCR content"):
            gate.check_ocr_text(case, {"data": [{"text": "ANTFLY Order"}]})

    def test_latency_regression_fails(self):
        gate.check_latency(
            {"sparse": {"median_ms": {"baseline": 100, "candidate": 104}}}, 5
        )
        with self.assertRaisesRegex(ValueError, "regression"):
            gate.check_latency(
                {"sparse": {"median_ms": {"baseline": 100, "candidate": 106}}}, 5
            )

    def test_stream_requires_exact_text_natural_finish_and_done(self):
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Ready"}}]}\n',
            b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n',
            b"data: [DONE]\n",
        ]
        self.assertTrue(gate.parse_stream(chunks, "Ready")["complete"])
        for invalid in (
            chunks[:-1],
            chunks[:1] + chunks[2:],
            [chunks[0].replace(b"Ready", b"Re")] + chunks[1:],
            [chunks[0], chunks[1].replace(b"stop", b"length"), chunks[2]],
        ):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                gate.parse_stream(invalid, "Ready")

    def test_reader_token_exhaustion_fails(self):
        response = {
            "model": gate.MODEL,
            "data": [{"text": "partial"}],
            "usage": {"completion_tokens": 8},
        }
        with patch.object(
            gate.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(response).encode()),
        ):
            with self.assertRaisesRegex(ValueError, "token bound"):
                gate.request(18091, "read", {"images": [{}], "max_tokens": 8})

    def test_http_error_preserves_server_reason_and_status(self):
        error = gate.urllib.error.HTTPError(
            "http://localhost",
            413,
            "Payload Too Large",
            {},
            io.BytesIO(b'{"message":"image budget exhausted"}'),
        )
        with patch.object(gate.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(gate.urllib.error.HTTPError) as caught:
                gate.request(18091, "read", {})
        self.assertEqual(413, caught.exception.code)
        self.assertIn("image budget exhausted", str(caught.exception))

    def test_phase_parser_preserves_transfer_and_frame_counts(self):
        self.assertEqual(
            [dict(phase="decode", frame_submits=27, tensor_download_bytes=0)],
            gate.phases(
                "log qwen3vl_metal_phase: phase=decode frame_submits=27 tensor_download_bytes=0"
            ),
        )


if __name__ == "__main__":
    unittest.main()
