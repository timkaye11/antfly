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

import unittest
from qualify_florence2_residency import check_encoder, parse_encoder


class ResidencyEvidenceTests(unittest.TestCase):
    def test_missing_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            check_encoder(parse_encoder("request completed without counters"))

    def test_download_or_access_is_rejected(self):
        for downloads, accesses in ((4, 0), (0, 1)):
            with self.assertRaises(ValueError):
                check_encoder(
                    [
                        {
                            "tensor_download_bytes": downloads,
                            "tensor_device_readbacks": accesses,
                        }
                    ]
                )

    def test_every_batch_chunk_must_stay_resident(self):
        records = parse_encoder(
            "florence_metal_phase: phase=encoder batch=2 tensor_download_bytes=0 tensor_device_readbacks=0 frame_submits=6\n"
            "florence_metal_phase: phase=encoder batch=1 tensor_download_bytes=4 tensor_device_readbacks=1 frame_submits=6"
        )
        self.assertEqual(len(records), 2)
        with self.assertRaises(ValueError):
            check_encoder(records)
        check_encoder(records[:1])


if __name__ == "__main__":
    unittest.main()
