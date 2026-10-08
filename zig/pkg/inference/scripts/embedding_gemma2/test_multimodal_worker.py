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

try:
    import numpy as np
    from PIL import Image as _Image
    OPTIONAL_MEDIA_DEPS = True
except ImportError:
    np = None
    OPTIONAL_MEDIA_DEPS = False

from oracle import cases, fixture_audio, float32_wav
from pytorch_worker import Worker
from qualify_endpoint import request_input
from contract import render_text


@unittest.skipUnless(OPTIONAL_MEDIA_DEPS, "optional NumPy/Pillow media dependencies are unavailable")
class MultimodalWorkerTest(unittest.TestCase):
    def test_float_wav_round_trip_is_bit_exact(self):
        samples = fixture_audio()
        decoded = Worker._decode_wav(float32_wav(samples))
        self.assertTrue(np.array_equal(samples, decoded))

    def test_ordered_mixed_group_preserves_placeholders_and_media(self):
        case = next(item for item in cases() if item["id"] == "mixed_ordered")
        text, images, audios = Worker._prepare_group(request_input(case), case["task_type"])
        self.assertEqual(text.count("<|image|>"), 1)
        self.assertEqual(text.count("<|audio|>"), 1)
        self.assertEqual((len(images), len(audios)), (1, 1))
        self.assertTrue(np.array_equal(audios[0], fixture_audio()))

    def test_invalid_wav_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid WAV"):
            Worker._decode_wav(b"not a wav")

    def test_non_16khz_audio_is_resampled_to_reference_rate(self):
        for rate in (8_000, 44_100, 48_000):
            count = rate // 10
            samples = np.linspace(-0.1, 0.1, count, dtype=np.float32)
            decoded = Worker._decode_wav(float32_wav(samples, rate))
            self.assertLessEqual(abs(len(decoded) - 1_600), 1)

    def test_stereo_audio_is_averaged_to_mono(self):
        left = np.array([.2, -.2], dtype=np.float32)
        stereo = np.stack((left, left * .5), axis=1)
        decoded = Worker._decode_wav(float32_wav(stereo))
        np.testing.assert_array_equal(decoded, left * .75)

    def test_split_text_receives_one_task_prefix(self):
        group = {"content": [{"type": "text", "text": "first "}, {"type": "text", "text": "second"}]}
        text, _, _ = Worker._prepare_group(group, "RETRIEVAL_QUERY")
        self.assertEqual(text, render_text("first ", "RETRIEVAL_QUERY") + "second")

    def test_manual_media_markers_are_not_duplicated(self):
        image_case = next(item for item in cases() if item["id"] == "image")
        image_part = request_input(image_case)["content"][0]
        group = {"content": [{"type": "text", "text": "before <|image|> after"}, image_part]}
        text, images, _ = Worker._prepare_group(group, "RETRIEVAL_DOCUMENT")
        self.assertEqual(text.count("<|image|>"), 1)
        self.assertEqual(len(images), 1)


if __name__ == "__main__":
    unittest.main()
