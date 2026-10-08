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

"""Compare native and PyTorch CUDA audio quality across rates and durations."""

import argparse
import base64
import json
from pathlib import Path
import urllib.request

from contract import cosine
from oracle import fixture_audio, float32_wav


def post(url, body, timeout):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def samples(rate: int, duration: float):
    import numpy as np
    t = np.arange(round(rate * duration), dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 440 * t / rate)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antfly-url", required=True)
    parser.add_argument("--pytorch-url", required=True)
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    cells = [(16_000, .1, 1), (8_000, 1, 1), (16_000, 1, 1), (16_000, 1, 2),
             (44_100, 1, 1), (48_000, 1, 1), (48_000, 10, 1), (16_000, 10, 1), (16_000, 31, 1)]
    for rate, duration, channels in cells:
        waveform = samples(rate, duration)
        if channels == 2:
            import numpy as np
            waveform = np.stack((waveform, waveform * .5), axis=1)
        wav = float32_wav(waveform, rate)
        group = {"content": [{"type": "media", "mime_type": "audio/wav", "data": base64.b64encode(wav).decode()}]}
        body = {"model": args.model, "input": group, "task_type": "RETRIEVAL_DOCUMENT"}
        native = post(args.antfly_url, body, args.timeout)
        reference = post(args.pytorch_url, body, args.timeout)
        a, b = native["data"][0]["embedding"], reference["data"][0]["embedding"]
        rows.append({"sample_rate": rate, "duration_seconds": duration, "channels": channels, "cosine": cosine(a, b),
                     "native_backend": native.get("backend"), "dimensions": len(a)})
    report = {"schema": "antfly.embedding_gemma2.audio_matrix.v1", "pass": all(row["cosine"] >= .999 and row["native_backend"] == "cuda" for row in rows), "minimum_cosine": min(row["cosine"] for row in rows), "rows": rows}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
