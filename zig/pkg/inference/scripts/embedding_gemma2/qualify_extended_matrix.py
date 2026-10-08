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

"""Gate extended image, mixed ordering, split text, and task-prefix cases."""

import argparse
import base64
import io
import json
import math
from pathlib import Path
import urllib.request

from contract import PROMPTS, cosine
from oracle import fixture_audio, float32_wav
from select_pytorch_baseline import exact_cell_inputs


def post(url, body, timeout):
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def image_part(width, height):
    import numpy as np
    from PIL import Image
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=-1).astype("uint8")
    stream = io.BytesIO(); Image.fromarray(pixels, "RGB").save(stream, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": uri}}


def audio_part():
    return {"type": "media", "mime_type": "audio/wav", "data": base64.b64encode(float32_wav(fixture_audio())).decode()}


def cases(processor):
    rows = []
    for width, height in ((32, 32), (128, 96), (896, 896), (1536, 256), (256, 1536)):
        rows.append((f"image-{width}x{height}", "RETRIEVAL_DOCUMENT", {"content": [image_part(width, height)]}))
    image, audio = image_part(128, 96), audio_part()
    text = {"type": "text", "text": "Find the matching media."}
    for name, content in (("mixed-text-image-audio", [text, image, audio]),
                          ("mixed-audio-text-image", [audio, text, image]),
                          ("mixed-image-image-text", [image, image, text]),
                          ("mixed-audio-audio-text", [audio, audio, text])):
        rows.append((name, "RETRIEVAL_QUERY", {"content": content}))
    rows.append(("split-text", "RETRIEVAL_QUERY", {"content": [{"type": "text", "text": "first "}, {"type": "text", "text": "second"}]}))
    rows.append(("manual-image-marker", "RETRIEVAL_DOCUMENT", {"content": [{"type": "text", "text": "before <|image|> after"}, image]}))
    rows.append(("manual-audio-marker", "RETRIEVAL_DOCUMENT", {"content": [{"type": "text", "text": "before <|audio|> after"}, audio]}))
    for task in PROMPTS:
        rows.append((f"task-{task.lower()}", task, exact_cell_inputs(processor, "natural_text", 1, 128, task)[0]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antfly-url", required=True); parser.add_argument("--pytorch-url", required=True)
    parser.add_argument("--model", default="embeddinggemma-2"); parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(); results = []
    from transformers import EmbeddingGemma2Processor
    processor = EmbeddingGemma2Processor.from_pretrained(args.processor_dir, local_files_only=True)
    for name, task, value in cases(processor):
        body = {"model": args.model, "input": value, "task_type": task}
        native, reference = post(args.antfly_url, body, args.timeout), post(args.pytorch_url, body, args.timeout)
        a, b = native["data"][0]["embedding"], reference["data"][0]["embedding"]
        norm = math.sqrt(sum(x*x for x in a))
        results.append({"case": name, "cosine": cosine(a, b), "norm": norm, "backend": native.get("backend"), "usage": native.get("usage")})
    report = {"schema": "antfly.embedding_gemma2.extended_matrix.v1", "minimum_cosine": min(row["cosine"] for row in results),
              "pass": all(row["cosine"] >= .999 and abs(row["norm"]-1) <= 1e-3 and row["backend"] == "cuda" for row in results), "rows": results}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, sort_keys=True)); return 0 if report["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
