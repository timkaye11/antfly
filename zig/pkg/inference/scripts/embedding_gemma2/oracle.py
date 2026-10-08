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

"""Generate deterministic FP32 or BF16 CUDA EmbeddingGemma 2 oracle vectors.

The input fixtures are synthesized locally, so the only network artifact is
the separately pinned checkpoint. Video is deliberately outside this lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import struct
import time

from contract import (
    FULL_DIMENSIONS,
    MAX_TOKENS,
    MODEL_ID,
    REVISION,
    TRAINED_DIMENSIONS,
    WEIGHT_BYTES,
    WEIGHT_SHA256,
    render_text,
    truncate_and_normalize,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_image():
    import numpy as np
    from PIL import Image

    y, x = np.mgrid[:96, :128]
    pixels = np.stack(((x * 2) % 256, (y * 3) % 256, (x + y) % 256), axis=-1).astype("uint8")
    return Image.fromarray(pixels, "RGB")


def fixture_audio():
    import numpy as np

    samples = np.arange(16_000, dtype=np.float32)
    return (0.2 * np.sin(2.0 * np.pi * 440.0 * samples / 16_000.0)).astype(np.float32)


def float32_wav(samples, sample_rate: int = 16_000) -> bytes:
    """Encode mono/stereo IEEE-float WAV without quantizing oracle samples."""
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    if channels not in (1, 2):
        raise ValueError("float WAV fixture supports mono or stereo")
    payload = samples.astype("<f4", copy=False).tobytes()
    block = channels * 4
    fmt = struct.pack("<HHIIHH", 3, channels, sample_rate, sample_rate * block, block, 32)
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(payload)) + b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload


def cases() -> list[dict]:
    return [
        {"id": "document", "task_type": "RETRIEVAL_DOCUMENT", "parts": [{"type": "text", "text": "Ant colonies coordinate through local signals."}]},
        {"id": "query", "task_type": "RETRIEVAL_QUERY", "parts": [{"type": "text", "text": "How do ants coordinate?"}]},
        {"id": "code_query", "task_type": "CODE_RETRIEVAL", "parts": [{"type": "text", "text": "function that computes cosine similarity"}]},
        {"id": "image", "task_type": "RETRIEVAL_DOCUMENT", "parts": [{"type": "image"}]},
        {"id": "audio", "task_type": "RETRIEVAL_DOCUMENT", "parts": [{"type": "audio"}]},
        {"id": "mixed_ordered", "task_type": "RETRIEVAL_QUERY", "parts": [{"type": "text", "text": "Find media matching this signal: "}, {"type": "image"}, {"type": "audio"}]},
    ]


def prepare_case(processor, case: dict):
    image = fixture_image()
    audio = fixture_audio()
    text = ""
    images = []
    audios = []
    for part in case["parts"]:
        if part["type"] == "text":
            text += render_text(part["text"], case["task_type"])
        elif part["type"] == "image":
            text += "<|image|>"
            images.append(image)
        elif part["type"] == "audio":
            text += "<|audio|>"
            audios.append(audio)
        else:
            raise ValueError(f"unsupported fixture part: {part['type']}")
    kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if images:
        kwargs["images"] = [images]
    if audios:
        kwargs["audio"] = [audios]
    encoded = processor(**kwargs)
    expanded_tokens = int(encoded["attention_mask"].sum().item())
    if expanded_tokens > MAX_TOKENS:
        raise ValueError(f"{case['id']}: {expanded_tokens} expanded tokens exceed {MAX_TOKENS}")
    return text, encoded, expanded_tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    import transformers
    import numpy
    import PIL
    import scipy
    import tokenizers
    from transformers import EmbeddingGemma2Model, EmbeddingGemma2Processor

    weight = args.model / "model.safetensors"
    if weight.stat().st_size != WEIGHT_BYTES or sha256(weight) != WEIGHT_SHA256:
        raise SystemExit("checkpoint weight bytes do not match the reviewed pin")
    if args.precision == "bf16" and args.device != "cuda":
        parser.error("bf16 oracle requires --device cuda")
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    processor = EmbeddingGemma2Processor.from_pretrained(args.model, local_files_only=True)
    # The upstream processor's configured audio cap is advisory. Token counts
    # remain derived from the frontend mask and the global 8192-token limit.
    processor.audio_seq_length = MAX_TOKENS
    model = EmbeddingGemma2Model.from_pretrained(args.model, local_files_only=True, dtype=dtype).to(args.device).eval()

    rows = []
    for case in cases():
        rendered, encoded, expanded_tokens = prepare_case(processor, case)
        encoded = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in encoded.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            vector = F.normalize((output * mask).sum(1) / mask.sum(1).clamp_min(1), p=2, dim=-1)[0].cpu().tolist()
        if len(vector) != FULL_DIMENSIONS:
            raise RuntimeError(f"unexpected output width: {len(vector)}")
        tensor_shapes = {
            key: list(value.shape)
            for key, value in encoded.items()
            if hasattr(value, "shape")
        }
        placeholder_counts = {
            "image": int((encoded["input_ids"] == processor.image_token_id).sum().item()),
            "audio": int((encoded["input_ids"] == processor.audio_token_id).sum().item()),
        }
        rows.append({
            **case,
            "rendered_text": rendered,
            "expanded_tokens": expanded_tokens,
            "input_ids": encoded["input_ids"][0].cpu().tolist(),
            "tensor_shapes": tensor_shapes,
            "placeholder_counts": placeholder_counts,
            "embeddings": {str(dim): truncate_and_normalize(vector, dim) for dim in TRAINED_DIMENSIONS},
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        })
    payload = {
        "schema": "antfly.embedding_gemma2.oracle.v1",
        "model": {"id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256},
        "runtime": {"precision": args.precision, "device": args.device, "torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__, "tokenizers": tokenizers.__version__},
        "contract": {"max_expanded_tokens": MAX_TOKENS, "pooling": "masked_mean", "include_prompt": True, "normalize": True, "dimensions": list(TRAINED_DIMENSIONS), "audio_implicit_truncation": False},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(rows), "runtime": payload["runtime"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
