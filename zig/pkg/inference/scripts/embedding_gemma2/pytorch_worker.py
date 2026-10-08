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

"""Minimal resident PyTorch CUDA reference endpoint for paired benchmarks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import platform
import subprocess
import threading
import struct

from contract import MAX_TOKENS, MODEL_ID, REVISION, WEIGHT_BYTES, WEIGHT_SHA256, render_text, truncate_and_normalize


ATTENTION_MODES = ("eager", "sdpa", "flex_attention")
COMPILE_MODES = ("none", "default", "max-autotune")


class Worker:
    def __init__(self, model_path: Path, attention: str, compile_mode: str, torch_threads: int):
        import torch
        import transformers
        import numpy
        import PIL
        import scipy
        import tokenizers
        from transformers import EmbeddingGemma2Model, EmbeddingGemma2Processor

        if attention not in ATTENTION_MODES:
            raise ValueError(f"unsupported attention mode: {attention}")
        if compile_mode not in COMPILE_MODES:
            raise ValueError(f"unsupported compile mode: {compile_mode}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        torch.set_num_threads(torch_threads)
        if attention == "sdpa" and not EmbeddingGemma2Model._supports_sdpa:
            raise RuntimeError("this Transformers build does not support SDPA for EmbeddingGemma2")
        if attention == "flex_attention" and not EmbeddingGemma2Model._supports_flex_attn:
            raise RuntimeError("this Transformers build does not support flex_attention for EmbeddingGemma2")
        weight = model_path / "model.safetensors"
        digest = hashlib.sha256()
        with weight.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if weight.stat().st_size != WEIGHT_BYTES or digest.hexdigest() != WEIGHT_SHA256:
            raise RuntimeError("checkpoint does not match the reviewed EmbeddingGemma 2 pin")

        self.torch = torch
        self.processor = EmbeddingGemma2Processor.from_pretrained(model_path, local_files_only=True)
        self.processor.audio_seq_length = MAX_TOKENS
        model = EmbeddingGemma2Model.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation=attention
        ).to("cuda").eval()
        actual_attention = model.config._attn_implementation
        if actual_attention != attention:
            raise RuntimeError(f"requested attention {attention}, loaded {actual_attention}")
        self.model = model
        if compile_mode != "none":
            self.model = torch.compile(self.model, mode=compile_mode)
        self.lock = threading.Lock()

        props = torch.cuda.get_device_properties(0)
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = None
        self.runtime = {
            "backend": "pytorch_cuda",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "weight_sha256": WEIGHT_SHA256,
            "attention": actual_attention,
            "compile": compile_mode,
            "precision": "bfloat16_weights_fp32_pooling",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pillow": PIL.__version__,
            "tokenizers": tokenizers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": driver,
            "cudnn": torch.backends.cudnn.version(),
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "torch_cpu_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }

    @staticmethod
    def _decode_data(value: str) -> bytes:
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        return base64.b64decode(encoded, validate=True)

    @classmethod
    def _prepare_group(cls, value, task_type: str):
        import numpy as np
        from PIL import Image

        if isinstance(value, str):
            return render_text(value, task_type), [], []
        if not isinstance(value, dict) or not isinstance(value.get("content"), list):
            raise ValueError("input must be text or an ordered content group")
        manual_media = any(
            isinstance(part, dict) and part.get("type") == "text" and
            ("<|image|>" in part.get("text", "") or "<|audio|>" in part.get("text", ""))
            for part in value["content"]
        )
        text, images, audios = "", [], []
        prefixed = False
        for part in value["content"]:
            if not isinstance(part, dict):
                raise ValueError("content parts must be objects")
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("text"), str):
                text += render_text(part["text"], task_type) if not prefixed else part["text"]
                prefixed = True
            elif kind == "image_url":
                source = part.get("image_url")
                source = source.get("url") if isinstance(source, dict) else source
                if not isinstance(source, str):
                    raise ValueError("image_url must contain a URL")
                images.append(Image.open(io.BytesIO(cls._decode_data(source))).convert("RGB"))
                if not manual_media:
                    text += "<|image|>"
            elif kind == "media" and isinstance(part.get("mime_type"), str) and part["mime_type"].startswith("audio/"):
                audios.append(cls._decode_wav(cls._decode_data(part.get("data", ""))))
                if not manual_media:
                    text += "<|audio|>"
            else:
                raise ValueError(f"unsupported content part: {kind}")
        return text, images, audios

    @staticmethod
    def _decode_wav(data: bytes):
        import numpy as np
        from scipy.signal import resample_poly
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise ValueError("invalid WAV container")
        fmt = payload = None
        offset = 12
        while offset + 8 <= len(data):
            kind, size = data[offset:offset+4], struct.unpack_from("<I", data, offset+4)[0]
            chunk = data[offset+8:offset+8+size]
            if kind == b"fmt ": fmt = chunk
            if kind == b"data": payload = chunk
            offset += 8 + size + (size & 1)
        if fmt is None or payload is None or len(fmt) < 16:
            raise ValueError("WAV lacks fmt or data chunk")
        encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
        if channels not in (1, 2) or rate <= 0:
            raise ValueError("reference worker requires mono/stereo WAV with a positive sample rate")
        if encoding == 3 and bits == 32:
            samples = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
        elif encoding == 1 and bits == 16:
            samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        else:
            raise ValueError("reference worker accepts PCM16 or IEEE-float32 WAV")
        if channels == 2:
            if len(samples) % 2:
                raise ValueError("stereo WAV has an incomplete frame")
            samples = samples.reshape(-1, 2).mean(axis=1, dtype=np.float32)
        if rate != 16_000:
            # Production reference policy: polyphase FIR resampling, with the
            # exact integer ratio reduced internally by scipy.
            samples = resample_poly(samples, 16_000, rate).astype(np.float32)
        return samples

    def embed(self, inputs, dimensions: int | None, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if isinstance(inputs, (str, dict)):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("input must be non-empty")
        groups = [self._prepare_group(value, task_type) for value in inputs]
        kwargs = {"text": [group[0] for group in groups], "padding": True, "truncation": False, "return_tensors": "pt"}
        if any(group[1] for group in groups):
            kwargs["images"] = [group[1] for group in groups]
        if any(group[2] for group in groups):
            kwargs["audio"] = [group[2] for group in groups]
        encoded = self.processor(**kwargs)
        if int(encoded["attention_mask"].sum(-1).max()) > MAX_TOKENS:
            raise ValueError("expanded input exceeds 8192 tokens")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with self.lock, self.torch.inference_mode():
            states = self.model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            vectors = self.torch.nn.functional.normalize(
                (states * mask).sum(1) / mask.sum(1).clamp_min(1), p=2, dim=-1
            )
            if not bool(self.torch.isfinite(vectors).all()) or bool((self.torch.linalg.vector_norm(vectors, dim=-1) == 0).any()):
                raise RuntimeError("model produced a non-finite or zero-norm embedding")
            vectors = vectors.cpu().tolist()
        if dimensions is not None:
            vectors = [truncate_and_normalize(vector, dimensions) for vector in vectors]
        return vectors


def handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                vectors = worker.embed(body["input"], body.get("dimensions"), body.get("task_type", "RETRIEVAL_DOCUMENT"))
                payload = {"backend": "pytorch_cuda", "runtime": worker.runtime, "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]}
                data = json.dumps(payload).encode()
                self.send_response(200)
            except Exception as exc:
                data = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, _format, *_args):
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--attention", choices=ATTENTION_MODES, default="sdpa")
    parser.add_argument("--compile", choices=COMPILE_MODES, default="none")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--warmup-batch-size", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 8), default=1)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.warmup_batch_size < 1 or args.warmup_tokens < 1:
        parser.error("warmup must be non-negative; warmup batch size and tokens must be positive")
    worker = Worker(args.model, args.attention, args.compile, args.torch_threads)
    warmup_input = [" ".join(["ant"] * args.warmup_tokens)] * args.warmup_batch_size
    for _ in range(args.warmup):
        worker.embed(warmup_input, None)
    worker.torch.cuda.synchronize()
    server = ThreadingHTTPServer((args.host, args.port), handler(worker))
    print(json.dumps({"listening": f"http://{args.host}:{args.port}/v1/embeddings", "runtime": worker.runtime, "warmup": {"iterations": args.warmup, "batch_size": args.warmup_batch_size, "approx_tokens": args.warmup_tokens}}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
