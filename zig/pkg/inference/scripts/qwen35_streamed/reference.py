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

"""Bounded CPU oracle: upstream ggml codecs, NumPy FP32 operations, explicit pread.

No model mmap or full FP32 matrix. Uses 128-row dequant tiles. Equations follow
llama.cpp 6184e92c57dcd34de8a3e381d7641a5e75250d5f/src/models/qwen35.cpp.
This is an independent equation oracle, not a full llama.cpp graph execution.
"""

import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
import resource
import struct
import time
from pathlib import Path

import numpy as np

TYPES = {
    0: (1, 4, None),
    8: (32, 34, "q8_0"),
    11: (256, 110, "q3_K"),
    12: (256, 144, "q4_K"),
    13: (256, 176, "q5_K"),
    14: (256, 210, "q6_K"),
    20: (32, 18, "iq4_nl"),
    21: (256, 110, "iq3_s"),
    23: (256, 136, "iq4_xs"),
}


class Model:
    def __init__(self, path, library):
        self.fd = os.open(path, os.O_RDONLY)
        fcntl.fcntl(self.fd, 48, 1)
        self.lib = C.CDLL(str(library))
        self.read_bytes = 0
        spec = json.loads(Path(__file__).with_name("artifact.json").read_text())[
            "files"
        ][0]
        if os.fstat(self.fd).st_size != spec["bytes"]:
            raise ValueError("artifact size mismatch")
        digest = hashlib.sha256()
        for offset in range(0, spec["bytes"], 4 * 1024 * 1024):
            digest.update(
                self.read(offset, min(4 * 1024 * 1024, spec["bytes"] - offset))
            )
        self.sha256 = digest.hexdigest()
        if self.sha256 != spec["sha256"]:
            raise ValueError("artifact hash mismatch")
        self.library_sha256 = hashlib.sha256(Path(library).read_bytes()).hexdigest()
        self.header = self.read(0, 16 * 1024 * 1024)
        self.at = 0
        if self.take(4) != b"GGUF" or self.num("I") != 3:
            raise ValueError("expected GGUF v3")
        nt, nk = (self.num("Q"), self.num("Q"))
        self.meta = {self.string(): self.value(self.num("I")) for _ in range(nk)}
        tensors = {}
        for _ in range(nt):
            name = self.string()
            dims = [self.num("Q") for _ in range(self.num("I"))]
            kind, offset = (self.num("I"), self.num("Q"))
            tensors[name] = (dims, kind, offset)
        align = self.meta.get("general.alignment", 32)
        base = (self.at + align - 1) // align * align
        self.tensors = {n: (d, k, o + base) for n, (d, k, o) in tensors.items()}
        del self.header
        self.state, self.conv, self.kv = ({}, {}, {})
        self.position = 0

    def take(self, n):
        x = self.header[self.at : self.at + n]
        self.at += n
        if len(x) != n:
            raise ValueError("header bound exceeded")
        return x

    def num(self, f):
        return struct.unpack("<" + f, self.take(struct.calcsize("<" + f)))[0]

    def string(self):
        return self.take(self.num("Q")).decode()

    def value(self, k):
        if k == 8:
            return self.string()
        if k == 9:
            t, n = (self.num("I"), self.num("Q"))
            return [self.value(t) for _ in range(n)]
        return self.num(
            {
                0: "B",
                1: "b",
                2: "H",
                3: "h",
                4: "I",
                5: "i",
                6: "f",
                7: "?",
                10: "Q",
                11: "q",
                12: "d",
            }[k]
        )

    def read(self, offset, n):
        pieces = []
        while n:
            b = os.pread(self.fd, min(n, 4 * 1024 * 1024), offset)
            if not b:
                raise OSError("short read")
            pieces.append(b)
            offset += len(b)
            n -= len(b)
            self.read_bytes += len(b)
        return b"".join(pieces)

    def rows(self, name, start, count):
        dims, k, offset = self.tensors[name]
        width = dims[0]
        block, size, codec = TYPES[k]
        rb = width // block * size
        raw = self.read(offset + start * rb, count * rb)
        if codec is None:
            return np.frombuffer(raw, dtype="<f4").reshape(count, width).copy()
        out = np.empty((count, width), np.float32)
        fn = getattr(self.lib, "dequantize_row_" + codec)
        fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        fn.restype = None
        fn(C.c_char_p(raw), out.ctypes.data, count * width)
        return out

    def weight(self, name):
        dims, _, _ = self.tensors[name]
        return self.rows(
            name, 0, int(np.prod(dims[1:])) if len(dims) > 1 else 1
        ).reshape(tuple(reversed(dims)))

    def mm(self, name, x):
        width, out = self.tensors[name][0]
        x = np.asarray(x, np.float32).reshape(-1, width)
        result = np.empty((len(x), out), np.float32)
        for start in range(0, out, 128):
            count = min(128, out - start)
            result[:, start : start + count] = x @ self.rows(name, start, count).T
        return result

    def step(self, ids, capture=None):
        hidden = np.concatenate([self.rows("token_embd.weight", i, 1) for i in ids])
        for layer in range(64):
            prefix = f"blk.{layer}."
            w = lambda n, prefix=prefix: self.weight(prefix + n)
            mm = lambda n, x, prefix=prefix: self.mm(prefix + n + ".weight", x)
            x = rms(hidden, w("attn_norm.weight"))
            if (layer + 1) % 4:
                mixed = mm("attn_qkv", x)
                gate = mm("attn_gate", x)
                beta = sig(mm("ssm_beta", x))
                alpha = mm("ssm_alpha", x) + w("ssm_dt.bias")
                decay = np.exp(np.logaddexp(np.float32(0), alpha) * w("ssm_a"))
                conv = self.conv.setdefault(layer, np.zeros((3, 10240), np.float32))
                state = self.state.setdefault(
                    layer, np.zeros((48, 128, 128), np.float32)
                )
                kernel = w("ssm_conv1d.weight").T
                values = []
                for i in range(len(ids)):
                    z = silu((conv * kernel[:3]).sum(axis=0) + mixed[i] * kernel[3])
                    conv[:2] = conv[1:]
                    conv[2] = mixed[i]
                    q = z[:2048].reshape(16, 128)
                    k = z[2048:4096].reshape(16, 128)
                    q = np.tile(
                        q / np.sqrt((q * q).sum(-1, keepdims=True) + np.float32(1e-06)),
                        (3, 1),
                    ) / np.sqrt(np.float32(128))
                    k = np.tile(
                        k / np.sqrt((k * k).sum(-1, keepdims=True) + np.float32(1e-06)),
                        (3, 1),
                    )
                    state *= decay[i, :, None, None]
                    prediction = np.einsum("hkv,hk->hv", state, k)
                    delta = (z[4096:].reshape(48, 128) - prediction) * beta[i, :, None]
                    state += k[:, :, None] * delta[:, None, :]
                    values.append(np.einsum("hkv,hk->hv", state, q))
                values = np.stack(values)
                attn = (
                    rms(values, w("ssm_norm.weight")) * silu(gate.reshape(-1, 48, 128))
                ).reshape(-1, 6144)
                hidden += mm("ssm_out", attn)
            else:
                qg = mm("attn_q", x).reshape(-1, 24, 512)
                q = rms(qg[:, :, :256], w("attn_q_norm.weight"))
                k = rms(mm("attn_k", x).reshape(-1, 4, 256), w("attn_k_norm.weight"))
                v = mm("attn_v", x).reshape(-1, 4, 256)
                keys, values = self.kv.setdefault(layer, ([], []))
                outputs = []
                for i in range(len(ids)):
                    qi = rope(q[i], self.position + i)
                    ki = rope(k[i], self.position + i)
                    keys.append(ki)
                    values.append(v[i])
                    ks = np.repeat(np.stack(keys), 6, axis=1)
                    vs = np.repeat(np.stack(values), 6, axis=1)
                    score = np.einsum("hd,thd->ht", qi, ks) * np.float32(0.0625)
                    prob = np.exp(score - score.max(-1, keepdims=True))
                    prob /= prob.sum(-1, keepdims=True)
                    outputs.append(
                        np.einsum("ht,thd->hd", prob, vs) * sig(qg[i, :, 256:])
                    )
                hidden += mm("attn_output", np.stack(outputs).reshape(-1, 6144))
            x = rms(hidden, w("post_attention_norm.weight"))
            hidden += mm("ffn_down", silu(mm("ffn_gate", x)) * mm("ffn_up", x))
            if capture:
                hidden.astype("<f4").tofile(
                    capture / f"layer-{layer:02}-pos-{self.position + len(ids)}.f32"
                )
            if capture and os.environ.get("ANTFLY_QWEN35_CAPTURE_STATE"):
                if (layer + 1) % 4:
                    self.state[layer].tofile(
                        capture / f"state-{layer:02}-pos-{self.position + len(ids)}.f32"
                    )
                    self.conv[layer].tofile(
                        capture / f"aux-{layer:02}-pos-{self.position + len(ids)}.f32"
                    )
                else:
                    np.stack(self.kv[layer][0]).tofile(
                        capture / f"state-{layer:02}-pos-{self.position + len(ids)}.f32"
                    )
                    np.stack(self.kv[layer][1]).tofile(
                        capture / f"aux-{layer:02}-pos-{self.position + len(ids)}.f32"
                    )
            if layer % 8 == 7:
                print(f"layer {layer + 1}/64", flush=True)
        self.position += len(ids)
        logits = self.mm(
            "output.weight", rms(hidden[-1:], self.weight("output_norm.weight"))
        )[0]
        return logits


def sig(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


def silu(x):
    return x * sig(x)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-06)) * w


def rope(x, pos):
    y = x.copy()
    angle = np.float32(pos) * np.power(
        np.float32(10000000.0), -2 * np.arange(32, dtype=np.float32) / 64
    )
    c, s = (np.cos(angle), np.sin(angle))
    y[:, :32] = x[:, :32] * c - x[:, 32:64] * s
    y[:, 32:64] = x[:, 32:64] * c + x[:, :32] * s
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--ggml-library", type=Path, required=True)
    p.add_argument("--ids", required=True, help="comma separated token ids")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capture", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.capture:
        a.capture.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in a.ids.split(",")]
    if not 0 < len(ids) <= 128 or a.steps < 1 or len(ids) + a.steps > 4096:
        p.error("bounded to 128 prompt tokens and 4K context")
    model = Model(a.model, a.ggml_library)
    tokens = []
    start = time.monotonic()
    for i in range(a.steps):
        logits = model.step(ids, a.capture)
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite logits")
        logits.astype("<f4").tofile(a.output / f"logits-{i}.f32")
        token = int(logits.argmax())
        tokens.append(token)
        ids = [token]
        print(f"token={token}", flush=True)
    (a.output / "receipt.json").write_text(
        json.dumps(
            {
                "tokens": tokens,
                "seconds": time.monotonic() - start,
                "bytes_read": model.read_bytes,
                "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "reference": "ggml codecs + NumPy independent FP32 equations",
                "model_sha256": model.sha256,
                "ggml_library_sha256": model.library_sha256,
            },
            indent=2,
        )
        + "\n"
    )
    os.close(model.fd)


if __name__ == "__main__":
    main()
