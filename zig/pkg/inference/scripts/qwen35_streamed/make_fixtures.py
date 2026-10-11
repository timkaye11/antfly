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

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
from reference import TYPES, Model

p = argparse.ArgumentParser(
    description="Regenerate bounded real-weight ggml dequant fixtures"
)
p.add_argument("model")
p.add_argument("--ggml-library", required=True)
a = p.parse_args()
m = Model(a.model, a.ggml_library)
out = Path(__file__).resolve().parents[2] / "src/backends/fixtures/qwen35_streamed"
out.mkdir(parents=True, exist_ok=True)
meta = []
for kind, (block, size, codec) in TYPES.items():
    if kind == 0:
        name = "output_norm.weight"
        raw = m.read(m.tensors[name][2], 768 * 4)
    else:
        for name, (dims, k, offset) in m.tensors.items():
            if k == kind and len(dims) == 2 and (dims[0] >= 256) and (dims[1] >= 3):
                rb = dims[0] // block * size
                raw = b"".join(
                    m.read(offset + i * rb, 256 // block * size) for i in (0, 1, 2)
                )
                break
        else:
            raise ValueError(kind)
    expected = np.empty((3, 256), np.float32)
    if codec:
        fn = getattr(m.lib, "dequantize_row_" + codec)
        fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        fn.restype = None
        fn(C.c_char_p(raw), expected.ctypes.data, 768)
    else:
        expected[:] = np.frombuffer(raw, dtype="<f4").reshape(3, 256)
    (out / f"{kind}.bin").write_bytes(raw)
    expected.astype("<f4").tofile(out / f"{kind}.f32")
    meta.append(
        {
            "type": kind,
            "source_tensor": name,
            "rows": [0, 1, 2],
            "columns": 256,
            "note": "F32 is contiguous first 768 norm elements reshaped to 3x256"
            if kind == 0
            else "first 256 columns of each row",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_sha256": hashlib.sha256(
                expected.astype("<f4").tobytes()
            ).hexdigest(),
        }
    )
(out / "provenance.json").write_text(
    json.dumps(
        {
            "ggml_revision": "6184e92c57dcd34de8a3e381d7641a5e75250d5f",
            "gguf_sha256": "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
            "cases": meta,
        },
        indent=2,
    )
    + "\n"
)
