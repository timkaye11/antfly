#!/usr/bin/env python
"""Regenerate the SigLIP preprocessing parity fixture.

Produces `input.png` (a real, non-square RGB image larger than 512 so the resize
antialiases on downscale) and `reference_chw_f16.bin`: the canonical SigLIP
pixel tensor [1,3,512,512] from HuggingFace `SiglipImageProcessor`
(google/siglip2-base-patch16-512): resize to 512x512 SQUARE with PIL BILINEAR
(resample=2, antialiased), rescale 1/255, normalize mean/std 0.5 -> [-1,1], RGB,
CHW. The Zig `preprocessSiglipBatch` must match this at cosine >= 0.99.

Run: python3 gen_reference.py [source_image]
"""
import os, sys
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
MID = "google/siglip2-base-patch16-512"
src = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/mm_encoder/arxiv_cache/2002.bin"

im = Image.open(src).convert("RGB")
# Non-square, larger than 512 on both axes so the resize genuinely downsamples.
im = im.resize((704, 408), Image.BICUBIC)
im.save(os.path.join(HERE, "input.png"), optimize=True)
print("input.png", im.size, os.path.getsize(os.path.join(HERE, "input.png")), "bytes")

ip = AutoImageProcessor.from_pretrained(MID)
print("processor:", type(ip).__name__, "size", ip.size, "resample", ip.resample,
      "mean", ip.image_mean, "std", ip.image_std)
pix = ip(images=im, return_tensors="np")["pixel_values"].astype(np.float32)  # [1,3,512,512]
print("pixel_values", pix.shape, "range", float(pix.min()), float(pix.max()))
pix.astype(np.float16).tofile(os.path.join(HERE, "reference_chw_f16.bin"))
print("wrote reference_chw_f16.bin", 3*512*512*2, "bytes")
