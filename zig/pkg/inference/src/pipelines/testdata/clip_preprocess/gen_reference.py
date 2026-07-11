#!/usr/bin/env python
"""Regenerate the CLIP-preprocessing parity fixtures.

Produces, for a deterministic synthetic 767x462 test image (non-square aspect +
high-frequency stripe bands that stress antialiased downsampling, and a resized
long edge of 371 that exercises torchvision's round-half-to-even center-crop):

  input.png              the test image (committed)
  reference_chw_f16.bin  torchvision Resize(224, BICUBIC) + CenterCrop(224) +
                         rescale(1/255) + CLIP mean/std normalize, laid out CHW,
                         little-endian float16 (committed)

The Zig unit test `clip preprocessing matches torchvision CLIP canonical` decodes
input.png, runs the server's CLIP preprocessing, and asserts cosine >= 0.99 vs
this reference. Run with a venv that has torchvision + PIL:
  /private/tmp/mm_bridge/venv/bin/python gen_reference.py
"""
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, TARGET = 767, 462, 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float64)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float64)


def synth_image():
    ys, xs = np.mgrid[0:H, 0:W]
    xf = xs / (W - 1)
    yf = ys / (H - 1)
    r = (xf * 255).astype(np.float64)
    g = (yf * 255).astype(np.float64)
    b = (np.hypot(xf - 0.5, yf - 0.5) / 0.707 * 255).astype(np.float64)
    img = np.stack([r, g, b], -1)
    # high-frequency vertical stripes (period 3px) in a horizontal band
    band_v = (ys > H * 0.15) & (ys < H * 0.4)
    img[band_v] = np.where(((xs[band_v] % 3) == 0)[:, None], 20.0, 235.0)
    # high-frequency horizontal stripes (period 2px) in another band
    band_h = (ys > H * 0.55) & (ys < H * 0.8)
    img[band_h] = np.where(((ys[band_h] % 2) == 0)[:, None], 10.0, 245.0)
    # a couple of solid blocks + a diagonal for structure
    img[40:120, 500:650] = np.array([200.0, 40.0, 120.0])
    diag = np.abs((xs * 0.6).astype(int) - ys) < 4
    img[diag] = np.array([255.0, 255.0, 0.0])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def main():
    im = synth_image()
    im.save(os.path.join(HERE, "input.png"))
    tf = T.Compose([T.Resize(TARGET, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(TARGET)])
    arr = np.asarray(tf(im), np.float64) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD               # HWC
    chw = arr.transpose(2, 0, 1).astype(np.float16)  # CHW
    chw.tofile(os.path.join(HERE, "reference_chw_f16.bin"))
    print(f"wrote input.png ({im.size}) and reference_chw_f16.bin ({chw.shape}, {chw.dtype})")
    print("resized dims (ow,oh) expected (371,224); crop offset left=round(73.5)=74")


if __name__ == "__main__":
    main()
