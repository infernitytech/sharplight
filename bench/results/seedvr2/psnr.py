#!/usr/bin/env python3
"""psnr.py <a.u8> <b.u8> <width> <height>: per-frame PSNR (dB) of two raw 8-bit RGB clips of the same size, and the worst frame."""
import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.uint8)
b = np.fromfile(sys.argv[2], dtype=np.uint8)
frame = int(sys.argv[3]) * int(sys.argv[4]) * 3
assert a.size == b.size and a.size % frame == 0, (a.size, b.size, frame)
values = []
for f in range(a.size // frame):
    d = a[f * frame:(f + 1) * frame].astype(np.float64) - b[f * frame:(f + 1) * frame]
    mse = (d * d).mean()
    values.append(float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse))
print("frames " + " ".join(f"{v:.1f}" for v in values))
print(f"worst {min(values):.2f} dB, mean {np.mean(values):.2f} dB")
