"""Preprocessing references for the tiny Gemma 3 and Idefics3 checkpoints of mk_tiny_siglip_vl.py: transformers' (PIL) image processor on
synthetic 8-bit images, value(y, x, c) = (7·x + 3·y + 50·c + (x·y) % 31) % 256 (the formula of the C# test).

Usage: python siglip_vl_preprocess_reference.py <model_dir>

Writes <model_dir>/reference_preprocess.json: per image size the processor's pixel values (views in its order, channels first) and, for Idefics3,
the tile grid (rows, cols).
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

model_dir = sys.argv[1]
processor = AutoImageProcessor.from_pretrained(model_dir, backend="pil")
cases = []
for width, height in [(300, 200), (150, 380), (64, 64), (97, 41)]:
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([(7 * x + 3 * y + 50 * c + (x * y) % 31) % 256 for c in range(3)], axis=-1).astype(np.uint8)
    out = processor(images=[Image.fromarray(rgb)], return_tensors="np", return_row_col_info=True) \
        if "Idefics3" in type(processor).__name__ else processor(images=[Image.fromarray(rgb)], return_tensors="np")
    pixels = np.asarray(out["pixel_values"], dtype=np.float32).reshape(-1)
    case = {"width": width, "height": height, "pixels": [round(float(v), 6) for v in pixels]}
    if "rows" in out:
        case.update(rows=int(np.asarray(out["rows"]).reshape(-1)[0]), cols=int(np.asarray(out["cols"]).reshape(-1)[0]))
    cases.append(case)
    print(width, height, pixels.shape, case.get("rows"), case.get("cols"))

with open(os.path.join(model_dir, "reference_preprocess.json"), "w") as f:
    json.dump({"cases": cases}, f)
