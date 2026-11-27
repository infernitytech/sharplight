"""Preprocessing references for the tiny Gemma 3 and Idefics3 checkpoints of mk_tiny_siglip_vl.py and for LFM2-VL (the real
LiquidAI/LFM2-VL-450M processor: its tiny checkpoint has other token bounds than the 64..256 llama.cpp fixes): transformers' image processor
(PIL; LFM2-VL's is torchvision only) on synthetic 8-bit images, value(y, x, c) = (7·x + 3·y + 50·c + (x·y) % 31) % 256 (the formula of the C# test).

Usage: python siglip_vl_preprocess_reference.py <model_dir>

Writes <model_dir>/reference_preprocess.json: per image size the processor's pixel values (views in its order, channels first) and, for Idefics3
and LFM2-VL, the tile grid (rows, cols); for LFM2-VL also every view's size ("views": [[width, height], ...]), as its views differ in size.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

model_dir = sys.argv[1]
lfm2 = json.load(open(os.path.join(model_dir, "config.json")))["model_type"] == "lfm2_vl"
processor = AutoImageProcessor.from_pretrained(model_dir) if lfm2 else AutoImageProcessor.from_pretrained(model_dir, backend="pil")
cases = []
sizes = [(300, 200), (150, 380), (97, 41), (1200, 700), (2000, 300), (640, 1600)] if lfm2 else [(300, 200), (150, 380), (64, 64), (97, 41)]
for width, height in sizes:
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([(7 * x + 3 * y + 50 * c + (x * y) % 31) % 256 for c in range(3)], axis=-1).astype(np.uint8)
    if lfm2:
        out = processor(images=[Image.fromarray(rgb)], return_tensors="np", return_row_col_info=True)
        ps = processor.encoder_patch_size
        views, chw = [], []
        for patches, (gh, gw) in zip(np.asarray(out["pixel_values"], dtype=np.float32), np.asarray(out["spatial_shapes"])):
            # The patches (row-major, each flattened (y, x, channel)) back to a channels-first view.
            view = patches[:gh * gw].reshape(gh, gw, ps, ps, 3).transpose(4, 0, 2, 1, 3).reshape(3, gh * ps, gw * ps)
            views.append([int(gw * ps), int(gh * ps)])
            chw.append(view.reshape(-1))
        pixels = np.concatenate(chw)
        case = {"width": width, "height": height, "pixels": [round(float(v), 6) for v in pixels], "views": views,
                "rows": int(np.asarray(out["image_rows"]).reshape(-1)[0]), "cols": int(np.asarray(out["image_cols"]).reshape(-1)[0])}
        cases.append(case)
        print(width, height, views, case["rows"], case["cols"])
        continue
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
