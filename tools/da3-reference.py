#!/usr/bin/env python3
"""Reference outputs of Depth Anything 3 (the authors' network, `depth_anything_3`, CPU) for the Sharplight depth tests.

Needs a checkout of https://github.com/ByteDance-Seed/Depth-Anything-3 (SRC = its `src` folder) and torch, omegaconf, addict, einops. Resizes IMAGE to a
SIZE x SIZE canvas (bicubic, ImageNet statistics, like the DPT processor with keep_aspect_ratio off), runs the single-view network and writes
`sharplight-sample.png` and `sharplight-reference.json` (depth, confidence) next to the weights.

Usage: da3-reference.py SRC MODEL_DIR IMAGE [SIZE]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def main() -> None:
    src, model_dir, image_path = sys.argv[1:4]
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 504
    sys.path.insert(0, src)
    from depth_anything_3.cfg import create_object
    from omegaconf import OmegaConf

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)["config"]
    cfg = OmegaConf.create(cfg)
    net = create_object(cfg)
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    missing = net.load_state_dict(state, strict=False)
    print("missing", [k for k in missing.missing_keys][:8], "unexpected", missing.unexpected_keys[:8])
    net.eval()

    image = Image.open(image_path).convert("RGB")
    sample = os.path.join(model_dir, "sharplight-sample.png")
    image.save(sample)
    image = Image.open(sample).convert("RGB")
    resized = image.resize((size, size), Image.BICUBIC)
    pixels = (np.asarray(resized, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(pixels).permute(2, 0, 1)[None, None]
    with torch.no_grad():
        out = net(x)
    depth = out["depth"][0, 0]
    conf = out["depth_conf"][0, 0]
    result = {"depth_shape": list(depth.shape), "depth": depth.flatten().tolist(), "conf": conf.flatten().tolist()}
    print(list(depth.shape), float(depth.min()), float(depth.max()))
    with open(os.path.join(model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
