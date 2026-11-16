#!/usr/bin/env python3
"""Reference outputs of timm image classifiers for the Sharplight vision tests.

Loads MODEL_DIR/model.safetensors (with the architecture and pretrained_cfg of MODEL_DIR/config.json) into timm, runs IMAGE through timm's own evaluation transform and
writes `sharplight-sample.png` (the image) and `sharplight-reference.json` (logits, pooled features) into MODEL_DIR, in the layout of `vision-reference.py`.

Usage: timm-reference.py MODEL_DIR IMAGE
"""
import json
import os
import sys

import timm
import torch
from PIL import Image


def main() -> None:
    model_dir, image_path = sys.argv[1:3]
    config = json.load(open(os.path.join(model_dir, "config.json")))
    model = timm.create_model(config["architecture"], pretrained=True, pretrained_cfg_overlay={"file": os.path.join(model_dir, "model.safetensors")}).eval()
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    image = Image.open(image_path).convert("RGB")
    image.save(os.path.join(model_dir, "sharplight-sample.png"))
    image = Image.open(os.path.join(model_dir, "sharplight-sample.png")).convert("RGB")
    pixels = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model.forward_head(model.forward_features(pixels), pre_logits=True)
        logits = model.get_classifier()(features)
    result = {"model_type": config["architecture"], "pixel_sample": pixels.flatten()[:16].tolist(), "logits": logits[0].tolist(), "pooled": features[0].tolist()}
    with open(os.path.join(model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})


if __name__ == "__main__":
    main()
