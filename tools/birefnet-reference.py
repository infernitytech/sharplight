#!/usr/bin/env python3
"""Reference matte of BiRefNet / RMBG-2.0 (the authors' remote code, CPU) for the Sharplight vision tests.

Writes `sharplight-reference.json` into the model directory: the pre-sigmoid matte logits of the last decoder output at the model size, and the
image as `sharplight-sample.png`.

Usage: birefnet-reference.py MODEL_DIR IMAGE [--size 1024]
Python environment: torch (CPU), transformers, torchvision, timm, einops, kornia (a venv, never the system Python).
"""
import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    sample = os.path.join(args.model_dir, "sharplight-sample.png")
    Image.open(args.image).convert("RGB").save(sample)
    image = Image.open(sample).convert("RGB")
    model = AutoModelForImageSegmentation.from_pretrained(args.model_dir, trust_remote_code=True).float().eval()
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pixels = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(pixels)[-1]
    logits = out[0, 0]
    result = {"size": args.size, "pixel_sample": pixels.flatten()[:16].tolist(), "logits": [round(v, 5) for v in logits.flatten().tolist()],
              "logits_shape": list(logits.shape)}
    with open(os.path.join(args.model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)
    print("range", logits.min().item(), logits.max().item())


if __name__ == "__main__":
    main()
