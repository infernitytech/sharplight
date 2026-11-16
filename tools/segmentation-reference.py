#!/usr/bin/env python3
"""Reference outputs of SAM 2 (transformers, CPU) for the Sharplight segmentation tests.

Writes `sharplight-reference.json` into the model directory: the sample image, for several prompts (points, points with a negative
one, a box) the low-resolution mask logits, IoU scores, object score and the post-processed binary masks (packed bits, base64).

Usage: segmentation-reference.py MODEL_DIR IMAGE
"""
import argparse
import base64
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

PROMPTS = [
    {"name": "point", "points": [[450, 200]], "labels": [1]},
    {"name": "points", "points": [[450, 200], [150, 300]], "labels": [1, 0]},
    {"name": "box", "box": [350, 25, 640, 375]},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image")
    args = parser.parse_args()
    sample = os.path.join(args.model_dir, "sharplight-sample.png")
    Image.open(args.image).convert("RGB").save(sample)
    image = Image.open(sample).convert("RGB")
    model = Sam2Model.from_pretrained(args.model_dir).eval()
    processor = Sam2Processor.from_pretrained(args.model_dir)
    result = {"image_size": [image.width, image.height], "prompts": []}
    for prompt in PROMPTS:
        kwargs = {}
        if "points" in prompt:
            kwargs["input_points"] = [[prompt["points"]]]
            kwargs["input_labels"] = [[prompt["labels"]]]
        if "box" in prompt:
            kwargs["input_boxes"] = [[prompt["box"]]]
        inputs = processor(images=image, return_tensors="pt", **kwargs)
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(out.pred_masks, inputs["original_sizes"], binarize=True)[0][0]  # [3, H, W]
        entry = dict(prompt)
        entry["pixel_sample"] = inputs["pixel_values"].flatten()[:16].tolist()
        entry["low_res_masks"] = out.pred_masks[0, 0].flatten().tolist()
        entry["iou_scores"] = out.iou_scores[0, 0].tolist()
        entry["object_score"] = out.object_score_logits[0, 0].flatten().tolist()
        entry["masks"] = [base64.b64encode(np.packbits(m.numpy().astype(np.uint8).flatten()).tobytes()).decode() for m in masks]
        entry["mask_areas"] = [int(m.sum()) for m in masks]
        result["prompts"].append(entry)
        print(prompt["name"], entry["iou_scores"], entry["mask_areas"], entry["object_score"])
    with open(os.path.join(args.model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
