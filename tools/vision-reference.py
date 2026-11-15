#!/usr/bin/env python3
"""Reference outputs of vision models for the Sharplight vision tests.

Runs a Hugging Face image model (transformers) on an image and writes `sharplight-reference.json` into the model directory: the
pooled image embedding, the class logits (classifiers) and, for dual-tower models, text embeddings and zero-shot logits. The image is
also stored as `sharplight-sample.png` there, so that Sharplight and the reference see the same pixels.

Usage: vision-reference.py MODEL_DIR IMAGE [--text "a cat" --text "a dog" ...]
Python environment: torch (CPU), transformers, pillow, numpy (a venv, never the system Python).
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageClassification, AutoModelForObjectDetection, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--size", type=int, default=0, help="resize to a square of this side (detection models with a fixed-size graph)")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image.save(os.path.join(args.model_dir, "sharplight-sample.png"))
    image = Image.open(os.path.join(args.model_dir, "sharplight-sample.png")).convert("RGB")
    config = AutoConfig.from_pretrained(args.model_dir)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    extra = {"size": {"height": args.size, "width": args.size}} if args.size else {}
    pixels = processor(images=image, return_tensors="pt", **extra)["pixel_values"]
    result: dict = {"model_type": config.model_type, "pixel_sample": pixels.flatten()[:16].tolist()}

    with torch.no_grad():
        if config.model_type in ("clip", "siglip"):
            model = AutoModel.from_pretrained(args.model_dir).eval()
            # Projected features computed from the towers (the return type of get_*_features differs between transformers versions).
            project_image = model.visual_projection if config.model_type == "clip" else (lambda x: x)
            project_text = model.text_projection if config.model_type == "clip" else (lambda x: x)
            result["image_embedding"] = project_image(model.vision_model(pixel_values=pixels).pooler_output)[0].tolist()
            if args.text:
                tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
                padding = "max_length" if config.model_type == "siglip" else True
                tokens = tokenizer(args.text, padding=padding, max_length=64 if config.model_type == "siglip" else None, return_tensors="pt")
                result["texts"] = args.text
                result["token_ids"] = tokens["input_ids"].tolist()
                result["text_embeddings"] = project_text(model.text_model(**tokens).pooler_output).tolist()
                outputs = model(pixel_values=pixels, **tokens)
                result["logits_per_image"] = outputs.logits_per_image[0].tolist()
        elif getattr(config, "architectures", None) and "ForObjectDetection" in config.architectures[0]:
            model = AutoModelForObjectDetection.from_pretrained(args.model_dir).eval()
            outputs = model(pixel_values=pixels)
            result["logits"] = outputs.logits[0].flatten().tolist()
            result["logits_shape"] = list(outputs.logits[0].shape)
            result["boxes"] = outputs.pred_boxes[0].flatten().tolist()
            found = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=[(image.height, image.width)])[0]
            result["detections"] = [
                {"label": model.config.id2label[int(l)], "score": float(sc), "box": [float(v) for v in b]}
                for sc, l, b in zip(found["scores"], found["labels"], found["boxes"])
            ]
        elif getattr(config, "architectures", None) and "ForImageClassification" in config.architectures[0]:
            model = AutoModelForImageClassification.from_pretrained(args.model_dir).eval()
            outputs = model(pixel_values=pixels, output_hidden_states=True)
            result["logits"] = outputs.logits[0].tolist()
            result["labels"] = [model.config.id2label[i] for i in range(len(model.config.id2label))]
            result["pooled"] = model.base_model(pixel_values=pixels).last_hidden_state[0, 0].tolist()
        else:
            model = AutoModel.from_pretrained(args.model_dir).eval()
            out = model(pixel_values=pixels)
            result["pooled"] = out.last_hidden_state[0, 0].tolist()
            result["patch_mean"] = out.last_hidden_state[0, 1 + getattr(config, "num_register_tokens", 0):].mean(0).tolist()

    with open(os.path.join(args.model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})


if __name__ == "__main__":
    main()
