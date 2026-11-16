#!/usr/bin/env python3
"""Reference outputs of Florence-2 (transformers, CPU) for the Sharplight vision tests.

Writes `sharplight-reference.json` into the model directory: for the tasks <OCR>, <CAPTION>, <DETAILED_CAPTION>, <OD> the prompt, the projected image features of the
vision tower (first task only), the encoder output of the first task, the greedy token ids (num_beams=1, the checkpoint's no-repeat n-gram size kept) and the decoded text.
The image is stored as `sharplight-sample.png`.

Usage: florence-reference.py MODEL_DIR IMAGE [--tag TAG]   (TAG: files sharplight-reference-TAG.json and sharplight-sample-TAG.png)
Python environment: torch (CPU), transformers, pillow (a venv, never the system Python).
"""
import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

TASKS = ["<OCR>", "<CAPTION>", "<DETAILED_CAPTION>", "<OD>"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    suffix = f"-{args.tag}" if args.tag else ""
    sample = os.path.join(args.model_dir, f"sharplight-sample{suffix}.png")
    Image.open(args.image).convert("RGB").save(sample)
    image = Image.open(sample).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Florence2ForConditionalGeneration.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    result = {"tasks": []}
    for index, task in enumerate(TASKS):
        inputs = processor(text=task, images=image, return_tensors="pt")
        entry = {"task": task, "input_ids": inputs["input_ids"][0].tolist(), "pixel_sample": inputs["pixel_values"].flatten()[:16].tolist()}
        with torch.no_grad():
            if index == 0:
                features = model.model.get_image_features(inputs["pixel_values"]).pooler_output[0]
                entry["image_features"] = [round(v, 6) for v in features.flatten().tolist()]
                embeds = model.model.get_input_embeddings()(inputs["input_ids"])
                special = inputs["input_ids"] == model.config.image_token_id
                embeds = embeds.masked_scatter(special.unsqueeze(-1), features.unsqueeze(0))
                encoded = model.model.language_model.encoder(inputs_embeds=embeds).last_hidden_state[0]
                entry["encoder_output"] = [round(v, 6) for v in encoded.flatten().tolist()]
            generated = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=128, num_beams=1, do_sample=False)
        ids = generated[0].tolist()
        entry["generated_ids"] = ids
        entry["text"] = processor.batch_decode(generated, skip_special_tokens=True)[0]
        entry["text_raw"] = processor.batch_decode(generated, skip_special_tokens=False)[0]
        result["tasks"].append(entry)
        print(task, repr(entry["text"]), len(ids))
    with open(os.path.join(args.model_dir, f"sharplight-reference{suffix}.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
