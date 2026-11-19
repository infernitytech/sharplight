#!/usr/bin/env python3
"""Reference outputs of GLM-OCR (transformers, CPU, greedy, no multi-token prediction) for the Sharplight tests.

Renders a few lines of printed text (or reads IMAGE) into `sharplight-sample.png` in the model directory, runs the chat template with the prompt
"Text Recognition:" and writes `sharplight-reference.json` (prompt ids, image grid, first-step top logits, generated ids and text).

Usage: glm-ocr-reference.py MODEL_DIR [IMAGE]
"""
import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

LINES = ["Sharplight reads printed text", "from document images.", "Invoice 2026-1118, total 1,234.50 EUR"]


def main() -> None:
    model_dir = sys.argv[1]
    sample = os.path.join(model_dir, "sharplight-sample.png")
    if len(sys.argv) > 2:
        Image.open(sys.argv[2]).convert("RGB").save(sample)
    else:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 30)
        image = Image.new("RGB", (640, 170), "white")
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(LINES):
            draw.text((16, 12 + 50 * i), line, fill="black", font=font)
        image.save(sample)
    image = Image.open(sample).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = GlmOcrForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Text Recognition:"}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        first = model(**inputs).logits[0, -1]
        ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    new = ids[0, inputs["input_ids"].shape[1]:].tolist()
    text = processor.decode(new, skip_special_tokens=True)
    top = torch.topk(first, 10)
    result = {
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "grid": inputs["image_grid_thw"][0].tolist(),
        "pixel_sample": inputs["pixel_values"].flatten()[:16].tolist(),
        "top_ids": top.indices.tolist(),
        "top_logits": top.values.tolist(),
        "token_ids": new,
        "text": text,
    }
    # The other recognition tasks of the model card, same image (greedy).
    result["tasks"] = {}
    for name, prompt in [("formula", "Formula Recognition:"), ("table", "Table Recognition:")]:
        msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        task_inputs = processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        task_inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(**task_inputs, max_new_tokens=64, do_sample=False)
        result["tasks"][name] = processor.decode(out[0, task_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(repr(text), new, result["grid"], result["tasks"])
    with open(os.path.join(model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
