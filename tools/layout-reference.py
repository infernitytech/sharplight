#!/usr/bin/env python3
"""Reference outputs of PP-DocLayoutV3 (transformers, CPU) for the Sharplight tests.

Renders a synthetic 800 x 800 document page (title, two text columns, a table, a figure with a caption, a page number) into `sharplight-sample.png` in the
model directory (or reads IMAGE, resized to 800 x 800 so both sides see the same pixels), runs the model and writes `sharplight-reference.json`: the raw
logits and boxes of all queries, the final detections of the image processor (label, score, box in pixels, reading order) and the mask areas of the kept
detections (the processor's 0.5 threshold on the 200 x 200 mask map).

Usage: layout-reference.py MODEL_DIR [IMAGE]
"""
import json
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
WORDS = "the layout of a page decides how its parts are read and which text belongs together in a column".split()


def paragraph(draw, x, y, width, lines, font):
    for i in range(lines):
        text = " ".join(WORDS[(i * 3 + j) % len(WORDS)] for j in range(9))
        draw.text((x, y + i * 20), text, fill="black", font=font)


def render(path):
    image = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(image)
    body = ImageFont.truetype(FONT, 14)
    draw.text((60, 40), "A Sample Document Page", fill="black", font=ImageFont.truetype(BOLD, 30))
    draw.text((60, 100), "1. Introduction", fill="black", font=ImageFont.truetype(BOLD, 18))
    paragraph(draw, 60, 130, 320, 9, body)
    paragraph(draw, 420, 130, 320, 9, body)
    draw.text((60, 340), "2. Results", fill="black", font=ImageFont.truetype(BOLD, 18))
    for r in range(5):
        for c in range(4):
            x, y = 60 + c * 100, 375 + r * 28
            draw.rectangle([x, y, x + 100, y + 28], outline="black")
            draw.text((x + 8, y + 6), f"{r * 4 + c + 1}.{c}", fill="black", font=body)
    draw.rectangle([480, 375, 740, 520], outline="black", width=2)
    draw.line([480, 520, 740, 375], fill="black", width=2)
    draw.text((490, 530), "Figure 1. A placeholder image.", fill="black", font=ImageFont.truetype(FONT, 12))
    paragraph(draw, 60, 560, 680, 8, body)
    draw.text((385, 760), "- 3 -", fill="black", font=body)
    image.save(path)


def main() -> None:
    model_dir = sys.argv[1]
    sample = os.path.join(model_dir, "sharplight-sample.png")
    if len(sys.argv) > 2:
        Image.open(sys.argv[2]).convert("RGB").resize((800, 800), Image.BICUBIC).save(sample)
    else:
        render(sample)
    image = Image.open(sample).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(model_dir, dtype=torch.float32).eval()
    inputs = processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=torch.tensor([[800, 800]]))[0]
    id2label = model.config.id2label
    masks = (outputs.out_masks[0].sigmoid() > 0.5)
    scores = outputs.logits[0].sigmoid()
    detections = []
    for score, label, box, order in zip(results["scores"], results["labels"], results["boxes"], results["order_seq"]):
        detections.append({"label": id2label[label.item()], "class": label.item(), "score": score.item(), "box": [round(v, 3) for v in box.tolist()], "order": order.item()})
    result = {
        "logits": outputs.logits[0].flatten().tolist(),
        "boxes": outputs.pred_boxes[0].flatten().tolist(),
        "order_logits_upper": [outputs.order_logits[0, i, i + 1:].tolist() for i in range(0, 300, 37)],
        "detections": detections,
        "mask_areas": {str(q): int(masks[q].sum()) for q in range(300) if scores[q].max() >= 0.5},  # per query with a class score of 0.5 or more
        "polygons": [None if p is None else p.tolist() for p in results["polygon_points"]],
    }
    for d in detections:
        print(d)
    with open(os.path.join(model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
