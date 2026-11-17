#!/usr/bin/env python3
"""Reference detections of an Ultralytics YOLO ONNX export for the Sharplight vision tests.

Runs the export with onnxruntime on the letterboxed image (longest side scaled into the input with Pillow's bilinear filter, gray 114 padding, 0..1 range: the
preprocessing Sharplight stores with the model) and decodes it the way Ultralytics' predictor does (best class per cell, score threshold, class-aware
torchvision NMS at IoU 0.7, instance masks from the prototypes upsampled to the input, cropped at the box, cut at logit 0). Writes `sharplight-reference.json` and
`sharplight-sample.png` next to the model.

Usage: yolo-reference.py MODEL_DIR IMAGE [--conf 0.25] [--kind detect|segment|end2end]
Python environment: torch (CPU), torchvision, onnxruntime, pillow, numpy (a venv, never the system Python).
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--kind", default="detect", choices=["detect", "segment", "end2end"])
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    sample = os.path.join(args.model_dir, "sharplight-sample.png")
    image.save(sample)
    image = Image.open(sample).convert("RGB")
    session = ort.InferenceSession(os.path.join(args.model_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    meta = session.get_inputs()[0]
    size_h, size_w = meta.shape[2], meta.shape[3]
    w, h = image.size
    ratio = min(size_w / w, size_h / h)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = image.resize((rw, rh), Image.BILINEAR)
    canvas = Image.new("RGB", (size_w, size_h), (114, 114, 114))
    left, top = (size_w - rw) // 2, (size_h - rh) // 2
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    outputs = session.run(None, {meta.name: pixels})
    result = {"letterbox": [rw, rh, left, top], "kind": args.kind, "conf": args.conf, "iou": args.iou, "output_sample": outputs[0].flatten()[:16].tolist()}

    detections = []
    if args.kind == "end2end":
        rows = outputs[0][0]
        for row in rows:
            if row[4] >= args.conf:
                x1, y1, x2, y2 = row[:4]
                detections.append({"class": int(row[5]), "score": float(row[4]), "box": [float((x1 - left) / ratio), float((y1 - top) / ratio), float((x2 - left) / ratio), float((y2 - top) / ratio)]})
    else:
        pred = torch.from_numpy(outputs[0][0]).T  # [cells, 4 + C (+ nm)]
        nm = outputs[1].shape[1] if args.kind == "segment" else 0
        classes = pred.shape[1] - 4 - nm
        scores, cls = pred[:, 4:4 + classes].max(1)
        keep = scores >= args.conf
        pred, scores, cls = pred[keep], scores[keep], cls[keep]
        xyxy = torch.cat([pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2], 1)
        kept = torchvision.ops.batched_nms(xyxy, scores, cls, args.iou)
        protos = torch.from_numpy(outputs[1][0]) if nm else None
        for i in kept.tolist():
            box = xyxy[i]
            entry = {"class": int(cls[i]), "score": float(scores[i]), "box": [float((box[0] - left) / ratio), float((box[1] - top) / ratio), float((box[2] - left) / ratio), float((box[3] - top) / ratio)]}
            entry["box"] = [max(0.0, entry["box"][0]), max(0.0, entry["box"][1]), min(float(w), entry["box"][2]), min(float(h), entry["box"][3])]
            if nm:
                coefficients = pred[i, 4 + classes:]
                mh, mw = protos.shape[1:]
                logits = (coefficients @ protos.reshape(nm, -1)).reshape(1, 1, mh, mw)
                logits = F.interpolate(logits, size=(size_h, size_w), mode="bilinear", align_corners=False)[0, 0]
                inside = torch.zeros_like(logits, dtype=torch.bool)
                inside[int(box[1].floor().clamp(0, size_h)):int(box[3].ceil().clamp(0, size_h)), int(box[0].floor().clamp(0, size_w)):int(box[2].ceil().clamp(0, size_w))] = True
                logits = logits[top:top + rh, left:left + rw]
                inside = inside[top:top + rh, left:left + rw]
                full = F.interpolate(logits[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
                inside_full = F.interpolate(inside[None, None].float(), size=(h, w), mode="nearest")[0, 0] > 0.5
                mask = (full > 0) & inside_full
                entry["mask_area"] = int(mask.sum())
                entry["mask_rle"] = np.packbits(mask.numpy().flatten()).tobytes().hex()
            detections.append(entry)
    result["detections"] = sorted(detections, key=lambda d: -d["score"])
    result["image_size"] = [w, h]
    with open(os.path.join(args.model_dir, "sharplight-reference.json"), "w") as f:
        json.dump(result, f)
    print(len(detections), "detections")
    for d in result["detections"]:
        print(d["class"], round(d["score"], 3), [round(v, 1) for v in d["box"]], d.get("mask_area"))


if __name__ == "__main__":
    main()
