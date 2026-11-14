"""Image-processor reference of the DeepSeek-OCR family (transformers' UnlimitedOcrImageProcessor, the same steps as DeepSeek-OCR's remote code).

Usage: python hf_ocr_preprocess_reference.py <model_dir> <width> <height>

Draws the 8-bit test pattern rgb(x, y) = ((7x + 13y) mod 256, (3x + 5y + 80) mod 256, (x·y / 64) mod 256) of the given size, runs the model's
preprocessor_config.json through the processor and writes <model_dir>/reference_preprocess_<width>x<height>.bin: int32 columns, rows, then the
float32 global view [3, size, size] and the tiles [n, 3, tile, tile] in raster order. Then runs the prompt "<image>\nFree OCR." with the image
through the processor and the model (greedy, 12 tokens) and writes the prompt ids and the generated ids to
<model_dir>/reference_generate_<width>x<height>.json.
"""
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.models.unlimited_ocr.image_processing_unlimited_ocr import UnlimitedOcrImageProcessor
from transformers.models.unlimited_ocr.processing_unlimited_ocr import UnlimitedOcrProcessor

model_dir, width, height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
x = np.arange(width)[None, :]
y = np.arange(height)[:, None]
image = np.stack([(7 * x + 13 * y) % 256, (3 * x + 5 * y + 80) % 256, (x * y // 64) % 256], axis=0).astype(np.uint8)
processor = UnlimitedOcrImageProcessor.from_pretrained(model_dir)
out = processor(images=[torch.from_numpy(image)], return_tensors="pt")
columns, rows = out["patches_grid"][0]
parts = [np.array([columns, rows], dtype=np.int32).tobytes(), out["pixel_values"][0].numpy().astype(np.float32).tobytes()]
if "pixel_values_local" in out:
    parts.append(torch.stack(list(out["pixel_values_local"])).numpy().astype(np.float32).tobytes())
with open(os.path.join(model_dir, f"reference_preprocess_{width}x{height}.bin"), "wb") as f:
    f.write(b"".join(parts))
print("grid", columns, rows, "tiles", out["num_local_patches"][0])

config = AutoConfig.from_pretrained(model_dir)
model = getattr(transformers, config.architectures[0]).from_pretrained(model_dir, dtype=torch.float32).eval()
full = UnlimitedOcrProcessor(processor, AutoTokenizer.from_pretrained(model_dir))
inputs = full(images=[torch.from_numpy(image)], text=["<image>\nFree OCR."], return_tensors="pt")
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=12, do_sample=False)
prompt_ids = inputs["input_ids"][0].tolist()
with open(os.path.join(model_dir, f"reference_generate_{width}x{height}.json"), "w") as f:
    json.dump({"prompt": "<image>\nFree OCR.", "prompt_ids": prompt_ids, "greedy": generated[0, len(prompt_ids):].tolist()}, f)
print("prompt rows", len(prompt_ids), "greedy", generated[0, len(prompt_ids):].tolist())
