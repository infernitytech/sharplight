"""Image references for the tiny DeepSeek-OCR family checkpoints (mk_tiny_deepseek_ocr.py) from transformers.

Usage: python hf_ocr_image_reference.py <model_dir> <columns> <rows> [<out_name>]

Feeds already resized, normalized views into the model, bypassing the image processor: the global view is the synthetic image of
hf_image_reference.py (sin(0.031·x + 0.017·(c+1)·y + 0.5·c)) at the processor's global size, the local tiles are cut from the same pattern
drawn over a canvas of columns × rows tiles (no tiles when columns·rows is 1). Writes <model_dir>/<out_name> (default reference_image.json):
the image rows the language model receives (embeddings), the prompt ids (BOS, one <image> per row, "\\nFree OCR."), the top logits of the
prompt's last position and 12 greedy ids.
"""
import json
import math
import os
import sys

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

model_dir, columns, rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out_name = sys.argv[4] if len(sys.argv) > 4 else "reference_image.json"
config = AutoConfig.from_pretrained(model_dir)
model = getattr(transformers, config.architectures[0]).from_pretrained(model_dir, dtype=torch.float32).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
with open(os.path.join(model_dir, "preprocessor_config.json")) as f:
    pre = json.load(f)
size, tile = pre["size"]["height"], pre["tile_size"]


def sine(width, height):
    x = torch.arange(width, dtype=torch.float64)[None, :]
    y = torch.arange(height, dtype=torch.float64)[:, None]
    return torch.stack([torch.sin(0.031 * x + 0.017 * (c + 1) * y + 0.5 * c) for c in range(3)]).float()


global_view = sine(size, size)[None]
kwargs = {}
tiles = columns * rows if columns * rows > 1 else 0
if tiles:
    canvas = sine(columns * tile, rows * tile)
    local = torch.stack([canvas[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] for r in range(rows) for c in range(columns)])
    kwargs = dict(pixel_values_local=local, num_local_patches=[tiles])
    if "UnlimitedOcr" in config.architectures[0]:
        kwargs["patches_grid"] = [[columns, rows]]
with torch.no_grad():
    features = model.model.get_image_features(global_view, **kwargs).pooler_output[0]
    image_id = config.image_token_id
    text = tokenizer("\nFree OCR.", add_special_tokens=False).input_ids
    ids = [tokenizer.bos_token_id] + [image_id] * features.shape[0] + text
    embeds = model.model.get_input_embeddings()(torch.tensor([ids]))
    mask = torch.tensor([ids]) == image_id
    embeds[mask] = features
    generated = []
    prompt_logits = None
    for _ in range(12):
        logits = model(inputs_embeds=embeds).logits[0, -1]
        if prompt_logits is None:
            prompt_logits = logits
        nxt = int(logits.argmax())
        generated.append(nxt)
        embeds = torch.cat([embeds, model.model.get_input_embeddings()(torch.tensor([[nxt]]))], dim=1)
top = torch.topk(prompt_logits, 5)
out = {"size": size, "tile_size": tile, "columns": columns, "rows": rows, "image_tokens": features.shape[0],
       "embeddings": features.flatten().tolist(), "prompt_ids": ids, "top_ids": top.indices.tolist(), "top_logits": top.values.tolist(),
       "greedy": generated}
with open(os.path.join(model_dir, out_name), "w") as f:
    json.dump(out, f)
print("image rows", features.shape[0], "greedy", generated, "top", top.indices.tolist())
