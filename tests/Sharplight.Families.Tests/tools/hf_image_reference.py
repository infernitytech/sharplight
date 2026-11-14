"""Image references for tiny Qwen-VL checkpoints from Hugging Face transformers (fp32).

Usage: python hf_image_reference.py <model_dir> [<width> <height>] [--new-tokens N]

Feeds a synthetic, already normalized image (value(y, x, c) = sin(0.031·x + 0.017·(c + 1)·y + 0.5·c), the same formula as the C# tests)
straight into the vision tower, bypassing resizing and normalization, so that only the tower and the language model are compared. Writes
<model_dir>/reference_image.json with the merged-token embeddings, the top-10 logits of the last prompt row of
"<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\\n<|im_start|>assistant\\n" and the greedy
continuation. "prompt_ids" holds one <|image_pad|> placeholder per image (the runtime expands it).
"""
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model_dir = args[0]
width, height = (int(args[1]), int(args[2])) if len(args) >= 3 else (280, 168)
new_tokens = int(sys.argv[sys.argv.index("--new-tokens") + 1]) if "--new-tokens" in sys.argv else 16

config = AutoConfig.from_pretrained(model_dir)
model = getattr(transformers, config.architectures[0]).from_pretrained(model_dir, dtype=torch.float32).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
vc = config.vision_config
ps, m, tp = vc.patch_size, vc.spatial_merge_size, vc.temporal_patch_size

y, x = np.mgrid[0:height, 0:width].astype(np.float64)
image = np.stack([np.sin(0.031 * x + 0.017 * (c + 1) * y + 0.5 * c) for c in range(3)], axis=0).astype(np.float32)   # [C, H, W]
gh, gw = height // ps, width // ps
frames = np.repeat(image[None], tp, axis=0)                                                           # [T, C, H, W]
patches = frames.reshape(1, tp, 3, gh // m, m, ps, gw // m, m, ps).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
pixel_values = torch.from_numpy(np.ascontiguousarray(patches).reshape(gh * gw, 3 * tp * ps * ps))
grid = torch.tensor([[1, gh, gw]])
tokens = gh * gw // (m * m)

with torch.no_grad():
    out = model.model.visual(pixel_values, grid_thw=grid)
    embeddings = out.pooler_output if hasattr(out, "pooler_output") else out
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
    expanded = []
    for t in prompt_ids:
        expanded.extend([pad] * tokens if t == pad else [t])
    input_ids = torch.tensor([expanded])
    mm = (input_ids == pad).long()   # token types, 1 = image row (the processors return them as mm_token_type_ids)
    logits = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=mm).logits[0, -1]
    top = torch.topk(logits, 10)
    generated = model.generate(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid, mm_token_type_ids=mm, max_new_tokens=new_tokens, do_sample=False,
                               min_new_tokens=new_tokens)[0, input_ids.shape[1]:].tolist()

result = {
    "width": width, "height": height, "prompt_ids": prompt_ids, "image_tokens": tokens,
    "embeddings": [round(v, 7) for v in embeddings.reshape(-1).tolist()],
    "top_ids": top.indices.tolist(), "top_logits": [round(v, 5) for v in top.values.tolist()], "greedy": generated,
}
with open(os.path.join(model_dir, "reference_image.json"), "w") as f:
    json.dump(result, f)
print("top", list(zip(result["top_ids"], result["top_logits"]))[:5])
print("greedy", generated)
