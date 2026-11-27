"""Image references for the tiny Gemma 3, Idefics3 and LFM2-VL checkpoints of mk_tiny_siglip_vl.py from Hugging Face transformers (fp32).

Usage: python siglip_vl_reference.py <model_dir> [--new-tokens N]

Feeds synthetic, already normalized views (value(y, x, c) = sin(0.031·x + 0.017·(c + 1)·y + 0.5·c), the formula of the C# tests) straight into
the tower, bypassing resizing and normalization, so that only the tower, the prompt layout and the language model are compared:
- gemma3: one 64×64 image; the prompt expands the chat template's <start_of_image> to <start_of_image>, the image rows and <end_of_image>
  (llama.cpp's layout; transformers' processor adds "\\n\\n" around it). token_type_ids mark the image rows, so the rows of the image attend
  to each other (transformers' blockwise mask, llama.cpp's non-causal image batch; issue #98).
  The input projection is rounded to f16 first, as llama.cpp's converter stores it.
- idefics3: a 2×1 tile grid cut from a 128×64 image plus an overview whose values are the negated 64×64 pattern; the prompt layout is the
  processor's own (Idefics3Processor.replace_image_token).
- lfm2_vl: a 2×1 grid of 128-pixel tiles cut from a 256×128 image plus a 96×32 thumbnail (negated pattern; its 2×6 patch grid resamples the 4×4
  position grid down along the rows and up along the columns); the views go through the NaFlex padding and attention mask as the processor
  would pass them, the prompt layout is Lfm2VlProcessor._build_image_tokens'.
Writes <model_dir>/reference_image.json: the embeddings (all views, in output order), the top-10 logits of the last prompt row, the greedy
continuation and "prompt_ids" with one placeholder per image (the runtime expands it).
"""
import json
import os
import sys

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

model_dir = sys.argv[1]
new_tokens = int(sys.argv[sys.argv.index("--new-tokens") + 1]) if "--new-tokens" in sys.argv else 16
config = AutoConfig.from_pretrained(model_dir)
model = getattr(transformers, config.architectures[0]).from_pretrained(model_dir, dtype=torch.float32).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir)


def sine(width, height):
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    return np.stack([np.sin(0.031 * x + 0.017 * (c + 1) * y + 0.5 * c) for c in range(3)], axis=0).astype(np.float32)   # [C, H, W]


result = {}
with torch.no_grad():
    if config.model_type == "gemma3":
        # llama.cpp's converter always stores mm.input_projection in f16; the reference uses the same rounded weight.
        w = model.model.multi_modal_projector.mm_input_projection_weight
        w.copy_(w.half().float())
        side = config.vision_config.image_size
        pixel_values = torch.from_numpy(sine(side, side))[None]
        embeddings = model.model.get_image_features(pixel_values).pooler_output.reshape(-1, config.text_config.hidden_size)
        text = "<bos><start_of_turn>user\n<start_of_image>Describe this image.<end_of_turn>\n<start_of_turn>model\n"
        prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
        boi, eoi = config.boi_token_index, config.eoi_token_index
        expanded = []
        for t in prompt_ids:
            expanded.extend([boi] + [config.image_token_index] * config.mm_tokens_per_image + [eoi] if t == boi else [t])
        input_ids = torch.tensor([expanded])
        token_type_ids = (input_ids == config.image_token_index).long()
        kwargs = dict(pixel_values=pixel_values, token_type_ids=token_type_ids)
        result.update(width=side, height=side)
    elif config.model_type == "lfm2_vl":
        ps = config.vision_config.patch_size
        tile, columns, rows = 128, 2, 1
        tw, th = 96, 32
        canvas = sine(columns * tile, rows * tile)
        views = [canvas[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] for r in range(rows) for c in range(columns)] + [-sine(tw, th)]
        patched = []
        shapes = []
        for v in views:
            c_, h_, w_ = v.shape
            patches = v.reshape(c_, h_ // ps, ps, w_ // ps, ps).transpose(1, 3, 2, 4, 0).reshape((h_ // ps) * (w_ // ps), ps * ps * c_)
            patched.append(patches)
            shapes.append([h_ // ps, w_ // ps])
        longest = max(p.shape[0] for p in patched)
        pixel_values = torch.zeros(len(views), longest, ps * ps * 3)
        mask = torch.zeros(len(views), longest, dtype=torch.int32)
        for i, p in enumerate(patched):
            pixel_values[i, :p.shape[0]] = torch.from_numpy(np.ascontiguousarray(p))
            mask[i, :p.shape[0]] = 1
        spatial = torch.tensor(shapes)
        embeddings = torch.cat(list(model.model.get_image_features(pixel_values=pixel_values, spatial_shapes=spatial, pixel_attention_mask=mask).pooler_output))
        processor = AutoProcessor.from_pretrained(model_dir)
        text = "<|startoftext|><|im_start|>user\n<image>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
        per_tile = (tile // ps // 2) ** 2
        thumb = (th // ps // 2) * (tw // ps // 2)
        expanded_text = text.replace("<image>", processor._build_image_tokens(rows, columns, per_tile, thumb, True, True))
        input_ids = torch.tensor([tokenizer(expanded_text, add_special_tokens=False).input_ids])
        kwargs = dict(pixel_values=pixel_values, spatial_shapes=spatial, pixel_attention_mask=mask)
        result.update(width=columns * tile, height=rows * tile, tile_columns=columns, tile_rows=rows, thumbnail_width=tw, thumbnail_height=th,
                      expanded_ids=input_ids[0].tolist())
    else:
        tile = config.vision_config.image_size
        columns, rows = 2, 1
        canvas = sine(columns * tile, rows * tile)
        views = [canvas[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] for r in range(rows) for c in range(columns)] + [-sine(tile, tile)]
        pixel_values = torch.from_numpy(np.ascontiguousarray(np.stack(views)))[None]                     # [1, views, C, T, T]
        embeddings = torch.cat(model.model.get_image_features(pixel_values).pooler_output)
        processor = AutoProcessor.from_pretrained(model_dir)
        text = "<|im_start|>User:<image>Describe this image.<end_of_utterance>\nAssistant:"
        prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
        expanded_text = text.replace("<image>", processor.replace_image_token({"rows": [[rows]], "cols": [[columns]]}, 0))
        input_ids = torch.tensor([tokenizer(expanded_text, add_special_tokens=False).input_ids])
        kwargs = dict(pixel_values=pixel_values, pixel_attention_mask=torch.ones(1, len(views), tile, tile, dtype=torch.bool))
        result.update(width=columns * tile, height=rows * tile, tile_columns=columns, tile_rows=rows, expanded_ids=input_ids[0].tolist())

    logits = model(input_ids=input_ids, **kwargs).logits[0, -1]
    top = torch.topk(logits, 10)
    generated = model.generate(input_ids=input_ids, **kwargs, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False)[0, input_ids.shape[1]:].tolist()

result.update({
    "prompt_ids": prompt_ids, "image_tokens": int(embeddings.shape[0]),
    "embeddings": [round(v, 7) for v in embeddings.reshape(-1).tolist()],
    "top_ids": top.indices.tolist(), "top_logits": [round(v, 5) for v in top.values.tolist()], "greedy": generated,
})
with open(os.path.join(model_dir, "reference_image.json"), "w") as f:
    json.dump(result, f)
print("tokens", result["image_tokens"], "top", list(zip(result["top_ids"], result["top_logits"]))[:5])
print("greedy", generated)
