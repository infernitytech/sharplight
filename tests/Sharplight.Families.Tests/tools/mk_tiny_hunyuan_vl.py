"""Writes a tiny random HunyuanOCR / Hunyuan-VL (HF hunyuan_vl) checkpoint for the family and vision tests.

Usage: python mk_tiny_hunyuan_vl.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds the HunyuanOCR tokenizer (tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja;
vocabulary 120818, image placeholder <｜hy_place▁holder▁no▁102｜> = 120120). The language model is Hunyuan dense (q/k RMS norms after the
rotary embedding, NTK alpha folded into the base) with XD-RoPE: four sections [4, 4, 4, 4] that partition the full 32-wide head (each half
of a rotation pair takes its own axis). The tower is tiny (2 layers, 32 wide, 2 heads, patch 16) with an 8x8 learned position grid (plus
the unused class row) resampled bilinearly, and the HunyuanVL perceiver: RMS norm, 2x2 convolution, GELU, 1x1 convolution, a newline column
per row, a linear projection, begin / end rows and a final RMS norm. Weights are drawn wider than the default init so outputs depend on
the input. Saved with transformers' names (model.language_model.*, model.vision_tower.*).
"""
import json
import os
import shutil
import sys

import torch
from transformers import HunYuanVLConfig, HunYuanVLForConditionalGeneration

tokenizer_dir, out = sys.argv[1], sys.argv[2]
torch.manual_seed(53)
text = dict(vocab_size=120818, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, rms_norm_eps=1e-5, max_position_embeddings=4096, attention_bias=False, tie_word_embeddings=True,
            bos_token_id=120000, eos_token_id=120020, eod_token_id=120020, pad_token_id=120002, sep_token_id=0,
            rope_parameters={"rope_type": "xdrope", "alpha": 1000.0, "rope_theta": 10000.0, "factor": 1.0, "xdrope_section": [4, 4, 4, 4]})
vision = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, patch_size=16, spatial_merge_size=2,
              max_image_size=128, rms_norm_eps=1e-5, hidden_act="gelu", out_hidden_size=64, text_hidden_size=64)
config = HunYuanVLConfig(text_config=text, vision_config=vision, image_token_id=120120, im_start_id=120118, im_end_id=120119,
                         im_newline_id=120121, tie_word_embeddings=True)
config.architectures = ["HunYuanVLForConditionalGeneration"]
model = HunYuanVLForConditionalGeneration(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if ("norm" in name or "rms" in name or "layer_norm" in name) and p.dim() == 1 and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif p.dim() >= 2:
            fan_in = p[0].numel()
            p.normal_(0.0, 1.0 if "embed_tokens" in name or "position_embedding" in name else (4.0 if "language_model" in name or name.startswith("model.layers") else 1.5) / fan_in ** 0.5)
        elif name.endswith(("image_newline", "image_begin", "image_end", "image_sep")):
            p.normal_(0.0, 1.0)
        else:
            p.normal_(0.0, 0.1)
model.save_pretrained(out, safe_serialization=True)
with open(os.path.join(out, "preprocessor_config.json"), "w") as f:
    json.dump({"image_processor_type": "HunYuanVLImageProcessor", "processor_class": "HunYuanVLProcessor",
               "image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
               "min_pixels": 16384, "max_pixels": 1048576, "patch_size": 16, "temporal_patch_size": 1, "merge_size": 2,
               "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True, "resample": 1,
               "rescale_factor": 0.00392156862745098}, f, indent=2)
for f in os.listdir(tokenizer_dir):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
print("wrote", out)
