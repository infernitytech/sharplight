"""Writes a tiny random PaddleOCR-VL (HF paddleocr_vl) checkpoint for the family tests.

Usage: python mk_tiny_paddleocr_vl.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds an ERNIE-4.5 tokenizer (e.g. ERNIE-4.5-0.3B-PT, vocabulary 103424). The language model is ERNIE-4.5 (no biases) with
sectioned, non-interleaved M-RoPE ([2, 3, 3] of a 16-wide head); the SigLIP-style vision tower is tiny (2 layers). Weights are drawn wider
than the default init so the greedy continuation depends on the input.
"""
import json
import os
import shutil
import sys

import torch
from transformers import PaddleOCRVLConfig, PaddleOCRVLForConditionalGeneration

tokenizer_dir, out = sys.argv[1], sys.argv[2]
torch.manual_seed(43)
text = dict(vocab_size=103424, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, rms_norm_eps=1e-5, max_position_embeddings=1024, tie_word_embeddings=False, use_bias=False,
            rope_parameters={"rope_type": "default", "rope_theta": 500000.0, "mrope_section": [2, 3, 3]})
vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14)
config = PaddleOCRVLConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)
config.architectures = ["PaddleOCRVLForConditionalGeneration"]
model = PaddleOCRVLForConditionalGeneration(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1 and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif p.dim() >= 2:
            p.normal_(0.0, 1.0 if "embed_tokens" in name else 1.5 / p.shape[-1] ** 0.5)
        else:
            p.normal_(0.0, 0.1)
    model.lm_head.weight.mul_(4.0)   # wider logit margins for the greedy comparison
model.save_pretrained(out, safe_serialization=True)
# The processor defaults; llama.cpp's mmproj converter needs min_pixels / max_pixels from this file.
with open(os.path.join(out, "preprocessor_config.json"), "w") as f:
    json.dump({"image_processor_type": "PaddleOCRVLImageProcessor", "image_mean": [0.48145466, 0.4578275, 0.40821073],
               "image_std": [0.26862954, 0.26130258, 0.27577711], "min_pixels": 147456, "max_pixels": 2359296, "patch_size": 14,
               "temporal_patch_size": 1, "merge_size": 2}, f, indent=2)
for f in os.listdir(tokenizer_dir):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "added_tokens.json"):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
print("wrote", out)
