"""Writes a tiny random Qwen2-VL (HF qwen2_vl) or Qwen2.5-VL (qwen2_5_vl) checkpoint for the family tests.

Usage: python mk_tiny_qwen2_vl.py <qwen2_vl|qwen2_5_vl> <tokenizer_dir> <out_dir>

<tokenizer_dir> holds a Qwen2 tokenizer (e.g. Qwen2.5-0.5B-Instruct). The language model is Qwen2 with sectioned M-RoPE ([2, 3, 3] of a
16-wide head); the vision tower is tiny (2 layers; Qwen2.5-VL: windowed layer 0, full-attention layer 1). The Qwen2-VL processor's
preprocessor_config.json is written too (tools/hf_image_reference.py produces the image references). Weights are drawn wider than the default init so the greedy continuation
depends on the input.
"""
import json
import os
import shutil
import sys

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLConfig, Qwen2VLForConditionalGeneration

kind, tokenizer_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.manual_seed(29 if kind == "qwen2_vl" else 37)
text = dict(vocab_size=151936, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            rms_norm_eps=1e-6, max_position_embeddings=1024, tie_word_embeddings=False,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [2, 3, 3]})
if kind == "qwen2_vl":
    config = Qwen2VLConfig(text_config=text, vision_config=dict(depth=2, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2),
                           tie_word_embeddings=False)
    config.architectures = ["Qwen2VLForConditionalGeneration"]
    model = Qwen2VLForConditionalGeneration(config).eval()
else:
    config = Qwen2_5_VLConfig(text_config=text, vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                                                                   fullatt_block_indexes=[1]),
                              tie_word_embeddings=False)
    config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
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
for f in os.listdir(tokenizer_dir):
    if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt"):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
with open(os.path.join(out, "preprocessor_config.json"), "w") as f:
    json.dump({"image_processor_type": "Qwen2VLImageProcessor", "image_mean": [0.48145466, 0.4578275, 0.40821073],
               "image_std": [0.26862954, 0.26130258, 0.27577711], "min_pixels": 3136, "max_pixels": 12845056, "patch_size": 14,
               "temporal_patch_size": 2, "merge_size": 2}, f, indent=2)
print("wrote", out)
