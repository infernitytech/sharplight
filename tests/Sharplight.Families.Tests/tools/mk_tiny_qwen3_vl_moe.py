"""Writes a tiny random Qwen3-VL-MoE (HF qwen3_vl_moe) checkpoint for the family tests.

Usage: python mk_tiny_qwen3_vl_moe.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds a Qwen3-VL tokenizer and processor files (e.g. hf-internal-testing's tiny-random-qwen3-vl-moe). The language model has
interleaved M-RoPE (sections [4, 2, 2] of a 16-wide head) and MoE layers with fused, [in, out]-ordered expert tensors; the vision tower has
DeepStack mergers on two layers. Weights are drawn wider than the default init so the greedy continuation depends on the input.
"""
import os
import shutil
import sys

import torch
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

tokenizer_dir, out = sys.argv[1], sys.argv[2]
torch.manual_seed(23)
config = Qwen3VLMoeConfig(
    text_config=dict(vocab_size=151936, hidden_size=64, intermediate_size=96, moe_intermediate_size=24, num_hidden_layers=3, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
                     mlp_only_layers=[], rms_norm_eps=1e-6, max_position_embeddings=1024, tie_word_embeddings=False,
                     rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [4, 2, 2], "mrope_interleaved": True}),
    vision_config=dict(depth=4, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                       temporal_patch_size=2, num_position_embeddings=2304, deepstack_visual_indexes=[1, 2]),
    tie_word_embeddings=False,
)
config.architectures = ["Qwen3VLMoeForConditionalGeneration"]
model = Qwen3VLMoeForConditionalGeneration(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1 and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif p.dim() >= 2:
            fan_in = p.shape[1] if "experts" in name else p.shape[-1]   # fused experts are [E, in, out]
            p.normal_(0.0, 1.0 if "embed_tokens" in name else 1.5 / fan_in ** 0.5)
        else:
            p.normal_(0.0, 0.1)
model.save_pretrained(out, safe_serialization=True)
for f in os.listdir(tokenizer_dir):
    if f not in ("config.json", "model.safetensors", "generation_config.json") and not f.startswith("."):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
print("wrote", out)
