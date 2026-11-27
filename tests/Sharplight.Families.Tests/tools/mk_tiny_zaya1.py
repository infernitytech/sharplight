"""Builds a tiny random ZAYA1 checkpoint (transformers `zaya`, Zyphra/ZAYA1-8B layout) with the real tokenizer.

Usage: python mk_tiny_zaya1.py <outdir> [tokenizer_dir]

tokenizer_dir defaults to ~/models/hf/ZAYA1-8B (tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja,
generation_config.json are copied from it; download them with `hf download Zyphra/ZAYA1-8B <file>`). Needs transformers >= 5.10 (ZayaForCausalLM).

Unlike the default init (unit residual scales, zero biases) the residual scales, the input scale,
the router state scale and the key temperature are drawn around 1, the biases around 0, the router MLP weights with a larger spread (decisive
top-1 choices; TINY_ROUTER_STD). The class bias of the skip expert stays -1 (TINY_SKIP_BIAS): transformers resets it to -1 when it loads a
checkpoint (the real ZAYA1-8B stores -1 as well), so with probabilities in [0, 1] the skip expert never wins there. A checkpoint built with
TINY_SKIP_BIAS=0 lets it win for part of the tokens; only llama.cpp (and Sharplight) honour that, so it is checked against llama.cpp's text.
TINY_LAYERS (default 6), TINY_SEED, TINY_WEIGHT_STD change the model for bisecting.
"""
import os
import shutil
import sys

import torch
from transformers import ZayaConfig, ZayaForCausalLM

out = sys.argv[1]
tokenizer_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/hf/ZAYA1-8B")
seed = int(os.environ.get("TINY_SEED", "0"))
weight_std = float(os.environ.get("TINY_WEIGHT_STD", "0.15"))
skip_bias = float(os.environ.get("TINY_SKIP_BIAS", "-1"))   # class bias of the skip expert (see the docstring)
router_std = float(os.environ.get("TINY_ROUTER_STD", "1.0"))   # router MLP (fc1, fc2, out_proj): spread logits keep the top-1 away from ties

config = ZayaConfig(vocab_size=262272, hidden_size=64, num_hidden_layers=int(os.environ.get("TINY_LAYERS", "6")), num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    moe_intermediate_size=32, num_experts=4, num_experts_per_tok=1, router_hidden_size=16, max_position_embeddings=4096,
                    rms_norm_eps=1e-5, tie_word_embeddings=True, bos_token_id=2, eos_token_id=106, pad_token_id=0,
                    rope_parameters={"hybrid": {"rope_type": "default", "rope_theta": 5_000_000.0, "partial_rotary_factor": 0.5},
                                     "hybrid_sliding": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5}})
torch.manual_seed(seed)
model = ZayaForCausalLM(config).to(torch.float32)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith(("_scale", "temp")) or (p.dim() == 1 and "norm" in n):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif n.endswith("_bias") or (p.dim() == 1 and n.endswith(".bias")):
            p.copy_(0.1 * torch.randn_like(p))
        elif "router_mlp." in n and n.endswith(".weight"):
            p.copy_(torch.randn_like(p) * router_std)
        elif "embed" in n or "lm_head" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.copy_(torch.randn_like(p) * weight_std)
    for layer in model.model.layers:
        biases = layer.mlp.gate.balancing_biases
        biases.copy_(0.02 * torch.randn_like(biases))
        biases[-1] = skip_bias
os.makedirs(out, exist_ok=True)
model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"]:
    if os.path.isfile(os.path.join(tokenizer_dir, name)):
        shutil.copy(os.path.join(tokenizer_dir, name), os.path.join(out, name))
print("done zaya1", sum(p.numel() for p in model.parameters()))
