"""Writes a tiny random Hunyuan v4 (HF hy_v4) checkpoint for the family tests.

Usage: python mk_tiny_hy_v4.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds any byte-level BPE tokenizer.json (its vocabulary size sets the model's). The model has MLA attention with a sigmoid
output gate and per-head sinks, a DSA indexer (top-k 64, so it selects every token of the test prompts), one dense layer followed by MoE
layers (sigmoid routing with a selection bias, two expert groups, clamped SwiGLU experts, one shared expert) and hyper-connections with 3
streams. Weights are drawn wider than the default init so the greedy continuation depends on the input.
"""
import os
import shutil
import sys

import torch
from transformers import HYV4Config, HYV4ForCausalLM

tokenizer_dir, out = sys.argv[1], sys.argv[2]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
import json
vocab = len(json.load(open(os.path.join(tokenizer_dir, "tokenizer.json")))["model"]["vocab"])

torch.manual_seed(41)
config = HYV4Config(
    vocab_size=vocab, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, routed_scaling_factor=1.5, norm_topk_prob=True, n_group=2, topk_group=1,
    q_lora_rank=48, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, index_topk=64, index_head_dim=16,
    index_n_heads=2, hc_mult=3, hc_magnitude=2.0, swiglu_limit=1.5, max_position_embeddings=256, rms_norm_eps=1e-5,
    bos_token_id=None, eos_token_id=None, pad_token_id=None, tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
)
config.architectures = ["HYV4ForCausalLM"]
model = HYV4ForCausalLM(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("e_score_correction_bias"):
            continue
        if "norm" in name and p.dim() == 1:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif name.endswith("sinks"):
            p.normal_(0.0, 1.0)
        elif name.endswith(".base") or name.endswith("hc_base"):
            p.normal_(0.0, 0.5)
        elif name.endswith(".scale") or name.endswith("hc_scale"):
            p.uniform_(0.5, 1.5)
        elif p.dim() >= 2:
            p.normal_(0.0, 1.0 if "embed" in name else 1.5 / p.shape[-1] ** 0.5)
        else:
            p.normal_(0.0, 0.1)
    model.lm_head.weight.mul_(4.0)   # sharper logits: wider greedy margins
    for module in model.modules():
        if hasattr(module, "e_score_correction_bias"):
            module.e_score_correction_bias.normal_(0.0, 0.1)
model.save_pretrained(out, safe_serialization=True)
for f in TOKENIZER_FILES:
    if os.path.exists(os.path.join(tokenizer_dir, f)):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
print("wrote", out)
