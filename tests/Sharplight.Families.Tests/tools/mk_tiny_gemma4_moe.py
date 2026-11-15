"""Writes a tiny random Gemma 4 mixture-of-experts language model (HF gemma4_text with enable_moe_block, Gemma4ForCausalLM) for the family tests.

Usage: python mk_tiny_gemma4_moe.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds a Gemma 4 tokenizer (e.g. tiny-random-gemma4-assistant-target). Like Gemma 4 26B-A4B: every layer runs a dense MLP and
routed experts side by side (softmax top-k routing on the router's own norm and input scale, renormalized weights times a per-expert scale,
fused gate/up experts), sliding / full layers with their own head sizes, a 30-capped soft logit cap. The router input scale, the per-expert
scales and the norms are drawn away from 1 so the test sees each of them; float32 weights wider than the default init so the greedy
continuation depends on the input.
"""
import os
import shutil
import sys

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

tokenizer_dir, out = sys.argv[1], sys.argv[2]
torch.manual_seed(int(os.environ.get("SEED", "7")))
config = Gemma4TextConfig(vocab_size=262144, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                          layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=8, max_position_embeddings=1024,
                          final_logit_softcapping=30.0, rms_norm_eps=1e-6, enable_moe_block=True, num_experts=8, top_k_experts=2,
                          moe_intermediate_size=32, tie_word_embeddings=False,
                          rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                                           "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
config.architectures = ["Gemma4ForCausalLM"]
model = Gemma4ForCausalLM(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if ("norm" in name or name.endswith("router.scale") or name.endswith("per_expert_scale")) and p.dim() == 1:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif p.dim() >= 2:
            p.normal_(0.0, float(os.environ.get("EMB", "0.5")) if "embed_tokens" in name else 1.5 / p.shape[-1] ** 0.5)
        else:
            p.normal_(0.0, 0.1)
model.save_pretrained(out, safe_serialization=True)
for f in os.listdir(tokenizer_dir):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "chat_template.jinja"):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
print("wrote", out)
