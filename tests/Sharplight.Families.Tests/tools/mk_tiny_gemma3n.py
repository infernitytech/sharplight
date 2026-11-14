"""Writes a tiny random Gemma 3n language model (HF gemma3n_text, Gemma3nForCausalLM) for the family tests.

Usage: python mk_tiny_gemma3n.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds a Gemma 3n tokenizer (e.g. tiny-random-gemma-3n). Four AltUp streams, LAuReL, per-layer embeddings, activation sparsity in
the first two layers, sliding / full layers with KV sharing in the last two, float32 weights drawn wider than the default init so the greedy
continuation depends on the input (the published tiny-random-gemma-3n is bfloat16 and nearly constant).
"""
import os
import shutil
import sys

import torch
from transformers import Gemma3nForCausalLM, Gemma3nTextConfig

tokenizer_dir, out = sys.argv[1], sys.argv[2]
torch.manual_seed(int(os.environ.get("SEED", "29")))
config = Gemma3nTextConfig(vocab_size=262400, vocab_size_per_layer_input=262144, hidden_size=64, hidden_size_per_layer_input=8,
                           intermediate_size=[96, 96, 128, 128, 96, 96], num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, laurel_rank=8, altup_num_inputs=4, num_kv_shared_layers=2, sliding_window=8,
                           layer_types=["sliding_attention", "full_attention"] * 3, activation_sparsity_pattern=[0.95, 0.95, 0.0, 0.0, 0.0, 0.0],
                           max_position_embeddings=1024, final_logit_softcapping=30.0, rms_norm_eps=1e-6)
config.architectures = ["Gemma3nForCausalLM"]
model = Gemma3nForCausalLM(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif "correct_output_scale" in name:
            p.normal_(0.0, 0.3)
        elif p.dim() >= 2:
            p.normal_(0.0, float(os.environ.get("EMB", "0.5")) if "embed_tokens" in name else 1.5 / p.shape[-1] ** 0.5)
        else:
            p.normal_(0.0, 0.1)
model.save_pretrained(out, safe_serialization=True)
for f in os.listdir(tokenizer_dir):
    if f.startswith("tokenizer") or f in ("special_tokens_map.json", "chat_template.jinja"):
        shutil.copy(os.path.join(tokenizer_dir, f), out)
print("wrote", out)
