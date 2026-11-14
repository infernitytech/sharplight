"""Writes tiny-random-k2-horizon: a K2-Horizon checkpoint made with the model's remote code, plus its reference.

Usage: python mk_tiny_k2_horizon.py <models/hf directory> <remote code directory>

The remote code directory holds modeling_k2_horizon.py and configuration_k2_horizon.py of IFM/K2-Horizon-36B (no transformers class exists);
they are copied next to the checkpoint and loaded with trust_remote_code. The tiny model has a dense first layer and three MoE layers with
routed value experts (MoVA), sigmoid routing with selection biases, a shared expert, RMS norms over 2 groups, per-head q/k norms, the softplus
output gate and YaRN (factor 4 over 64 positions, beta_fast 128, beta_slow 4). Greedy ids and prompt logits go to
<out>/reference_logits.json and stdout.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

root, code = os.path.expanduser(sys.argv[1]), sys.argv[2]
out = os.path.join(root, "tiny-random-k2-horizon")
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"

os.makedirs(out, exist_ok=True)
for f in ["modeling_k2_horizon.py", "configuration_k2_horizon.py"]:
    shutil.copy(os.path.join(code, f), out)
config = dict(
    architectures=["K2HorizonForCausalLM"], model_type="k2_horizon",
    auto_map={"AutoConfig": "configuration_k2_horizon.K2HorizonConfig", "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM"},
    vocab_size=1024, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_head_dim=16,
    intermediate_size=96, mlp_only_layers=[0], decoder_sparse_step=1, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    num_shared_experts=1, moe_gate_bias=True, router_score_func="sigmoid", router_scaling_factor=2.5, norm_topk_prob=True,
    mova_num_experts=8, mova_num_experts_per_tok=2, layernorm_num_groups=2, query_key_norm=True, attention_gate_func="softplus",
    max_position_embeddings=256, rms_norm_eps=1e-6, bos_token_id=0, eos_token_id=0, tie_word_embeddings=False, hidden_act="silu",
    rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0, "original_max_position_embeddings": 64,
                     "beta_fast": 128.0, "beta_slow": 4.0, "truncate": True},
    use_sliding_window=False, sliding_window=None, dtype="float32")
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), out)

torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(out, trust_remote_code=True), trust_remote_code=True, dtype=torch.float32)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("gate.bias") or n.endswith("v_router.bias"):
            p.copy_(0.05 * torch.randn_like(p))      # selection biases
        elif p.dim() == 1:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * 0.15)
model.save_pretrained(out, safe_serialization=True)
model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32).eval()

tokenizer = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
ids = tokenizer(PROMPT, return_tensors="pt").input_ids
generated, margins, prompt_logits = [], [], None
with torch.no_grad():
    for _ in range(24):
        logits = model(input_ids=ids).logits[0, -1]
        if prompt_logits is None:
            prompt_logits = logits.tolist()
        top = torch.topk(logits, 2)
        margins.append(float(top.values[0] - top.values[1]))
        generated.append(int(top.indices[0]))
        ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)
print("ids:", ",".join(map(str, generated)))
print("min margin: %.5f" % min(margins))
json.dump({"prompt": PROMPT, "prompt_ids": tokenizer(PROMPT).input_ids, "logits": prompt_logits},
          open(os.path.join(out, "reference_logits.json"), "w"))
print("wrote", out)
