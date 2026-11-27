"""Builds a tiny random Raon-1 mini checkpoint (HF `raon1_mini`, KRAFTON's remote code) with the real tokenizer, in float32.

Usage: python mk_tiny_raon1.py <outdir> [source_dir]

source_dir defaults to ~/models/hf/Raon-1-mini and must hold the remote code (configuration_raon1_mini.py, modeling_raon1_mini.py) and the
tokenizer files (tokenizer.json, tokenizer_config.json, chat_template.jinja) of KRAFTON/Raon-1-mini; they are copied next to the weights.
generation_config.json is not copied (its presence penalty would change the greedy references). Needs transformers >= 5.16.

Six layers (RDA except the 1-based MLA layers 3 and 6, a dense first layer, MoE with 8 experts / top 2 and a shared expert), 4 heads of 16,
4 Raon_mHC streams. The init draws every mHC parameter, norm and gate away from its default (alpha around 1, eta and the biases around 0) so the
chart mixing, the write gate and the bounded decay all matter. TINY_STEERING=1 adds reasoning-effort steering vectors on layers 2 and 4 (the
default effort "high", alpha 0.5 so they move the logits); llama.cpp's converter does not take them, so that variant is HF-only.
TINY_SEED, TINY_WEIGHT_STD change the model for bisecting.
"""
import json
import os
import shutil
import sys

import torch

out = sys.argv[1]
source = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/hf/Raon-1-mini")
seed = int(os.environ.get("TINY_SEED", "0"))
weight_std = float(os.environ.get("TINY_WEIGHT_STD", "0.15"))
steering = os.environ.get("TINY_STEERING", "0") == "1"

os.makedirs(out, exist_ok=True)
for name in ["configuration_raon1_mini.py", "modeling_raon1_mini.py", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    shutil.copy(os.path.join(source, name), os.path.join(out, name))

config_json = {
    "architectures": ["Raon1MiniForCausalLM"],
    "model_type": "raon1_mini",
    "auto_map": {
        "AutoConfig": "configuration_raon1_mini.Raon1MiniConfig",
        "AutoModel": "modeling_raon1_mini.Raon1MiniModel",
        "AutoModelForCausalLM": "modeling_raon1_mini.Raon1MiniForCausalLM",
    },
    "dtype": "float32",
    "vocab_size": 259840, "hidden_size": 64, "num_hidden_layers": 6, "num_attention_heads": 4, "head_dim": 16,
    "tie_word_embeddings": False, "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_parameters": None,
    "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0, "intermediate_size": 96,
    "first_k_dense_replace": 1, "moe_layer_freq": 1, "num_experts": 8, "num_experts_per_token": 2, "moe_intermediate_size": 32,
    "num_shared_experts": 1, "moe_router_activation_func": "sigmoid", "moe_renormalize": True, "routed_scaling_factor": 1.0,
    "q_lora_rank": 32, "kv_lora_rank": 32, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "mla_use_nope": True, "mla_use_output_gate": True, "raon_mhc_streams": 4, "rda_decay_rank": 16,
    "rda_config": {"num_heads": 4, "head_dim": 16, "conv_kernel_size": 4, "full_rank_gate": True, "gate_lower_bound": -5.0,
                   "mla_layers": [3, 6], "rda_layers": [1, 2, 4, 5]},
    "pad_token_id": 259200, "eos_token_id": [259202, 259200],
}
if steering:
    config_json["reasoning_effort_steering"] = {"alpha": {"none": 0, "low": 1.0, "medium": 0.7, "high": 0.5, "xhigh": 0.2, "max": 0},
                                                "default": "high", "layers": [2, 4]}
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(config_json, f, indent=2)

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

config = AutoConfig.from_pretrained(out, trust_remote_code=True)
torch.manual_seed(seed)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.float32)
with torch.no_grad():
    for n, p in model.named_parameters():
        leaf = n.rsplit(".", 1)[-1]
        if leaf in ("alpha_pre", "alpha_post", "alpha_z"):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif leaf in ("b_pre", "b_post", "b_z", "eta"):
            p.copy_(0.5 * torch.randn_like(p))
        elif leaf in ("phi_pre", "phi_post", "phi_z"):
            p.copy_(0.1 * torch.randn_like(p))
        elif leaf == "A_log":
            p.copy_(0.5 * torch.randn_like(p))
        elif leaf in ("dt_bias", "e_score_correction_bias"):
            p.copy_(0.3 * torch.randn_like(p))
        elif p.dim() == 1 and "norm" in n:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif "conv1d" in n:
            p.copy_(0.4 * torch.randn_like(p))
        elif "embed" in n or "lm_head" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        elif n.endswith("gate.weight") and ".moe." in n:
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.copy_(torch.randn_like(p) * weight_std)
    for name, buffer in model.named_buffers():
        if name.endswith("reasoning_effort_steering"):
            buffer.copy_(torch.randn_like(buffer))
model.save_pretrained(out, safe_serialization=True)
# save_pretrained rewrites config.json; keep the remote-code auto_map and the original dtype.
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(config_json, f, indent=2)
print("done raon1", sum(p.numel() for p in model.parameters()))
