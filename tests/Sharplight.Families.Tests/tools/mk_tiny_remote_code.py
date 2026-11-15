"""Builds tiny random checkpoints (HF format, real tokenizer files) of families whose modeling code ships with the checkpoint (remote code, an
auto_map in config.json) rather than with transformers: the real repository's config.json, modeling / configuration files and tokenizer files are
downloaded, the config is shrunk, and the model is instantiated through AutoModelForCausalLM.from_config(trust_remote_code=True) and saved with its code.

Usage: python mk_tiny_remote_code.py <family> <outdir>       families: see FAMILIES
The greedy reference then comes from hf_greedy_reference.py, which runs the shipped modeling code for such checkpoints.
"""
import json
import os
import sys
import urllib.request

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import remote_code_compat  # noqa: F401  (transformers 4.x remote code on transformers 5)

# family -> (repository, config overrides; layer_types also sets num_hidden_layers)
FAMILIES = {
    # LFM3: short-conv layers, sliding-window and full attention layers (partial zero-tail rotation on the full ones), sandwich norms.
    "lfm3": ("LiquidAI/LFM3-1.7B", dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                        sliding_window=5, max_position_embeddings=4096,
                                        layer_types=["conv", "full_attention", "conv", "sliding_attention", "full_attention", "sliding_attention"])),
    # Spark 2.5 / X2.6: fused q_k_v_proj, per-head sigmoid output gate, sliding layers on their own base, partial rotation of the full layers, GELU MLP.
    "spark2_5": ("XHToken/Spark-X2.6-4B", dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                              sliding_window=5, max_position_embeddings=4096,
                                              layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
                                                           "sliding_attention", "full_attention"])),
    # Clover (block-diffusion Qwen3.5-MoE): decoded causally (the commit pass); its preview-only tensors are random and unused. Seed 1: the
    # seed-0 weights make a chaotic model (torch's own f16 run of it moves the prompt logits by 35 of 44), too sensitive for the f16 GEMMs.
    "clover_moe": ("Diffuspace/Clover-35B-A3B-Preview", dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                                            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                                                            linear_value_head_dim=16, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                                                            shared_expert_intermediate_size=48, max_position_embeddings=4096, _seed=1,
                                                            layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
                                                            + ["full_attention"])),
    # MiMo-V2.6-Pro (mimo_v2): fused qkv_proj, sliding layers with sinks and twice the key/value heads, value heads smaller than key heads, value
    # scale, partial rotation; a dense first layer, then sigmoid-routed experts. The vision and audio towers are left out of the tiny model.
    "mimo_v2": ("XiaomiMiMo/MiMo-V2.6-Pro-RL", dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=1, head_dim=24,
                                                   v_head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=24,
                                                   swa_v_head_dim=16, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                                                   num_hidden_layers=6, hybrid_layer_pattern=[0, 1, 1, 1, 1, 0], moe_layer_freq=[0, 1, 1, 1, 1, 1],
                                                   sliding_window=5, sliding_window_size=5, max_position_embeddings=4096, vision_config=None,
                                                   audio_config=None)),
    # Mage-VL (llava_onevision2): a Qwen3 text decoder on 1D positions below a OneVision encoder (tiny and random; the text import skips it).
    "llava_onevision2": ("microsoft/Mage-VL-Cowork", dict(
        text_config=dict(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_hidden_layers=4,
                         layer_types=["full_attention"] * 4, max_window_layers=4, max_position_embeddings=4096),
        vision_config=dict(hidden_size=64, intermediate_size=64, num_attention_heads=2, num_hidden_layers=1, out_hidden_size=64,
                           text_hidden_size=64))),
}

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja",
                   "tokenizer.model", "vocab.json", "merges.txt"]


def fetch(repo, name, target):
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    headers = {"Authorization": "Bearer " + open(token_file).read().strip()} if os.path.exists(token_file) else {}
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read()
    except Exception:
        return False
    with open(target, "wb") as f:
        f.write(data)
    return True


def code_files(repo):
    info = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}").read())
    return [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".py") and "/" not in s["rfilename"]]


def main():
    family, out = sys.argv[1], sys.argv[2]
    repo, overrides = FAMILIES[family]
    overrides = dict(overrides)
    weight_std = float(os.environ.get("TINY_WEIGHT_STD", overrides.pop("_weight_std", 0.15)))   # std of the non-norm, non-embedding weights
    router_std = float(os.environ.get("TINY_ROUTER_STD", overrides.pop("_router_std", weight_std)))   # std of the expert routers (mlp.gate)
    seed = int(os.environ.get("TINY_SEED", overrides.pop("_seed", 0)))
    os.makedirs(out, exist_ok=True)
    for name in code_files(repo) + TOKENIZER_FILES:
        fetch(repo, name, os.path.join(out, name))
    fetch(repo, "config.json", os.path.join(out, "config.json"))
    config = json.load(open(os.path.join(out, "config.json")))
    for extra in (overrides, json.loads(os.environ.get("TINY_OVERRIDES", "{}"))):   # TINY_OVERRIDES: extra ones, e.g. to bisect a dimension
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)   # a nested config (text_config, vision_config) is overridden key by key
            else:
                config[key] = value
    if "layer_types" in overrides:
        config["num_hidden_layers"] = len(overrides["layer_types"])
    config.pop("quantization_config", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(out, trust_remote_code=True), trust_remote_code=True, dtype=torch.float32)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1 and "norm" in n:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif n.endswith("mlp.gate.weight"):
                p.copy_(torch.randn_like(p) * router_std)   # a larger spread keeps the top-k choice away from ties
            elif "embed" in n or "lm_head" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * weight_std)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    print("done", family, sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
