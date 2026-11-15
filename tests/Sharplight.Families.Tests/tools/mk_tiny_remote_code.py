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
    os.makedirs(out, exist_ok=True)
    for name in code_files(repo) + TOKENIZER_FILES:
        fetch(repo, name, os.path.join(out, name))
    fetch(repo, "config.json", os.path.join(out, "config.json"))
    config = json.load(open(os.path.join(out, "config.json")))
    config.update(overrides)
    if "layer_types" in overrides:
        config["num_hidden_layers"] = len(overrides["layer_types"])
    config.pop("quantization_config", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(out, trust_remote_code=True), trust_remote_code=True, dtype=torch.float32)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1 and "norm" in n:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif "embed" in n or "lm_head" in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    print("done", family, sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
