"""
Transformers reference for the GGUF files that llama.cpp's converter writes from the tiny Kimi K3 / K3.1 checkpoints (mk_tiny_kimi_k3.py).

The converter keeps only the first num_heads values of the per-key-channel A_log [head_dim] and uses them as one decay rate per head, and
llama.cpp normalizes the compressed query and latent with rms_norm_eps instead of the remote code's default 1e-6. This script evaluates the
remote modeling code with those semantics: A_log is set to 0 and every head's rows of f_b_proj and dt_bias are scaled by exp(A_log[h]) (so
exp(A) * (f + dt) is the per-head product), and the latent-attention norms use rms_norm_eps. It prints the greedy ids and per-step margins.

    python kimi_k3_gguf_reference.py <checkpoint directory> [tokens]
"""
import json, os, sys
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_k3_reference import load_text_model

PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"

d = os.path.expanduser(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
cfg = json.load(open(os.path.join(d, "config.json")))["text_config"]
state = {k[len("language_model."):]: v for k, v in load_file(os.path.join(d, "model.safetensors")).items()}
heads = cfg["linear_attn_config"]["num_heads"]
dims = cfg["linear_attn_config"]["head_dim"]
for name in [k for k in state if k.endswith("self_attn.A_log")]:
    prefix = name[: -len("A_log")]
    rate = state[name].float()[:heads].exp().repeat_interleave(dims)          # [heads * D]: exp(A_log[h]) for every channel of head h
    state[prefix + "f_b_proj.weight"] = state[prefix + "f_b_proj.weight"] * rate[:, None]
    state[prefix + "dt_bias"] = state[prefix + "dt_bias"] * rate
    state[name] = torch.zeros_like(state[name])
model = load_text_model(d, cfg, state)
for layer in model.model.layers:
    for norm in ("q_a_layernorm", "kv_a_layernorm"):
        if hasattr(layer.self_attn, norm):
            getattr(layer.self_attn, norm).variance_epsilon = cfg["rms_norm_eps"]

ids = torch.tensor([AutoTokenizer.from_pretrained(d)(PROMPT).input_ids])
generated, margins = [], []
with torch.no_grad():
    for _ in range(n):
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
        top = logits.topk(2)
        generated.append(int(top.indices[0]))
        margins.append(float(top.values[0] - top.values[1]))
        ids = torch.cat([ids, top.indices[:1].view(1, 1)], 1)
print("ids:", ", ".join(map(str, generated)))
print("min margin: %.5f" % min(margins))
print("margins:", " ".join("%.4f" % m for m in margins))
