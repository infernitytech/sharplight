"""Writes tiny-random-hy-v4-mhc: tiny-random-hy-v4 (mk_tiny_hy_v4.py) with DeepSeek-V4's manifold-constrained hyper-connections, plus its reference.

Usage: python mk_tiny_hy_v4_mhc.py <tiny-random-hy-v4 dir> <out_dir>

Every hyper-connection module of the layers gets the stream mixing matrix of DeepSeek-V4 (fn and base gain N² rows, scale a third entry,
config hc_sinkhorn_iters = 20). No released model combines Hunyuan v4 with DeepSeek-V4's mHC; the checkpoint exercises the Sinkhorn mixing
of the shared hyper-connection block against transformers' DeepseekV4HyperConnection, which runs inside the Hunyuan v4 layers here:
X' = post * block(collapsed) + C^T X with (post, C, collapsed) from DeepseekV4HyperConnection (post without Hunyuan's + eps). The greedy
ids and prompt logits go to <out_dir>/reference_logits.json and stdout.
"""
import json
import os
import shutil
import sys
import types

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, HYV4ForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection

src, out = sys.argv[1], sys.argv[2]
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
ITERS = 20

torch.manual_seed(5)
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if f not in ("model.safetensors", "config.json", "reference_logits.json"):
        shutil.copy(os.path.join(src, f), out)
config = json.load(open(os.path.join(src, "config.json")))
config["hc_sinkhorn_iters"] = ITERS
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
n = config["hc_mult"]
state = load_file(os.path.join(src, "model.safetensors"))
for k in [k for k in state if ".hc_attn_layer." in k or ".hc_mlp_layer." in k]:
    v = state[k]
    if k.endswith("hc_fn"):
        state[k] = torch.cat([v, torch.randn(n * n, v.shape[1]) * 1.5 / v.shape[1] ** 0.5])
    elif k.endswith("hc_base"):
        state[k] = torch.cat([v, torch.randn(n * n) * 0.5])
    elif k.endswith("hc_scale"):
        state[k] = torch.cat([v, torch.tensor([1.3])])
save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

# The reference: the stock Hunyuan v4 model with its layer hyper-connections replaced by DeepSeek-V4's.
model = HYV4ForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
cfg = types.SimpleNamespace(hc_mult=n, hc_sinkhorn_iters=ITERS, hc_eps=config.get("hc_eps", 1e-6), rms_norm_eps=config["rms_norm_eps"],
                            hidden_size=config["hidden_size"])


def mhc(prefix):
    module = DeepseekV4HyperConnection(cfg)
    with torch.no_grad():
        module.fn.copy_(state[prefix + "hc_fn"])
        module.base.copy_(state[prefix + "hc_base"])
        module.scale.copy_(state[prefix + "hc_scale"])
    return module


def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, position_embeddings=None,
            prev_topk_indices=None, **kwargs):
    post, comb, collapsed = self.attn_mhc(hidden_states)
    attn, _, topk = self.self_attn(hidden_states=self.input_layernorm(collapsed), attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings,
                                   prev_topk_indices=prev_topk_indices, **kwargs)
    hidden_states = post.unsqueeze(-1) * attn.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), hidden_states)
    post, comb, collapsed = self.ffn_mhc(hidden_states)
    mlp = self.mlp(self.post_attention_layernorm(collapsed))
    return post.unsqueeze(-1) * mlp.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), hidden_states), topk


for i, layer in enumerate(model.model.layers):
    layer.attn_mhc = mhc(f"model.layers.{i}.hc_pre.hc_attn_layer.")
    layer.ffn_mhc = mhc(f"model.layers.{i}.hc_pre.hc_mlp_layer.")
    layer.forward = types.MethodType(forward, layer)

tokenizer = AutoTokenizer.from_pretrained(src)
ids = tokenizer(PROMPT, return_tensors="pt").input_ids
generated, margins, prompt_logits = [], [], None
with torch.no_grad():
    for _ in range(24):
        logits = model(input_ids=ids).logits[0, -1]
        prompt_logits = prompt_logits or logits.tolist()
        top = torch.topk(logits, 2)
        margins.append(float(top.values[0] - top.values[1]))
        generated.append(int(top.indices[0]))
        ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)
print("ids:", ",".join(str(i) for i in generated))
print("min margin: %.5f" % min(margins))
with open(os.path.join(out, "reference_logits.json"), "w") as f:
    json.dump({"prompt": PROMPT, "prompt_ids": tokenizer(PROMPT).input_ids, "logits": prompt_logits}, f)
