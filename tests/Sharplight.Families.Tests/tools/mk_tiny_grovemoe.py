"""Writes tiny-random-grovemoe: a GroveMoE checkpoint in the layout of the original remote code (inclusionAI/GroveMoE-Inst), plus its reference.

Usage: python mk_tiny_grovemoe.py <models/hf directory> [--llamacpp]

GroveMoE has no transformers class: the config says qwen3_moe with architectures GroveMoeForCausalLM. The reference is transformers'
Qwen3MoeForCausalLM with every MoE block replaced by GroveMoeSparseMoeBlock written out from modeling_grove_moe.py (revision c4c69e5):
experts chosen by the top-k of sigmoid(logits), weighted by their softmax probabilities renormalized over the selection, plus one adjugate
("chunk") expert per pair of routed experts that every slot also runs on the same input with the same weight, final = 0.05 * small + large.
Top-4 of 8 experts makes slots of the same pair common. The adjugate experts' inner size is head_dim, as llama.cpp's converter assumes.

Greedy ids and prompt logits go to <out>/reference_logits.json and stdout. With --llamacpp the ids of llama.cpp 8ca5a80's graph are printed
instead (adjugate experts fed the routed experts' output and weighted by the probability of expert id // 2, renormalized), to confirm where
llama.cpp's output comes from.
"""
import json
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

root = os.path.expanduser(sys.argv[1])
emulate_llamacpp = "--llamacpp" in sys.argv[2:]
out = os.path.join(root, "tiny-random-grovemoe")
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
EXPERTS, TOP_K, GROUP, HEAD_DIM = 8, 4, 2, 16

config = dict(vocab_size=1024, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=HEAD_DIM,
              intermediate_size=96, moe_intermediate_size=32, num_experts=EXPERTS, num_experts_per_tok=TOP_K, norm_topk_prob=True,
              decoder_sparse_step=1, mlp_only_layers=[], max_position_embeddings=256, rope_theta=1000000.0, rms_norm_eps=1e-6,
              bos_token_id=0, eos_token_id=0, tie_word_embeddings=False, hidden_act="silu", attention_bias=False)


class Mlp(nn.Module):
    def __init__(self, hidden, inner):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroveMoeSparseMoeBlock(nn.Module):
    """modeling_grove_moe.GroveMoeSparseMoeBlock (the loops over experts written as per-slot sums)."""

    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Linear(hidden, EXPERTS, bias=False)
        self.register_buffer("expert_bias", torch.zeros(EXPERTS))
        self.experts = nn.ModuleList([Mlp(hidden, config["moe_intermediate_size"]) for _ in range(EXPERTS)])
        self.chunk_experts = nn.ModuleList([Mlp(hidden, HEAD_DIM) for _ in range(EXPERTS // GROUP)])

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        router_logits = self.gate(x)
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        _, selected = torch.topk(torch.sigmoid(router_logits).float(), TOP_K, dim=-1)
        groups = selected // GROUP
        weights = probs.gather(-1, groups if emulate_llamacpp else selected)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        large = torch.zeros_like(x)
        small = torch.zeros_like(x)
        large_weights = probs.gather(-1, selected)
        large_weights = large_weights / large_weights.sum(dim=-1, keepdim=True)
        for t in range(x.shape[0]):
            for s in range(TOP_K):
                large[t] += self.experts[int(selected[t, s])](x[t]) * large_weights[t, s]
        chunk_input = large if emulate_llamacpp else x
        for t in range(x.shape[0]):
            for s in range(TOP_K):
                small[t] += self.chunk_experts[int(groups[t, s])](chunk_input[t]) * weights[t, s]
        return (0.05 * small + large).reshape(shape)


if not emulate_llamacpp:
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config)).eval()
    for layer in model.model.layers:
        layer.mlp = GroveMoeSparseMoeBlock(config["hidden_size"])
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif ".mlp.gate." in n:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    os.makedirs(out, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items() if "rotary" not in k}
    save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    published = dict(config, model_type="qwen3_moe", architectures=["GroveMoeForCausalLM"],
                     auto_map={"AutoModelForCausalLM": "modeling_grove_moe.GroveMoeForCausalLM"}, torch_dtype="float32")
    json.dump(published, open(os.path.join(out, "config.json"), "w"), indent=2)
    tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(os.path.join(tokenizer_dir, f), out)
else:
    from safetensors.torch import load_file
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config)).eval()
    for layer in model.model.layers:
        layer.mlp = GroveMoeSparseMoeBlock(config["hidden_size"])
    model.load_state_dict(load_file(os.path.join(out, "model.safetensors")), strict=False)

tokenizer = AutoTokenizer.from_pretrained(out)
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
print("llama.cpp graph" if emulate_llamacpp else "reference", "ids:", ",".join(map(str, generated)))
print("min margin: %.5f" % min(margins))
if not emulate_llamacpp:
    json.dump({"prompt": PROMPT, "prompt_ids": tokenizer(PROMPT).input_ids, "logits": prompt_logits},
              open(os.path.join(out, "reference_logits.json"), "w"))
    print("wrote", out)
