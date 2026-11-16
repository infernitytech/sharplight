#!/usr/bin/env python3
"""Writes a tiny random MiMo model (HF model_type "mimo", MiMoForCausalLM: Qwen2 layers plus one multi-token-prediction layer
model.mtp_layers.0) for the speculative-decoding tests, and its reference drafts (<model>/mimo_mtp_reference.json).

Layout as XiaomiMiMo/MiMo-7B-RL: Qwen2 decoder with q/k/v biases, untied LM head, and per MTP layer token_layernorm, hidden_layernorm,
input_proj [H, 2H] reading [hidden | embedding], a Qwen2 decoder layer (input_layernorm, self_attn, post_attention_layernorm, mlp) and
final_layernorm (modeling_mimo.py, MiMoMTPLayers). 2 layers, hidden 64, 4 heads / 2 KV heads of 16; the tokenizer is copied from a
Qwen3 checkpoint (default ~/models/hf/Qwen3-8B).

Reference: transformers' Qwen2ForCausalLM runs the trunk (the checkpoint read as qwen2, the MTP tensors ignored) and gives the
greedy next token ("pending") and the final-norm hidden states (hook on model.norm); the MTP layer is transformers' own
Qwen2DecoderLayer loaded with mtp_layers.0 plus the norms and the projection of MiMoMTPLayers. Row i reads (hidden_i, token after i),
the last prompt row's argmax is the first draft, and every further draft appends a row with the previous draft and the head's own
output (after final_layernorm). Three drafts per prompt; the minimum top-2 logit gap is printed.

Usage: mk_tiny_mimo_mtp.py [OUT_DIR] [TOKENIZER_DIR]   (default OUT_DIR ~/models/hf/tiny-random-mimo-mtp)
Needs numpy, torch and transformers.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

HOME = os.path.expanduser("~")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "models/hf/tiny-random-mimo-mtp")
TOK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "models/hf/Qwen3-8B")

HIDDEN, LAYERS, INTER, HEADS, KV, HEAD_DIM, VOCAB = 64, 2, 128, 4, 2, 16, 151936
EPS, THETA = 1e-5, 640000.0
DRAFTS = 3
PROMPTS = [[785, 6722, 315, 9625, 374, 12095, 13, 576, 6722], [40, 1075, 311, 1349, 6467, 911, 279], [16, 17, 18, 16, 17, 18, 16, 17, 18, 16]]

rng = np.random.default_rng(int(os.environ.get("MIMO_SEED", "3")))


def lin(rows, cols, scale=0.1):
    return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)


def vec(size, scale=0.05, base=0.0):
    return (base + rng.standard_normal(size) * scale).astype(np.float32)


def layer_tensors(prefix):
    return {
        prefix + "input_layernorm.weight": vec(HIDDEN, 0.1, 1.0),
        prefix + "post_attention_layernorm.weight": vec(HIDDEN, 0.1, 1.0),
        prefix + "self_attn.q_proj.weight": lin(HEADS * HEAD_DIM, HIDDEN),
        prefix + "self_attn.q_proj.bias": vec(HEADS * HEAD_DIM),
        prefix + "self_attn.k_proj.weight": lin(KV * HEAD_DIM, HIDDEN),
        prefix + "self_attn.k_proj.bias": vec(KV * HEAD_DIM),
        prefix + "self_attn.v_proj.weight": lin(KV * HEAD_DIM, HIDDEN),
        prefix + "self_attn.v_proj.bias": vec(KV * HEAD_DIM),
        prefix + "self_attn.o_proj.weight": lin(HIDDEN, HEADS * HEAD_DIM),
        prefix + "mlp.gate_proj.weight": lin(INTER, HIDDEN),
        prefix + "mlp.up_proj.weight": lin(INTER, HIDDEN),
        prefix + "mlp.down_proj.weight": lin(HIDDEN, INTER),
    }


tensors = {"model.embed_tokens.weight": lin(VOCAB, HIDDEN, 0.5), "model.norm.weight": vec(HIDDEN, 0.1, 1.0), "lm_head.weight": lin(VOCAB, HIDDEN, 0.5)}
for i in range(LAYERS):
    tensors.update(layer_tensors(f"model.layers.{i}."))
mtp = "model.mtp_layers.0."
tensors.update(layer_tensors(mtp))
tensors[mtp + "token_layernorm.weight"] = vec(HIDDEN, 0.1, 1.0)
tensors[mtp + "hidden_layernorm.weight"] = vec(HIDDEN, 0.1, 1.0)
tensors[mtp + "final_layernorm.weight"] = vec(HIDDEN, 0.1, 1.0)
tensors[mtp + "input_proj.weight"] = lin(HIDDEN, 2 * HIDDEN, 0.15)

common = {
    "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645, "hidden_act": "silu", "hidden_size": HIDDEN,
    "intermediate_size": INTER, "max_position_embeddings": 32768, "num_attention_heads": HEADS, "head_dim": HEAD_DIM,
    "num_hidden_layers": LAYERS, "num_key_value_heads": KV, "rms_norm_eps": EPS, "rope_theta": THETA, "tie_word_embeddings": False,
    "use_sliding_window": False, "vocab_size": VOCAB, "attention_bias": True, "torch_dtype": "float32",
}
config = dict(common, architectures=["MiMoForCausalLM"], model_type="mimo", num_nextn_predict_layers=1)

os.makedirs(OUT, exist_ok=True)
save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config, f, indent=1)
for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    if os.path.exists(os.path.join(TOK, name)):
        shutil.copy(os.path.join(TOK, name), OUT)

# ---- reference ---------------------------------------------------------------------------------------------------------------
qcfg = Qwen2Config(**common)
qcfg._attn_implementation = "eager"
model = Qwen2ForCausalLM(qcfg).eval()
state = {k: torch.from_numpy(v) for k, v in tensors.items() if not k.startswith("model.mtp_layers.")}
model.load_state_dict(state, strict=True)
layer = Qwen2DecoderLayer(qcfg, 0).eval()
layer.load_state_dict({k[len(mtp):]: torch.from_numpy(v) for k, v in tensors.items() if k.startswith(mtp) and k[len(mtp):] in layer.state_dict()}, strict=True)


def norm(name):
    n = Qwen2RMSNorm(HIDDEN, eps=EPS)
    n.weight.data = torch.from_numpy(tensors[mtp + name])
    return n


token_norm, hidden_norm, final_norm = norm("token_layernorm.weight"), norm("hidden_layernorm.weight"), norm("final_layernorm.weight")
proj = torch.from_numpy(tensors[mtp + "input_proj.weight"])
embed, head = model.model.embed_tokens.weight, model.lm_head.weight
captured = {}
model.model.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("hidden", out.detach()))


def run_head(hidden, tokens):
    n = tokens.shape[0]
    x = torch.cat([hidden_norm(hidden), token_norm(embed[tokens])], dim=-1) @ proj.T
    positions = torch.arange(n).view(1, n)
    position_embeddings = model.model.rotary_emb(x[None], positions)
    mask = torch.full((n, n), float("-inf")).triu(1).view(1, 1, n, n)
    out = layer(x[None], attention_mask=mask, position_ids=positions, position_embeddings=position_embeddings)
    out = out[0] if isinstance(out, tuple) else out
    normed = final_norm(out[0])
    return normed @ head.T, normed


cases, gaps = [], []
with torch.no_grad():
    for prompt in PROMPTS:
        logits = model(input_ids=torch.tensor([prompt])).logits[0]
        pending = int(logits[-1].argmax())
        hidden = captured["hidden"][0]
        tokens = torch.tensor(prompt[1:] + [pending])
        drafts = []
        for _ in range(DRAFTS):
            draft_logits, normed = run_head(hidden, tokens)
            top = draft_logits[-1].topk(2).values
            gaps.append(float(top[0] - top[1]))
            draft = int(draft_logits[-1].argmax())
            drafts.append(draft)
            hidden = torch.cat([hidden, normed[-1:]])
            tokens = torch.cat([tokens, torch.tensor([draft])])
        cases.append({"prompt": prompt, "pending": pending, "drafts": drafts})

with open(os.path.join(OUT, "mimo_mtp_reference.json"), "w") as f:
    json.dump(cases, f, indent=1)
print(json.dumps(cases))
print(f"minimum top-2 logit gap {min(gaps):.3g}")
