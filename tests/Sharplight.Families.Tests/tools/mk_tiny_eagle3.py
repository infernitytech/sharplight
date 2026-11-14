#!/usr/bin/env python3
"""Writes a tiny random Llama target and a tiny random EAGLE-3 drafter for it (Hugging Face layouts, f32 safetensors) for the speculative-decoding
tests, plus the drafter's reference drafts.

Target: Llama, 6 layers, hidden 64, 4 heads / 2 KV heads, llama3 rope scaling, tied embeddings, the Llama-3.2 tokenizer (default
~/models/hf/Llama-3.2-1B-Instruct). Drafter (EAGLE / SpecForge LlamaForCausalLMEagle3 layout: embed_tokens, fc, midlayer.*, norm, lm_head, d2t,
t2d; its own embedding since its width differs from the target's): hidden 48, 4 heads / 2 KV heads of 16, a 512-token draft vocabulary, fusing the inputs of target layers 1, 3
and 5 (eagle_aux_hidden_state_layer_ids).

The reference (<drafter>/eagle3_reference.json) follows EAGLE-3 decoding as llama.cpp and vLLM do it: transformers gives the target's layer
inputs (output_hidden_states) and greedy next token for each prompt; a float64 numpy forward of the drafter then proposes three tokens: its row
i reads (prompt[i + 1] or the target's next token, fc(features_i)), the last row's argmax is the first draft, and every further draft reads the
previous draft with the drafter's own pre-norm output. Draft ids are mapped to target ids through d2t. The minimum top-2 logit gap is printed.

Usage: mk_tiny_eagle3.py [TARGET_DIR] [DRAFTER_DIR] [TOKENIZER_DIR]
       (defaults ~/models/hf/tiny-random-llama-eagle3-target, ~/models/hf/tiny-random-eagle3)
GGUF versions: convert_hf_to_gguf.py TARGET_DIR, and convert_hf_to_gguf.py DRAFTER_DIR --target-model-dir TARGET_DIR.
EAGLE3_SEED (default 6) picks the weights; 6 gives drafts that differ along the chain. Needs numpy, torch and transformers.
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

HOME = os.path.expanduser("~")
TARGET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "models/hf/tiny-random-llama-eagle3-target")
DRAFTER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "models/hf/tiny-random-eagle3")
TOK = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HOME, "models/hf/Llama-3.2-1B-Instruct")

VOCAB = 128256
ROPE = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
THETA = 500000.0
EPS = 1e-5

T_HIDDEN, T_LAYERS, T_INTER, T_HEADS, T_KV, T_HEAD_DIM = 64, 6, 128, 4, 2, 16
D_HIDDEN, D_INTER, D_HEADS, D_KV, D_HEAD_DIM, D_VOCAB = 48, 96, 4, 2, 16, 512
AUX = [1, 3, 5]
DRAFTS = 3
PROMPTS = [[128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13],
           [128000, 40, 1093, 311, 1373, 6603, 922, 279, 2294, 23651, 315, 3823, 7138, 323, 13]]

rng = np.random.default_rng(int(os.environ.get("EAGLE3_SEED", "6")))


def lin(out_features, in_features, scale=0.08):
    return (rng.standard_normal((out_features, in_features)) * scale).astype(np.float32)


def norm(size):
    return (1.0 + rng.standard_normal(size) * 0.1).astype(np.float32)


def write_safetensors(path, items):
    header = {}
    offset = 0
    for name, array in items.items():
        dtype = {np.dtype(np.float32): "F32", np.dtype(np.int64): "I64", np.dtype(np.bool_): "BOOL"}[array.dtype]
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for array in items.values():
            f.write(np.ascontiguousarray(array).tobytes())


# --- target ---
target = {"model.embed_tokens.weight": (rng.standard_normal((VOCAB, T_HIDDEN)) * 0.5).astype(np.float32), "model.norm.weight": norm(T_HIDDEN)}
for i in range(T_LAYERS):
    p = f"model.layers.{i}."
    target[p + "input_layernorm.weight"] = norm(T_HIDDEN)
    target[p + "post_attention_layernorm.weight"] = norm(T_HIDDEN)
    target[p + "self_attn.q_proj.weight"] = lin(T_HEADS * T_HEAD_DIM, T_HIDDEN)
    target[p + "self_attn.k_proj.weight"] = lin(T_KV * T_HEAD_DIM, T_HIDDEN)
    target[p + "self_attn.v_proj.weight"] = lin(T_KV * T_HEAD_DIM, T_HIDDEN)
    target[p + "self_attn.o_proj.weight"] = lin(T_HIDDEN, T_HEADS * T_HEAD_DIM)
    target[p + "mlp.gate_proj.weight"] = lin(T_INTER, T_HIDDEN)
    target[p + "mlp.up_proj.weight"] = lin(T_INTER, T_HIDDEN)
    target[p + "mlp.down_proj.weight"] = lin(T_HIDDEN, T_INTER)
target_config = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": T_HIDDEN, "intermediate_size": T_INTER,
    "num_hidden_layers": T_LAYERS, "num_attention_heads": T_HEADS, "num_key_value_heads": T_KV, "head_dim": T_HEAD_DIM,
    "hidden_act": "silu", "rms_norm_eps": EPS, "rope_theta": THETA, "rope_scaling": ROPE, "max_position_embeddings": 4096,
    "vocab_size": VOCAB, "tie_word_embeddings": True, "bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009],
    "attention_bias": False, "mlp_bias": False, "torch_dtype": "float32",
}
os.makedirs(TARGET, exist_ok=True)
write_safetensors(os.path.join(TARGET, "model.safetensors"), target)
with open(os.path.join(TARGET, "config.json"), "w") as f:
    json.dump(target_config, f, indent=1)
for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if os.path.exists(os.path.join(TOK, name)):
        shutil.copy(os.path.join(TOK, name), os.path.join(TARGET, name))

# --- drafter ---
draft_ids = np.sort(rng.choice(np.arange(256, 20000), D_VOCAB, replace=False)).astype(np.int64)
d2t = draft_ids - np.arange(D_VOCAB, dtype=np.int64)
t2d = np.zeros(VOCAB, dtype=np.bool_)
t2d[draft_ids] = True
drafter = {
    "embed_tokens.weight": (rng.standard_normal((VOCAB, D_HIDDEN)) * 0.5).astype(np.float32),
    "fc.weight": lin(D_HIDDEN, 3 * T_HIDDEN),
    "midlayer.input_layernorm.weight": norm(D_HIDDEN),
    "midlayer.hidden_norm.weight": norm(D_HIDDEN),
    "midlayer.self_attn.q_proj.weight": lin(D_HEADS * D_HEAD_DIM, 2 * D_HIDDEN),
    "midlayer.self_attn.k_proj.weight": lin(D_KV * D_HEAD_DIM, 2 * D_HIDDEN),
    "midlayer.self_attn.v_proj.weight": lin(D_KV * D_HEAD_DIM, 2 * D_HIDDEN),
    "midlayer.self_attn.o_proj.weight": lin(D_HIDDEN, D_HEADS * D_HEAD_DIM),
    "midlayer.post_attention_layernorm.weight": norm(D_HIDDEN),
    "midlayer.mlp.gate_proj.weight": lin(D_INTER, D_HIDDEN),
    "midlayer.mlp.up_proj.weight": lin(D_INTER, D_HIDDEN),
    "midlayer.mlp.down_proj.weight": lin(D_HIDDEN, D_INTER),
    "norm.weight": norm(D_HIDDEN),
    "lm_head.weight": lin(D_VOCAB, D_HIDDEN, 0.3),
    "d2t": d2t,
    "t2d": t2d,
}
drafter_config = {
    "architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama", "hidden_size": D_HIDDEN, "intermediate_size": D_INTER,
    "num_hidden_layers": 1, "num_attention_heads": D_HEADS, "num_key_value_heads": D_KV, "head_dim": D_HEAD_DIM, "hidden_act": "silu",
    "rms_norm_eps": EPS, "rope_theta": THETA, "rope_scaling": ROPE, "max_position_embeddings": 4096, "vocab_size": VOCAB,
    "draft_vocab_size": D_VOCAB, "target_hidden_size": T_HIDDEN, "eagle_aux_hidden_state_layer_ids": AUX, "tie_word_embeddings": False,
    "bos_token_id": 128000, "eos_token_id": 128001, "torch_dtype": "float32",
}
os.makedirs(DRAFTER, exist_ok=True)
write_safetensors(os.path.join(DRAFTER, "model.safetensors"), drafter)
with open(os.path.join(DRAFTER, "config.json"), "w") as f:
    json.dump(drafter_config, f, indent=1)

# --- reference ---
import torch  # noqa: E402
from transformers import LlamaForCausalLM  # noqa: E402

model = LlamaForCausalLM.from_pretrained(TARGET, torch_dtype=torch.float32).eval()
W = {k: v.astype(np.float64) for k, v in drafter.items() if v.dtype == np.float32}
emb = W["embed_tokens.weight"]


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def inv_freq(dim):
    freqs = 1.0 / THETA ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    low, high = ROPE["original_max_position_embeddings"] / ROPE["low_freq_factor"], ROPE["original_max_position_embeddings"] / ROPE["high_freq_factor"]
    out = []
    for f in freqs:
        wavelen = 2 * np.pi / f
        if wavelen < high:
            out.append(f)
        elif wavelen > low:
            out.append(f / ROPE["factor"])
        else:
            smooth = (ROPE["original_max_position_embeddings"] / wavelen - ROPE["low_freq_factor"]) / (ROPE["high_freq_factor"] - ROPE["low_freq_factor"])
            out.append((1 - smooth) * f / ROPE["factor"] + smooth * f)
    return np.array(out)


FREQ = inv_freq(D_HEAD_DIM)


def rope(x, pos):  # x [H, D], neox (rotate_half)
    ang = pos * FREQ
    cos, sin = np.cos(np.concatenate([ang, ang])), np.sin(np.concatenate([ang, ang]))
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cos + rot * sin


class Drafter:
    def __init__(self):
        self.k, self.v = [], []

    def row(self, token, g, pos):
        """One drafter row: returns (draft-vocabulary logits, pre-norm output)."""
        e = rms(emb[token], W["midlayer.input_layernorm.weight"])
        gn = rms(g, W["midlayer.hidden_norm.weight"])
        a = np.concatenate([e, gn])
        q = rope((W["midlayer.self_attn.q_proj.weight"] @ a).reshape(D_HEADS, D_HEAD_DIM), pos)
        self.k.append(rope((W["midlayer.self_attn.k_proj.weight"] @ a).reshape(D_KV, D_HEAD_DIM), pos))
        self.v.append((W["midlayer.self_attn.v_proj.weight"] @ a).reshape(D_KV, D_HEAD_DIM))
        K, V = np.stack(self.k), np.stack(self.v)
        out = np.zeros((D_HEADS, D_HEAD_DIM))
        for h in range(D_HEADS):
            kv = h // (D_HEADS // D_KV)
            s = K[:, kv] @ q[h] / np.sqrt(D_HEAD_DIM)
            p = np.exp(s - s.max())
            out[h] = (p / p.sum()) @ V[:, kv]
        x = W["midlayer.self_attn.o_proj.weight"] @ out.reshape(-1) + g
        hn = rms(x, W["midlayer.post_attention_layernorm.weight"])
        gate = W["midlayer.mlp.gate_proj.weight"] @ hn
        x = x + W["midlayer.mlp.down_proj.weight"] @ (gate / (1 + np.exp(-gate)) * (W["midlayer.mlp.up_proj.weight"] @ hn))
        return W["lm_head.weight"] @ rms(x, W["norm.weight"]), x


def gap(logits):
    top = np.sort(logits)[-2:]
    return top[1] - top[0]


reference = []
min_gap = np.inf
for prompt in PROMPTS:
    with torch.no_grad():
        out = model(torch.tensor([prompt]), output_hidden_states=True)
    pending = int(out.logits[0, -1].argmax())
    features = np.concatenate([out.hidden_states[i][0].double().numpy() for i in AUX], -1)
    g = features @ W["fc.weight"].T
    d = Drafter()
    successors = prompt[1:] + [pending]
    for i, token in enumerate(successors):
        logits, pre = d.row(token, g[i], i)
    drafts = []
    for j in range(DRAFTS):
        best = int(logits.argmax())
        min_gap = min(min_gap, gap(logits))
        drafts.append(int(draft_ids[best]))
        if j + 1 < DRAFTS:
            logits, pre = d.row(drafts[-1], pre, len(prompt) + j)
    reference.append({"prompt": prompt, "pending": pending, "drafts": drafts})

with open(os.path.join(DRAFTER, "eagle3_reference.json"), "w") as f:
    json.dump(reference, f, indent=1)
print(json.dumps(reference))
print(f"wrote {TARGET} and {DRAFTER}; minimum draft logit gap {min_gap:.4f}")
