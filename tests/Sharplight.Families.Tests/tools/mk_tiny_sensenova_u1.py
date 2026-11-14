"""
Writes a tiny random SenseNova-U1 checkpoint (NEOChatModel, remote code that transformers does not ship) and its float64 numpy reference.

    python mk_tiny_sensenova_u1.py <models/hf directory> [<new tokens>]

The checkpoint, <dir>/tiny-random-sensenova-u1/, has the published layout: a Qwen3 language model under language_model.* with the
config under llm_config, the spatial q/k norms (q_norm_hw / k_norm_hw) next to the temporal ones, the image-generation branch (*_mot_gen,
fm_modules.*) and the vision embeddings, which a text runtime skips. Tokenizer: the 1024-token GPT-2-style one of tiny-random-GPTNeoXForCausalLM.

The reference follows llama.cpp's graph (src/models/sensenova-u1.cpp) for text tokens, whose positions are (t, 0, 0): each q/k head is RMS-normed
in two halves (temporal weight q_norm, spatial weight q_norm_hw), the temporal half is rotated (NeoX pairing over head_dim / 2 dimensions,
rope_theta) and the spatial quarters see position 0 (no rotation). It prints the greedy ids and writes the prompt logits to
reference_logits.json (the format of hf_greedy_reference.py --logits).
"""
import json, os, shutil, sys
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

root = os.path.expanduser(sys.argv[1])
new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 24
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, FFN, LAYERS, VOCAB, EPS, THETA = 64, 4, 2, 16, 128, 3, 1024, 1e-6, 10000.0
out = os.path.join(root, "tiny-random-sensenova-u1")
rng = np.random.default_rng(3)   # seed 3: the smallest top-2 logit margin over the 24 steps is 0.02


def weight(*shape):
    return (rng.standard_normal(shape) * 0.15).astype(np.float32)


def norm(n):
    return (1.0 + 0.3 * rng.standard_normal(n)).astype(np.float32)


t = {"language_model.model.embed_tokens.weight": weight(VOCAB, HIDDEN), "language_model.model.norm.weight": norm(HIDDEN),
     "language_model.lm_head.weight": weight(VOCAB, HIDDEN)}
for il in range(LAYERS):
    p = f"language_model.model.layers.{il}."
    t[p + "input_layernorm.weight"] = norm(HIDDEN)
    t[p + "post_attention_layernorm.weight"] = norm(HIDDEN)
    t[p + "self_attn.q_proj.weight"] = weight(HEADS * HEAD_DIM, HIDDEN)
    t[p + "self_attn.k_proj.weight"] = weight(KV_HEADS * HEAD_DIM, HIDDEN)
    t[p + "self_attn.v_proj.weight"] = weight(KV_HEADS * HEAD_DIM, HIDDEN)
    t[p + "self_attn.o_proj.weight"] = weight(HIDDEN, HEADS * HEAD_DIM)
    for n in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
        t[p + f"self_attn.{n}.weight"] = norm(HEAD_DIM // 2)
    t[p + "mlp.gate_proj.weight"] = weight(FFN, HIDDEN)
    t[p + "mlp.up_proj.weight"] = weight(FFN, HIDDEN)
    t[p + "mlp.down_proj.weight"] = weight(HIDDEN, FFN)
    # The image-generation branch: present in the checkpoint, unused for text.
    t[p + "mlp_mot_gen.up_proj.weight"] = weight(FFN, HIDDEN)
    t[p + "input_layernorm_mot_gen.weight"] = norm(HIDDEN)
    t[p + "self_attn.k_norm_hw_mot_gen.weight"] = norm(HEAD_DIM // 2)
t["vision_model.embeddings.patch_embedding.weight"] = weight(32, 3, 4, 4)
t["fm_modules.timestep_embedder.mlp.0.weight"] = weight(32, 16)

os.makedirs(out, exist_ok=True)
save_file({k: torch.from_numpy(v) for k, v in t.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
config = {
    "architectures": ["NEOChatModel"], "model_type": "neo_chat", "downsample_ratio": 0.5, "eos_token_id": 0, "pad_token_id": 0,
    "tie_word_embeddings": False, "torch_dtype": "float32",
    "llm_config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": False, "bos_token_id": 0, "eos_token_id": 0,
                   "head_dim": HEAD_DIM, "hidden_act": "silu", "hidden_size": HIDDEN, "intermediate_size": FFN, "max_position_embeddings": 256,
                   "max_position_embeddings_hw": 10000, "num_attention_heads": HEADS, "num_hidden_layers": LAYERS,
                   "num_key_value_heads": KV_HEADS, "rms_norm_eps": EPS, "rope_scaling": None, "rope_theta": THETA, "rope_theta_hw": 10000.0,
                   "sliding_window": None, "use_sliding_window": False, "vocab_size": VOCAB, "tie_word_embeddings": False},
    "vision_config": {"architectures": ["NEOVisionModel"], "model_type": "neo_vision", "hidden_size": 32, "llm_hidden_size": HIDDEN,
                      "downsample_ratio": 0.5, "patch_size": 4, "num_channels": 3},
}
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    if os.path.exists(os.path.join(tokenizer_dir, f)):
        shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))

W = {k: v.astype(np.float64) for k, v in t.items()}


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1 + np.exp(-x))


def norm_and_rotate(x, p, kind, T):
    # x: [T, H, D]; each half is normed with its own weight, then the temporal half rotates (NeoX pairing within it).
    half = HEAD_DIM // 2
    x = np.concatenate([rms(x[..., :half], W[p + f"self_attn.{kind}_norm.weight"]), rms(x[..., half:], W[p + f"self_attn.{kind}_norm_hw.weight"])], -1)
    inv = THETA ** (-np.arange(0, half, 2) / half)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half // 2], x[..., half // 2:half]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin, x[..., half:]], -1)


def forward(ids):
    T = len(ids)
    x = W["language_model.model.embed_tokens.weight"][ids]
    causal = np.tril(np.ones((T, T), bool))
    for il in range(LAYERS):
        p = f"language_model.model.layers.{il}."
        h = rms(x, W[p + "input_layernorm.weight"])
        q = norm_and_rotate((h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, HEADS, HEAD_DIM), p, "q", T)
        k = norm_and_rotate((h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, KV_HEADS, HEAD_DIM), p, "k", T)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, KV_HEADS, HEAD_DIM)
        k = np.repeat(k, HEADS // KV_HEADS, axis=1)
        v = np.repeat(v, HEADS // KV_HEADS, axis=1)
        s = np.einsum("thd,jhd->thj", q, k) / np.sqrt(HEAD_DIM)
        s = np.where(causal[:, None, :], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("thj,jhd->thd", s, v).reshape(T, -1)
        x = x + o @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        x = x + (silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    return rms(x[-1], W["language_model.model.norm.weight"]) @ W["language_model.lm_head.weight"].T


prompt_ids = AutoTokenizer.from_pretrained(out)(PROMPT).input_ids
ids, generated, margins, prompt_logits = list(prompt_ids), [], [], None
for _ in range(new_tokens):
    logits = forward(ids)
    if prompt_logits is None:
        prompt_logits = logits.tolist()
    order = np.argsort(-logits)
    margins.append(float(logits[order[0]] - logits[order[1]]))
    ids.append(int(order[0]))
    generated.append(int(order[0]))
json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids, "logits": prompt_logits}, open(os.path.join(out, "reference_logits.json"), "w"))
print("wrote", out)
print("ids:", ",".join(map(str, generated)))
print("min margin: %.5f" % min(margins))
