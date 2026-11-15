"""
Float64 numpy reference of llama.cpp's granite-switch graph (src/models/granite-switch.cpp) for a GGUF of mk_tiny_granite_switch.py: greedy ids of a
prompt given as token ids, with the smallest top-2 logit margin met on the way. Granite Switch has no transformers class. Per token, a single causal
router head (query 1, key +gain on adapter activation tokens and -gain elsewhere, value the adapter's slot or 0) picks the adapter slot (clamped,
rounded; slot 0 is the base model); adapter tokens are embedded as their substitute tokens; every projection adds B[slot] (A[slot] x).
With --base the router is ignored (slot 0 everywhere), to show the adapters change the output. With --logits <directory> the logits of the
last prompt position are written to <directory>/reference_logits.json (as tools/hf_greedy_reference.py does), which FamilyGenerationTests compares.

    PYTHONPATH=<llama.cpp>/gguf-py python ref_granite_switch.py <file.gguf> <new tokens> [--base] [--logits <dir>] <prompt id> [<prompt id> ...]
"""
import json, os, sys
import numpy as np
from gguf import GGUFReader

argv = sys.argv[1:]
logits_dir = None
if "--logits" in argv:
    i = argv.index("--logits")
    logits_dir = argv[i + 1]
    del argv[i:i + 2]
args = [a for a in argv if a != "--base"]
base_only = "--base" in argv
reader = GGUFReader(args[0])
steps = int(args[1])
prompt = [int(a) for a in args[2:]]
arch = "graniteswitch"


def field(key, default=None):
    f = reader.fields.get(f"{arch}.{key}")
    return default if f is None else f.contents()


W = {t.name: np.asarray(t.data, dtype=np.float64).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}
layers = field("block_count")
heads = field("attention.head_count")
kv_heads = field("attention.head_count_kv")
eps = field("attention.layer_norm_rms_epsilon")
base = field("rope.freq_base")
attn_scale = field("attention.scale")
emb_scale = field("embedding_scale")
res_scale = field("residual_scale")
logit_scale = field("logit_scale")
gain = field("adapters.router_gain", 15.0)
n_adapters = field("adapters.count")
activate = list(field("adapters.token_ids_activate"))
substitute = list(field("adapters.token_ids_substitute"))
slot_of = {tok: i + 1 for i, tok in enumerate(activate)}
sub_of = dict(zip(activate, substitute))
emb = W["token_embd.weight"]
head_dim = emb.shape[1] // heads if "blk.0.attn_q.lora_b" not in W else W["blk.0.attn_q.lora_b"].shape[1] // heads


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, pos):
    # ggml NORM rope: pairs (2i, 2i+1); x [heads, head_dim].
    i = np.arange(head_dim // 2)
    a = pos * base ** (-2.0 * i / head_dim)
    c, s = np.cos(a), np.sin(a)
    out = x.copy()
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out


def lora(name, x, slot):
    # A [slots, rank, in], B [slots, out, rank].
    return W[name + ".lora_b"][slot] @ (W[name + ".lora_a"][slot] @ x)


def run(tokens):
    """Logits of the last token after running the whole sequence (no cache: every row recomputed, as causal attention gives the same)."""
    n = len(tokens)
    keys = np.array([gain if t in slot_of else -gain for t in tokens])
    vals = np.array([float(slot_of.get(t, 0)) for t in tokens])
    slots = []
    for i in range(n):
        w = np.exp(keys[: i + 1] - keys[: i + 1].max())
        r = float((w * vals[: i + 1]).sum() / w.sum())
        slots.append(0 if base_only else int(np.floor(min(max(r, 0.0), n_adapters) + 0.5)))
    x = np.stack([emb[sub_of.get(t, t)] for t in tokens]) * emb_scale
    for l in range(layers):
        p = f"blk.{l}."
        h = rms(x, W[p + "attn_norm.weight"])
        qkv = h @ W[p + "attn_qkv.weight"].T
        qd, kd = heads * head_dim, kv_heads * head_dim
        q = np.stack([qkv[i, :qd] + lora(p + "attn_q", h[i], slots[i]) for i in range(n)]).reshape(n, heads, head_dim)
        k = np.stack([qkv[i, qd:qd + kd] + lora(p + "attn_k", h[i], slots[i]) for i in range(n)]).reshape(n, kv_heads, head_dim)
        v = np.stack([qkv[i, qd + kd:] + lora(p + "attn_v", h[i], slots[i]) for i in range(n)]).reshape(n, kv_heads, head_dim)
        q = np.stack([rope(q[i], i) for i in range(n)])
        k = np.stack([rope(k[i], i) for i in range(n)])
        att = np.zeros((n, heads, head_dim))
        group = heads // kv_heads
        for i in range(n):
            for hh in range(heads):
                s = k[: i + 1, hh // group] @ q[i, hh] * attn_scale
                e = np.exp(s - s.max())
                att[i, hh] = (e / e.sum()) @ v[: i + 1, hh // group]
        att = att.reshape(n, -1)
        o = np.stack([W[p + "attn_output.weight"] @ att[i] + lora(p + "attn_output", att[i], slots[i]) for i in range(n)])
        x = x + o * res_scale
        h = rms(x, W[p + "ffn_norm.weight"])
        f = []
        for i in range(n):
            gate = W[p + "ffn_gate.weight"] @ h[i] + lora(p + "ffn_gate", h[i], slots[i])
            up = W[p + "ffn_up.weight"] @ h[i] + lora(p + "ffn_up", h[i], slots[i])
            a = gate / (1 + np.exp(-gate)) * up
            f.append(W[p + "ffn_down.weight"] @ a + lora(p + "ffn_down", a, slots[i]))
        x = x + np.stack(f) * res_scale
    out = W.get("output.weight", emb)
    return rms(x[-1], W["output_norm.weight"]) @ out.T / logit_scale, slots


tokens = list(prompt)
ids, margin = [], np.inf
for step in range(steps):
    logits, slots = run(tokens)
    if step == 0:
        if logits_dir is not None:
            with open(os.path.join(logits_dir, "reference_logits.json"), "w") as f:
                json.dump({"prompt_ids": prompt, "logits": logits.tolist()}, f)
        print("slots", slots)
        print("prompt logits[:8]", np.round(logits[:8], 6).tolist())
    top = np.argsort(logits)[::-1][:2]
    margin = min(margin, logits[top[0]] - logits[top[1]])
    ids.append(int(top[0]))
    tokens.append(int(top[0]))
print("ids", ids)
print("min margin", margin)
