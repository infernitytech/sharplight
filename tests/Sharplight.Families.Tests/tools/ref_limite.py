"""
Float64 numpy reference of llama.cpp's limite graph (src/models/limite.cpp) for the tiny GGUF of mk_tiny_gguf.py: greedy ids of a prompt given as
token ids, and the smallest top-2 logit margin. Limite's Hugging Face code is remote code with bf16 rotary tables, so this is the reference.

    PYTHONPATH=<llama.cpp>/gguf-py python ref_limite.py <file.gguf> <new tokens> <prompt id> [<prompt id> ...]
"""
import math, sys
import numpy as np
from gguf import GGUFReader

reader = GGUFReader(sys.argv[1])
arch = "limite"


def field(key, default=None):
    f = reader.fields.get(f"{arch}.{key}")
    return default if f is None else f.contents()


W = {t.name: np.asarray(t.data, dtype=np.float64).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}
layers = field("block_count")
heads = field("attention.head_count")
kv_heads = field("attention.head_count_kv")
hidden = field("embedding_length")
head_dim = hidden // heads
eps = field("attention.layer_norm_rms_epsilon")
scale = field("attention.scale")
xsa_eps = field("attention.xsa_epsilon", 0.0)
window = field("attention.sliding_window")
swa = field("attention.sliding_window_pattern")
rope_dim = field("rope.dimension_count")
base = field("rope.freq_base")
cap = field("final_logit_sigmoid_capping")
n_taps = field("mudd.tap_count", 0)
tap_idx = field("mudd.tap_indices", [])


def rms(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rope(x, T):
    # ggml "normal" rope: adjacent pairs (2i, 2i + 1) over the first rope_dim channels.
    inv = base ** (-np.arange(0, rope_dim, 2) / rope_dim)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., 0:rope_dim:2], x[..., 1:rope_dim:2]
    out = x.copy()
    out[..., 0:rope_dim:2] = a * cos - b * sin
    out[..., 1:rope_dim:2] = a * sin + b * cos
    return out


def forward(ids):
    T = len(ids)
    x = rms(W["token_embd.weight"][ids])
    ve = W["value_embd.weight"][ids].reshape(T, kv_heads, head_dim)
    hist = []
    pos = np.arange(T)
    for il in range(layers):
        p = f"blk.{il}."
        hist.append(x)
        resid, cur = x, x
        if p + "mudd_up.weight" in W:
            taps = tap_idx[il * n_taps:(il + 1) * n_taps]
            w = gelu(rms(x) @ W[p + "mudd_down.weight"].T) @ W[p + "mudd_up.weight"].T + W[p + "mudd_up.bias"]
            cur = sum(w[:, j:j + 1] * hist[t] for j, t in enumerate(taps))
            resid = sum(w[:, n_taps + j:n_taps + j + 1] * hist[t] for j, t in enumerate(taps))
        h = rms(cur)
        q = (h @ W[p + "attn_q.weight"].T).reshape(T, heads, head_dim)
        k = (h @ W[p + "attn_k.weight"].T).reshape(T, kv_heads, head_dim)
        v = (h @ W[p + "attn_v.weight"].T).reshape(T, kv_heads, head_dim)
        if p + "attn_ve_gate.weight" in W:
            v = v + ve * sigmoid(h @ W[p + "attn_ve_gate.weight"].T)[:, :, None]
        q, k = rms(q), rms(k)
        if swa[il]:
            q, k = rope(q, T), rope(k, T)
        g = heads // kv_heads
        kk, vv = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
        s = np.einsum("thd,jhd->thj", q, kk) * scale
        mask = pos[None, :] <= pos[:, None]
        if swa[il]:
            mask &= pos[:, None] - pos[None, :] < window
        s = np.where(mask[:, None, :], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        y = np.einsum("thj,jhd->thd", s, vv)
        if p + "attn_xsa_alpha.weight" in W:
            vn = v / np.maximum(np.sqrt((v * v).sum(-1, keepdims=True)), xsa_eps)
            vn = np.repeat(vn, g, axis=1)
            proj = (y * vn).sum(-1, keepdims=True) * W[p + "attn_xsa_alpha.weight"][None, :, None]
            y = y - vn * proj
        if p + "attn_gate.weight" in W:
            y = y * sigmoid(h @ W[p + "attn_gate.weight"].T)[:, :, None]
        x = resid * W[p + "attn_resid_scale.weight"] + y.reshape(T, -1) @ W[p + "attn_output.weight"].T
        f = rms(x)
        f = (silu(f @ W[p + "ffn_gate.weight"].T) * (f @ W[p + "ffn_up.weight"].T)) @ W[p + "ffn_down.weight"].T
        x = x * W[p + "ffn_resid_scale.weight"] + f
    logits = rms(x[-1]) @ W.get("output.weight", W["token_embd.weight"]).T
    if cap is not None and cap[0] != 0 and cap[2] != 0:
        logits = cap[0] * sigmoid((logits + cap[1]) / cap[2])
    return logits


ids = [int(v) for v in sys.argv[3:]]
generated, margin = [], math.inf
for _ in range(int(sys.argv[2])):
    logits = forward(ids)
    order = np.argsort(-logits)
    margin = min(margin, logits[order[0]] - logits[order[1]])
    ids.append(int(order[0]))
    generated.append(int(order[0]))
print("ids:", ",".join(map(str, generated)))
print("min margin: %.5f" % margin)
