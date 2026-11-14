"""
Float64 numpy reference of llama.cpp's mistral-large4 graph (src/models/mistral-large4.cpp) for the tiny GGUF of mk_tiny_gguf.py: greedy ids of a
prompt given as token ids, plus the smallest gap between the indexer's last selected and first dropped score over all rows and indexed layers (a
gap near 0 means the selection depends on rounding, so the file is not a usable test case). Mistral Large 4 has no transformers class, and
llama.cpp's sparse prefill and its token-by-token decoding of the same file do not agree with each other, so this is the reference.

    PYTHONPATH=<llama.cpp>/gguf-py python ref_mistral_large4.py <file.gguf> <new tokens> <prompt id> [<prompt id> ...]
"""
import math, sys
import numpy as np
from gguf import GGUFReader

reader = GGUFReader(sys.argv[1])
arch = "mistral-large4"


def field(key, default=None):
    f = reader.fields.get(f"{arch}.{key}")
    if f is None:
        return default
    value = f.contents()
    return value


tensors = {t.name: np.asarray(t.data, dtype=np.float64).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}
layers = field("block_count")
heads = field("attention.head_count")
latent = field("attention.kv_lora_rank")
rope = field("rope.dimension_count")
head_k = field("attention.key_length_mla")
rms_eps = field("attention.layer_norm_rms_epsilon")
ln_eps = field("attention.layer_norm_epsilon")
top_k = field("attention.indexer.top_k")
rope_layers = field("attention.rope_pattern")
kv_source = field("attention.kv_source_map")
indexer_full = field("attention.indexer.types")
softcap = field("final_logit_softcapping", 0.0)
dense_lead = field("leading_dense_block_count", 0)
experts_used = field("expert_used_count")
base = field("rope.freq_base")
factor = field("rope.scaling.factor", 1.0)
orig_ctx = field("rope.scaling.original_context_length", 0)
beta_fast = field("rope.scaling.yarn_beta_fast", 32.0)
beta_slow = field("rope.scaling.yarn_beta_slow", 1.0)
yarn = field("rope.scaling.type") == "yarn"
freq_factors = tensors.get("rope_freqs.weight", np.ones(rope // 2))


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + rms_eps) * w


def layer_norm(x, w, b):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + ln_eps) * w + b


def angles(pos):
    # ggml rope (NORM, interleaved pairs) with freq factors and YaRN's ramp; the magnitude is 1 (log multiplier 1, see llama-context.cpp).
    i = np.arange(rope // 2)
    extrap = pos * base ** (-2.0 * i / rope) / freq_factors
    if not yarn:
        return extrap
    corr = lambda beta: rope * math.log(orig_ctx / (beta * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(0.0, math.floor(corr(beta_fast))), min(rope - 1.0, math.ceil(corr(beta_slow)))
    ramp = 1 - np.clip((i - low) / max(0.001, high - low), 0, 1)
    return extrap / factor * (1 - ramp) + extrap * ramp


def rotate(x, positions):
    # x [T, ..., rope]: pairs (2i, 2i+1).
    out = x.copy()
    for t, pos in enumerate(positions):
        a = angles(pos)
        c, s = np.cos(a), np.sin(a)
        x0, x1 = x[t, ..., 0::2], x[t, ..., 1::2]
        out[t, ..., 0::2] = x0 * c - x1 * s
        out[t, ..., 1::2] = x0 * s + x1 * c
    return out


def ssmax(prefix, positions):
    return np.log(positions + 1.0)[:, None] * tensors[prefix + ".weight"][None, :] + tensors[prefix + ".bias"][None, :]


def silu(x):
    return x / (1 + np.exp(-x))


def forward(ids):
    T = len(ids)
    positions = np.arange(T, dtype=np.float64)
    x = tensors["token_embd.weight"][ids]
    caches, selection, gap = {}, None, math.inf
    causal = np.tril(np.ones((T, T), dtype=bool))
    for il in range(layers):
        p = f"blk.{il}."
        has_rope = bool(rope_layers[il])
        rope_l = rope if has_rope else 0
        h = rms(x, tensors[p + "attn_norm.weight"])
        if indexer_full[il]:
            qn = layer_norm(h, tensors[p + "indexer.q_norm.weight"], tensors[p + "indexer.q_norm.bias"])
            dims = tensors[p + "indexer.attn_k.weight"].shape[0]
            iq = (qn @ tensors[p + "indexer.attn_q_b.weight"].T).reshape(T, -1, dims)
            iq = iq * ssmax(p + "indexer.ssmax", positions)[:, :, None]
            ik = layer_norm(h @ tensors[p + "indexer.attn_k.weight"].T, tensors[p + "indexer.k_norm.weight"], tensors[p + "indexer.k_norm.bias"])
            if has_rope:
                iq = np.concatenate([rotate(iq[..., :rope], positions), iq[..., rope:]], axis=-1)
                ik = np.concatenate([rotate(ik[:, :rope], positions), ik[:, rope:]], axis=-1)
            w = h @ tensors[p + "indexer.proj.weight"].T / math.sqrt(dims * iq.shape[1])
            scores = np.einsum("th,thj->tj", w, np.maximum(np.einsum("thd,jd->thj", iq, ik), 0))
            selection = np.zeros((T, T), dtype=bool)
            for t in range(T):
                visible = scores[t, : t + 1]
                if t + 1 <= top_k:
                    selection[t, : t + 1] = True
                    continue
                order = np.argsort(-visible, kind="stable")
                selection[t, order[:top_k]] = True
                gap = min(gap, visible[order[top_k - 1]] - visible[order[top_k]])
        # Latent attention.
        nope_l = head_k - rope_l
        q = (h @ tensors[p + "attn_q.weight"].T).reshape(T, heads, head_k)
        q_nope, q_pe = q[..., :nope_l], q[..., nope_l:]
        if kv_source[il] == il:
            c = h @ tensors[p + "attn_kv_a_mqa.weight"].T
            lat = rms(c[:, :latent], tensors[p + "attn_kv_a_norm.weight"])
            k_pe = rotate(c[:, latent:], positions) if has_rope else np.zeros((T, 0))
            caches[il] = (lat, k_pe)
        lat, k_pe = caches[kv_source[il]]
        q_abs = np.einsum("hln,thn->thl", tensors[p + "attn_k_b.weight"], q_nope)
        scores = np.einsum("thl,jl->thj", q_abs, lat)
        if has_rope:
            scores = scores + np.einsum("thr,jr->thj", rotate(q_pe, positions), k_pe)
        scores = scores * ssmax(p + "attn_ssmax", positions)[:, :, None] / math.sqrt(head_k)
        mask = causal & selection if selection is not None else causal
        scores = np.where(mask[:, None, :], scores, -np.inf)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        o = np.einsum("thj,jl->thl", probs, lat)
        v = np.einsum("hvl,thl->thv", tensors[p + "attn_v_b.weight"], o).reshape(T, -1)
        x = x + v @ tensors[p + "attn_output.weight"].T
        # Feed-forward.
        h = rms(x, tensors[p + "ffn_norm.weight"])
        if il < dense_lead:
            f = (silu(h @ tensors[p + "ffn_gate.weight"].T) * (h @ tensors[p + "ffn_up.weight"].T)) @ tensors[p + "ffn_down.weight"].T
        else:
            logits = h @ tensors[p + "ffn_gate_inp.weight"].T
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            f = np.zeros_like(h)
            for t in range(T):
                top = np.argsort(-probs[t], kind="stable")[:experts_used]
                weights = probs[t, top] / probs[t, top].sum()
                for e, wgt in zip(top, weights):
                    g = silu(tensors[p + "ffn_gate_exps.weight"][e] @ h[t]) * (tensors[p + "ffn_up_exps.weight"][e] @ h[t])
                    f[t] += wgt * (tensors[p + "ffn_down_exps.weight"][e] @ g)
            f = f + (silu(h @ tensors[p + "ffn_gate_shexp.weight"].T) * (h @ tensors[p + "ffn_up_shexp.weight"].T)) @ tensors[p + "ffn_down_shexp.weight"].T
        x = x + f
    logits = rms(x[-1], tensors["output_norm.weight"]) @ tensors["output.weight"].T
    if softcap:
        logits = softcap * np.tanh(logits / softcap)
    return logits, gap


ids = [int(v) for v in sys.argv[3:]]
generated, min_gap = [], math.inf
for _ in range(int(sys.argv[2])):
    logits, gap = forward(ids)
    min_gap = min(min_gap, gap)
    order = np.argsort(-logits)
    ids.append(int(order[0]))
    generated.append(int(order[0]))
print("ids:", ", ".join(map(str, generated)))
print("min indexer gap:", min_gap)
