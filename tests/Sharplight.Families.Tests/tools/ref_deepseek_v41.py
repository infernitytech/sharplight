"""
Float64 numpy reference of llama.cpp's deepseek41 graph (src/models/deepseek41.cpp) for the tiny GGUF of mk_tiny_deepseek_v41.py: greedy ids of
a prompt given as token ids (every step recomputes the whole sequence), plus the smallest relative gap between the last selected and the first
dropped indexer score over all rows and indexer layers (a gap near 0 means a selection depends on rounding). DeepSeek-V4.1 has no transformers
class; this is the reference of the Sharplight test case, with llama.cpp's own output as a second check.

    PYTHONPATH=<llama.cpp>/gguf-py python ref_deepseek_v41.py <file.gguf> <new tokens> <prompt id> [<prompt id> ...]
"""
import math, sys
import numpy as np
from gguf import GGUFReader

reader = GGUFReader(sys.argv[1])
arch = "deepseek41"


def field(key, default=None):
    f = reader.fields.get(f"{arch}.{key}")
    return default if f is None else f.contents()


tensors = {t.name: np.asarray(t.data, dtype=np.float64).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}
layers = field("block_count")
hidden = field("embedding_length")
heads = field("attention.head_count")
head_dim = field("attention.key_length")
rope = field("rope.dimension_count")
eps = field("attention.layer_norm_rms_epsilon")
window = field("attention.sliding_window")
ratios = field("attention.compress_ratios")
top_k = field("attention.indexer.top_k")
index_heads = field("attention.indexer.head_count")
index_dim = field("attention.indexer.key_length")
groups = field("attention.output_group_count")
streams = field("hyper_connection.count")
sinkhorn = field("hyper_connection.sinkhorn_iterations")
hc_eps = field("hyper_connection.epsilon")
used = field("expert_used_count")
w_scale = field("expert_weights_scale")
w_norm = field("expert_weights_norm")
clamp = field("swiglu_clamp_exp")
clamp_shared = field("swiglu_clamp_shexp")
base, comp_base = field("rope.freq_base"), field("attention.compress_rope_freq_base")
factor, orig_ctx = field("rope.scaling.factor", 1.0), field("rope.scaling.original_context_length", 0)
beta_fast, beta_slow = field("rope.scaling.yarn_beta_fast", 32.0), field("rope.scaling.yarn_beta_slow", 1.0)
kv_layers, index_layers = list(field("attention.kv_source_layers")), list(field("attention.indexer.source_layers"))
cand_src = field("attention.indexer.candidate_source_layer", -1)
cand_blocks, cand_size = field("attention.indexer.candidate_top_k_blocks", 0), field("attention.indexer.candidate_block_size", 0)
e_layers = list(field("ple.layers", []))
n_gram, e_heads, e_dim = field("ple.ngram_size", 0), field("ple.heads_per_ngram", 0), field("embedding_length_per_layer_input", 0)
mult, offs, primes = field("ple.layer_multipliers", []), field("ple.head_offsets", []), field("ple.head_vocab_sizes", [])
token_map, pad_id = field("engram.token_map", []), field("engram.pad_token_id", 0)
n_cols = (n_gram - 1) * e_heads

kv_src, idx_src = [-1] * layers, [-1] * layers
last_kv = last_idx = -1
for il in range(layers):
    last_kv = il if il in kv_layers else last_kv
    last_idx = il if il in index_layers else last_idx
    if ratios[il]:
        kv_src[il], idx_src[il] = last_kv, last_idx


def rms(x, w=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def angles(pos, compressed):
    # ggml rope (NORM: interleaved pairs); the compressed layers use their base with YaRN's ramp at magnitude 1 (dsv4_rope_attn_factor).
    i = np.arange(rope // 2)
    b = comp_base if compressed else base
    extrap = pos * b ** (-2.0 * i / rope)
    if not compressed or factor == 1.0:
        return extrap
    corr = lambda beta: rope * math.log(orig_ctx / (beta * 2 * math.pi)) / (2 * math.log(b))
    low, high = max(0.0, math.floor(corr(beta_fast))), min(rope - 1.0, math.ceil(corr(beta_slow)))
    ramp = 1 - np.clip((i - low) / max(0.001, high - low), 0, 1)
    return extrap / factor * (1 - ramp) + extrap * ramp


def rotate_tail(x, positions, compressed, sign=1.0):
    # x [T, n, d]: the last `rope` dimensions, pairs (2i, 2i+1), rotated by sign * angle at the row's position.
    out = x.copy()
    for t, pos in enumerate(positions):
        a = sign * angles(pos, compressed)
        c, s = np.cos(a), np.sin(a)
        tail = x[t, ..., x.shape[-1] - rope:]
        x0, x1 = tail[..., 0::2], tail[..., 1::2]
        out[t, ..., x.shape[-1] - rope::2] = x0 * c - x1 * s
        out[t, ..., x.shape[-1] - rope + 1::2] = x0 * s + x1 * c
    return out


def engram_ids(e, tokens):
    # engram.py NgramHashState.forward as llama_model_deepseek41::engram_hash: blocked shifts read the pad id.
    m = mult[e * n_gram:(e + 1) * n_gram]
    pr, of = primes[e * n_cols:(e + 1) * n_cols], offs[e * n_cols:(e + 1) * n_cols]
    pad = token_map[pad_id]
    cid = lambda t: token_map[t] if 0 <= t < len(token_map) else -1
    out = np.zeros((len(tokens), n_cols), dtype=np.int64)
    mask = (1 << 64) - 1
    for i in range(len(tokens)):
        toks, blocked = [], False
        for s in range(n_gram):
            c = cid(tokens[i - s]) if i - s >= 0 else -1
            blocked = blocked or c == -1
            toks.append(pad if blocked else c)
        rolling = (toks[0] * m[0]) & mask
        for o in range(1, n_gram):
            rolling ^= (toks[o] * m[o]) & mask
            for h in range(e_heads):
                col = (o - 1) * e_heads + h
                out[i, col] = rolling % pr[col] + of[col]
    return out


def engram(x, p, e, tokens):
    rows = tensors["per_layer_token_embd.weight"][engram_ids(e, tokens)].reshape(len(tokens), -1)
    kv = rows @ tensors[p + "engram_kv.weight"].T
    key = kv[:, :streams * hidden].reshape(-1, streams, hidden)
    value = kv[:, streams * hidden:]
    dot = (rms(x) * rms(key) * tensors[p + "engram_gate.weight"]).sum(-1) / math.sqrt(hidden)
    gate = 1 / (1 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)))
    return x + value[:, None, :] * gate[:, :, None]


def hc_mixes(x, p):
    flat = rms(x.reshape(len(x), -1))
    m = flat @ tensors[p + "fn.weight"].T
    scale, bias = tensors[p + "scale.weight"], tensors[p + "base.weight"]
    sig = lambda v: 1 / (1 + np.exp(-v))
    pre = sig(m[:, :streams] * scale[0] + bias[:streams]) + hc_eps
    post = 2 * sig(m[:, streams:2 * streams] * scale[1] + bias[streams:2 * streams])
    comb = (m[:, 2 * streams:] * scale[2] + bias[2 * streams:]).reshape(-1, streams, streams)   # [T, src, dst]
    comb = np.exp(comb - comb.max(-1, keepdims=True))
    comb = comb / comb.sum(-1, keepdims=True) + hc_eps
    comb = comb / (comb.sum(1, keepdims=True) + hc_eps)
    for _ in range(1, sinkhorn):
        comb = comb / (comb.sum(2, keepdims=True) + hc_eps)
        comb = comb / (comb.sum(1, keepdims=True) + hc_eps)
    return pre, post, comb


def hc_pre(x, pre):
    return (x * pre[:, :, None]).sum(1)


def hc_post(y, residual, post, comb):
    return post[:, :, None] * y[:, None, :] + np.einsum("tsd,tsh->tdh", comb, residual)


def softmax_last(v):
    v = np.exp(v - v.max(-1, keepdims=True))
    return v / v.sum(-1, keepdims=True)


gaps = []


def top(scores, k):
    # Indices of the k best scores above -inf (all of them when there are at most k; +inf pins one), and the relative gap at the cut.
    finite = np.flatnonzero(scores > -np.inf)
    if len(finite) <= k:
        return finite
    order = finite[np.argsort(-scores[finite], kind="stable")]
    a, b = scores[order[k - 1]], scores[order[k]]
    if np.isfinite(a):
        gaps.append(abs(a - b) / max(abs(a), abs(b), 1e-30))
    return np.sort(order[:k])


def forward(tokens):
    T = len(tokens)
    pos = np.arange(T)
    x = np.repeat(tensors["token_embd.weight"][tokens][:, None, :], streams, axis=1)
    comp = {}      # kv source layer -> (entries [n, D], index keys [n, Di])
    selected = {}  # index source layer -> per row the selected entries
    candidates = None
    pre_prev = None
    for il in range(layers):
        p = f"blk.{il}."
        if il in e_layers:
            x = engram(x, p, e_layers.index(il), tokens)
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(x, p + "hc_attn_")
        cur = x[:, 0] if pre_prev is None else hc_pre(x, pre_prev)
        cur = rms(cur, tensors[p + "attn_norm.weight"])

        r = ratios[il]
        compressed = r != 0
        qr = rms(cur @ tensors[p + "attn_q_a.weight"].T, tensors[p + "attn_q_a_norm.weight"])
        q = rotate_tail((qr @ tensors[p + "attn_q_b.weight"].T).reshape(T, heads, head_dim), pos, compressed)
        kv = rms(cur @ tensors[p + "attn_kv.weight"].T, tensors[p + "attn_kv_a_norm.weight"])
        kv = rotate_tail(kv[:, None, :], pos, compressed)[:, 0]
        if r and kv_src[il] == il:
            v = cur @ tensors[p + "attn_compressor_kv.weight"].T
            s = cur @ tensors[p + "attn_compressor_gate.weight"].T if r > 1 else np.zeros_like(v)
            n = T // r
            pooled = np.array([(softmax_last(s[w * r:(w + 1) * r].T) * v[w * r:(w + 1) * r].T).sum(-1) for w in range(n)]).reshape(n, head_dim)
            latent = rms(pooled, tensors[p + "attn_compressor_norm.weight"])
            first = np.arange(n) * r
            ik = rms(latent @ tensors[p + "indexer.attn_k.weight"].T, tensors[p + "indexer.k_norm.weight"])
            comp[il] = (rotate_tail(latent[:, None, :], first, True)[:, 0], rotate_tail(ik[:, None, :], first, True)[:, 0])

        sel = None
        if r:
            entries, index_keys = comp[kv_src[il]]
            visible = (pos + 1) // r
            if idx_src[il] == il:
                iq = rotate_tail((qr @ tensors[p + "indexer.attn_q_b.weight"].T).reshape(T, index_heads, index_dim), pos, True)
                iw = (cur @ tensors[p + "indexer.proj.weight"].T) / math.sqrt(index_dim * index_heads)
                sel, blocks = [], []
                for t in range(T):
                    score = np.full(len(entries), -np.inf)
                    nv = visible[t]
                    score[:nv] = (np.maximum(iq[t] @ index_keys[:nv].T, 0) * iw[t][:, None]).sum(0)
                    if cand_src == il:
                        nb = (nv + cand_size - 1) // cand_size
                        best = np.array([score[b * cand_size:(b + 1) * cand_size].max() for b in range(nb)])
                        if nb:
                            best[nb - 1] = np.inf
                        blocks.append(set(top(best, cand_blocks).tolist()) if nb > cand_blocks else set(range(nb)))
                    if 0 <= cand_src < il:
                        allowed = np.array([j // cand_size in candidates[t] for j in range(len(score))], dtype=bool)
                        score = np.where(allowed, score, -np.inf)
                    sel.append(top(score, top_k))
                if cand_src == il:
                    candidates = blocks
                selected[il] = sel
            else:
                sel = selected[idx_src[il]]

        out = np.zeros((T, heads, head_dim))
        sinks = tensors[p + "attn_sinks.weight"]
        for t in range(T):
            keys = [kv[j] for j in range(max(0, t - window + 1), t + 1)]
            if r:
                keys += [entries[e] for e in sel[t]]
            K = np.array(keys)
            logits = q[t] @ K.T / math.sqrt(head_dim)                                   # [H, n]
            m = np.maximum(logits.max(-1), sinks)
            e = np.exp(logits - m[:, None])
            out[t] = (e @ K) / (e.sum(-1) + np.exp(sinks - m))[:, None]
        out = rotate_tail(out, pos, compressed, sign=-1.0).reshape(T, groups, -1)
        wa = tensors[p + "attn_output_a.weight"].reshape(groups, -1, heads * head_dim // groups)
        oa = np.einsum("grk,tgk->tgr", wa, out).reshape(T, -1)
        attn = oa @ tensors[p + "attn_output_b.weight"].T
        x = hc_post(attn, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(x, p + "hc_ffn_")
        pre_prev = ffn_pre
        cur = rms(hc_pre(x, attn_pre), tensors[p + "ffn_norm.weight"])
        probs = np.sqrt(np.log1p(np.exp(cur @ tensors[p + "ffn_gate_inp.weight"].T)))
        choice = probs + tensors[p + "exp_probs_b.bias"]
        y = np.zeros_like(cur)
        for t in range(T):
            ids = np.argsort(-choice[t], kind="stable")[:used]
            wts = probs[t, ids]
            if w_norm and used > 1:
                wts = wts / (wts.sum() + 1e-20)
            wts = wts * w_scale
            for ex, wt in zip(ids, wts):
                g = np.minimum(tensors[p + "ffn_gate_exps.weight"][ex] @ cur[t], clamp[il])
                u = np.clip(tensors[p + "ffn_up_exps.weight"][ex] @ cur[t], -clamp[il], clamp[il])
                y[t] += wt * (tensors[p + "ffn_down_exps.weight"][ex] @ (g / (1 + np.exp(-g)) * u))
        g = np.minimum(cur @ tensors[p + "ffn_gate_shexp.weight"].T, clamp_shared[il])
        u = np.clip(cur @ tensors[p + "ffn_up_shexp.weight"].T, -clamp_shared[il], clamp_shared[il])
        y += (g / (1 + np.exp(-g)) * u) @ tensors[p + "ffn_down_shexp.weight"].T
        x = hc_post(y, residual, ffn_post, ffn_comb)

    h = rms(hc_pre(x, pre_prev), tensors["output_norm.weight"])
    return h @ tensors["output.weight"].T


count = int(sys.argv[2])
tokens = [int(a) for a in sys.argv[3:]]
generated = []
for _ in range(count):
    logits = forward(tokens + generated)
    generated.append(int(np.argmax(logits[-1])))
print("ids:", ", ".join(map(str, generated)))
print(f"smallest indexer gap: {min(gaps) if gaps else float('nan'):.3e}")
