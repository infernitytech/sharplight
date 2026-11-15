"""
Float64 numpy reference of llama.cpp's metis16 graph (src/models/metis16.cpp) for the tiny GGUF of mk_tiny_gguf.py: greedy ids of a prompt given as
token ids, and the smallest top-2 logit margin. Metis 1.6's Hugging Face code is remote code, so this is the reference.

    PYTHONPATH=<llama.cpp>/gguf-py python ref_metis16.py <file.gguf> <new tokens> <prompt id> [<prompt id> ...]
"""
import math, sys
import numpy as np
from gguf import GGUFReader

reader = GGUFReader(sys.argv[1])
arch = "metis16"


def field(key, default=None):
    f = reader.fields.get(f"{arch}.{key}")
    return default if f is None else f.contents()


W = {t.name: np.asarray(t.data, dtype=np.float64).reshape([int(d) for d in reversed(t.shape)]) for t in reader.tensors}
virtual = field("block_count")
passes = field("metis.pass_count")
phys = virtual // passes
kv_heads = field("attention.head_count_kv")
heads = field("attention.head_count")
hidden = field("embedding_length")
head_dim = hidden // heads
eps = field("attention.layer_norm_rms_epsilon")
ssm_eps = field("ssm.norm_epsilon")
n_mem = field("metis.memory_length")
limit = field("metis.continuation_logit_limit")
used = field("expert_used_count")
ssm_heads = field("ssm.time_step_rank")
inner = field("ssm.inner_size")
state_size = field("ssm.state_size")
groups = field("ssm.group_count")
conv_kernel = field("ssm.conv_kernel")


def rms(x, e=eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + e)


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def softplus(x):
    return np.log1p(np.exp(x))


def mamba2(h, p):
    T = h.shape[0]
    hd, n, g = inner // ssm_heads, state_size, groups
    proj = h @ W[p + "ssm_in.weight"].T
    z, xbc, dt = proj[:, :inner], proj[:, inner:2 * inner + 2 * g * n], proj[:, 2 * inner + 2 * g * n:]
    pad = np.concatenate([np.zeros((conv_kernel - 1, xbc.shape[1])), xbc])
    cw = W[p + "ssm_conv1d.weight"]
    conv = np.stack([(pad[t:t + conv_kernel] * cw.T).sum(0) for t in range(T)]) + W[p + "ssm_conv1d.bias"]
    conv = silu(conv)
    x = conv[:, :inner].reshape(T, ssm_heads, hd)
    B = conv[:, inner:inner + g * n].reshape(T, g, n)
    C = conv[:, inner + g * n:].reshape(T, g, n)
    dt = softplus(dt + W[p + "ssm_dt.bias"])
    A, D = W[p + "ssm_a"].reshape(-1), W[p + "ssm_d"].reshape(-1)
    group_of = np.arange(ssm_heads) // (ssm_heads // g)
    s = np.zeros((ssm_heads, hd, n))
    y = np.zeros((T, ssm_heads, hd))
    for t in range(T):
        s = s * np.exp(dt[t] * A)[:, None, None] + (dt[t][:, None, None] * x[t][:, :, None]) * B[t][group_of][:, None, :]
        y[t] = (s * C[t][group_of][:, None, :]).sum(-1) + D[:, None] * x[t]
    y = y.reshape(T, inner) * silu(z)
    y = rms(y.reshape(T, g, inner // g), ssm_eps).reshape(T, inner) * W[p + "ssm_norm.weight"].reshape(-1)
    return y @ W[p + "ssm_out.weight"].T


def attention(h, p, pas, n_kv):
    T = h.shape[0]
    qkv = h @ W[p + "attn_qkv.weight"].T
    qkv = qkv + W[p + "attn_pass_gate.weight"][pas] * ((h @ W[p + "attn_pass_lora_down.weight"].T) @ W[p + "attn_pass_lora_up.weight"].T)
    q = qkv[:, :heads * head_dim].reshape(T, heads, head_dim)
    k = qkv[:, heads * head_dim:(heads + n_kv) * head_dim].reshape(T, n_kv, head_dim)
    v = qkv[:, (heads + n_kv) * head_dim:].reshape(T, n_kv, head_dim)
    k, v = np.repeat(k, heads // n_kv, axis=1), np.repeat(v, heads // n_kv, axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / math.sqrt(head_dim)
    scores = scores + np.triu(np.full((T, T), -np.inf), 1)
    y = np.einsum("hts,shd->thd", softmax(scores), v)
    return y.reshape(T, -1) @ W[p + "attn_output.weight"].T


def forward(ids):
    T = len(ids)
    x = W["token_embd.weight"][ids][:, None, :] + W["stream_embd.weight"].T[None]           # [T, S, H]
    bank_k, bank_v = [], []

    def mem_read():
        nonlocal x
        q = x @ W["depth_mem_query.weight"].T + W["depth_mem_query.bias"]                   # [T, S, M]
        K, V = np.stack(bank_k, 1), np.stack(bank_v, 1)                                    # [T, e, M]
        a = softmax(np.einsum("tsm,tem->tse", q, K) / math.sqrt(n_mem))
        ret = np.einsum("tse,tem->tsm", a, V)
        proj = ret @ W["depth_mem_out.weight"].T + W["depth_mem_out.bias"]                 # [T, S, H]
        gate = sigmoid((x * W["depth_mem_stream_gate.weight"][None]).sum(-1) + W["depth_mem_stream_gate.bias"])
        x = x + proj * gate[:, :, None]
        return proj.mean(1)

    def mem_write(summ, ent, conf, cont, entry):
        meta = np.concatenate([summ, ent, conf, cont if cont is not None else 0 * conf], -1)
        e = x.mean(1) @ W["depth_mem_state_write.weight"].T + meta @ W["depth_mem_meta_write.weight"].T + W["depth_mem_entry_bias.weight"][entry]
        bank_k.append(e @ W["depth_mem_key.weight"].T + W["depth_mem_key.bias"])
        bank_v.append(e @ W["depth_mem_value.weight"].T + W["depth_mem_value.bias"])

    def features(state, summary, diff, history):
        f = state @ W["depth_mem_route_state.weight"].T
        if summary is not None:
            f = f + summary @ W["depth_mem_route_retr.weight"].T
        if diff is not None:
            f = f + diff @ W["depth_mem_route_diff.weight"].T
        if history is not None:
            f = f + history @ W["depth_mem_route_hist.weight"].T
        return np.tanh(f + W["depth_mem_route.bias"])

    def hc_read(p, kind, pas):
        mixed = np.einsum("ik,tkh->tih", W[p + f"hc_{kind}_mix.weight"][pas], x)
        src = np.einsum("k,tkh->th", W[p + f"hc_{kind}_read.weight"][pas], x)
        return mixed, rms(src) * W[p + ("attn_norm.weight" if kind == "attn" else "ffn_norm.weight")]

    def hc_write(mixed, p, kind, pas, update):
        return mixed + W[p + f"hc_{kind}_write.weight"][pas][None, :, None] * update[:, None, :]

    prev_state = x.mean(1)
    history, cont, entry = None, None, 0
    for pas in range(passes):
        for il_p in range(phys):
            il = pas * phys + il_p
            p = f"blk.{il_p}."
            is_attn = kv_heads[il] > 0
            writes = is_attn or il_p == phys - 1
            retrieved = mem_read() if bank_k else None
            feats = features(x.mean(1), retrieved, None, history)
            mixed, cur = hc_read(p, "attn", pas)
            cur = attention(cur, p, pas, kv_heads[il]) if is_attn else mamba2(cur, p)
            x = hc_write(mixed, p, "attn", pas, cur)
            mixed, cur = hc_read(p, "ffn", pas)
            latent = cur @ W[p + "ffn_latent_down.weight"].T
            logits = np.concatenate([latent, feats], -1) @ W[p + "ffn_gate_inp.weight"].T + W[p + "ffn_gate_inp.bias"]
            selected = np.argsort(-(logits + W[p + "exp_probs_b.bias"]), -1, kind="stable")[:, :used]
            weights = softmax(np.take_along_axis(logits, selected, -1))
            moe = np.zeros_like(latent)
            for t in range(T):
                for j, e in enumerate(selected[t]):
                    gu = W[p + "ffn_gate_up_exps.weight"][e] @ latent[t]
                    half = gu.shape[0] // 2
                    moe[t] += weights[t, j] * (W[p + "ffn_down_exps.weight"][e] @ (silu(gu[:half]) * gu[half:]))
            shared = (silu(latent @ W[p + "ffn_gate_shexp.weight"].T) * (latent @ W[p + "ffn_up_shexp.weight"].T)) @ W[p + "ffn_down_shexp.weight"].T
            cur = (moe + shared) @ W[p + "ffn_latent_up.weight"].T
            summ = np.einsum("tk,tkr->tr", weights, W[p + "ffn_exp_embd.weight"][selected])
            if writes:
                probs = softmax(logits)
                ent = -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(-1, keepdims=True)
                conf = probs.max(-1, keepdims=True)
            x = hc_write(mixed, p, "ffn", pas, cur)
            history = 0.5 * summ if history is None else 0.5 * history + 0.5 * summ
            if is_attn:
                mem_write(summ, ent, conf, cont, entry)
                entry += 1
        retrieved = mem_read()
        state = x.mean(1)
        diff = state - prev_state
        feats = features(state, retrieved, diff, history)
        h = silu(state @ W["cont_hidden_state.weight"].T + retrieved @ W["cont_hidden_retr.weight"].T + diff @ W["cont_hidden_diff.weight"].T
                 + feats @ W["cont_hidden_route.weight"].T + W["cont_hidden.bias"])
        logit = h @ W["cont_output.weight"].T + W["cont_output.bias"]
        scale = np.maximum(np.sqrt((state * state).mean(-1, keepdims=True)), 1e-6)
        innov = np.sqrt((diff * diff).mean(-1, keepdims=True))
        logit = logit + np.tanh(innov / scale) - 0.5
        cont = sigmoid(np.clip(logit, -limit, limit))
        mem_write(summ, ent, conf, cont, entry)
        entry += 1
        prev_state = state
    head = W["output.weight"] if "output.weight" in W else W["token_embd.weight"]
    return rms(x.mean(1)[-1]) * W["output_norm.weight"] @ head.T


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
