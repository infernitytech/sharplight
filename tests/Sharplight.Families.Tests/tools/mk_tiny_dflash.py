#!/usr/bin/env python3
"""Writes two tiny random DFlash drafters (llama.cpp `dflash` GGUF) for the tiny EAGLE-3 test target, plus their reference drafts.

Target: the tiny random Llama of mk_tiny_eagle3.py (6 layers, hidden 64, llama3 rope, tied embeddings; HF directory for the reference,
its f32 GGUF for the engines). Both drafters fuse the inputs of target layers 1, 3 and 5 (GGUF target_layers are layer-input ids), have hidden 64
(they share the target's token embedding and LM head), 2 layers of 4 heads / 2 KV heads of 16, an MLP of 128, a block of 8 and the Llama-3.2
tokenizer of the target with <|reserved_special_token_0|> (128002) as the mask token.

- tiny-random-dflash-f32.gguf: plain DFlash; layer 1 has a sliding window (4096, longer than any test context).
- tiny-random-dflash2-f32.gguf: DFlash2: a kernel-2 dynamic convolution (group 16) around attention and MLP, a selector lattice (rank 8,
  top 4 candidates, logits scaled by 0.5 and soft-capped at 30), attention sinks, a value scale, post-attention / post-MLP norms, layer
  output scales and degenerate M-RoPE sections [8, 0, 0, 0].

The reference (<gguf>.reference.json next to each file) is a float64 numpy forward of llama.cpp's dflash graph (src/models/dflash.cpp,
common/speculative.cpp): the KV cache of position p holds rope(k_norm(wk g_p)) and wv g_p with g = rms_norm(fc(features_p)) · enc.output_norm;
the block [pending, mask x 3] runs non-causally over the cache and itself; plain drafts are the argmax of block rows 1..3, DFlash2 drafts walk
the lattice. Two flows are simulated with greedy verification by the target (transformers, f32): "engine" (the prompt is decoded, the pending
token is the target's greedy token, as Sharplight's engine does) and "llama" (as llama-speculative-simple: the last prompt token is pending).
Each records the first proposal and, over 24 generated tokens, every proposal with the length of the context it continues and its smallest top-2 logit (or lattice score) gap, drafted /
accepted counts and the output. The minimum gap of all proposals is printed.

Usage: mk_tiny_dflash.py [TARGET_HF_DIR] [OUT_DIR] [GGUF_PY_DIR]
       (defaults ~/models/hf/tiny-random-llama-eagle3-target, ~/models/gguf, the gguf-py of the llama.cpp checkout in GGUF_PY)
DFLASH_SEED (default 3) picks the weights. Needs numpy, torch, transformers and llama.cpp's gguf-py.
"""
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
TARGET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "models/hf/tiny-random-llama-eagle3-target")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "models/gguf")
GGUF_PY = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("GGUF_PY", "")
if GGUF_PY:
    sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

TARGET_GGUF = os.environ.get("DFLASH_TARGET_GGUF", os.path.join(HOME, "models/gguf/tiny-random-llama-eagle3-target-f32.gguf"))
T_HIDDEN, T_VOCAB = 64, 128256
HIDDEN, LAYERS, HEADS, KV, HD, FFN = 64, 2, 4, 2, 16, 128
TAPS = [1, 3, 5]
BLOCK, DRAFTS, MASK = 8, 3, 128002
EPS, THETA = 1e-6, 1000000.0
PROMPTS = [[128000, 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13],
           [128000, 40, 1093, 311, 1373, 6603, 922, 279, 2294, 23651, 315, 3823, 7138, 323, 13]]
GENERATE = 24

rng = np.random.default_rng(int(os.environ.get("DFLASH_SEED", "3")))


def lin(out_features, in_features, scale=0.15):
    return (rng.standard_normal((out_features, in_features)) * scale).astype(np.float32)


def norm(size):
    return (1.0 + rng.standard_normal(size) * 0.1).astype(np.float32)


def make_weights(v2):
    w = {"fc.weight": lin(HIDDEN, len(TAPS) * T_HIDDEN, 0.1), "enc.output_norm.weight": norm(HIDDEN), "output_norm.weight": norm(HIDDEN)}
    for il in range(LAYERS):
        p = f"blk.{il}."
        w[p + "attn_norm.weight"] = norm(HIDDEN)
        w[p + "attn_q.weight"] = lin(HEADS * HD, HIDDEN)
        w[p + "attn_k.weight"] = lin(KV * HD, HIDDEN)
        w[p + "attn_v.weight"] = lin(KV * HD, HIDDEN)
        w[p + "attn_output.weight"] = lin(HIDDEN, HEADS * HD)
        w[p + "attn_q_norm.weight"] = norm(HD)
        w[p + "attn_k_norm.weight"] = norm(HD)
        w[p + "ffn_norm.weight"] = norm(HIDDEN)
        w[p + "ffn_gate.weight"] = lin(FFN, HIDDEN)
        w[p + "ffn_up.weight"] = lin(FFN, HIDDEN)
        w[p + "ffn_down.weight"] = lin(HIDDEN, FFN)
        if v2:
            groups = HIDDEN // 16
            w[p + "attn_conv_base"] = (rng.standard_normal((2, 2, HIDDEN)) * 0.5).astype(np.float32)
            w[p + "attn_conv_proj.weight"] = lin(2 * 2 * groups, HIDDEN, 0.2)
            w[p + "ffn_conv_base"] = (rng.standard_normal((2, 2, HIDDEN)) * 0.5).astype(np.float32)
            w[p + "ffn_conv_proj.weight"] = lin(2 * 2 * groups, HIDDEN, 0.2)
            w[p + "attn_sinks.weight"] = (rng.standard_normal(HEADS) * 0.5).astype(np.float32)
            w[p + "post_attention_norm.weight"] = norm(HIDDEN)
            w[p + "post_ffw_norm.weight"] = norm(HIDDEN)
            w[p + "layer_output_scale.weight"] = np.array([0.8 + 0.1 * il], dtype=np.float32)
    if v2:
        w["selector_predecessor.weight"] = (rng.standard_normal((T_VOCAB, 8)) * 0.6).astype(np.float32)
        w["selector_successor.weight"] = (rng.standard_normal((T_VOCAB, 8)) * 0.6).astype(np.float32)
        w["selector_hidden.weight"] = lin(8, HIDDEN, 0.3)
    return w


VARIANTS = {
    "tiny-random-dflash-f32.gguf": dict(v2=False, sliding=[0, 1], window=4096),
    "tiny-random-dflash2-f32.gguf": dict(v2=True, sliding=[0, 0], window=0, value_scale=0.9, logit_scale=0.5, softcap=30.0, top_k=4, rank=8,
                                         sections=[8, 0, 0, 0]),
}


def write_gguf(path, w, cfg):
    reader = gguf.GGUFReader(TARGET_GGUF)
    writer = gguf.GGUFWriter(path, "dflash")
    a = "dflash."
    writer.add_uint32(a + "block_count", LAYERS)
    writer.add_uint32(a + "context_length", 4096)
    writer.add_uint32(a + "embedding_length", HIDDEN)
    writer.add_uint32(a + "feed_forward_length", FFN)
    writer.add_uint32(a + "attention.head_count", HEADS)
    writer.add_uint32(a + "attention.head_count_kv", KV)
    writer.add_uint32(a + "attention.key_length", HD)
    writer.add_uint32(a + "attention.value_length", HD)
    writer.add_float32(a + "rope.freq_base", THETA)
    writer.add_float32(a + "attention.layer_norm_rms_epsilon", EPS)
    writer.add_uint32(a + "block_size", BLOCK)
    writer.add_array(a + "target_layers", TAPS)
    if cfg["window"]:
        writer.add_uint32(a + "attention.sliding_window", cfg["window"])
        writer.add_array(a + "attention.sliding_window_pattern", [bool(s) for s in cfg["sliding"]])
    if cfg["v2"]:
        writer.add_uint32(a + "conv_kernel_size", 2)
        writer.add_uint32(a + "conv_group_size", 16)
        writer.add_uint32(a + "selector_rank", cfg["rank"])
        writer.add_uint32(a + "selector_top_k", cfg["top_k"])
        writer.add_float32(a + "attention.value_scale", cfg["value_scale"])
        writer.add_float32(a + "logit_scale", cfg["logit_scale"])
        writer.add_float32(a + "final_logit_softcapping", cfg["softcap"])
        writer.add_array(a + "rope.dimension_sections", cfg["sections"])
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer."):
            continue
        kind = field.types[0]
        if kind == gguf.GGUFValueType.ARRAY:
            writer.add_key_value(key, field.contents(), kind, field.types[1])
        else:
            writer.add_key_value(key, field.contents(), kind)
    writer.add_uint32("tokenizer.ggml.mask_token_id", MASK)
    for name, array in w.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# --- numpy forward of llama.cpp's dflash graph ---
def rms(x, weight=None):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return y if weight is None else y * weight


def rope(x, pos):  # x [..., H, D] at positions pos [...], neox pairs (i, i + D/2)
    freq = THETA ** (-np.arange(0, HD, 2, dtype=np.float64) / HD)
    ang = np.asarray(pos, dtype=np.float64)[..., None, None] * freq
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = x[..., : HD // 2], x[..., HD // 2:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)


def silu(x):
    return x / (1 + np.exp(-x))


def conv(x, coefficients, base, side):  # x [n, H] rows of one block, coefficients [n, 2*K*G], base [2, K, H]
    groups = HIDDEN // 16
    c = coefficients.reshape(len(x), 2, 2, groups)[:, side]   # [n, K, G]
    out = np.zeros_like(x)
    for tap in range(2):
        weight = np.repeat(c[:, tap], 16, axis=-1) + base[side, tap]
        shifted = np.zeros_like(x)
        shifted[tap:] = x[: len(x) - tap] if tap else x
        out += weight * shifted
    return out


class Drafter:
    def __init__(self, w, cfg, embed):
        self.w = {k: v.astype(np.float64) for k, v in w.items()}
        self.cfg = cfg
        self.embed = embed

    def propose(self, features, pending):
        """Drafts after `pending` at position len(features), the cache holding the features of positions 0..len(features)-1."""
        w, cfg = self.w, self.cfg
        n_ctx = len(features)
        g = rms(features @ w["fc.weight"].T, w["enc.output_norm.weight"])
        tokens = [pending] + [MASK] * DRAFTS
        pos = np.arange(n_ctx, n_ctx + len(tokens))
        x = self.embed[tokens].astype(np.float64)
        for il in range(LAYERS):
            p = f"blk.{il}."
            kc = rope(rms((g @ w[p + "attn_k.weight"].T).reshape(n_ctx, KV, HD), w[p + "attn_k_norm.weight"]), np.arange(n_ctx))
            vc = (g @ w[p + "attn_v.weight"].T).reshape(n_ctx, KV, HD)
            n = rms(x, w[p + "attn_norm.weight"])
            if cfg["v2"]:
                ac = n @ w[p + "attn_conv_proj.weight"].T
                n = conv(n, ac, w[p + "attn_conv_base"], 0)
            q = rope(rms((n @ w[p + "attn_q.weight"].T).reshape(-1, HEADS, HD), w[p + "attn_q_norm.weight"]), pos)
            k = rope(rms((n @ w[p + "attn_k.weight"].T).reshape(-1, KV, HD), w[p + "attn_k_norm.weight"]), pos)
            v = (n @ w[p + "attn_v.weight"].T).reshape(-1, KV, HD)
            K, V = np.concatenate([kc, k]), np.concatenate([vc, v])
            kpos = np.arange(len(K))
            out = np.zeros((len(tokens), HEADS, HD))
            for t in range(len(tokens)):
                visible = np.ones(len(K), dtype=bool)
                if cfg["window"] and cfg["sliding"][il]:
                    visible = pos[t] - kpos < cfg["window"]
                for h in range(HEADS):
                    s = K[visible, h // (HEADS // KV)] @ q[t, h] / np.sqrt(HD)
                    m = s.max()
                    e = np.exp(s - m)
                    denominator = e.sum()
                    if p + "attn_sinks.weight" in w:
                        m2 = max(m, w[p + "attn_sinks.weight"][h])
                        e = np.exp(s - m2)
                        denominator = e.sum() + np.exp(w[p + "attn_sinks.weight"][h] - m2)
                    out[t, h] = (e / denominator) @ V[visible, h // (HEADS // KV)]
            o = out.reshape(len(tokens), -1) @ w[p + "attn_output.weight"].T
            if cfg.get("value_scale"):
                o = o * cfg["value_scale"]
            if cfg["v2"]:
                o = conv(o, ac, w[p + "attn_conv_base"], 1)
                o = rms(o, w[p + "post_attention_norm.weight"])
            h = o + x
            f = rms(h, w[p + "ffn_norm.weight"])
            if cfg["v2"]:
                fc = f @ w[p + "ffn_conv_proj.weight"].T
                f = conv(f, fc, w[p + "ffn_conv_base"], 0)
            m = (silu(f @ w[p + "ffn_gate.weight"].T) * (f @ w[p + "ffn_up.weight"].T)) @ w[p + "ffn_down.weight"].T
            if cfg["v2"]:
                m = conv(m, fc, w[p + "ffn_conv_base"], 1)
                m = rms(m, w[p + "post_ffw_norm.weight"])
            x = m + h
            if p + "layer_output_scale.weight" in w:
                x = x * w[p + "layer_output_scale.weight"][0]
        hidden = rms(x, w["output_norm.weight"])
        logits = hidden @ self.embed.T
        if not cfg["v2"]:
            drafts = [int(np.argmax(logits[i])) for i in range(1, len(tokens))]
            gaps = [float(np.diff(np.sort(logits[i])[-2:])[0]) for i in range(1, len(tokens))]
            return drafts, min(gaps)
        logits = cfg["softcap"] * np.tanh(logits * cfg["logit_scale"] / cfg["softcap"])
        k = cfg["top_k"]
        cand = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
        unary = np.take_along_axis(logits, cand, -1)
        gate = hidden @ w["selector_hidden.weight"].T
        drafts, gaps = [], []
        pred = 0
        for i in range(1, len(tokens)):
            prev_ids = [tokens[0]] * k if i == 1 else list(cand[i - 1])
            prev = w["selector_predecessor.weight"][prev_ids] * gate[i]        # [k(p), rank]
            succ = w["selector_successor.weight"][cand[i]]                     # [k(c), rank]
            scores = prev @ succ.T + unary[i]                                  # [p, c]
            row = scores[pred]
            best = int(np.argmax(row))
            gaps.append(float(np.diff(np.sort(row)[-2:])[0]))
            pred = best
            drafts.append(int(cand[i][best]))
        return drafts, min(gaps)


def main():
    import torch
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(TARGET, torch_dtype=torch.float32).eval()
    embed = model.model.embed_tokens.weight.detach().double().numpy()

    def target(tokens):
        with torch.no_grad():
            out = model(torch.tensor([tokens]), output_hidden_states=True)
        feats = np.concatenate([out.hidden_states[i][0].double().numpy() for i in TAPS], -1)
        return feats, out.logits[0].double().numpy().argmax(-1)

    for name, cfg in VARIANTS.items():
        w = make_weights(cfg["v2"])
        path = os.path.join(OUT, name)
        write_gguf(path, w, cfg)
        drafter = Drafter(w, cfg, embed)
        min_gap = np.inf
        reference = []
        for prompt in PROMPTS:
            entry = {"prompt": prompt}
            for flow in ("engine", "llama"):
                if flow == "engine":
                    committed = list(prompt)
                    pending = int(target(committed)[1][-1])
                else:
                    committed, pending = list(prompt[:-1]), prompt[-1]
                output, proposals, gaps, lengths, drafted, accepted = [], [], [], [], 0, 0
                generated = 0 if flow == "llama" else 1
                if flow == "engine":
                    output.append(pending)
                while generated < GENERATE:
                    feats, _ = target(committed)
                    drafts, gap = drafter.propose(feats, pending)
                    min_gap = min(min_gap, gap)
                    proposals.append(drafts)
                    gaps.append(round(gap, 5))
                    lengths.append(len(committed) + 1)
                    _, greedy = target(committed + [pending] + drafts)
                    tail = greedy[len(committed):]
                    keep = 0
                    while keep < len(drafts) and drafts[keep] == int(tail[keep]):
                        keep += 1
                    drafted += len(drafts)
                    accepted += keep
                    committed += [pending] + drafts[:keep]
                    pending = int(tail[keep])
                    output += drafts[:keep] + [pending]
                    generated += keep + 1
                entry[flow] = {"pending": int(output[0]) if flow == "engine" else prompt[-1], "first_drafts": proposals[0], "proposals": proposals, "gaps": gaps,
                               "context_lengths": lengths,
                               "drafted": drafted, "accepted": accepted, "output": output}
            reference.append(entry)
        with open(path + ".reference.json", "w") as f:
            json.dump(reference, f, indent=1)
        print(name, json.dumps([{"first": e["engine"]["first_drafts"], "acc": [e[fl]["accepted"] for fl in ("engine", "llama")],
                                 "drafted": [e[fl]["drafted"] for fl in ("engine", "llama")]} for e in reference]))
        print(f"wrote {path}; minimum draft gap {min_gap:.4f}")


main()
