"""
Writes tiny random GGUF files for llama.cpp architectures that have no Hugging Face class in transformers (so no tiny-random checkpoint
and no converter path): Qwen v1 (qwen), XVERSE (xverse) and ChatGLM (chatglm), the RMS-normed RWKV distillations ARWKV7 (arwkv7, with and without output gate) and QRWKV6 (rwkv6qwen2), and PLaMo-2 (plamo2), plus the later llama.cpp-only families (write_family). The weights are random, written with llama.cpp's tensor names
and hyper-parameter keys; llama.cpp (llama-completion) provides the reference continuation. The tokenizer fields are copied from a GGUF
that already carries the 1024-token GPT-2-style vocabulary of tiny-random-GPTNeoXForCausalLM, e.g. tiny-random-falcon-f32.gguf converted
from ~/models/hf/tiny-random-falcon with convert_hf_to_gguf.py.

    PYTHONPATH=<llama.cpp>/gguf-py python mk_tiny_gguf.py <tokenizer gguf> <output directory> [name ...]

Files: <output directory>/tiny-random-<name>-f32.gguf for every name of the models table below (only the named ones when names are given). The
bidirectional embedding families (eurobert, neo-bert, llama-embed, gemma-embedding, modern-bert, jina-bert-v2) get their references from llamacpp_embeddings.py. Norm weights
and biases are randomized so that every tensor matters.
"""
import os, sys
import numpy as np
from gguf import GGUFReader, GGUFWriter

HIDDEN, HEADS, KV_HEADS, FFN, LAYERS, CONTEXT = 64, 4, 2, 128, 3, 256
HEAD_DIM = HIDDEN // HEADS
rng = np.random.default_rng(0)


def weight(*shape):
    return (rng.standard_normal(shape) * 0.15).astype(np.float32)


def norm(n):
    return (1.0 + 0.3 * rng.standard_normal(n)).astype(np.float32)


def bias(n):
    return (0.3 * rng.standard_normal(n)).astype(np.float32)


def copy_tokenizer(reader, writer):
    for key, field in reader.fields.items():
        if not key.startswith("tokenizer."):
            continue
        value = field.contents()
        if field.types[0].name == "ARRAY":
            writer.add_key_value(key, value, field.types[0], field.types[1])
        else:
            writer.add_key_value(key, value, field.types[0])


def write_model(path, arch, tokenizer_path, kv_heads, rope_dim, layer_tensors, ffn_length):
    reader = GGUFReader(tokenizer_path)
    vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    writer = GGUFWriter(path, arch)
    writer.add_name(f"tiny random {arch}")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(ffn_length)
    writer.add_head_count(HEADS)
    if kv_heads != HEADS:
        writer.add_head_count_kv(kv_heads)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(rope_dim)
    writer.add_rope_freq_base(10000.0)
    copy_tokenizer(reader, writer)
    tensors = {"token_embd.weight": weight(vocab, HIDDEN), "output_norm.weight": norm(HIDDEN), "output.weight": weight(vocab, HIDDEN)}
    for layer in range(LAYERS):
        for name, value in layer_tensors().items():
            tensors[f"blk.{layer}.{name}"] = value
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", path)


def qwen1_layer():
    # Multi-head attention with one fused, biased q/k/v tensor; the gate and up projections are half the stored feed-forward length.
    return {"attn_norm.weight": norm(HIDDEN), "attn_qkv.weight": weight(3 * HIDDEN, HIDDEN), "attn_qkv.bias": bias(3 * HIDDEN),
            "attn_output.weight": weight(HIDDEN, HIDDEN), "ffn_norm.weight": norm(HIDDEN), "ffn_gate.weight": weight(FFN, HIDDEN),
            "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}


def xverse_layer():
    kv = KV_HEADS * HEAD_DIM
    return {"attn_norm.weight": norm(HIDDEN), "attn_q.weight": weight(HIDDEN, HIDDEN), "attn_k.weight": weight(kv, HIDDEN),
            "attn_v.weight": weight(kv, HIDDEN), "attn_output.weight": weight(HIDDEN, HIDDEN), "ffn_norm.weight": norm(HIDDEN),
            "ffn_gate.weight": weight(FFN, HIDDEN), "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}


def chatglm_layer():
    qkv = HIDDEN + 2 * KV_HEADS * HEAD_DIM
    return {"attn_norm.weight": norm(HIDDEN), "attn_qkv.weight": weight(qkv, HIDDEN), "attn_qkv.bias": bias(qkv),
            "attn_output.weight": weight(HIDDEN, HIDDEN), "ffn_norm.weight": norm(HIDDEN), "ffn_up.weight": weight(2 * FFN, HIDDEN),
            "ffn_down.weight": weight(HIDDEN, FFN)}


def write_rwkv(path, arch, tokenizer_path, seed, hidden, head_size, parameters, layer_tensors):
    # Recurrent models: no rotary or head-count keys beyond llama.cpp's required (unused) head count of 0. Each model has its own seed so that
    # naming a subset of the models does not change the weights.
    global rng
    rng = np.random.default_rng(seed)
    reader = GGUFReader(tokenizer_path)
    vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    writer = GGUFWriter(path, arch)
    writer.add_name(f"tiny random {arch}")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(hidden)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(0)
    writer.add_wkv_head_size(head_size)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_token_shift_count(1)
    parameters(writer)
    copy_tokenizer(reader, writer)
    tensors = {"token_embd.weight": weight(vocab, hidden), "output_norm.weight": norm(hidden), "output.weight": weight(vocab, hidden)}
    for layer in range(LAYERS):
        for name, value in layer_tensors(layer, hidden).items():
            tensors[f"blk.{layer}.{name}"] = value
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", path)


# llama.cpp's CPU WKV7 kernel reads whole SIMD blocks of a head (32 or 64 floats): ARWKV7 needs the real head size of 64.
ARWKV_HIDDEN, ARWKV_HEAD_SIZE = 128, 64
DECAY_RANK, ICLR_RANK, VALUE_RANK, GATE_RANK, MIX_RANK = 16, 16, 8, 16, 8


def decay_bias(n):
    return (0.5 * rng.standard_normal(n) - 0.5).astype(np.float32)


def arwkv7_layer(layer, hidden, gated):
    # RWKV-7 time-mix (llama.cpp tensor layout: lora matrices stored [rank, C] / [C, rank], fused interpolation factors [6 or 5, 1, 1, C]),
    # RMS norms and a SwiGLU feed-forward in place of the channel-mix. The first layer's value residual tensors are unused.
    value_rank = ICLR_RANK if layer == 0 else VALUE_RANK
    t = {"attn_norm.weight": norm(hidden), "ffn_norm.weight": norm(hidden),
         "time_mix_w0.weight": decay_bias(hidden), "time_mix_w1.weight": weight(DECAY_RANK, hidden), "time_mix_w2.weight": weight(hidden, DECAY_RANK),
         "time_mix_a0.weight": bias(hidden), "time_mix_a1.weight": weight(ICLR_RANK, hidden), "time_mix_a2.weight": weight(hidden, ICLR_RANK),
         "time_mix_v0.weight": bias(hidden), "time_mix_v1.weight": weight(value_rank, hidden), "time_mix_v2.weight": weight(hidden, value_rank),
         "time_mix_lerp_fused.weight": rng.uniform(0.1, 0.9, (6 if gated else 5, 1, 1, hidden)).astype(np.float32),
         "time_mix_k_k.weight": norm(hidden), "time_mix_k_a.weight": norm(hidden), "time_mix_r_k.weight": weight(hidden),
         "time_mix_key.weight": weight(hidden, hidden), "time_mix_value.weight": weight(hidden, hidden),
         "time_mix_receptance.weight": weight(hidden, hidden), "time_mix_output.weight": weight(hidden, hidden),
         "ffn_gate.weight": weight(FFN, hidden), "ffn_up.weight": weight(FFN, hidden), "ffn_down.weight": weight(hidden, FFN)}
    if gated:
        t["time_mix_g1.weight"] = weight(GATE_RANK, hidden)
        t["time_mix_g2.weight"] = weight(hidden, GATE_RANK)
        t["time_mix_ln.weight"] = norm(hidden)
        t["time_mix_ln.bias"] = bias(hidden)
    return t


def arwkv7_parameters(gated):
    def write(writer):
        writer.add_decay_lora_rank(DECAY_RANK)
        writer.add_iclr_lora_rank(ICLR_RANK)
        writer.add_value_residual_mix_lora_rank(VALUE_RANK)
        writer.add_gate_lora_rank(GATE_RANK if gated else 0)
    return write


def rwkv6qwen2_parameters(writer):
    writer.add_time_mix_extra_dim(MIX_RANK)
    writer.add_time_decay_extra_dim(DECAY_RANK)
    writer.add_head_count_kv(KV_HEADS)


def rwkv6qwen2_layer(layer, hidden):
    # QRWKV6: grouped key/value heads (KV_HEADS of HEADS), biased r/k/v, sigmoid gate, no bonus, group norm or channel-mix; the five interpolation
    # offsets are stored [5·rank, C] (down) and [5, C, rank] (up), the fused factors [5, 1, 1, C] in the order w, k, v, r, g.
    kv = KV_HEADS * HEAD_DIM
    return {"attn_norm.weight": norm(hidden), "ffn_norm.weight": norm(hidden),
            "time_mix_w1.weight": weight(5 * MIX_RANK, hidden), "time_mix_w2.weight": weight(5, hidden, MIX_RANK),
            "time_mix_lerp_x.weight": rng.uniform(0.1, 0.9, hidden).astype(np.float32),
            "time_mix_lerp_fused.weight": rng.uniform(0.1, 0.9, (5, 1, 1, hidden)).astype(np.float32),
            "time_mix_decay.weight": decay_bias(hidden), "time_mix_decay_w1.weight": weight(DECAY_RANK, hidden),
            "time_mix_decay_w2.weight": weight(hidden, DECAY_RANK),
            "time_mix_key.weight": weight(kv, hidden), "time_mix_key.bias": bias(kv),
            "time_mix_value.weight": weight(kv, hidden), "time_mix_value.bias": bias(kv),
            "time_mix_receptance.weight": weight(hidden, hidden), "time_mix_receptance.bias": bias(hidden),
            "time_mix_gate.weight": weight(hidden, hidden), "time_mix_output.weight": weight(hidden, hidden),
            "ffn_gate.weight": weight(FFN, hidden), "ffn_up.weight": weight(FFN, hidden), "ffn_down.weight": weight(hidden, FFN)}


def write_plamo2(path, tokenizer_path, seed):
    # PLaMo-2: a Mamba layer (1) between two attention layers, every layer with sandwich norms and a fused gate+up feed-forward block. (The real models start
    # with a Mamba layer, which makes llama.cpp b11177 take them for attention-free: it leaves the rotary width at 0 and hangs in its rope kernel.)
    # llama.cpp loads a few of its norm / decay tensors without the ".weight" suffix.
    global rng
    rng = np.random.default_rng(seed)
    reader = GGUFReader(tokenizer_path)
    vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    heads = [HEADS, 0, HEADS]
    kv_heads = [KV_HEADS, 0, KV_HEADS]
    state, conv_kernel, dt_rank_dim = 16, 4, max(64, HIDDEN // 16)
    writer = GGUFWriter(path, "plamo2")
    writer.add_name("tiny random plamo2")
    writer.add_vocab_size(vocab)
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(LAYERS)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_ssm_state_size(state)
    writer.add_ssm_conv_kernel(conv_kernel)
    writer.add_ssm_time_step_rank(HEADS)
    writer.add_ssm_inner_size(HIDDEN)
    writer.add_ssm_group_count(0)
    copy_tokenizer(reader, writer)
    tensors = {"token_embd.weight": weight(vocab, HIDDEN), "output_norm.weight": norm(HIDDEN), "output.weight": weight(vocab, HIDDEN)}
    for layer in range(LAYERS):
        t = {"attn_norm.weight": norm(HIDDEN), "post_attention_norm": norm(HIDDEN), "ffn_norm.weight": norm(HIDDEN), "post_ffw_norm": norm(HIDDEN),
             "ffn_up.weight": weight(2 * FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}
        if heads[layer] == 0:
            t.update({"ssm_in.weight": weight(2 * HIDDEN, HIDDEN), "ssm_conv1d.weight": weight(HIDDEN, conv_kernel),
                      "ssm_x.weight": weight(dt_rank_dim + 2 * state, HIDDEN), "ssm_dt.weight": weight(HEADS, dt_rank_dim), "ssm_dt.bias": bias(HEADS),
                      "ssm_a": (-np.exp(0.5 * rng.standard_normal(HEADS))).astype(np.float32), "ssm_d": norm(HEADS),
                      "ssm_out.weight": weight(HIDDEN, HIDDEN), "ssm_dt_norm": norm(dt_rank_dim), "ssm_b_norm": norm(state), "ssm_c_norm": norm(state)})
        else:
            qkv = (heads[layer] + 2 * kv_heads[layer]) * HEAD_DIM
            t.update({"attn_qkv.weight": weight(qkv, HIDDEN), "attn_q_norm.weight": norm(heads[layer] * HEAD_DIM).reshape(heads[layer], HEAD_DIM),
                      "attn_k_norm.weight": norm(kv_heads[layer] * HEAD_DIM).reshape(kv_heads[layer], HEAD_DIM),
                      "attn_output.weight": weight(HIDDEN, heads[layer] * HEAD_DIM)})
        for name, value in t.items():
            tensors[f"blk.{layer}.{name}"] = value
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", path)


def write_family(path, arch, seed, layer_tensors, parameters=None, layers=LAYERS, kv_heads=KV_HEADS, layer_norm=False, global_tensors=None, rope_dim=HEAD_DIM,
                 heads=HEADS, ffn=FFN):
    # The llama-like families of the later bring-up rounds: one seed per model (independent of which models are written), the shared keys, an
    # RMS (or biased layer) output norm and an untied output head; layer_tensors(layer) returns the tensors of one block, global_tensors(vocab) any
    # replacement of the global ones (a None value drops a tensor) and parameters(writer) the family's own keys. heads, kv_heads and ffn may be
    # per-layer lists.
    global rng
    rng = np.random.default_rng(seed)
    reader = GGUFReader(tokenizer)
    vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    writer = GGUFWriter(path, arch)
    writer.add_name(f"tiny random {arch}")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    if layer_norm:
        writer.add_layer_norm_eps(1e-5)
    else:
        writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(rope_dim)
    writer.add_rope_freq_base(10000.0)
    if parameters:
        parameters(writer)
    copy_tokenizer(reader, writer)
    tensors = {"token_embd.weight": weight(vocab, HIDDEN), "output_norm.weight": norm(HIDDEN), "output.weight": weight(vocab, HIDDEN)}
    if layer_norm:
        tensors["output_norm.bias"] = bias(HIDDEN)
    for name, value in (global_tensors(vocab) if global_tensors else {}).items():
        if value is None:
            tensors.pop(name)
        else:
            tensors[name] = value
    for layer in range(layers):
        for name, value in layer_tensors(layer).items():
            tensors[f"blk.{layer}.{name}"] = value
    for name, value in tensors.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", path)


def llama_block(kv_heads=KV_HEADS, attn_bias=False, out_bias=False, layer_norm=False, gated=True, ffn_bias=False, fused_qkv=False, ffn_norm=True):
    # One pre-norm block with llama.cpp's tensor names; the variations cover the layer-norm / biased / fused / ungated families.
    kv = kv_heads * HEAD_DIM
    t = {"attn_norm.weight": norm(HIDDEN), "attn_output.weight": weight(HIDDEN, HIDDEN), "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}
    if fused_qkv:
        t["attn_qkv.weight"] = weight(HIDDEN + 2 * kv, HIDDEN)
    else:
        t.update({"attn_q.weight": weight(HIDDEN, HIDDEN), "attn_k.weight": weight(kv, HIDDEN), "attn_v.weight": weight(kv, HIDDEN)})
    if attn_bias:
        if fused_qkv:
            t["attn_qkv.bias"] = bias(HIDDEN + 2 * kv)
        else:
            t.update({"attn_q.bias": bias(HIDDEN), "attn_k.bias": bias(kv), "attn_v.bias": bias(kv)})
    if out_bias:
        t["attn_output.bias"] = bias(HIDDEN)
    if ffn_norm:
        t["ffn_norm.weight"] = norm(HIDDEN)
    if layer_norm:
        t["attn_norm.bias"] = bias(HIDDEN)
        if ffn_norm:
            t["ffn_norm.bias"] = bias(HIDDEN)
    if gated:
        t["ffn_gate.weight"] = weight(FFN, HIDDEN)
    if ffn_bias:
        t.update({"ffn_up.bias": bias(FFN), "ffn_down.bias": bias(HIDDEN)})
        if gated:
            t["ffn_gate.bias"] = bias(FFN)
    return t


def plamo3_block():
    # llama.cpp loads the two post norms without the ".weight" suffix.
    qkv = HIDDEN + 2 * KV_HEADS * HEAD_DIM
    return {"attn_norm.weight": norm(HIDDEN), "attn_qkv.weight": weight(qkv, HIDDEN), "attn_q_norm.weight": norm(HEAD_DIM), "attn_k_norm.weight": norm(HEAD_DIM),
            "attn_output.weight": weight(HIDDEN, HIDDEN), "post_attention_norm": norm(HIDDEN), "ffn_norm.weight": norm(HIDDEN),
            "post_ffw_norm": norm(HIDDEN), "ffn_up.weight": weight(2 * FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}


def with_tensors(block, **extra):
    t = block()
    for name, make in extra.items():
        t[name.replace("__", ".")] = make()
    return t


def plamo3_parameters(writer):
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern(2)
    writer.add_rope_freq_base_swa(1000.0)


def spark_parameters(writer):
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern([True, False, True])
    writer.add_rope_freq_base_swa(1000.0)
    writer.add_rope_dimension_count_swa(HEAD_DIM)


EXPERTS, EXPERTS_USED, EXPERT_FFN = 4, 2, 48


def moe_block(head_dim=HEAD_DIM, shared=1):
    # A Llama attention block (head size head_dim) with routed SwiGLU experts and, when shared > 0, a shared SwiGLU expert of shared * EXPERT_FFN.
    q, kv = HEADS * head_dim, KV_HEADS * head_dim
    t = {"attn_norm.weight": norm(HIDDEN), "attn_q.weight": weight(q, HIDDEN), "attn_k.weight": weight(kv, HIDDEN), "attn_v.weight": weight(kv, HIDDEN),
         "attn_output.weight": weight(HIDDEN, q), "ffn_norm.weight": norm(HIDDEN), "ffn_gate_inp.weight": weight(EXPERTS, HIDDEN),
         "ffn_gate_exps.weight": weight(EXPERTS, EXPERT_FFN, HIDDEN), "ffn_up_exps.weight": weight(EXPERTS, EXPERT_FFN, HIDDEN),
         "ffn_down_exps.weight": weight(EXPERTS, HIDDEN, EXPERT_FFN)}
    if shared:
        t.update({"ffn_gate_shexp.weight": weight(shared * EXPERT_FFN, HIDDEN), "ffn_up_shexp.weight": weight(shared * EXPERT_FFN, HIDDEN),
                  "ffn_down_shexp.weight": weight(HIDDEN, shared * EXPERT_FFN)})
    return t


def moe_parameters(shared=1, dense_lead=0, scale=None, norm_weights=None):
    def add(writer):
        writer.add_expert_count(EXPERTS)
        writer.add_expert_used_count(EXPERTS_USED)
        writer.add_expert_feed_forward_length(EXPERT_FFN)
        writer.add_expert_shared_count(shared)
        if dense_lead:
            writer.add_leading_dense_block_count(dense_lead)
        if scale is not None:
            writer.add_expert_weights_scale(scale)
        if norm_weights is not None:
            writer.add_expert_weights_norm(norm_weights)
    return add


def bailing_parameters(writer):
    moe_parameters(norm_weights=True)(writer)
    writer.add_key_length(24)
    writer.add_value_length(24)


LATENT, NOPE, ROPE, VALUE = 32, 16, 8, 16


def plm_block():
    # Multi-head latent attention with a direct query projection and the combined key/value up-projection, ungated squared-ReLU MLP.
    return {"attn_norm.weight": norm(HIDDEN), "attn_q.weight": weight(HEADS * (NOPE + ROPE), HIDDEN), "attn_kv_a_mqa.weight": weight(LATENT + ROPE, HIDDEN),
            "attn_kv_a_norm.weight": norm(LATENT), "attn_kv_b.weight": weight(HEADS * (NOPE + VALUE), LATENT), "attn_output.weight": weight(HIDDEN, HEADS * VALUE),
            "ffn_norm.weight": norm(HIDDEN), "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}


def plm_parameters(writer):
    writer.add_kv_lora_rank(LATENT)
    writer.add_key_length(NOPE + ROPE)
    writer.add_value_length(VALUE)


def qk_norm_block(block=None, **extra):
    # A block with per-head q/k norms (and any extra tensors, "__" standing for ".").
    t = (block or llama_block)()
    t.update({"attn_q_norm.weight": norm(HEAD_DIM), "attn_k_norm.weight": norm(HEAD_DIM)})
    for name, make in extra.items():
        t[name.replace("__", ".")] = make()
    return t


def talkie_block():
    # No norm weights at all; a per-head scalar q gain ([heads, 1]) and the per-layer embedding skip scale.
    t = {name: value for name, value in llama_block().items() if not name.endswith("norm.weight")}
    t.update({"attn_q_norm.weight": norm(HEADS).reshape(HEADS, 1), "layer_output_scale.weight": (0.5 * rng.standard_normal(1)).astype(np.float32)})
    return t


def arctic_block():
    # A dense SwiGLU MLP of hidden width next to softmax-routed experts of FFN width behind their own norm.
    t = {name: value for name, value in moe_block(shared=0).items() if "exps" not in name}
    t.update({"ffn_gate.weight": weight(HIDDEN, HIDDEN), "ffn_up.weight": weight(HIDDEN, HIDDEN), "ffn_down.weight": weight(HIDDEN, HIDDEN),
              "ffn_norm_exps.weight": norm(HIDDEN), "ffn_gate_exps.weight": weight(EXPERTS, FFN, HIDDEN), "ffn_up_exps.weight": weight(EXPERTS, FFN, HIDDEN),
              "ffn_down_exps.weight": weight(EXPERTS, HIDDEN, FFN)})
    return t


DECI_HEADS, DECI_KV, DECI_FFN = [4, 4, 0, 4], [2, 0, 0, 4], [128, 96, 64, 128]


def deci_block(layer):
    # Layer 1 replaces attention by one linear map, layer 2 has no attention; the feed-forward width varies.
    heads, kv, ffn = DECI_HEADS[layer], DECI_KV[layer], DECI_FFN[layer]
    t = {"ffn_norm.weight": norm(HIDDEN), "ffn_gate.weight": weight(ffn, HIDDEN), "ffn_up.weight": weight(ffn, HIDDEN), "ffn_down.weight": weight(HIDDEN, ffn)}
    if heads:
        t.update({"attn_norm.weight": norm(HIDDEN), "attn_output.weight": weight(HIDDEN, HIDDEN)})
    if kv:
        t.update({"attn_q.weight": weight(HIDDEN, HIDDEN), "attn_k.weight": weight(kv * HEAD_DIM, HIDDEN), "attn_v.weight": weight(kv * HEAD_DIM, HIDDEN)})
    return t


def grok_block():
    # GELU experts plus a dense GELU MLP, sandwich norms around both sub-layers.
    t = moe_block(shared=0)
    t.update({"ffn_gate.weight": weight(FFN, HIDDEN), "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN),
              "attn_output_norm.weight": norm(HIDDEN), "layer_output_norm.weight": norm(HIDDEN)})
    return t


def grok_parameters(writer):
    moe_parameters(shared=0)(writer)
    writer.add_float32("grok.attention.output_scale", 0.2)
    writer.add_embedding_scale(2.0)
    writer.add_logit_scale(0.6)
    writer.add_final_logit_softcapping(20.0)


def sliding_parameters(writer, base_swa=1000.0):
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern([True, False, True])
    if base_swa:
        writer.add_rope_freq_base_swa(base_swa)


def maple_parameters(writer):
    moe_parameters(shared=0)(writer)
    sliding_parameters(writer)
    writer.add_swiglu_clamp_exp([0.8, 0.0, 1.2])


STEP_HEADS, STEP_KV = [4, 4, 2], [2, 2, 1]


def step35_block(layer):
    # A dense first layer, then experts with a selection bias and a shared expert; per-layer head counts and a per-head output gate.
    heads, kv = STEP_HEADS[layer], STEP_KV[layer]
    t = llama_block() if layer == 0 else moe_block()
    t.update({"attn_q.weight": weight(heads * HEAD_DIM, HIDDEN), "attn_k.weight": weight(kv * HEAD_DIM, HIDDEN), "attn_v.weight": weight(kv * HEAD_DIM, HIDDEN),
              "attn_output.weight": weight(HIDDEN, heads * HEAD_DIM), "attn_q_norm.weight": norm(HEAD_DIM), "attn_k_norm.weight": norm(HEAD_DIM),
              "attn_gate.weight": weight(heads, HIDDEN)})
    if layer:
        t["exp_probs_b.bias"] = bias(EXPERTS)
    return t


def step35_parameters(writer):
    moe_parameters(dense_lead=1)(writer)
    writer.add_expert_shared_feed_forward_length(EXPERT_FFN)
    writer.add_key_length(HEAD_DIM)
    writer.add_value_length(HEAD_DIM)
    sliding_parameters(writer)
    writer.add_rope_dimension_count_swa(HEAD_DIM)
    writer.add_swiglu_clamp_exp([0.0, 1.0, 0.9])
    writer.add_swiglu_clamp_shexp([1.1, 0.7, 0.0])


def kolibri1_parameters(writer):
    moe_parameters(norm_weights=True)(writer)
    writer.add_expert_shared_feed_forward_length(EXPERT_FFN)
    writer.add_uint32("kolibri1.expert_gating_func", 2)
    sliding_parameters(writer, base_swa=None)


def ricursive_parameters(writer):
    writer.add_sliding_window(8)
    writer.add_embedding_scale(2.0)
    writer.add_swiglu_clamp([0.8, 0.0, 1.0, 0.9, 0.7])


LFM3_KINDS, LFM3_HEAD = ["conv", "sliding", "conv", "full"], 24


def lfm3_block(layer):
    # Gated short convolution, or attention with q/k norms, a head size other than hidden / heads and a per-head sigmoid gate; sandwich norms.
    t = {"attn_norm.weight": norm(HIDDEN), "ffn_norm.weight": norm(HIDDEN), "post_attention_norm.weight": norm(HIDDEN), "post_ffw_norm.weight": norm(HIDDEN),
         "ffn_gate.weight": weight(FFN, HIDDEN), "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}
    if LFM3_KINDS[layer] == "conv":
        t.update({"shortconv.conv.weight": weight(HIDDEN, 3), "shortconv.in_proj.weight": weight(3 * HIDDEN, HIDDEN),
                  "shortconv.out_proj.weight": weight(HIDDEN, HIDDEN)})
    else:
        q, kv = HEADS * LFM3_HEAD, KV_HEADS * LFM3_HEAD
        t.update({"attn_q.weight": weight(q, HIDDEN), "attn_k.weight": weight(kv, HIDDEN), "attn_v.weight": weight(kv, HIDDEN), "attn_output.weight": weight(HIDDEN, q),
                  "attn_q_norm.weight": norm(LFM3_HEAD), "attn_k_norm.weight": norm(LFM3_HEAD), "attn_gate.weight": weight(HEADS, HIDDEN)})
    return t


def lfm3_parameters(writer):
    writer.add_shortconv_l_cache(3)
    writer.add_key_length(LFM3_HEAD)
    writer.add_value_length(LFM3_HEAD)
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern([kind == "sliding" for kind in LFM3_KINDS])
    writer.add_rope_freq_base_swa(1000.0)


def lfm3_globals(vocab):
    # The converter expresses the partial rotation (factor 0.5) of the full-attention layers as frequency factors that stop the upper pairs.
    pairs = LFM3_HEAD // 2
    return {"rope_freqs.weight": np.array([1.0] * (pairs // 2) + [1e30] * (pairs - pairs // 2), dtype=np.float32)}


LATENT = 32


def maion_coder_block(layer):
    # q/k norms and sandwich norms; every second layer routes its experts in a latent space of LATENT channels.
    t = qk_norm_block(post_attention_norm__weight=lambda: norm(HIDDEN), post_ffw_norm__weight=lambda: norm(HIDDEN))
    if layer % 2 == 1:
        for name in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            t.pop(name)
        t.update({"ffn_gate_inp.weight": weight(EXPERTS, HIDDEN), "ffn_latent_down.weight": weight(LATENT, HIDDEN), "ffn_latent_up.weight": weight(HIDDEN, LATENT),
                  "ffn_gate_exps.weight": weight(EXPERTS, EXPERT_FFN, LATENT), "ffn_up_exps.weight": weight(EXPERTS, EXPERT_FFN, LATENT),
                  "ffn_down_exps.weight": weight(EXPERTS, LATENT, EXPERT_FFN)})
    return t


def maion_coder_parameters(writer):
    moe_parameters(shared=0)(writer)
    writer.add_interleave_moe_layer_step(2)
    writer.add_moe_latent_size(LATENT)
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern([True, True, True, False])


def beam_block(layer):
    # A dense first layer, then experts with a selection bias; q/k norms, a per-channel gate with one bias per head, sandwich norms.
    t = qk_norm_block(llama_block if layer == 0 else lambda: moe_block(shared=0), attn_gate__weight=lambda: weight(HIDDEN, HIDDEN),
                      attn_gate__bias=lambda: bias(HEADS), post_attention_norm__weight=lambda: norm(HIDDEN), post_ffw_norm__weight=lambda: norm(HIDDEN))
    if layer:
        t["exp_probs_b.bias"] = bias(EXPERTS)
    return t


def beam_parameters(writer):
    moe_parameters(shared=0, dense_lead=1, scale=2.5, norm_weights=True)(writer)
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern([True, True, False])
    writer.add_rope_freq_base_swa(1000.0)
    writer.add_float32("beam.attention.temperature_scale", 0.5)
    writer.add_uint32("beam.attention.temperature_length", 4)
    writer.add_final_logit_softcapping(20.0)


def bidirectional(extra=None):
    # Embedding models: llama.cpp reads attention.causal = false and runs them without a KV cache.
    def parameters(writer):
        writer.add_causal_attention(False)
        if extra:
            extra(writer)
    return parameters


def neo_bert_block():
    # Fused q/k/v, fused SwiGLU up projection (gate rows first), RMS norms.
    return {"attn_norm.weight": norm(HIDDEN), "attn_qkv.weight": weight(3 * HIDDEN, HIDDEN), "attn_output.weight": weight(HIDDEN, HIDDEN),
            "ffn_norm.weight": norm(HIDDEN), "ffn_up.weight": weight(2 * FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}


def gemma_embedding_block():
    return {"attn_norm.weight": norm(HIDDEN), "attn_q.weight": weight(HIDDEN, HIDDEN), "attn_k.weight": weight(HEAD_DIM, HIDDEN),
            "attn_v.weight": weight(HEAD_DIM, HIDDEN), "attn_q_norm.weight": norm(HEAD_DIM), "attn_k_norm.weight": norm(HEAD_DIM),
            "attn_output.weight": weight(HIDDEN, HIDDEN), "post_attention_norm.weight": norm(HIDDEN), "ffn_norm.weight": norm(HIDDEN),
            "ffn_gate.weight": weight(FFN, HIDDEN), "ffn_up.weight": weight(FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN),
            "post_ffw_norm.weight": norm(HIDDEN)}


DENSE = 96


def gemma_embedding_parameters(writer):
    # Window 8 is symmetric (keys at most 4 positions away); layers 0-4 slide, layer 5 is global; Dense 64 -> 96 -> 64 after pooling.
    writer.add_sliding_window(8)
    writer.add_rope_freq_base_swa(10000.0)
    writer.add_uint32("gemma-embedding.dense_2_feat_in", HIDDEN)
    writer.add_uint32("gemma-embedding.dense_2_feat_out", DENSE)
    writer.add_uint32("gemma-embedding.dense_3_feat_in", DENSE)
    writer.add_uint32("gemma-embedding.dense_3_feat_out", HIDDEN)


def modern_bert_block(layer):
    # Bias-free layer norms, no norm in front of the first attention, fused q/k/v and GeGLU up projection (activated half first).
    t = {"attn_qkv.weight": weight(3 * HIDDEN, HIDDEN), "attn_output.weight": weight(HIDDEN, HIDDEN), "ffn_norm.weight": norm(HIDDEN),
         "ffn_up.weight": weight(2 * FFN, HIDDEN), "ffn_down.weight": weight(HIDDEN, FFN)}
    if layer != 0:
        t["attn_norm.weight"] = norm(HIDDEN)
    return t


def modern_bert_parameters(writer):
    # Local window 8, every third layer global from the first.
    writer.add_sliding_window(8)
    writer.add_sliding_window_pattern(3)
    writer.add_rope_freq_base_swa(10000.0)
    writer.add_string("modern-bert.hidden_act", "gelu")


def jina_bert_v2_block():
    # Post-norm BERT layer with whole-projection q/k layer norms, a second attention norm and a fused GeGLU up projection (activated half
    # first); every linear has a bias except the q/k/v ones of real checkpoints, which have them too.
    t = {}
    for name, rows in [("attn_q", HIDDEN), ("attn_k", HIDDEN), ("attn_v", HIDDEN), ("attn_output", HIDDEN)]:
        t[f"{name}.weight"] = weight(rows, HIDDEN)
        t[f"{name}.bias"] = bias(rows)
    for name in ["attn_q_norm", "attn_k_norm", "attn_output_norm", "attn_norm_2", "layer_output_norm"]:
        t[f"{name}.weight"] = norm(HIDDEN)
        t[f"{name}.bias"] = bias(HIDDEN)
    t.update({"ffn_up.weight": weight(2 * FFN, HIDDEN), "ffn_up.bias": bias(2 * FFN), "ffn_down.weight": weight(HIDDEN, FFN), "ffn_down.bias": bias(HIDDEN)})
    return t


def jina_bert_v2_parameters(writer):
    writer.add_token_type_count(2)


tokenizer, out = sys.argv[1], os.path.expanduser(sys.argv[2])
only = sys.argv[3:]
os.makedirs(out, exist_ok=True)
models = {
    # Qwen v1 stores twice the gate/up length as the feed-forward length; the rotary dimension is the head size (neox pairing).
    "qwen1": lambda path: write_model(path, "qwen", tokenizer, HEADS, HEAD_DIM, qwen1_layer, 2 * FFN),
    "xverse": lambda path: write_model(path, "xverse", tokenizer, KV_HEADS, HEAD_DIM, xverse_layer, FFN),
    # ChatGLM rotates the first half of every head on interleaved pairs.
    "chatglm": lambda path: write_model(path, "chatglm", tokenizer, KV_HEADS, HEAD_DIM // 2, chatglm_layer, FFN),
    "arwkv7": lambda path: write_rwkv(path, "arwkv7", tokenizer, 1, ARWKV_HIDDEN, ARWKV_HEAD_SIZE, arwkv7_parameters(True), lambda layer, hidden: arwkv7_layer(layer, hidden, True)),
    "arwkv7-nogate": lambda path: write_rwkv(path, "arwkv7", tokenizer, 2, ARWKV_HIDDEN, ARWKV_HEAD_SIZE, arwkv7_parameters(False), lambda layer, hidden: arwkv7_layer(layer, hidden, False)),
    "plamo2": lambda path: write_plamo2(path, tokenizer, 4),
    "rwkv6qwen2": lambda path: write_rwkv(path, "rwkv6qwen2", tokenizer, 3, HIDDEN, HEAD_DIM, rwkv6qwen2_parameters, rwkv6qwen2_layer),
    # Later bring-up rounds (write_family, one seed each).
    "refact": lambda path: write_family(path, "refact", 10, lambda layer: llama_block(kv_heads=1), kv_heads=1),
    "jais": lambda path: write_family(path, "jais", 11, lambda layer: llama_block(attn_bias=True, out_bias=True, layer_norm=True, ffn_bias=True, fused_qkv=True),
                                      parameters=lambda w: w.add_max_alibi_bias(8.0), layer_norm=True),
    "orion": lambda path: write_family(path, "orion", 12, lambda layer: llama_block(layer_norm=True), layer_norm=True),
    # CodeShell without token_embd: llama.cpp embeds through the output projection.
    "codeshell": lambda path: write_family(path, "codeshell", 13, lambda layer: llama_block(attn_bias=True, out_bias=True, layer_norm=True, gated=False, ffn_bias=True, fused_qkv=True),
                                          layer_norm=True, global_tensors=lambda vocab: {"token_embd.weight": None}),
    "plamo": lambda path: write_family(path, "plamo", 14, lambda layer: llama_block(ffn_norm=False)),
    "plamo3": lambda path: write_family(path, "plamo3", 15, lambda layer: plamo3_block(), parameters=plamo3_parameters),
    "pangu-embedded": lambda path: write_family(path, "pangu-embedded", 16, lambda layer: llama_block(attn_bias=True, out_bias=True)),
    "maincoder": lambda path: write_family(path, "maincoder", 17, lambda layer: with_tensors(llama_block, attn_q_norm__weight=lambda: norm(HEAD_DIM), attn_k_norm__weight=lambda: norm(HEAD_DIM))),
    "spark2_5": lambda path: write_family(path, "spark2_5", 18, lambda layer: with_tensors(llama_block, attn_gate__weight=lambda: weight(HEADS, HIDDEN)), parameters=spark_parameters,
                                          rope_dim=8),
    # Baichuan's position scheme depends on the depth: 32 layers rotary (7B), 40 layers ALiBi (13B).
    "baichuan": lambda path: write_family(path, "baichuan", 19, lambda layer: llama_block(), layers=32),
    "baichuan-alibi": lambda path: write_family(path, "baichuan", 20, lambda layer: llama_block(), layers=40),
    # DeepSeek V1: a leading dense layer, then experts (top-k weights not renormalized, scaled) plus a shared expert of two expert widths.
    "deepseek": lambda path: write_family(path, "deepseek", 23, lambda layer: llama_block() if layer == 0 else moe_block(shared=2),
                                          parameters=moe_parameters(shared=2, dense_lead=1, scale=1.5)),
    # Bailing MoE V1: a head size (rope.dimension_count) other than hidden / heads, renormalized top-k weights.
    "bailingmoe": lambda path: write_family(path, "bailingmoe", 22, lambda layer: moe_block(head_dim=24), parameters=bailing_parameters,
                                            rope_dim=24),
    # PLM: latent attention without query compression, squared-ReLU MLP, output head tied to the embedding.
    "plm": lambda path: write_family(path, "plm", 24, lambda layer: plm_block(), parameters=plm_parameters, rope_dim=ROPE, kv_heads=HEADS,
                                     global_tensors=lambda vocab: {"output.weight": None}),
    # Talkie: weight-less norms, a per-head q gain after rope, the embedding skip and a logit scale.
    "talkie": lambda path: write_family(path, "talkie", 25, lambda layer: talkie_block(), parameters=lambda w: w.add_logit_scale(0.7),
                                        global_tensors=lambda vocab: {"output_norm.weight": None}),
    # Nanbeige: the three layers run twice with the output norm in between.
    "nanbeige": lambda path: write_family(path, "nanbeige", 26, lambda layer: llama_block(), parameters=lambda w: w.add_uint32("nanbeige.num_loops", 2)),
    "arctic": lambda path: write_family(path, "arctic", 27, lambda layer: arctic_block(), parameters=moe_parameters(shared=0)),
    "deci": lambda path: write_family(path, "deci", 28, deci_block, layers=4, heads=DECI_HEADS, kv_heads=DECI_KV, ffn=DECI_FFN),
    "grok": lambda path: write_family(path, "grok", 29, lambda layer: grok_block(), parameters=grok_parameters),
    "maple": lambda path: write_family(path, "maple", 30, lambda layer: qk_norm_block(lambda: moe_block(shared=0)), parameters=maple_parameters),
    "step35": lambda path: write_family(path, "step35", 31, step35_block, parameters=step35_parameters, heads=STEP_HEADS, kv_heads=STEP_KV),
    "kolibri1": lambda path: write_family(path, "kolibri1", 32, lambda layer: qk_norm_block(moe_block, exp_probs_b__bias=lambda: bias(EXPERTS),
                                          post_attention_norm__weight=lambda: norm(HIDDEN), post_ffw_norm__weight=lambda: norm(HIDDEN)), parameters=kolibri1_parameters),
    # Ricursive: five layers with the default pattern (layer 4 global and NoPE), value residual, per-channel gate, sandwich norms.
    "ricursive": lambda path: write_family(path, "ricursive", 33, lambda layer: qk_norm_block(llama_block, attn_gate__weight=lambda: weight(HIDDEN, HIDDEN),
                                           attn_vres_lambda__weight=lambda: bias(1), post_attention_norm__weight=lambda: norm(HIDDEN),
                                           post_ffw_norm__weight=lambda: norm(HIDDEN)), parameters=ricursive_parameters, layers=5),
    # LFM3 (llama.cpp after b11177): conv / sliding attention / conv / full attention, see lfm3_block.
    "lfm3": lambda path: write_family(path, "lfm3", 34, lfm3_block, parameters=lfm3_parameters, layers=len(LFM3_KINDS),
                                      kv_heads=[0 if kind == "conv" else KV_HEADS for kind in LFM3_KINDS], rope_dim=LFM3_HEAD, global_tensors=lfm3_globals),
    "maion-coder": lambda path: write_family(path, "maion-coder", 35, maion_coder_block, parameters=maion_coder_parameters, layers=4),
    "beam": lambda path: write_family(path, "beam", 36, beam_block, parameters=beam_parameters),
    # Bidirectional embedding models (references from llama-server's /embedding, tools/llamacpp_embeddings.py).
    "eurobert": lambda path: write_family(path, "eurobert", 40, lambda layer: llama_block(), parameters=bidirectional(),
                                          global_tensors=lambda vocab: {"output.weight": None}),
    "neo-bert": lambda path: write_family(path, "neo-bert", 41, lambda layer: neo_bert_block(), parameters=bidirectional(), kv_heads=HEADS,
                                          global_tensors=lambda vocab: {"output.weight": None, "output_norm.weight": None, "enc.output_norm.weight": norm(HIDDEN)}),
    "llama-embed": lambda path: write_family(path, "llama-embed", 42, lambda layer: llama_block(), parameters=bidirectional(),
                                             global_tensors=lambda vocab: {"output.weight": None}),
    "gemma-embedding": lambda path: write_family(path, "gemma-embedding", 43, lambda layer: gemma_embedding_block(), layers=6, kv_heads=1,
                                                 parameters=bidirectional(gemma_embedding_parameters),
                                                 global_tensors=lambda vocab: {"output.weight": None, "dense_2.weight": weight(DENSE, HIDDEN),
                                                                               "dense_3.weight": weight(HIDDEN, DENSE)}),
    "modern-bert": lambda path: write_family(path, "modern-bert", 44, modern_bert_block, layers=4, kv_heads=HEADS, layer_norm=True,
                                             parameters=bidirectional(modern_bert_parameters),
                                             global_tensors=lambda vocab: {"output.weight": None, "output_norm.bias": None,
                                                                           "token_embd_norm.weight": norm(HIDDEN)}),
    # JinaBERT v2: no positions (ALiBi, llama.cpp's fixed maximum bias 8), two token types of which row 0 is added.
    "jina-bert-v2": lambda path: write_family(path, "jina-bert-v2", 45, lambda layer: jina_bert_v2_block(), kv_heads=HEADS, layer_norm=True,
                                              parameters=bidirectional(jina_bert_v2_parameters),
                                              global_tensors=lambda vocab: {"output.weight": None, "output_norm.weight": None, "output_norm.bias": None,
                                                                            "token_types.weight": weight(2, HIDDEN), "token_embd_norm.weight": norm(HIDDEN),
                                                                            "token_embd_norm.bias": bias(HIDDEN)}),
}
for name, write in models.items():
    if not only or name in only:
        write(os.path.join(out, f"tiny-random-{name}-f32.gguf"))
