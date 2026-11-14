"""
Writes tiny random GGUF files for llama.cpp architectures that have no Hugging Face class in transformers (so no tiny-random checkpoint
and no converter path): Qwen v1 (qwen), XVERSE (xverse) and ChatGLM (chatglm), the RMS-normed RWKV distillations ARWKV7 (arwkv7, with and without output gate) and QRWKV6 (rwkv6qwen2), and PLaMo-2 (plamo2), plus the later llama.cpp-only families (write_family). The weights are random, written with llama.cpp's tensor names
and hyper-parameter keys; llama.cpp (llama-completion) provides the reference continuation. The tokenizer fields are copied from a GGUF
that already carries the 1024-token GPT-2-style vocabulary of tiny-random-GPTNeoXForCausalLM, e.g. tiny-random-falcon-f32.gguf converted
from ~/models/hf/tiny-random-falcon with convert_hf_to_gguf.py.

    PYTHONPATH=<llama.cpp>/gguf-py python mk_tiny_gguf.py <tokenizer gguf> <output directory> [name ...]

Files: <output directory>/tiny-random-{qwen1,xverse,chatglm,arwkv7,arwkv7-nogate,rwkv6qwen2,plamo2}-f32.gguf (only the named ones when names are given). Norm weights
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


def write_family(path, arch, seed, layer_tensors, parameters=None, layers=LAYERS, kv_heads=KV_HEADS, layer_norm=False, global_tensors=None, rope_dim=HEAD_DIM):
    # The llama-like families of the later bring-up rounds: one seed per model (independent of which models are written), the shared keys, an
    # RMS (or biased layer) output norm and an untied output head; layer_tensors(layer) returns the tensors of one block, global_tensors(vocab) any
    # replacement of the global ones (a None value drops a tensor) and parameters(writer) the family's own keys.
    global rng
    rng = np.random.default_rng(seed)
    reader = GGUFReader(tokenizer)
    vocab = len(reader.fields["tokenizer.ggml.tokens"].contents())
    writer = GGUFWriter(path, arch)
    writer.add_name(f"tiny random {arch}")
    writer.add_context_length(CONTEXT)
    writer.add_embedding_length(HIDDEN)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(FFN)
    writer.add_head_count(HEADS)
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
}
for name, write in models.items():
    if not only or name in only:
        write(os.path.join(out, f"tiny-random-{name}-f32.gguf"))
