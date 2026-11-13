"""
Writes tiny random GGUF files for llama.cpp architectures that have no Hugging Face class in transformers (so no tiny-random checkpoint
and no converter path): Qwen v1 (qwen), XVERSE (xverse) and ChatGLM (chatglm), and the RMS-normed RWKV distillation ARWKV7 (arwkv7, with and without output gate). The weights are random, written with llama.cpp's tensor names
and hyper-parameter keys; llama.cpp (llama-completion) provides the reference continuation. The tokenizer fields are copied from a GGUF
that already carries the 1024-token GPT-2-style vocabulary of tiny-random-GPTNeoXForCausalLM, e.g. tiny-random-falcon-f32.gguf converted
from ~/models/hf/tiny-random-falcon with convert_hf_to_gguf.py.

    PYTHONPATH=<llama.cpp>/gguf-py python mk_tiny_gguf.py <tokenizer gguf> <output directory> [name ...]

Files: <output directory>/tiny-random-{qwen1,xverse,chatglm,arwkv7,arwkv7-nogate}-f32.gguf (only the named ones when names are given). Norm weights
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
}
for name, write in models.items():
    if not only or name in only:
        write(os.path.join(out, f"tiny-random-{name}-f32.gguf"))
