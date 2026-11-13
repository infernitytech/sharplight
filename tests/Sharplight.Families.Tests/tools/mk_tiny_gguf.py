"""
Writes tiny random GGUF files for llama.cpp architectures that have no Hugging Face class in transformers (so no tiny-random checkpoint
and no converter path): Qwen v1 (qwen), XVERSE (xverse) and ChatGLM (chatglm). The weights are random, written with llama.cpp's tensor names
and hyper-parameter keys; llama.cpp (llama-completion) provides the reference continuation. The tokenizer fields are copied from a GGUF
that already carries the 1024-token GPT-2-style vocabulary of tiny-random-GPTNeoXForCausalLM, e.g. tiny-random-falcon-f32.gguf converted
from ~/models/hf/tiny-random-falcon with convert_hf_to_gguf.py.

    PYTHONPATH=<llama.cpp>/gguf-py python mk_tiny_gguf.py <tokenizer gguf> <output directory>

Files: <output directory>/tiny-random-{qwen1,xverse,chatglm}-f32.gguf. Norm weights and biases are randomized so that every tensor matters.
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


tokenizer, out = sys.argv[1], os.path.expanduser(sys.argv[2])
os.makedirs(out, exist_ok=True)
# Qwen v1 stores twice the gate/up length as the feed-forward length; the rotary dimension is the head size (neox pairing).
write_model(os.path.join(out, "tiny-random-qwen1-f32.gguf"), "qwen", tokenizer, HEADS, HEAD_DIM, qwen1_layer, 2 * FFN)
write_model(os.path.join(out, "tiny-random-xverse-f32.gguf"), "xverse", tokenizer, KV_HEADS, HEAD_DIM, xverse_layer, FFN)
# ChatGLM rotates the first half of every head on interleaved pairs.
write_model(os.path.join(out, "tiny-random-chatglm-f32.gguf"), "chatglm", tokenizer, KV_HEADS, HEAD_DIM // 2, chatglm_layer, FFN)
