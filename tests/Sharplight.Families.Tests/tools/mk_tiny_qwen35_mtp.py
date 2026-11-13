#!/usr/bin/env python3
"""Writes a tiny random Qwen3.5 text model with one MTP layer (Hugging Face layout, f32 safetensors) for the speculative-decoding
tests: 4 layers (3 Gated DeltaNet, 1 gated attention), hidden 64, 2 key / 4 value linear-attention heads (tiled value heads),
plus the MTP block (mtp.fc, the pre-fc norms, one attention layer, mtp.norm). The tokenizer and chat template are copied from a
Qwen3.5 checkpoint (default ~/models/hf/Qwen3.5-0.8B), so the vocabulary is the real one.

Usage: mk_tiny_qwen35_mtp.py [OUT_DIR] [TOKENIZER_DIR]   (default OUT_DIR ~/models/hf/tiny-random-qwen3.5-mtp)
Needs only numpy.
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

HOME = os.path.expanduser("~")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, "models/hf/tiny-random-qwen3.5-mtp")
TOK = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HOME, "models/hf/Qwen3.5-0.8B")

HIDDEN, LAYERS, INTER = 64, 4, 128
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32
LIN_K_HEADS, LIN_V_HEADS, LIN_K_DIM, LIN_V_DIM, CONV = 2, 4, 16, 16, 4
VOCAB = 248320

config = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "attention_bias": False,
    "attn_output_gate": True,
    "bos_token_id": 248044,
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "head_dim": HEAD_DIM,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "intermediate_size": INTER,
    "layer_types": ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(LAYERS)],
    "linear_conv_kernel_dim": CONV,
    "linear_key_head_dim": LIN_K_DIM,
    "linear_num_key_heads": LIN_K_HEADS,
    "linear_num_value_heads": LIN_V_HEADS,
    "linear_value_head_dim": LIN_V_DIM,
    "max_position_embeddings": 4096,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": KV_HEADS,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "vocab_size": VOCAB,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default",
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25},
}

rng = np.random.default_rng(1234)
tensors = {}


def lin(name, out_features, in_features, scale=0.08):
    tensors[name] = (rng.standard_normal((out_features, in_features)) * scale).astype(np.float32)


def norm(name, size):
    # Qwen3.5 RMS norms store w with an implicit +1.
    tensors[name] = (rng.standard_normal(size) * 0.1).astype(np.float32)


def attention(prefix):
    lin(prefix + "self_attn.q_proj.weight", HEADS * HEAD_DIM * 2, HIDDEN)
    lin(prefix + "self_attn.k_proj.weight", KV_HEADS * HEAD_DIM, HIDDEN)
    lin(prefix + "self_attn.v_proj.weight", KV_HEADS * HEAD_DIM, HIDDEN)
    lin(prefix + "self_attn.o_proj.weight", HIDDEN, HEADS * HEAD_DIM)
    norm(prefix + "self_attn.q_norm.weight", HEAD_DIM)
    norm(prefix + "self_attn.k_norm.weight", HEAD_DIM)


def mlp_and_norms(prefix):
    norm(prefix + "input_layernorm.weight", HIDDEN)
    norm(prefix + "post_attention_layernorm.weight", HIDDEN)
    lin(prefix + "mlp.gate_proj.weight", INTER, HIDDEN)
    lin(prefix + "mlp.up_proj.weight", INTER, HIDDEN)
    lin(prefix + "mlp.down_proj.weight", HIDDEN, INTER)


tensors["model.embed_tokens.weight"] = (rng.standard_normal((VOCAB, HIDDEN)) * 0.5).astype(np.float32)
norm("model.norm.weight", HIDDEN)
for i in range(LAYERS):
    p = f"model.layers.{i}."
    mlp_and_norms(p)
    if (i + 1) % 4:
        conv_dim = 2 * LIN_K_HEADS * LIN_K_DIM + LIN_V_HEADS * LIN_V_DIM
        lin(p + "linear_attn.in_proj_qkv.weight", conv_dim, HIDDEN)
        lin(p + "linear_attn.in_proj_z.weight", LIN_V_HEADS * LIN_V_DIM, HIDDEN)
        lin(p + "linear_attn.in_proj_b.weight", LIN_V_HEADS, HIDDEN)
        lin(p + "linear_attn.in_proj_a.weight", LIN_V_HEADS, HIDDEN)
        lin(p + "linear_attn.out_proj.weight", HIDDEN, LIN_V_HEADS * LIN_V_DIM)
        tensors[p + "linear_attn.conv1d.weight"] = (rng.standard_normal((conv_dim, 1, CONV)) * 0.3).astype(np.float32)
        tensors[p + "linear_attn.A_log"] = rng.uniform(-1.0, 1.0, LIN_V_HEADS).astype(np.float32)
        tensors[p + "linear_attn.dt_bias"] = rng.uniform(-1.0, 1.0, LIN_V_HEADS).astype(np.float32)
        tensors[p + "linear_attn.norm.weight"] = (1.0 + rng.standard_normal(LIN_V_DIM) * 0.1).astype(np.float32)
    else:
        attention(p)

lin("mtp.fc.weight", HIDDEN, 2 * HIDDEN)
norm("mtp.pre_fc_norm_embedding.weight", HIDDEN)
norm("mtp.pre_fc_norm_hidden.weight", HIDDEN)
norm("mtp.norm.weight", HIDDEN)
attention("mtp.layers.0.")
mlp_and_norms("mtp.layers.0.")


def write_safetensors(path, items):
    header = {}
    offset = 0
    for name, array in items.items():
        size = array.nbytes
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for array in items.values():
            f.write(np.ascontiguousarray(array).tobytes())


os.makedirs(OUT, exist_ok=True)
write_safetensors(os.path.join(OUT, "model.safetensors"), tensors)
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config, f, indent=1)
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    source = os.path.join(TOK, name)
    if os.path.exists(source):
        shutil.copy(source, os.path.join(OUT, name))
print(f"wrote {OUT} ({sum(a.nbytes for a in tensors.values()) / 1e6:.1f} MB)")
