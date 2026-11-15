"""
Writes a tiny random Granite Switch checkpoint (GraniteSwitchForCausalLM, model_type granite_switch) in the layout of the published checkpoints,
for which no transformers class exists: a dense Granite decoder whose q/k/v, o, gate/up and down projections carry per-adapter LoRA pairs
(lora_A / lora_B, stacked over the adapters, the fused q/k/v and gate/up projections with one pair per slice), adapter activation tokens with
their substitute tokens, and a weightless switch layer counted in num_hidden_layers. The tokenizer is the tiny Granite one; " France" (the
fourth token of the test prompt) activates the second adapter, so the prompt runs the base model up to it and the adapter from it on.

    python mk_tiny_granite_switch.py [<output directory>]     (default ~/models/hf/tiny-random-granite-switch)

Convert with llama.cpp's converter (tools/convert_tiny_gguf.py) and check against tools/ref_granite_switch.py and llama.cpp.
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file

home = os.path.expanduser("~/models/hf")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(home, "tiny-random-granite-switch")
tokenizer = os.path.join(home, "tiny-random-granite")

hidden, heads, kv_heads, head_dim, ffn, layers, adapters, rank, vocab = 64, 4, 2, 16, 128, 3, 3, 8, 49155
config = {
    "architectures": ["GraniteSwitchForCausalLM"],
    "model_type": "granite_switch",
    "adapter_names": ["first", "second", "third"],
    "adapter_ranks": [rank] * adapters,
    "adapter_token_ids": [48990, 45600, 48991],
    "adapter_substitute_token_ids": [328, 312, 322],
    "num_adapters": adapters,
    "max_lora_rank": rank,
    "control_token_gain": 15.0,
    "lora_target_modules": ["o_proj", "qkv_proj", "shared_input_linear", "shared_output_linear"],
    "hidden_size": hidden,
    "intermediate_size": ffn,
    "shared_intermediate_size": ffn,
    "num_attention_heads": heads,
    "num_key_value_heads": kv_heads,
    "projection_head_dim": head_dim,
    "num_hidden_layers": layers + 1,   # the weightless switch layer is counted
    "layer_types": ["full_attention"] * (layers + 1),
    "num_local_experts": 0,
    "num_experts_per_tok": 2,
    "attention_multiplier": 0.125,
    "embedding_multiplier": 2.0,
    "residual_multiplier": 0.5,
    "logits_scaling": 8.0,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_bias": False,
    "position_embedding_type": "rope",
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "max_position_embeddings": 4096,
    "tie_word_embeddings": True,
    "vocab_size": vocab,
    "bos_token_id": 0,
    "eos_token_id": 0,
    "pad_token_id": 0,
    "dtype": "float32",
}

g = torch.Generator().manual_seed(1234)


def rand(*shape, std):
    return (torch.randn(*shape, generator=g) * std).contiguous()


def norm():
    return (1 + rand(hidden, std=0.1)).contiguous()


def lora(out_rows, in_cols):
    """One adapter-stacked LoRA pair as stored: A [adapters, 1, rank, in], B [adapters, 1, out, rank]."""
    return rand(adapters, 1, rank, in_cols, std=0.25), rand(adapters, 1, out_rows, rank, std=0.25)


q_size, kv_size = heads * head_dim, kv_heads * head_dim
t = {
    "model.embed_tokens.weight": rand(vocab, hidden, std=0.5),
    "model.norm.weight": norm(),
    # The published checkpoints' switch buffers (the converter and the importers drop them).
    "model.switch.codebook": rand(4, 8, std=1.0),
    "model.switch.control_to_substitute_lut": torch.tensor(config["adapter_substitute_token_ids"], dtype=torch.int64),
}
for i in range(layers):
    p = f"model.layers.{i}."
    t[p + "input_layernorm.weight"] = norm()
    t[p + "post_attention_layernorm.weight"] = norm()
    t[p + "self_attn.qkv_proj.base_layer.weight"] = rand(q_size + 2 * kv_size, hidden, std=0.25)
    for s, rows in enumerate([q_size, kv_size, kv_size]):
        a, b = lora(rows, hidden)
        t[p + f"self_attn.qkv_proj.lora_A_slices.{s}"] = a
        t[p + f"self_attn.qkv_proj.lora_B_slices.{s}"] = b
    t[p + "self_attn.o_proj.base_layer.weight"] = rand(hidden, q_size, std=0.25)
    t[p + "self_attn.o_proj.lora_A"], t[p + "self_attn.o_proj.lora_B"] = lora(hidden, q_size)
    t[p + "shared_mlp.input_linear.base_layer.weight"] = rand(2 * ffn, hidden, std=0.25)
    for s in range(2):
        a, b = lora(ffn, hidden)
        t[p + f"shared_mlp.input_linear.lora_A_slices.{s}"] = a
        t[p + f"shared_mlp.input_linear.lora_B_slices.{s}"] = b
    t[p + "shared_mlp.output_linear.base_layer.weight"] = rand(hidden, ffn, std=0.25)
    t[p + "shared_mlp.output_linear.lora_A"], t[p + "shared_mlp.output_linear.lora_B"] = lora(hidden, ffn)

os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"):
    if os.path.exists(os.path.join(tokenizer, name)):
        shutil.copy(os.path.join(tokenizer, name), out)
print("wrote", out)
