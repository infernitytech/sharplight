#!/usr/bin/env python3
"""Writes tiny AQLM checkpoints (random codes and codebooks in the aqlm HF layout) with the reference dequantization of every weight.

  python tools/mk-tiny-aqlm.py <output-dir>

Needs torch and `pip install --no-deps aqlm` (aqlm 1.1.7; its `_dequantize_weight` is the reference). For each variant
(<name> = num_codebooks x nbits, out/in group size) the output has <name>/ (config.json + model.safetensors, a tiny llama whose
linears are AQLM modules) and <name>/sharplight-reference/dense.safetensors: the reference's dequantized F16 weight of every module.
"""
import json
import sys
from pathlib import Path

import torch
from aqlm.utils import _dequantize_weight, unpack_int_data
from safetensors.torch import save_file

HIDDEN, INTER, LAYERS, VOCAB, HEADS, KV_HEADS = 64, 128, 2, 128, 4, 2
VARIANTS = {
    "aqlm-1x8-g8": (1, 8, 1, 8, torch.float16),
    "aqlm-2x8-g8": (2, 8, 1, 8, torch.float16),
    "aqlm-4x6-og2": (4, 6, 2, 8, torch.float16),
    "aqlm-8x8-g8": (8, 8, 1, 8, torch.float16),
    "aqlm-1x10-bf16": (1, 10, 1, 8, torch.bfloat16),
}


def linears():
    head = HIDDEN // HEADS
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        yield f"{p}.self_attn.q_proj", HIDDEN, HIDDEN
        yield f"{p}.self_attn.k_proj", KV_HEADS * head, HIDDEN
        yield f"{p}.self_attn.v_proj", KV_HEADS * head, HIDDEN
        yield f"{p}.self_attn.o_proj", HIDDEN, HIDDEN
        yield f"{p}.mlp.gate_proj", INTER, HIDDEN
        yield f"{p}.mlp.up_proj", INTER, HIDDEN
        yield f"{p}.mlp.down_proj", HIDDEN, INTER


def main():
    out = Path(sys.argv[1])
    g = torch.Generator().manual_seed(2026)
    for name, (ncb, nbits, og, ig, dtype) in VARIANTS.items():
        d = out / name
        (d / "sharplight-reference").mkdir(parents=True, exist_ok=True)
        tensors, dense = {}, {}
        for module, o, k in linears():
            codebooks = (torch.randn(ncb, 2**nbits, og, ig, generator=g) * 0.05).to(dtype)
            # the library stores codes as int8 / int16 / int32 by width; unsigned 8-bit codes wrap into the signed range
            raw = torch.randint(0, 2**nbits, (o // og, k // ig, ncb), generator=g)
            codes = raw.to(torch.uint8).view(torch.int8) if nbits <= 8 else raw.to(torch.int16)
            scales = (torch.rand(o // og, 1, 1, 1, generator=g) * 2 + 0.5).to(dtype)
            tensors[f"{module}.codebooks"] = codebooks
            tensors[f"{module}.codes"] = codes
            tensors[f"{module}.scales"] = scales
            dense[f"{module}.weight"] = _dequantize_weight(unpack_int_data(codes, nbits), codebooks, scales).contiguous()
        for n, shape in [("model.embed_tokens.weight", (VOCAB, HIDDEN)), ("lm_head.weight", (VOCAB, HIDDEN)), ("model.norm.weight", (HIDDEN,))]:
            tensors[n] = torch.randn(*shape, generator=g).to(dtype)
        for i in range(LAYERS):
            for n in ("input_layernorm", "post_attention_layernorm"):
                tensors[f"model.layers.{i}.{n}.weight"] = torch.ones(HIDDEN).to(dtype)
        save_file(tensors, str(d / "model.safetensors"))
        save_file(dense, str(d / "sharplight-reference" / "dense.safetensors"))
        cfg = {
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": HIDDEN, "intermediate_size": INTER,
            "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB,
            "max_position_embeddings": 256, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": False,
            "torch_dtype": "bfloat16" if dtype == torch.bfloat16 else "float16", "hidden_act": "silu",
            "quantization_config": {
                "quant_method": "aqlm", "in_group_size": ig, "out_group_size": og, "num_codebooks": ncb,
                "nbits_per_codebook": nbits, "linear_weights_not_to_quantize": [],
            },
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        print(name, "ok")


if __name__ == "__main__":
    main()
