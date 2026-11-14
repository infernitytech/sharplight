#!/usr/bin/env python3
"""Builds the tiny quantized Hugging Face checkpoints of tests/Sharplight.Families.Tests/QuantizedCheckpointTests.cs.

A small random Llama (hidden 256, MLP 768, 2 layers, the vocabulary-1024 tokenizer of tiny-random-GPTNeoXForCausalLM) is quantized with the real libraries where they run
(llm-compressor for compressed-tensors NVFP4 / MXFP4 / integer weights, AutoAWQ, bitsandbytes through transformers) and, for the
GPTQ layouts (auto-gptq / gptqmodel, which do not install here), packed by this script following gptqmodel's packing code
(nn_modules/qlinear/__init__.py: codes as a little-endian bit stream along K per column, zero points along N, v1 stores zero - 1).
AMD Quark needs a JIT-compiled extension that does not build here: its MXFP4 case re-labels the llm-compressor MXFP4 codes in
Quark's export layout (quark/torch/utils/pack.py Pack_fp4: U8 X.weight with the first value in the low nibble, U8 E8M0
X.weight_scale per 32 values; config as Quark's exporter writes it). MLX cases use mlx.core.quantize of the CPU build of MLX.
Every case is written to <dir>/tiny-llama-<case>/; the tests skip a case whose directory is missing. Reference dequantized
weights and logits: hf_quant_reference.py on every output directory.

usage: mk_tiny_quantized.py <models/hf directory> [case ...]   (writes <models/hf>/quant/tiny/)

Needs torch, transformers, compressed-tensors, llmcompressor, autoawq, bitsandbytes (a CUDA GPU for AutoAWQ and bitsandbytes).
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

HIDDEN, MLP, LAYERS = 256, 768, 2
LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def base_model(tokenizer_dir, out):
    # Vocabulary 1024 (the tokenizer's), so a checkpoint stays a few MB.
    """The random BF16 Llama all cases start from (weights a bit larger than the default init so the logits are peaked)."""
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    config = LlamaConfig(vocab_size=1024, hidden_size=HIDDEN, intermediate_size=MLP, num_hidden_layers=LAYERS,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=2048,
                         tie_word_embeddings=True, initializer_range=0.05, bos_token_id=0, eos_token_id=0,
                         torch_dtype="bfloat16")
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def calibration(tokenizer_dir, samples=16, length=64):
    from datasets import Dataset
    torch.manual_seed(1)
    ids = torch.randint(10, 1000, (samples, length))
    return Dataset.from_dict({"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()})


def llmcompressor_case(base, out, recipe, data=None):
    from llmcompressor import oneshot
    model = LlamaForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    kwargs = dict(model=model, recipe=recipe)
    if data is not None:
        kwargs.update(dataset=data, num_calibration_samples=len(data), max_seq_length=64)
    oneshot(**kwargs)
    model.save_pretrained(out, save_compressed=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def compressed_tensors_cases(base, out_dir, tokenizer_dir):
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    data = calibration(tokenizer_dir)
    yield "ct-nvfp4a16", lambda out: llmcompressor_case(base, out, QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"]))
    yield "ct-mxfp4a16", lambda out: llmcompressor_case(base, out, QuantizationModifier(targets="Linear", scheme="MXFP4A16", ignore=["lm_head"]))
    yield "ct-w4a16-g128", lambda out: llmcompressor_case(base, out, GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]), data)
    yield "ct-w4a16-actorder", lambda out: llmcompressor_case(base, out, GPTQModifier(
        targets="Linear", ignore=["lm_head"], config_groups={"group_0": {"targets": ["Linear"], "weights": {
            "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128, "actorder": "weight"}}}), data)
    yield "ct-w4a16-asym-g32", lambda out: llmcompressor_case(base, out, QuantizationModifier(
        targets="Linear", ignore=["lm_head"], config_groups={"group_0": {"targets": ["Linear"], "weights": {
            "num_bits": 4, "type": "int", "symmetric": False, "strategy": "group", "group_size": 32}}}))
    yield "ct-w8a16-channel", lambda out: llmcompressor_case(base, out, QuantizationModifier(targets="Linear", scheme="W8A16", ignore=["lm_head"]))
    yield "ct-w8a8-int", lambda out: llmcompressor_case(base, out, QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]))


def awq_case(base, out, version):
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base)
    torch.manual_seed(2)
    texts = [" ".join(str(int(x)) for x in torch.randint(0, 1000, (48,))) for _ in range(16)]
    model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": version},
                   calib_data=texts, n_parallel_calib_samples=4, max_calib_samples=16, max_calib_seq_len=512)
    model.save_quantized(out)
    tokenizer.save_pretrained(out)


def bnb_case(base, out, **settings):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=BitsAndBytesConfig(**settings), device_map="cuda",
                                                 torch_dtype=torch.bfloat16)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def pack_stream(codes, bits):
    """Codes [K, N] (unsigned) as gptqmodel packs qweight: a little-endian bit stream along K per column, in int32 words [K*bits/32, N]."""
    k, n = codes.shape
    words = torch.zeros((k * bits + 31) // 32, n, dtype=torch.int64)
    for i in range(k):
        position = i * bits
        word, offset = position // 32, position % 32
        words[word] |= (codes[i].to(torch.int64) << offset) & 0xFFFFFFFF
        if offset + bits > 32:
            words[word + 1] |= codes[i].to(torch.int64) >> (32 - offset)
    return words.to(torch.uint32).view(torch.int32) if hasattr(torch, "uint32") else words.to(torch.int32)


def gptq_case(base, out, bits, group, sym, act_order, version):
    """RTN-quantized weights in the GPTQ checkpoint layout (the packing, not the GPTQ solver, is what the importer reads)."""
    from safetensors.torch import load_file, save_file
    torch.manual_seed(3)
    state = load_file(os.path.join(base, "model.safetensors"))
    maxq = 2 ** bits - 1
    result = {}
    for name, tensor in state.items():
        stem = name[:-len(".weight")] if name.endswith(".weight") else None
        if stem is None or not any(stem.endswith(linear) for linear in LINEARS):
            result[name] = tensor
            continue
        w = tensor.float()                                          # [N, K]
        n, k = w.shape
        g = k if group == -1 else group
        perm = torch.randperm(k) if act_order else torch.arange(k)
        g_idx = torch.empty(k, dtype=torch.int32)
        g_idx[perm] = (torch.arange(k) // g).to(torch.int32)       # act-order: groups follow the permutation
        groups = k // g
        scales = torch.empty(groups, n)
        zeros = torch.empty(groups, n, dtype=torch.int64)
        codes = torch.empty(k, n, dtype=torch.int64)
        for gi in range(groups):
            columns = (g_idx == gi).nonzero().flatten()
            block = w[:, columns]
            lo, hi = block.min(dim=1).values.clamp(max=0), block.max(dim=1).values.clamp(min=0)
            if sym:
                amax = torch.maximum(lo.abs(), hi)
                scale = (2 * amax / maxq).clamp(min=1e-5)
                zero = torch.full((n,), (maxq + 1) // 2, dtype=torch.int64)
            else:
                scale = ((hi - lo) / maxq).clamp(min=1e-5)
                zero = torch.round(-lo / scale).to(torch.int64).clamp(0, maxq)
            scale = scale.half().float()
            scales[gi] = scale
            zeros[gi] = zero
            codes[columns] = torch.clamp(torch.round(block / scale[:, None]) + zero[:, None], 0, maxq).to(torch.int64).T
        stored_zeros = (zeros - 1) & maxq if version == "gptq" else zeros
        result[stem + ".qweight"] = pack_stream(codes, bits)
        result[stem + ".qzeros"] = pack_stream(stored_zeros.T.contiguous(), bits).T.contiguous()
        result[stem + ".scales"] = scales.half()
        result[stem + ".g_idx"] = g_idx
    os.makedirs(out, exist_ok=True)
    save_file(result, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(base, "config.json")) as f:
        config = json.load(f)
    config["quantization_config"] = {"quant_method": "gptq", "bits": bits, "group_size": group, "desc_act": act_order, "sym": sym,
                                     "checkpoint_format": version}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def quark_case(source, out):
    """The MXFP4 weights of the compressed-tensors case in the Quark export layout and config."""
    from safetensors.torch import load_file, save_file
    state = load_file(os.path.join(source, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        if name.endswith(".weight_packed"):
            result[name[:-len("_packed")]] = tensor
        else:
            result[name] = tensor
    os.makedirs(out, exist_ok=True)
    save_file(result, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    mx = {"dtype": "fp4", "qscheme": "per_group", "ch_axis": -1, "group_size": 32, "is_dynamic": False, "scale_format": "e8m0",
          "scale_type": "float", "round_method": "half_even", "scale_calculation_mode": "even", "observer_cls": "PerBlockMXObserver"}
    config["quantization_config"] = {"quant_method": "quark", "quant_mode": "eager_mode", "exclude": ["lm_head"],
                                     "export": {"pack_method": "reorder", "weight_format": "real_quantized", "kv_cache_group": []},
                                     "global_quant_config": {"weight": mx, "input_tensors": dict(mx, is_dynamic=True), "output_tensors": None,
                                                             "bias": None},
                                     "layer_quant_config": {}, "layer_type_quant_config": {}}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    AutoTokenizer.from_pretrained(source).save_pretrained(out)


def mlx_case(base, out, bits, group):
    """mlx.core.quantize (CPU build of MLX) of every linear and the embedding, saved as mlx-lm saves it."""
    import mlx.core as mx
    import numpy as np
    from safetensors.numpy import save_file
    from safetensors.torch import load_file
    state = load_file(os.path.join(base, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        stem = name[:-len(".weight")] if name.endswith(".weight") else None
        if stem is not None and (any(stem.endswith(linear) for linear in LINEARS) or stem.endswith("embed_tokens")):
            w = mx.array(tensor.float().numpy()).astype(mx.bfloat16)
            q, scales, biases = mx.quantize(w, group_size=group, bits=bits)
            result[name] = np.array(q)
            result[stem + ".scales"] = np.array(scales.astype(mx.float32)).astype(np.float32)
            result[stem + ".biases"] = np.array(biases.astype(mx.float32)).astype(np.float32)
        else:
            result[name] = tensor.float().numpy()
    # MLX stores bf16; numpy has no bf16, so the float32 copies are converted back below.
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "model.safetensors")
    save_file(result, path, metadata={"format": "mlx"})
    from safetensors.torch import load_file as load_torch, save_file as save_torch
    torch_state = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in load_torch(path).items()}
    save_torch(torch_state, path, metadata={"format": "mlx"})
    with open(os.path.join(base, "config.json")) as f:
        config = json.load(f)
    config["quantization"] = {"group_size": group, "bits": bits}
    config["quantization_config"] = {"group_size": group, "bits": bits}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def main():
    root = os.path.expanduser(sys.argv[1])
    tokenizer_dir, out_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM"), os.path.join(root, "quant", "tiny")
    wanted = set(sys.argv[2:])
    os.makedirs(out_dir, exist_ok=True)
    base = base_model(tokenizer_dir, os.path.join(out_dir, "tiny-llama-bf16"))
    cases = list(compressed_tensors_cases(base, out_dir, tokenizer_dir))
    cases += [
        ("awq-gemm", lambda out: awq_case(base, out, "GEMM")),
        ("awq-gemv", lambda out: awq_case(base, out, "GEMV")),
        ("bnb-nf4-double", lambda out: bnb_case(base, out, load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                bnb_4bit_compute_dtype=torch.bfloat16)),
        ("bnb-nf4", lambda out: bnb_case(base, out, load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)),
        ("bnb-fp4", lambda out: bnb_case(base, out, load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)),
        ("bnb-int8", lambda out: bnb_case(base, out, load_in_8bit=True)),
        ("gptq-4bit-g128-sym", lambda out: gptq_case(base, out, 4, 128, True, False, "gptq")),
        ("gptq-4bit-g128-actorder", lambda out: gptq_case(base, out, 4, 128, False, True, "gptq")),
        ("gptq-4bit-g32-v2", lambda out: gptq_case(base, out, 4, 32, False, False, "gptq_v2")),
        ("gptq-3bit-g128", lambda out: gptq_case(base, out, 3, 128, False, False, "gptq")),
        ("gptq-8bit-channel", lambda out: gptq_case(base, out, 8, -1, True, False, "gptq")),
        ("quark-mxfp4", lambda out: quark_case(os.path.join(out_dir, "tiny-llama-ct-mxfp4a16"), out)),
        ("mlx-4bit-g64", lambda out: mlx_case(base, out, 4, 64)),
        ("mlx-4bit-g32", lambda out: mlx_case(base, out, 4, 32)),
        ("mlx-8bit-g64", lambda out: mlx_case(base, out, 8, 64)),
    ]
    for name, build in cases:
        out = os.path.join(out_dir, "tiny-llama-" + name)
        if (wanted and name not in wanted) or os.path.exists(os.path.join(out, "config.json")):
            continue
        print("building", name, flush=True)
        try:
            build(out)
        except Exception as error:  # keep going: one library failing must not block the others
            print(f"  {name} failed: {error!r}", flush=True)
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
