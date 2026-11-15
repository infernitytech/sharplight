#!/usr/bin/env python3
"""More tiny quantized Hugging Face checkpoints (see mk_tiny_quantized.py, same base model, same output directory): the formats
of the second quantization round, each written by the format's own library where it runs on CPU or the GPU of this machine.

usage: mk_tiny_quantized_formats.py <models/hf directory> [case ...]   (writes <models/hf>/quant/tiny/tiny-llama-<case>)

Cases (library): mlx-2bit-g64 / mlx-3bit-g64 / mlx-5bit-g64 / mlx-6bit-g64 / mlx-mxfp4 / mlx-nvfp4 / mlx-mxfp8 (mlx.core.quantize,
CPU build of MLX), and the cases added by the other sections of this file.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mk_tiny_quantized import LINEARS, base_model  # noqa: E402


def write_config(base, out, **quant):
    with open(os.path.join(base, "config.json")) as f:
        config = json.load(f)
    config.update(quant)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def mlx_case(base, out, bits, group, mode="affine"):
    """mlx.core.quantize (CPU build of MLX) of every linear and the embedding, saved as mlx-lm saves it (any mode)."""
    import mlx.core as mx
    state = load_file(os.path.join(base, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        stem = name[:-len(".weight")] if name.endswith(".weight") else None
        if stem is not None and (any(stem.endswith(linear) for linear in LINEARS) or stem.endswith("embed_tokens")):
            w = mx.array(tensor.float().numpy()).astype(mx.bfloat16)
            parts = mx.quantize(w, group_size=group, bits=bits, mode=mode)
            result[name] = torch.from_numpy(np.array(parts[0]).view(np.int32).copy()).view(torch.int32)   # U32 words as raw bits
            scales = parts[1]
            if mode == "affine":
                result[stem + ".scales"] = torch.from_numpy(np.array(scales.astype(mx.float32))).to(torch.bfloat16)
                result[stem + ".biases"] = torch.from_numpy(np.array(parts[2].astype(mx.float32))).to(torch.bfloat16)
            else:
                result[stem + ".scales"] = torch.from_numpy(np.array(scales)).to(torch.uint8)
        else:
            result[name] = tensor
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "model.safetensors")
    save_file({k: v.contiguous() for k, v in result.items()}, path, metadata={"format": "mlx"})
    patch_u32(path)
    settings = {"group_size": group, "bits": bits}
    if mode != "affine":
        settings["mode"] = mode
    write_config(base, out, quantization=settings, quantization_config=settings)


def patch_u32(path):
    """torch cannot write U32: the I32 weight tensors of the file are re-labelled U32 in the safetensors header (same bytes)."""
    with open(path, "r+b") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
        for key, entry in header.items():
            if key != "__metadata__" and entry["dtype"] == "I32" and key.endswith(".weight"):
                entry["dtype"] = "U32"
        text = json.dumps(header, separators=(",", ":")).encode()
        text += b" " * (size - len(text))
        if len(text) != size:
            raise SystemExit("header grew")
        f.seek(8)
        f.write(text)


def main():
    root = os.path.expanduser(sys.argv[1])
    tokenizer_dir, out_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM"), os.path.join(root, "quant", "tiny")
    wanted = set(sys.argv[2:])
    os.makedirs(out_dir, exist_ok=True)
    base = base_model(tokenizer_dir, os.path.join(out_dir, "tiny-llama-bf16"))
    cases = [
        ("mlx-2bit-g64", lambda out: mlx_case(base, out, 2, 64)),
        ("mlx-3bit-g64", lambda out: mlx_case(base, out, 3, 64)),
        ("mlx-5bit-g64", lambda out: mlx_case(base, out, 5, 64)),
        ("mlx-6bit-g64", lambda out: mlx_case(base, out, 6, 64)),
        ("mlx-mxfp4", lambda out: mlx_case(base, out, 4, 32, "mxfp4")),
        ("mlx-nvfp4", lambda out: mlx_case(base, out, 4, 16, "nvfp4")),
        ("mlx-mxfp8", lambda out: mlx_case(base, out, 8, 32, "mxfp8")),
    ]
    cases += extra_cases(base, out_dir)
    for name, build in cases:
        out = os.path.join(out_dir, "tiny-llama-" + name)
        if (wanted and name not in wanted) or os.path.exists(os.path.join(out, "config.json")):
            continue
        print("building", name, flush=True)
        try:
            build(out)
        except Exception as error:  # keep going: one library failing must not block the others
            import traceback
            traceback.print_exc()
            print(f"  {name} failed: {error!r}", flush=True)
            shutil.rmtree(out, ignore_errors=True)


def autoround_case(base, out, fmt, bits, group, sym):
    """AutoRound (CPU, two tuning iterations: the packing is what the importer reads, not the rounding quality)."""
    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base)
    data = [" ".join(str(i) for i in range(j, j + 80)) for j in range(16)]
    tuner = AutoRound(model, tokenizer, bits=bits, group_size=group, sym=sym, iters=2, nsamples=8, seqlen=64, batch_size=4, device="cpu",
                      dataset=data, low_gpu_mem_usage=False)
    tuner.quantize_and_save(out, format=fmt)
    inner = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))]
    if inner:   # AutoRound saves into <out>/<model>-w<bits>g<group>/
        for name in os.listdir(os.path.join(out, inner[0])):
            shutil.move(os.path.join(out, inner[0], name), os.path.join(out, name))
        os.rmdir(os.path.join(out, inner[0]))


def bnb_case(base, out, **settings):
    """bitsandbytes through transformers (CPU build of bitsandbytes, or a CUDA device with SHARPLIGHT_BNB_DEVICE=cuda)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    device = os.environ.get("SHARPLIGHT_BNB_DEVICE", "cpu")
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=BitsAndBytesConfig(**settings), device_map=device,
                                                 dtype=torch.bfloat16)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def hqq_case(base, out, nbits, group, axis=1):
    """HQQ (hqq 0.2.8, CPU): every linear is quantized by HQQLinear and stored as the layer's state_dict (X.W_q, X.scale, X.zero,
    X.shape, X.nbits, X.group_size, X.axis, X.packing, ...), the keys transformers' HQQ integration saved in safetensors
    (transformers 5 cannot quantize with HQQ yet, so the layers are built directly)."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    state = load_file(os.path.join(base, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        stem = name[:-len(".weight")] if name.endswith(".weight") else None
        if stem is not None and any(stem.endswith(linear) for linear in LINEARS):
            linear = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
            linear.weight.data = tensor.to(torch.float16)
            layer = HQQLinear(linear, BaseQuantizeConfig(nbits=nbits, group_size=group, axis=axis), compute_dtype=torch.float16, device="cpu")
            for key, value in layer.state_dict().items():
                result[stem + "." + key] = value.contiguous()
        else:
            result[name] = tensor.contiguous()
    os.makedirs(out, exist_ok=True)
    save_file(result, os.path.join(out, "model.safetensors"))
    quant = {"quant_method": "hqq", "quant_config": {"weight_quant_params": {"nbits": nbits, "group_size": group, "axis": axis}, "skip_modules": ["lm_head"]}}
    write_config(base, out, quantization_config=quant)


def eetq_case(base, out):
    """EETQ int8 weights written as transformers' EetqLinear stores them (X.weight I8 [K, N] preprocessed for the Ampere mixed GEMM,
    X.weight_scales F16 [N]), quantized by EETQ's own C++ (tools/eetq_harness, built by build.sh; SHARPLIGHT_EETQ_LIB names the
    library): the CUDA kernels of the eetq package refuse the Blackwell GPUs of this machine, the preprocessing is CPU code."""
    import ctypes
    lib = ctypes.CDLL(os.environ["SHARPLIGHT_EETQ_LIB"])
    state = load_file(os.path.join(base, "model.safetensors"))
    result, reference = {}, {}
    for name, tensor in state.items():
        stem = name[:-len(".weight")] if name.endswith(".weight") else None
        if stem is not None and any(stem.endswith(linear) for linear in LINEARS) and all(d % 64 == 0 for d in tensor.shape):
            w = tensor.to(torch.float16).t().contiguous()                      # [K, N]
            k, n = w.shape
            processed = torch.zeros(k, n, dtype=torch.int8)
            unprocessed = torch.zeros(k, n, dtype=torch.int8)
            scales = torch.zeros(n, dtype=torch.float16)
            lib.eetq_quant_int8(ctypes.c_void_p(w.data_ptr()), k, n, ctypes.c_void_p(processed.data_ptr()),
                                ctypes.c_void_p(unprocessed.data_ptr()), ctypes.c_void_p(scales.data_ptr()))
            result[name] = processed
            result[stem + ".weight_scales"] = scales
            reference[stem] = (unprocessed.to(torch.float16) * scales[None, :]).t().contiguous()   # w8a16 GEMM: half(q) * half(scale)
        else:
            result[name] = tensor.contiguous()
    os.makedirs(out, exist_ok=True)
    save_file(result, os.path.join(out, "model.safetensors"))
    torch.save(reference, os.path.join(out, "eetq-reference.pt"))
    quant = {"quant_method": "eetq", "weights": "int8", "modules_to_not_convert": None}
    write_config(base, out, quantization_config=quant)


def extra_cases(base, out_dir):
    return [
        ("eetq-int8", lambda out: eetq_case(base, out)),
        ("hqq-4bit-g64", lambda out: hqq_case(base, out, 4, 64)),
        ("hqq-8bit-g128", lambda out: hqq_case(base, out, 8, 128)),
        ("hqq-3bit-g64", lambda out: hqq_case(base, out, 3, 64)),
        ("hqq-2bit-g32", lambda out: hqq_case(base, out, 2, 32)),
        ("hqq-1bit-g64", lambda out: hqq_case(base, out, 1, 64)),
        ("hqq-4bit-g64-axis0", lambda out: hqq_case(base, out, 4, 64, 0)),
        ("bnb-nf4", lambda out: bnb_case(base, out, load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)),
        ("bnb-fp4", lambda out: bnb_case(base, out, load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)),
        ("bnb-int8", lambda out: bnb_case(base, out, load_in_8bit=True)),
        ("autoround-gptq-4bit-sym", lambda out: autoround_case(base, out, "auto_round:auto_gptq", 4, 128, True)),
        ("autoround-gptq-4bit-asym-g32", lambda out: autoround_case(base, out, "auto_round:auto_gptq", 4, 32, False)),
        ("autoround-gptq-2bit-g64", lambda out: autoround_case(base, out, "auto_round:auto_gptq", 2, 64, False)),
        ("autoround-awq-4bit", lambda out: autoround_case(base, out, "auto_round:auto_awq", 4, 128, False)),
    ]


if __name__ == "__main__":
    main()
