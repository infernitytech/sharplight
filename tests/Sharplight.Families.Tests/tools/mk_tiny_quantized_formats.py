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


def extra_cases(base, out_dir):
    return []


if __name__ == "__main__":
    main()
