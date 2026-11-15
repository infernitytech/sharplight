#!/usr/bin/env python3
"""Tiny quantized Hugging Face checkpoints of the third quantization round (same base model and output directory as
mk_tiny_quantized.py / mk_tiny_quantized_formats.py), each written by the format's own library code:

usage: mk_tiny_quantized_formats3.py <models/hf directory> [case ...]   (writes <models/hf>/quant/tiny/tiny-llama-<case>)

Cases (library; the Python environment each needs is noted):
  awq-gemv-fast, awq-gemv-fast-g64   AutoAWQ 0.2.9 WQLinear_GEMVFast.from_linear                       (formats-a3 venv, CPU)
  awq-marlin                         AutoAWQ 0.2.9 WQLinear_Marlin.from_linear                         (formats-a3 venv, CPU)
  gptq-marlin-4bit-g128, gptq-marlin-8bit-channel, gptq-marlin-4bit-g64
                                     RTN symmetric GPTQ codes (mk_tiny_quantized.gptq_case), repacked by vLLM's gptq_marlin_repack and
                                     marlin_permute_scales (CUDA op; vllm venv with CUDA_VISIBLE_DEVICES set)
  quanto-int8, quanto-int4, quanto-float8
                                     optimum-quanto 0.2.7 quantize + freeze, state dict saved like QuantizedModelForCausalLM  (formats-a3 venv)
  torchao-int8wo, torchao-int4wo-plain, torchao-float8wo   see torchao_case                            (formats-a3 venv)
  quark-int4-g128, quark-int8-channel, quark-mxfp6-e3m2, quark-mxfp6-e2m3
                                     weights packed by quark.torch.utils.pack (Pack_4_bits / Pack_8_bits / Pack_mxfp6)  (formats-a3 venv)
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mk_tiny_quantized import LINEARS, base_model  # noqa: E402


def is_linear(name):
    return name.endswith(".weight") and any(name[:-len(".weight")].endswith(linear) for linear in LINEARS)


def write(base, out, state, quantization_config, metadata=None, extra_config=None):
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, "model.safetensors"), metadata=metadata or {"format": "pt"})
    with open(os.path.join(base, "config.json")) as f:
        config = json.load(f)
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    config.update(extra_config or {})
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def awq_scales_zeros(w, group, w_bit=4):
    """AutoAWQ's AwqQuantizer.pseudo_quantize_tensor (zero point branch): per group scales [N, G] and zeros [N, G]."""
    grouped = w.float().reshape(-1, group)
    max_int = 2 ** w_bit - 1
    scales = (grouped.amax(dim=1, keepdim=True) - grouped.amin(dim=1, keepdim=True)).clamp(min=1e-5) / max_int
    zeros = (-torch.round(grouped.amin(dim=1, keepdim=True) / scales)).clamp_(0, max_int)
    n = w.shape[0]
    return scales.view(n, -1).half(), zeros.view(n, -1).half()


def awq_gemv_fast_case(base, out, group):
    from awq.modules.linear.gemv_fast import WQLinear_GEMVFast
    state = load_file(os.path.join(base, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        if not is_linear(name):
            result[name] = tensor
            continue
        stem = name[:-len(".weight")]
        linear = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
        linear.weight.data = tensor.half()
        scales, zeros = awq_scales_zeros(linear.weight.data, group)
        module = WQLinear_GEMVFast.from_linear(linear, 4, group, scales=scales, zeros=zeros)
        result[stem + ".qweight"], result[stem + ".qzeros"], result[stem + ".scales"] = module.qweight, module.qzeros, module.scales
    write(base, out, result, {"quant_method": "awq", "bits": 4, "group_size": group, "version": "gemv_fast", "zero_point": True,
                              "modules_to_not_convert": None})


def awq_marlin_case(base, out, group):
    from awq.modules.linear.marlin import WQLinear_Marlin
    state = load_file(os.path.join(base, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        if not is_linear(name):
            result[name] = tensor
            continue
        stem = name[:-len(".weight")]
        linear = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
        linear.weight.data = tensor.half()
        grouped = linear.weight.data.float().reshape(tensor.shape[0], -1, group)
        scales = (grouped.abs().amax(dim=2) / 8).clamp(min=1e-5).half()
        module = WQLinear_Marlin.from_linear(linear, 4, group, scales=scales)
        result[stem + ".qweight"], result[stem + ".scales"] = module.qweight, module.scales
    write(base, out, result, {"quant_method": "awq", "bits": 4, "group_size": group, "version": "marlin", "zero_point": False,
                              "modules_to_not_convert": None})


def gptq_marlin_case(base, out, bits, group):
    """RTN symmetric GPTQ codes repacked as vLLM does for the Marlin kernel (gptq_marlin_repack + marlin_permute_scales)."""
    import tempfile
    from mk_tiny_quantized import gptq_case
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    source = os.path.join(os.environ.get("SCRATCH", tempfile.gettempdir()), f"gptq-sym-{bits}-{group}")
    shutil.rmtree(source, ignore_errors=True)
    gptq_case(base, source, bits, group, True, False, "gptq")
    state = load_file(os.path.join(source, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        if name.endswith((".qzeros", ".g_idx")):
            continue
        if name.endswith(".qweight"):
            stem = name[:-len(".qweight")]
            k, n = tensor.shape[0] * 32 // bits, tensor.shape[1]
            result[name] = ops.gptq_marlin_repack(tensor.cuda(), k, n, bits).cpu()
            result[stem + ".scales"] = marlin_permute_scales(state[stem + ".scales"].cuda(), k, n, group).cpu()
        elif not name.endswith(".scales"):
            result[name] = tensor
    write(base, out, result, {"quant_method": "gptq", "bits": bits, "group_size": group, "desc_act": False, "sym": True,
                              "checkpoint_format": "marlin"})
    shutil.rmtree(source, ignore_errors=True)


def main():
    root = os.path.expanduser(sys.argv[1])
    tokenizer_dir, out_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM"), os.path.join(root, "quant", "tiny")
    wanted = set(sys.argv[2:])
    os.makedirs(out_dir, exist_ok=True)
    base = base_model(tokenizer_dir, os.path.join(out_dir, "tiny-llama-bf16"))
    cases = [
        ("awq-gemv-fast", lambda out: awq_gemv_fast_case(base, out, 128)),
        ("awq-gemv-fast-g64", lambda out: awq_gemv_fast_case(base, out, 64)),
        ("awq-marlin", lambda out: awq_marlin_case(base, out, 128)),
        ("gptq-marlin-4bit-g128", lambda out: gptq_marlin_case(base, out, 4, 128)),
        ("gptq-marlin-4bit-g64", lambda out: gptq_marlin_case(base, out, 4, 64)),
        ("gptq-marlin-8bit-channel", lambda out: gptq_marlin_case(base, out, 8, -1)),
    ]
    cases += extra_cases(base)
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


def extra_cases(base):
    return []


if __name__ == "__main__":
    main()
