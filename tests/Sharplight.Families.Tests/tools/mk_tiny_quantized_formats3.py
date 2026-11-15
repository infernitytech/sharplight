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
  torchao-int8wo, torchao-int8wo-g128, torchao-float8wo   see torchao_case                     (formats-a3 venv, CPU; int4 needs mslk, absent)
  quark-int4-g128, quark-int4-g64-sym, quark-int4-g128-f32scale, quark-int8-channel, quark-mxfp6-e3m2, quark-mxfp6-e2m3,
                                     quark-mx-fp6-e3m2 (in-stream)
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
    from awq_standalone import load
    WQLinear_GEMVFast = load("gemv_fast").WQLinear_GEMVFast
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
    from awq_standalone import load
    WQLinear_Marlin = load("marlin").WQLinear_Marlin
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


def quark_config(weight, pack_method="reorder"):
    return {"quant_method": "quark", "export": {"pack_method": pack_method, "kv_cache_group": []},
            "global_quant_config": {"weight": weight, "input_tensors": None, "output_tensors": None, "bias": None, "target_device": None},
            "layer_quant_config": {}, "layer_type_quant_config": {}, "exclude": ["lm_head"], "algo_config": None, "softmax_quant_spec": None}


def quark_int_case(base, out, bits, scheme, group, symmetric, f16_scales):
    """Integer weights packed by quark.torch.utils.pack (Pack_4_bits / Pack_8_bits, as the exporter calls them) from RTN codes.
    int4 / int8 are signed, the zero point of an asymmetric weight a signed integer. f16_scales: the scales are rounded to f16 values
    (stored as f32, as Quark stores them), else the natural f32 scales (which no f16-scaled block type holds)."""
    from quark.torch.utils.pack import Pack_4_bits, Pack_8_bits
    state = load_file(os.path.join(base, "model.safetensors"))
    lo_q, hi_q = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    dtype = f"int{bits}"
    result = {}
    for name, tensor in state.items():
        if not is_linear(name):
            result[name] = tensor
            continue
        stem = name[:-len(".weight")]
        w = tensor.float()                                                      # [N, K]
        n, k = w.shape
        g = k if scheme == "per_channel" else group
        grouped = w.reshape(n, k // g, g)
        lo, hi = grouped.amin(dim=2).clamp(max=0), grouped.amax(dim=2).clamp(min=0)
        if symmetric:
            scale = (torch.maximum(lo.abs(), hi) / hi_q).clamp(min=1e-6)
            zero = torch.zeros_like(scale, dtype=torch.int32)
        else:
            scale = ((hi - lo) / (hi_q - lo_q)).clamp(min=1e-6)
            zero = (lo_q - torch.round(lo / scale)).to(torch.int32)
        if f16_scales:
            scale = scale.half().float()
        codes = torch.clamp(torch.round(grouped / scale[:, :, None]) + zero[:, :, None], lo_q, hi_q).to(torch.int32).reshape(n, k)
        pack = (Pack_4_bits if bits == 4 else Pack_8_bits)(scheme, dtype)
        result[stem + ".weight"] = pack.pack(codes, True)
        if scheme == "per_group":
            result[stem + ".weight_scale"] = scale.t().contiguous()           # [K / group, N]
            if not symmetric:
                result[stem + ".weight_zero_point"] = pack.pack(zero, True)
        else:
            result[stem + ".weight_scale"] = scale[:, 0].contiguous()         # [N]
            if not symmetric:
                result[stem + ".weight_zero_point"] = pack.pack(zero[:, 0].contiguous(), True)
    spec = {"dtype": dtype, "is_dynamic": False, "qscheme": scheme, "ch_axis": 0 if scheme == "per_channel" else 1, "group_size": group if scheme == "per_group" else None,
            "symmetric": symmetric, "round_method": "half_even", "scale_type": "float", "scale_format": None, "zero_point_type": "int32",
            "observer_cls": "PerGroupMinMaxObserver" if scheme == "per_group" else "PerChannelMinMaxObserver"}
    write(base, out, result, quark_config(spec))


FP6_FORMATS = {"fp6_e3m2": (3, 2), "fp6_e2m3": (2, 3)}


def fp6_values(e_bits, m_bits):
    """The 64 FP6 codes' values as quark.torch.utils.pack.Pack_fp6.unpack computes them."""
    ebias = (1 << (e_bits - 1)) - 1
    mbias = 23 - m_bits
    codes = torch.arange(64, dtype=torch.int32)
    bits = ((codes & 0x1F) << mbias) | ((codes & 0x20) << 26)
    return bits.view(torch.float32) * (2.0 ** (127 - ebias))


def fp6_quantize(values, e_bits, m_bits):
    """Nearest FP6 code of each value (ties to the lower code index): codes [..] int32."""
    table = fp6_values(e_bits, m_bits)
    return (values[..., None] - table).abs().argmin(dim=-1).to(torch.int32), table


def quark_mxfp6_case(base, out, element, embedded):
    """OCP MXFP6 weights packed by Quark's own packers: Pack_fp6 (stand-alone dtype, U8 weight + E8M0 weight_scale) or Pack_mxfp6
    (dtype mx: the scale is the first byte of every 25-byte block of the weight)."""
    from quark.torch.utils.pack import Pack_fp6, Pack_mxfp6
    e_bits, m_bits = FP6_FORMATS[element]
    emax = int(torch.log2(fp6_values(e_bits, m_bits).abs().max()).floor())
    state = load_file(os.path.join(base, "model.safetensors"))
    result = {}
    for name, tensor in state.items():
        if not is_linear(name):
            result[name] = tensor
            continue
        stem = name[:-len(".weight")]
        w = tensor.float()
        n, k = w.shape
        blocks = w.reshape(n, k // 32, 32)
        exponent = torch.floor(torch.log2(blocks.abs().amax(dim=2).clamp(min=1e-30))) - emax          # power-of-two scale per block
        scaled = blocks / (2.0 ** exponent)[:, :, None]
        codes, table = fp6_quantize(scaled, e_bits, m_bits)
        quantized = table[codes.long()]                                                            # [N, K/32, 32] element values
        if embedded:
            stream = torch.cat([(2.0 ** exponent)[:, :, None], quantized], dim=2).reshape(n, k // 32 * 33)
            packed = Pack_mxfp6(None, "mx", e_bits, m_bits).pack(stream, True)
            check = Pack_mxfp6(None, "mx", e_bits, m_bits).unpack(packed, True)
            assert torch.equal(check, stream), "Pack_mxfp6 does not round-trip"
            result[stem + ".weight"] = packed.contiguous()
        else:
            packed = Pack_fp6(None, element, e_bits, m_bits).pack(quantized.reshape(n, k), True)
            assert torch.equal(Pack_fp6(None, element, e_bits, m_bits).unpack(packed, True), quantized.reshape(n, k))
            result[stem + ".weight"] = packed.contiguous()
            result[stem + ".weight_scale"] = (exponent + 127).to(torch.uint8)
    if embedded:
        spec = {"dtype": "mx", "mx_element_dtype": element, "group_size": 32, "qscheme": "per_group", "ch_axis": -1, "is_dynamic": False,
                "symmetric": True, "scale_format": "e8m0"}
    else:
        spec = {"dtype": element, "group_size": 32, "qscheme": "per_group", "ch_axis": -1, "is_dynamic": False, "symmetric": True,
                "scale_format": "e8m0", "scale_type": "float"}
    write(base, out, result, quark_config(spec))


def quanto_case(base, out, weights, group_size=None):
    """optimum-quanto 0.2.7: every linear quantized (quantize + freeze) and its state dict saved like QuantizedModelForCausalLM
    (X.weight._data[._data], ._scale, ._shift, input_scale / output_scale). The reference (quanto-reference.pt) is the library's own
    dequantization with scale and shift cast to f32 (quanto computes in the module's dtype, f16 here, rounding every step)."""
    from optimum.quanto import freeze, qfloat8, qint2, qint4, qint8, quantize
    qtype = {"qint8": qint8, "qint4": qint4, "qint2": qint2, "qfloat8": qfloat8}[weights]
    state = load_file(os.path.join(base, "model.safetensors"))
    result, reference = {}, {}
    for name, tensor in state.items():
        if not is_linear(name):
            result[name] = tensor
            continue
        stem = name[:-len(".weight")]
        module = torch.nn.Sequential(torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False))
        module[0].weight.data = tensor.to(torch.float16)
        quantize(module, weights=qtype)
        freeze(module)
        for key, value in module.state_dict().items():
            result[stem + key[len("0"):]] = value.contiguous()
        packed = module[0].weight
        reference[stem] = dequantize_f32(packed)
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in result.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    torch.save(reference, os.path.join(out, "quanto-reference.pt"))
    with open(os.path.join(base, "config.json")) as f:
        config = json.load(f)
    config["quantization_config"] = {"quant_method": "quanto", "weights": weights, "activations": None, "modules_to_not_convert": ["lm_head"]}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def dequantize_f32(weight):
    """The quanto dequantization of a frozen weight with its scale (and shift) as f32."""
    import copy
    clone = copy.copy(weight)
    clone._scale = weight._scale.float()
    if hasattr(weight, "_shift"):
        clone._shift = weight._shift.float()
    return clone.dequantize().float()


def torchao_case(base, out, kind, group_size=None):
    """torchao 0.18 through transformers (TorchAoConfig, CPU; safetensors with the tensor subclass parts as X._weight_qdata /
    X._weight_scale / X._weight_zero_point). The reference (torchao-reference.pt) is the library's dequantization of every weight in f32:
    (qdata - zero point) * scale for Int8Tensor, qdata * scale for Float8Tensor, from the tensors the library holds."""
    from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
    from torchao.quantization import PerGroup
    config = (Int8WeightOnlyConfig(granularity=PerGroup(group_size)) if group_size else Int8WeightOnlyConfig()) if kind == "int8" else Float8WeightOnlyConfig()
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=TorchAoConfig(quant_type=config), device_map="cpu", dtype=torch.bfloat16)
    reference = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if hasattr(weight, "qdata"):
            q = weight.qdata.float()
            zero = getattr(weight, "zero_point", None)
            scale = weight.scale.float()
            n, k = q.shape
            groups = scale.numel() // n
            q = q.reshape(n, groups, k // groups)
            q = q if zero is None else q - zero.float().reshape(n, groups, 1)
            reference[name] = (q * scale.reshape(n, groups, 1)).reshape(n, k)
    model.save_pretrained(out, safe_serialization=True)
    torch.save(reference, os.path.join(out, "torchao-reference.pt"))
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


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
    return [
        ("torchao-int8wo", lambda out: torchao_case(base, out, "int8")),
        ("torchao-int8wo-g128", lambda out: torchao_case(base, out, "int8", 128)),
        ("torchao-float8wo", lambda out: torchao_case(base, out, "float8")),
        ("quanto-int8", lambda out: quanto_case(base, out, "qint8")),
        ("quanto-int4", lambda out: quanto_case(base, out, "qint4")),
        ("quanto-int2", lambda out: quanto_case(base, out, "qint2")),
        ("quanto-float8", lambda out: quanto_case(base, out, "qfloat8")),
        ("quark-int4-g128", lambda out: quark_int_case(base, out, 4, "per_group", 128, False, True)),
        ("quark-int4-g64-sym", lambda out: quark_int_case(base, out, 4, "per_group", 64, True, True)),
        ("quark-int4-g128-f32scale", lambda out: quark_int_case(base, out, 4, "per_group", 128, False, False)),
        ("quark-int8-channel", lambda out: quark_int_case(base, out, 8, "per_channel", None, True, True)),
        ("quark-mxfp6-e3m2", lambda out: quark_mxfp6_case(base, out, "fp6_e3m2", False)),
        ("quark-mxfp6-e2m3", lambda out: quark_mxfp6_case(base, out, "fp6_e2m3", False)),
        ("quark-mx-fp6-e3m2", lambda out: quark_mxfp6_case(base, out, "fp6_e3m2", True)),
    ]


if __name__ == "__main__":
    main()
