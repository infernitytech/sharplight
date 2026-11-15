"""
Reference dequantized weights and logits for the checkpoints of mk_tiny_quantized_formats.py (same output as hf_quant_reference.py,
which this script extends with the formats of the second quantization round):

    python hf_quant_reference_formats.py <checkpoint dir> [...]

  MLX FP modes   mlx.core.dequantize(mode="mxfp4" | "nvfp4" | "mxfp8") (CPU build of MLX; the values are exactly BF16)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_quant_reference as base  # noqa: E402


def mlx_float_modes(state, quant):
    import mlx.core as mx
    import numpy as np
    for key in sorted(k for k in state if k.endswith(".scales")):
        stem = key[:-len(".scales")]
        q = mx.array(state[stem + ".weight"].numpy().view(np.uint32))
        scales = mx.array(state[key].numpy())
        weight = mx.dequantize(q, scales, None, group_size=quant["group_size"], bits=quant["bits"], mode=quant["mode"])
        yield stem, torch.from_numpy(np.array(weight.astype(mx.float32))).to(torch.bfloat16), {stem + ".weight", key}


def autoround(directory, state, quant):
    """AutoRound's own dequantizers: the AWQ packing through export_to_awq.utils.dequantize_gemm, the GPTQ packing through its
    torch QuantLinear (auto_round_extension.torch.qlinear_torch_zp) applied to the identity matrix (a product with a one-hot row is
    exact, so the output is the library's dequantized weight in the f16 it computes in)."""
    awq = "awq" in quant.get("packing_format", "")
    bits, group = quant["bits"], quant["group_size"]
    weights, consumed = {}, set()
    for key in sorted(k for k in state if k.endswith(".qweight")):
        stem = key[:-len(".qweight")]
        qweight, qzeros, scales = state[key], state[stem + ".qzeros"], state[stem + ".scales"]
        if awq:
            from auto_round.export.export_to_awq.utils import dequantize_gemm
            weight = dequantize_gemm(qweight, qzeros, scales, bits, group).T
        else:
            from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
            k, n = qweight.shape[0] * 32 // bits, qweight.shape[1]
            layer = QuantLinear(bits, group, k, n, False)
            layer.qweight, layer.qzeros, layer.scales = qweight, qzeros, scales
            if hasattr(layer, "post_init"):
                layer.post_init()
            with torch.no_grad():
                weight = layer(torch.eye(k, dtype=torch.float16)).T
        weights[stem] = weight.to(torch.float16).contiguous()
        consumed |= {key, stem + ".qzeros", stem + ".scales", stem + ".g_idx"}
    return weights, {k: v for k, v in state.items() if k not in consumed}


def hqq(state):
    """hqq's own Quantizer.dequantize (hqq 0.2.8) with the scale and zero as stored (f16 / bf16 values) and the arithmetic in f32:
    (unpacked code - zero) * scale. HQQ's default compute dtype (f16) rounds (code - zero) and the product to f16 as well; the f32
    result is the exact value those roundings approximate."""
    from hqq.core.quantize import Quantizer
    text = lambda t: "".join(chr(int(c)) for c in t)
    for key in sorted(k for k in state if k.endswith(".W_q")):
        stem = key[:-len(".W_q")]
        meta = {"nbits": int(state[stem + ".nbits"]), "group_size": int(state[stem + ".group_size"]),
                "shape": torch.Size([int(d) for d in state[stem + ".shape"]]), "scale": state[stem + ".scale"].float(),
                "zero": state[stem + ".zero"].float(), "axis": int(state[stem + ".axis"]), "packing": text(state[stem + ".packing"]),
                "view_as_float": False, "compute_dtype": torch.float32}
        if meta["packing"] == "":
            meta["packing"] = None
        weight = Quantizer.dequantize(state[key], meta)
        used = {k for k in state if k.startswith(stem + ".") and k[len(stem) + 1:] != "bias"}
        yield stem, weight, used


def eetq(directory, state):
    """The w8a16 GEMM's dequantization (half(q) * half(scale)) of the unprocessed int8 matrix EETQ's quantizer returned."""
    reference = torch.load(os.path.join(directory, "eetq-reference.pt"))
    weights, consumed = {}, set()
    for stem, weight in reference.items():
        weights[stem] = weight
        consumed |= {stem + ".weight", stem + ".weight_scales"}
    return weights, {k: v for k, v in state.items() if k not in consumed}


original = base.dequantized


def dequantized(directory, config):
    quant = config.get("quantization") or {}
    method = config.get("quantization_config", {}).get("quant_method")
    if method == "eetq":
        return eetq(directory, base.tensors(directory))
    if method == "hqq":
        state = base.tensors(directory)
        weights, consumed = {}, set()
        for stem, weight, used in hqq(state):
            weights[stem] = weight
            consumed |= used
        return weights, {k: v for k, v in state.items() if k not in consumed}
    if config.get("quantization_config", {}).get("quant_method") in ("auto-round", "autoround"):
        return autoround(directory, base.tensors(directory), config["quantization_config"])
    if config.get("quantization_config", {}).get("quant_method") is None and quant.get("mode", "affine") != "affine":
        state = base.tensors(directory)
        weights, consumed = {}, set()
        for stem, weight, used in mlx_float_modes(state, quant):
            weights[stem] = weight
            consumed |= used
        return weights, {k: v for k, v in state.items() if k not in consumed}
    return original(directory, config)


base.dequantized = dequantized

if __name__ == "__main__":
    base.main()
