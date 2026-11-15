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


EXTRA = {}
original = base.dequantized


def dequantized(directory, config):
    quant = config.get("quantization") or {}
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
