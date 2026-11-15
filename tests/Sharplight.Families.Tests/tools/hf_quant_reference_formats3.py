"""
Reference dequantized weights and logits for the checkpoints of mk_tiny_quantized_formats3.py (same output as hf_quant_reference.py,
which this script extends with the formats of the third quantization round):

    python hf_quant_reference_formats3.py <checkpoint dir> [...]

  AWQ gemv_fast        no Python dequantizer exists (CUDA kernel only): AutoAWQ's pack_intweight (awq/modules/linear/gemv_fast.py)
                       inverted step by step (checked against the packer on random codes), value = q * scale + qzeros in f16
                       (the kernel is one half-precision fma, so one rounding)
  Marlin (GPTQ / AWQ)  hf_quant_reference_marlin.py (vLLM Marlin GEMM on an identity matrix); run it first
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_quant_reference as base  # noqa: E402
import hf_quant_reference_formats as formats  # noqa: E402


def unpack_gemv_fast(qweight):
    """Inverse of awq.modules.linear.gemv_fast.pack_intweight (interleave 4, kstride 64): codes [N, K]."""
    n4, k = qweight.shape
    n = n4 * 4
    words = qweight.numpy().view(np.uint16)
    nibbles = np.stack([(words >> (4 * t)) & 0xF for t in range(4)], axis=-1)            # [N/4, K, 4]
    x = nibbles.reshape(n4, k // 64, 64, 4)
    x = x.reshape(n4, k // 64, 4, 64).transpose(0, 2, 1, 3).reshape(n, k)                 # undo the row interleave
    x = x.reshape(n, k // 32, 4, 2, 4).transpose(0, 1, 2, 4, 3).reshape(n, k)             # undo the [0,2,4,6,1,3,5,7] reorder
    x = x.reshape(n, k // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4).reshape(n, k)             # undo the 4x4x2 transpose
    return torch.from_numpy(x.astype(np.int32))


def check_unpack():
    from awq.modules.linear.gemv_fast import pack_intweight
    codes = torch.randint(0, 16, (64, 128), dtype=torch.int32)
    assert torch.equal(unpack_gemv_fast(pack_intweight(codes, interleave=4, kstride=64)), codes), "gemv_fast unpack does not invert pack_intweight"


def awq_gemv_fast(state, quant):
    check_unpack()
    group = quant["group_size"]
    for key in sorted(k for k in state if k.endswith(".qweight")):
        stem = key[:-len(".qweight")]
        codes = unpack_gemv_fast(state[key])
        n, k = codes.shape
        scales = state[stem + ".scales"].T[:, :k // group].double().repeat_interleave(group, dim=1)       # [N, K]
        zeros = state[stem + ".qzeros"].T[:, :k // group].double().repeat_interleave(group, dim=1)
        yield stem, (codes.double() * scales + zeros).to(torch.float16), {key, stem + ".scales", stem + ".qzeros"}


def marlin(directory, state):
    path = os.path.join(directory, "sharplight-reference", "marlin_weights.safetensors")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run hf_quant_reference_marlin.py first (vllm environment, GPU)")
    with safe_open(path, "pt") as f:
        for stem in sorted(f.keys()):
            yield stem, f.get_tensor(stem), {stem + ".qweight", stem + ".scales", stem + ".g_idx", stem + ".workspace"}


previous = base.dequantized


def dequantized(directory, config):
    quant = config.get("quantization_config") or {}
    state = base.tensors(directory)
    modules = None
    if quant.get("quant_method") == "awq" and quant.get("version") == "gemv_fast":
        modules = awq_gemv_fast(state, quant)
    elif (quant.get("quant_method") == "awq" and quant.get("version") == "marlin") or quant.get("checkpoint_format") == "marlin":
        modules = marlin(directory, state)
    if modules is None:
        return previous(directory, config)
    weights, consumed = {}, set()
    for stem, weight, used in modules:
        weights[stem] = weight
        consumed |= used
    return weights, {k: v for k, v in state.items() if k not in consumed}


base.dequantized = dequantized

if __name__ == "__main__":
    base.main()
