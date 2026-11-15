"""
Marlin-format checkpoints (GPTQ checkpoint_format "marlin", AutoAWQ version "marlin"): the library's own dequantization, taken from
vLLM's Marlin GEMM kernel (CUDA) applied to an identity matrix (every product has one non-zero term, so the f16 output is exactly the weight
the kernel dequantizes). Run in the vllm environment with a GPU:

    CUDA_VISIBLE_DEVICES=0 python hf_quant_reference_marlin.py <checkpoint dir> [...]

Writes <checkpoint dir>/sharplight-reference/marlin_weights.safetensors (F16 [N, K] per module, named <module>), which
hf_quant_reference_formats3.py reads in place of its own dequantizer for these checkpoints.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear, marlin_make_workspace_new
    from vllm.scalar_type import scalar_types
    for directory in sys.argv[1:]:
        quant = json.load(open(os.path.join(directory, "config.json")))["quantization_config"]
        bits = quant["bits"]
        wtype = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
        weights = {}
        with safe_open(os.path.join(directory, "model.safetensors"), "pt") as f:
            for key in sorted(k for k in f.keys() if k.endswith(".qweight")):
                stem = key[:-len(".qweight")]
                qweight, scales = f.get_tensor(key).cuda(), f.get_tensor(stem + ".scales").cuda()
                k, n = qweight.shape[0] * 16, scales.shape[1]
                workspace = marlin_make_workspace_new(qweight.device)
                out = apply_gptq_marlin_linear(torch.eye(k, dtype=torch.float16, device="cuda"), qweight, scales, torch.empty(0, device="cuda"),
                                               workspace, wtype, n, k, use_fp32_reduce=True)
                weights[stem] = out.T.contiguous().cpu()
        os.makedirs(os.path.join(directory, "sharplight-reference"), exist_ok=True)
        save_file(weights, os.path.join(directory, "sharplight-reference", "marlin_weights.safetensors"))
        print(directory, len(weights), "weights")


if __name__ == "__main__":
    main()
