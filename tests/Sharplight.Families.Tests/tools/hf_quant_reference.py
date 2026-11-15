"""
Reference dequantized weights and logits for the quantized Hugging Face checkpoints of QuantizedCheckpointTests
(the tiny ones of mk_tiny_quantized.py and small real Hub checkpoints):

    python hf_quant_reference.py <checkpoint dir> [...]

Writes <checkpoint dir>/sharplight-reference/reference.safetensors (a subdirectory, so the checkpoint's own files stay as they are) with
  dequant.<module>  the weight of every quantized linear (tiny models) or of the linears of layer 0 (real models), dequantized by the
                    quantization library's own code, in the dtype that code returns:
                    compressed-tensors  the format's compressor (BaseCompressor registry) .decompress
                    ModelOpt NVFP4      modelopt NVFP4QTensor.dequantize (f32)
                    AutoAWQ GEMM        awq.utils.packing_utils.dequantize_gemm (f16)
                    AutoAWQ GEMV        no Python dequantizer exists (CUDA kernel only): unpacked here following
                                        awq/modules/linear/gemv.py from_linear, (q - z) * s in f16 as dequantize_gemm computes
                    bitsandbytes        bitsandbytes.functional.dequantize_4bit (f32 output) / int8_vectorwise_dequant
                    GPTQ                gptqmodel does not install here: unpacked following gptqmodel's dequantize_weight
                                        (nn_modules/qlinear: shifts along K per column, v1 zeros + 1), scales[g_idx] * (q - z) in f16
                    Quark MXFP4         quark's extension does not build here: Pack_fp4.unpack transcribed (f32)
                    MLX                 mlx.core.dequantize (CPU build of MLX, BF16)
  ids               the prompt token ids (I32)
  logits            transformers' f32 forward of the model with the dequantized weights (activations not quantized), [T, V]
                    for the tiny models, the last position only for real ones
Run with torch, transformers, safetensors, compressed-tensors, nvidia-modelopt, autoawq and bitsandbytes (a CUDA GPU for bitsandbytes).
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"


def tensors(directory):
    result = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(directory, name), "pt") as f:
                for key in f.keys():
                    result[key] = f.get_tensor(key)
    return result


def compressed_tensors(state, quant):
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    group = next(iter(quant["config_groups"].values()))
    scheme = QuantizationScheme(targets=group["targets"], weights=QuantizationArgs(**group["weights"]))
    compressor = BaseCompressor.get_value_from_registry(group.get("format") or quant["format"])
    stems = {k[:-len(".weight_scale")] for k in state if k.endswith(".weight_scale")}
    for stem in sorted(stems):
        local = {k[len(stem) + 1:]: v for k, v in state.items() if k.startswith(stem + ".") and k.count(".") == stem.count(".") + 1}
        yield stem, compressor.decompress(local, scheme)["weight"], set(stem + "." + k for k in local)


def modelopt_nvfp4(state):
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    for key in sorted(k for k in state if k.endswith(".weight_scale_2")):
        stem = key[:-len(".weight_scale_2")]
        packed = state[stem + ".weight"]
        shape = torch.Size([packed.shape[0], packed.shape[1] * 2])
        weight = NVFP4QTensor(shape, torch.float32, packed).dequantize(
            dtype=torch.float32, scale=state[stem + ".weight_scale"], double_scale=state[key], block_sizes={-1: 16})
        yield stem, weight, {stem + ".weight", stem + ".weight_scale", key, stem + ".input_scale"}


def awq(state, quant):
    from awq.utils.packing_utils import dequantize_gemm
    group = quant["group_size"]
    for key in sorted(k for k in state if k.endswith(".qweight")):
        stem = key[:-len(".qweight")]
        qweight, qzeros, scales = state[key], state[stem + ".qzeros"], state[stem + ".scales"]
        if quant.get("version", "gemm").lower() == "gemm":
            weight = dequantize_gemm(qweight, qzeros, scales, 4, group).T
        else:
            n, k = qweight.shape[0], qweight.shape[1] * 8
            shifts = torch.arange(8, dtype=torch.int32) * 4
            codes = ((qweight[:, :, None] >> shifts) & 0xF).reshape(n, k)
            zeros = ((qzeros[:, :, None] >> shifts) & 0xF).reshape(n, -1)[:, :k // group]
            weight = (codes - zeros.repeat_interleave(group, dim=1)) * scales[:, :k // group].repeat_interleave(group, dim=1)
        yield stem, weight, {key, stem + ".qzeros", stem + ".scales"}


def gptq(state, quant):
    bits = quant["bits"]
    v1 = quant.get("checkpoint_format", "gptq") == "gptq"
    for key in sorted(k for k in state if k.endswith(".qweight")):
        stem = key[:-len(".qweight")]
        qweight, qzeros, scales = state[key], state[stem + ".qzeros"], state[stem + ".scales"]
        k, n = qweight.shape[0] * 32 // bits, qweight.shape[1]
        g_idx = state.get(stem + ".g_idx", torch.arange(k, dtype=torch.int32) // (k if quant["group_size"] == -1 else quant["group_size"]))

        def unpack(words, count):   # a little-endian bit stream along dim 0
            w = words.to(torch.int64) & 0xFFFFFFFF
            out = torch.empty(count, words.shape[1], dtype=torch.int64)
            for i in range(count):
                position = i * bits
                word, offset = position // 32, position % 32
                value = w[word] >> offset
                if offset + bits > 32:
                    value = value | (w[word + 1] << (32 - offset))
                out[i] = value & ((1 << bits) - 1)
            return out

        codes = unpack(qweight, k)                                     # [K, N]
        zeros = unpack(qzeros.T.contiguous(), n).T                     # [G, N]
        if v1:
            zeros = zeros + 1
        weight = (scales[g_idx.long()] * (codes - zeros[g_idx.long()]).to(torch.float16)).T
        yield stem, weight, {key, stem + ".qzeros", stem + ".scales", stem + ".g_idx"}


def bitsandbytes(state):
    import bitsandbytes.functional as F
    for key in sorted(k for k in state if ".quant_state.bitsandbytes__" in k):
        stem = key[:key.index(".quant_state.")][:-len(".weight")]
        parts = {k[len(stem) + len(".weight."):]: v for k, v in state.items() if k.startswith(stem + ".weight.")}
        used = {stem + ".weight"} | {stem + ".weight." + k for k in parts}
        quant_state = F.QuantState.from_dict(dict(parts), device=torch.device("cuda"))
        quant_state.dtype = torch.float32
        weight = F.dequantize_4bit(state[stem + ".weight"].cuda(), quant_state).cpu()
        yield stem, weight, used
    for key in sorted(k for k in state if k.endswith(".SCB")):
        stem = key[:-len(".SCB")]
        weight = F.int8_vectorwise_dequant(state[stem + ".weight"], state[key].float()[:, None])
        yield stem, weight, {stem + ".weight", key, stem + ".weight_format"}


def quark_mxfp4(state):
    """Quark's Pack_fp4.unpack (quark/torch/utils/pack.py: low nibble first, E2M1 bits placed in an f32) times 2^(e - 127)."""
    for key in sorted(k for k in state if k.endswith(".weight_scale")):
        stem = key[:-len(".weight_scale")]
        packed = state[stem + ".weight"]
        unpacked = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.int32)
        unpacked[:, 0::2] = (packed & 0x0F).to(torch.int32)
        unpacked[:, 1::2] = (packed >> 4).to(torch.int32)
        values = (((unpacked & 0x07) << 22) | ((unpacked & 0x08) << 28)).view(torch.float32) * (2.0 ** 126)
        scale = torch.exp2(state[key].to(torch.float32) - 127).repeat_interleave(32, dim=1)
        yield stem, values * scale, {stem + ".weight", key}


def mlx_affine(state, quant):
    """mlx.core.dequantize (CPU build) over f32 copies of the BF16 scales and biases, rounded once to BF16.

    mx.dequantize in BF16 rounds scale * q and then + bias separately; MLX's quantized matmul accumulates in f32 instead,
    so the f32 dequantization is the one inference sees."""
    import mlx.core as mx
    import numpy as np
    for key in sorted(k for k in state if k.endswith(".scales")):
        stem = key[:-len(".scales")]
        to_mx = lambda t: mx.array(t.float().numpy())
        q = mx.array(state[stem + ".weight"].numpy())
        weight = mx.dequantize(q, to_mx(state[key]), to_mx(state[stem + ".biases"]), group_size=quant["group_size"], bits=quant["bits"])
        yield stem, torch.from_numpy(np.array(weight.astype(mx.float32))).to(torch.bfloat16), {stem + ".weight", key, stem + ".biases"}


def dequantized(directory, config):
    state = tensors(directory)
    quant = config.get("quantization_config") or {}
    method = quant.get("quant_method")
    if method == "compressed-tensors":
        modules = compressed_tensors(state, quant)
    elif method == "modelopt" or any(k.endswith(".weight_scale_2") for k in state):
        modules = modelopt_nvfp4(state)
    elif method == "awq":
        modules = awq(state, quant)
    elif method == "gptq":
        modules = gptq(state, quant)
    elif method == "bitsandbytes":
        modules = bitsandbytes(state)
    elif method == "quark":
        modules = quark_mxfp4(state)
    elif method is None and "quantization" in config:
        modules = mlx_affine(state, config["quantization"])
    else:
        raise SystemExit(f"{directory}: unknown quantization {method}")
    weights, consumed = {}, set()
    for stem, weight, used in modules:
        weights[stem] = weight
        consumed |= used
    dense = {k: v for k, v in state.items() if k not in consumed and not k.endswith((".input_scale", ".input_global_scale", ".weight_shape", ".k_scale", ".v_scale"))}
    return weights, dense


def main():
    for directory in sys.argv[1:]:
        config = json.load(open(os.path.join(directory, "config.json")))
        weights, dense = dequantized(directory, config)
        model_config = AutoConfig.from_pretrained(directory)
        model_config.quantization_config = None
        if hasattr(model_config, "quantization_config"):
            del model_config.quantization_config
        model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.float32)
        full = {k: v.float() for k, v in dense.items()}
        full.update({stem + ".weight": w.float() for stem, w in weights.items()})
        missing, unexpected = model.load_state_dict(full, strict=False)
        missing = [m for m in missing if not (m == "lm_head.weight" and model_config.tie_word_embeddings)]
        if missing or unexpected:
            raise SystemExit(f"{directory}: missing {missing[:5]}, unexpected {unexpected[:5]}")
        model.tie_weights()
        tiny = model_config.vocab_size <= 4096
        if tiny:
            ids = torch.Generator().manual_seed(5)
            ids = torch.randint(3, model_config.vocab_size, (24,), generator=ids)
        else:
            ids = AutoTokenizer.from_pretrained(directory)(PROMPT, return_tensors="pt").input_ids[0]
        with torch.no_grad():
            logits = model(ids[None]).logits[0]
        out = {"ids": ids.to(torch.int32), "logits": (logits if tiny else logits[-1:]).contiguous()}
        for stem, weight in weights.items():
            if tiny or ".layers.0." in stem:
                out["dequant." + stem] = weight.contiguous()
        os.makedirs(os.path.join(directory, "sharplight-reference"), exist_ok=True)
        save_file(out, os.path.join(directory, "sharplight-reference", "reference.safetensors"))
        print(directory, len(weights), "weights;", {str(w.dtype) for w in weights.values()}, "top-1", int(logits[-1].argmax()))


if __name__ == "__main__":
    main()
