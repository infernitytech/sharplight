#!/usr/bin/env python3
"""Reference dense weights of a VPTQ checkpoint (quant_method vptq; see docs/model-support.md).

  python tools/vptq-reference-dense.py <model-dir> [--original <model-dir>] <module-prefix> [<module-prefix> ...]

The `vptq` package's PyTorch fallback `vptq.ops.quant_gemm.dequant` with `vptq.utils.pack.unpack_index_tensor` (github.com/microsoft/VPTQ,
main), transcribed operation by operation in the module's dtype (the CUDA package does not build here): gather the centroids by index,
add the residual centroids, prepend the outlier columns, undo the permutation, multiply by weight_scale and add weight_bias. With
--original the relative Frobenius error against the unquantized weights is printed.
Written to <model-dir>/sharplight-reference/dense.safetensors.
"""
import glob
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def unpack_index_tensor(packed_tensor, index_bits, num_elements, res_bits=0, num_res_elements=0):
    total_bits = index_bits + res_bits
    wf = torch.arange(0, 32, 1).view(1, 1, 1, -1)
    out = torch.bitwise_right_shift(torch.unsqueeze(packed_tensor, -1), wf)
    torch.bitwise_and(out, 1, out=out)
    pad_size = (packed_tensor.shape[-1] * 32) % (index_bits * num_elements + res_bits * num_res_elements)
    out = out.reshape(*packed_tensor.shape[:-1], -1)
    if pad_size > 0:
        out = out[..., :-pad_size]
    out = out.reshape(*packed_tensor.shape[:-1], -1, total_bits)
    wf1 = torch.arange(0, total_bits, 1).view(1, 1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)
    unpacked_indices = out.to(torch.uint64).view(torch.int64)
    indices = (unpacked_indices & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
    res_indices = None
    if res_bits > 0:
        res_indices = ((unpacked_indices >> index_bits) & ((1 << index_bits) - 1)).view(torch.uint64).to(torch.int64)
    return indices, res_indices


def dequant(f, m, c):
    get = lambda name: f.get_tensor(m + name) if (m + name) in f.keys() else None
    vector_len, outlier_vector_len = c["vector_lens"][1], c["vector_lens"][0]
    num_centroids, num_outlier_centroids = c["num_centroids"][1], c["num_centroids"][0]
    num_res_centroids = c["num_res_centroids"][1]
    num_codebooks, group_size, outlier_size = c["group_num"], c["group_size"], c["outlier_size"]
    out_features = c["out_features"]
    enable_residual = num_res_centroids > 0
    enable_outlier = outlier_vector_len > 1 and num_outlier_centroids > 0
    padding = (-out_features) % vector_len
    outlier_padding = (-out_features) % outlier_vector_len if enable_outlier else 0
    indices = get(".indices")
    centroids = get(".centroids.weight").view(num_codebooks, num_centroids, vector_len)
    res_indices = get(".res_indices")
    if c.get("is_indice_packed"):
        index_bits = math.ceil(math.log2(num_centroids))
        index_res_bits = math.ceil(math.log2(num_res_centroids)) if enable_residual else 0
        indices, res_indices = unpack_index_tensor(indices.long() & 0xFFFFFFFF, index_bits, group_size, index_res_bits, group_size)
    else:
        indices = indices.view(torch.uint16).to(torch.int64)
        if enable_residual:
            res_indices = res_indices.view(torch.uint16).to(torch.int64)
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, vector_len).reshape(num_codebooks, -1, vector_len)
    selected = torch.gather(centroids, 1, indices).view(num_codebooks, -1, group_size, vector_len)
    qweight = selected.permute(0, 1, 3, 2).reshape(num_codebooks, -1, group_size).permute(1, 0, 2).reshape(-1, num_codebooks * group_size)
    if enable_residual:
        res_centroids = get(".res_centroids.weight").view(num_codebooks, num_res_centroids, vector_len)
        res_indices = res_indices.unsqueeze(-1).expand(-1, -1, -1, vector_len).reshape(num_codebooks, -1, vector_len)
        sel = torch.gather(res_centroids, 1, res_indices).reshape(num_codebooks, -1, group_size, vector_len)
        sel = sel.permute(0, 1, 3, 2).reshape(num_codebooks, -1, group_size).permute(1, 0, 2).reshape(-1, num_codebooks * group_size)
        qweight = qweight + sel
    if padding > 0:
        qweight = qweight[:-padding, :]
    if enable_outlier:
        oc = get(".outlier_centroids.weight").view(1, num_outlier_centroids, outlier_vector_len)
        oi = get(".outlier_indices").view(torch.uint16).to(torch.int64)
        oi = oi.unsqueeze(-1).expand(-1, -1, -1, outlier_vector_len).reshape(1, -1, outlier_vector_len)
        oc = torch.gather(oc, 1, oi).reshape(1, -1, outlier_size, outlier_vector_len).permute(0, 1, 3, 2)
        qo = oc.reshape(-1, outlier_size)
        if outlier_padding > 0:
            qo = qo[:-outlier_padding, ]
        qweight = torch.cat([qo, qweight], dim=1)
    if c.get("enable_perm"):
        invert_perm = torch.argsort(get(".perm").view(torch.uint16).to(torch.int64) if get(".perm").dtype == torch.int16 else get(".perm"))
        qweight = qweight[:, invert_perm]
    if c.get("enable_norm"):
        qweight = qweight * get(".weight_scale") + get(".weight_bias")
    return qweight


args = sys.argv[1:]
d = Path(args.pop(0))
original = None
if args and args[0] == "--original":
    args.pop(0)
    original = args.pop(0)
prefixes = args
layers = json.load(open(d / "config.json"))["quantization_config"]["config_for_layers"]
orig = {}
if original:
    for fn in glob.glob(original + "/*.safetensors"):
        with safe_open(fn, "pt") as g:
            for key in g.keys():
                if any(key.startswith(p) for p in prefixes):
                    orig[key] = g.get_tensor(key).float()
out = {}
with safe_open(str(d / "model.safetensors"), "pt") as f:
    for m, c in layers.items():
        if not any(m.startswith(p) for p in prefixes):
            continue
        w = dequant(f, m, c)
        if m + ".weight" in orig:
            w0 = orig[m + ".weight"]
            print(m, "relative error %.4f" % ((w.float() - w0).norm() / w0.norm()).item())
        out[m + ".weight"] = w.contiguous()
(d / "sharplight-reference").mkdir(exist_ok=True)
save_file(out, str(d / "sharplight-reference" / "dense.safetensors"))
print(len(out), "modules")
