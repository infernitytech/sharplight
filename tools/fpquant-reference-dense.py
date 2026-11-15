#!/usr/bin/env python3
"""Reference dense weights of an FP-Quant checkpoint (IST-DASLab fp_quant 0.3 semantics, see docs/model-support.md).

  python tools/fpquant-reference-dense.py <model-dir> <module-prefix> [<module-prefix> ...]

Per module: dq = e2m1(nibble, low nibble first) * scale / weight_global_scale (MXFP4: scale = 2^(e8m0 - 127) per 32 values,
NVFP4: e4m3 per 16 values) in the rotated domain, then W[o, g, i] = sum_j H[i, j] dq[o, g, j] (H = forward_hadamard_matrix; the library
rotates activations as x_rot[g, j] = sum_i x[g, i] H[i, j], so the rotation folds into the weight), computed in float32 and rounded to
bf16.
Written to <model-dir>/sharplight-reference/dense.safetensors.
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

d = Path(sys.argv[1])
prefixes = sys.argv[2:]
cfg = json.load(open(d / "config.json"))["quantization_config"]
nv = cfg["forward_dtype"] == "nvfp4"
grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
fp4 = torch.cat([grid, -grid])
out = {}
with safe_open(str(d / "model.safetensors"), "pt") as f:
    for key in f.keys():
        if not key.endswith(".qweight") or not any(key.startswith(p) for p in prefixes):
            continue
        m = key[: -len(".qweight")]
        q = f.get_tensor(key)
        s = f.get_tensor(m + ".scales")
        gs = f.get_tensor(m + ".weight_global_scale").float()
        H = f.get_tensor(m + ".forward_hadamard_matrix").float()
        codes = torch.stack([q & 15, q >> 4], -1).long().reshape(q.shape[0], -1)
        vals = fp4[codes]
        sc = s.view(torch.float8_e4m3fn).float() if nv else torch.pow(2.0, s.float() - 127)
        block = 16 if nv else 32
        dq = (vals.reshape(q.shape[0], -1, block) * sc.reshape(q.shape[0], -1, 1)).reshape(q.shape[0], -1) / gs
        g = H.shape[0]
        w = (dq.reshape(q.shape[0], -1, g) @ H.T).reshape(q.shape[0], -1)
        out[m + ".weight"] = w.bfloat16().contiguous()
(d / "sharplight-reference").mkdir(exist_ok=True)
save_file(out, str(d / "sharplight-reference" / "dense.safetensors"))
print(len(out), "modules")
