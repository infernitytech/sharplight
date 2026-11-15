#!/usr/bin/env python3
"""Writes the reference dequantization (aqlm 1.1.7 `_dequantize_weight`) of selected modules of an AQLM checkpoint.

  python tools/aqlm-reference-dense.py <model-dir> <module-prefix> [<module-prefix> ...]

The weights of every module whose name starts with a prefix go to <model-dir>/sharplight-reference/dense.safetensors.
"""
import sys
from pathlib import Path

from aqlm.utils import _dequantize_weight, unpack_int_data
from safetensors import safe_open
from safetensors.torch import save_file

d = Path(sys.argv[1])
prefixes = sys.argv[2:]
out = {}
with safe_open(str(d / "model.safetensors"), "pt") as f:
    for key in f.keys():
        if key.endswith(".codes") and any(key.startswith(p) for p in prefixes):
            module = key[: -len(".codes")]
            codebooks = f.get_tensor(module + ".codebooks")
            codes = f.get_tensor(module + ".codes")
            scales = f.get_tensor(module + ".scales")
            out[module + ".weight"] = _dequantize_weight(unpack_int_data(codes, codebooks.shape[1].bit_length() - 1), codebooks, scales).contiguous()
(d / "sharplight-reference").mkdir(exist_ok=True)
save_file(out, str(d / "sharplight-reference" / "dense.safetensors"))
print(len(out), "modules")
