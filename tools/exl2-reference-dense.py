#!/usr/bin/env python3
"""Reference dense weights of an ExLlamaV2 (EXL2) checkpoint, from exllamav2 itself (see docs/model-support.md).

  python tools/exl2-reference-dense.py <model-dir> <module-prefix> [<module-prefix> ...]

Loads the model with exllamav2 (built from source against the local CUDA; the GPU is taken from CUDA_VISIBLE_DEVICES) and writes
`ExLlamaV2Linear.get_weight_tensor_dq()` (the `reconstruct` kernel: f16 (code - 2^(bits-1)) * scale, rows un-permuted), transposed to
[out, in], for the matching modules. It also decodes the stored tensors independently in PyTorch (per-column bit streams per group,
scale = f16((q + 1)^2) * f16(q_scale_max / 256), dense row j = stored row q_invperm[j]) and asserts that both agree bit for bit.
exllamav2's MLP loads down_proj without its permutation and reorders the output rows of gate_proj / up_proj by down_proj's q_perm
instead (`ExLlamaV2MLP.load`, `unmap` / `output_map`), so for those three the comparison undoes that reordering; the written
reference is the canonical weight.
Written to <model-dir>/sharplight-reference/dense.safetensors.
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.linear import ExLlamaV2Linear


def decode(f, m):
    qw = f.get_tensor(m + ".q_weight").long() & 0xFFFFFFFF
    qs = f.get_tensor(m + ".q_scale").long() & 0xFFFFFFFF
    qmax = f.get_tensor(m + ".q_scale_max")
    qg = f.get_tensor(m + ".q_groups").long().reshape(-1, 2)
    inv = f.get_tensor(m + ".q_invperm").long()
    K, N = inv.shape[0], qw.shape[1]
    nibbles = torch.stack([(qs >> (4 * i)) & 15 for i in range(8)], -1).reshape(qs.shape[0], N)
    scale = ((nibbles + 1) ** 2).half() * (qmax / 256)[:, None]  # exllamav2 divides q_scale_max by 256 at load
    stored = torch.empty(K, N, dtype=torch.half)
    row = 0
    for g in range(qg.shape[0]):
        bits, start = qg[g, 0].item(), qg[g, 1].item()
        rows = (qg[g + 1, 1].item() - start) * 32 // bits if g + 1 < qg.shape[0] else K - row
        words = qw[start: start + (rows * bits + 31) // 32]
        stream = ((words[:, None, :] >> torch.arange(32)[None, :, None]) & 1).reshape(-1, N)[: rows * bits].reshape(rows, bits, N)
        codes = (stream << torch.arange(bits)[None, :, None]).sum(1)
        stored[row: row + rows] = (codes - (1 << (bits - 1))).half() * scale[g][None, :]
        row += rows
    return stored[inv].T.contiguous()


d = Path(sys.argv[1])
prefixes = sys.argv[2:]
config = ExLlamaV2Config(str(d))
model = ExLlamaV2(config)
model.load()
out = {}
files = list(d.glob("*.safetensors"))
for module in model.modules:
    for sub in [module] + list(getattr(module, "submodules", [])):
        if isinstance(sub, ExLlamaV2Linear) and any(sub.key.startswith(p) for p in prefixes) and sub.q_handle is not None:
            dense = sub.get_weight_tensor_dq().T.contiguous().cpu()
            for fn in files:
                with safe_open(str(fn), "pt") as f:
                    if sub.key + ".q_weight" in f.keys():
                        mine = decode(f, sub.key)
                        mlp = sub.key.rsplit(".", 1)[0]
                        if sub.key.endswith((".gate_proj", ".up_proj")) and mlp + ".down_proj.q_invperm" in f.keys():
                            down_perm = torch.argsort(f.get_tensor(mlp + ".down_proj.q_invperm").long())
                            expect = mine[down_perm]
                        elif sub.key.endswith(".down_proj"):
                            expect = mine[:, torch.argsort(f.get_tensor(sub.key + ".q_invperm").long())]
                        else:
                            expect = mine
                        assert torch.equal(expect.view(torch.int16), dense.view(torch.int16)), sub.key
                        out[sub.key + ".weight"] = mine
(d / "sharplight-reference").mkdir(exist_ok=True)
save_file(out, str(d / "sharplight-reference" / "dense.safetensors"))
print(len(out), "modules, PyTorch decode bit-identical to exllamav2 reconstruct")
