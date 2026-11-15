#!/usr/bin/env python3
"""Reference dense weights of a HIGGS checkpoint (transformers quant_method higgs, FLUTE-packed; see docs/model-support.md).

  python tools/higgs-reference-dense.py <model-dir> [--original <bf16-model-dir>] <module-prefix> [<module-prefix> ...]

FLUTE (github.com/HanGuo97/flute) has no CPU dequantizer and its kernels are not built for every GPU, so this script carries a
transcription of FLUTE's packer (`flute/utils.py` `_pack_4bit` with `packbits_utils.pack_integer_tensors`: int16 words, four 4-bit
values least significant first) and its exact inverse, and checks the inverse on random codes for both tile widths before use. The
tile width of a module is FLUTE's `TileP` of `tune_metadata[module].template_id` (64 when id % 48 < 16 for 4 bits, else 32; checked
against `qgemm_kernel_raw_generated_configs.pth`). Dequantization: h[n, k] = tables2[code][k % 2] * scales[n, k // group size], then the
orthonormal Sylvester Hadamard matrix of hadamard_size applied per block (FLUTE rotates activations with it before the GEMM),
float32, rounded to the scales' dtype. With --original the relative Frobenius error against the unquantized weights is printed (about
0.088 for 4-bit p=2; a wrong layout gives about 1.4, a wrong normalization about 21).
Written to <model-dir>/sharplight-reference/dense.safetensors.
"""
import glob
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def pack4(W, P):
    K, N = W.shape
    c = W.view(K // 2, 2, N // (4 * P), 4, P).transpose(-1, -2)
    n = torch.zeros_like(c)
    n[:, 0, :, :, 0] = c[:, 1, :, :, 0]
    n[:, 0, :, :, 1] = c[:, 0, :, :, 0]
    n[:, 0, :, :, 2] = c[:, 1, :, :, 1]
    n[:, 0, :, :, 3] = c[:, 0, :, :, 1]
    n[:, 1, :, :, 0] = c[:, 1, :, :, 2]
    n[:, 1, :, :, 1] = c[:, 0, :, :, 2]
    n[:, 1, :, :, 2] = c[:, 1, :, :, 3]
    n[:, 1, :, :, 3] = c[:, 0, :, :, 3]
    flat = n.reshape(-1).long()
    bits = ((flat[:, None] >> torch.arange(4)) & 1).reshape(-1, 16)
    words = (bits << torch.arange(16)).sum(-1)
    words = torch.where(words >= 32768, words - 65536, words).to(torch.int16)
    return words.view(K, -1).T.contiguous()


def unpack4(Q, P):
    n4, K = Q.shape
    q = Q.T.contiguous().to(torch.int32) & 0xFFFF
    vals = torch.stack([(q >> (4 * e)) & 15 for e in range(4)], -1)
    W = torch.empty(K, n4 * 4, dtype=torch.int64)
    k = torch.arange(K)
    c = torch.arange(n4)
    for e in range(4):
        t = 2 * (k % 2) + e // 2
        rows = 2 * (k // 2) + 1 - e % 2
        cols = (c // P)[None, :] * 4 * P + t[:, None] * P + (c % P)[None, :]
        W[rows[:, None].expand(-1, n4), cols] = vals[:, :, e].long()
    return W


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


args = sys.argv[1:]
d = Path(args.pop(0))
original = None
if args and args[0] == "--original":
    args.pop(0)
    original = args.pop(0)
prefixes = args
torch.manual_seed(0)
for P in (32, 64):
    codes = torch.randint(0, 16, (256, 1024), dtype=torch.uint8)
    assert torch.equal(unpack4(pack4(codes, P), P), codes.long()), P
cfg = json.load(open(d / "config.json"))["quantization_config"]
assert cfg.get("bits", 4) == 4 and cfg.get("p", 2) == 2
had = cfg.get("hadamard_size", 512)
H = hadamard(had)
orig = {}
if original:
    for fn in glob.glob(original + "/*.safetensors"):
        with safe_open(fn, "pt") as g:
            for key in g.keys():
                if any(key.startswith(p) for p in prefixes):
                    orig[key] = g.get_tensor(key).float()
out = {}
with safe_open(str(d / "model.safetensors"), "pt") as f:
    for key in f.keys():
        if not key.endswith(".tables2") or not any(key.startswith(p) for p in prefixes):
            continue
        m = key[: -len(".tables2")]
        Q = f.get_tensor(m + ".weight")
        S = f.get_tensor(m + ".scales")
        grid = f.get_tensor(key).float().reshape(-1, 2)
        tid = cfg["tune_metadata"][m]["template_id"]
        W = unpack4(Q, 64 if tid % 48 < 16 else 32)
        K, N = W.shape
        code = (W[0::2] * 16 + W[1::2]).T
        h = grid[code].reshape(N, K) * S.float().repeat_interleave(K // S.shape[1], 1)
        w = (h.reshape(N, -1, had) @ H).reshape(N, K)
        if m + ".weight" in orig:
            w0 = orig[m + ".weight"]
            print(m, "relative error %.4f" % ((w - w0).norm() / w0.norm()).item())
        out[m + ".weight"] = w.to(S.dtype).contiguous()
(d / "sharplight-reference").mkdir(exist_ok=True)
save_file(out, str(d / "sharplight-reference" / "dense.safetensors"))
print(len(out), "modules")
