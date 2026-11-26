"""Write a dequantized copy of a quantized Hugging Face checkpoint, for references that run transformers on the exact weights.

    python -I tools/dequantize-hf-checkpoint.py <checkpoint dir> <out dir> [--gguf-py <llama.cpp gguf-py dir>]
        [--qkv-slices N:Q:K:V] [--rename OLD=NEW ...]

Every quantized weight is decoded with the arithmetic of the reference loaders and written in BF16 where that is exact, else F32; the
quantization_config is removed from config.json (top level and text_config) and hf_quant_config.json is not copied. Read:
  - FP8 E4M3 X.weight with X.weight_scale_inv / X.weight_scale (F32, BF16, F8_E8M0 or U8 exponent bytes) or X.scale (F8_E8M0), per tensor,
    per row or per block (quantization_config.weight_block_size, also scales tiled per equal slice of the rows, as MiMo-V2.6's pre-sharded
    fused qkv_proj: vLLM mimo_v2 _shard_fp8_qkv_proj): value = code * scale (vLLM dequant_mxfp8_to_bf16 / block FP8).
  - MXFP4: U8 / I8 X.weight packed two E2M1 codes per byte (low nibble first) with a U8 / F8_E8M0 scale per 32 values.
  - NVFP4 compressed-tensors (X.weight_packed, F8_E4M3 X.weight_scale per 16, X.weight_global_scale): code * scale / global_scale.
  - NVFP4 ModelOpt (U8 X.weight, F8_E4M3 X.weight_scale per 16, X.weight_scale_2): code * scale * scale_2.
  - ggml blocks packed in safetensors (ModelOpt MIXED_PRECISION "packing": "ggml"; U8 [N, K / block, bytes]): decoded by gguf-py's
    reference dequantizers (gguf-py from --gguf-py).
Activation and KV-cache calibration scales are dropped. --qkv-slices reorders every fused qkv_proj stored as N slices [Q_i | K_i | V_i] of
Q, K and V rows each (MiMo-V2.6-Pro's FP8 release, as vLLM's mimo_v2 loader reads it) into [Q | K | V], the layout transformers' code splits. --rename replaces a part of every tensor name (Nemotron-3.5-Bolt's remote code
names its decoder `model`, the release stores it as `backbone`).
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
qkv_slices = [int(v) for v in sys.argv[sys.argv.index("--qkv-slices") + 1].split(":")] if "--qkv-slices" in sys.argv else None
if "--gguf-py" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--gguf-py") + 1])
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("model.safetensors.index.json", "hf_quant_config.json"):
        shutil.copy(f, out / f.name)
config = json.loads((src / "config.json").read_text())
quant = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config") or {}
block = quant.get("weight_block_size")
ggml_groups = {}
for group in (quant.get("config_groups") or {}).values():
    if group.get("packing") == "ggml":
        ggml_groups[group["quant_algo"]] = (group["group_size"], group["block_payload_bytes"])
for section in [config, config.get("text_config", {})]:
    section.pop("quantization_config", None)
(out / "config.json").write_text(json.dumps(config, indent=2))

tensors = {}
for f in sorted(src.glob("*.safetensors")):
    h = safe_open(f, framework="pt")
    for k in h.keys():
        tensors[k] = (h, k)


def load(name):
    h, k = tensors[name]
    return h.get_tensor(k)


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
IGNORED = (".input_scale", ".input_global_scale", ".k_scale", ".v_scale", ".q_scale", ".prob_scale", ".output_scale", ".input_zero_point")


def exponent(scale):
    return torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127)


def fp4(codes):
    b = codes.view(torch.uint8).to(torch.int64)
    return torch.stack([E2M1[b & 15], E2M1[b >> 4]], dim=-1).flatten(-2)


def scale_rows(scale, rows, block_rows):
    """Row index map of a block-scale tensor (also tiled per equal slice of the rows)."""
    srows = scale.shape[-2]
    if -(-rows // block_rows) == srows:
        return torch.arange(rows) // block_rows
    for chunks in range(2, srows + 1):
        if rows % chunks == 0 and chunks * -(-(rows // chunks) // block_rows) == srows:
            c = rows // chunks
            r = torch.arange(rows)
            return (r // c) * -(-c // block_rows) + (r % c) // block_rows
    raise ValueError(f"scale rows {srows} do not tile {rows} rows")


def fp8(weight, scale, name):
    w = weight.to(torch.float32)
    s = exponent(scale) if scale.dtype in (torch.uint8, torch.float8_e8m0fnu) else scale.to(torch.float32)
    if s.numel() == 1:
        return w * s.reshape(())
    if s.dim() == 1 or s.shape[-1] == 1:
        return w * s.reshape(-1, 1)
    rows, cols = w.shape
    br = block[0] if block else -(-rows // s.shape[0])
    bc = block[1] if block else -(-cols // s.shape[1])
    rmap = scale_rows(s, rows, br)
    cmap = torch.arange(cols) // bc
    return w * s[rmap][:, cmap]


out_tensors, kinds = {}, {}
done = set()
for name in sorted(tensors):
    if name in done or name.endswith(IGNORED):
        continue
    t = load(name)
    stem = name.rsplit(".", 1)[0]
    if name.endswith(".weight_packed"):
        scale = load(stem + ".weight_scale").to(torch.float32)
        g = load(stem + ".weight_global_scale").to(torch.float32)
        v = fp4(t).reshape(t.shape[0], -1, 16) * scale.unsqueeze(-1) / g
        out_tensors[stem + ".weight"] = v.reshape(t.shape[0], -1)
        done |= {stem + ".weight_scale", stem + ".weight_global_scale"}
        kinds["nvfp4 (compressed-tensors)"] = kinds.get("nvfp4 (compressed-tensors)", 0) + 1
        continue
    if not name.endswith(".weight"):
        if name.endswith((".weight_scale", ".weight_scale_inv", ".weight_scale_2", ".weight_global_scale")) or (
                name.endswith(".scale") and t.dtype == torch.float8_e8m0fnu and stem + ".weight" in tensors):
            continue
        out_tensors[name] = t
        continue
    scale_name = next((stem + s for s in (".weight_scale_inv", ".weight_scale", ".scale") if stem + s in tensors), None)
    algo = next((a for a in ggml_groups if t.dtype == torch.uint8 and t.dim() == 3 and t.shape[-1] == ggml_groups[a][1]), None)
    if algo is not None and scale_name is None:
        import numpy as np
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        values = dequantize(t.numpy().reshape(t.shape[0], -1), GGMLQuantizationType[algo])
        out_tensors[name] = torch.from_numpy(np.ascontiguousarray(values)).to(torch.float32).reshape(t.shape[0], -1)
        kinds[algo] = kinds.get(algo, 0) + 1
        continue
    if scale_name is None:
        out_tensors[name] = t
        continue
    scale = load(scale_name)
    done.add(scale_name)
    if t.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e4m3fn and stem + ".weight_scale_2" in tensors:
        raise ValueError(f"{name}: FP8 weight with an FP8 scale")
    if t.dtype == torch.float8_e4m3fn:
        out_tensors[name] = fp8(t, scale, name)
        kinds["fp8"] = kinds.get("fp8", 0) + 1
    elif t.dtype in (torch.uint8, torch.int8) and scale.dtype == torch.float8_e4m3fn:   # ModelOpt NVFP4
        s2 = load(stem + ".weight_scale_2").to(torch.float32) if stem + ".weight_scale_2" in tensors else torch.tensor(1.0)
        done.add(stem + ".weight_scale_2")
        v = fp4(t).reshape(*t.shape[:-1], -1, 16) * scale.to(torch.float32).unsqueeze(-1) * s2
        out_tensors[name] = v.reshape(*t.shape[:-1], -1)
        kinds["nvfp4 (modelopt)"] = kinds.get("nvfp4 (modelopt)", 0) + 1
    elif t.dtype in (torch.uint8, torch.int8):                                           # MXFP4
        v = fp4(t).reshape(*t.shape[:-1], -1, 32) * exponent(scale).unsqueeze(-1)
        out_tensors[name] = v.reshape(*t.shape[:-1], -1)
        kinds["mxfp4"] = kinds.get("mxfp4", 0) + 1
    else:
        raise ValueError(f"{name}: {t.dtype} weight with scale {scale_name} ({scale.dtype}) not read")

if qkv_slices:
    n, q, k, v = qkv_slices
    for name in [x for x in out_tensors if x.endswith("qkv_proj.weight")]:
        t = out_tensors[name].reshape(n, q + k + v, -1)
        out_tensors[name] = torch.cat([t[:, :q].flatten(0, 1), t[:, q:q + k].flatten(0, 1), t[:, q + k:].flatten(0, 1)], 0)
for pair in [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--rename"]:
    old, new = pair.split("=", 1)
    out_tensors = {k.replace(old, new): v for k, v in out_tensors.items()}
for name, t in list(out_tensors.items()):
    if t.dtype == torch.float32 and t.dim() >= 2 and torch.equal(t.to(torch.bfloat16).to(torch.float32), t):
        out_tensors[name] = t.to(torch.bfloat16)
    elif t.dtype in (torch.float8_e4m3fn, torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(f"{name}: {t.dtype} left undecoded")
save_file({k: v.contiguous() for k, v in out_tensors.items()}, out / "model.safetensors", metadata={"format": "pt"})
print(f"{len(out_tensors)} tensors, decoded {kinds} -> {out}")
