#!/usr/bin/env python3
"""PyTorch side of the dequantized-checkpoint parity test (tests/Sharplight.Families.Tests/DequantizedCheckpointParityTests.cs).

  python tools/dequantized-parity.py <mode> <model-dir> [<original-dir>] [--layer0 <file.pt>]

Prints the perplexity of the fixed parity text (transformers, teacher-forced, the whole text in one forward pass) and greedy continuations
of three prompts. Modes:
  plain         an unquantized checkpoint as transformers loads it
  fp_quant      the FP-Quant checkpoint through the real library (transformers + fp_quant + qutlass: FP4 weights AND activations)
  fp_quant-dense  <original-dir> with every projection replaced by the FP-Quant dense weight (tools/fpquant-reference-dense.py math)
  fp_quant-emulated  <original-dir> whose projections compute fp_quant's own activation pseudo-quantization (its Triton kernel:
                rotation x @ H, FP4 rounding, act_global_scale) times the decoded rotated-domain weight e2m1 * scale / weight_global_scale:
                the library's arithmetic with the weights as this importer reads them, without the rotation folded in
  fp_quant-gemm   no model: per projection of layers 0 and 27, random exact FP4 activations (codes with block scale 1, so the
                library's activation quantization is lossless) go through the real QuTLASS FP4 x FP4 GEMM with the checkpoint's weight codes
                and scales (alpha = 1 / weight_global_scale); the same activations un-rotated (x @ H) times the importer's dense weight must
                agree up to bf16 rounding (prints the relative error per module)
  higgs-dense   <original-dir> with the HIGGS dense weights (tools/higgs-reference-dense.py math), in the checkpoint's F16
  vptq-dense    <original-dir> with the VPTQ dense weights (tools/vptq-reference-dense.py math)
  exl2-dense    the EXL2 checkpoint's own config and unquantized tensors with the dense weights (tools/exl2-reference-dense.py math,
                bit-identical to exllamav2's reconstruct)
--layer0 saves the output of decoder layer 0 for the parity text (float32 [tokens, hidden]) to compare modes with each other.
The tools/*-reference-dense.py scripts must be importable from this directory (they are run as modules here, without writing files).
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TEXT = ("The history of astronomy is the story of how people learned to read the sky. Early farmers watched the rising of certain "
        "stars to know when to plant, and sailors steered by the pole star long before anyone understood why it stood still. "
        "Later, careful measurements of planetary motion led to the laws that describe orbits, and telescopes revealed moons, "
        "rings and distant galaxies. Today, spacecraft carry instruments far beyond the atmosphere, where the view is clear. "
        "A small kitchen garden needs little more than sunlight, loose soil and patience. Tomatoes want warmth and a stake to "
        "climb, beans fix their own nitrogen, and a row of marigolds keeps some pests away. Water in the morning, not at noon, "
        "and pull weeds while they are young. By late summer the harvest usually outgrows the family, and neighbours get "
        "baskets of zucchini whether they asked for them or not.")
PROMPTS = ["The capital of France is", "def fibonacci(n):", "The three primary colors are"]
TOOLS = Path(__file__).resolve().parent


def tool_functions(name):
    """The function definitions of a tools/*-reference-dense.py script (everything before its command-line part)."""
    source = (TOOLS / name).read_text()
    marker = "\nargs = sys.argv" if "\nargs = sys.argv" in source else "\nd = Path(sys.argv[1])"
    namespace = {"__name__": "reference"}
    exec(compile(source.split(marker)[0], name, "exec"), namespace)
    return namespace


def fp_quant_dense(ck, name, rotated=False):
    cfg = json.load(open(Path(ck) / "config.json"))["quantization_config"]
    nv = cfg["forward_dtype"] == "nvfp4"
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    fp4 = torch.cat([grid, -grid])
    f = safe_open(str(Path(ck) / "model.safetensors"), "pt")
    q = f.get_tensor(name + ".qweight")
    s = f.get_tensor(name + ".scales")
    gs = f.get_tensor(name + ".weight_global_scale").float()
    H = f.get_tensor(name + ".forward_hadamard_matrix").float()
    codes = torch.stack([q & 15, q >> 4], -1).long().reshape(q.shape[0], -1)
    sc = s.view(torch.float8_e4m3fn).float() if nv else torch.pow(2.0, s.float() - 127)
    block = 16 if nv else 32
    dq = (fp4[codes].reshape(q.shape[0], -1, block) * sc.reshape(q.shape[0], -1, 1)).reshape(q.shape[0], -1) / gs
    if rotated:
        return dq, H, f.get_tensor(name + ".act_global_scale").float(), cfg
    return (dq.reshape(q.shape[0], -1, H.shape[0]) @ H.T).reshape(q.shape[0], -1)


def fp_quant_gemm(ck):
    from fp_quant.module.qutlass_ops import matmul_mxf4_bf16_tn_op, matmul_nvf4_bf16_tn_op
    cfg = json.load(open(Path(ck) / "config.json"))["quantization_config"]
    nv = cfg["forward_dtype"] == "nvfp4"
    block = 16 if nv else 32
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    fp4 = torch.cat([grid, -grid])
    f = safe_open(str(Path(ck) / "model.safetensors"), "pt")
    torch.manual_seed(0)
    worst = 0.0
    for key in sorted(f.keys()):
        if not key.endswith(".qweight") or not (".layers.0." in key or ".layers.27." in key):
            continue
        m = key[: -len(".qweight")]
        wq = f.get_tensor(key).cuda()
        ws = f.get_tensor(m + ".scales").cuda()
        gs = f.get_tensor(m + ".weight_global_scale").float().cuda()
        K = wq.shape[1] * 2
        xq = torch.randint(0, 256, (128, K // 2), dtype=torch.uint8, device="cuda")
        if nv:
            xs = torch.full((128, K // block), 0x38, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
            y = matmul_nvf4_bf16_tn_op(xq, wq, xs, ws, (1.0 / gs).float())
        else:
            xs = torch.full((128, K // block), 127, dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
            y = matmul_mxf4_bf16_tn_op(xq, wq, xs, ws, (1.0 / gs).float())
        xr = fp4[torch.stack([xq.cpu() & 15, xq.cpu() >> 4], -1).long().reshape(128, K)]
        H = f.get_tensor(m + ".forward_hadamard_matrix").float()
        x = (xr.reshape(128, -1, H.shape[0]) @ H.T).reshape(128, K)  # x @ H == xr (H orthonormal)
        dense = fp_quant_dense(ck, m).bfloat16().float()
        mine = x @ dense.T
        err = ((y.float().cpu() - mine).norm() / mine.norm()).item()
        worst = max(worst, err)
        print(m, "relative error %.5f" % err)
    print("worst %.5f" % worst)


argv = sys.argv[1:]
layer0 = None
if "--layer0" in argv:
    layer0 = argv.pop(argv.index("--layer0") + 1)
    argv.remove("--layer0")
mode, ck = argv[0], argv[1]
original = argv[2] if len(argv) > 2 else None
if mode == "fp_quant-gemm":
    fp_quant_gemm(ck)
    sys.exit(0)
if mode in ("plain", "fp_quant"):
    tok = AutoTokenizer.from_pretrained(original or ck)
    model = AutoModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map="cuda")
elif mode == "exl2-dense":
    tok = AutoTokenizer.from_pretrained(ck)
    config = AutoConfig.from_pretrained(ck)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    decode = tool_functions("exl2-reference-dense.py")["decode"]
    state = {}
    for fn in Path(ck).glob("*.safetensors"):
        with safe_open(str(fn), "pt") as f:
            for key in f.keys():
                if key.endswith(".q_groups"):
                    state[key[: -len(".q_groups")] + ".weight"] = decode(f, key[: -len(".q_groups")])
                elif not key.split(".")[-1].startswith("q_"):
                    state[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all("rotary" in k or k == "lm_head.weight" for k in missing), (missing, unexpected)
    if "lm_head.weight" in missing:
        model.tie_weights()
    model = model.cuda()
else:
    tok = AutoTokenizer.from_pretrained(original)
    cfg = json.load(open(Path(ck) / "config.json"))["quantization_config"]
    dtype = torch.float16 if mode == "higgs-dense" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(original, dtype=dtype)
    if mode == "higgs-dense":
        ns = tool_functions("higgs-reference-dense.py")
        f = safe_open(str(Path(ck) / "model.safetensors"), "pt")
        H = ns["hadamard"](cfg["hadamard_size"])
    elif mode == "vptq-dense":
        dequant = tool_functions("vptq-reference-dense.py")["dequant"]
        f = safe_open(str(Path(ck) / "model.safetensors"), "pt")
    for name, p in model.named_parameters():
        m = name[: -len(".weight")]
        if mode == "fp_quant-dense" and ".layers." in name and name.endswith("_proj.weight"):
            p.data = fp_quant_dense(ck, m).to(dtype)
        elif mode == "fp_quant-emulated" and ".layers." in name and name.endswith("_proj.weight"):
            from fp_quant.module.pseudoquant_linear_fns import forward_pseudoquantize
            from fp_quant.utils import FPQuantDtype
            dq, H, ags, qcfg = fp_quant_dense(ck, m, rotated=True)
            linear = model.get_submodule(m)
            linear.weight.data = dq.to(dtype)

            def emulate(x, linear=linear, H=H.cuda().to(dtype), ags=ags.cuda(), kind=FPQuantDtype(qcfg["forward_dtype"]), method=qcfg["forward_method"]):
                xq, _ = forward_pseudoquantize(x.flatten(end_dim=-2).contiguous(), H, ags, kind, method)
                return torch.nn.functional.linear(xq.to(x.dtype), linear.weight).unflatten(0, x.shape[:-1])

            linear.forward = emulate
        elif mode == "higgs-dense" and m + ".tables2" in f.keys():
            Q = f.get_tensor(m + ".weight")
            S = f.get_tensor(m + ".scales")
            grid = f.get_tensor(m + ".tables2").float().reshape(-1, 2)
            tid = cfg["tune_metadata"][m]["template_id"]
            W = ns["unpack4"](Q, 64 if tid % 48 < 16 else 32)
            K, N = W.shape
            code = (W[0::2] * 16 + W[1::2]).T
            h = grid[code].reshape(N, K) * S.float().repeat_interleave(K // S.shape[1], 1)
            p.data = (h.reshape(N, -1, H.shape[0]) @ H).reshape(N, K).to(S.dtype).to(dtype)
        elif mode == "vptq-dense" and m in cfg["config_for_layers"]:
            p.data = dequant(f, m, cfg["config_for_layers"][m]).to(dtype)
    model = model.cuda()
model.eval()
with torch.no_grad():
    ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
    result = model(ids, output_hidden_states=True)
    nll = torch.nn.functional.cross_entropy(result.logits[0, :-1].float(), ids[0, 1:])
    if layer0:
        torch.save(result.hidden_states[1][0].float().cpu(), layer0)
    print(mode, Path(ck).name, "tokens", ids.shape[1], "perplexity %.3f" % nll.exp().item())
    for prompt in PROMPTS:
        x = tok(prompt, return_tensors="pt").input_ids.cuda()
        print(repr(tok.decode(model.generate(x, max_new_tokens=24, do_sample=False)[0])))
