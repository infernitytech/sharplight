"""transformers reference (fp32 compute) for a truncated Qwen3.5-MoE checkpoint written by tools/qwen35moe-truncate.py, on its
dequantized weights.

  python -I tools/qwen35moe-truncated-reference.py <dir>

Dequantizes FP8 (block 128x128, weight_scale_inv), GPTQ (4-bit, g_idx; zero point 8 when symmetric, else the stored value plus
one) and ModelOpt NVFP4 (e2m1 codes, e4m3 scale per 16, global weight_scale_2; activations not quantized) in float32, fuses the
per-expert projections into transformers' 3-D expert parameters, prints greedy ids of the prompts and writes reference_logits.json
(prompt ids and last-position logits of prompt 1).
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

d = Path(sys.argv[1])
PROMPTS = ["The capital of France is", "Once upon a time, in a small village by the sea, there lived"]
raw = {}
SYMMETRIC = (json.loads((d / "config.json").read_text()).get("quantization_config") or {}).get("sym", False)
with safe_open(d / "model.safetensors", "pt") as f:
    for k in f.keys():
        raw[k] = f.get_tensor(k)

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def dense(prefix):
    """The float32 weight of module <prefix> (prefix without .weight)."""
    if prefix + ".qweight" in raw:
        q, z, s, g = raw[prefix + ".qweight"], raw[prefix + ".qzeros"], raw[prefix + ".scales"].float(), raw[prefix + ".g_idx"].long()
        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        w = ((q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 15).reshape(-1, q.shape[1])          # [in, out]
        zeros = ((z.unsqueeze(2) >> shifts.view(1, 1, 8)) & 15).reshape(z.shape[0], -1) + 1  # [groups, out]
        if SYMMETRIC:
            zeros = torch.full_like(zeros, 8)  # symmetric GPTQ: the zero point is 2^(bits-1) whatever qzeros stores (as vLLM's kernels)
        return ((w - zeros[g]).float() * s[g]).t().contiguous()
    w = raw[prefix + ".weight"]
    if prefix + ".weight_scale_inv" in raw:
        s = raw[prefix + ".weight_scale_inv"].float()
        r, c = w.shape
        return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:r, :c]
    if prefix + ".weight_scale_2" in raw:
        codes = torch.stack([w & 15, w >> 4], dim=-1).reshape(w.shape[0], -1).long()
        scale = raw[prefix + ".weight_scale"].float().repeat_interleave(16, 1)
        return E2M1[codes] * scale * raw[prefix + ".weight_scale_2"].float()
    return w.float()


config = Qwen3_5MoeTextConfig.from_pretrained(d)
config.quantization_config = None
model = Qwen3_5MoeForCausalLM(config).float().eval()
state = {}
for name in model.state_dict():
    if name.endswith("experts.gate_up_proj"):
        p = name[: -len(".gate_up_proj")]
        state[name] = torch.stack([torch.cat([dense(f"{p}.{e}.gate_proj"), dense(f"{p}.{e}.up_proj")]) for e in range(config.num_experts)])
    elif name.endswith("experts.down_proj"):
        p = name[: -len(".down_proj")]
        state[name] = torch.stack([dense(f"{p}.{e}.down_proj") for e in range(config.num_experts)])
    elif name.endswith(".weight") and name[: -len(".weight")] + ".weight_scale_inv" in raw or name[: -len(".weight")] + ".qweight" in raw \
            or name[: -len(".weight")] + ".weight_scale_2" in raw:
        state[name] = dense(name[: -len(".weight")])
    else:
        state[name] = raw[name].float()
model.load_state_dict(state, strict=True)
del raw
tok = AutoTokenizer.from_pretrained(d)
result = {}
with torch.no_grad():
    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=16, do_sample=False)
        new = out[0, ids.shape[1]:].tolist()
        print(json.dumps({"prompt": prompt, "ids": new, "text": tok.decode(new)}))
        result[prompt] = new
        if i == 0:
            logits = model(ids).logits[0, -1].tolist()
            (d / "reference_logits.json").write_text(json.dumps({"prompt_ids": ids[0].tolist(), "logits": logits}))
(d / "reference_ids.json").write_text(json.dumps(result))
