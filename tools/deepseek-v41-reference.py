"""DeepSeek-V4.1 reference logits from the release's own inference code (inference/model.py, engram.py), on the CPU in float32.

    python -I tools/deepseek-v41-reference.py <inference dir> <converted checkpoint dir> <config.json> <prompt> [--top 5] [--quantize-activations]

<inference dir> holds the release's inference/*.py; the checkpoint is the output of its convert.py (--model-parallel 1 --expert-dtype fp4)
and <config.json> its ModelArgs (the release's inference/config.json adjusted to a truncated checkpoint). The release's kernels (tilelang,
CUDA only) are replaced by torch code with the same arithmetic: fp8_gemm / fp4_gemm multiply by the dequantized weights (E4M3 codes times
the E8M0 scale of their 32 x 32 block, E2M1 codes times the E8M0 scale of their 32 values), sparse_attn is the gathered softmax attention
with the per-head sink, hc_split_sinkhorn the kernel's Sinkhorn iterations; the BF16 parameters are widened to float32. The simulated activation quantization of the release (FP8 per
32 values before every quantized GEMM and on the KV cache, FP4 on the indexer and compressed latents) is left out unless
--quantize-activations is given, so the default reference is the model's arithmetic in float32 on the exact weights. Prints the prompt ids
(the tokenizer of the checkpoint, BOS added) and the top-k next-token log-probabilities as C# tuples.
"""
import json
import sys
import types

import torch

FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
E4M3_MAX = 448.0


def e8m0(s):
    return torch.exp2(s.view(torch.uint8).float() - 127)


def quantize_fp8(x, block):
    if not QUANTIZE:
        return x
    shape = x.shape
    v = x.float().reshape(-1, shape[-1] // block, block)
    amax = v.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / E4M3_MAX)))
    return ((v / scale).to(torch.float8_e4m3fn).float() * scale).reshape(shape).to(x.dtype)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    y = quantize_fp8(x, block_size)
    if inplace:
        x.copy_(y)
        return x
    return y, None


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if inplace:
        return x
    return x, None


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    n, k = b.shape
    s = e8m0(b_s) if b_s.dtype in (torch.float8_e8m0fnu, torch.uint8) else b_s.float()
    w = b.float() * s.repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    return (a.float() @ w.T).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    codes = b.view(torch.uint8).long()
    w = torch.stack([FP4[codes & 15], FP4[codes >> 4]], -1).flatten(-2)
    w = w * e8m0(b_s).repeat_interleave(32, 1)
    return (a.float() @ w.T).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    gathered = torch.stack([kv[i][idx[i].clamp(min=0)] for i in range(b)])          # [b, m, topk, d]
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    peak = torch.maximum(scores.amax(-1, keepdim=True), torch.full_like(scores[..., :1], -1e30))
    weights = torch.exp(scores - peak)
    total = weights.sum(-1, keepdim=True) + torch.exp(attn_sink.float()[None, None, :, None] - peak)
    return (torch.einsum("bmht,bmtd->bmhd", weights / total, gathered.float())).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


inference, checkpoint, config_path, prompt = sys.argv[1:5]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 5
QUANTIZE = "--quantize-activations" in sys.argv
kernel = types.ModuleType("kernel")
for f in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
    setattr(kernel, f.__name__, f)
sys.modules["kernel"] = kernel
sys.path.insert(0, inference)
import model as release  # noqa: E402
from safetensors.torch import load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)
args = release.ModelArgs(**json.load(open(config_path)))
args.max_batch_size, args.max_seq_len, args.temperature = 1, 256, 0
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
net = release.Transformer(args, tokenizer)
missing, unexpected = load_model(net, f"{checkpoint}/model0-mp1.safetensors", strict=False)
# the image routing biases exist only with the vision tower enabled
assert not missing and all(k.endswith("gate.bias_vl") for k in unexpected), (missing, unexpected)
for parameter in net.parameters():
    if parameter.dtype == torch.bfloat16:   # the release builds some modules in BF16 explicitly; the reference computes in float32
        parameter.data = parameter.data.float()
ids = tokenizer.encode(prompt, add_special_tokens=True)
print("prompt ids", ids)
_, logits, _ = net(torch.tensor([ids]), 0)
logp = torch.log_softmax(logits[0].float(), -1)
values, indices = logp.topk(top)
print("expected [" + ", ".join(f"({i}, {v:.3f})" for i, v in zip(indices.tolist(), values.tolist())) + "]")
