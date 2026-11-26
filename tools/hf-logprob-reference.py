"""Top-k next-token log-probabilities from transformers in float32 (references of the truncated-checkpoint cases of LogprobReferenceTests).

    python -I tools/hf-logprob-reference.py <model dir> <prompt> [--top 5] [--class <transformers class>] [--remote] [--greedy N]
        [--mamba-ssm-reference]

Loads the checkpoint (a dequantized copy for quantized releases, tools/dequantize-hf-checkpoint.py) in float32 on the CPU, tokenizes the
prompt with the checkpoint's tokenizer (special tokens added as the tokenizer config says), and prints the prompt ids, the top-k
log-probabilities of the next token as C# tuples, and optionally N greedy ids. --mamba-ssm-reference provides mamba_ssm's gated RMS norm
(mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn, as its torch reference rms_norm_ref computes it) for remote code that imports it
unconditionally (Nemotron-H) and otherwise runs its torch path.
"""
import sys

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer


def option(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    """mamba_ssm rms_norm_ref."""
    dtype = x.dtype
    x, weight = x.float(), weight.float()
    if z is not None and not norm_before_gate:
        x = x * torch.nn.functional.silu(z.float())
    groups = x.reshape(*x.shape[:-1], -1, group_size or x.shape[-1])
    out = (groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + eps)).reshape(x.shape) * weight
    if bias is not None:
        out = out + bias.float()
    if z is not None and norm_before_gate:
        out = out * torch.nn.functional.silu(z.float())
    return out.to(dtype)


if "--mamba-ssm-reference" in sys.argv:
    import importlib.machinery
    import types
    for name in ["mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.layernorm_gated"]:
        sys.modules[name] = types.ModuleType(name)
        sys.modules[name].__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"].rmsnorm_fn = rmsnorm_fn

model_dir, prompt = sys.argv[1], sys.argv[2]
top = int(option("--top", 5))
remote = "--remote" in sys.argv
greedy = int(option("--greedy", 0))
torch.set_grad_enabled(False)
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=remote)
cls = getattr(transformers, option("--class")) if option("--class") else transformers.AutoModelForCausalLM
model = cls.from_pretrained(model_dir, config=config, dtype=torch.float32, trust_remote_code=remote).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=remote)
ids = tokenizer(prompt, return_tensors="pt").input_ids
print("prompt ids", ids[0].tolist())
logits = model(input_ids=ids).logits[0, -1].float()
logp = torch.log_softmax(logits, -1)
values, indices = logp.topk(top)
print("expected [" + ", ".join(f"({i}, {v:.3f})" for i, v in zip(indices.tolist(), values.tolist())) + "]")
if greedy:
    out = model.generate(ids, max_new_tokens=greedy, do_sample=False)
    print("greedy", out[0, ids.shape[1]:].tolist())
