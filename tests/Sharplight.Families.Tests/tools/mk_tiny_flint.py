"""Builds a tiny random Flint checkpoint (HF format) with real tokenizer files: python mk_tiny_flint.py <outdir>"""
import sys, os, json, shutil, urllib.request, torch
from transformers import FlintConfig, FlintForCausalLM

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
cfg = json.loads(urllib.request.urlopen("https://huggingface.co/RecursiveSI/flint/raw/main/config.json").read())
types = ["full_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                layer_types=types, mlp_only_layers=[0, 1], moe_intermediate_size=32, shared_expert_intermediate_size=48, num_experts=8,
                num_experts_per_tok=3, sliding_window=int(os.environ.get("SW", "7")), no_rope_layer_indices=[0, 1, 5], max_position_embeddings=4096))
cfg.pop("dtype", None); cfg.pop("torch_dtype", None)
config = FlintConfig(**cfg)
model = FlintForCausalLM(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("sinks"):
            p.copy_(torch.randn_like(p) * 2.0 if not os.environ.get("NOSINK") else torch.full_like(p, -80.0))
        elif p.dim() == 1 and "norm" in n:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif n.endswith("gate.weight"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "embed" in n or "lm_head" in n:
            p.copy_(torch.randn_like(p) * 0.5)
        else:
            p.copy_(torch.randn_like(p) * 0.15)
model = model.to(torch.bfloat16)
model.save_pretrained(out, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "special_tokens_map.json"]:
    try:
        urllib.request.urlretrieve("https://huggingface.co/RecursiveSI/flint/resolve/main/" + f, os.path.join(out, f))
    except Exception as e:
        print("skip", f, e)
print("done", sum(p.numel() for p in model.parameters()))
