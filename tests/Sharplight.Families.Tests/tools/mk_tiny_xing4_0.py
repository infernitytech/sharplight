"""Writes tiny-random-xing4-0: a Xing4.0 checkpoint made with the model's remote code, plus its reference.

Usage: python mk_tiny_xing4_0.py <models/hf directory> <remote code directory> [--unclamped]

The remote code directory holds modeling_xing4_0.py and configuration_xing4_0.py of XingChen-AGI/Xing4.0-29B-A4B (no transformers class
exists); they are copied next to the checkpoint and loaded with trust_remote_code. The tokenizer (SentencePiece, which llama.cpp's converter
needs for this architecture) comes from tiny-random-MixtralForCausalLM. The tiny model has DeepSeek-V3's latent attention with a query
LoRA and interleaved rope dimensions, YaRN (factor 4 over 64 positions, mscale_all_dim 1), a dense first layer and two MoE layers with
sigmoid routing, selection biases and a shared expert, all on 4 hyper-connection streams with Xing's mixing (clamped logits, 20 Sinkhorn
iterations, row-wise mixing) averaged before the output norm. The first layer's attention module gets a large mixing scale, so its
logits reach the ±30 clamp; --unclamped writes tiny-random-xing4-0-unclamped instead, whose logits stay inside it (llama.cpp 8ca5a80 runs
the mixing in a fused kernel without the clamp). Greedy ids and prompt logits go to <out>/reference_logits.json and stdout.
"""
import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

root, code = os.path.expanduser(sys.argv[1]), sys.argv[2]
unclamped = "--unclamped" in sys.argv[3:]
out = os.path.join(root, "tiny-random-xing4-0-unclamped" if unclamped else "tiny-random-xing4-0")
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"

os.makedirs(out, exist_ok=True)
for f in ["modeling_xing4_0.py", "configuration_xing4_0.py"]:
    shutil.copy(os.path.join(code, f), out)
config = dict(
    architectures=["Xing4_0ForCausalLM"], model_type="xing4_0",
    auto_map={"AutoConfig": "configuration_xing4_0.Xing4_0Config", "AutoModelForCausalLM": "modeling_xing4_0.Xing4_0ForCausalLM"},
    vocab_size=32000, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=3, num_nextn_predict_layers=0,
    num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2, routed_scaling_factor=2.0,
    kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=16, v_head_dim=16, qk_nope_head_dim=16, topk_method="noaux_tc", n_group=1, topk_group=1,
    first_k_dense_replace=1, norm_topk_prob=True, scoring_func="sigmoid", hidden_act="silu", max_position_embeddings=256, rms_norm_eps=1e-6,
    bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, rope_theta=10000.0, rope_interleave=True,
    rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0,
                  "mscale_all_dim": 1.0},
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, mhc_h_res_clamp_min=-30, mhc_h_res_clamp_max=30, dtype="float32")
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
tokenizer_dir = os.path.join(root, "tiny-random-MixtralForCausalLM")
for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), out)

# The remote code's _init_weights names the hyper-connection parameters fn / base / scale (they are hc_fn / hc_base / hc_scale), so it
# skips those modules (they are set below); every other module, the rotary buffers included, keeps its initializer.
model_class = get_class_from_dynamic_module("modeling_xing4_0.Xing4_0ForCausalLM", out)
remote = sys.modules[model_class.__module__]
init_weights = remote.Xing4_0PreTrainedModel._init_weights
remote.Xing4_0PreTrainedModel._init_weights = (
    lambda self, module: None if isinstance(module, remote.Xing4_0HyperConnection) else init_weights(self, module))

torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(out, trust_remote_code=True), trust_remote_code=True, dtype=torch.float32)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("hc_fn"):
            p.copy_(torch.randn_like(p) * 1.5 / p.shape[1] ** 0.5)
        elif n.endswith("hc_base"):
            p.copy_(0.5 * torch.randn_like(p))
        elif n.endswith("hc_scale"):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif p.dim() == 1:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * 0.15)
    for n, b in model.named_buffers():
        if n.endswith("e_score_correction_bias"):
            b.copy_(0.05 * torch.randn_like(b))
    if not unclamped:
        model.model.layers[0].attn_hc.hc_scale[2] = 25.0     # mixing logits beyond the clamp
model.save_pretrained(out, safe_serialization=True)
model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32).eval()

tokenizer = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
ids = tokenizer(PROMPT, return_tensors="pt").input_ids
generated, margins, prompt_logits = [], [], None
with torch.no_grad():
    for _ in range(24):
        logits = model(input_ids=ids).logits[0, -1]
        if prompt_logits is None:
            prompt_logits = logits.tolist()
        top = torch.topk(logits, 2)
        margins.append(float(top.values[0] - top.values[1]))
        generated.append(int(top.indices[0]))
        ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)
print("prompt ids:", ",".join(map(str, tokenizer(PROMPT).input_ids)))
print("ids:", ",".join(map(str, generated)))
print("min margin: %.5f" % min(margins))
json.dump({"prompt": PROMPT, "prompt_ids": tokenizer(PROMPT).input_ids, "logits": prompt_logits},
          open(os.path.join(out, "reference_logits.json"), "w"))
print("wrote", out)
