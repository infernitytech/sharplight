"""Writes tiny-random-nemotron-h-latent: the tiny random Nemotron-H of tiny-random-nemotron-h (Mamba-2, MoE and attention layers,
pattern ME*ME*) with a latent mixture of experts (moe_latent_size 16, as Nemotron 3 Super / Ultra): the routed experts read the
rows projected by fc1_latent_proj and their sum is projected back by fc2_latent_proj, while the router and the shared expert read
the full rows. Prints the transformers greedy ids of the test prompt (float32, 16 tokens).
Convert it with llama.cpp's convert_hf_to_gguf.py --outtype f32 for the GGUF case.

Usage: python mk_tiny_nemotron_h_latent.py <tiny-random-nemotron-h dir> <output dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, NemotronHConfig

source, target = Path(sys.argv[1]), Path(sys.argv[2])
config = json.loads((source / "config.json").read_text())
config["moe_latent_size"] = 16
config["hybrid_override_pattern"] = "ME*ME*"   # no dense MLP layer: llama.cpp's nemotron_h_moe reads every feed-forward layer as MoE
config["time_step_limit"] = (0.0, float("inf"))
for key in ["architectures", "transformers_version", "dtype"]:
    config.pop(key, None)
target.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1234)
model = AutoModelForCausalLM.from_config(NemotronHConfig(**config), dtype=torch.float32).eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if name.endswith("e_score_correction_bias"):
            parameter.copy_(torch.randn_like(parameter) * 0.1)
        elif parameter.dim() >= 2:
            parameter.copy_(torch.randn_like(parameter) * 0.08)
model.save_pretrained(target)
for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]:
    if (source / name).exists():
        shutil.copy(source / name, target / name)

tokenizer = AutoTokenizer.from_pretrained(target)
prompt = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
ids = tokenizer(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=16, do_sample=False)
print(",".join(str(i) for i in out[0, ids.shape[1]:].tolist()))
