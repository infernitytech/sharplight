"""Wraps the tiny latent Nemotron-H checkpoint (tools/mk_tiny_nemotron_h_latent.py) as the Nemotron 3.5 Super-VL / Bolt release lays it out
(model_type nemotron_h_omni): the language model's tensors below `language_model.`, its config as `llm_config`, plus a stand-in vision tower and
projector tensor that the text import must skip. With --puzzle the language model becomes a heterogeneous ("Puzzle", Bolt) one written the
way Bolt's config describes it (llm_config.model_type nemotron_h_puzzle, per-layer `block_configs`): the second MoE layer keeps the first 6 of
its 8 experts and selects 3, the first selects 2 — the layers of tools/mk_tiny_nemotron_h_puzzle.py, so llama.cpp's text for that GGUF is the
reference.

Usage: python mk_tiny_nemotron_h_omni.py <tiny-random-nemotron-h-latent dir> <out dir> [--puzzle]
"""
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

source, target = sys.argv[1], sys.argv[2]
puzzle = "--puzzle" in sys.argv[3:]
os.makedirs(target, exist_ok=True)
for name in os.listdir(source):
    if name not in ("config.json", "model.safetensors") and os.path.isfile(os.path.join(source, name)):
        shutil.copy(os.path.join(source, name), os.path.join(target, name))

llm = json.load(open(os.path.join(source, "config.json")))
state = load_file(os.path.join(source, "model.safetensors"))
moe_layers = [i for i, kind in enumerate(llm["layers_block_type"]) if kind == "moe"]
if puzzle:
    kept = {moe_layers[1]: 6}
    used = {moe_layers[0]: 2, moe_layers[1]: 3}
    for name in list(state):
        m = re.match(r"backbone\.layers\.(\d+)\.mixer\.(experts\.(\d+)\.|gate\.)", name)
        if not m or int(m.group(1)) not in kept:
            continue
        layer, keep = int(m.group(1)), kept[int(m.group(1))]
        if m.group(3) is not None and int(m.group(3)) >= keep:
            del state[name]
        elif m.group(2) == "gate.":
            state[name] = state[name][:keep].contiguous()   # router rows / selection bias of the kept experts
    llm["model_type"] = "nemotron_h_puzzle"
    llm["architectures"] = ["NemotronHPuzzleForCausalLM"]
    llm["block_configs"] = [
        {"block_type": "moe", "moe_intermediate_size": llm["moe_intermediate_size"], "n_routed_experts": kept.get(i, llm["n_routed_experts"]),
         "num_experts_per_tok": used[i]} if kind == "moe" else {"block_type": {"linear_attention": "mamba", "full_attention": "attention"}.get(kind, kind)}
        for i, kind in enumerate(llm["layers_block_type"])
    ]
    for key in ("layers_block_type", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size"):
        llm.pop(key)

state = {"language_model." + ("lm_head.weight" if k == "lm_head.weight" else k): v for k, v in state.items()}
torch.manual_seed(0)
state["vision_model.radio_model.model.blocks.0.attn.qkv.weight"] = torch.randn(48, 16)   # stand-ins the text import skips
state["mlp1.1.weight"] = torch.randn(64, 16)
save_file(state, os.path.join(target, "model.safetensors"), metadata={"format": "pt"})
config = {
    "architectures": ["NemotronH_Omni_Reasoning_V3"],
    "model_type": "nemotron_h_omni",
    "llm_config": llm,
    "vision_config": {"model_type": "radio", "hidden_size": 16},
    "tie_word_embeddings": False,
}
with open(os.path.join(target, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("wrote", target, "puzzle" if puzzle else "")
