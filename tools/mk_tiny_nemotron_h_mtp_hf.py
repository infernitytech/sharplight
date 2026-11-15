"""Writes tiny-random-nemotron-h-mtp: tiny-random-nemotron-h-latent with a multi-token-prediction head laid out as the Nemotron 3
Ultra / 3.5 checkpoints ship it (num_nextn_predict_layers 1, mtp_layers_block_type ["full_attention", "moe"]): mtp.layers.0 holds the
NextN input fusion (enorm, hnorm, eh_proj) and an attention block, mtp.layers.1 a mixture of experts and the head norm
(final_layernorm). The head's weights are random; the trunk is unchanged, so its greedy ids stay those of the source checkpoint.
llama.cpp's convert_hf_to_gguf.py converts it to the block layout tools/mk_tiny_nemotron_h_mtp.py writes.

Usage: python mk_tiny_nemotron_h_mtp_hf.py <tiny-random-nemotron-h-latent dir> <output dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

source, target = Path(sys.argv[1]), Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
for item in source.iterdir():
    if item.name not in ("config.json", "model.safetensors"):
        shutil.copy(item, target / item.name)

config = json.loads((source / "config.json").read_text())
config["num_nextn_predict_layers"] = 1
config["mtp_layers_block_type"] = ["full_attention", "moe"]
(target / "config.json").write_text(json.dumps(config, indent=2))

tensors = load_file(source / "model.safetensors")
kinds = config["layers_block_type"]
attention, moe = kinds.index("full_attention"), kinds.index("moe")
generator = torch.Generator().manual_seed(1234)
hidden = config["hidden_size"]
for name, tensor in list(tensors.items()):
    for layer, mtp in [(attention, "mtp.layers.0."), (moe, "mtp.layers.1.")]:
        prefix = f"backbone.layers.{layer}."
        if name.startswith(prefix + "mixer."):
            tensors[mtp + name[len(prefix):]] = (torch.randn(tensor.shape, generator=generator) * 0.08).to(tensor.dtype)
for name in ["mtp.layers.0.enorm.weight", "mtp.layers.0.hnorm.weight", "mtp.layers.0.norm.weight", "mtp.layers.1.norm.weight",
             "mtp.layers.1.final_layernorm.weight"]:
    tensors[name] = torch.ones(hidden)
tensors["mtp.layers.0.eh_proj.weight"] = torch.randn((hidden, 2 * hidden), generator=generator) * 0.08
save_file({k: v.contiguous() for k, v in tensors.items()}, target / "model.safetensors", metadata={"format": "pt"})
