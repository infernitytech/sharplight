"""Greedy reference token ids of the tiny multi-head-latent-attention checkpoints (transformers, fp32 compute).

Usage: python ref_latent_decoder.py ~/models/hf/tiny-random-deepseek-v3.1 [more dirs...]
Prints the prompt ids and the generated ids per model; FamilyGenerationTests.cs holds the copied results.
"""
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
NEW_TOKENS = 16


def main(paths):
    torch.manual_seed(0)
    for path in paths:
        tokenizer = AutoTokenizer.from_pretrained(path)
        config = AutoConfig.from_pretrained(path)
        rope = getattr(config, "rope_parameters", None)
        if config.model_type == "minicpm3" and rope is not None and "factor" not in rope:
            # transformers' MiniCPM3 reads rope_parameters["factor"] for every non-default rope type; longrope has none.
            rope["factor"] = 1.0
        model = AutoModelForCausalLM.from_pretrained(path, config=config, torch_dtype=torch.float32)
        model.eval()
        ids = tokenizer(PROMPT, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False, num_beams=1)
        print(path)
        print("  prompt   ", ids[0].tolist())
        print("  generated", out[0, ids.shape[1]:].tolist())


if __name__ == "__main__":
    main(sys.argv[1:])
