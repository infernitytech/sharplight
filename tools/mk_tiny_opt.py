"""Tiny random OPT checkpoint (pre-norm, ReLU, learned positions with offset 2) with facebook/opt-125m's slow tokenizer files
(vocab.json + merges.txt, no tokenizer.json), and the transformers fp32 greedy ids of the Families HF prompt.

Usage: python -I mk_tiny_opt.py <facebook/opt-125m dir> <out dir>
"""
import os
import shutil
import sys

import torch
from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"


def main():
    source, out = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    config = OPTConfig(vocab_size=50272, hidden_size=64, num_hidden_layers=2, ffn_dim=128, num_attention_heads=4, max_position_embeddings=128,
                       word_embed_proj_dim=64, do_layer_norm_before=True, activation_function="relu", bos_token_id=2, eos_token_id=2, pad_token_id=1)
    model = OPTForCausalLM(config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.5)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for name in ["vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(os.path.join(source, name), os.path.join(out, name))
    tok = AutoTokenizer.from_pretrained(out)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        res = model.generate(ids, max_new_tokens=24, min_new_tokens=24, do_sample=False, repetition_penalty=None)
    print("prompt ids", ids[0].tolist())
    print("ids", ",".join(str(i) for i in res[0, ids.shape[1]:].tolist()))


if __name__ == "__main__":
    main()
