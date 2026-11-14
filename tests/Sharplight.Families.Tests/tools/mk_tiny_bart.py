"""Writes tiny random BART-family checkpoints for the encoder-decoder family tests.

Usage: python mk_tiny_bart.py <tokenizer_dir> <out_root>

<tokenizer_dir> holds a BART byte-level BPE tokenizer (e.g. hf-internal-testing's tiny-random-BartForCausalLM, vocabulary 1024). Writes
<out_root>/tiny-random-bart (BartForConditionalGeneration: post-norm LayerNorms, learned positions with BART's offset of 2, embedding norm,
GELU, final logits bias) and <out_root>/tiny-random-mbart (MBartForConditionalGeneration: pre-norm with final layer norms, embeddings scaled by
sqrt(d_model)), each with reference_embeddings.json (mean-pooled encoder states) for EmbeddingFamilyTests.
"""
import json
import os
import shutil
import sys

import torch
from transformers import BartConfig, BartForConditionalGeneration, MBartConfig, MBartForConditionalGeneration

tokenizer_dir, out_root = sys.argv[1], sys.argv[2]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
COMMON = dict(vocab_size=1024, d_model=64, encoder_layers=2, decoder_layers=3, encoder_attention_heads=4, decoder_attention_heads=4,
              encoder_ffn_dim=128, decoder_ffn_dim=96, max_position_embeddings=256, activation_function="gelu", dropout=0.0,
              attention_dropout=0.0, activation_dropout=0.0, pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
              forced_eos_token_id=None)

VARIANTS = {
    "tiny-random-bart": (61, BartConfig, BartForConditionalGeneration, dict(scale_embedding=False)),
    "tiny-random-mbart": (62, MBartConfig, MBartForConditionalGeneration, dict(scale_embedding=True)),
}

for name, (seed, config_class, model_class, extra) in VARIANTS.items():
    torch.manual_seed(seed)
    config = config_class(**COMMON, **extra)
    config.architectures = [model_class.__name__]
    model = model_class(config).eval()
    with torch.no_grad():
        # Wider weights than the 0.02 init so the tiny model's greedy output depends on its input; non-trivial norms and biases.
        for pname, p in model.named_parameters():
            if p.dim() == 2:
                p.normal_(0.0, 1.0 if "embed" in pname or "shared" in pname else 1.5 / p.shape[1] ** 0.5)
            elif "layer_norm" in pname or "layernorm" in pname:
                p.copy_((1.0 if pname.endswith("weight") else 0.0) + 0.2 * torch.randn_like(p))
            else:
                p.normal_(0.0, 0.1)
        model.final_logits_bias.normal_(0.0, 0.5)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tokenizer_dir, f)):
            shutil.copy(os.path.join(tokenizer_dir, f), out)
    gen = torch.Generator().manual_seed(7)
    inputs = [torch.randint(3, 1000, (n,), generator=gen).tolist() for n in (5, 37, 150)]
    with torch.no_grad():
        embeddings = [model.get_encoder()(input_ids=torch.tensor([ids])).last_hidden_state[0].mean(0).tolist() for ids in inputs]
    with open(os.path.join(out, "reference_embeddings.json"), "w") as f:
        json.dump({"inputs": inputs, "embeddings": embeddings, "pooling": "mean"}, f)
    print("wrote", out)
