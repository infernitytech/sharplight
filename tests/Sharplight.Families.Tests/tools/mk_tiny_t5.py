"""Writes tiny random T5 checkpoints (T5ForConditionalGeneration) for the encoder-decoder family tests.

Usage: python mk_tiny_t5.py <tokenizer_dir> <out_root>

<tokenizer_dir> holds a T5 SentencePiece tokenizer (e.g. a copy of hf-internal-testing's tiny-random-T5Model, vocabulary 32100). Writes
<out_root>/tiny-random-t5 (original T5: ReLU feed-forward, decoder output scaled by d_model^-0.5 before the tied head, a short bias
saturation distance so long prompts exercise the clamped buckets, more decoder than encoder layers) and <out_root>/tiny-random-t5-v1_1
(T5 v1.1 / Flan-T5: gated tanh-GELU feed-forward, unscaled output, attention width different from d_model). Transformers 5 always ties
T5's LM head to the shared embedding, so neither checkpoint has a separate lm_head (the flan-t5-small GGUF case covers one).
"""
import json
import os
import shutil
import sys

import torch
from transformers import T5Config, T5ForConditionalGeneration

tokenizer_dir, out_root = sys.argv[1], sys.argv[2]
TOKENIZER_FILES = ["spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

# Seeds chosen for distinct greedy tokens with top-two margins well above the f16 rounding of the GPU paths.
VARIANTS = {
    "tiny-random-t5": dict(seed=52, d_model=64, d_kv=16, num_heads=4, d_ff=128, num_layers=2, num_decoder_layers=3,
                           relative_attention_num_buckets=16, relative_attention_max_distance=12, feed_forward_proj="relu",
                           tie_word_embeddings=True),
    "tiny-random-t5-v1_1": dict(seed=69, d_model=64, d_kv=16, num_heads=3, d_ff=96, num_layers=3, num_decoder_layers=2,
                                relative_attention_num_buckets=32, relative_attention_max_distance=128, feed_forward_proj="gated-gelu",
                                tie_word_embeddings=False),
}

for name, spec in VARIANTS.items():
    spec = dict(spec)
    torch.manual_seed(spec.pop("seed"))
    config = T5Config(vocab_size=32100, layer_norm_epsilon=1e-6, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
                      initializer_factor=1.0, **spec)
    config.architectures = ["T5ForConditionalGeneration"]
    model = T5ForConditionalGeneration(config).eval()
    with torch.no_grad():
        # Wider projections than T5's init (whose tiny outputs barely depend on the decoder input, so greedy decoding repeats one
        # token) and larger relative-bias entries, so the position bias visibly shapes the attention.
        for pname, p in model.named_parameters():
            if p.dim() == 2 and "relative_attention_bias" not in pname:
                p.normal_(0.0, 1.0 if pname == "shared.weight" else 1.5 / p.shape[1] ** 0.5)
        for module in model.modules():
            if hasattr(module, "relative_attention_bias"):
                module.relative_attention_bias.weight.normal_(0.0, 1.0)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tokenizer_dir, f)):
            shutil.copy(os.path.join(tokenizer_dir, f), out)
    # Encoder embeddings (EmbeddingFamilyTests): mean-pooled final encoder states of a few sequences, one longer than the bias range.
    gen = torch.Generator().manual_seed(7)
    inputs = [torch.randint(3, 32000, (n,), generator=gen).tolist() for n in (5, 37, 150)]
    with torch.no_grad():
        embeddings = [model.encoder(input_ids=torch.tensor([ids])).last_hidden_state[0].mean(0).tolist() for ids in inputs]
    with open(os.path.join(out, "reference_embeddings.json"), "w") as f:
        json.dump({"inputs": inputs, "embeddings": embeddings, "pooling": "mean"}, f)
    print("wrote", out)
