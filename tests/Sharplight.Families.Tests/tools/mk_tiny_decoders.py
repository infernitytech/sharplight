"""
Builds tiny random Hugging Face checkpoints for decoder families that have no tiny-random model in ~/models/hf:
GPT-2, GPT-J, Nemotron (dense), Gemma 1 and Falcon with the GPT-2-style tokenizer of tiny-random-GPTNeoXForCausalLM (vocabulary 1024).

    python mk_tiny_decoders.py <models/hf directory>

Every model is written to <dir>/tiny-random-<family>/ in fp32 safetensors; the tests skip a family whose directory is missing.
Norm weights and biases are randomized so that every parameter matters.
"""
import os, sys, shutil, torch
from transformers import (AutoModelForCausalLM, GPT2Config, GPTJConfig, NemotronConfig, GemmaConfig, FalconConfig)

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
VOCAB = 1024

configs = {
    "gpt2": GPT2Config(vocab_size=VOCAB, n_positions=256, n_embd=64, n_layer=3, n_head=4, bos_token_id=0, eos_token_id=0),
    "gptj": GPTJConfig(vocab_size=VOCAB, n_positions=256, n_embd=64, n_layer=3, n_head=4, rotary_dim=8, bos_token_id=0, eos_token_id=0),
    "nemotron": NemotronConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                               num_key_value_heads=2, partial_rotary_factor=0.5, max_position_embeddings=256, bos_token_id=0, eos_token_id=0),
    "gemma": GemmaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=1, head_dim=16, max_position_embeddings=256, bos_token_id=0, eos_token_id=0, pad_token_id=0),
    "falcon": FalconConfig(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_kv_heads=2,
                           new_decoder_architecture=True, parallel_attn=True, bias=False, max_position_embeddings=256, bos_token_id=0, eos_token_id=0),
}

for name, config in configs.items():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p) if n.endswith("weight") else 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    out = os.path.join(root, f"tiny-random-{name}")
    model.save_pretrained(out, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(tokenizer_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))
    print("wrote", out)
