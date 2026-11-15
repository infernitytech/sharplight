"""Builds tiny random PaliGemma checkpoints (PaliGemmaForConditionalGeneration, the real tokenizer) for the text-decoder import:
`paligemma` (Gemma text model) and `paligemma2` (Gemma 2 text model). The text configs leave head_dim out, as the released ones do (the Gemma
default of 256 applies, not hidden size over head count). The SigLIP tower and projector are tiny and random; the text import skips them.

Usage: python mk_tiny_paligemma.py paligemma|paligemma2 <outdir>
"""
import os
import sys
import urllib.request

import torch
from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

TOKENIZER_REPO = "google/paligemma-3b-pt-224"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

TEXT = {
    "paligemma": dict(model_type="gemma", hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=1,
                      num_hidden_layers=4, vocab_size=257216, max_position_embeddings=8192),
    "paligemma2": dict(model_type="gemma2", hidden_size=72, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                       num_hidden_layers=4, vocab_size=257216, max_position_embeddings=8192, query_pre_attn_scalar=256,
                       attn_logit_softcapping=50.0, final_logit_softcapping=30.0, sliding_window=5,
                       hidden_activation="gelu_pytorch_tanh"),
}


def fetch(name, target):
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    token = os.environ.get("HF_TOKEN") or (open(token_file).read().strip() if os.path.exists(token_file) else "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    url = f"https://huggingface.co/{TOKENIZER_REPO}/resolve/main/{name}"
    with open(target, "wb") as f:
        f.write(urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read())


def main():
    family, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    text = dict(TEXT[family])
    vision = dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=1,
                  patch_size=14, image_size=28, projection_dim=text["hidden_size"])
    config = PaliGemmaConfig(text_config=text, vision_config=vision, image_token_index=257152, projection_dim=text["hidden_size"],
                             bos_token_id=2, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(0)
    model = PaliGemmaForConditionalGeneration(config).to(torch.float32)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("language_model.norm.weight"):
                p.copy_(7.0 + 0.3 * torch.randn_like(p))  # a final norm of about 8 spreads the tied head's small logits apart
            elif p.dim() == 1 and "norm" in n:
                p.copy_(0.3 * torch.randn_like(p))        # Gemma norms are 1 + w
            elif "embed" in n:
                p.copy_(torch.randn_like(p) * 0.05)   # scaled by sqrt(hidden) on input
            else:
                p.copy_(torch.randn_like(p) * 0.3)
    model.save_pretrained(out, safe_serialization=True)
    for name in TOKENIZER_FILES:
        fetch(name, os.path.join(out, name))
    print("done", family, sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
