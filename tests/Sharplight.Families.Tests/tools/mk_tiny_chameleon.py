"""
Builds tiny random Chameleon checkpoints (transformers ChameleonForConditionalGeneration, text part) for Sharplight.Families.Tests:

  tiny-random-chameleon       pre-norm layers, 4 query / 2 key-value heads, q/k layer norms with a different gain and bias for every head
  tiny-random-chameleon-mha   pre-norm layers, multi-head attention and q/k norm parameters shared by all heads as in the real checkpoints
                              (llama.cpp's converter keeps only the first head's and assumes as many key-value heads as query heads)
  tiny-random-chameleon-swin  the same with swin_norm layers (norms after attention and MLP)

    python mk_tiny_chameleon.py <models/hf directory> [name ...]

The vocabulary has 8448 entries: the GPT-2-style tokenizer of tiny-random-GPTNeoXForCausalLM (1024 tokens) padded with plain tokens, because
llama.cpp suppresses the image-token logits [4, 8196) of the real Chameleon vocabulary whatever the file says. vocabulary_map names those ids
IMGIMG* so that transformers suppresses the same range. The VQ-GAN image tokenizer is randomly initialized and not used.
Reference ids and logits: hf_greedy_reference.py --logits; GGUF: convert_tiny_gguf.py.
"""
import json, os, shutil, sys
import torch
from transformers import ChameleonConfig, ChameleonForConditionalGeneration

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
VOCAB = 8448
FIRST_IMAGE, END_IMAGE = 4, 8196


def image_token_name(i):
    # Chameleon names image token i "IMGIMG" + the digits of i written as letters A-J + "Z".
    return "IMGIMG" + "".join(chr(ord("A") + int(d)) for d in str(i - FIRST_IMAGE)) + "Z"


def vocabulary_map():
    tok = json.load(open(os.path.join(tokenizer_dir, "tokenizer.json")))
    names = {name: index for name, index in tok["model"]["vocab"].items() if not FIRST_IMAGE <= index < END_IMAGE}
    names.update({image_token_name(i): i for i in range(FIRST_IMAGE, END_IMAGE)})
    names["<image>"] = END_IMAGE
    return names


def write_tokenizer(out):
    """The 1024-token tokenizer with plain tokens appended up to VOCAB, so that every generated id decodes."""
    tok = json.load(open(os.path.join(tokenizer_dir, "tokenizer.json")))
    vocab = tok["model"]["vocab"]
    for i in range(len(vocab), VOCAB):
        vocab[f"tok{i}"] = i
    json.dump(tok, open(os.path.join(out, "tokenizer.json"), "w"))
    for f in ["tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))


def build(name, swin, shared_heads, kv_heads, seed):
    config = ChameleonConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=kv_heads, max_position_embeddings=256, swin_norm=swin, vocabulary_map=vocabulary_map(),
                             bos_token_id=0, eos_token_id=0, pad_token_id=1,
                             vq_config=dict(embed_dim=16, num_embeddings=64, base_channels=32, channel_multiplier=[1], num_res_blocks=1,
                                            resolution=16, latent_channels=16, attn_resolutions=[]))
    torch.manual_seed(seed)
    model = ChameleonForConditionalGeneration(config)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.startswith("model.vqmodel."):
                continue
            if p.dim() == 1 or "_norm." in n:
                p.copy_(1.0 + 0.3 * torch.randn_like(p) if n.endswith("weight") else 0.3 * torch.randn_like(p))
                if shared_heads and p.dim() == 2:
                    p.copy_(p[0:1].expand_as(p))
            else:
                p.copy_(torch.randn_like(p) * 0.15)
    out = os.path.join(root, f"tiny-random-{name}")
    model.save_pretrained(out, safe_serialization=True)
    write_tokenizer(out)
    print("wrote", out)


MODELS = {
    "chameleon": dict(swin=False, shared_heads=False, kv_heads=2, seed=0),
    "chameleon-swin": dict(swin=True, shared_heads=True, kv_heads=4, seed=1),
    "chameleon-mha": dict(swin=False, shared_heads=True, kv_heads=4, seed=1),
}
for name, arguments in MODELS.items():
    if len(sys.argv) <= 2 or name in sys.argv[2:]:
        build(name, **arguments)
