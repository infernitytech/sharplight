"""Writes a tiny random Gemma 3 (HF gemma3), Idefics3 / SmolVLM (idefics3) or LFM2-VL (lfm2_vl) vision-language checkpoint for the image tests.

Usage: python mk_tiny_siglip_vl.py <gemma3|idefics3|lfm2_vl> <tokenizer_dir> <out_dir>

<tokenizer_dir> holds the real model's tokenizer and processor files (gemma3: e.g. unsloth/gemma-3-4b-it; idefics3:
HuggingFaceTB/SmolVLM-256M-Instruct, *.json / *.model / *.txt only); they are copied, with the processor configs scaled to the tiny tower.
Both towers are SigLIP (2 layers, width 32, patch 8, 64-pixel images = an 8×8 patch grid):
- gemma3: 4×4 average pooling to 2×2 = 4 image tokens, soft-embedding norm and input projection; Gemma 3 text model (6 layers, head_dim 32).
- idefics3: 64-pixel tiles, pixel shuffle with scale factor 2 (16 tokens per tile) and the linear connector; Llama text model (2 layers).
- lfm2_vl (tokenizer: LiquidAI/LFM2-VL-450M): a SigLIP 2 NaFlex tower instead (patch 16, the size llama.cpp's converter assumes; a 4×4 learned
  position grid resampled to every view), pixel unshuffle by 2, LayerNorm and GELU MLP; LFM2 text model (conv, conv, attention, conv layers).
Weights are drawn wider than the default init so the greedy continuation depends on the input. tools/siglip_vl_reference.py writes the
references, tools/convert_tiny_mmproj.py the GGUF / mmproj files.
"""
import json
import os
import shutil
import sys

import torch
from transformers import (Gemma3Config, Gemma3ForConditionalGeneration, Idefics3Config, Idefics3ForConditionalGeneration, Lfm2VlConfig,
                          Lfm2VlForConditionalGeneration)

kind, tokenizer_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.manual_seed({"gemma3": 53, "idefics3": 59}.get(kind, 67))
vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=64, patch_size=8,
              hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
if kind == "gemma3":
    # Six layers: Gemma 3's pattern of five sliding-window layers and one global layer (llama.cpp assumes it).
    text = dict(vocab_size=262208, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
                head_dim=32, query_pre_attn_scalar=32, sliding_window=512, max_position_embeddings=1024, rms_norm_eps=1e-6,
                layer_types=["sliding_attention"] * 5 + ["full_attention"])
    config = Gemma3Config(text_config=text, vision_config=dict(vision, model_type="siglip_vision_model", vision_use_head=False), mm_tokens_per_image=4,
                          boi_token_index=255999, eoi_token_index=256000, image_token_index=262144, tie_word_embeddings=True)
    config.architectures = ["Gemma3ForConditionalGeneration"]
    model = Gemma3ForConditionalGeneration(config).eval()
elif kind == "lfm2_vl":
    text = dict(model_type="lfm2", vocab_size=65536, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, layer_types=["conv", "conv", "full_attention", "conv"], conv_L_cache=3, norm_eps=1e-5,
                rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}, max_position_embeddings=2048, block_auto_adjust_ff_dim=False,
                tie_word_embeddings=True)
    vision_l = dict(model_type="siglip2_vision_model", hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                    patch_size=16, num_patches=16, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6, vision_use_head=False)
    config = Lfm2VlConfig(text_config=text, vision_config=vision_l, downsample_factor=2, projector_hidden_size=48, projector_bias=True,
                          projector_use_layernorm=True, image_token_id=396, tie_word_embeddings=True)
    config.architectures = ["Lfm2VlForConditionalGeneration"]
    model = Lfm2VlForConditionalGeneration(config).eval()
else:
    text = dict(vocab_size=49280, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                rms_norm_eps=1e-5, rope_theta=100000.0, max_position_embeddings=2048, tie_word_embeddings=False, model_type="llama")
    config = Idefics3Config(text_config=text, vision_config=dict(vision, max_image_size={"longest_edge": 64}), scale_factor=2, image_token_id=49190,
                            tie_word_embeddings=False)
    config.architectures = ["Idefics3ForConditionalGeneration"]
    model = Idefics3ForConditionalGeneration(config).eval()

with torch.no_grad():
    for name, p in model.named_parameters():
        if kind == "lfm2_vl" and "conv.conv" in name:
            p.normal_(0, 0.3)
        elif "norm" in name and name.endswith("weight"):
            # Gemma's RMS norms store w with the scale 1 + w; SigLIP's LayerNorms the scale itself.
            p.copy_((torch.randn_like(p) * 0.1) + (0.0 if "language_model" in name and kind == "gemma3" else 1.0))
            if "soft_emb_norm" in name:
                p.copy_(torch.randn_like(p) * 0.1)
        elif name.endswith("bias"):
            p.normal_(0, 0.05)
        else:
            p.normal_(0, 0.08)

model.save_pretrained(out, safe_serialization=True)
if kind in ("gemma3", "lfm2_vl"):
    # The Hub's (transformers 4) tensor names, which llama.cpp's converter expects and transformers 5 still loads.
    from safetensors.torch import load_file, save_file
    path = os.path.join(out, "model.safetensors")
    tensors = load_file(path)
    tower = "vision_tower." if kind == "gemma3" else "model.vision_tower."
    save_file({(tower + "vision_model." + k[len(tower):] if k.startswith(tower) else k): v for k, v in tensors.items()}, path,
              metadata={"format": "pt"})
for f in os.listdir(tokenizer_dir):
    if f.endswith((".json", ".model", ".txt", ".jinja")) and f not in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))

pre = json.load(open(os.path.join(out, "preprocessor_config.json")))
proc = json.load(open(os.path.join(out, "processor_config.json"))) if os.path.exists(os.path.join(out, "processor_config.json")) else {}
if kind == "gemma3":
    pre.update(size={"height": 64, "width": 64}, image_seq_length=4)
    proc.update(image_seq_length=4)
elif kind == "lfm2_vl":
    # 128-pixel tiles of 8×8 patches (16 tokens each), images of 4..16 tokens.
    pre.update(tile_size=128, min_image_tokens=4, max_image_tokens=16, max_num_patches=64)
else:
    pre.update(size={"longest_edge": 128}, max_image_size={"longest_edge": 64})
    proc.update(image_seq_len=16)
json.dump(pre, open(os.path.join(out, "preprocessor_config.json"), "w"), indent=2)
json.dump(proc, open(os.path.join(out, "processor_config.json"), "w"), indent=2)
print("wrote", out)
