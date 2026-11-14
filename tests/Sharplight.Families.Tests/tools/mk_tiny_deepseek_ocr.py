"""Writes tiny random DeepSeek-OCR family checkpoints (llama.cpp arch deepseek2-ocr) for the family and vision tests.

Usage: python mk_tiny_deepseek_ocr.py <tokenizer_dir> <out_root> [sam12]

<tokenizer_dir> holds the DeepSeek-OCR tokenizer (tokenizer.json, tokenizer_config.json, special_tokens_map.json; vocabulary 129280).
Writes, below <out_root>:
  tiny-random-deepseek-ocr2      HF deepseek_ocr2 (transformers): SAM ViT-B + Qwen2 query encoder, DeepSeek-V2 MoE text model without MLA
  tiny-random-unlimited-ocr      HF unlimited_ocr (transformers): SAM ViT-B + CLIP-L tower; its "reference" sliding window is wider than the
                                 tests' sequences (full attention, as llama.cpp's graph)
  tiny-random-deepseek-ocr       the unlimited_ocr weights in DeepSeek-OCR's own remote-code layout (model_type deepseek_vl_v2,
                                 DeepseekOCRForCausalLM, language_config / vision_config.width): the same model (the CLIP patch embedding,
                                 unused by the model, is added for llama.cpp's mmproj converter)
  tiny-random-unlimited-ocr-sam12 and tiny-random-deepseek-ocr-sam12 (only these with "sam12"): the same layouts with a 12-layer SAM whose
                                 global layers are 2, 5, 8 and 11, the layers llama.cpp's DeepSeek-OCR graph hard-codes, and a 12-layer CLIP
                                 (llama.cpp's mmproj converter names tower blocks, the SAM's and the 4 neck tensors, up to the CLIP depth);
                                 the vision tests use llama.cpp's conversion of tiny-random-deepseek-ocr-sam12 (convert_hf_to_gguf.py --outtype f32,
                                 without and with --mmproj) as ~/models/gguf/tiny-random-deepseek-ocr-sam12-f32.gguf and
                                 ~/models/gguf/mmproj-tiny-random-deepseek-ocr-sam12-f32.gguf, and the references of hf_ocr_image_reference.py
                                 (1 1 reference_image_global.json; 2 3) in tiny-random-unlimited-ocr-sam12
transformers saves these checkpoints with the original tensor names (model.layers.*, model.sam_model.*, model.vision_model.*, ...).
The towers are narrow and shallow but keep the real image geometry (1024-pixel global view, 640 / 768-pixel tiles, 16-pixel patches, 14-patch
windows, a 16 × 16 CLIP grid), so they exercise every structural feature: windowed SAM layers whose tile grid needs padding, relative position
tables that are resampled for the tile grid, learned position grids that are resampled (bicubic with antialiasing) for the tiles, and they run
in llama.cpp, whose DeepSeek-OCR preprocessing has these sizes built in.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file
from transformers import DeepseekOcr2Config, DeepseekOcr2ForConditionalGeneration, UnlimitedOcrConfig, UnlimitedOcrForConditionalGeneration

tokenizer_dir, out_root = sys.argv[1], sys.argv[2]
only_sam12 = len(sys.argv) > 3 and sys.argv[3] == "sam12"

TEXT = dict(vocab_size=129280, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=4, n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=2, max_position_embeddings=1024,
            rms_norm_eps=1e-6, bos_token_id=0, eos_token_id=1, routed_scaling_factor=1.0, n_group=1, topk_group=1, topk_method="greedy")
SAM = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_dim=64, output_channels=16, image_size=1024, patch_size=16,
           window_size=14, global_attn_indexes=[1], layer_norm_eps=1e-6)
GLOBAL_SIZE, TILE_SIZE = 1024, 640


def randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ("norm" in name or "layrnorm" in name) and p.dim() == 1 and name.endswith("weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif "rel_pos" in name or "pos_embed" in name or "position_embedding" in name or "class_embedding" in name:
                p.normal_(0.0, 0.5)
            elif p.dim() >= 2:
                fan_in = p[0].numel() if p.dim() == 4 else p.shape[-1]
                p.normal_(0.0, 1.0 if "embed_tokens" in name or "query_" in name else 1.5 / fan_in ** 0.5)
            else:
                p.normal_(0.0, 0.1)
        model.lm_head.weight.mul_(8.0)   # wider logit margins for the greedy comparison


def copy_tokenizer(out):
    for f in os.listdir(tokenizer_dir):
        if f.startswith("tokenizer") or f == "special_tokens_map.json":
            shutil.copy(os.path.join(tokenizer_dir, f), out)


def processor_config(out, tile_size, max_patches):
    with open(os.path.join(out, "preprocessor_config.json"), "w") as f:
        json.dump({"image_processor_type": "DeepseekOcr2ImageProcessor" if max_patches == 6 else "UnlimitedOcrImageProcessor",
                   "size": {"height": GLOBAL_SIZE, "width": GLOBAL_SIZE}, "tile_size": tile_size, "min_patches": 2, "max_patches": max_patches,
                   "crop_to_patches": True, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "background_color": [127, 127, 127],
                   "do_rescale": True, "rescale_factor": 1 / 255, "do_normalize": True, "resample": 3}, f, indent=2)


def write_v1(unlimited, v1, sam1, enc1, sam, max_patches):
    """DeepSeek-OCR's remote-code layout (deepseek_vl_v2, DeepseekOCRForCausalLM) of the weights of an unlimited_ocr checkpoint."""
    os.makedirs(v1, exist_ok=True)
    state = load_file(os.path.join(unlimited, "model.safetensors"))
    torch.manual_seed(53)
    state["model.vision_model.embeddings.patch_embedding.weight"] = torch.randn(enc1["hidden_size"], 3, 14, 14) * 0.02
    save_file(state, os.path.join(v1, "model.safetensors"), metadata={"format": "pt"})
    language = dict(TEXT, first_k_dense_replace=1, architectures=["DeepseekV2ForCausalLM"], kv_lora_rank=None, q_lora_rank=None,
                    qk_nope_head_dim=0, qk_rope_head_dim=0, v_head_dim=0, use_mla=False, lm_head=True, rm_head=False, torch_dtype="float32")
    language.pop("routed_scaling_factor")
    v1_config = dict(language, architectures=["DeepseekOCRForCausalLM"], model_type="deepseek_vl_v2", language_config=language,
                     candidate_resolutions=[[GLOBAL_SIZE, GLOBAL_SIZE]], global_view_pos="head", tile_tag="2D", torch_dtype="float32",
                     projector_config={"input_dim": sam1["downsample_channels"][-1] + enc1["hidden_size"], "model_type": "mlp_projector",
                                       "n_embed": TEXT["hidden_size"], "projector_type": "linear"},
                     vision_config={"image_size": GLOBAL_SIZE, "mlp_ratio": sam["mlp_dim"] / sam["hidden_size"], "model_name": "deeplip_b_l",
                                    "model_type": "vision",
                                    "width": {"clip-l-14-224": {"heads": enc1["num_attention_heads"], "image_size": enc1["image_size"],
                                                                "layers": enc1["num_hidden_layers"], "patch_size": 14, "width": enc1["hidden_size"],
                                                                # Sizes the remote code fixes for the real model, given for the tiny one.
                                                                "intermediate_size": enc1["intermediate_size"]},
                                              "sam_vit_b": {"downsample_channels": sam1["downsample_channels"],
                                                            "global_attn_indexes": sam["global_attn_indexes"], "heads": sam["num_attention_heads"],
                                                            "layers": sam["num_hidden_layers"], "width": sam["hidden_size"], "mlp_dim": sam["mlp_dim"],
                                                            "output_channels": sam["output_channels"], "window_size": sam["window_size"]}}})
    with open(os.path.join(v1, "config.json"), "w") as f:
        json.dump(v1_config, f, indent=2)
    copy_tokenizer(v1)
    processor_config(v1, TILE_SIZE, max_patches)
    for f in ("reference_logits.json",):
        if os.path.exists(os.path.join(unlimited, f)):
            shutil.copy(os.path.join(unlimited, f), v1)
    print("wrote", v1)


ENC1 = dict(hidden_size=48, intermediate_size=192, num_hidden_layers=2, num_attention_heads=4, image_size=224, patch_size=14, layer_norm_eps=1e-5)


def write_unlimited(out, sam, enc, seed):
    """An unlimited_ocr checkpoint (transformers) with the given SAM and CLIP; returns the SAM config with its downsample widths."""
    sam1 = dict(sam, downsample_channels=[24, 48])   # the last width is the CLIP width: SAM features are its patch embeddings
    config = UnlimitedOcrConfig(text_config=dict(TEXT, first_k_dense_replace=1, sliding_window=4096),
                                vision_config=dict(sam_config=sam1, encoder_config=enc), image_token_id=128815)
    config.architectures = ["UnlimitedOcrForConditionalGeneration"]
    model = UnlimitedOcrForConditionalGeneration(config).eval()
    randomize(model, seed)
    model.save_pretrained(out, safe_serialization=True)
    copy_tokenizer(out)
    processor_config(out, TILE_SIZE, 32)
    print("wrote", out)
    return sam1


if only_sam12:
    # The mmproj converter names tower blocks up to the CLIP depth, so the CLIP gets as many layers as the SAM.
    sam12 = dict(SAM, num_hidden_layers=12, global_attn_indexes=[2, 5, 8, 11])
    enc12 = dict(ENC1, num_hidden_layers=12)
    unlimited12 = os.path.join(out_root, "tiny-random-unlimited-ocr-sam12")
    write_v1(unlimited12, os.path.join(out_root, "tiny-random-deepseek-ocr-sam12"), write_unlimited(unlimited12, sam12, enc12, 54), enc12, sam12, 9)
    sys.exit(0)


# DeepSeek-OCR-2: SAM → stride-2 convolutions to the encoder width → Qwen2 encoder over [patches | learned queries].
ocr2 = os.path.join(out_root, "tiny-random-deepseek-ocr2")
text2 = dict(TEXT, mlp_layer_types=["dense", "sparse", "sparse"])
sam2 = dict(SAM, downsample_channels=[24, 48])
enc2 = dict(hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6,
            max_position_embeddings=2048)
config = DeepseekOcr2Config(text_config=text2, vision_config=dict(sam_config=sam2, encoder_config=enc2), image_token_id=128815)
config.architectures = ["DeepseekOcr2ForConditionalGeneration"]
model = DeepseekOcr2ForConditionalGeneration(config).eval()
randomize(model, 61)
model.save_pretrained(ocr2, safe_serialization=True)
copy_tokenizer(ocr2)
processor_config(ocr2, 768, 6)
print("wrote", ocr2)

# Unlimited-OCR (and DeepSeek-OCR, the same model): SAM → convolutions → CLIP-L over the SAM feature map; [CLIP | SAM] → linear projector.
unlimited = os.path.join(out_root, "tiny-random-unlimited-ocr")
text1 = dict(TEXT, first_k_dense_replace=1, sliding_window=4096)
sam1 = dict(SAM, downsample_channels=[24, 48])   # the last width is the CLIP width: SAM features are its patch embeddings
enc1 = dict(hidden_size=48, intermediate_size=192, num_hidden_layers=2, num_attention_heads=4, image_size=224, patch_size=14, layer_norm_eps=1e-5)
config = UnlimitedOcrConfig(text_config=text1, vision_config=dict(sam_config=sam1, encoder_config=enc1), image_token_id=128815)
config.architectures = ["UnlimitedOcrForConditionalGeneration"]
model = UnlimitedOcrForConditionalGeneration(config).eval()
randomize(model, 52)
model.save_pretrained(unlimited, safe_serialization=True)
copy_tokenizer(unlimited)
processor_config(unlimited, TILE_SIZE, 32)
print("wrote", unlimited)

# DeepSeek-OCR's remote-code layout of the same weights.
write_v1(unlimited, os.path.join(out_root, "tiny-random-deepseek-ocr"), sam1, enc1, SAM, 9)
