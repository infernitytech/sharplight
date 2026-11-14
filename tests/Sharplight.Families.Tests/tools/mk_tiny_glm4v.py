"""Writes a tiny random GLM-4.1V (HF glm4v) checkpoint for the family and vision tests.

Usage: python mk_tiny_glm4v.py <tokenizer_dir> <out_dir>

<tokenizer_dir> holds a GLM-4 tokenizer (e.g. tiny-random-glm-4, vocabulary 151552); <|image|> (id 151343, as in GLM-4.1V's tokenizer) is
added. The language model is GLM-4 (q/k/v biases, sandwich norms, fused gate/up, rotary on the first half of each head in interleaved pairs)
with M-RoPE sections [2, 3, 3] (transformers takes the head width from hidden_size / heads: 2 heads of 32, 16 channels rotated); the tower is tiny (2 layers, 32 wide, 2 heads) with a 4×4 learned position grid resampled bicubically,
a 2×2 convolution merging patches and GLM-4V's projector. Weights are drawn wider than the default init so outputs depend on the input.
"""
import json
import os
import sys

import torch
from transformers import AutoTokenizer, Glm4vConfig, Glm4vForConditionalGeneration

tokenizer_dir, out = sys.argv[1], sys.argv[2]
torch.manual_seed(47)
text = dict(vocab_size=151552, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, rms_norm_eps=1e-5, max_position_embeddings=1024, attention_bias=True,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3], "partial_rotary_factor": 0.5})
vision = dict(hidden_size=32, intermediate_size=48, depth=2, num_heads=2, out_hidden_size=64, image_size=56, patch_size=14,
              spatial_merge_size=2, temporal_patch_size=2, rms_norm_eps=1e-5)
config = Glm4vConfig(text_config=text, vision_config=vision, image_token_id=151343, tie_word_embeddings=False)
config.architectures = ["Glm4vForConditionalGeneration"]
model = Glm4vForConditionalGeneration(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1 and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif p.dim() >= 2:
            fan_in = p[0].numel()
            p.normal_(0.0, 1.0 if "embed_tokens" in name or "position_embedding" in name else 1.5 / fan_in ** 0.5)
        else:
            p.normal_(0.0, 0.1)
    model.lm_head.weight.mul_(4.0)   # wider logit margins for the greedy comparison
model.save_pretrained(out, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
assert tokenizer.convert_tokens_to_ids("<|image|>") == 151343
tokenizer.save_pretrained(out)
with open(os.path.join(out, "preprocessor_config.json"), "w") as f:
    json.dump({"image_processor_type": "Glm4vImageProcessor", "image_mean": [0.48145466, 0.4578275, 0.40821073],
               "image_std": [0.26862954, 0.26130258, 0.27577711], "size": {"shortest_edge": 12544, "longest_edge": 9633792},
               "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2}, f, indent=2)
print("wrote", out)
