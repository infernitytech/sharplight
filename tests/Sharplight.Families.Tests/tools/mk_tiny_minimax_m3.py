"""
Builds a tiny random MiniMax-M3 checkpoint (transformers MiniMaxM3SparseForConditionalGeneration with a tiny vision tower that is not
used) for Sharplight.Families.Tests:

  tiny-random-minimax-m3   2 heads per key-value group, per-head Gemma (1 + w) q/k RMS norms, partial rope, one leading dense layer and
                           three MoE layers (sigmoid router with selection bias, renormalized and scaled weights, one shared expert,
                           clamped SwiGLU-OAI experts), MiniMax sparse attention on the MoE layers whose block budget covers the test
                           context (so attention is dense, which is exact there)

    python mk_tiny_minimax_m3.py <models/hf directory>

transformers saves the checkpoint in the published layout (language_model.model.*, block_sparse_moe experts w1/w2/w3,
sparse_attention_config), which is also what llama.cpp's converter reads. Reference ids and logits: hf_greedy_reference.py --logits;
GGUF: convert_tiny_gguf.py.
"""
import os, shutil, sys
import torch
from transformers import MiniMaxM3VLConfig, MiniMaxM3SparseForConditionalGeneration

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
layers = 4
text = dict(vocab_size=1024, hidden_size=64, intermediate_size=32, dense_intermediate_size=96, shared_intermediate_size=32,
            num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16, rotary_dim=8,
            rope_parameters=dict(rope_type="default", rope_theta=5000000.0, partial_rotary_factor=0.5),
            num_local_experts=8, num_experts_per_tok=2, routed_scaling_factor=2.0, swiglu_alpha=1.702, swiglu_limit=7.0,
            mlp_layer_types=["dense"] + ["sparse"] * (layers - 1),
            layer_types=["full_attention"] + ["minimax_m3_sparse"] * (layers - 1),
            index_n_heads=2, index_head_dim=16, index_block_size=16, index_topk_blocks=16, index_local_blocks=1,
            max_position_embeddings=256, bos_token_id=0, eos_token_id=0, pad_token_id=1, tie_word_embeddings=False)
vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=56, patch_size=14,
              temporal_patch_size=2, spatial_merge_size=2)
config = MiniMaxM3VLConfig(text_config=text, vision_config=vision, projector_hidden_size=64, image_token_index=1000, video_token_index=1001)
torch.manual_seed(0)
model = MiniMaxM3SparseForConditionalGeneration(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            continue
        if n.endswith("norm.weight"):
            p.copy_(0.3 * torch.randn_like(p))          # Gemma norms scale by 1 + w
        elif n.endswith("e_score_correction_bias"):
            p.copy_(0.2 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * 0.15)
    for n, b in model.named_buffers():
        if n.endswith("e_score_correction_bias"):
            b.copy_(0.2 * torch.randn_like(b))
out = os.path.join(root, "tiny-random-minimax-m3")
model.save_pretrained(out, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))
print("wrote", out)
