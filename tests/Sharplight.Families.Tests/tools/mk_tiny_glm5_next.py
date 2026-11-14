"""
Builds a tiny random GLM-5.3-Flash checkpoint (transformers Glm5NextForConditionalGeneration with a tiny vision tower that is not used) for
Sharplight.Families.Tests:

  tiny-random-glm5-next   Kimi delta-attention layers with the bounded forget gate and DSA layers (no-rope multi-head latent attention whose
                          k-pool indexer runs in its dense regime: index_topk covers the test context), the second DSA layer (right after the
                          first) sharing its selection; a dense first layer and MoE layers (sigmoid router with selection bias, shared expert),
                          all MLPs with the clamped SwiGLU; manifold-constrained hyper-connections (4 streams, Sinkhorn-normalized mixing)
                          collapsed by their mean before the output norm

    python mk_tiny_glm5_next.py <models/hf directory>

Reference ids and logits: hf_greedy_reference.py --logits; GGUF: convert_tiny_gguf.py.
"""
import os, shutil, sys
import torch
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
layers = 4
text = dict(vocab_size=1024, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
            routed_scaling_factor=2.5, n_group=1, topk_group=1, norm_topk_prob=True, kv_lora_rank=32, q_lora_rank=48,
            qk_rope_head_dim=0, qk_nope_head_dim=16, v_head_dim=16, index_topk=256, index_head_dim=16, index_n_heads=2, index_kpool=16,
            layer_types=["linear_attention", "indexed_attention", "indexed_attention", "linear_attention"],
            indexer_types=["full", "full", "shared", "full"], mlp_layer_types=["dense"] + ["sparse"] * (layers - 1),
            swiglu_limit=1.5, linear_head_dim=16, linear_num_heads=4, linear_conv_kernel_dim=4, linear_lower_bound=-5.0,
            hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20, rms_norm_eps=1e-5, max_position_embeddings=256,
            bos_token_id=0, eos_token_id=0, pad_token_id=1, tie_word_embeddings=False)
vision = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64, projection_intermediate_size=64,
              patch_size=14, spatial_merge_size=2, temporal_patch_size=2, image_size=56)
config = Glm5NextConfig(text_config=text, vision_config=vision, image_token_id=1000, video_token_id=1001, image_start_token_id=1002,
                        image_end_token_id=1003, video_start_token_id=1004, video_end_token_id=1005)
torch.manual_seed(0)
model = Glm5NextForConditionalGeneration(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if ".visual." in n:
            continue
        if n.endswith(("norm.weight", "layernorm.weight")):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif n.endswith("norm.bias"):
            p.copy_(0.2 * torch.randn_like(p))
        elif "_hc." in n and n.endswith("scale"):
            p.copy_(0.5 + 0.5 * torch.rand_like(p))
        elif "_hc." in n and n.endswith("base"):
            p.copy_(0.5 * torch.randn_like(p))
        elif n.endswith("A_log"):
            p.copy_(torch.log(torch.rand_like(p) * 4 + 1))
        elif n.endswith("dt_bias"):
            p.copy_(0.5 * torch.randn_like(p))
        elif n.endswith("conv1d.weight"):
            p.copy_(0.4 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * 0.15)
    for n, b in model.named_buffers():
        if n.endswith("e_score_correction_bias"):
            b.copy_(0.2 * torch.randn_like(b))
out = os.path.join(root, "tiny-random-glm5-next")
model.save_pretrained(out, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))
print("wrote", out)
