"""
Builds a tiny random DeepSeek-V4 checkpoint (transformers DeepseekV4ForCausalLM) for Sharplight.Families.Tests:

  tiny-random-deepseek-v4   one sliding-window layer, two compressed sparse attention layers (CSA: overlapping windows of 4 positions, a
                            lightning indexer over its own compressed keys keeping 4 entries) and one heavily compressed layer (HCA: windows
                            of 128 positions), every layer with a sliding window of 8 raw positions, attention sinks, the grouped low-rank
                            output projection and inverse rope on the output; two hash-routed MoE layers (fixed token-to-expert table) and two
                            sqrt-softplus routed layers with selection bias, clamped SwiGLU experts and shared expert; manifold-constrained
                            hyper-connections (4 streams, Sinkhorn mixing) with the learned head; yarn on the compressed layers' rope.

The compression rates are the ones llama.cpp hard-codes (4 and 128), so the converted GGUF runs there too; the reference prompt is longer than
128 tokens so that the heavily compressed layer emits an entry.

    python mk_tiny_deepseek_v4.py <models/hf directory>

Reference ids and logits: hf_greedy_reference.py --logits with the prompt of DeepSeekV4Cases; GGUF: convert_tiny_gguf.py.
"""
import os, shutil, sys
import torch
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
config = DeepseekV4Config(
    vocab_size=1024, hidden_size=64, moe_intermediate_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    head_dim=128, q_lora_rank=32, partial_rotary_factor=0.25, num_experts_per_tok=2, n_routed_experts=8, n_shared_experts=1,
    scoring_func="sqrtsoftplus", norm_topk_prob=True, routed_scaling_factor=1.5, max_position_embeddings=1024, rope_theta=10000.0,
    layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention", "compressed_sparse_attention"],
    compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}, compress_rope_theta=160000.0,
    rope_parameters={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64, "beta_fast": 32.0, "beta_slow": 1.0},
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe"], swiglu_limit=1.5,
    sliding_window=8, o_groups=2, o_lora_rank=16, index_n_heads=8, index_head_dim=64, index_topk=4, rms_norm_eps=1e-6,
    bos_token_id=0, eos_token_id=0, pad_token_id=1, tie_word_embeddings=False)
torch.manual_seed(0)
model = DeepseekV4ForCausalLM(config)
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith(("norm.weight", "layernorm.weight")):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif n.endswith("_hc.scale") or n.endswith("hc_scale"):
            p.copy_(0.5 + 0.5 * torch.rand_like(p))
        elif n.endswith("_hc.base") or n.endswith("hc_base"):
            p.copy_(0.5 * torch.randn_like(p))
        elif n.endswith("sinks"):
            p.copy_(torch.randn_like(p))
        elif n.endswith("position_bias"):
            p.copy_(0.5 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * 0.15)
    for n, b in model.named_buffers():
        if n.endswith("e_score_correction_bias"):
            b.copy_(0.2 * torch.randn_like(b))
        elif n.endswith("tid2eid"):
            # Distinct experts per token (the frozen hash table of the real model).
            b.copy_(torch.stack([torch.randperm(config.n_routed_experts)[:config.num_experts_per_tok] for _ in range(b.shape[0])]))
out = os.path.join(root, "tiny-random-deepseek-v4")
model.save_pretrained(out, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))
print("wrote", out)
