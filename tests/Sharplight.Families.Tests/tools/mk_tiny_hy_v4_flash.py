"""Writes a tiny random Hunyuan v4 Flash (HF hy_v4_flash, tencent/Hy4-flash) checkpoint for the family tests.

Usage: python mk_tiny_hy_v4_flash.py <tokenizer_dir> <out_dir>

There is no transformers class and the Hub repository ships no modeling code, so the checkpoint is written directly: config.json and
model.safetensors with the released tensor names (model.layers.{i}.self_attn.compressed_indexer.*, engram_layer.*, ...). The reference is
llama.cpp HEAD (src/models/hy-v4-flash.cpp, run on the GGUF its converter writes from this directory; see llamacpp_greedy.c).

The model: 5 layers, sparse (pooled indexer: pools of 4, index_topk 8, so one picked history pool plus the query's own pool) and
sliding-window (window 8, twice the heads) MLA layers over one latent geometry, gated MLA with sinks, sandwich norms, one dense layer then
latent MoE layers (sigmoid routing with a selection bias, latent 24 < hidden 64, clamped SwiGLU experts, a shared expert), hyper-connections
with 3 streams and two Engram modules (3-grams, 2 heads per order) at a sliding and a sparse layer.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

tokenizer_dir, out = sys.argv[1], sys.argv[2]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
vocab = len(json.load(open(os.path.join(tokenizer_dir, "tokenizer.json")))["model"]["vocab"])

hidden, layers, heads, swa_heads = 64, 5, 4, 8
q_rank, kv_rank, nope, rope, v_dim = 48, 32, 16, 8, 16
experts, used, moe_inter, latent, inter = 8, 2, 32, 24, 96
hc = 3
idx_heads, idx_dim, kpool = 8, 16, 4
layer_types = ["deepseek_sparse_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention"]
engram_ids = [2, 4]   # 1-based: layers 1 (sliding) and 3 (sparse)
ngram, engram_heads_per_ngram, engram_embed = 3, 2, 16
engram_dim = engram_embed // engram_heads_per_ngram
engram_cols = (ngram - 1) * engram_heads_per_ngram
primes = {"2": [[31, 37], [41, 43]], "4": [[47, 53], [59, 61]]}
compressed = vocab // 2

config = dict(
    architectures=["HYV4FlashForCausalLM"], model_type="hy_v4_flash", vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
    moe_intermediate_size=moe_inter, routed_expert_hidden_size=latent, num_hidden_layers=layers, num_attention_heads=heads,
    num_key_value_heads=heads, swa_num_attention_heads=swa_heads, q_lora_rank=q_rank, kv_lora_rank=kv_rank, qk_nope_head_dim=nope,
    qk_rope_head_dim=rope, qk_head_dim=nope + rope, v_head_dim=v_dim, head_dim=v_dim, swa_qk_nope_head_dim=nope, swa_qk_rope_head_dim=rope,
    swa_qk_head_dim=nope + rope, swa_v_head_dim=v_dim, n_routed_experts=experts, n_shared_experts=1, num_experts_per_tok=used,
    routed_scaling_factor=1.5, norm_topk_prob=True, n_group=1, topk_group=1, moe_router_use_sigmoid=True, moe_router_enable_expert_bias=True,
    first_k_dense_replace=1, mlp_layer_types=["dense"] + ["sparse"] * (layers - 1), swiglu_limit=1.5, hidden_act="silu",
    layer_types=layer_types, indexer_types=["full"] * layers, sliding_window=8, window_size=[7, 0], sandwich_norm=True,
    dsa_skip_attn_post_norm=False, use_dsa=True, use_compressed_indexer=True, index_topk=8, index_n_heads=idx_heads, index_head_dim=idx_dim,
    compressed_indexer_compress_ratio=kpool, compressed_indexer_group_topk=2, compressed_indexer_candidate_topk=8,
    compressed_indexer_head_dim=idx_dim, compressed_indexer_num_attention_heads=idx_heads, compressed_indexer_key_pooling="gated",
    learnable_sink=True, gated_mla=True, gating_type="elementwise", enable_ihc=True, hc_mult=hc, hc_eps=1e-6, hc_magnitude=2.0,
    engram_layer_ids=engram_ids, engram_max_ngram_size=ngram, engram_n_head_per_ngram=engram_heads_per_ngram,
    engram_n_embed_per_ngram=engram_embed, engram_pad_id=1, engram_seed=0, engram_layer_head_vocab_sizes=primes,
    engram_vocab_size=[compressed, compressed],
    rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                     "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1000000.0}},
    rms_norm_eps=1e-5, max_position_embeddings=256, num_nextn_predict_layers=0, tie_word_embeddings=False, enable_lm_head_fp32=True,
    bos_token_id=None, eos_token_id=None, pad_token_id=None, torch_dtype="float32",
)

g = torch.Generator().manual_seed(83)


def randn(*shape, std=1.0, mean=0.0):
    return (torch.randn(*shape, generator=g) * std + mean).float()


def linear(out_features, in_features, gain=1.5):
    return randn(out_features, in_features, std=gain / in_features ** 0.5)


def norm(n):
    return randn(n, std=0.2, mean=1.0)


t = {"model.embed_tokens.weight": randn(vocab, hidden), "model.norm.weight": norm(hidden), "lm_head.weight": linear(vocab, hidden, 6.0),
     "model.hc_head.hc_head_fn": linear(hc, hc * hidden), "model.hc_head.hc_head_base": randn(hc, std=0.5),
     "model.hc_head.hc_head_scale": torch.rand(1, generator=g) + 0.5}
for il, kind in enumerate(layer_types):
    p = f"model.layers.{il}."
    n_head = swa_heads if kind == "sliding_attention" else heads
    t[p + "input_layernorm.weight"] = norm(hidden)
    t[p + "post_attention_layernorm.weight"] = norm(hidden)
    for m in ("attn", "mlp"):
        t[p + f"hc_{m}_layer.hc_pre.hc_fn"] = linear(2 * hc, hc * hidden)
        t[p + f"hc_{m}_layer.hc_pre.hc_base"] = randn(2 * hc, std=0.5)
        t[p + f"hc_{m}_layer.hc_pre.hc_scale"] = torch.rand(2, generator=g) + 0.5
    a = p + "self_attn."
    t[a + "q_a_proj.weight"] = linear(q_rank, hidden)
    t[a + "q_a_layernorm.weight"] = norm(q_rank)
    t[a + "q_b_proj.weight"] = linear(n_head * (nope + rope), q_rank)
    t[a + "kv_a_proj_with_mqa.weight"] = linear(kv_rank + rope, hidden)
    t[a + "kv_a_layernorm.weight"] = norm(kv_rank)
    t[a + "kv_b_proj.weight"] = linear(n_head * (nope + v_dim), kv_rank)
    t[a + "o_proj.weight"] = linear(hidden, n_head * v_dim)
    t[a + "linear_gate.weight"] = linear(n_head * v_dim, hidden)
    t[a + "learnable_sink_param"] = randn(n_head)
    t[a + "post_proj_norm.weight"] = norm(hidden)
    if kind != "sliding_attention":
        c = a + "compressed_indexer."
        t[c + "token_key_projection.weight"] = linear(idx_dim, hidden)
        t[c + "token_key_norm.weight"] = norm(idx_dim)
        t[c + "key_value_projection.weight"] = linear(idx_dim, idx_dim)
        t[c + "key_gate_projection.weight"] = linear(idx_dim, idx_dim, 3.0)
        t[c + "intra_group_ape"] = randn(kpool, idx_dim)
        t[c + "key_norm.weight"] = norm(idx_dim)
        t[c + "query_projection.weight"] = linear(idx_heads * idx_dim, q_rank, 3.0)
        t[c + "head_weight_projection.weight"] = linear(idx_heads, hidden)
    mp = p + "mlp."
    if il == 0:
        t[mp + "gate_proj.weight"] = linear(inter, hidden)
        t[mp + "up_proj.weight"] = linear(inter, hidden)
        t[mp + "down_proj.weight"] = linear(hidden, inter)
        t[mp + "post_fc2_norm.weight"] = norm(hidden)
    else:
        t[mp + "gate.weight"] = linear(experts, hidden, 3.0)
        t[mp + "gate.e_score_correction_bias"] = randn(experts, std=0.1)
        t[mp + "experts.gate_up_proj"] = randn(experts, 2 * moe_inter, latent, std=1.5 / latent ** 0.5)
        t[mp + "experts.down_proj"] = randn(experts, latent, moe_inter, std=1.5 / moe_inter ** 0.5)
        t[mp + "routed_expert_down_proj.weight"] = linear(latent, hidden)
        t[mp + "routed_expert_up_proj.weight"] = linear(hidden, latent)
        t[mp + "shared_experts.gate_proj.weight"] = linear(moe_inter, hidden)
        t[mp + "shared_experts.up_proj.weight"] = linear(moe_inter, hidden)
        t[mp + "shared_experts.down_proj.weight"] = linear(hidden, moe_inter)
        t[mp + "combine_output_norm.weight"] = norm(hidden)
    if il + 1 in engram_ids:
        e = p + "engram_layer."
        rows = sum(sum(group) for group in primes[str(il + 1)])
        t[e + "multi_head_embedding.embedding.weight"] = randn(rows, engram_dim)
        for s in range(hc):
            t[e + f"key_projs.{s}.weight"] = linear(hidden, engram_cols * engram_dim)
            t[e + f"key_projs.{s}.bias"] = randn(hidden, std=0.2)
            t[e + f"norm_key.{s}.weight"] = norm(hidden)
            t[e + f"norm_query.{s}.weight"] = norm(hidden)
            t[e + f"unified_per_hc_scales.{s}"] = randn(hidden, std=0.3)
        t[e + "value_proj.weight"] = linear(hidden, engram_cols * engram_dim)
        t[e + "value_proj.bias"] = randn(hidden, std=0.2)
        for c in range(engram_cols):
            t[e + f"lookup_per_head_norms.{c}.weight"] = norm(engram_dim)
        t[e + "unified_pre_gamma"] = norm(hidden)
        # token id -> compressed id: onto [0, compressed) (every id used, so the converter reads it as a dense range), the same in every layer
        mapping = torch.cat([torch.arange(compressed), torch.randint(0, compressed, (vocab - compressed,), generator=torch.Generator().manual_seed(7))])
        t[e + "lookup_table"] = mapping[torch.randperm(vocab, generator=torch.Generator().manual_seed(8))].to(torch.int64)

os.makedirs(out, exist_ok=True)
save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
for name in TOKENIZER_FILES:
    if os.path.exists(os.path.join(tokenizer_dir, name)):
        shutil.copy(os.path.join(tokenizer_dir, name), out)
print("wrote", out)
