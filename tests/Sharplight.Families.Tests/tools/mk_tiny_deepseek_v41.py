"""
Writes a tiny random DeepSeek-V4.1 GGUF (llama.cpp deepseek41; there is no transformers class) for Sharplight.Families.Tests:

  tiny-random-deepseek-v41.gguf   five layers: 0 compresses every position (ratio 1) with its own indexer, 1 attends its sliding window only,
                                  2 compresses windows of 2 positions with its own indexer that also picks 3 candidate blocks of 4 entries,
                                  3 attends layer 2's compressed caches with its own indexer restricted to those candidate blocks and 4 attends
                                  them with layer 3's selection (top-k 4 everywhere); index keys derived from the entries; Engram n-gram memory
                                  (n-grams of 3, 2 heads per order) on layers 1 and 3; single-pass hyper-connections (4 streams, Sinkhorn mixing,
                                  no head); sqrt-softplus routed experts with selection bias, clamped SwiGLU, a shared expert; yarn on the
                                  compressed layers' rope; f32 throughout.

The tokenizer (GPT-2 style, 1024 tokens) is copied from the tiny DeepSeek-V4 GGUF. The Engram hash constants follow the layout of llama.cpp's
converter (distinct primes above the hashed vocabulary, odd multipliers bounded against int64 overflow); the token map folds the vocabulary
onto 300 hashed ids.

    PYTHONPATH=<llama.cpp>/gguf-py python mk_tiny_deepseek_v41.py <tiny-random-deepseek-v4-f32.gguf> <output .gguf>

Reference ids: ref_deepseek_v41.py (numpy) and llama.cpp (llama-cli / llama-simple with -ctk f32 -ctv f32 -fa off).
"""
import math, sys
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

source, output = sys.argv[1:3]
arch = "deepseek41"
vocab, hidden, layers, heads, head_dim, rope_dims, q_rank = 1024, 64, 5, 4, 128, 32, 32
experts, used, shared, ff = 8, 2, 1, 32
index_heads, index_dim, top_k = 8, 64, 4
streams, groups, o_rank, window = 4, 2, 16, 8
ratios = [1, 0, 2, 2, 2]
kv_layers, index_layers = [0, 2], [0, 2, 3]
candidate_layer, candidate_blocks, candidate_size = 2, 3, 4
engram_layers, order, heads_per_order, engram_dim, hashed_vocab, pad_id = [1, 3], 3, 2, 16, 300, 1
rng = np.random.default_rng(0)


def rand(*shape, scale=0.15):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def norm(n):
    return (1.0 + 0.3 * rng.standard_normal(n)).astype(np.float32)


def is_prime(n):
    return n >= 2 and (n == 2 or (n % 2 and all(n % d for d in range(3, math.isqrt(n) + 1, 2))))


# Engram hash constants (EngramLayout.from_args / compute_hash_multipliers of the reference).
seen, sizes = set(), []
for _ in engram_layers:
    for _ in range(order - 1):
        cur = hashed_vocab - 1
        for _ in range(heads_per_order):
            cand = cur + 1
            while not is_prime(cand) or cand in seen:
                cand += 1
            cur = cand
            seen.add(cur)
            sizes.append(cur)
offsets = [sum(sizes[:i]) for i in range(len(sizes))]
bound = max(1, (np.iinfo(np.int64).max // hashed_vocab) // 2)
multipliers = []
for layer in engram_layers:
    values = np.random.default_rng(10007 * layer).integers(low=0, high=bound, size=(order,), dtype=np.int64)
    multipliers += [int(v) * 2 + 1 for v in values]
token_map = [(t * 7) % hashed_vocab for t in range(vocab)]

w = GGUFWriter(output, arch)
w.add_type("model")
w.add_name("tiny-random-deepseek-v41")
w.add_block_count(layers)
w.add_context_length(1024)
w.add_embedding_length(hidden)
w.add_head_count(heads)
w.add_head_count_kv(1)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_rope_freq_base(10000.0)
w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
w.add_rope_scaling_factor(4.0)
w.add_rope_scaling_orig_ctx_len(64)
w.add_rope_scaling_yarn_beta_fast(32.0)
w.add_rope_scaling_yarn_beta_slow(1.0)
w.add_layer_norm_rms_eps(1e-6)
w.add_expert_count(experts)
w.add_expert_used_count(used)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SQRTSOFTPLUS)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
w.add_rope_dimension_count(rope_dims)
w.add_q_lora_rank(q_rank)
w.add_sliding_window(window)
w.add_expert_feed_forward_length(ff)
w.add_expert_shared_count(shared)
w.add_expert_weights_scale(1.5)
w.add_expert_weights_norm(True)
w.add_swiglu_clamp_exp([1.5] * layers)
w.add_swiglu_clamp_shexp([1.5] * layers)
w.add_indexer_head_count(index_heads)
w.add_indexer_key_length(index_dim)
w.add_indexer_top_k(top_k)
w.add_attention_output_group_count(groups)
w.add_attention_output_lora_rank(o_rank)
w.add_attention_compress_ratios(ratios)
w.add_attention_compress_rope_freq_base(160000.0)
w.add_hyper_connection_count(streams)
w.add_hyper_connection_sinkhorn_iterations(20)
w.add_hyper_connection_epsilon(1e-6)
w.add_hash_layer_count(0)
w.add_attention_kv_source_layers(kv_layers)
w.add_indexer_source_layers(index_layers)
w.add_indexer_candidate_source_layer(candidate_layer)
w.add_indexer_candidate_top_k_blocks(candidate_blocks)
w.add_indexer_candidate_block_size(candidate_size)
w.add_key_value(gguf.Keys.PerLayerEmbedding.LAYERS.format(arch=arch), engram_layers, GGUFValueType.ARRAY, GGUFValueType.UINT32)
w.add_ple_ngram_size(order)
w.add_ple_heads_per_ngram(heads_per_order)
w.add_embedding_length_per_layer_input(engram_dim)
w.add_ple_layer_multipliers(multipliers)
w.add_ple_head_offsets(offsets)
w.add_ple_head_vocab_sizes(sizes)
w.add_engram_pad_token_id(pad_id)
w.add_engram_token_map(token_map)

reader = GGUFReader(source)
for key, field in reader.fields.items():
    if not key.startswith("tokenizer.ggml."):
        continue
    kind = field.types[0]
    if kind == GGUFValueType.ARRAY:
        w.add_key_value(key, field.contents(), kind, field.types[-1])
    else:
        w.add_key_value(key, field.contents(), kind)

w.add_tensor("token_embd.weight", rand(vocab, hidden, scale=1.0))
w.add_tensor("output_norm.weight", norm(hidden))
w.add_tensor("output.weight", rand(vocab, hidden))
w.add_tensor("per_layer_token_embd.weight", rand(sum(sizes), engram_dim, scale=1.0))
mix = (2 + streams) * streams
for il in range(layers):
    p = f"blk.{il}."
    t = {
        "attn_norm.weight": norm(hidden), "attn_sinks.weight": rand(heads, scale=1.0),
        "attn_q_a.weight": rand(q_rank, hidden), "attn_q_a_norm.weight": norm(q_rank), "attn_q_b.weight": rand(heads * head_dim, q_rank),
        "attn_kv.weight": rand(head_dim, hidden), "attn_kv_a_norm.weight": norm(head_dim),
        "attn_output_a.weight": rand(o_rank * groups, heads * head_dim // groups), "attn_output_b.weight": rand(hidden, groups * o_rank),
        "hc_attn_fn.weight": rand(mix, streams * hidden), "hc_attn_base.weight": rand(mix, scale=0.5),
        "hc_attn_scale.weight": (0.5 + 0.5 * rng.random(3)).astype(np.float32),
        "hc_ffn_fn.weight": rand(mix, streams * hidden), "hc_ffn_base.weight": rand(mix, scale=0.5),
        "hc_ffn_scale.weight": (0.5 + 0.5 * rng.random(3)).astype(np.float32),
        "ffn_norm.weight": norm(hidden), "ffn_gate_inp.weight": rand(experts, hidden), "exp_probs_b.bias": rand(experts, scale=0.1),
        "ffn_gate_exps.weight": rand(experts, ff, hidden), "ffn_down_exps.weight": rand(experts, hidden, ff), "ffn_up_exps.weight": rand(experts, ff, hidden),
        "ffn_gate_shexp.weight": rand(ff * shared, hidden), "ffn_down_shexp.weight": rand(hidden, ff * shared), "ffn_up_shexp.weight": rand(ff * shared, hidden),
    }
    if il in kv_layers:
        t["attn_compressor_kv.weight"] = rand(head_dim, hidden)
        t["attn_compressor_norm.weight"] = norm(head_dim)
        if ratios[il] > 1:
            t["attn_compressor_gate.weight"] = rand(head_dim, hidden, scale=0.5)
        t["indexer.attn_k.weight"] = rand(index_dim, head_dim)
        t["indexer.k_norm.weight"] = norm(index_dim)
    if il in index_layers:
        t["indexer.attn_q_b.weight"] = rand(index_heads * index_dim, q_rank)
        t["indexer.proj.weight"] = rand(index_heads, hidden)
    if il in engram_layers:
        t["engram_kv.weight"] = rand((streams + 1) * hidden, (order - 1) * heads_per_order * engram_dim)
        t["engram_gate.weight"] = rand(streams, hidden, scale=1.0)
    for name, value in t.items():
        w.add_tensor(p + name, value)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
