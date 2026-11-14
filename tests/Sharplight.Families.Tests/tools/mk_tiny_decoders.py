"""
Builds tiny random Hugging Face checkpoints for decoder families that have no tiny-random model in ~/models/hf:
GPT-2, GPT-J, Nemotron (dense), Gemma 1, Falcon, BLOOM and MPT (both with ALiBi), plus the families of the later bring-up rounds
(EXTRA below: transformers model types configured through AutoConfig.for_model), with the GPT-2-style tokenizer of
tiny-random-GPTNeoXForCausalLM (vocabulary 1024).

    python mk_tiny_decoders.py <models/hf directory> [family ...]

With family names only those models are written (so that a later addition does not rewrite checkpoints existing references were taken from).

Every model is written to <dir>/tiny-random-<family>/ in fp32 safetensors; the tests skip a family whose directory is missing.
Norm weights and biases are randomized so that every parameter matters. Reference ids and logits: hf_greedy_reference.py --logits.
"""
import os, sys, shutil, torch
import transformers
from transformers import (AutoConfig, AutoModelForCausalLM, BloomConfig, GPT2Config, GPTJConfig, NemotronConfig, GemmaConfig, FalconConfig, MptConfig)

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
VOCAB = 1024
COMMON = dict(vocab_size=VOCAB, bos_token_id=0, eos_token_id=0, pad_token_id=0)

configs = {
    "gpt2": GPT2Config(vocab_size=VOCAB, n_positions=256, n_embd=64, n_layer=3, n_head=4, bos_token_id=0, eos_token_id=0),
    "gptj": GPTJConfig(vocab_size=VOCAB, n_positions=256, n_embd=64, n_layer=3, n_head=4, rotary_dim=8, bos_token_id=0, eos_token_id=0),
    "nemotron": NemotronConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                               num_key_value_heads=2, partial_rotary_factor=0.5, max_position_embeddings=256, bos_token_id=0, eos_token_id=0),
    "gemma": GemmaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=1, head_dim=16, max_position_embeddings=256, bos_token_id=0, eos_token_id=0, pad_token_id=0),
    "falcon": FalconConfig(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_kv_heads=2,
                           new_decoder_architecture=True, parallel_attn=True, bias=False, max_position_embeddings=256, bos_token_id=0, eos_token_id=0),
    "bloom": BloomConfig(vocab_size=VOCAB, hidden_size=64, n_layer=3, n_head=4, bos_token_id=0, eos_token_id=0, pad_token_id=0),
    # llama.cpp's converter reads attn_config.alibi_bias_max and clip_qkv; no q/k layer norms.
    "mpt": MptConfig(vocab_size=VOCAB, d_model=64, n_heads=4, n_layers=3, expansion_ratio=2, max_seq_len=256, no_bias=True,
                     attn_config={"attn_type": "multihead_attention", "alibi": True, "alibi_bias_max": 8, "clip_qkv": None, "qk_ln": False,
                                  "softmax_scale": None, "attn_uses_sequence_id": False, "attn_pdrop": 0, "prefix_lm": False},
                     bos_token_id=0, eos_token_id=0, pad_token_id=0),
}

# Later families: name -> (transformers model_type, config arguments on top of COMMON).
EXTRA = {
    # Softmax routing (not renormalized), sigmoid-gated shared expert; layer 1 is dense (mlp_only_layers).
    "qwen2-moe": ("qwen2_moe", dict(hidden_size=64, intermediate_size=96, moe_intermediate_size=48, shared_expert_intermediate_size=80,
                                    num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
                                    mlp_only_layers=[1], max_position_embeddings=256)),
    # The same without the dense layer, which llama.cpp's converter cannot map (source of the qwen2moe GGUF case).
    "qwen2-moe-llamacpp": ("qwen2_moe", dict(hidden_size=64, intermediate_size=96, moe_intermediate_size=48, shared_expert_intermediate_size=80,
                                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, num_experts=4,
                                             num_experts_per_tok=2, max_position_embeddings=256)),
    # StarCoder 1 / SantaCoder: GPT-2 layout with linear weights and one key/value head.
    "gpt-bigcode": ("gpt_bigcode", dict(n_embd=64, n_layer=3, n_head=4, n_positions=256, multi_query=True)),
    # Jais 2: layer norms, squared-ReLU MLP, biases.
    "jais2": ("jais2", dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=256, attention_bias=True, mlp_bias=True)),
    # Granite SWA: sinks, sliding layers (window 8 so that the prompt exceeds it), a NoPE layer and a second rotary base through layer_rope_theta.
    "granite-swa": ("granite_swa", dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=256, sliding_window=8, embedding_multiplier=2.0, residual_multiplier=0.5,
                                        attention_multiplier=0.2, logits_scaling=2.0,
                                        layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
                                        layer_rope_theta=[0, 1000.0, 1000.0, 10000.0])),
    "granitemoe-swa": ("granitemoe_swa", dict(hidden_size=64, intermediate_size=48, shared_intermediate_size=80, num_hidden_layers=4,
                                              num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
                                              max_position_embeddings=256, sliding_window=8, embedding_multiplier=2.0, residual_multiplier=0.5,
                                              attention_multiplier=0.2, logits_scaling=2.0)),
    # Mellum 2: Qwen3-MoE layers, sliding layers with their own rope base, a dense layer (mlp_layer_types); the llama.cpp variant is all-MoE.
    "mellum": ("mellum", dict(hidden_size=64, intermediate_size=96, moe_intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=256, sliding_window=8,
                              layer_types=["sliding_attention"] * 3 + ["full_attention"], mlp_layer_types=["dense", "sparse", "sparse", "sparse"])),
    "mellum-llamacpp": ("mellum", dict(hidden_size=64, moe_intermediate_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                       head_dim=16, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=256, sliding_window=8,
                                       layer_types=["sliding_attention"] * 3 + ["full_attention"])),
    # DBRX: layer norms without bias, fused q/k/v clamped to +-clip_qkv, softmax routing renormalized, fused [E*F, H] experts.
    "dbrx": ("dbrx", dict(d_model=64, n_heads=4, n_layers=3, max_seq_len=256,
                          attn_config=dict(kv_n_heads=2, clip_qkv=0.6, rope_theta=10000.0),
                          ffn_config=dict(ffn_hidden_size=48, moe_num_experts=4, moe_top_k=2))),
    # Mistral 4: latent attention, YaRN, query temperature beyond 16 positions, softmax routing over 2 groups (scored by their best two experts),
    # a leading dense layer and a shared expert.
    "mistral4": ("mistral4", dict(hidden_size=64, intermediate_size=96, moe_intermediate_size=24, num_hidden_layers=3, num_attention_heads=4,
                                  num_key_value_heads=4, kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
                                  n_routed_experts=8, n_group=2, topk_group=1, num_experts_per_tok=2, first_k_dense_replace=1, n_shared_experts=1,
                                  routed_scaling_factor=1.5, max_position_embeddings=256,
                                  rope_parameters=dict(rope_type="yarn", rope_theta=10000.0, factor=4.0, original_max_position_embeddings=16,
                                                       beta_fast=32.0, beta_slow=1.0, mscale_all_dim=1.0, mscale=1.0, llama_4_scaling_beta=0.1))),
}
for name, (model_type, arguments) in EXTRA.items():
    configs[name] = AutoConfig.for_model(model_type, **{**COMMON, **arguments})

# Most families use seed 0 and weights of scale 0.15; the ALiBi models tend to repeat one token or tie their logits at it with those, so BLOOM and MPT pick their own.
seeds = {"bloom": 3, "mpt": 2}
scales = {"bloom": 0.5, "mpt": 0.5}
# Causal-LM classes the auto mapping does not list.
classes = {"mistral4": "Mistral4ForCausalLM"}
only = sys.argv[2:]

for name, config in configs.items():
    if only and name not in only:
        continue
    torch.manual_seed(seeds.get(name, 0))
    model = getattr(transformers, classes[name])._from_config(config) if name in classes else AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p) if n.endswith("weight") else 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * scales.get(name, 0.15))
    out = os.path.join(root, f"tiny-random-{name}")
    model.save_pretrained(out, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        src = os.path.join(tokenizer_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))
    print("wrote", out)
