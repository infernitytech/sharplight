"""
Builds tiny random Hugging Face checkpoints for the routed-MoE / wave-6 families that have no tiny-random model in ~/models/hf:
Hunyuan MoE (hunyuan_v1_moe), dots.llm1 (dots1, with group-limited routing), EXAONE MoE (exaone_moe), BitNet (bitnet),
Ministral 3 (ministral3), Arcee Trinity (afmoe), MiMo-V2-Flash (mimo_v2_flash) and Laguna (laguna), all with the GPT-2-style tokenizer of tiny-random-GPTNeoXForCausalLM (vocabulary 1024), plus
tiny-random-bailing-moe-v2-llamacpp: Ling 2.0 re-sized from tiny-random-bailing-moe-v2 so that llama.cpp can convert and run it
(hidden size = heads x head_dim); convert it with convert_hf_to_gguf.py --outtype f32 for the GGUF case.

    python mk_tiny_moe.py <models/hf directory> [family ...]

With family names (for example afmoe mimo-v2-flash) only those transformers-based checkpoints are written and the Hunyuan / Ling
variants below are skipped, so a later addition does not rewrite the checkpoints that existing references were taken from.

Every model is written to <dir>/tiny-random-<family>/ in fp32 safetensors; the tests skip a family whose directory is missing.
Norm weights, biases and router selection biases are randomized so that every parameter matters.
"""
import json, os, sys, shutil, torch
import transformers

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
VOCAB = 1024
common = dict(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
              max_position_embeddings=256, bos_token_id=0, eos_token_id=0, pad_token_id=0, tie_word_embeddings=False)
moe = dict(moe_intermediate_size=32, num_experts_per_tok=2)

configs = {
    "hunyuan-v1-moe": transformers.HunYuanMoEV1Config(**common, intermediate_size=96, num_experts=8, moe_topk=2,
                                                      rope_parameters={"rope_type": "dynamic", "rope_theta": 10000.0, "alpha": 1000.0, "factor": 1.0}),
    "dots1": transformers.Dots1Config(**common, **moe, intermediate_size=96, n_routed_experts=16, n_shared_experts=1, n_group=4,
                                      topk_group=2, first_k_dense_replace=1, norm_topk_prob=True, routed_scaling_factor=2.5),
    "exaone-moe": transformers.ExaoneMoeConfig(**common, **moe, intermediate_size=96, num_experts=8, num_shared_experts=1,
                                               first_k_dense_replace=1, mlp_layer_types=["dense", "sparse", "sparse"],
                                               sliding_window=16, sliding_window_pattern=3,
                                               layer_types=["sliding_attention", "sliding_attention", "full_attention"],
                                               norm_topk_prob=True, routed_scaling_factor=2.5),
    # llama.cpp's dots1 loader has no grouped-query attention (key/value width = query width): multi-head variant for the GGUF reference.
    "dots1-llamacpp": transformers.Dots1Config(**{**common, "num_key_value_heads": 4}, **moe, intermediate_size=96, n_routed_experts=16,
                                               n_shared_experts=1, n_group=4, topk_group=2, first_k_dense_replace=1, norm_topk_prob=True,
                                               routed_scaling_factor=2.5),
    "hy-v3": transformers.HYV3Config(**common, **moe, intermediate_size=96, num_experts=8, num_shared_experts=1, router_scaling_factor=2.5),
    "afmoe": transformers.AfmoeConfig(**{**common, "num_hidden_layers": 4}, moe_intermediate_size=32, num_experts_per_tok=2, intermediate_size=96,
                                      num_experts=8, num_shared_experts=1, num_dense_layers=1, route_scale=2.5, sliding_window=16,
                                      global_attn_every_n_layers=4, mup_enabled=True, route_norm=True),
    "mimo-v2-flash": transformers.MiMoV2FlashConfig(**{**common, "num_hidden_layers": 6, "num_key_value_heads": 1}, moe_intermediate_size=32,
                                                    num_experts_per_tok=2, intermediate_size=96, n_routed_experts=8, routed_scaling_factor=2.5,
                                                    v_head_dim=8, sliding_window=16, attention_value_scale=0.7,
                                                    # Keys llama.cpp's converter reads (the transformers class ignores them).
                                                    swa_head_dim=16, swa_num_attention_heads=4, swa_v_head_dim=8, swa_num_key_value_heads=2,
                                                    hybrid_layer_pattern=[0, 1, 1, 1, 1, 0], topk_method="noaux_tc", layernorm_epsilon=1e-5,
                                                    rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
                                                                     "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}}),
    # Per-head softplus gate, per-layer head counts, one global layer in front of three sliding ones (own rope base and width), YaRN on the global layer.
    "laguna": transformers.LagunaConfig(**{**common, "num_hidden_layers": 4}, moe_intermediate_size=32, shared_expert_intermediate_size=48,
                                        num_experts_per_tok=2, intermediate_size=96, num_experts=8, moe_routed_scaling_factor=2.5,
                                        sliding_window=16, num_attention_heads_per_layer=[4, 6, 6, 6], gating="per-head",
                                        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
                                        rope_parameters={"full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "partial_rotary_factor": 0.5,
                                                                            "factor": 4.0, "original_max_position_embeddings": 64},
                                                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}}),
    # Full attention only with a per-channel gate (the Laguna M.1 layout).
    "laguna-full": transformers.LagunaConfig(**{**common, "num_hidden_layers": 3}, moe_intermediate_size=32, shared_expert_intermediate_size=48,
                                             num_experts_per_tok=2, intermediate_size=96, num_experts=8, moe_routed_scaling_factor=2.5,
                                             sliding_window=0, gating="per-element"),
    "bitnet": transformers.BitNetConfig(**common, intermediate_size=96),
    "ministral3": transformers.Ministral3Config(**common, intermediate_size=96,
                                                rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
                                                                 "original_max_position_embeddings": 16, "beta_fast": 32.0,
                                                                 "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                                                                 "llama_4_scaling_beta": 0.1}),
}

only = sys.argv[2:]
for name, config in configs.items():
    if only and name not in only:
        continue
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p) if n.endswith("weight") else 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * 0.15)
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(0.05 * torch.randn_like(b))
    out = os.path.join(root, f"tiny-random-{name}")
    model.save_pretrained(out, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(tokenizer_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))
    print("wrote", out)
    if name == "mimo-v2-flash":
        # llama.cpp's converter wants the flat rope parameters of the published config (the sliding base is the separate swa_rope_theta key).
        flat = os.path.join(root, "tiny-random-mimo-v2-flash-llamacpp")
        shutil.copytree(out, flat, dirs_exist_ok=True)
        flat_cfg = json.load(open(os.path.join(flat, "config.json")))
        flat_cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 500000.0, "partial_rotary_factor": 0.5}
        flat_cfg["swa_rope_theta"] = 10000.0
        json.dump(flat_cfg, open(os.path.join(flat, "config.json"), "w"), indent=2)
        print("wrote", flat)

if only:
    sys.exit(0)

# Hunyuan MoE for llama.cpp's converter: it reads per-layer lists (moe_intermediate_size, moe_topk, num_shared_expert) and accepts only the
# 32k / 256k contexts of the real model; the weights are those of tiny-random-hunyuan-v1-moe.
hunyuan_source = os.path.join(root, "tiny-random-hunyuan-v1-moe")
hunyuan_out = os.path.join(root, "tiny-random-hunyuan-v1-moe-llamacpp")
os.makedirs(hunyuan_out, exist_ok=True)
for f in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
    shutil.copy(os.path.join(hunyuan_source, f), os.path.join(hunyuan_out, f))
hunyuan_cfg = json.load(open(os.path.join(hunyuan_source, "config.json")))
hunyuan_layers = hunyuan_cfg["num_hidden_layers"]
hunyuan_cfg.update(moe_intermediate_size=[hunyuan_cfg["intermediate_size"]] * hunyuan_layers, moe_topk=[hunyuan_cfg["moe_topk"]] * hunyuan_layers,
                   num_shared_expert=[1] * hunyuan_layers, max_position_embeddings=32768)
json.dump(hunyuan_cfg, open(os.path.join(hunyuan_out, "config.json"), "w"), indent=2)
print("wrote", hunyuan_out)

# Ling 2.0 (bailing_moe_v2) is remote code without a transformers class: write its tensors directly, with q/k/v sizes llama.cpp
# accepts (hidden = heads * head_dim), so that llama.cpp's converter and CPU build provide the reference. Tokenizer and the
# remaining config come from tiny-random-bailing-moe-v2.
from safetensors.torch import save_file
source = os.path.join(root, "tiny-random-bailing-moe-v2")
cfg = json.load(open(os.path.join(source, "config.json")))
cfg.update(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=96, moe_intermediate_size=32,
           num_hidden_layers=3, num_experts=16, num_experts_per_tok=4, n_group=4, topk_group=2, first_k_dense_replace=1,
           tie_word_embeddings=False, max_position_embeddings=256)
H, I, E, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_experts"], cfg["vocab_size"]
qkv = (cfg["num_attention_heads"] + 2 * cfg["num_key_value_heads"]) * cfg["head_dim"]
torch.manual_seed(0)
rand = lambda *shape: torch.randn(*shape) * 0.15
norm = lambda n: 1.0 + 0.3 * torch.randn(n)
t = {"model.word_embeddings.weight": rand(V, H), "model.norm.weight": norm(H), "lm_head.weight": rand(V, H)}
for il in range(cfg["num_hidden_layers"]):
    p = f"model.layers.{il}."
    t.update({p + "input_layernorm.weight": norm(H), p + "post_attention_layernorm.weight": norm(H),
              p + "attention.query_key_value.weight": rand(qkv, H), p + "attention.dense.weight": rand(H, H),
              p + "attention.query_layernorm.weight": norm(cfg["head_dim"]), p + "attention.key_layernorm.weight": norm(cfg["head_dim"])})
    if il < cfg["first_k_dense_replace"]:
        t.update({p + "mlp.gate_proj.weight": rand(I, H), p + "mlp.up_proj.weight": rand(I, H), p + "mlp.down_proj.weight": rand(H, I)})
        continue
    m = cfg["moe_intermediate_size"]
    t.update({p + "mlp.gate.weight": rand(E, H), p + "mlp.gate.expert_bias": 0.05 * torch.randn(E),
              p + "mlp.shared_experts.gate_proj.weight": rand(m, H), p + "mlp.shared_experts.up_proj.weight": rand(m, H),
              p + "mlp.shared_experts.down_proj.weight": rand(H, m)})
    for e in range(E):
        t.update({f"{p}mlp.experts.{e}.gate_proj.weight": rand(m, H), f"{p}mlp.experts.{e}.up_proj.weight": rand(m, H),
                  f"{p}mlp.experts.{e}.down_proj.weight": rand(H, m)})
out = os.path.join(root, "tiny-random-bailing-moe-v2-llamacpp")
os.makedirs(out, exist_ok=True)
save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
    if os.path.exists(os.path.join(source, f)):
        shutil.copy(os.path.join(source, f), os.path.join(out, f))
print("wrote", out)
