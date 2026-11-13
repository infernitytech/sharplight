"""
Builds tiny random Hugging Face checkpoints for the routed-MoE / wave-6 families that have no tiny-random model in ~/models/hf:
Hunyuan MoE (hunyuan_v1_moe), dots.llm1 (dots1, with group-limited routing), EXAONE MoE (exaone_moe), BitNet (bitnet) and
Ministral 3 (ministral3), all with the GPT-2-style tokenizer of tiny-random-GPTNeoXForCausalLM (vocabulary 1024).

    python mk_tiny_moe.py <models/hf directory>

Every model is written to <dir>/tiny-random-<family>/ in fp32 safetensors; the tests skip a family whose directory is missing.
Norm weights, biases and router selection biases are randomized so that every parameter matters.
"""
import os, sys, shutil, torch
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
    "bitnet": transformers.BitNetConfig(**common, intermediate_size=96),
    "ministral3": transformers.Ministral3Config(**common, intermediate_size=96,
                                                rope_parameters={"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
                                                                 "original_max_position_embeddings": 16, "beta_fast": 32.0,
                                                                 "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                                                                 "llama_4_scaling_beta": 0.1}),
}

for name, config in configs.items():
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
