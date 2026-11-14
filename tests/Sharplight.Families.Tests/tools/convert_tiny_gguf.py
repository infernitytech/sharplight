"""
Converts tiny transformers checkpoints (written by mk_tiny_decoders.py) to f32 GGUF with llama.cpp's converter. Families the converter takes as
they are (deepseek_v32, glm_moe_dsa, ...) only need its tokenizer check relaxed; the linear-attention hybrids need more, as the converter only
knows their original remote-code checkpoints: kimi_linear (KimiLinearForCausalLM, as is), bailing_hybrid (renamed
BailingMoeV3ForCausalLM with the original config keys) and minimax (renamed MiniMaxText01ForCausalLM with the original layernorm_* / attn_type_list /
rotary_dim keys; llama.cpp supports only equal alpha factors and beta factors of 1, so the checkpoint must be written with those).
The converter is patched to take the GPT-2 style tokenizer.json of the tiny checkpoints.

    python convert_tiny_gguf.py <llama.cpp source> <checkpoint directory> <output .gguf>
"""
import json, os, shutil, sys, tempfile

llama, source, output = sys.argv[1:4]
sys.path.insert(0, llama)
sys.path.insert(0, os.path.join(llama, "gguf-py"))

with tempfile.TemporaryDirectory() as work:
    for name in os.listdir(source):
        if name != "reference_logits.json":
            shutil.copy(os.path.join(source, name), work)
    path = os.path.join(work, "config.json")
    config = json.load(open(path))
    kind = config["model_type"]

    def rewrite(rename):
        """Renames (or, where rename returns None, drops) the tensors of the copied checkpoint."""
        from safetensors.torch import load_file, save_file
        weights = os.path.join(work, "model.safetensors")
        tensors = {rename(name): value for name, value in load_file(weights).items()}
        tensors.pop(None, None)
        save_file(tensors, weights, metadata={"format": "pt"})

    import convert_hf_to_gguf
    from conversion.base import TextModel
    TextModel.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
    if kind in ("Dream", "llada"):
        # Masked diffusion: llama.cpp's LLaDA converter leaves the mask token to the tokenizer, which the tiny one lacks.
        from conversion import dream, llada
        model_class = dream.DreamModel if kind == "Dream" else llada.LLaDAModel
        base_parameters = model_class.set_gguf_parameters
        def diffusion_parameters(self, base=base_parameters):
            base(self)
            if kind == "llada":
                self.gguf_writer.add_mask_token_id(self.hparams["mask_token_id"])
        model_class.set_gguf_parameters = diffusion_parameters
    if kind == "llada_moe":
        # LLaDA-MoE's converter writes the original vocabulary's mask token id; the tiny checkpoint's comes from its config.
        from conversion import llada
        base_moe = llada.LLaDAMoEModel.set_gguf_parameters
        def moe_parameters(self, base=base_moe):
            add = self.gguf_writer.add_mask_token_id
            self.gguf_writer.add_mask_token_id = lambda token: None
            base(self)
            self.gguf_writer.add_mask_token_id = add
            add(self.hparams["mask_token_id"])
        llada.LLaDAMoEModel.set_gguf_parameters = moe_parameters
    if kind in ("deepseek_v32", "glm_moe_dsa"):
        config.setdefault("scoring_func", "sigmoid")   # transformers' default, which the converter does not assume
    if kind == "deepseek_v32":
        from conversion import deepseek
        deepseek.DeepseekV32Model.set_vocab = lambda self: self._set_vocab_gpt2()   # skips the add_bos_token assertion
    elif kind == "glm_moe_dsa":
        from conversion import glm
        glm.GlmMoeDsaModel.set_vocab = lambda self: self._set_vocab_gpt2()   # the tiny tokenizer has no GLM special tokens

    if kind in ("kimi_linear", "bailing_hybrid"):
        config["score_function"] = "sigmoid"
        config["n_group"] = config["num_expert_group"]
        config["num_experts_per_tok"] = config["num_experts_per_token"]
        config["norm_topk_prob"] = config.get("moe_renormalize", True)
    if kind == "bailing_hybrid":
        config["architectures"] = ["BailingMoeV3ForCausalLM"]
        config["no_kda_lora"] = True
        config["gated_attention_proj_granularity_type"] = "head_wise"
        config["head_dim"] = config["linear_head_dim"]
        config["moe_shared_expert_intermediate_size"] = config["moe_intermediate_size"] * config["num_shared_experts"]
    elif kind == "minimax":
        alpha = {config[k] for k in ("full_attn_alpha_factor", "linear_attn_alpha_factor", "mlp_alpha_factor")}
        beta = {config[k] for k in ("full_attn_beta_factor", "linear_attn_beta_factor", "mlp_beta_factor")}
        if len(alpha) != 1 or beta != {1.0}:
            sys.exit("llama.cpp's minimax-01 needs equal alpha factors and beta factors of 1")
        config["architectures"] = ["MiniMaxText01ForCausalLM"]
        for part in ("full_attention", "linear_attention", "mlp"):
            config[f"layernorm_{part}_alpha"] = config["mlp_alpha_factor"]
            config[f"layernorm_{part}_beta"] = 1.0
        config["attn_type_list"] = [0 if t == "linear_attention" else 1 for t in config["layer_types"]]
        config["rotary_dim"] = config["head_dim"]
        config["rope_theta"] = config.get("rope_parameters", {}).get("rope_theta", 10000.0)
        from conversion import minimax
        minimax.MiniMaxText01Model._get_suppress_tokens = lambda self: None
        # The converter leaves the layer pattern to llama.cpp's default (a softmax layer every 8th); write it explicitly.
        base_parameters = minimax.MiniMaxText01Model.set_gguf_parameters
        def parameters(self):
            base_parameters(self)
            self.gguf_writer.add_recurrent_layers([t == 0 for t in self.hparams["attn_type_list"]])
        minimax.MiniMaxText01Model.set_gguf_parameters = parameters
        # The decay buffers transformers saves are derived from the slopes, which llama.cpp computes itself.
        rewrite(lambda name: None if name.endswith(("_decay", ".slope_rate")) else name)
    elif kind == "kimi_linear":
        # The original checkpoints name the dense feed-forward of the leading layers mlp.*; transformers saves it under block_sparse_moe.
        def dense(name):
            for part in ("gate_proj", "up_proj", "down_proj"):
                name = name.replace(f".block_sparse_moe.{part}.", f".mlp.{part}.")
            return name
        rewrite(dense)
    if kind in ("kimi_linear", "bailing_hybrid", "minimax"):
        # An unknown model type keeps the converter from loading the config through transformers, whose attribute maps drop the original keys.
        config["model_type"] = kind + "_original"
    json.dump(config, open(path, "w"), indent=1)

    sys.argv = [sys.argv[0], work, "--outtype", "f32", "--outfile", output]
    convert_hf_to_gguf.main()
