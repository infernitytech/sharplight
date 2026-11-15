"""
Converts tiny transformers checkpoints (written by mk_tiny_decoders.py) to f32 GGUF with llama.cpp's converter. Families the converter takes as
they are (deepseek_v32, glm_moe_dsa, ...) only need its tokenizer check relaxed; the linear-attention hybrids need more, as the converter only
knows their original remote-code checkpoints: kimi_linear (KimiLinearForCausalLM, as is), bailing_hybrid (renamed
BailingMoeV3ForCausalLM with the original config keys) and minimax (renamed MiniMaxText01ForCausalLM with the original layernorm_* / attn_type_list /
rotary_dim keys; llama.cpp supports only equal alpha factors and beta factors of 1, so the checkpoint must be written with those) and
minimax_m3_vl (the published sparse_attention_config / moe_layer_freq / n_shared_experts / scoring_func keys added) and glm5_next
(the published linear_attn_config / mla_use_nope / first_k_dense_replace keys added). kimi_k3 (Kimi K3 / K3.1 in the published layout,
mk_tiny_kimi_k3.py) is taken as it is, except that latent-attention layers without an indexer are allowed (indexer.types false).
inkling (Inkling in the published layout, mk_tiny_inkling.py) is taken as it is, with the tokens past unpadded_vocab_size marked unused and
the GPT-2 pre-tokenizer of the tiny tokenizer.
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
    if os.environ.get("TINY_PRE", "gpt-2") != "auto":   # auto: the converter's own hash lookup, for the real tokenizers (GLM-4, HunyuanOCR, DeepSeek-OCR)
      TextModel.get_vocab_base_pre = lambda self, tokenizer: os.environ.get("TINY_PRE", "gpt-2")   # TINY_PRE=falcon for the Falcon file that mk_tiny_gguf.py takes its tokenizer from
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
    if kind in ("deepseek_v32", "glm_moe_dsa", "dots1", "exaone_moe"):
        config.setdefault("scoring_func", "sigmoid")   # transformers' default, which the converter does not assume
    if kind == "deepseek_v32":
        from conversion import deepseek
        deepseek.DeepseekV32Model.set_vocab = lambda self: self._set_vocab_gpt2()   # skips the add_bos_token assertion
    elif kind == "glm_moe_dsa":
        from conversion import glm
        glm.GlmMoeDsaModel.set_vocab = lambda self: self._set_vocab_gpt2()   # the tiny tokenizer has no GLM special tokens

    if kind == "inkling_mm_model":
        # The tiny tokenizer is GPT-2's (no Inkling end-of-turn ids), and its ids past unpadded_vocab_size are real tokens, which the
        # converter's padded-vocabulary check refuses: they are marked unused, as in the published vocabulary.
        from conversion import inkling
        import gguf as gguf_module
        base_vocab = TextModel.get_vocab_base
        def vocab(self):
            tokens, types, pre = base_vocab(self)
            unpadded = self.hparams.get("unpadded_vocab_size") or len(tokens)
            return tokens, [gguf_module.TokenType.UNUSED if i >= unpadded else t for i, t in enumerate(types)], pre
        TextModel.get_vocab_base = vocab
        # The converter names Inkling's o200k-style pre-tokenizer; the tiny tokenizer splits like GPT-2.
        inkling_vocab = inkling.InklingModel.get_vocab_base
        inkling.InklingModel.get_vocab_base = lambda self: inkling_vocab(self)[:2] + ("gpt-2",)
        inkling.InklingModel.set_vocab = lambda self: self._set_vocab_gpt2()
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
    elif kind == "minimax_m3_vl":
        # The converter reads the published config keys (sparse_attention_config, moe_layer_freq, n_shared_experts) that transformers
        # folds into flat ones, and the published name mlp.* of the leading dense layers (transformers saves them under block_sparse_moe).
        text = config["text_config"]
        text["sparse_attention_config"] = dict(sparse_num_index_heads=text["index_n_heads"], sparse_index_dim=text["index_head_dim"],
                                               sparse_topk_blocks=text["index_topk_blocks"], sparse_block_size=text["index_block_size"],
                                               sparse_local_block=text["index_local_blocks"],
                                               sparse_attention_freq=[int(t == "minimax_m3_sparse") for t in text["layer_types"]])
        text["moe_layer_freq"] = [int(t == "sparse") for t in text["mlp_layer_types"]]
        text["n_shared_experts"] = 1
        text["scoring_func"] = "sigmoid"
        def dense(name):
            for part in ("gate_proj", "up_proj", "down_proj"):
                name = name.replace(f".block_sparse_moe.{part}.", f".mlp.{part}.")
            return name
        rewrite(dense)
    elif kind == "glm5_next":
        # The converter reads the published linear_attn_config / mla_use_nope / first_k_dense_replace keys.
        text = config["text_config"]
        text["linear_attn_config"] = dict(num_heads=text["linear_num_heads"], head_dim=text["linear_head_dim"],
                                          short_conv_kernel_size=text["linear_conv_kernel_dim"], gate_lower_bound=text["linear_lower_bound"])
        text["mla_use_nope"] = True
        text["first_k_dense_replace"] = text["mlp_layer_types"].index("sparse")
        from conversion import glm
        glm.Glm5NextModel.set_vocab = lambda self: self._set_vocab_gpt2()   # the tiny tokenizer has no GLM special tokens
    elif kind == "qwen4_exp":
        # transformers renames the published "full_attention" layer type to "indexed_attention", which the converter does not know
        # (it would write no QSA compress ratio for the trunk layers).
        from conversion import qwen4exp
        base_parameters = qwen4exp.Qwen4ExpTextModel.set_gguf_parameters

        def set_gguf_parameters(self):
            self.hparams["layer_types"] = ["full_attention" if t == "indexed_attention" else t for t in self.hparams["layer_types"]]
            base_parameters(self)
        qwen4exp.Qwen4ExpTextModel.set_gguf_parameters = set_gguf_parameters
    elif kind == "kimi_k3":
        from conversion import kimi_k3
        kimi_k3.KimiK3Model.set_vocab = lambda self: self._set_vocab_gpt2()   # the tiny tokenizer is not Kimi's tiktoken vocabulary
        set_k31_parameters = kimi_k3.KimiK31Model.set_gguf_parameters

        def set_shared_indexer_parameters(self):
            # The converter expects an indexer on every full-attention layer; the others share the previous selection (indexer.types false).
            dsa = self.hparams["dsa_config"]
            own = list(dsa["dsa_layers"])
            dsa["dsa_layers"] = list(self.hparams["linear_attn_config"]["full_attn_layers"])
            add_types = self.gguf_writer.add_indexer_types
            self.gguf_writer.add_indexer_types = lambda value: add_types([(il + 1) in own for il in range(len(value))])
            try:
                set_k31_parameters(self)
            finally:
                dsa["dsa_layers"] = own
                del self.gguf_writer.add_indexer_types
        kimi_k3.KimiK31Model.set_gguf_parameters = set_shared_indexer_parameters
    elif kind == "deepseek_v4":
        # The converter reads the native checkpoint layout: unprefixed native tensor names (attn.kv_norm, hc_head_fn, embed), the
        # compress_ratios / num_hash_layers keys and the compressed layers' yarn as rope_scaling. Its expert writer repacks FP4 experts to
        # MXFP4; the tiny f32 experts are stacked as they are.
        rates = config["compress_rates"]
        config["compress_ratios"] = [rates.get(t, 0) for t in config["layer_types"]]
        config["num_hash_layers"] = config["mlp_layer_types"].count("hash_moe")
        config["num_nextn_predict_layers"] = 0
        compress = config["rope_parameters"]["compress"]
        config["rope_scaling"] = dict(type="yarn", factor=compress["factor"], beta_fast=compress["beta_fast"], beta_slow=compress["beta_slow"],
                                      original_max_position_embeddings=compress["original_max_position_embeddings"])
        def native(name):
            name = name.removeprefix("model.")
            name = {"embed_tokens.weight": "embed.weight", "hc_head.hc_fn": "hc_head_fn", "hc_head.hc_base": "hc_head_base",
                    "hc_head.hc_scale": "hc_head_scale"}.get(name, name)
            return name[:-len(".attn.norm.weight")] + ".attn.kv_norm.weight" if name.endswith(".attn.norm.weight") else name
        rewrite(native)
        from conversion import deepseek
        import numpy as np
        deepseek.DeepseekV4Model.set_vocab = lambda self: self._set_vocab_gpt2()
        def stack_experts(self, bid, proj, tensor_key):
            names = [f"layers.{bid}.ffn.experts.{e}.{proj}.weight" for e in range(self.hparams["n_routed_experts"])]
            from conversion.base import LazyTorchTensor
            data = np.stack([LazyTorchTensor.to_eager(self.model_tensors[n]()).float().numpy() for n in names])
            self.gguf_writer.add_tensor(self.format_tensor_name(tensor_key, bid), data)
            return names
        deepseek.DeepseekV4Model._write_mxfp4_expert_tensor = stack_experts
    if kind in ("kimi_linear", "bailing_hybrid", "minimax", "minimax_m3_vl"):
        # An unknown model type keeps the converter from loading the config through transformers, whose attribute maps drop the original keys.
        config["model_type"] = kind + "_original"
    json.dump(config, open(path, "w"), indent=1)

    sys.argv = [sys.argv[0], work, "--outtype", "f32", "--outfile", output, *os.environ.get("TINY_CONVERT_ARGS", "").split()]   # e.g. --mtp
    convert_hf_to_gguf.main()
