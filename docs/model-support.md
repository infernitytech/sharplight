# Model support survey and bring-up plan

Status: survey written 2026-11-13 against llama.cpp `master` (src/llama-arch.cpp `LLM_ARCH_NAMES`, src/models/*.cpp graph builders, tools/mtmd/models) and the Sharplight tree as found in the working copy. Reference runtime for parity outputs: llama.cpp build b11177 (`llama-completion`).

Contents: 1 summary, 2 what Sharplight supports today, 3 building blocks ranked by unlock value, 4 architecture matrix, 5 bring-up waves, 6 Qwen3.8-Next (qwen4exp) deep dive, 7 multimodal projectors, 8 test assets and llama.cpp reference outputs, 9 how the references were produced.

## 1. Summary

llama.cpp lists **172** architecture names (`LLM_ARCH_NAMES`, excluding `(unknown)`; `clip` is the mmproj pseudo-architecture). Classification used throughout:

- **supported**: imports (GGUF and/or HF) and runs end to end on the CPU and Vulkan providers through an existing adapter, with a test or fixture behind it.
- **partial**: the graph can be expressed with ops and kernels Sharplight already has. What is missing is an importer mapping (GGUF arch name, hyper-parameters, tensor names, HF `model_type`) and/or a small adapter feature (LayerNorm, parallel residual, branch scale, routing bias, an activation). No new kernel.
- **missing**: needs a new op/kernel, a new cache kind, a new model topology (encoder, encoder-decoder) or a runtime feature (diffusion, speculative decoding).

| Status | Architectures |
|---|---|
| supported | 128 |
| partial | 5 |
| missing | 39 |
| total | 172 |

Counts as of 2026-11-14, recounted from the matrix in section 4 (the survey started at 15 / 90 / 67; before the first 2026-11-14 round the matrix stood at 79 / 39 / 54, the summary had not been updated). Of the 39 missing ones, 3 are TTS/audio-codec decoders that are out of scope for a text runtime (qwen3tts, pockettts, wavtokenizer-dec). The 2026-11-14 round moved 23 rows to supported: qwen2moe, starcoder, refact, jais, jais2, orion, codeshell, plamo, plamo3, pangu-embedded, maincoder, spark2_5, baichuan, granite_swa, mellum, dbrx, deepseek, bailingmoe, mistral4, plm and cohere2moe (importer mappings, `moe.shared_scale`, latent-decoder query temperature; tiny random models checked against transformers logits/greedy ids and llama.cpp b11177 greedy ids, `tests/Sharplight.Families.Tests/MappedFamilyCases.cs`), plus bert and nomic-bert, which the encoder adapter already ran (`Sharplight.Embeddings.Tests`). The second 2026-11-14 round moved 16 more: the embedding families eurobert, neo-bert, llama-embed, gemma-embedding, modern-bert and jina-bert-v2 (bidirectional decoder layers with symmetric sliding windows and Dense projections after pooling, encoder ALiBi; tiny models vs transformers and llama-server `/embedding`, real ModernBERT-base and EuroBERT-210m vs transformers and jina-embeddings-v2-small-en vs llama.cpp, `tests/Sharplight.Families.Tests/EmbeddingFamilyTests.cs`), and talkie, nanbeige, arctic, deci, grok, maple, step35, kolibri1, ricursive and phimoe (clamped SwiGLU, sparsemixer routing, layer loops, value residual, embedding skip, attention-free and MLP-free layers; tiny GGUFs vs llama.cpp greedy ids, real Nanbeige4.2-3B). ALiBi now runs in the Vulkan attention kernels, so bloom, mpt, refact, jais and Baichuan-13B no longer fall back to the CPU. The third 2026-11-14 round moved 10 more: lfm3, maion-coder, beam, muse-spark and clover_moe (tiny GGUFs vs llama.cpp 8ca5a80, which b11177 predates; LFM3-1.7B vs llama.cpp), t5 (new `encoder_decoder` adapter with relative position bias and cross-attention, also running the HF-only bart and mbart; flan-t5-small vs llama.cpp and transformers), qwen3vl and qwen3vlmoe (DeepStack towers and M-RoPE on the decoder's multimodal graph; Qwen3-VL-2B with an image vs transformers and llama.cpp), qwen2vl (language model, tiny models vs transformers and llama.cpp b11177) and hy_v4 (hyper-connections as the shared `StreamHyperConnections` block, attention sinks and output gate on latent attention; tiny models vs transformers); t5encoder went from missing to partial (mapped, not yet verified).

Headline findings:

1. Qwen3.5/3.6/3.8 **dense** (`qwen35`) works. Three further Qwen architectures matter next: `qwen35moe` (Qwen3.5/3.6-35B-A3B) and `qwen3next` (Qwen3-Next-80B, Qwen3-Coder-Next) are *partial*: Qwen35Adapter plus the MoE sub-graph that DecoderAdapter already has, plus a sigmoid-gated shared expert. `qwen4exp` (what the Hub calls **Qwen3.8-Flash-Next**, HF `qwen4_exp`) is *missing*: it adds hyper-connections, a sparse-attention indexer and n-gram hash embeddings (section 6).
2. The generic decoder adapter already covers a surprising amount (sliding window, softcap, sinks, QK-norm, partial/NoPE rope, per-layer head sizes, KV sharing, PLE, sigmoid MoE with shared expert, longrope/yarn). The bulk of the 90 partial architectures needs only LayerNorm + parallel residual, branch scaling, routing bias/groups and mappings.
3. Weight formats were a second, independent gap and are now closed by quantisation track Q: F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_NL, IQ4_XS, TQ1_0/TQ2_0, MXFP4, NVFP4, FP8 (E4M3 / E5M2 and block-scaled E4M3, section 2.4) and the Bonsai Q1_0/Q2_0/PQ2_0/PTQ1_0 have CPU and Vulkan compute paths (dequantisation bit-exact against ggml, see `tests/Sharplight.Cpu.Tests/GgmlDequantTests.cs`). Only Q8_K (activation-only) and the GGUF `f8_e4m3` type id are registered without kernels. Unsloth `UD-*` quants mix Q2_K/Q3_K/IQ tensors, gpt-oss is MXFP4, so this gated real models regardless of architecture. Concrete example (tensor types read from the GGUF headers): Qwen3.8-Flash-Next `UD-Q4_K_XL` uses only Q8_0, Q5_1, Q4_K, Q5_K, IQ4_NL, BF16 and F32, while the other community quants of the same model contain IQ2_S, IQ3_S or IQ3_XXS tensors (and one unknown ggml type id 42) (the AtomicChat file named `4.27bpw-Q4_K_M` is in fact IQ2_S/IQ3_S/IQ4_NL/Q8_0 plus a Q5_1 table). Quant names do not predict kernel coverage: check the tensor types.

## 2. What Sharplight supports today

### 2.1 Importers and adapters

| Canonical arch | Adapter | GGUF `general.architecture` | HF `model_type` |
|---|---|---|---|
| `decoder` | DecoderAdapter | llama (incl. Mixtral), qwen2, qwen3, gemma, gemma2, gemma3, gemma4, phi2, phi3, olmoe, qwen3moe, muse-glimmer, flint, smollm3, olmo, olmo2, glm4, chatglm, ernie4_5, hunyuan-dense, exaone, exaone4, seed_oss, internlm2, qwen, xverse, openelm, granite, granitemoe, minicpm, starcoder2, command-r, cohere2, falcon, gptneox, gpt2, nemotron, stablelm, apertus, arcee, qwen2moe, starcoder, refact, jais, jais2, orion, codeshell, plamo, plamo3, pangu-embedded, maincoder, spark2_5, baichuan, granite_swa, mellum, dbrx, deepseek, bailingmoe, cohere2moe, qwen3vl, qwen3vlmoe, qwen2vl (and the families listed in GgufArchitectureMappings) | llama, mistral, mixtral, qwen2, qwen3, gemma, gemma2, gemma3_text, gemma4(_text/_unified(_text)), phi, phi3, olmoe, qwen3_moe, muse_glimmer, flint, smollm3, olmo, olmo2, olmo3, glm4, ernie4_5, hunyuan_v1_dense, exaone, exaone4, seed_oss, internlm2, granite, granitemoe, minicpm, starcoder2, cohere, cohere2, falcon, gpt_neox, gptj, gpt2, nemotron, stablelm, apertus, arcee, qwen2_moe, gpt_bigcode, jais2, granite_swa, granitemoe_swa, granitemoeshared, mellum, dbrx, cohere2_moe, qwen3_vl, qwen3_vl_moe, qwen2_vl, qwen2_5_vl (and the families listed in HfArchitectureMappings) |
| `latent_decoder` | LatentDecoderAdapter | deepseek2, minicpm3, mistral4, plm | deepseek_v2, deepseek_v3, kimi_k2, glm4_moe_lite, minicpm3, mistral4, hy_v4 |
| `encoder_decoder` | EncoderDecoderAdapter | t5, t5encoder | t5, mt5, umt5, bart, mbart |
| `encoder` | EncoderAdapter | bert, nomic-bert | bert, roberta, xlm-roberta, nomic_bert |
| `qwen3_5` | Qwen35Adapter (+ Qwen35Multimodal, VisionTowerGraph) | qwen35 (+ `clip` projector `qwen3vl_merger`) | qwen3_5, qwen3_5_text |
| `nemotron_h` | NemotronHAdapter | nemotron_h, nemotron_h_moe | nemotron_h |
| vision tower | VisionTowerGraph | clip (only `qwen3vl_merger`, with or without DeepStack layers) | embedded in Qwen3.5 and Qwen3-VL(-MoE) HF checkpoints |

Format-specific code lives only in Sharplight.Gguf / Sharplight.HuggingFace / Sharplight.Onnx; adding a family is one `IGgufArchitectureMapping` / `IHfArchitectureMapping` class plus a `Register` call (GgufArchitectureMapping.cs, HfArchitectureMapping.cs). The ONNX importer is generic (ONNX zoo tests: detection, SAM, diffusion VAE/UNet, whisper decoder, CLIP/SigLIP, ...), so BERT-style encoders are reachable through ONNX today but not through GGUF/HF.

### 2.2 What DecoderAdapter expresses (documented in its XML remarks)

GQA/MQA; per-head QK RMS norm (weighted or unweighted), value norm; partial rotary, neox/gptj pairing, local rope base, yarn/llama3/longrope frequency factors, attention scale; sliding window with per-layer pattern; logit soft-capping and final soft-capping; attention sinks; sigmoid attention output gate; per-layer head dim / kv heads / KV sharing / NoPE layers; k = v layers; sandwich norms (`attention.out_norm`, `mlp.out_norm`); embedding scale, embedding norm, logit scale; gated or ungated MLP (silu/gelu/gelu_tanh/relu) with optional biases; MoE (softmax or sigmoid scores, renormalisation, weight scale, rank-3 expert weights through MatMulId, ungated shared expert, leading dense layers); Gemma 3n/4 per-layer embeddings. Waves 0 and 2 added: layer norm (with bias, or non-parametric), parallel residual, residual-branch scale, ReLU^2 / xIELU / SwiGLU-OAI activations, learned absolute positions, q/k norm after rope, per-layer head counts, q/k/v clamping, biased output head (`DecoderNorm`, `IMlpActivation` and the `Sequential`/`Parallel` layer topology in Sharplight.Architectures). Later rounds added routing bias and groups, ALiBi (CPU and Vulkan kernels: scalar, split-K decode and flash attention), bidirectional decoder layers (`attention.causal` = false: embedding graph only, symmetric sliding window, `embedding_projection.{i}` Dense layers after pooling), `moe.shared_scale` and query temperature on the latent decoder as well.

### 2.3 Graph ops available (Sharplight.Core StandardOps)

Attention, CachedAttention, CacheWrite, Rope, RmsNorm, LayerNorm, GroupNorm, L2Norm, Linear, MatMul, BatchMatMul, **MatMulId** (MoE), TopK, Softmax, Gather*/Scatter*, CumSum, Trilu, **CausalConv1d**, **GatedDeltaRule** (Qwen3.5 linear attention), **SelectiveScan** (Mamba-2 SSD), the ONNX-style elementwise/reduction/spatial set (Conv, Pool, Resize, GridSample, ...). **SelectiveScan** also covers Mamba-1 (A as [H, N]), **Wkv7**, **Wkv6** (RWKV recurrences); all of them have the multi-sequence form. CachedAttention takes `v_dims` (values = the first v_dims entries of each cache row, so the MLA latent cache serves as K and V). Not present: KDA/lightning variants, attention score bias, sparse/indexed attention, int hash ops.

### 2.4 Weight formats with compute kernels

CPU (Quant.cs and its partials Quant.KQuants/Iq/Fp4/Fp8/Ternary.cs) and Vulkan (Sharplight.Shaders/Kernels/common.slang, quant_k.slang, quant_iq.slang, quant_fp8.slang; the lowering table is GraphLowerer.Support.cs): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4; Bonsai Q1_0, Q2_0, Q2_0_G128, Tern1_G128; FP8 `f8e4m3`, `f8e5m2` (one byte per value) and the block-scaled `f8e4m3_b32` { f32 d; u8 q[32] } / `f8e4m3_b128` { f32 d; u8 q[128] } (value = d * e4m3(q), blocks along K). The IQ lookup grids come from ggml-common.h via tools/gen-quant-grids.py (src/Sharplight.Cpu/Grids, Kernels/grid_*.slang). Q8_K fails at load.

FP8 Hugging Face checkpoints (the vLLM formats; `HfFp8Weights` in Sharplight.HuggingFace): an `F8_E4M3` `X.weight` with a separate scale is converted while it is read (nothing cached) into a block-scaled type, before the architecture mapping runs, so fused-projection splits, V-head row reorders and expert stacking work on whole rows unchanged (transposes and column permutations that cut blocks decode to BF16 first).
- DeepSeek-V3 / Qwen FP8 (`quantization_config.quant_method = "fp8"`, `weight_block_size [128, 128]`): BF16/F32 `X.weight_scale_inv` [ceil(N/128), ceil(K/128)], a multiplier despite its name.
- ModelOpt (`hf_quant_config.json`, e.g. Nemotron-3-Nano-30B-A3B-FP8) and compressed-tensors: `X.weight_scale` per tensor, per channel or per block (also a multiplier). `X.input_scale` is ignored (activations are not statically quantized), as are KV-cache scales (`k_scale` / `v_scale`) and `kv_cache_quant_algo`: caches use the `--kv-type` setting.
- Target: `f8e4m3_b128` when K and the scale's K granularity are multiples of 128, else `f8e4m3_b32` (the scale repeated in every block it covers, e.g. Nemotron's expert down projections with K = 1856), else BF16. Modules a checkpoint does not quantize are BF16 in the file and pass through. Scale-less `F8_E4M3` / `F8_E5M2` tensors load as `f8e4m3` / `f8e5m2`.
- Kernels: the f32 matvecs decode through half bits (load8_fp8), the f16 tensor-core GEMMs (gemm_cm2, the MoE grouped GEMM) through packed halves (load16h_fp8 / load8h_fp8). Batches of 2-16 rows run on the skinny f16 tensor-core GEMM (`matmul_skinny.slang`, weights read once for all rows, for matrices with at least 1024 output rows; `SHARPLIGHT_SKINNY_GEMM=0` returns them to the multi-row f32 matvec), batches of 17+ rows (`SHARPLIGHT_FP8_GEMM_MIN_ROWS`) of the block-scaled `f8e4m3_b128` / `f8e4m3_b32` matmuls (K % 128 == 0) on the fp8 tensor-core GEMM (`gemm_fp8.slang`, device feature `matrix_fp8`: activations quantized to E4M3 with one f32 scale per 128 values, 16x16x32 f8E4m3 cooperative matrices with f32 accumulation, one scaling per weight block); `SHARPLIGHT_FP8_GEMM=0` or `--disable-features matrix_fp8` return them to the f16 GEMM. This covers every scaled HF FP8 weight (per-tensor and per-channel scales become block scales on import); unscaled `f8e4m3` / `f8e5m2` tensors and the MoE expert GEMMs stay on the f16 GEMMs. Tested: tests/Sharplight.Cpu.Tests/Fp8DequantTests.cs, Vulkan QuantFormatTests, tests/Sharplight.HuggingFace.Tests/Fp8ImportTests.cs, and Families Fp8CheckpointTests (Qwen3.8-27B-FP8 vs Q8_0, Nemotron-3-Nano-30B-A3B-FP8 vs Q4_K_M; `SHARPLIGHT_TEST_LARGE=1`).

### 2.5 Models that are actually tested

| Test project | Models |
|---|---|
| Sharplight.Families.Tests (HF ids + GGUF text vs transformers/llama.cpp) | gemma-4-E2B-it (HF, BF16, Q8_0), NVIDIA-Nemotron-3-Nano-4B (HF, Q8_0), tiny-random-nemotron-h, NVIDIA-Nemotron-3.5-Bolt-E4B (UD-Q4_K_XL), tiny-random-muse-glimmer, Muse-Glimmer-30B (UD-Q4_K_XL), tiny-random-flint |
| Sharplight.HuggingFace.Tests (HF vs GGUF canonical parity, DecoderParityTests) | Llama-3.2-1B-Instruct, Qwen2.5-0.5B-Instruct (Q8/f16), Qwen3-0.6B, SmolLM2-360M-Instruct, Phi-3.5-mini-instruct, OLMoE-1B-7B-0924-Instruct, gemma-2-2b-it, gemma-3-1b-it, gemma-4-E2B-it, gemma-4-12B-it, Nemotron-3-Nano-4B, Muse-Glimmer-30B, Bonsai-1.7B, Ternary-Bonsai-1.7B/2-2B |
| Qwen35ParityTests, Sharplight.Vision.Tests | Qwen3.5-0.8B (HF + BF16 GGUF + mmproj), Qwen3.8-27B mmproj (BF16) vs llama.cpp |
| Sharplight.Gguf.Tests | metadata/tensor parsing incl. Qwen3.8-27B Q8_0 / Q4_K_M / NVFP4 files, synthetic GGUFs |
| Sharplight.Vulkan.Tests / Cpu.Tests | op-level: SSM (Nemotron-H shapes), MoE, LLM ops, IQ4, Bonsai quant, fusion, submission budget |
| Sharplight.Onnx.Tests | ONNX model zoo |

Waves 0 and 2 (tests/Sharplight.Families.Tests/DecoderFamilyCases.cs): Mistral-7B-Instruct-v0.3, SmolLM3, OLMo 1/2, GLM-4, ERNIE-4.5, Hunyuan, EXAONE 3.5/4, Seed-OSS, InternLM2.5, OpenELM, Granite 3.x dense and MoE, MiniCPM4.1, Mixtral (tiny), Phi-2, StarCoder2, Command-R/Aya, Cohere2, Falcon (tiny), Pythia, GPT-J (tiny), GPT-2, Nemotron-Mini, StableLM 2, Apertus, AFM-4.5B, Gemma 1 (tiny). Not covered by any model test: Mamba/RWKV models. Wave 4 (FamilyGenerationTests): DeepSeek-V2-Lite-Chat and MiniCPM3-4B Q4_K_M, tiny-random deepseek-v3.1 / glm-4.7-flash / minicpm3.

## 3. Building blocks ranked by unlock value

Score = sum over non-supported architectures that need the block of a 1-3 popularity weight (3 = widely deployed in 2025-26, 1 = niche). Importer-only items (`MAP`, `MMPROJ`) are excluded; they are listed in the matrix.

### 3.1 Top 10 (curated from the score table below)

| # | Block | Codes | Unlocks |
|---|---|---|---|
| 1 | MoE inside the hybrid adapter + sigmoid-gated shared expert + qwen3next layouts | MOEHYB, SGATE, QKVZ | qwen3next, qwen35moe, clover_moe; first half of qwen4exp. Qwen3.6-35B-A3B and Qwen3-Coder-Next are the two real test models. Small: reuse DecoderAdapter's router/expert sub-graph. |
| 2 | Aux-loss-free routing (selection bias, sigmoid, renorm) + group-limited top-k | RBIAS, RGRP | glm4moe, ernie4_5-moe, bailingmoe2, dots1, lfm2moe, minimax-m2, hy_v3, exaone-moe, mimo2, afmoe, laguna, beam, kolibri1, smallthinker + every DeepSeek-V3 style model. NemotronH already has the bias. |
| 3 | LayerNorm + parallel residual in the decoder adapter | LN, PAR | phi2, falcon, gptj, gptneox, starcoder(2), olmo, command-r, cohere2(moe), nemotron, stablelm, orion, codeshell, jais2, dbrx, arctic. Ops exist; wiring only. |
| 4 | Multi-head latent attention (latent KV cache, decoupled rope, absorbed decode) | MLA | deepseek2 (V2/V3/R1/Kimi-K2/GLM-4.7-Flash/Moonlight), minicpm3, mistral4, plm, deepseek32, glm-dsa, kimi-linear. The single largest new kernel/cache feature. |
| 5 | Branch/residual scaling and muP multipliers (+ ReLU^2, xIELU, SwiGLU-OAI activations) | RES, ACT | granite, granitemoe, granitehybrid, falcon-h1, minicpm, arcee, jais2, nemotron, apertus, gpt-oss. Elementwise graph only. |
| 6 | Mamba-2 hybrid layer factory (generalise NemotronHAdapter) and pure Mamba-2 | M2H, ADAPTM2 | granitehybrid (Granite 4.0-H), falcon-h1, mamba2, jamba (with M1), plamo2. |
| 7 | Sparse-attention indexer (DSA/QSA) | IDX | qwen4exp, deepseek32, glm-dsa, glm5-next, kimi-k31, minimax-m3, dots3note, mistral-large4, hy_v4. Dense masked fallback first, sparse kernel later. |
| 8 | Hyper-connections | HC | qwen4exp, deepseek4/41, xing4_0, kimi-k3/k31, glm5-next, metis16. Pure graph (norm, low-rank gate, scatter). |
| 9 | Bidirectional encoder adapter with pooling heads | ENC, POOL (+ LN, APOS, ALIBI) | bert, modern-bert, nomic-bert(-moe), neo-bert, jina-bert-v2/v3, eurobert, gemma-embedding(2), t5encoder: embeddings and rerankers, the main RAG workload. |
| 10 | Quantisation kernels: Q2_K/Q3_K, IQ1-IQ3, MXFP4/NVFP4, TQ | (formats) | Every UD-quant of a large model, gpt-oss (MXFP4), BitNet (TQ2_0). Independent of architecture; unblocks the sub-100 GB Qwen3.8-Flash-Next files (IQ1/IQ2/IQ3 based, 66-94 GB). |

Next in line: the remaining vision towers on the decoder's multimodal graph (glm4v, paddleocr, Qwen2/2.5-VL), KDA/lightning recurrences (kimi-linear, bailingmoe3, minimax), the sparse regime of the DSA/QSA indexer (deepseek32, glm-dsa, hy_v4 beyond `index_topk`), Gemma 3n (AltUp, LAuReL), masked-diffusion decoding.

### 3.2 Block catalog and score table

| Code | Block | Archs needing it | Score | Effort | Sharplight today |
|---|---|---|---|---|---|
| `LN` | **LayerNorm (mean-centred, bias) in the generic decoder adapter**. Falcon, GPT-NeoX/J, Phi-2, StarCoder(2), OLMo-1, Command-R/Cohere (no bias), Nemotron-4 (1+w), Bloom, MPT, BERT. | 26 | 34 | S | LayerNorm op exists (used by VisionTowerGraph); decoder adapter has RMS only |
| `RBIAS` | **Aux-loss-free routing: selection bias (e_score_correction_bias), sigmoid scores, optional renorm + scale**. top-k chosen on score+bias, weights taken from the unbiased score. Nemotron-H adapter has it; the generic decoder adapter only has softmax/sigmoid without bias. | 18 | 31 | S | **done** in the shared `MoeOptions`/`MoeRouter`/`MoeBlock` (decoder, hybrid, NemotronH, Qwen3.5 and latent_decoder adapters): `moe.router_activation` softmax / sigmoid / softmax_topk / sqrt_softplus, `mlp.router_bias` selection bias, `moe.normalize_weights`, `moe.weights_scale`, `moe.weights_before_experts` |
| `MLA` | **Multi-head latent attention**. q_lora / kv_lora compression, decoupled rope dims, absorbed form with a latent KV cache (llama.cpp caches kv_lora_rank + rope_dim per token) and, for prefill, the expanded MHA form; yarn mscale on the softmax scale. | 15 | 24 | L | **done** (LatentAttention block in `latent_decoder`: latent cache [kv_lora normed latent, roped k_pe], absorbed queries via per-head MatMulId, CachedAttention `v_dims`); prefill also uses the absorbed form (expanded prefill and an MQA-shaped decode kernel still to do) |
| `RGRP` | **Group-limited top-k routing (n_group / topk_group)**. experts split into groups, best groups chosen by (top-2 sum of) scores, then top-k inside them (DeepSeek-V3, Ling, Kimi-K2, GLM-4.5 n_group=1 degenerates). | 11 | 22 | S | **done** in `MoeRouter` (`moe.group_count`, `moe.group_used_count`, `moe.group_score` top2_sum / max); latent_decoder uses it too |
| `IDX` | **Sparse-attention indexer (DSA / QSA / lightning indexer)**. a small per-layer scoring attention (few heads, head_dim 128, q/k norm) picks top-k visible tokens (budget 2048; QSA pools keys in blocks of compress_ratio) and the main attention only attends to them. | 11 | 20 | L | missing: scoring op + gather/top-k + sparse attention path (a masked dense fallback is enough for correctness) |
| `RES` | **muP-style scalings: residual/branch scale, attention multiplier, embedding & logit scale**. Granite (residual_scale, attention_scale, embedding_scale, logits_scaling), MiniCPM (scale_depth/sqrt(L)), Falcon-H1 (mup multipliers), Cohere logit scale, MiniMax. | 10 | 19 | S | embedding_scale, logit_scale, attention.scale exist; residual (branch) scale missing |
| `HC` | **Hyper-connections (multi-stream residual)**. hc_count (4) parallel residual streams [hidden, hc, T]; before each mixer/MoE a grouped RMS norm + low-rank (hc_lowrank 320) gate + sigmoid collapses the streams, results are scattered back with learned inject weights; final head mixer replaces output_norm (DeepSeek-V4 uses Sinkhorn-normalised mixing instead of low-rank). | 9 | 16 | M | **done**: qwen4exp's gated low-rank form in `HyperConnections`; the per-token collapse/expansion form (Hunyuan v4, DeepSeek-V4 mHC with optional Sinkhorn mixing) as the shared block `StreamHyperConnections` (`hyper_connections.*`), used by the latent decoder |
| `POOL` | **Pooling and heads for embedding/rerank models**. mean / CLS / last-token pooling, L2 normalise, classification (rank) head, token-type embeddings. | 11 | 15 | S | missing in the runtime API (Qwen3-Embedding works as a decoder with last-token pooling only if exposed) |
| `ENC` | **Bidirectional encoder graph**. non-causal attention, post-LN BERT blocks / ModernBERT local-global rope / NomicBERT SwiGLU, no KV cache, batch of independent sequences. | 11 | 15 | M | Attention op supports non-causal use (vision tower); no text encoder adapter, no GGUF/HF text-encoder mapping |
| `PAR` | **Parallel residual (attention and MLP read the same normed input)**. x + attn(n(x)) + mlp(n(x)); Falcon, GPT-NeoX/J, Phi-2, Cohere, Command-R, StableLM-12B, Cohere2. | 9 | 12 | S | missing in DecoderAdapter (graph wiring only) |
| `ACT` | **Extra activations**. ReLU^2 ungated MLP (Arcee, Nemotron-4, Jais2, PLM), xIELU (Apertus), clamped SwiGLU-OAI with alpha (gpt-oss, MiniMax-M3, Ricursive), GELU-new variants. | 7 | 12 | S | **done**: ReLU^2, xIELU and the clamped SwiGLU-OAI (`swiglu_oai`, `mlp.swiglu_limit`) |
| `SGATE` | **Sigmoid-gated shared expert**. shared expert output multiplied by sigmoid(h . w_gate) (llama.cpp ffn_gate_inp_shexp, [hidden]). Decoder's shared expert is added ungated. | 5 | 12 | S | missing (one Linear + Sigmoid + Mul) |
| `MROPE` | **M-RoPE / interleaved M-RoPE with sections**. rope.sections over (t,h,w[,e]); Qwen35Adapter has it for its own graph, DecoderAdapter does not. | 6 | 10 | S | **done** in the decoder adapter too (`rope.sections`, `rope.sections_interleaved`; 4 position axes in the multimodal graph, plain RoPE on text-only graphs): qwen3vl, qwen3vlmoe, qwen2vl |
| `MOEHYB` | **MoE FFN inside the hybrid (gated-delta + attention) adapter**. Qwen35Adapter layers use a dense SwiGLU MLP; qwen3next/qwen35moe/qwen4exp replace it by routed experts (softmax top-k, norm-topk, 256-512 experts) plus a shared expert. The router/expert code already exists in DecoderAdapter (MatMulId, TopK, Softmax) and NemotronHAdapter. | 4 | 10 | S | ops exist; factor the MoE sub-graph out of DecoderAdapter and call it from Qwen35Adapter |
| `M2H` | **Mamba-2 inside a generic hybrid layer stack (parallel or interleaved with attention/MoE)**. Granite-4.0-H (interleaved Mamba2 + attention + MoE with shared expert), Falcon-H1 (attention and Mamba2 in parallel in one layer, multipliers), Jamba, PLaMo-2. | 5 | 8 | M | SelectiveScan + CausalConv1d exist and NemotronHAdapter composes them; needs a shared per-layer block factory |
| `KDA` | **Kimi Delta Attention (per-channel-gated delta rule) / lightning variants**. gated delta rule with a per-key-channel decay instead of a per-head scalar (Kimi-Linear, Ling-3/Ring-linear, GLM5-next, Kimi K3). | 5 | 8 | M | GatedDeltaRule exists with scalar gate; needs per-channel gate variant |
| `TMP` | **Attention temperature tuning and chunked-local attention**. Llama 4 NoPE layers scale q by log(floor((pos+1)/floor_scale)+1)*attn_scale+1; iRoPE chunked attention mask; Mistral-3/4 use the same scale. | 3 | 6 | S | **done**: per-layer query temperature (`attention.temperature_scale/length/offset/layers`) and chunked attention (`attention.chunk_size/chunk_layers`, CachedAttention `chunk` on CPU and Vulkan) |
| `ALIBI` | **ALiBi / additive score bias in cached attention**. Bloom, MPT, Baichuan-13B, Refact, Jais, Jina-BERT-v2; same plumbing serves T5 relative-position buckets. | 5 | 5 | M | **done** as the provider-neutral attribute `alibi_slopes` (float[Hq]) on CachedAttention / Attention (`attention.alibi_max_bias` in the decoder); the CPU provider and the Vulkan attention kernels (scalar, split-K decode, flash; slopes baked in as a specialization) implement it, for causal and bidirectional (symmetric |i - j|) attention |
| `CONV` | **Gated short-conv layer (LFM2/LFM3)**. in_proj -> (B, C, x); y = C * conv_k3(B * x); out_proj; interleaved with GQA layers and (lfm2moe) MoE. | 3 | 5 | S | CausalConv1d exists; needs adapter + state plumbing |
| `YARNM` | **YaRN attention factor on MLA softmax scale**. mscale^2 folded into the MLA scale (rope.attention_scale exists for non-MLA). | 2 | 5 | S | **done** with MLA (`attention.scale` of latent_decoder) |
| `DSTACK` | **DeepStack vision injection**. Qwen3-VL adds vision-tower intermediate features to the hidden state of the first N decoder layers. ClipGgufMapping throws NotSupported for is_deepstack_layers today. | 2 | 5 | M | **done**: `vision.deepstack_layers` taps in VisionTowerGraph (merged, normed, projected, appended to the embedding row), added after decoder layer k in the multimodal graph |
| `NGRAM` | **PLE n-gram hash embeddings**. Qwen4Exp has one such layer (HF `ple_layer_ids` [2], GGUF `ple.layers` [1]): the last 3 token ids are hashed per head (16 heads = (ngram_size-1) x heads_per_ngram 8, each with its own multiplier, row offset and ~20M-row slice) into one flat [160, ~320M] table (38 GB in Q5_1), then a short causal conv (k=4) over the 4 hyper-connection streams and key/value projections add the result into the stream. Gather + CausalConv1d exist; the 64-bit hash must run host-side (row ids as an int input) or as an int64 op. | 2 | 5 | M | missing: int hash/row-id computation + conv-history state per sequence; the table must be memory-mapped, not materialised |
| `WKV7` | **RWKV-7 time-mix (generalised delta rule)**. token-shift lerps, LoRA-parametrised decay/in-context-lr/gate, WKV7 recurrence, LN, channel-mix ReLU^2. | 2 | 5 | L | missing: Wkv7 op + RWKV adapter |
| `DIFF` | **Masked-diffusion decoding loop**. bidirectional transformer + iterative unmasking sampler (Dream, LLaDA, RND1). | 4 | 4 | L | missing: runtime feature, not graph |
| `APOS` | **Learned absolute position embeddings**. GPT-2, StarCoder, BERT family: token + position (+ token-type) embedding sum. | 3 | 4 | S | Gather + Add exist; adapter lacks it |
| `M1` | **Mamba-1 selective scan**. per-channel A [d_inner, d_state], dt_rank low-rank dt projection, x_proj; conv k=4; Mamba, Falcon-Mamba, Jamba, Codestral-Mamba. | 3 | 4 | M | missing: SelectiveScan1 op (CPU + Slang) and its chunked prefill path |
| `SPECDEC` | **Speculative-decoding drafters (EAGLE3, DFlash, MTP assistants)**. draft graphs that consume target hidden states. | 3 | 3 | L | partial: MTP for the qwen3_5 family and NextN heads of DeepSeek-V3 (`latent_decoder`) / GLM-4.5 (`decoder`, uniform layers) from HF and GGUF (`MtpDraftHead` + `MtpSpeculator`, rollback of recurrent state, shared embedding / LM head, device-side verification); verified on tiny random checkpoints only (no real DeepSeek / GLM checkpoint was run); EAGLE3 / DFlash drafters missing |
| `RELPOS` | **Relative-position bias buckets**. T5 style learned bucketed bias added to scores. | 2 | 3 | M | **done**: `position_bias` / `position_bias_range` attention attributes on CPU and Vulkan (`RelativePositionBias`) |
| `WKV6` | **RWKV-6 time-mix**. data-dependent decay with LoRA, WKV6 recurrence, token shift, group norm. | 2 | 3 | M | missing: Wkv6 op (WKV7 kernel can share structure) |
| `LOOP` | **Weight-shared layer loops**. Nanbeige (num_loops), HRM-Text (H/L cycles): the same layer repeated with distinct KV slots. | 2 | 3 | S | graph feature: reuse weights, separate cache keys |
| `QKVZ` | **qwen3next fused projection layouts**. qwen3next stores in_proj_qkvz / in_proj_ba with head-interleaved layout (llama.cpp ssm_in + ssm_beta_alpha, and a per-layer split rule in the loader); Qwen3.5 uses separate qkv/z/b/a. | 1 | 3 | S | mapping + lazy row permutation in Gguf/HF importers |
| `MXFP4` | **MXFP4 / NVFP4 weight formats**. gpt-oss ships MXFP4 experts; many 2026 checkpoints ship NVFP4. CPU and Slang decoders, matvec, GEMM and MatMulId exist and are tested against ggml (quantisation track Q); gpt-oss still needs expert bias and SwiGLU-OAI from the decoder track. | 1 | 3 | M | **done** (kernels; gpt-oss GGUF and HF MXFP4 `*_blocks`/`*_scales` experts import) |
| `EBIAS` | **Per-expert biases in MatMulId**. gpt-oss experts have gate/up/down biases. | 1 | 3 | S | **done** (`experts.{gate,up,gate_up,down}.bias` gathered per selected expert) |
| `LIN` | **Lightning / linear attention with fixed decay**. MiniMax-01/M3 lightning attention: block-wise kv state with exponential decay, softmax layer every 8th. | 2 | 2 | M | missing: new recurrent op (close to GatedDeltaRule without delta correction) |
| `MOELATENT` | **MoE in a latent space (down/up projection around the experts)**. Nemotron-H MoE latent (supported), Muse-Spark, Maion-Coder. | 2 | 2 | S | exists in NemotronHAdapter |
| `ALTUP` | **Gemma 3n AltUp (4 parallel streams with predict/correct)**. plus LAuReL low-rank residual and activation sparsity; PLE and KV sharing already exist. | 1 | 2 | M | missing |
| `LAUREL` | **Gemma 3n LAuReL**. low-rank learned residual. | 1 | 2 | S | missing |
| `XATTN` | **Cross-attention encoder-decoder (T5)**. encoder runs once, decoder cross-attends to cached encoder K/V; relative-position bias buckets; no RoPE. | 1 | 2 | L | **done**: `encoder_decoder` adapter (T5, BART, mBART), encoder pass in InferenceSession, Vulkan cross-attention kernels |
| `ADAPTM2` | **Pure Mamba-2 model adapter**. mamba2-130m...: all layers SSM; a degenerate case of NemotronHAdapter once its mapping accepts `mamba2`. | 1 | 2 | S | adapter exists in effect; mapping + hparam names missing |
| `TQ` | **Ternary TQ1_0/TQ2_0/I2_S formats**. BitNet b1.58 official GGUFs. | 1 | 2 | M | **done** for TQ2_0 (bitnet-b1.58-2B-4T runs on CPU and Vulkan); I2_S not yet |
| `SUBNORM` | **BitNet sub-layer norms**. extra RMS norms before attn-out and ffn-down projections. | 1 | 2 | S | **done** (`attention.sub_norm.weight`, `mlp.sub_norm.weight`) |
| `LAYERVAR` | **Per-layer structural variation (skip attention / FFN, NAS)**. Deci / Nemotron-51B layers without attention or with linear replacement. | 1 | 1 | S | partly (per-layer arrays exist for heads/kv) |
| `SPARSEMIXER` | **Phi-3.5-MoE sparsemixer routing**. 2-step masked top-2 with jitter thresholds. | 1 | 1 | S | missing |
| `GROVE` | **GroveMoE adjugate/chunk experts**. small shared 'adjugate' experts per expert group. | 1 | 1 | M | missing |
| `VEXPERT` | **CogVLM visual-expert weights**. separate qkv/ffn for image tokens. | 1 | 1 | M | missing |
| `MOVA` | **MoVA value experts**. V = sum_k w_k silu(W_k x) routed per token. | 1 | 1 | M | missing |

Effort: S = adapter/graph wiring (days), M = new op or runtime plumbing (about a week), L = new kernel family or model topology (weeks).

Importer-only codes used in the matrix: `MAP` = GGUF arch/hparam/tensor mapping and HF model_type mapping only; `MAPMOE` = expert tensors under an existing arch name (Mixtral); `MMPROJ` = needs a projector/tower mapping (section 7).

## 4. Architecture matrix

Columns: llama.cpp arch id | representative models | llama.cpp building blocks (from src/models/<arch>.cpp and hparams loading) | status | what Sharplight lacks (block codes, section 3.2) | HF `model_type` | test assets downloaded (section 8).

Trait abbreviations: GQA/MQA/MHA, MLA (latent attention), SWA (sliding window), NoPE (layers without rope), QKN (QK norm), pRoPE (partial rope), MRoPE, Agate (attention output gate), RMS/LN (norm type), postN (sandwich norms), SwiGLU/GeGLU/ReLU2, MoE[...] (routing: sm = softmax, sg = sigmoid, bias, grp = group-limited, shared), M1/M2 (Mamba 1/2), GDN, KDA, WKV6/7, conv (short conv), par (parallel residual), lscale/rscale/escale (logit/residual/embedding scale), PLE, MTP, HC, IDX.

### Llama-like

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `llama` | Llama 1-3.x, Mistral 7B, Mixtral (experts under this arch), SmolLM2, Yi, Falcon3 | GQA, RMS, SwiGLU, rope; Mixtral: MoE softmax top-2 | **supported** | - | llama, mistral, mixtral | hf/tiny-random-MixtralForCausalLM |
| `llama4` | Llama 4 Scout/Maverick | GQA, RMS, chunked-local attn + NoPE layers, qk-l2norm, MoE sigmoid + shared, attn temperature | **supported** | - (vision tower not imported) | llama4_text | hf/tiny-random-llama-4 (incl. a 160-token prompt across the 128-token chunk) |
| `deci` | Llama-3.x-Nemotron-51B (NAS) | per-layer GQA/skip-attn/ffn-mult, RMS | **supported** (per-layer heads / kv heads / ffn widths, linear-replacement and attention-free layers; a layer without MLP keeps its attention residual as in HF, llama.cpp drops it) | - | deci | gguf/tiny-random-deci-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `mistral3` | Ministral-3, Mistral-Small-3.x (+pixtral mmproj) | GQA, RMS, SwiGLU, yarn, llama4-style attn temperature | **supported** | `MMPROJ` (Pixtral) | mistral3 / ministral3 | gguf/Ministral-3-3B-Instruct-2512 (+mmproj), hf/tiny-random-mistral-3, hf/tiny-random-ministral3 |
| `llama-embed` | Llama embedding models (pooled) | llama graph + pooling head | **supported** (causal unless the file declares `attention.causal` = false, as in llama.cpp) | - | - | gguf/tiny-random-llama-embed-f32 (bidirectional, vs llama-server /embedding; tests/Sharplight.Families.Tests/EmbeddingFamilyTests.cs) |

### Qwen

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `qwen` | Qwen (v1) | MHA, RMS, SwiGLU, fused qkv | **supported (GGUF, tiny random vs llama.cpp)**| -| qwen | gguf/tiny-random-qwen1-f32 |
| `qwen2` | Qwen2/2.5, DeepSeek-R1-Distill-Qwen | GQA, qkv bias, RMS, SwiGLU | **supported** | - | qwen2 | - |
| `qwen2moe` | Qwen1.5-MoE, Qwen2-57B-A14B | MoE softmax + sigmoid-gated shared expert | **supported** | - | qwen2_moe | hf/tiny-random-qwen2-moe, gguf/tiny-random-qwen2-moe-f32.gguf |
| `qwen2vl` | Qwen2-VL / Qwen2.5-VL (LLM part) | GQA, M-RoPE, qkv bias | **supported** (language model only, text input: sectioned non-interleaved M-RoPE in the decoder; the Qwen2/2.5-VL towers are not imported; tiny models vs transformers and their converted GGUFs vs llama.cpp b11177 greedy ids, CPU and Vulkan; no real checkpoint run) | `MMPROJ` (qwen2vl_merger / qwen25vl towers) | qwen2_vl, qwen2_5_vl | hf/tiny-random-qwen2-vl-wide, hf/tiny-random-qwen2-5-vl-wide, gguf/tiny-random-qwen2-vl-wide-f32, gguf/tiny-random-qwen2-5-vl-wide-f32 |
| `qwen3` | Qwen3 dense, Qwen3-Embedding/Reranker | GQA, QK-norm, RMS, SwiGLU | **supported** | - | qwen3 | - |
| `qwen3moe` | Qwen3-30B-A3B, 235B-A22B | QK-norm, MoE softmax norm-topk (no shared) | **supported** | - | qwen3_moe | - |
| `qwen3vl` | Qwen3-VL dense | qwen3 + M-RoPE (interleaved) + DeepStack injection | **supported** (text and images: DeepStack tower vs llama.cpp b11177 (`llama-mtmd-debug`, worst corner 2.7e-3), GGUF + mmproj describes the test image on CPU and Vulkan, HF Qwen3-VL-2B-Instruct with an image vs transformers: prompt logits within 0.007 and 32 greedy ids, CPU and Vulkan. Q4_K_M continuation differs from llama-mtmd-cli at step 24 where the top-2 margin is 0.28) | - | qwen3_vl | gguf/Qwen3-VL-2B-Instruct (+mmproj), hf/Qwen3-VL-2B-Instruct |
| `qwen3vlmoe` | Qwen3-VL-MoE | qwen3moe + M-RoPE + DeepStack | **supported** (HF: fused experts in either order, router, tower with DeepStack through the shared Qwen3-VL path; tiny model vs transformers greedy ids on CPU and Vulkan; GGUF through the qwen3vl mapping with `expert_count`; no real checkpoint under 4 GB) | - | qwen3_vl_moe | hf/tiny-random-qwen3-vl-moe-wide |
| `sensenova_u1` | SenseNova-U1 | qwen3 + temporal-only M-RoPE (image gen) | **missing** | `MROPE`, `IMGGEN` | - | - |

### Qwen hybrid

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `qwen3next` | Qwen3-Next-80B-A3B, Qwen3-Coder-Next | GDN (3/4 layers) + gated attention (1/4, partial rope 0.25, QK-norm), MoE softmax norm-topk + sigmoid-gated shared expert, conv k=4, MTP | **supported** (MTP not used) | - | qwen3_next | gguf/Qwen3-Coder-Next, hf/Qwen3-Coder-Next (config+tokenizer), hf/tiny-random-qwen3-next-moe |
| `qwen35` | Qwen3.5/3.6/3.8 dense (0.8B...27B) | GDN + gated attention, M-RoPE partial, dense SwiGLU, MTP, vision tower | **supported** | - | qwen3_5 (text) | gguf/Qwen3.5-0.8B (existing), hf/Qwen3.5-0.8B (existing) |
| `qwen35moe` | Qwen3.5-35B-A3B/122B/397B, Qwen3.6-35B-A3B | qwen35 + MoE softmax (256 experts, top-8) + sigmoid-gated shared expert | **supported** | - | qwen3_5_moe | gguf/Qwen3.6-35B-A3B (+mmproj), hf/Qwen3.6-35B-A3B (config+tokenizer), hf/tiny-random-qwen3.5-moe, hf/tiny-random-qwen3.6-moe, hf/tiny-random-qwen3.8-moe |
| `qwen4exp` | Qwen3.8-Flash-Next (hf: qwen4_exp, ~200B total params) | GDN + gated attention, MoE 512 experts top-10 + shared, hyper-connections (4 streams, low-rank), QSA sparse-attention indexer (top-k 2048, pooled keys), PLE n-gram hash embeddings (layer 2), M-RoPE, MTP | **partial** (HC, NGRAM done; IDX dense regime only; MTP not used) | `IDX` (block selection) | qwen4_exp | gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL (+mmproj), hf/Qwen3.8-Flash-Next (config+tokenizer), hf/tiny-random-qwen3.8-flash-next |
| `clover_moe` | clover-moe (qwen3-next-like) | GDN + attention + MoE shared | **supported** (GGUF through the qwen3_5 adapter: the tiny Qwen3.5-MoE checkpoint retargeted to clover_moe vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan, b11177 predates it; per-layer `attention.recurrent_layers` must be periodic; not run on a real model) | - | - | gguf/tiny-random-clover-moe-f32 |

### Gemma

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `gemma` | Gemma 1 | MQA/MHA, RMS(1+w), GeGLU, escale | **supported**| -| gemma | - |
| `gemma2` | Gemma 2 | GQA, SWA alt, attn+final softcap, sandwich norms, GeGLU | **supported** | - | gemma2 | - |
| `gemma3` | Gemma 3 (+ SigLIP mmproj) | GQA, QK-norm, SWA 5:1 local rope base, sandwich norms | **supported** | - | gemma3_text | hf/tiny-random-gemma-3 |
| `gemma3n` | Gemma 3n E2B/E4B | AltUp, LAuReL, PLE, KV sharing, sparse FFN activation | **missing** | `ALTUP`, `LAUREL`, `PLE` | gemma3n | gguf/gemma-3n-E2B-it, hf/tiny-random-gemma-3n |
| `gemma4` | Gemma 4 (E2B/E4B/12B/MoE) | PLE, KV sharing, per-layer head dims, NoPE globals, k=v, MoE sigmoid | **supported** | - | gemma4 | - |
| `gemma4-assistant` | Gemma 4 MTP assistants | drafter (centroid head, KV from target) | **missing** | `SPECDEC` | - | - |
| `gemma-embedding` | EmbeddingGemma | bidirectional gemma3 + pooling | **supported** (bidirectional decoder layers, symmetric window, Dense projections after pooling; embeddinggemma-300m is gated and was not run) | - | gemma3_text (`use_bidirectional_attention`, sentence-transformers Dense modules) | hf/tiny-random-embeddinggemma (vs transformers), gguf/tiny-random-gemma-embedding-f32 (vs llama.cpp) |
| `gemma-embedding2` | EmbeddingGemma 2 | bidirectional gemma + PLE + pooling | **missing** | `ENC`, `POOL` | - | - |

### Phi

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `phi2` | Phi-1/1.5/2 | LN, parallel residual, partial rope, biased GELU MLP | **supported**| -| phi | gguf/phi-2, hf/tiny-random-PhiForCausalLM |
| `phi3` | Phi-3/3.5/4-mini (longrope), Phi-4 | fused qkv/gate-up, RMS, SwiGLU, longrope, SWA opt | **supported** | - | phi3 | gguf/Phi-4-mini-instruct, hf/tiny-random-phi3 |
| `phimoe` | Phi-3.5-MoE | LN, MoE with sparsemixer routing, longrope | **supported** (HF and GGUF: layer norms + sparsemixer routing vs transformers on CPU and Vulkan. Deviation: llama.cpp runs phimoe through its phi3 graph with RMS norms + bias and softmax top-2 routing, so its greedy ids differ from transformers and Sharplight; the GGUF mapping follows the model. llama.cpp's converter only takes the checkpoint renamed to `PhiMoEForCausalLM` with `original_max_position_embeddings`, see `tools/convert_phimoe_gguf.py`) | - | phimoe | hf/tiny-random-phimoe, gguf/tiny-random-phimoe-f32 (vs transformers) |

### Meta

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `muse-glimmer` | Muse Glimmer 30B | sandwich norms, sigmoid attn gate, SWA + NoPE, softcap | **supported** | - | muse_glimmer | - |
| `muse-spark` | Muse Spark | gated attn, norm-preserving residual, MoE in latent space | **supported** (GGUF, tiny random vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan, b11177 predates it; the 816B checkpoint was not run) | - | - | gguf/tiny-random-muse-spark-f32 |

### NVIDIA

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `nemotron` | Nemotron-4 / Minitron / Nemotron-Mini | LN(1+w), ReLU2 ungated MLP, partial rope | **supported**| -| nemotron | gguf/Nemotron-Mini-4B-Instruct |
| `nemotron_h` | Nemotron-H, Nemotron-3-Nano | Mamba2 + attention + MLP hybrid, ReLU2 | **supported** | - | nemotron_h | - |
| `nemotron_h_moe` | Nemotron-3-Nano-30B-A3B | Mamba2 + attn + MoE sigmoid bias shared latent | **supported** | - | - | - |

### Cohere

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `command-r` | Command-R, Command-R+, Aya-23, Aya-Expanse | LN(no bias), parallel residual, logit scale, QK-norm (R+) | **supported**| -| cohere | gguf/aya-expanse-8b, hf/tiny-CohereForCausalLM |
| `cohere2` | Command-R7B, Command-A | LN, parallel residual, SWA 3:1 + NoPE, logit scale | **supported**| -| cohere2 | gguf/c4ai-command-r7b-12-2024, hf/tiny-Cohere2ForCausalLM |

### OLMo

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `olmo` | OLMo 1/1-0724 | non-parametric LN, SwiGLU, clip_qkv | **supported**| -| olmo | hf/tiny-random-OlmoForCausalLM |
| `olmo2` | OLMo-2, OLMo-3 (SWA) | post-norm only, full-width QK-norm, SWA (OLMo3) | **supported**| -| olmo2, olmo3 | gguf/OLMo-2-0425-1B-Instruct, hf/OLMo-2-0425-1B-Instruct, hf/tiny-random-Olmo2ForCausalLM |
| `olmoe` | OLMoE-1B-7B | QK-norm full-width, MoE softmax top-8 | **supported** | - | olmoe | - |

### Granite

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `granite` | Granite 3.x dense, 4.0 dense | GQA, mup scalings (embed/residual/attn/logit) | **supported**| -| granite | gguf/granite-3.3-2b-instruct, hf/tiny-random-granite |
| `granitemoe` | Granite 3.x MoE (1B-A400M, 3B-A800M) | granite + fused-expert MoE softmax | **supported**| -| granitemoe | gguf/granite-3.1-1b-a400m-instruct, hf/granite-3.1-1b-a400m-instruct, hf/tiny-random-granite-moe |
| `graniteswitch` | granite-switch | granite + MTP/switch heads | **partial** | `RES`, `MAP` | - | - |
| `granite_swa` | granite SWA | granite + SWA + sinks | **supported** | - | granite_swa, granitemoe_swa, granitemoeshared | hf/tiny-random-granite-swa, hf/tiny-random-granitemoe-swa, gguf/tiny-random-granitemoe-swa-f32.gguf |

### GLM

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `chatglm` | ChatGLM3, GLM-4-9B-chat | MQA-ish GQA, partial interleaved rope, fused qkv | **supported (GGUF, tiny random vs llama.cpp)**| -| chatglm | gguf/tiny-random-chatglm-f32 |
| `glm4` | GLM-4-0414 (9B/32B), GLM-Z1 | GQA, sandwich norms, partial rope, fused gate-up | **supported**| -| glm4 | gguf/GLM-4-9B-0414, hf/tiny-random-glm-4 |
| `glm4moe` | GLM-4.5/4.6/4.7, GLM-4.5-Air | QK-norm, partial rope, MoE sigmoid + bias + groups + shared, dense lead, MTP | **supported** | - (MTP layers dropped) | glm4_moe | hf/tiny-random-glm-4-moe |
| `glm-dsa` | GLM-5 / DSA models | MLA + DSA indexer, MoE | **missing** | `MLA`, `IDX`, `RGRP` | glm_moe_dsa | - |
| `glm5-next` | GLM-5-Next | MLA + indexer + HC + linear-attn layers | **missing** | `MLA`, `IDX`, `HC`, `KDA` | - | - |

### MLA/DeepSeek-like

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `mistral4` | Mistral Large/Small 4 (MLA) | MLA, MoE, yarn (inherits deepseek2 graph) | **supported** | - | mistral4 | hf/tiny-random-mistral4, gguf/tiny-random-mistral4-f32.gguf, gguf/tiny-random-mistral4-temperature-f32.gguf |
| `mistral-large4` | Mistral Large 4 | MLA + indexer, MoE shared, yarn, attn temp | **missing** | `MLA`, `IDX`, `RGRP` | - | - |
| `minicpm3` | MiniCPM3-4B | MLA, RMS | **supported** (latent_decoder) | - | minicpm3 | gguf/MiniCPM3-4B, hf/tiny-random-minicpm3 |
| `deepseek` | DeepSeek-LLM-MoE (v1) | MHA, MoE softmax + shared, dense lead | **supported** | - | deepseek | gguf/tiny-random-deepseek-f32.gguf |
| `deepseek2` | DeepSeek V2/V2-Lite/V3/R1, Kimi-K2, GLM-4.7-Flash, Moonlight | MLA (absorbed cache), MoE sigmoid/softmax + bias + groups + shared, yarn mscale, MTP | **supported** (latent_decoder; MTP layers loaded but unused) | - | deepseek_v2, deepseek_v3, kimi_k2 | gguf/DeepSeek-V2-Lite-Chat, hf/tiny-random-deepseek-v3.1, hf/tiny-random-glm-4.7-flash |
| `deepseek2-ocr` | DeepSeek-OCR | deepseek2 text decoder + vision | **missing** | `MLA`, `MMPROJ` | - | - |
| `deepseek32` | DeepSeek-V3.2 | MLA + DSA lightning indexer | **missing** | `MLA`, `IDX`, `RGRP` | deepseek_v32 | hf/tiny-random-deepseek_v32 |
| `deepseek4` | DeepSeek-V4 | hyper-connections (Sinkhorn), compressed/sparse attention, sinks, MoE hash/groups | **missing** | `HC`, `IDX`, `RGRP` | - | - |
| `deepseek41` | DeepSeek-V4.1 | as v4 + PLE n-gram | **missing** | `HC`, `IDX`, `NGRAM` | - | - |
| `xing4_0` | xing4.0 | MLA + manifold HC + MoE | **missing** | `MLA`, `HC`, `RGRP` | - | - |
| `plm` | PLM-1.8B | MLA + ReLU2 | **supported** | - | - | gguf/tiny-random-plm-f32.gguf |

### MoE

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `dbrx` | DBRX | LN, clip_qkv, MoE softmax top-4 fused experts | **supported** | - | dbrx | hf/tiny-random-dbrx, gguf/tiny-random-dbrx-f32.gguf |
| `arctic` | Snowflake Arctic | dense FFN + parallel MoE residual | **supported** (dense MLP next to a separately normed expert branch, `moe_norm`) | - | - | gguf/tiny-random-arctic-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `grok` | Grok-1 | MoE softmax top-2, attn softcap, escale/lscale, sandwich norms | **supported** (router logit soft cap applied as in the reference implementation; llama.cpp reads it but does not apply it) | - | grok-1 | gguf/tiny-random-grok-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `exaone-moe` | EXAONE-MoE (K-EXAONE) | exaone4 + MoE sigmoid bias + shared, MTP | **supported** | - (GGUF mapping not verified on a real file) | exaone_moe | hf/tiny-random-exaone-moe |
| `bailingmoe` | Ling-lite, Ling-plus | MoE softmax + shared, dense lead | **supported** | - | - | gguf/tiny-random-bailingmoe-f32.gguf |
| `bailingmoe2` | Ling-mini-2.0, Ling-flash-2.0, Ring | QK-norm, MoE sigmoid + bias + groups + shared, MTP | **supported** | - | bailing_moe_v2 | hf/tiny-random-bailing-moe-v2-llamacpp (+ its f32 GGUF, vs llama.cpp) |
| `bailingmoe3` | Ling 3 / Ring-linear | MLA + KDA/GDN hybrid + MoE | **missing** | `MLA`, `KDA`, `GDN`, `RGRP` | - | - |
| `dots1` | dots.llm1 | QK-norm, MoE sigmoid + bias + shared | **supported** | - (GGUF mapping not verified on a real file) | dots1 | hf/tiny-random-dots1 |
| `dots3note` | dots.3 | MLA + indexer + SWA | **missing** | `MLA`, `IDX` | - | - |
| `afmoe` | AFM-MoE (Trinity) | SWA 3:1 (rope on the sliding layers only), sigmoid attention gate, sandwich norms, mup embedding scale, MoE sigmoid + bias + shared | **supported** (not run on a real model) | - | afmoe | hf/tiny-random-afmoe, gguf/tiny-random-afmoe-f32 |
| `laguna` | Laguna (poolside) | MoE sigmoid + bias + shared, softplus out gate (per head or per channel), per-layer head counts, own rope base/width on the SWA layers, yarn on the global layers | **supported** (not run on a real model; router logit softcapping not supported) | - | laguna | hf/tiny-random-laguna, hf/tiny-random-laguna-full, gguf/tiny-random-laguna-f32, gguf/tiny-random-laguna-full-f32 |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B, 300B-A47B | MoE softmax + bias + shared, dense lead | **supported** | - | ernie4_5_moe | gguf/ERNIE-4.5-21B-A3B-PT, hf/tiny-random-ernie-4.5-moe |
| `hunyuan-moe` | Hunyuan-A13B | QK-norm, MoE + shared, dynamic NTK | **supported** | - (GGUF mapping not verified on a real file) | hunyuan_v1_moe | hf/tiny-random-hunyuan-v1-moe |
| `hy_v3` | Hunyuan v3 | MoE sigmoid + bias + shared, MTP | **supported** (MTP layers ignored; not run on a real model) | - | hy_v3 | hf/tiny-random-hy-v3, gguf/tiny-random-hy-v3-f32 |
| `hy_v4` | Hunyuan v4 | MLA-like + indexer + HC | **supported** (HF, latent_decoder: MLA with sigmoid output gate and attention sinks, clamped SwiGLU experts, hyper-connections; tiny model vs transformers on CPU and Vulkan, plus a variant with DeepSeek-V4 mHC modules (Sinkhorn stream mixing) monkeypatched into the reference. Deviation: the DSA indexer runs only in its dense regime, the context is capped at `index_topk`. GGUF mapping not done (llama.cpp b11177 predates hy-v4)) | `IDX` (sparse regime) | hy_v4 | hf/tiny-random-hy-v4, hf/tiny-random-hy-v4-mhc |
| `gpt-oss` | gpt-oss-20b/120b | MoE top-4 softmax + expert biases, clamped SwiGLU-OAI, sinks, alternating SWA, yarn, MXFP4 weights | **supported** | - | gpt_oss | gguf/gpt-oss-20b (MXFP4), hf/tiny-random-gpt-oss, hf/tiny-random-gpt-oss-mxfp4 |
| `smallthinker` | SmallThinker 4B-A0.6B/21B-A3B | MoE (sigmoid gating), SWA+NoPE | **supported** | - (HF remote code not mapped) | smallthinker | gguf/SmallThinker-4BA0.6B-Instruct |
| `grovemoe` | GroveMoE | adjugate/chunk experts, QK-norm | **missing** | `GROVE` | - | - |
| `minimax-m2` | MiniMax-M2/M2.5 | full-width QK-norm, partial rope, MoE sigmoid + bias | **supported** | - (GGUF mapping not verified on a real file) | minimax_m2 | hf/tiny-random-minimax-m2 |
| `mimo2` | MiMo-V2-Flash | SWA + sinks, value head size < key head size, value scale, MoE sigmoid bias, MTP | **supported** (MTP layers ignored; not run on a real model) | - | mimo_v2_flash | hf/tiny-random-mimo-v2-flash, gguf/tiny-random-mimo-v2-flash-f32 |
| `step35` | Step-3.5-Flash | SWA, gated attn, MoE sigmoid + shared | **supported** (clamped SwiGLU, per-layer heads, per-head gate; MTP draft layers not used; not run on a real model, 196B) | - | - | gguf/tiny-random-step35-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `maion-coder` | Maion-Coder | SWA + NoPE, MoE latent | **supported** (GGUF, tiny random vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan, b11177 predates it; not run on a real model) | - | - | gguf/tiny-random-maion-coder-f32 |
| `mellum` | Mellum (JetBrains) | SWA + MoE, MTP | **supported** | - | mellum | hf/tiny-random-mellum, hf/tiny-random-mellum-llamacpp, gguf/tiny-random-mellum-llamacpp-f32.gguf |
| `k2-horizon` | K2-Horizon | MoVA value experts, MoE sigmoid + shared, yarn | **missing** | `MOVA`, `RGRP` | - | - |
| `beam` | Beam | SWA + NoPE, attn temp (power form), biased per-head gate, norm-preserving residual, softcap, MoE sigmoid | **supported** (GGUF, tiny random vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan, b11177 predates it; not run on a real model) | - | - | gguf/tiny-random-beam-f32 |
| `kolibri1` | Kolibri1 | SWA + NoPE, MoE sigmoid + shared | **supported** | - | - | gguf/tiny-random-kolibri1-f32 (tiny GGUF vs a newer llama.cpp build (b11177 lacks it) greedy ids), hf/tiny-random-kolibri1 (vs transformers) |
| `flint` | Flint | SWA/NoPE, sigmoid MoE + shared, sinks | **supported** | - | flint | - |
| `cohere2moe` | Command-A MoE | SWA + NoPE, MoE sigmoid + shared, parallel residual, LN | **supported** | - | cohere2_moe | hf/tiny-random-cohere2-moe, gguf/tiny-random-cohere2-moe-f32.gguf |

### Other dense

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `plamo` | PLaMo 1 | parallel residual style, RMS | **supported** | - | - | gguf/tiny-random-plamo-f32.gguf |
| `plamo3` | PLaMo 3 | SWA, QK-norm, sandwich | **supported** | - | - | gguf/tiny-random-plamo3-f32.gguf |
| `internlm2` | InternLM2/2.5, InternLM3 | GQA, fused wqkv, RMS | **supported**| -| internlm2 | gguf/internlm2_5-1_8b-chat, hf/tiny-random-internlm2 |
| `minicpm` | MiniCPM 2B, MiniCPM4/4.1 | RMS, muP scalings (embed/residual/logit), rope longrope | **supported**| -| minicpm, minicpm4 | gguf/MiniCPM4.1-8B, hf/tiny-random-minicpm4 |
| `openelm` | OpenELM | per-layer heads/ffn mult, QK-norm | **supported**| -| openelm | gguf/OpenELM-270M-Instruct |
| `bitnet` | BitNet b1.58 2B/3B | ternary weights, sub-norms (attn/ffn), ReLU2 | **supported** | - (I2_S) | bitnet | gguf/bitnet-b1.58-2B-4T, hf/tiny-random-bitnet |
| `exaone` | EXAONE 3.0/3.5 | GQA, RMS, SwiGLU | **supported**| -| exaone | gguf/EXAONE-3.5-2.4B-Instruct, hf/tiny-random-exaone |
| `exaone4` | EXAONE 4.0 | QK-norm, SWA 3:1 + NoPE globals, post-norms | **supported**| -| exaone4 | gguf/EXAONE-4.0-1.2B, hf/EXAONE-4.0-1.2B |
| `chameleon` | Chameleon | QK-LN, swin norm, VQ image tokens | **missing** | `VQ`, `LN` | - | - |
| `arcee` | Arcee AFM-4.5B | ReLU2 ungated MLP, yarn | **supported**| -| arcee | gguf/AFM-4.5B |
| `ernie4_5` | ERNIE-4.5 0.3B-VL, dense | GQA, interleaved rope, RMS | **supported**| -| ernie4_5 | gguf/ERNIE-4.5-0.3B-PT, hf/ERNIE-4.5-0.3B-PT, hf/tiny-random-ernie-4.5 |
| `hunyuan-dense` | Hunyuan 0.5B-7B | QK-norm, GQA, NTK-alpha rope | **supported**| -| hunyuan_v1_dense | gguf/Hunyuan-0.5B-Instruct, hf/Hunyuan-0.5B-Instruct, hf/tiny-random-hunyuan-dense-v1 |
| `hunyuan_vl` | HunyuanOCR / VL | xdrope M-RoPE + mmproj | **partial** | `MROPE`, `MAP`, `MMPROJ` | - | - |
| `smollm3` | SmolLM3-3B | NoPE every 4th layer, GQA | **supported**| -| smollm3 | gguf/SmolLM3-3B, hf/SmolLM3-3B, hf/tiny-random-smollm3 |
| `seed_oss` | Seed-OSS-36B | GQA, qkv bias, RMS | **supported**| -| seed_oss | hf/tiny-random-seed-oss |
| `apertus` | Apertus 8B/70B | xIELU activation, QK-norm | **supported**| -| apertus | gguf/Apertus-8B-Instruct-2509, hf/tiny-random-apertus |
| `cogvlm` | CogVLM | visual-expert qkv/ffn | **missing** | `VEXPERT`, `MMPROJ` | - | - |
| `pangu-embedded` | openPangu-Embedded | GQA, RMS | **supported** | - | - | gguf/tiny-random-pangu-embedded-f32.gguf |
| `paddleocr` | PaddleOCR-VL | ernie-like + M-RoPE | **partial** | `MROPE`, `MMPROJ` | - | - |
| `spark2_5` | Spark 2.5 | SWA pattern, gated attn, GeGLU | **supported** | - | - | gguf/tiny-random-spark2_5-f32.gguf |
| `maincoder` | Maincoder | QK-norm llama-like | **supported** | - | - | gguf/tiny-random-maincoder-f32.gguf |
| `talkie` | Talkie | QK-norm after rope, logit scale | **supported** (weight-less norms, per-head q gain, embedding skip) | - | - | gguf/tiny-random-talkie-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `nanbeige` | Nanbeige4/4.1/4.2-3B | looped layers (shared weights, per-loop KV) | **supported** (layer loops with per-loop KV caches) | - | - | gguf/Nanbeige4.2-3B Q4_K_M (vs llama.cpp b11177 text), gguf/tiny-random-nanbeige-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `ricursive` | Ricursive | SWA/NoPE, value residual, gated attn, swiglu clamp | **supported** (value residual, clamped SwiGLU, NoPE global layers) | - | - | gguf/tiny-random-ricursive-f32 (tiny GGUF vs a newer llama.cpp build (b11177 lacks it) greedy ids) |
| `maple` | Maple | SWA pattern, QK-norm, MoE | **supported** (matches the newer llama.cpp build, which clamps the gate before silu; b11177's ids differ) | - | - | gguf/tiny-random-maple-f32 (tiny GGUF vs a newer llama.cpp build (b11177 lacks it) greedy ids) |

### GPT-style

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `starcoder` | StarCoder 1, SantaCoder | MQA, LN, learned abs pos, biased GELU | **supported** | - | gpt_bigcode | hf/tiny-random-gpt-bigcode, gguf/tiny-random-gpt-bigcode-f32.gguf |
| `starcoder2` | StarCoder2 3B/7B/15B | GQA, SWA, LN, biased GELU MLP, rope | **supported**| -| starcoder2 | gguf/starcoder2-3b, hf/tiny-random-Starcoder2ForCausalLM |
| `gpt2` | GPT-2 | MHA, LN, learned abs pos, biased GELU, fused qkv | **supported**| -| gpt2 | gguf/gpt2 |
| `gptj` | GPT-J | parallel residual, interleaved partial rope, LN | **supported**| -| gptj | hf/tiny-random-GPTJForCausalLM |
| `gptneox` | Pythia, StableLM-alpha, RedPajama | parallel residual (opt), LN, partial rope, fused qkv | **supported**| -| gpt_neox | gguf/pythia-160m, hf/tiny-random-GPTNeoXForCausalLM |
| `falcon` | Falcon 7B/40B/180B | MQA/GQA, LN, parallel residual, GELU, rope | **supported**| -| falcon | hf/tiny-random-FalconForCausalLM |
| `mpt` | MPT-7B/30B | ALiBi, LN, QK-clip, GELU | **supported** (q/k layer norms (`qk_ln`) not supported; not run on a real model) | - | mpt | hf/tiny-random-mpt, gguf/tiny-random-mpt-f32 |
| `bloom` | BLOOM | ALiBi, LN, GELU | **supported** (`bloom` pre-tokenizer checked against llama.cpp) | - | bloom | gguf/bloom-560m (Q4_K_M vs llama.cpp Vulkan), hf/tiny-random-bloom, gguf/tiny-random-bloom-f32 |
| `refact` | Refact | ALiBi, RMS | **supported** | - | - | gguf/tiny-random-refact-f32.gguf |
| `baichuan` | Baichuan 1/2 (13B: ALiBi) | RMS, SwiGLU, rope (7B) or ALiBi (13B) | **supported** | - | baichuan | gguf/tiny-random-baichuan-f32.gguf, gguf/tiny-random-baichuan-alibi-f32.gguf |
| `stablelm` | StableLM 2 / 3B | LN, QK-norm (per head), partial rope, SwiGLU | **supported**| -| stablelm | gguf/stablelm-2-1_6b, hf/tiny-random-StableLmForCausalLM |
| `codeshell` | CodeShell | LN, rope, GELU | **supported** | - | - | gguf/tiny-random-codeshell-f32.gguf |
| `orion` | Orion-14B | LN, SwiGLU, rope | **supported** | - | - | gguf/tiny-random-orion-f32.gguf |
| `xverse` | XVERSE | RMS, SwiGLU, rope | **supported (GGUF, tiny random vs llama.cpp)**| -| - | gguf/tiny-random-xverse-f32 |
| `jais` | Jais | ALiBi, LN, SwiGLU | **supported** | - | jais | gguf/tiny-random-jais-f32.gguf |
| `jais2` | Jais 2 | LN, ReLU2, rope | **supported** | - | jais2 | hf/tiny-random-jais2, gguf/tiny-random-jais2-f32.gguf |

### SSM

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `mamba` | Mamba 1 (130M-2.8B), Falcon-Mamba, Codestral-Mamba(2) | Mamba1 selective scan (per-channel A, dt_rank), conv4, RMS | **supported** | - | mamba, falcon_mamba | gguf/mamba-130m-hf, hf/mamba-130m-hf, hf/tiny-random-MambaForCausalLM |
| `mamba2` | Mamba2, Codestral Mamba | Mamba2 SSD, groups, gated RMS | **supported** | - | mamba2 | gguf/mamba2-130m-hf, hf/mamba2-130m-hf |
| `jamba` | Jamba, Jamba-Reasoning-3B | Mamba1 + attention interleave + MoE | **supported** | - (MoE layers untested: the local Jamba-Reasoning-3B is dense) | jamba | gguf/AI21-Jamba-Reasoning-3B |
| `falcon-h1` | Falcon-H1 0.5B-34B | parallel attention + Mamba2 in each layer, mup multipliers | **supported** | - | falcon_h1 | gguf/Falcon-H1-0.5B-Instruct, hf/Falcon-H1-0.5B-Instruct, hf/tiny-random-falcon-h1 |
| `granitehybrid` | Granite 4.0 H micro/tiny/small | Mamba2 + attention layers + MoE shared, mup scalings | **supported** | - | granitemoehybrid | gguf/granite-4.0-h-350m, gguf/granite-4.0-h-tiny, hf/granite-4.0-h-350m, hf/tiny-random-granite-4.0-h, hf/tiny-random-granite-moe-hybrid |
| `plamo2` | PLaMo 2 | Mamba-2 style layers (own layout) + attention + sandwich norms | **supported** (GGUF, tiny random vs llama.cpp; not run on a real model) | - | - | gguf/tiny-random-plamo2-f32 |

### RNN

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `rwkv6` | RWKV v6 World | WKV6 time-mix, token-shift, LN, ChannelMix | **supported** | - | rwkv6 (remote code) | gguf/rwkv-6-world-1b6 |
| `rwkv6qwen2` | RWKV6-Qwen2 (QRWKV6) | gated linear attention (Wkv6 with zero bonus), GQA key/value heads, RMS + SwiGLU | **supported** (GGUF, tiny random vs llama.cpp; not run on a real model) | - | - | gguf/tiny-random-rwkv6qwen2-f32 |
| `rwkv7` | RWKV v7 G1 (0.1B-13B) | WKV7 (generalised delta rule), LoRA decay/a/v/g, LN | **supported** | - | rwkv7 (remote code) | gguf/rwkv7-g1h-1.5b |
| `arwkv7` | ARWKV7 (Qwen-distilled) | WKV7 + RMS + SwiGLU, optional output gate / group norm | **supported** (GGUF, tiny random vs llama.cpp; not run on a real model) | - | - | gguf/tiny-random-arwkv7-f32, gguf/tiny-random-arwkv7-nogate-f32 |

### Hybrid conv

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `lfm2` | LFM2/LFM2.5 350M-2.6B | gated short-conv (k=3) + GQA attn, QK-norm | **supported** | - | lfm2 | gguf/LFM2.5-1.2B-Instruct, hf/LFM2.5-1.2B-Instruct, hf/tiny-random-lfm2 |
| `lfm2moe` | LFM2-8B-A1B, LFM2.5-8B-A1B | lfm2 + MoE sigmoid bias | **supported** | - | lfm2_moe | gguf/LFM2.5-8B-A1B, hf/tiny-random-lfm2-moe |
| `lfm3` | LFM3-1.7B, LFM3-VL/Omni (text part) | gated conv + sliding/full attention (own rope bases, partial rope via `rope_freqs`), QK-norm, per-head sigmoid gate, sandwich norms | **supported** (GGUF: tiny random and real LFM3-1.7B Q8_0 vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan; b11177 predates lfm3. The non-causal "d1" decision variant (bidirectional full-attention layers, no cache) is not supported; no transformers class) | - | - | gguf/LFM3-1.7B, gguf/tiny-random-lfm3-f32 |

### Linear attn

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `minimax-01` | MiniMax-Text-01 / M1 | lightning linear attention + softmax every 8th, MoE | **missing** | `LIN`, `MOE` | - | - |
| `minimax-m3` | MiniMax-M3 | indexer + lightning + SwiGLU-OAI | **missing** | `IDX`, `LIN`, `ACT` | - | - |
| `kimi-linear` | Kimi-Linear-48B-A3B | KDA (per-channel gated delta) 3:1 + MLA, MoE sigmoid + shared, conv | **missing** | `KDA`, `MLA`, `RBIAS` | kimi_linear | hf/tiny-random-kimi-linear |
| `kimi-k3` | Kimi K3 | KDA + MLA + HC | **missing** | `KDA`, `MLA`, `HC` | - | hf/tiny-random-kimi-k3 |
| `kimi-k31` | Kimi K3.1 | KDA + MLA + HC + indexer | **missing** | `KDA`, `MLA`, `HC`, `IDX` | - | - |

### Encoder

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `bert` | BERT, MiniLM, bge, e5, rerankers | bidirectional, post-LN, abs pos + token types, GELU, pooling | **supported** | - | bert, roberta, xlm-roberta | all-MiniLM-L6-v2, bge-small-en-v1.5, bge-reranker-base / -v2-m3 (Sharplight.Embeddings.Tests) |
| `modern-bert` | ModernBERT, Granite Embedding R2 | bidirectional, local/global rope, GeGLU, LN | **supported** (canonical decoder with `attention.causal` = false; llama.cpp's converter rejects the masked-LM checkpoints, so the real model was checked through the HF importer only) | - | modernbert | hf/ModernBERT-base (vs transformers), hf/tiny-random-modernbert (vs transformers), gguf/tiny-random-modern-bert-f32 (vs llama.cpp) |
| `nomic-bert` | nomic-embed-text v1/1.5 | bidirectional, rope, SwiGLU | **supported** | - | nomic_bert | nomic-embed-text-v1.5 f16 / Q8_0 GGUF (Sharplight.Embeddings.Tests) |
| `nomic-bert-moe` | nomic-embed-text-v2-moe | bidirectional + MoE | **missing** | `ENC`, `POOL`, `LN` | - | - |
| `neo-bert` | NeoBERT | bidirectional, RMS, SwiGLU, rope | **supported** (GGUF; NeoBERT's HF code is remote-only) | - | - | gguf/tiny-random-neo-bert-f32 (vs llama.cpp) |
| `jina-bert-v2` | jina-embeddings-v2 | bidirectional, ALiBi | **supported** (GGUF; encoder adapter with ALiBi, q/k projection norms, second attention norm, GeGLU) | - | - | gguf/real-embed/jina-embeddings-v2-small-en-f32 (converted, vs llama.cpp), gguf/tiny-random-jina-bert-v2-f32 (vs llama.cpp) |
| `jina-bert-v3` | jina-embeddings-v3 | bidirectional, rope, LoRA tasks | **missing** | `ENC`, `POOL` | - | - |
| `eurobert` | EuroBERT | bidirectional llama-like | **supported** | - | eurobert | hf/EuroBERT-210m (vs transformers), hf/tiny-random-eurobert (vs transformers), gguf/tiny-random-eurobert-f32 (vs llama.cpp) |

### Encoder-decoder

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `t5` | T5, Flan-T5, UL2 | cross-attention, relative-position buckets, RMS(no bias), GeGLU/ReLU | **supported** (encoder_decoder adapter: encoder pass per request, cross-attention over the encoder output (CPU and Vulkan kernels), relative position bias; flan-t5-small Q4_K_M vs llama.cpp b11177, tiny T5 / T5 v1.1 and flan-t5-small f32 vs transformers, CPU and Vulkan. Deviations: llama.cpp never scales the tied LM head of the original T5 (transformers does; Sharplight follows transformers), and flan-t5-small Q4_K_M departs from the f32 HF greedy ids at step 4) | - | t5, mt5, umt5 | gguf/flan-t5-small, hf/flan-t5-small, hf/tiny-random-t5, hf/tiny-random-t5-v1_1 |
| `t5encoder` | T5 encoder (embeddings) | bidirectional + relative buckets | **partial** (GGUF mapping onto the encoder half of the encoder_decoder adapter with pooling; not yet checked against a reference) | `MAP` (verification) | - | - |

### Diffusion

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `dream` | Dream-7B | masked diffusion, bidirectional qwen2-like | **missing** | `DIFF` | Dream | - |
| `llada` | LLaDA-8B | masked diffusion, llama-like | **missing** | `DIFF` | llada | - |
| `llada-moe` | LLaDA-MoE | masked diffusion + MoE | **missing** | `DIFF` | - | - |
| `rnd1` | RND1 (Qwen3-MoE diffusion) | bidirectional qwen3moe | **missing** | `DIFF` | - | - |

### Spec-decode

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `eagle3` | EAGLE3 drafters | drafter reading target hidden states | **missing** | `SPECDEC` | - | - |
| `dflash` | DFlash drafters | diffusion-flash drafter | **missing** | `SPECDEC` | - | - |

### Multimodal

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `clip` | mmproj / vision towers | ViT encoders | **supported** | - | - | - |

### Exotic

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `clef` | clef (decision head, GDN hybrid) | GDN hybrid + decision-order scoring head | **missing** | `GDN`, `CUSTOMHEAD` | - | - |
| `vllm-decision2` | vllm-decision2 | LayerNorm transformer + decision head | **missing** | `CUSTOMHEAD` | - | - |
| `metis16` | metis16 | Mamba2 + HC + memory/route retrieval | **missing** | `HC`, `M2H`, `CUSTOM` | - | - |
| `hrm_text` | HRM-Text 1B | alternating low/high stacks, H/L cycles | **missing** | `LOOP` | - | - |
| `limite` | Limite | XSA, MUDD cross-layer taps, value embeddings | **missing** | `CUSTOM` | - | - |

### Out of scope

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `qwen3tts` | Qwen3-TTS | speech codec LM | **missing** | `TTS` | - | - |
| `pockettts` | PocketTTS | TTS flow decoder | **missing** | `TTS` | - | - |
| `wavtokenizer-dec` | WavTokenizer decoder | audio codec decoder | **missing** | `TTS` | - | - |

Notes on the matrix:

- `llama` is marked supported because plain Llama/Mistral/SmolLM work. Mixtral (arch `llama` + `llama.expert_count`) falls through the MoE keys in DecoderGgufMapping (`MAPMOE`) and the HF `mixtral` model_type is unmapped; both are quick fixes.
- Embedding/reranker variants of decoder archs (Qwen3-Embedding, Qwen3-Reranker, `llama-embed`) load as their base arch; the pooling head (`POOL`) is a runtime-level feature.
- HF-only `model_type`s without a llama.cpp architecture (e.g. RWKV v4/v5 `rwkv`, `bart`) are outside this survey.

## 5. Bring-up plan

Ordering weighs 2025-26 deployment popularity, number of architectures each wave unlocks, and risk. Every wave names its first test model from section 8; the llama.cpp output recorded there is the parity target, and each tiny-random HF checkpoint is the cheap structure/graph test (compare HF-transformers ids as Families.Tests already does for nemotron-h and flint).

**Wave 0 - importer-only quick wins (days, no new ops).** Register mappings in the Gguf and HuggingFace importers for architectures that DecoderAdapter already expresses: Mixtral (`MAPMOE`), smollm3, olmo2/olmo3, glm4 (dense), ernie4_5, hunyuan-dense, exaone, exaone4, seed_oss, internlm2, qwen (v1), xverse, gemma (1), openelm, pangu-embedded, maincoder, afm/arcee (after ACT), chatglm. Test: SmolLM3-3B, GLM-4-9B-0414, ERNIE-4.5-0.3B, Hunyuan-0.5B, EXAONE-3.5/4.0, OLMo-2-1B, InternLM2.5-1.8B and the matching tiny-random HF checkpoints. Also add the Mistral-7B-Instruct-v0.3 case that is missing from the tests.

**Wave 1 - Qwen3.x family (top priority).**

- 1a: `qwen35moe` and `qwen3next`. Factor the MoE sub-graph (softmax top-k, norm-topk, MatMulId experts, shared expert) out of DecoderAdapter, add `SGATE`, call it from Qwen35Adapter in place of the dense MLP; implement the qwen3next layouts (`QKVZ`: fused in_proj_qkvz / in_proj_ba, partial rope 0.25, no M-RoPE). Tests: tiny-random-qwen3.5-moe / qwen3.6-moe / qwen3.8-moe / qwen3-next-moe (HF), then Qwen3.6-35B-A3B UD-Q4_K_M (22 GB, fits the 96 GB GPU) and Qwen3-Coder-Next Q4_K_M (48.5 GB, the ~50 GB Next-class file). These two also exercise 256/512-expert MatMulId at scale and the Q4_K/Q5_K/Q6_K paths.
- 1b: `qwen3vl`/`qwen3vlmoe` text + DeepStack (`MROPE`, `DSTACK`, extend `qwen3vl_merger` mapping) with Qwen3-VL-2B-Instruct and its mmproj.
- 1c: `qwen4exp` (Qwen3.8-Flash-Next): `HC`, `IDX` (dense masked QSA fallback first), `NGRAM` on top of 1a. See section 6. The first target file is Unsloth `UD-Q4_K_XL` (111 GB, only kernel-covered tensor types, see 8.1); nothing at or below 50 GB exists for this model (smallest is 66 GB IQ1_S), and all of the smaller quants need IQ1/IQ2/IQ3 kernels from the quantisation track below.
- **Status (Wave 1):** one MoE block (`MoeBlock`/`MoeRouter`/`MoeOptions`: softmax or sigmoid scores, optional selection bias, normalisation, scale, separate or fused gate/up experts, ungated experts, gated/ungated/sigmoid-gated shared expert, expert activation through `IMlpActivation`) now serves Qwen35Adapter, DecoderAdapter (Mixtral, Granite-MoE, ...), HybridAdapter (Jamba, Granite 4, LFM2-MoE) and NemotronH. `qwen35moe` and `qwen3next` match transformers (tiny models) and llama.cpp (Qwen3.6-35B-A3B, Qwen3-Coder-Next) on CPU and Vulkan; `qwen3vl` text matches transformers and llama.cpp (Qwen3-VL-2B; image path with DeepStack and M-RoPE image positions not done, `qwen3_vl_moe` HF not mapped); `qwen4exp` HC + NGRAM match transformers on tiny-random-qwen3.8-flash-next (max logit diff 4e-6 CPU, 8e-6 Vulkan), with the indexer in its dense regime (context capped at the token budget, 2051) and MTP not used; the GGUF path matches transformers too (tiny model converted with llama.cpp's converter, fused tensors split back), and the 111 GB UD-Q4_K_XL matches llama.cpp with CPU offload (78 GB on the GPU, 26 GB of layers on the CPU, weights mmapped). Its 28.8 GB IQ4_NL n-gram table exposed an NVIDIA driver issue: a 64-bit `OpPtrAccessChain` index on a byte pointer is truncated to 32 bits, so `load8` now computes row starts on integers (other kernels that address single tensors beyond 4 GiB through pointer offsets would need the same). Bench (`--bench 512`, best of runs on a shared GPU; llama-bench b11177 Vulkan / CUDA): Qwen3.6-35B-A3B prefill 5.4k tok/s (7.5k / 4.2k), decode 216 tok/s (172 / 68); Qwen3-Coder-Next prefill 3.3k (5.0k / 4.4k), decode 154 (179 / 148); Qwen3-VL-2B decode 155 tok/s; Qwen3.8-Flash-Next with 33 of 48 layers on the GPU (llama-bench `-ngl 33`, p128/n32) prefill 186 tok/s (CUDA 107, Vulkan 71), decode 13.5 tok/s (CUDA 18.8, Vulkan 13.9).
- Parallel **quantisation track Q**: Q2_K, Q3_K, IQ2/IQ3/IQ1 dequant + matmul (CPU, Slang), then MXFP4/NVFP4 and TQ1_0/TQ2_0. Needed by Wave 1c and Wave 5. **Status: done** (CPU, Vulkan; MXFP4 and TQ2_0 real models still wait for their architectures). Verification: dequantisation bit-exact against llama.cpp's own decoders (`tools/gen-quant-reference.py`, `GgmlDequantTests`); matvec/GEMM/gather/MatMulId on Vulkan against the CPU oracle (`QuantFormatTests`); Qwen3-0.6B requantised to each format with llama-quantize (`tools/make-quant-test-models.py`, `QuantizedModelTests` compares next-token distributions with llama.cpp). The IQ lookup grids are device constants whose address is the first member of the kernel parameter block (`HAS_GRID`; a grid in kernel-local memory made the IQ decoders 3-5x slower). `sharplight --bench 512` against llama-bench b11177 (RTX PRO 6000, shared with other jobs, so noisy): on Qwen3-0.6B every new format prefills 23-55k tok/s and decodes 690-720 tok/s (launch bound), at or above llama.cpp CUDA/Vulkan except Q3_K_S decode on the busy GPU; on Mistral-7B requantised, Q2_K prefill 8.2k tok/s (llama.cpp CUDA 7.3k, Vulkan 7.0k) and decode 174 (285/195), Q3_K_S prefill 6.9k (9.4k/7.6k) and decode 138 (249/188), MXFP4 prefill 7.4k (16.4k/9.8k) and decode 156 (268/211); the existing Q4_K_M row of the same model is 6.3k/146 against 10.2k/212 (CUDA) and 6.5k/147 (Vulkan), so the new formats sit where Q4_K does relative to llama.cpp. CPU decode of the new formats matches Sharplight's Q4_K (about half of llama.cpp's int8 dot-product kernels).

**Wave 2 - mainstream dense families that need LayerNorm/parallel residual/scales.** `LN`, `PAR`, `RES`, `ACT` in DecoderAdapter, then: granite, granitemoe, minicpm/minicpm4, phi2, starcoder2, command-r, cohere2, falcon, gptneox/Pythia, gptj, gpt2/starcoder (`APOS`), olmo, nemotron (Nemotron-4/Mini), stablelm, apertus, arcee. Tests: granite-3.3-2b, granite-3.1-1b-a400m, phi-2, StarCoder2-3B, aya-expanse-8b, c4ai-command-r7b, Nemotron-Mini-4B, MiniCPM4.1-8B, Apertus-8B, AFM-4.5B, pythia-160m, gpt2, stablelm-2-1.6b.

**Wave 3 - routed MoE zoo.** `RBIAS` + `RGRP` lifted into the shared MoE helper: glm4moe (GLM-4.5/4.6/4.7), ernie4_5-moe (ERNIE-4.5-21B-A3B), bailingmoe2 (Ling), dots1, lfm2moe (with Wave 4 CONV), minimax-m2, hunyuan-moe, hy_v3, exaone-moe, mimo2, afmoe, laguna, smallthinker, llama4 (`TMP`) and mistral3 (`TMP`, Pixtral mmproj). Tests: ERNIE-4.5-21B-A3B-PT Q4_K_M, LFM2.5-8B-A1B, SmallThinker-4BA0.6B, Ministral-3-3B plus tiny-random glm-4-moe / bailing-moe-v2 / minimax-m2 / llama-4 / mistral-3.

**Wave 4 - MLA (DeepSeek/Kimi).** `MLA` (latent cache + absorbed decode + expanded prefill), then deepseek2 (DeepSeek-V2-Lite first, then V3/R1/Kimi-K2/GLM-4.7-Flash via the same arch), minicpm3, mistral4, plm; then `IDX` for deepseek32/glm-dsa and `KDA` for kimi-linear. Tests: DeepSeek-V2-Lite-Chat Q4_K_M (16B MoE with MLA), MiniCPM3-4B, tiny-random deepseek-v3.1 / glm-4.7-flash / deepseek_v32 / kimi-linear.

**Wave 5 - SSM, RNN and short-conv hybrids.** `M2H` factory: granitehybrid (Granite 4.0-H micro/tiny/small), falcon-h1, mamba2; `M1`: mamba, jamba, plamo2; `CONV`: lfm2/lfm3 (and lfm2moe from Wave 3); `WKV7` then `WKV6`: rwkv7 (G1 series), arwkv7, rwkv6, rwkv6qwen2. Tests: granite-4.0-h-350m (0.2 GB) and -tiny (4.2 GB), Falcon-H1-0.5B, mamba-130m / mamba2-130m, Jamba-Reasoning-3B, LFM2.5-1.2B, rwkv7-g1h-1.5B, rwkv-6-world-1.6B.

**Wave 6 - gpt-oss and format-driven models.** `MXFP4` kernels, `EBIAS`, SwiGLU-OAI (`ACT`): gpt-oss-20b/120b (alternating SWA, sinks and yarn already exist). BitNet (`TQ`, `SUBNORM`). Test: gpt-oss-20b MXFP4 (12 GB), bitnet-b1.58-2B-4T TQ2_0.

**Wave 7 - encoders, encoder-decoder, diffusion, exotic.** `ENC`+`POOL`+`LN`+`APOS`: bert, nomic-bert(-moe), modern-bert, neo-bert, jina, eurobert, gemma-embedding; `ALIBI`: bloom, mpt, jais, baichuan-13B, refact; `XATTN`+`RELPOS`: t5/t5encoder (flan-t5-small); `DIFF`: dream, llada(-moe), rnd1; `ALTUP`+`LAUREL`: gemma3n; `LOOP`: nanbeige, hrm_text; spec-decode drafters; hyper-connection DeepSeek-V4/Kimi-K3/GLM-5 family (`HC` already built in Wave 1c) when checkpoints reach a practical size. The remaining one-off architectures in the Exotic group are low priority.

Out of scope for the text runtime: qwen3tts, pockettts, wavtokenizer-dec (audio/TTS decoders), and the image/audio towers of omni models unless a product need appears.

## 6. Qwen3.8-Next (`qwen4exp`) deep dive

Naming on the Hub: `Qwen/Qwen3.8-Flash-Next` is the 'Next' hybrid of the 3.8 generation; its config says `architectures: Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp(_text)`; llama.cpp arch `qwen4exp`. The dense `Qwen3.8-27B` is plain `qwen35` and already supported. `Qwen3-Coder-Next` (80B-A3B) is `qwen3_next` / `qwen3next`; `Qwen3.5/3.6-35B-A3B` are `qwen3_5_moe` / `qwen35moe`.

Config highlights (HF config.json, text part): 48 layers, hidden 2560, `full_attention_interval` 4 (3 gated-delta layers : 1 full-attention layer), 24 attention heads / 2 KV heads / head_dim 256, partial rotary 0.25, interleaved M-RoPE sections [11,11,10], rope theta 1e7, 512 routed experts top-10 with expert width 640 plus a shared expert (640), linear attention 16 key heads x128 / 48 value heads x128, conv kernel 4, vocabulary 248320, one MTP layer, max positions 262144. Q8_0 is 188 GB (about 200B parameters total), BF16 about 400 GB over 131 safetensors shards: **no small HF checkpoint exists**, so the HF side is config + tokenizer + `model.safetensors.index.json` plus a tiny-random checkpoint.

What is new compared with `qwen35` / `qwen3next` (read from src/models/qwen4exp.cpp):

| Feature | hparams / tensors | Needed in Sharplight |
|---|---|---|
| MoE FFN on every layer | `ffn_gate_inp`, fused `ffn_gate_up_exps` (or separate), `ffn_down_exps`, shared expert with scalar sigmoid gate `ffn_gate_inp_shexp` | `MOEHYB`, `SGATE` (Wave 1a) |
| Hyper-connections, 4 streams, low-rank 320 | per layer two HC modules (`hc_attn_*`, `hc_ffn_*`: norm [hidden,hc], down+inject merged [4*hidden -> 320+4], up [320 -> 4*hidden]); final head `hc_head_{norm,down,up}` replaces `output_norm`; residual state is [hidden, hc, T] | `HC` |
| QSA sparse-attention indexer on full-attention layers | `index_q_proj` [hidden -> 4*128], `index_k_proj` [hidden -> 128], `index_{q,k}_norm`; top-k budget 2048, compress ratio 4 (pooled keys, raw+pooled rows, tail always kept) | `IDX` |
| PLE n-gram hash embedding on one linear layer | GGUF `ple.layers`=[1] (HF `ple_layer_ids`=[2]), n-gram size 3, 8 heads per n-gram order -> 16 heads, per-head offsets/vocab sizes (~20M rows each) and 3 hash multipliers, `per_layer_tok_embd` flat [160, ~320M] Q5_1 = 38 GB, `embedding_length_per_layer_input`=160, `ple_key/value/norm_*`, `ple_conv1d` (k=4 over the 4 streams), conv history in the recurrent cache | `NGRAM` |
| Gated-delta layers | same as qwen35 but 48 value heads, merged `wqkv_gate` and `ssm_beta_alpha` tensors | supported op; mapping |
| Full-attention layers | `wq` = [q | gate] interleaved per head, QK-norm, M-RoPE | present (Qwen35Adapter) |
| MTP block | `eh_proj`, `enorm`, `hnorm` per stream, own HC head | optional, defer |
| Quantised weights | Unsloth UD-Q4_K_XL 111 GB (supported tensor types), UD-IQ4_XS 94 GB / UD-Q3_K_XL 90 GB (contain IQ3_XXS and IQ2_S tensors), smallest UD-IQ1_S 66 GB; the 38 GB n-gram table is its own tensor (Q5_1) | quant track Q for everything below Q4_K_XL |

HF checkpoint layout (from `model.safetensors.index.json`, 1658 tensors): the trunk lives under `model.language_model.*`, the vision tower under `model.visual.*`, MTP under `mtp.*` (31 tensors). Per layer: `attn_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}` and the same four for `mlp_hyper_connection`; `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm,out_proj}` (separate projections, like Qwen3.5) or `self_attn.{q,k,v,o}_proj` + `{q,k}_norm` + `indexer.{index_qk_proj,q_layernorm,k_layernorm}` (fused indexer q/k projection); `mlp.gate`, **fused 3-D expert tensors** `mlp.experts.{gate_up_proj,down_proj}` (no per-expert tensors to stack), `mlp.shared_expert.{gate,up,down}_proj` and `mlp.shared_expert_gate`; one layer additionally `ple.{key_proj,norm_key,norm_query,norm_conv,conv1d}` and `ple.ple_embedding.{layer_multipliers,ngram_embedding.shard_0..127}` (the n-gram table is split into 128 shards that the importer must concatenate; llama.cpp stores one flat `per_layer_tok_embd`). Final `hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}` replaces `model.norm`. The GGUF side is the mirror image with merged `wqkv_gate` / `ssm_beta_alpha` / `hc_*_down_inject` tensors.

Recommended order: Wave 1a first (it is also what Qwen3.6-35B-A3B needs), then `HC` (self-contained graph feature, testable on tiny-random-qwen3.8-flash-next with the HF transformers reference), then `NGRAM`, then `IDX` with a dense masked fallback that reproduces top-k selection exactly (a model-quality-neutral approximation for context <= budget 2048, where the indexer selects every token anyway).

## 7. Multimodal projectors (mmproj)

llama.cpp `tools/mtmd/models/` implements these vision/audio front ends (file names = projector families). Sharplight supports exactly one projector type today (`clip.projector_type = qwen3vl_merger`, the Qwen3.5/3.6/3.8 tower) and rejects DeepStack towers.

| Family | llama.cpp projector files | Used by | Status |
|---|---|---|---|
| Qwen vision | qwen2vl, qwen3vl (DeepStack), qwen3.5 merger (`qwen3vl_merger`) | Qwen2-VL/2.5-VL, Qwen3-VL(-MoE), Qwen3.5/3.6/3.8 | qwen3_5 tower supported; qwen2vl/qwen3vl missing (`DSTACK`) |
| SigLIP / LLaVA-like | siglip, llava, internvl, minicpmv, mobilenetv5 (Gemma 3n), gemma4v/uv, glm4v, kimivl, kimik25, lfm3vl, step3vl, magevl, mimovl, youtuvl, ling3vl, exaone4_5, nemotron-v2-vl, granite4-vision | Gemma 3/4, SmolVLM, InternVL, MiniCPM-V, GLM-4.xV, Kimi-VL, LFM2-VL, Step3-VL | missing (ViT tower graph exists in VisionTowerGraph; per-family resamplers/mergers and preprocessing missing) |
| Pixtral | pixtral | Mistral-3 / Ministral-3 / Pixtral | missing |
| Llama 4 | llama4 | Llama 4 | missing |
| OCR / document | deepseekocr(2), paddleocr, dotsocr, dots3note, hunyuanvl, cogvlm | DeepSeek-OCR, PaddleOCR-VL, dots.ocr, HunyuanOCR, CogVLM | missing |
| Audio encoders | whisper-enc, conformer, voxtral-rt, qwen3a, granite-speech, parakeet, nemotron-transcribe, mimo-audio, lfm3a, gemma4a/ua, muse-spark-audio, nemotron-omni | Ultravox/Voxtral, Qwen3-Omni, Granite-Speech, Gemma 4 audio, ASR | missing (whisper/parakeet ONNX paths exist for ASR separately) |
| TTS / generation | chatterbox, pockettts, qwen3tts, soprano, drama3, sensenova-u1, yasa2 | speech/image generation | out of scope |
| Model-native | muse-glimmer, minimax-m3, deepseek4v | Meta Muse, MiniMax-M3, DeepSeek-V4 | missing |

mmproj assets downloaded: Qwen3.8-Flash-Next-UD-Q4_K_XL (F16), Qwen3.6-35B-A3B (F16), Qwen3-VL-2B-Instruct (F16), gemma-3-4b-it (f16), Ministral-3-3B-Instruct-2512 (BF16), LFM2-VL-450M (F16). Priority: Qwen3-VL (DeepStack) with Wave 1b, then SigLIP/Gemma-3 and Pixtral.

## 8. Test assets and llama.cpp reference outputs

### 8.1 GGUF models (`~/models/gguf/<name>/`)

Reference = llama.cpp b11177 CUDA build, `llama-completion`, greedy (`--temp 0 --top-k 1 --seed 1`), 16 tokens, prompt `The capital of France is`, `-c 512 -no-cnv --no-display-prompt`. Models up to ~25 GB fully offloaded (`-ngl 99`); Qwen3-Coder-Next (48 GB) used llama.cpp's default memory fitting; Qwen3.8-Flash-Next (111 GB) ran with `-ngl 0` (CPU weights, mmap) because another process held about 50 GB of the 96 GB GPU. `\n` marks a newline in the output. Output text begins right after the prompt.

| Dir | File | Size | arch (llama.cpp) | file type | rc | llama.cpp output (16 tokens) |
|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite-Chat | DeepSeek-V2-Lite-Chat-Q4_K_M.gguf | 10.36 GB | deepseek2 | Q4_K_M | 0 | ` Paris.\n\nThe official language of France is French.\n\nThe currency\n\n` |
| MiniCPM3-4B | minicpm3-4b-q4_k_m.gguf | 2.47 GB | minicpm3 | Q4_K_M | 0 | ` Paris.\n\n- Paris, the capital of France, is a city of\n\n` |
| GLM-4-9B-0414 | GLM-4-9B-0414-Q4_K_M.gguf | 6.17 GB | glm4 | Q4_K_M | 0 | ` Paris. It is one of the most beautiful cities in the world. It is\n\n` |
| granite-4.0-h-350m | granite-4.0-h-350m-Q4_K_M.gguf | 0.22 GB | granitehybrid | Q4_K_M | 0 | ` Paris. [end of text]\n\n\n` |
| granite-4.0-h-tiny | granite-4.0-h-tiny-Q4_K_M.gguf | 4.23 GB | granitehybrid | Q4_K_M | 0 | ` Paris." That's a fact. But the sample output is "Paris". So\n\n` |
| granite-3.1-1b-a400m-instruct | granite-3.1-1b-a400m-instruct-Q4_K_M.gguf | 0.82 GB | granitemoe | Q4_K_M | 0 | ` Paris.\n\nThe capital of Germany is Berlin.\n\nThe\n\n` |
| granite-3.3-2b-instruct | granite-3.3-2b-instruct-Q4_K_M.gguf | 1.55 GB | granite | Q4_K_M | 0 | ` Paris.\nThe capital of Germany is Berlin.\nThe capital of\n\n` |
| Falcon-H1-0.5B-Instruct | Falcon-H1-0.5B-Instruct-Q4_K_M.gguf | 0.31 GB | falcon-h1 | Q4_K_M | 0 | ` Paris. Paris is known for its iconic landmarks such as the Eiffel\n\n` |
| mamba-130m-hf | mamba-130m-hf.Q4_K_M.gguf | 89.9 MB | mamba | Q4_K_M | 0 | ` now part of the city of Font-les-Mines.\n\nHistory\n\n` |
| mamba2-130m-hf | mamba2-130m-hf-q8_0.gguf | 0.18 GB | mamba2 | Q8_0 | 0 | ` a city of the French people, and the capital of France is a city of\n\n` |
| rwkv-6-world-1b6 | rwkv-6-world-1.6b-Q8_0.gguf | 1.75 GB | rwkv6 | Q8_0 | 0 | ` Paris.\nThe capital of the United States is Washington, D.C.\n\n` |
| rwkv7-g1h-1.5b | rwkv7-g1h-1.5b-20260710-ctx10240.i1-Q4_K_M.gguf | 1.09 GB | rwkv7 | Q4_K_M | 0 | ` Paris.\nThe capital of France is Paris.\nThe capital of France is\n\n` |
| phi-2 | phi-2.Q4_K_M.gguf | 1.79 GB | phi2 | Q4_K_M | 0 | ` Paris.\n [end of text]\n\n\n` |
| gemma-3n-E2B-it | gemma-3n-E2B-it-Q4_K_M.gguf | 3.03 GB | gemma3n | Q4_K_M | 0 | ` Paris.\n\nThis statement is true.\n\nFinal Answer: The final answer is\n\n` |
| gemma-3-4b-it | gemma-3-4b-it-Q4_K_M.gguf | 2.49 GB | gemma3 | Q4_K_M | 0 | ` Paris.\n\nParis is a global center for art, fashion, gastronomy, and\n\n` |
| gemma-3-4b-it | mmproj-model-f16.gguf | 0.85 GB | clip | mmproj | - | (projector, no text generation) |
| c4ai-command-r7b-12-2024 | c4ai-command-r7b-12-2024-Q4_K_M.gguf | 5.06 GB | cohere2 | Q4_K_M | 0 | ` Paris, which is also the largest city in the country. The city is located\n\n` |
| aya-expanse-8b | aya-expanse-8b-Q4_K_M.gguf | 5.06 GB | command-r | Q4_K_M | 0 | ` Paris. It is located in the north-central part of the country, on\n\n` |
| starcoder2-3b | starcoder2-3b-Q4_K_M.gguf | 1.85 GB | starcoder2 | Q4_K_M | 0 | ` Paris.",\n        "The capital of Germany is Berlin.",\n        "The\n\n` |
| internlm2_5-1_8b-chat | internlm2_5-1_8b-chat-Q4_K_M.gguf | 1.17 GB | internlm2 | Q4_K_M | 0 | ` Paris. The French language is spoken in France, and also in several countries in\n\n` |
| MiniCPM4.1-8B | MiniCPM4.1-8B-Q4_K_M.gguf | 4.97 GB | minicpm | Q4_K_M | 0 | ` Paris.\n[A]. Berlin\n[B]. Madrid\n[\n\n` |
| ERNIE-4.5-0.3B-PT | ERNIE-4.5-0.3B-PT-Q4_K_M.gguf | 0.24 GB | ernie4_5 | Q4_K_M | 0 | `...\n\nThe capital of France is Paris. [end of text]\n\n\n` |
| ERNIE-4.5-21B-A3B-PT | ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf | 13.33 GB | ernie4_5-moe | Q4_K_M | 0 | ` Paris. The capital of Italy is Rome. The capital of Spain is Madrid.\n\n` |
| Hunyuan-0.5B-Instruct | tencent_Hunyuan-0.5B-Instruct-Q4_K_M.gguf | 0.35 GB | hunyuan-dense | Q4_K_M | 0 | ` " " " " " " " " " " " " " " " "\n\n` |
| SmolLM3-3B | SmolLM3-Q4_K_M.gguf | 1.92 GB | smollm3 | Q4_K_M | 0 | ` Paris. The Eiffel Tower is a famous landmark in Paris. The E\n\n` |
| EXAONE-3.5-2.4B-Instruct | EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf | 1.64 GB | exaone | Q4_K_M | 0 | ` Paris. Paris is located in northern France on the Seine River. Paris is\n\n` |
| EXAONE-4.0-1.2B | EXAONE-4.0-1.2B-Q4_K_M.gguf | 0.81 GB | exaone4 | Q4_K_M | 0 | ` is is is is is is is [end of text]\n\n\n` |
| gpt-oss-20b | gpt-oss-20b-MXFP4.gguf | 12.11 GB | gpt-oss | MXFP4_MOE | 0 | ` Paris."\n\nSure! Here's a simple example of a Python program that uses the\n\n` |
| LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Instruct-Q4_K_M.gguf | 0.73 GB | lfm2 | Q4_K_M | 0 | ` the city of Paris. It is the most populous city in the country and\n\n` |
| LFM2.5-8B-A1B | LFM2.5-8B-A1B-Q4_K_M.gguf | 5.16 GB | lfm2moe | Q4_K_M | 0 | ` Paris. city of London is the capital of the United Kingdom. capital of France\n\n` |
| OLMo-2-0425-1B-Instruct | OLMo-2-0425-1B-Instruct-Q4_K_M.gguf | 0.94 GB | olmo2 | Q4_K_M | 0 | ` Paris, and the largest city in France is Lyon. Lyon is known for its\n\n` |
| Ministral-3-3B-Instruct-2512 | Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf | 0.84 GB | clip | mmproj | - | (projector, no text generation) |
| Ministral-3-3B-Instruct-2512 | Ministral-3-3B-Instruct-2512-Q4_K_M.gguf | 2.15 GB | mistral3 | Q4_K_M | 0 | ` Paris. It is the largest city in France and the second largest city in Europe\n\n` |
| stablelm-2-1_6b | stablelm-2-1_6b.q4_k_m.gguf | 1.03 GB | stablelm | Q4_K_M | 0 | ` a city of great historical and cultural significance. The city is the capital of France\n\n` |
| gpt2 | gpt2.Q4_K_M.gguf | 0.11 GB | gpt2 | Q4_K_M | 0 | ` the capital of the Republic of France, and the capital of the Republic of the\n\n` |
| pythia-160m | pythia-160m.Q4_K_M.gguf | 0.11 GB | gptneox | Q4_K_M | 0 | ` located in the city of St-Germain-en-Laye. The\n\n` |
| bloom-560m | bloom-560m.Q4_K_M.gguf | 0.56 GB | bloom | Q4_K_M | 0 | ` Paris. The capital of Belgium is Brussels. The capital of Germany is Berlin.\n\n` |
| flan-t5-small | flan-t5-small.Q4_K_M.gguf | 94.5 MB | t5 | Q4_K_M | 0 | ` France [end of text]\n\n\n` |
| Nemotron-Mini-4B-Instruct | Nemotron-Mini-4B-Instruct-Q4_K_M.gguf | 2.70 GB | nemotron | Q4_K_M | 0 | ` Paris, which is also the country's largest city.\n\n- The\n\n` |
| AI21-Jamba-Reasoning-3B | jamba-reasoning-3b-Q4_K_M.gguf | 1.93 GB | jamba | Q4_K_M | 0 | ` Paris.\n\nThe capital of Germany is Berlin.\n\nThe capital of\n\n` |
| Apertus-8B-Instruct-2509 | Apertus-8B-Instruct-2509-Q4_K_M.gguf | 5.06 GB | apertus | Q4_K_M | 0 | ` Paris, which is also the country's largest city. thus, the answer is\n\n` |
| AFM-4.5B | AFM-4.5B-Q4_K_M.gguf | 2.92 GB | arcee | Q4_K_M | 0 | ` Paris," is a true statement.\n\nTrue.\n\nThe Eiffel Tower is\n\n` |
| SmallThinker-4BA0.6B-Instruct | SmallThinker-4B-A0.6B-Instruct.Q8_0.gguf | 4.55 GB | smallthinker | Q8_0 | 0 | ` Paris. actually, the capital of France is Paris. The capital of France is\n\n` |
| OpenELM-270M-Instruct | OpenELM-270M-Instruct.Q4_K_M.gguf | 0.18 GB | openelm | Q4_K_M | 0 | ` the city of Paris, which is the largest metropolis in the country. The\n\n` |
| Nanbeige4.2-3B | Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf | 2.68 GB | nanbeige | Q4_K_M | 0 | ` Paris.\n</think>\n\nHere are the **top 5 most important things\n\n` |
| Qwen3-VL-2B-Instruct | Qwen3VL-2B-Instruct-Q4_K_M.gguf | 1.11 GB | qwen3vl | Q4_K_M | 0 | ` Paris, and the capital of Germany is Berlin. If the capital of France is\n\n` |
| Qwen3-VL-2B-Instruct | mmproj-Qwen3VL-2B-Instruct-F16.gguf | 0.82 GB | clip | mmproj | - | (projector, no text generation) |
| Phi-4-mini-instruct | Phi-4-mini-instruct-Q4_K_M.gguf | 2.49 GB | phi3 | Q4_K_M | 0 | ` Paris. Paris is the capital of France. Paris is located in France. Paris\n\n` |
| Mistral-7B-Instruct-v0.3 | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | 4.37 GB | llama | Q4_K_M | 0 | ` Paris, but the largest city is Marseille.\n\nMarseille\n\n` |
| bitnet-b1.58-2B-4T | bitnet-2b4t-tq2_0.gguf | 1.20 GB | bitnet | TQ2_0 | 0 | ` the capital of France. The capital of France is the capital of France.The\n\n` |
| LFM2-VL-450M | LFM2-VL-450M-Q8_0.gguf | 0.38 GB | lfm2 | Q8_0 | 0 | ` Paris. The capital of France is Paris. [end of text]\n\n\n` |
| LFM2-VL-450M | mmproj-LFM2-VL-450M-F16.gguf | 0.19 GB | clip | mmproj | - | (projector, no text generation) |
| Qwen3.8-Flash-Next-UD-Q4_K_XL | Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf (+3 shards) | 111.33 GB | qwen4exp | Q4_K_M | 0 | ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome.\n\n` |
| Qwen3.8-Flash-Next-UD-Q4_K_XL | mmproj-F16.gguf | 0.90 GB | clip | mmproj | - | (projector, no text generation) |
| Qwen3-Coder-Next | Qwen3-Coder-Next-Q4_K_M.gguf | 48.53 GB | qwen3next | Q4_K_M | 0 | ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome.\n\n` |
| Qwen3.6-35B-A3B | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | 22.13 GB | qwen35moe | Q4_K_M | 0 | ` Paris, a city renowned for its iconic landmarks such as the Eiffel Tower,\n\n` |
| Qwen3.6-35B-A3B | mmproj-F16.gguf | 0.90 GB | clip | mmproj | - | (projector, no text generation) |

Notes: (a) The Qwen3.8-Flash-Next GGUF directory is named after its quant (`...-UD-Q4_K_XL`, 111 GB, Unsloth) while its HF config lives in `hf/Qwen3.8-Flash-Next`. No GGUF of this model at or below ~50 GB exists (smallest: 66 GB IQ1_S). An AtomicChat file labelled `4.27bpw-Q4_K_M` (94.5 GB) was downloaded first and deleted again: its tensors are IQ2_S/IQ3_S/IQ4_NL/Q8_0 plus a Q5_1 n-gram table, which Sharplight cannot run today; llama.cpp b11177 loaded it and greedy-generated ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome.` (a different quant, so not a parity reference). (b) Qwen3-Coder-Next 'Q4_K_M' is a plain 48.5 GB Unsloth file (the ~50 GB Next-class model). (c) Outputs of chat/instruct models given a raw prompt without a chat template are sometimes degenerate (EXAONE-4.0, Hunyuan-0.5B); that is what llama.cpp produces and is still a valid greedy reference. (d) rwkv-6/7, mamba, jamba, flan-t5 and the other non-transformer models all loaded and generated in b11177.

### 8.2 Hugging Face checkpoints (`~/models/hf/<name>/`)

Tiny random checkpoints (weights random, structure real) are for graph/shape/mapping tests against transformers; `config + tokenizer` entries are real model configs without weights.

| Dir | Size | Contents |
|---|---|---|
| tiny-random-qwen3.8-flash-next | 33.8 MB | model_type `qwen4_exp`, 1 safetensors |
| tiny-random-qwen3-next-moe | 21.6 MB | model_type `qwen3_next`, 1 safetensors |
| tiny-random-qwen3.5-moe | 30.1 MB | model_type `qwen3_5_moe`, 1 safetensors |
| tiny-random-qwen3.6-moe | 32.3 MB | model_type `qwen3_5_moe`, 1 safetensors |
| tiny-random-qwen3.8-moe | 36.0 MB | model_type `qwen3_5_moe_text`, 1 safetensors |
| tiny-random-qwen3-vl-moe | 22.2 MB | model_type `qwen3_vl_moe`, 1 safetensors |
| tiny-random-deepseek-v3.1 | 20.2 MB | model_type `deepseek_v3`, 1 safetensors |
| tiny-random-deepseek_v32 | 16.8 MB | model_type `deepseek_v32`, 1 safetensors |
| tiny-random-minicpm3 | 87.2 MB | model_type `minicpm3`, 1 safetensors |
| tiny-random-glm-4 | 29.7 MB | model_type `glm4`, 1 safetensors |
| tiny-random-glm-4-moe | 25.1 MB | model_type `glm4_moe`, 1 safetensors |
| tiny-random-glm-4.7-flash | 31.8 MB | model_type `glm4_moe_lite`, 1 safetensors |
| tiny-random-falcon-h1 | 26.2 MB | model_type `falcon_h1`, 1 safetensors |
| tiny-random-granite-4.0-h | 21.4 MB | model_type `granitemoehybrid`, 1 safetensors |
| tiny-random-granite-moe-hybrid | 21.4 MB | model_type `granitemoehybrid`, 1 safetensors |
| tiny-random-granite | 11.4 MB | model_type `granite`, 1 safetensors |
| tiny-random-granite-moe | 17.9 MB | model_type `granitemoe`, 1 safetensors |
| tiny-random-kimi-linear | 14.5 MB | model_type `kimi_linear`, 1 safetensors |
| tiny-random-kimi-k3 | 0.14 GB | model_type `kimi_k3`, 1 safetensors |
| tiny-random-lfm2 | 13.5 MB | model_type `lfm2`, 1 safetensors |
| tiny-random-lfm2-moe | 15.0 MB | model_type `lfm2_moe`, 1 safetensors |
| tiny-random-gpt-oss | 41.6 MB | model_type `gpt_oss`, 1 safetensors |
| tiny-random-gpt-oss-mxfp4 | 41.0 MB | model_type `gpt_oss`, 1 safetensors |
| tiny-random-smollm3 | 33.8 MB | model_type `smollm3`, 1 safetensors |
| tiny-random-seed-oss | 16.9 MB | model_type `seed_oss`, 1 safetensors |
| tiny-random-hunyuan-dense-v1 | 6.3 MB | model_type `hunyuan_v1_dense`, 1 safetensors |
| tiny-random-hunyuan-moe | 21.7 MB | model_type `hunyuan`, 1 safetensors |
| tiny-random-ernie-4.5 | 14.8 MB | model_type `ernie4_5`, 1 safetensors |
| tiny-random-ernie-4.5-moe | 14.5 MB | model_type `ernie4_5_moe`, 1 safetensors |
| tiny-random-minicpm4 | 17.4 MB | model_type `minicpm`, 1 safetensors |
| tiny-random-bailing-moe-v2 | 18.2 MB | model_type `architectures BailingMoeV2ForCausalLM (no model_type; remote code)`, 1 safetensors |
| tiny-random-mistral-3 | 36.8 MB | model_type `mistral3`, 1 safetensors |
| tiny-random-apertus | 21.5 MB | model_type `apertus`, 1 safetensors |
| tiny-random-phi-moe | 8.6 MB | model_type `phimoe`, 1 safetensors |
| tiny-random-llama-4 | 41.6 MB | model_type `llama4`, 1 safetensors |
| tiny-random-gemma-3n | 62.9 MB | model_type `gemma3n`, 1 safetensors |
| tiny-random-minimax-m2 | 28.5 MB | model_type `minimax_m2`, 1 safetensors |
| tiny-random-exaone | 21.5 MB | model_type `exaone`, 1 safetensors |
| tiny-random-internlm2 | 0.10 GB | model_type `internlm2`, 1 safetensors |
| tiny-random-phi3 | 14.9 MB | model_type `phi3`, 1 safetensors |
| tiny-Cohere2ForCausalLM | 25.6 MB | model_type `cohere2`, 1 safetensors |
| tiny-CohereForCausalLM | 24.2 MB | model_type `cohere`, 1 safetensors |
| tiny-random-T5Model | 3.2 MB | model_type `t5`, config + tokenizer only (the hub repo ships pytorch_model.bin, not downloaded; convert to safetensors if needed) |
| tiny-random-BloomForCausalLM | 0.5 MB | model_type `bloom`, 1 safetensors |
| tiny-random-GPTNeoXForCausalLM | 0.5 MB | model_type `gpt_neox`, 1 safetensors |
| tiny-random-gemma-3 | 56.4 MB | model_type `gemma3`, 1 safetensors |
| tiny-random-glm-4.5 | 25.1 MB | model_type `glm4_moe`, 1 safetensors |
| tiny-random-qwen3 | 35.5 MB | model_type `qwen3`, 1 safetensors |
| tiny-random-ernie-4.5-vl-moe | 5.9 MB | model_type `ernie4_5_moe_vl`, 1 safetensors |
| tiny-random-MambaForCausalLM | 0.2 MB | model_type `mamba`, 1 safetensors |
| tiny-random-OlmoForCausalLM | 16.5 MB | model_type `olmo`, 1 safetensors |
| tiny-random-Olmo2ForCausalLM | 35.4 MB | model_type `olmo2`, 1 safetensors |
| tiny-random-Starcoder2ForCausalLM | 0.2 MB | model_type `starcoder2`, 1 safetensors |
| tiny-random-MixtralForCausalLM | 12.6 MB | model_type `mixtral`, 1 safetensors |
| tiny-random-FalconForCausalLM | 0.3 MB | model_type `falcon`, 1 safetensors |
| tiny-random-StableLmForCausalLM | 0.8 MB | model_type `stablelm`, 1 safetensors |
| tiny-random-PhiForCausalLM | 0.4 MB | model_type `phi`, 1 safetensors |
| tiny-random-RwkvForCausalLM | 0.0 MB | model_type `rwkv`, config + tokenizer only (the hub repo ships pytorch_model.bin, not downloaded; convert to safetensors if needed) |
| tiny-random-Qwen2MoeForCausalLM | 55.1 MB | model_type `qwen2_moe`, 1 safetensors |
| tiny-random-GPTJForCausalLM | 0.1 MB | model_type `gptj`, config + tokenizer only (the hub repo ships pytorch_model.bin, not downloaded; convert to safetensors if needed) |
| tiny-random-MptForCausalLM | 0.0 MB | model_type `mpt`, config + tokenizer only (the hub repo ships pytorch_model.bin, not downloaded; convert to safetensors if needed) |
| tiny-random-ModernBertForMaskedLM | 10.4 MB | model_type `modernbert`, 1 safetensors |
| tiny-random-BartForCausalLM | 0.1 MB | model_type `bart`, config + tokenizer only (the hub repo ships pytorch_model.bin, not downloaded; convert to safetensors if needed) |
| ERNIE-4.5-0.3B-PT | 0.73 GB | model_type `ernie4_5`, 1 safetensors |
| Hunyuan-0.5B-Instruct | 1.08 GB | model_type `hunyuan_v1_dense`, 1 safetensors |
| granite-4.0-h-350m | 0.69 GB | model_type `granitemoehybrid`, 1 safetensors |
| Falcon-H1-0.5B-Instruct | 1.05 GB | model_type `falcon_h1`, 1 safetensors |
| mamba-130m-hf | 0.52 GB | model_type `mamba`, 1 safetensors |
| mamba2-130m-hf | 0.52 GB | model_type `mamba2`, 1 safetensors |
| OLMo-2-0425-1B-Instruct | 2.98 GB | model_type `olmo2`, 1 safetensors |
| granite-3.1-1b-a400m-instruct | 2.67 GB | model_type `granitemoe`, 1 safetensors |
| EXAONE-4.0-1.2B | 2.57 GB | model_type `exaone4`, 1 safetensors |
| LFM2.5-1.2B-Instruct | 2.35 GB | model_type `lfm2`, 1 safetensors |
| Qwen3-VL-2B-Instruct | 4.27 GB | model_type `qwen3_vl`, 1 safetensors |
| SmolLM3-3B | 6.17 GB | model_type `smollm3`, 2 safetensors |
| Qwen3.8-Flash-Next | 23.1 MB | model_type `qwen4_exp`, config + tokenizer + safetensors index only (weights deliberately not downloaded: too large) |
| Qwen3-Coder-Next | 18.3 MB | model_type `qwen3_next`, config + tokenizer + safetensors index only (weights deliberately not downloaded: too large) |
| Qwen3.6-35B-A3B | 23.0 MB | model_type `qwen3_5_moe`, config + tokenizer + safetensors index only (weights deliberately not downloaded: too large) |

## 9. Reproducing the references

```bash
D=$LLAMA_ROOT/runtime.llama-b11177-cuda13   # or -vulkan / -cpu
LD_LIBRARY_PATH=$D $D/llama-completion -m MODEL.gguf -p 'The capital of France is' -n 16 \
    --temp 0 --top-k 1 --seed 1 -c 512 -no-cnv --no-display-prompt -t 16 --no-warmup -ngl 99 </dev/null
```

The GPU is shared with other applications: for a model that does not fit, drop `-ngl` (llama.cpp then fits automatically) or use the cpu build. Greedy outputs of 16 tokens are sensitive to quantisation and to the backend's float summation order; use them as a first parity gate (identical text expected for the same file on CPU) and compare logits/token ids for finer checks. Quantised GGUFs from different repackagers of the same model are not interchangeable references.

