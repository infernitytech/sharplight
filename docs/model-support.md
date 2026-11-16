# Model support survey and bring-up plan

Status: survey written 2026-11-13 against llama.cpp `master` (src/llama-arch.cpp `LLM_ARCH_NAMES`, src/models/*.cpp graph builders, tools/mtmd/models) and the Sharplight tree as found in the working copy. Reference runtime for parity outputs: llama.cpp build b11177 (`llama-completion`).

Contents: 1 summary, 2 what Sharplight supports today, 3 building blocks ranked by unlock value, 4 architecture matrix, 5 bring-up waves, 6 Qwen3.8-Next (qwen4exp) deep dive, 7 multimodal projectors, 8 test assets and llama.cpp reference outputs, 9 how the references were produced, 10 the real-checkpoint sweep (round 10), 11 bring-up round 11 (the sweep's open rows), 12 text to speech, 13 speech to text.

## 1. Summary

llama.cpp lists **172** architecture names (`LLM_ARCH_NAMES`, excluding `(unknown)`; `clip` is the mmproj pseudo-architecture). Classification used throughout:

- **supported**: imports (GGUF and/or HF) and runs end to end on the CPU and Vulkan providers through an existing adapter, with a test or fixture behind it.
- **partial**: the graph can be expressed with ops and kernels Sharplight already has. What is missing is an importer mapping (GGUF arch name, hyper-parameters, tensor names, HF `model_type`) and/or a small adapter feature (LayerNorm, parallel residual, branch scale, routing bias, an activation). No new kernel.
- **missing**: needs a new op/kernel, a new cache kind, a new model topology (encoder, encoder-decoder) or a runtime feature (diffusion, speculative decoding).

| Status | Architectures |
|---|---|
| supported | 168 |
| partial | 1 |
| missing | 3 |
| total | 172 |

Counts as of 2026-11-14, recounted from the matrix in section 4 (the survey started at 15 / 90 / 67; before the first 2026-11-14 round the matrix stood at 79 / 39 / 54, the summary had not been updated). Of the 4 missing ones, 3 are TTS/audio-codec decoders that are out of scope for a text runtime (qwen3tts, pockettts, wavtokenizer-dec). The 2026-11-14 round moved 23 rows to supported: qwen2moe, starcoder, refact, jais, jais2, orion, codeshell, plamo, plamo3, pangu-embedded, maincoder, spark2_5, baichuan, granite_swa, mellum, dbrx, deepseek, bailingmoe, mistral4, plm and cohere2moe (importer mappings, `moe.shared_scale`, latent-decoder query temperature; tiny random models checked against transformers logits/greedy ids and llama.cpp b11177 greedy ids, `tests/Sharplight.Families.Tests/MappedFamilyCases.cs`), plus bert and nomic-bert, which the encoder adapter already ran (`Sharplight.Embeddings.Tests`). The second 2026-11-14 round moved 16 more: the embedding families eurobert, neo-bert, llama-embed, gemma-embedding, modern-bert and jina-bert-v2 (bidirectional decoder layers with symmetric sliding windows and Dense projections after pooling, encoder ALiBi; tiny models vs transformers and llama-server `/embedding`, real ModernBERT-base and EuroBERT-210m vs transformers and jina-embeddings-v2-small-en vs llama.cpp, `tests/Sharplight.Families.Tests/EmbeddingFamilyTests.cs`), and talkie, nanbeige, arctic, deci, grok, maple, step35, kolibri1, ricursive and phimoe (clamped SwiGLU, sparsemixer routing, layer loops, value residual, embedding skip, attention-free and MLP-free layers; tiny GGUFs vs llama.cpp greedy ids, real Nanbeige4.2-3B). ALiBi now runs in the Vulkan attention kernels, so bloom, mpt, refact, jais and Baichuan-13B no longer fall back to the CPU. The third 2026-11-14 round moved 10 more: lfm3, maion-coder, beam, muse-spark and clover_moe (tiny GGUFs vs llama.cpp 8ca5a80, which b11177 predates; LFM3-1.7B vs llama.cpp), t5 (new `encoder_decoder` adapter with relative position bias and cross-attention, also running the HF-only bart and mbart; flan-t5-small vs llama.cpp and transformers), qwen3vl and qwen3vlmoe (DeepStack towers and M-RoPE on the decoder's multimodal graph; Qwen3-VL-2B with an image vs transformers and llama.cpp), qwen2vl (language model, tiny models vs transformers and llama.cpp b11177) and hy_v4 (hyper-connections as the shared `StreamHyperConnections` block, attention sinks and output gate on latent attention; tiny models vs transformers); t5encoder went from missing to partial (mapped, not yet verified). The fourth 2026-11-14 round moved 3 more: gemma3n (AltUp streams, LAuReL and Gaussian activation sparsity as decoder blocks on the Gemma 4 path; tiny float32 model vs transformers, its converted GGUF vs the same ids, gemma-3n-E2B-it Q4_K_M vs llama.cpp b11177, CPU and Vulkan), paddleocr (language model: ERNIE-4.5 with sectioned M-RoPE; tiny model vs transformers and llama.cpp 11139, which has the architecture b11177 lacks) and t5encoder (tiny GGUF vs transformers' T5EncoderModel). The fifth 2026-11-14 round moved 3 more: kimi-linear and bailingmoe3 (Kimi delta attention: `GatedDeltaRule` with a decay per key channel, CPU and Vulkan, as the shared `ChannelDeltaAttention` mixer on the latent decoder; latent attention without rope and with a per-head output gate) and minimax-01 (lightning attention: the additive, decay-only form of `GatedDeltaRule` with time-invariant gates as the shared `LightningAttention` mixer on the hybrid stack, residuals from the normalized input with per-layer scales); tiny float32 models vs transformers (greedy ids and prompt logits, CPU and Vulkan, plus recurrent rollback), converted tiny GGUFs (`tools/convert_tiny_gguf.py`) vs llama.cpp b11177 greedy ids, `tests/Sharplight.Families.Tests/LinearAttentionFamilyCases.cs`. MiniMax deviation: transformers rotates the whole head of the softmax layers, the original model and llama.cpp only `rotary_dim` channels; the importers follow `rotary_dim` when the config has it. The sixth 2026-11-14 round moved deepseek32 and glm-dsa (DSA indexer in its dense regime, as for hy_v4: the context is capped at `index_topk`, where the indexer keeps every visible token; tiny models with a budget covering the test sequence vs transformers, the deepseek32 GGUF vs llama.cpp b11177; llama.cpp's glm-dsa graph leaves the transformers ids at the 4th token on the same weights, so that file is checked against transformers). The seventh 2026-11-14 round moved the masked-diffusion models dream, llada, llada-moe and rnd1 (a bidirectional `denoise` graph with logits for every row on the shared decoder, and `DiffusionGenerator` for the decoding loop: Dream's timestep schedule or LLaDA's block schedule, confidence / margin / entropy unmasking, greedy; exposed to `IChatClient` users through `DiffusionGenerationBackend`, which `SharplightChatClient` picks for diffusion models). Tiny random models (`tools/mk_tiny_diffusion.py`: transformers' forward pass with an all-visible mask plus the reference decoding loops written out from Dream's `diffusion_generate` and LLaDA's `generate.py`) and their converted GGUFs give the same 24 ids on CPU and Vulkan (`tests/Sharplight.Families.Tests/DiffusionGenerationTests.cs`). llama.cpp b11177 ships no `llama-diffusion-cli`, so the GGUFs were checked with one built from the b11139 source: its sampler always draws through a `dist` sampler, so it is deterministic only with `--top-k 1` (every confidence 1, earliest masked positions first); in that mode all four GGUFs give the same text as the same loop on transformers. LLaDA-MoE's Hugging Face checkpoints reuse LLaDA's model type `llada` with Qwen3-MoE tensor names, so only its GGUF is mapped. RND1's own sampler keeps the highest-entropy positions masked on a linear schedule; `DiffusionUnmasking.Entropy` with the timestep schedule is the closest setting. The Qwen2-VL and Qwen2.5-VL image towers now run too (windowed attention, RMS / LayerNorm, gated / quick-GELU MLPs; HF and mmproj, verified against transformers and llama.cpp with Qwen2-VL-2B and Qwen2.5-VL-3B). The eighth 2026-11-14 round added the sparse regime of the DSA indexer, so deepseek32, glm-dsa and hy_v4 are no longer capped at `index_topk`: `IndexerTopK` scores the visible keys of the indexer key cache per row (Σ_h w·relu(q·k)) and picks the k best (exact positions, ties to the lower position; a workgroup radix select shared with `TopK` on Vulkan), and `CachedAttention` with `selection` attends only to those positions; both have the multi-sequence form. The indexer runs only when the context exceeds the budget. Tiny models with `index_topk` 8 and 16 indexer heads (scores re-initialized so that torch.topk has no ties at the boundary) match transformers on CPU and Vulkan (prompt logits, 24 greedy ids, Hunyuan v4 with a shared indexer layer), batched decoding matches single-sequence decoding, and the deepseek32 GGUF matches llama.cpp b11177; llama.cpp's glm-dsa graph leaves the transformers ids at the 6th token, as in the dense regime. The masked-diffusion models are now served by the continuous-batching engine as well (and so by the serving host's chat-model loader and its OpenAI / Anthropic endpoints): `LlmModel.CreateBatchDecoder` returns a `DiffusionBatchDecoder` for them, and the engine keeps one `DiffusionSequence` (the schedule state shared with `DiffusionGenerator`) per running request, advancing all of them one denoising step per scheduler step in packed passes, streaming each block's tokens once it is final; `GenerationRequest.Diffusion` (and `ChatRequest` / `CompletionRequest.Diffusion`) carries the schedule. Requests of different lengths denoised together give exactly the single-sequence generator's ids, and the reference prompt the transformers reference, on CPU and Vulkan (`tests/Sharplight.Families.Tests/DiffusionServingTests.cs`). Sampling settings, log-probabilities and minimum lengths do not apply to diffusion decoding; output constraints are rejected. The ninth 2026-11-14 round moved jina-bert-v3 (jina-embeddings-v3: the encoder adapter's post-norm layers with rope and a fused qkv; Hugging Face `jina_embeddings_v3` and GGUF `jina-bert-v3`, tiny models vs transformers and llama.cpp b11177) and nomic-bert-moe (nomic-embed-text-v2-moe: the shared `MoeBlock` in the encoder adapter; tiny GGUF vs llama.cpp b11177). The tenth 2026-11-14 round moved chameleon (per-head q/k layer norms with a gain and bias per head, `attention.qk_norm_type`, swin-norm layers as post-norm, the image-token logits suppressed through `logits.suppressed_ranges`; HF and GGUF, tiny models vs transformers and llama.cpp 8ca5a80, whose swin-norm graph feeds the MLP the normed attention output instead of the residual stream, so the swin GGUF is checked against transformers) and gemma-embedding2 (EmbeddingGemma 2: the bidirectional decoder with per-layer head sizes, unscaled attention, v norm and per-layer embeddings computed from a projection of the token embeddings alone; HF only, tiny model vs transformers, no llama.cpp release has the architecture) to supported, and minimax-m3 and glm5-next from missing to partial: MiniMax-M3 is the decoder with per-head Gemma q/k norms, partial rope, dense and MoE layers with the clamped SwiGLU-OAI and a shared expert (HF `minimax_m3_vl` and GGUF `minimax-m3`), GLM-5.3-Flash the latent decoder with Kimi delta attention, rope-less latent attention (0 rotary dimensions), SwiGLU clamped in every MLP (`mlp.swiglu_clamp`) and DeepSeek-V4 hyper-connections averaged by the final head (HF `glm5_next` and GGUF `glm5-next`). Both select keys with indexers that differ from DSA's token indexer (MiniMax: blocks of keys per key-value group; GLM-5.3-Flash: k-pools of keys plus the current tail), and only their dense regime is implemented, so the context is capped at the indexer budget (`index_topk_blocks` x `index_block_size`, `index_topk`); tiny models whose budget covers the test sequence match transformers and llama.cpp 8ca5a80 on CPU and Vulkan, batched decoding matches single-sequence decoding (`tests/Sharplight.Families.Tests/ExtendedFamilyCases.cs`, `ExtendedFamilyServingTests.cs`). The eleventh 2026-11-14 round (bring-up round 8) moved minimax-m3 and glm5-next to supported: `IndexerTopK` now selects aligned groups of positions (`group`), scored by the best position of the group (`group_score` "max", MiniMax) or by a softmax-gated pooled key over complete groups (`group_score` "pooled", GLM-5.3-Flash, the cache rows holding key and gate), with the latest groups forced (`local`), the incomplete group appended (`tail`) and one selection per indexer head (`per_head`, output [T, H, width]); `CachedAttention` takes such a per-head selection (query head h reads selection head h / (Hq / Hs)). Both run on CPU and Vulkan (one workgroup per row and selection head, radix select over the groups) with the multi-sequence and paged forms, so neither model is capped at its budget any more. MiniMax's block indexer is the decoder's `BlockSparseIndexer` (`attention.block_indexer.*`), GLM-5.3-Flash's k-pool indexer the latent decoder's `SparseAttentionIndexer` with `pool_size` / `pool_tail`. Tiny models in the sparse regime (budgets of 12 and 8 positions, indexer weights re-initialized by `tools/mk_tiny_group_sparse.py` so that no group scores tie at the boundary) match transformers on CPU and Vulkan, HF and GGUF; the GLM-5.3-Flash GGUF also matches llama.cpp 8ca5a80, while llama.cpp's batched block-sparse MiniMax prefill continues differently from transformers (token-by-token decoding gives the transformers ids); batched decoding matches single-sequence decoding. The same round moved mistral-large4 and dots3note (GGUF only) and eagle3 to supported. Mistral Large 4 is the latent decoder with per-layer rope (`rope.layers`: rope-less layers keep the whole head in the latent), layers reading an earlier layer's latent cache (`attention.kv_source_layers`), scalable softmax (the query scaled by s_h·ln(pos+1)+b_h per head, `ScalableSoftmax`, on the attention and the indexer queries), a LayerNorm'd indexer query, learned yarn factors and final logit softcapping; dots.3 adds sliding-window layers with their own MLA geometry, head count and rope base (`attention.sliding.*`), an RMS norm on the rope key and a per-head sigmoid attention gate. Tiny GGUFs (`tools/mk_tiny_gguf.py`) match llama.cpp 8ca5a80 on CPU and Vulkan (mistral-large4: llama.cpp with flash attention or token by token, and a float64 numpy reference of its graph, `tools/ref_mistral_large4.py`; llama.cpp's batched prefill without flash attention continues differently). EAGLE-3 drafters load as the `draft` auxiliary source of their target (`LlmLoadOptions.Auxiliary`): the target's decode graphs export the inputs of the three fused layers (`hidden_taps`), and the drafter (`Eagle3DraftHead`: fc over the fused features, one decoder layer over the normed embedding and features, its own vocabulary mapped through `d2t`) becomes the draft graph that `MtpSpeculator` runs, its rows [target features | fed-back pre-norm state]. A tiny Llama target and drafter (`tools/mk_tiny_eagle3.py`, HF and converted GGUF) propose the drafts of a transformers + numpy reference, and greedy output with speculation equals output without, on CPU and Vulkan (`SpeculativeDecodingTests`). DFlash drafters (block diffusion over the target's features, with the DFlash2 and LiLiCorr variants) are still missing. The twelfth 2026-11-14 round (bring-up round 9) moved 16 missing and all 3 partial rows to supported (168 / 0 / 4). A Vulkan attention head size that is not a multiple of 8 no longer fails loading: the per-head kernel reads one key element per load for such heads (tiny Llama with head size 12 vs transformers, kernel cases with head sizes 12 and 20). Vision / OCR: deepseek2-ocr (DeepSeek-OCR, Unlimited-OCR and DeepSeek-OCR-2 from HF and GGUF + mmproj: DeepSeek-V2's MoE with multi-head attention, and a SAM ViT-B stage with windowed relative-position attention and a convolution neck over a padded 1024-pixel global view and tiles, followed by CLIP-L with newline and view-separator rows or, for OCR-2, a Qwen2 query encoder over the SAM features and learned queries; tiny checkpoints match transformers to 2.5e-5 relative; llama-mtmd-cli's global view is 2e-3 (OCR) / 9e-3 (OCR-2) from transformers, and its tile positions read the class token's position from the wrong row), hunyuan_vl (XD-RoPE as a generic per-element rope-section variant, the perceiver projector; text identical to llama.cpp 8ca5a80, its tower differs by its (n+0.1)/grid resampling and tanh GELU) and cogvlm (visual experts as the generic per-token weight-set block `TokenWeightSets`, the EVA2-CLIP tower; llama.cpp 8ca5a80 samples the first prompt row because its graph lacks output-row selection, its tower assigns the patches shifted position rows and its converter drops the visual experts, so the remote code is the reference). Tiny GGUF + mmproj pairs of all six tiny VL families are now checked end to end against transformers (`Tiny_vl_gguf_mmproj_image_prompt_matches_transformers`). Latent decoder: kimi-k3 / kimi-k31 (block attention residuals `AttentionResiduals`, the situ activation, a norm on the routed-expert sum of latent MoE, per-channel delta-attention decay; K3.1 adds a DSA indexer without rope and layers sharing the previous selection; the published MXFP4 compressed-tensors checkpoint and Kimi's tiktoken tokenizer load as is; llama.cpp's converter keeps one decay per head, so the GGUFs have their own reference, and its batched prefill differs from its token-by-token decoding, which matches), xing4_0 (manifold hyper-connections with a mixing clamp, transposed mixing and eps options of `StreamHyperConnections`; llama.cpp's fused mixing kernel has no clamp, so its ids are checked on an unclamped model) and deepseek4 (compressed KV caches as a new cache kind, `StateDefinition.Compression`, with the `WindowPool` op and a `compression` attribute on CacheWrite / CachedAttention / IndexerTopK, single-sequence, multi-sequence and paged; hash routing from `router.token_experts`; tiny model vs transformers and llama.cpp 8ca5a80 with f32 caches; Vulkan kernels for every compressed-cache op, no CPU fallback) and deepseek41 (GGUF: Engram per-layer n-gram embeddings (`EngramMemory`, `NGramHash` with `stop_cuts`), layers sharing compressed caches and selections, two-level candidate selection (`IndexerTopK` `candidate_group`); tiny GGUF `tools/mk_tiny_deepseek_v41.py` vs a float64 numpy reference of llama.cpp's graph (`tools/ref_deepseek_v41.py`) and llama.cpp 8ca5a80 decoding token by token, f32 caches; llama.cpp's batched prompt decode continues differently from its own token-by-token decoding). Decoder: grovemoe (adjugate experts in the shared `MoeBlock`; llama.cpp computes the adjugate branch differently from the remote code, which is the reference), k2-horizon (`ValueExperts`, MoVA routed value experts, RMS norms over channel groups, a softplus output gate), hrm_text (`HierarchicalCycles`: low and high stacks of shared-weight layers with per-pass KV caches), sensenova_u1 (q/k norms per part of the head, `attention.qk_norm_groups`; text only), limite (value embeddings, exclusive self attention, MUDD `DenseConnections`, a sigmoid logit cap; HF remote-code checkpoints with the converter's folds applied at import), graniteswitch (`AdapterRouter`, a one-head attention over a per-token table choosing each row's adapter, and `SwitchedLora` deltas through MatMulId) and metis16 (canonical `depth_memory`: multi-pass Mamba-2 / attention over parallel streams with `StaticStreamMixing`, a per-token `DepthMemory`, routing-feature-aware latent experts and a continuation head; GGUF only). qwen4exp's QSA indexer now runs its sparse regime (`IndexerTopK` group score "mean" with an optional pool norm and rope; the context is no longer capped). Gemma 4 assistants run as the target's draft graph (canonical `shared_kv_draft`: the target exports its final hidden state and the keys and values of its last full and sliding layers; centroid-masked head; drafts match transformers, whose sliding layers see window + 1 keys where llama.cpp's see window - 1). The decision models clef and vllm-decision2 score options of question spans through the generic `score_spans` graph and `SpanScorer` (`Sharplight.Llm.Scoring`), head chosen by `span_head.type` (`joint_schema`, `endpoint`). Every family has tiny models vs transformers, the remote code or a numpy reference of the llama.cpp graph, and vs llama.cpp 8ca5a80 where it has the architecture, on CPU and Vulkan, with batched decoding matching single-sequence decoding (case files CrossLayerFamilyCases, DeepseekOcrFamilyCases, MoeVariantFamilyCases, AdapterFamilyCases, KimiK3FamilyCases, DeepSeekV4Cases, SparseQsaCases, VisionLanguageFamilyCases; SpanScoringTests, SpeculativeDecodingTests, TinyDeepseekOcrImageTests). CPU and Vulkan segments of one compilation now share the states they touch (`StateHandoffTests`). Still missing: dflash (skipped this round) and the three TTS decoders. The 2026-11-15 round (bring-up round 10) swept released checkpoints instead of architectures (section 10): of 1138 GGUF files in 88 Hub repositories, the header-only check and the real import now agree on every one, 1051 import, and the rejections are draft files, files without an architecture, dflash / inkling, unknown fork-only ggml types, NVFP4 per-tensor scales and incomplete uploads; greedy output equals llama.cpp 8ca5a80 on Vulkan and CPU for 9 checkpoints and differs elsewhere only by rounding (f32-reference correlation ≥ 0.989). The round added latent and heterogeneous (Puzzle) experts and MTP-block handling for Nemotron-H, managed decoders for every ggml block format and the gemma-embedding2 GGUF mapping. Bring-up round 11 (section 11) moved dflash to partial (plain DFlash and DFlash2 drafters; the LiLiCorr variant is open) and worked through the sweep's open rows: HF mappings for lfm3, spark2_5, clover_moe, nemotron_h_omni, mimo_v2, llava_onevision2 and paligemma (partial), the fork architecture inkling (a row of its own in section 4, outside the 172) and the fork-only ggml types 64–66 and 144–149; the three TTS decoders remain missing.

Headline findings:

1. Qwen3.5/3.6/3.8 **dense** (`qwen35`) works. Three further Qwen architectures matter next: `qwen35moe` (Qwen3.5/3.6-35B-A3B) and `qwen3next` (Qwen3-Next-80B, Qwen3-Coder-Next) are *partial*: Qwen35Adapter plus the MoE sub-graph that DecoderAdapter already has, plus a sigmoid-gated shared expert. `qwen4exp` (what the Hub calls **Qwen3.8-Flash-Next**, HF `qwen4_exp`) is *missing*: it adds hyper-connections, a sparse-attention indexer and n-gram hash embeddings (section 6).
2. The generic decoder adapter already covers a surprising amount (sliding window, softcap, sinks, QK-norm, partial/NoPE rope, per-layer head sizes, KV sharing, PLE, sigmoid MoE with shared expert, longrope/yarn). The bulk of the 90 partial architectures needs only LayerNorm + parallel residual, branch scaling, routing bias/groups and mappings.
3. Weight formats were a second, independent gap and are now closed by quantisation track Q: F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/IQ2_S, IQ3_XXS/IQ3_S, IQ4_NL, IQ4_XS, TQ1_0/TQ2_0, MXFP4, NVFP4, FP8 (E4M3 / E5M2 and block-scaled E4M3, section 2.4) and the Bonsai Q1_0/Q2_0/PQ2_0/PTQ1_0 have CPU and Vulkan compute paths (dequantisation bit-exact against ggml, see `tests/Sharplight.Cpu.Tests/GgmlDequantTests.cs`). Only Q8_K (activation-only) and the GGUF `f8_e4m3` type id are registered without kernels. Unsloth `UD-*` quants mix Q2_K/Q3_K/IQ tensors, gpt-oss is MXFP4, so this gated real models regardless of architecture. Concrete example (tensor types read from the GGUF headers): Qwen3.8-Flash-Next `UD-Q4_K_XL` uses only Q8_0, Q5_1, Q4_K, Q5_K, IQ4_NL, BF16 and F32, while the other community quants of the same model contain IQ2_S, IQ3_S or IQ3_XXS tensors (and one unknown ggml type id 42) (the AtomicChat file named `4.27bpw-Q4_K_M` is in fact IQ2_S/IQ3_S/IQ4_NL/Q8_0 plus a Q5_1 table). Quant names do not predict kernel coverage: check the tensor types.

## 2. What Sharplight supports today

### 2.1 Importers and adapters

| Canonical arch | Adapter | GGUF `general.architecture` | HF `model_type` |
|---|---|---|---|
| `decoder` | DecoderAdapter | llama (incl. Mixtral), qwen2, qwen3, gemma, gemma2, gemma3, gemma4, phi2, phi3, olmoe, qwen3moe, muse-glimmer, flint, smollm3, olmo, olmo2, glm4, chatglm, ernie4_5, hunyuan-dense, exaone, exaone4, seed_oss, internlm2, qwen, xverse, openelm, granite, granitemoe, minicpm, starcoder2, command-r, cohere2, falcon, gptneox, gpt2, nemotron, stablelm, apertus, arcee, qwen2moe, starcoder, refact, jais, jais2, orion, codeshell, plamo, plamo3, pangu-embedded, maincoder, spark2_5, baichuan, granite_swa, mellum, dbrx, deepseek, bailingmoe, cohere2moe, qwen3vl, qwen3vlmoe, qwen2vl (and the families listed in GgufArchitectureMappings) | llama, mistral, mixtral, qwen2, qwen3, gemma, gemma2, gemma3_text, gemma4(_text/_unified(_text)), phi, phi3, olmoe, qwen3_moe, muse_glimmer, flint, smollm3, olmo, olmo2, olmo3, glm4, ernie4_5, hunyuan_v1_dense, exaone, exaone4, seed_oss, internlm2, granite, granitemoe, minicpm, starcoder2, cohere, cohere2, falcon, gpt_neox, gptj, gpt2, nemotron, stablelm, apertus, arcee, qwen2_moe, gpt_bigcode, jais2, granite_swa, granitemoe_swa, granitemoeshared, mellum, dbrx, cohere2_moe, qwen3_vl, qwen3_vl_moe, qwen2_vl, qwen2_5_vl (and the families listed in HfArchitectureMappings) |
| `latent_decoder` | LatentDecoderAdapter | deepseek2, minicpm3, mistral4, plm, kimi-linear, bailingmoe3, deepseek32, glm-dsa, glm5-next, mistral-large4, dots3note | deepseek_v2, deepseek_v3, kimi_k2, glm4_moe_lite, minicpm3, mistral4, hy_v4, kimi_linear, bailing_hybrid, deepseek_v32, glm_moe_dsa, glm5_next |
| `encoder_decoder` | EncoderDecoderAdapter | t5, t5encoder | t5, mt5, umt5, bart, mbart |
| `encoder` | EncoderAdapter | bert, nomic-bert | bert, roberta, xlm-roberta, nomic_bert |
| `qwen3_5` | Qwen35Adapter (+ Qwen35Multimodal, VisionTowerGraph) | qwen35 (+ `clip` projector `qwen3vl_merger`) | qwen3_5, qwen3_5_text |
| `nemotron_h` | NemotronHAdapter | nemotron_h, nemotron_h_moe | nemotron_h |
| vision tower | VisionTowerGraph | clip (only `qwen3vl_merger`, with or without DeepStack layers) | embedded in Qwen3.5 and Qwen3-VL(-MoE) HF checkpoints |

Format-specific code lives only in Sharplight.Import.Gguf / Sharplight.Import.HuggingFace / Sharplight.Import.Onnx; adding a family is one `IGgufArchitectureMapping` / `IHfArchitectureMapping` class plus a `Register` call (GgufArchitectureMapping.cs, HfArchitectureMapping.cs). The ONNX importer is generic (ONNX zoo tests: detection, SAM, diffusion VAE/UNet, whisper decoder, CLIP/SigLIP, ...), so BERT-style encoders are reachable through ONNX today but not through GGUF/HF.

### 2.2 What DecoderAdapter expresses (documented in its XML remarks)

GQA/MQA; per-head QK RMS norm (weighted or unweighted), value norm; partial rotary, neox/gptj pairing, local rope base, yarn/llama3/longrope frequency factors, attention scale; sliding window with per-layer pattern; logit soft-capping and final soft-capping; attention sinks; sigmoid attention output gate; per-layer head dim / kv heads / KV sharing / NoPE layers; k = v layers; sandwich norms (`attention.out_norm`, `mlp.out_norm`); embedding scale, embedding norm, logit scale; gated or ungated MLP (silu/gelu/gelu_tanh/relu) with optional biases; MoE (softmax or sigmoid scores, renormalisation, weight scale, rank-3 expert weights through MatMulId, ungated shared expert, leading dense layers); Gemma 3n/4 per-layer embeddings. Waves 0 and 2 added: layer norm (with bias, or non-parametric), parallel residual, residual-branch scale, ReLU^2 / xIELU / SwiGLU-OAI activations, learned absolute positions, q/k norm after rope, per-layer head counts, q/k/v clamping, biased output head (`DecoderNorm`, `IMlpActivation` and the `Sequential`/`Parallel` layer topology in Sharplight.Architectures). Later rounds added routing bias and groups, ALiBi (CPU and Vulkan kernels: scalar, split-K decode and flash attention), bidirectional decoder layers (`attention.causal` = false: embedding graph only, symmetric sliding window, `embedding_projection.{i}` Dense layers after pooling), `moe.shared_scale` and query temperature on the latent decoder as well.

### 2.3 Graph ops available (Sharplight.Core StandardOps)

Attention, CachedAttention, CacheWrite, Rope, RmsNorm, LayerNorm, GroupNorm, L2Norm, Linear, MatMul, BatchMatMul, **MatMulId** (MoE), TopK, Softmax, Gather*/Scatter*, CumSum, Trilu, **CausalConv1d**, **GatedDeltaRule** (Qwen3.5 linear attention), **SelectiveScan** (Mamba-2 SSD), the ONNX-style elementwise/reduction/spatial set (Conv, Pool, Resize, GridSample, ...). **SelectiveScan** also covers Mamba-1 (A as [H, N]), **Wkv7**, **Wkv6** (RWKV recurrences); all of them have the multi-sequence form. CachedAttention takes `v_dims` (values = the first v_dims entries of each cache row, so the MLA latent cache serves as K and V). GatedDeltaRule also takes a decay per key channel ([T, H, Dk], KDA), time-invariant gates ([H]) and the additive rule (lightning attention). **IndexerTopK** (DSA lightning-indexer scoring and top-k over the indexer key cache) and CachedAttention's `selection` input (attention over given key positions) cover sparse attention. Not present: attention score bias, int hash ops.

### 2.4 Weight formats with compute kernels

CPU (Quant.cs and its partials Quant.KQuants/Iq/Fp4/Fp8/Ternary/Grouped.cs) and Vulkan (Sharplight.Shaders/Kernels/common.slang, quant_k.slang, quant_iq.slang, quant_fp8.slang, quant_group.slang; the lowering table is GraphLowerer.Support.cs): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4; Bonsai Q1_0, Q2_0, Q2_0_G128, Tern1_G128; FP8 `f8e4m3`, `f8e5m2` (one byte per value) and the block-scaled `f8e4m3_b32` { f32 d; u8 q[32] } / `f8e4m3_b128` { f32 d; u8 q[128] } (value = d * e4m3(q), blocks along K); the group-scaled formats of quantized Hugging Face checkpoints (below): `nvfp4_gs` { f32 g; u8 d[4]; u8 qs[32] } (an NVFP4 block with its global scale), `q4z_g32` / `q4z_g128` { f16 d; f16 z; u8 qs[G/2] } and `q8z_g32` / `q8z_g128` { f16 d; f16 z; u8 q[G] } (value = (q - z) * d, 4-bit runs laid out like Q4_0), `nf4_b64` { f32 d; u8 qs[32] } (value = d * NF4[q]), `fp4_b64` (the same layout with the bitsandbytes FP4 code book), `q8a_g32` { f32 d; i8 q[32] } (value = q * d / 127, the LLM.int8 row absmax form) and `q2s_g128` { f32 d; u8 qs[32] } (ternary codes, value = (code - 1) * d, the BitNet I2_S layout). The IQ lookup grids come from ggml-common.h via tools/gen-quant-grids.py (src/Sharplight.Cpu/Grids, Kernels/grid_*.slang). Q8_K fails at load.

Quantized Hugging Face checkpoints (`HfQuantizedWeights` in Sharplight.Import.HuggingFace: `HfFp4Weights`, `HfIntegerWeights` with `HfGptqLayouts` / `HfAwqLayouts` / `HfAwqGemvFastLayouts` / `HfMarlinLayouts` / `HfCompressedIntLayouts` / `HfQuarkLayouts` / `HfQuantoLayouts` / `HfTorchAoLayouts` / `HfHqqLayouts` / `HfEetqLayouts`, `HfQuarkFloatWeights`, `HfBitsAndBytesWeights`). As with FP8, each quantized weight is re-blocked row by row while it is read, before the architecture mapping; weights stay quantized (the BF16 fallbacks are listed). Every format is checked bit for bit against the library's own dequantization on the CPU and on Vulkan (tests/Sharplight.Import.HuggingFace.Tests/QuantizedImportTests.cs, checkpoints made by the libraries with tests/Sharplight.Families.Tests/tools/mk_tiny_quantized.py, references from hf_quant_reference.py) and end to end against transformers' logits (Families QuantizedCheckpointTests).
- NVFP4: ModelOpt (`quant_algo NVFP4`; U8 `X.weight`, E4M3 `X.weight_scale` per 16 values, f32 `X.weight_scale_2`) and compressed-tensors `nvfp4-pack-quantized` (`X.weight_packed`, `X.weight_global_scale`, a divisor). A global scale that is a power of two folds exactly into the E4M3 block scales (`nvfp4`); otherwise `nvfp4_gs` keeps it per 64-value block (5.0 bits per weight instead of 4.5). GGUF NVFP4 tensors with an output multiplier `X.scale` (one per tensor or per expert, which llama.cpp multiplies into the matmul output: LFM3 NVFP4, the Nemotron-3.5 Super-VL NVFP4 and Bolt files) become `nvfp4_gs` the same way, at import in Sharplight.Import.Gguf (`GgufNvfp4Scales`); their `X.input_scale` is ignored, as in llama.cpp. ModelOpt's dequantization (fp4 * (e4m3 * scale_2) in f32) is matched bit for bit; compressed-tensors' division by the global scale is matched after its rounding to BF16. Activations are not quantized (W4A4 checkpoints run as W4A16). K % 64 != 0 decodes to BF16.
- MXFP4: compressed-tensors `mxfp4-pack-quantized` (U8 E8M0 `X.weight_scale` per 32 values; also when the `quantization_config` sits in `text_config`, as in Kimi-K3 and K3.1-Pro), DeepSeek's own FP4 experts (DeepSeek-V4, `expert_dtype fp4`: I8 `X.weight` with an F8_E8M0 `X.scale` per 32 values, the FP8 weights next to them with an F8_E8M0 `X.scale` per 128 x 128 block) and AMD Quark (`quant_method quark`, weight spec `fp4` / group 32 / `e8m0`: U8 `X.weight` and `X.weight_scale`; Quark FP8 E4M3 weight specs take the FP8 path), dense or MoE, map to `mxfp4` exactly (OCP MX layout, first value in the low nibble; gpt-oss checkpoints as before). Quark's own library needs a JIT-built extension, so its case is checked against a transcription of its `Pack_fp4.unpack`, which agrees with compressed-tensors' MXFP4 dequantization on the same codes.
- GPTQ (auto-gptq / gptqmodel, `quant_method gptq`): `qweight` I32 [K·b/32, N] as a little-endian bit stream along K (2, 3, 4 and 8 bits; 3-bit codes straddle words), `qzeros` packed along N, f16 `scales` [G, N], `g_idx`. `checkpoint_format gptq` (or absent) stores zero - 1, `gptq_v2` the zero itself. Group sizes 32, 64, 128 and -1 (per channel). Act-order (`desc_act`, a non-trivial `g_idx`): the input columns are sorted so that every group is contiguous, the permutation is kept as `WeightInfo.InputOrder`, and `GraphBuilder.MatMul` gathers the activations into that order (one gather per input and permutation, shared by the projections reading it). MoE experts with different permutations cannot be stacked and fail at load.
- AWQ (AutoAWQ, `quant_method awq`): GEMM packing (`qweight` [K, N/8], interleaved nibble order 0 2 4 6 1 3 5 7) and GEMV packing (`qweight` [N, K/8], sequential; zero and scale widths padded to multiples of 8 groups), 4 bits, asymmetric.
- compressed-tensors integer weights: `pack-quantized` (`weight_packed` I32 along K, codes biased by 2^(b-1), optional `weight_zero_point` packed along N, `weight_g_idx`) and `int-quantized` (I8 `weight`, per-channel or group `weight_scale`); W4A16, W4A16_ASYM, W8A16 and W8A8 (activations not quantized) schemes. Sparse (non-dense `sparsity_config`) checkpoints are rejected.
- Integer targets: `q4z_*` for 2-4 bits, `q8z_*` for 5-8 bits; `*_g128` when the group size and K are multiples of 128, else `*_g32` (the scale repeated per 32-value block of a larger group). Exact because the scales are f16 (BF16 scales are checked to be representable in f16) and the zero points integers: (q - z) * d in f32 equals the libraries' f16 / BF16 results after their rounding. Group sizes that are not a multiple of 32, scales that do not fit f16 and act-order groups that do not sort into equal contiguous runs decode to BF16.
- bitsandbytes (serialized `load_in_4bit` / `load_in_8bit` checkpoints): NF4 with or without double-quantized absmax (U8 codes with `nested_absmax` per 256 blocks, `nested_quant_map`, `nested_offset`) maps to `nf4_b64` with the f32 absmax, bit-exact with `dequantize_4bit` (the code book is read from a device constant buffer like the IQ grids). FP4 (`bnb_4bit_quant_type fp4`) maps to `fp4_b64` (same layout, FP4 code book from a device constant buffer), bit-exact with `dequantize_4bit`. LLM.int8 (I8 `weight` and `SCB`, value = q * SCB / 127) maps to `q8a_g32` (the row absmax `SCB` repeated per 32-value block), bit-exact with `int8_vectorwise_dequant`; its outlier threshold splits activations at inference time into an int8 and an f16 product, and since activations are not quantized here every column takes the precise path. References: bitsandbytes (CPU build; `hf_quant_reference.py`).
- HQQ (`quant_method hqq`, hqq 0.2.8 `HQQLinear` state: `X.W_q`, `X.scale`, `X.zero`, `X.shape`, `X.nbits`, `X.group_size`, `X.axis`, `X.packing`; value = (code - zero) * scale with a float f16 zero point): 1 to 8 bits (`8bit_u8`, `4bit_u8`, `2bit_u8`, `1bit_u8`, `3bit_32` packings), kept quantized as `q4z_*` / `q8z_*` (the block type stores an f16 zero point, so non-integer zeros are exact; 1-4 bits `q4z_*`, 5-8 `q8z_*`). `axis 1` (groups along the row) stays quantized; `axis 0` assigns every column of a row its own group (the group of a value is its flat position modulo N K / group size), which no block type expresses, so it **decodes to BF16 at import**. Scales / zero points that are themselves quantized (`quant_scale` / `quant_zero`) are rejected. Checkpoints are written by hqq itself (transformers 5.23 cannot quantize with HQQ; `mk_tiny_quantized_formats.py`), bit-exact (f32 arithmetic on the stored scale and zero, hqq's `Quantizer.dequantize`; hqq's default f16 compute dtype rounds the same products to f16).
- EETQ (`quant_method eetq`, int8 weight-only: I8 `X.weight` [K, N] preprocessed for the Ampere mixed-type GEMM, F16 `X.weight_scales` [N]): the layout reader (`HfEetqLayouts`) inverts EETQ's weight preprocessing (16-row tile permutation, transpose, 64-column tile interleave, +128 bias, byte swap), verified against EETQ's own C++ preprocessing (`tools/eetq_harness`, built with build.sh against an EETQ checkout, since the package's CUDA kernels do not run on Blackwell) and kept quantized as `q8z_*` with the zero point 128, bit-exact with the w8a16 GEMM's `half(q) * half(scale)`. K and N must be multiples of 64.
- MLX quantization (mlx-lm checkpoints: `quantization {group_size, bits[, mode]}` without a method, per-module overrides): U32 `X.weight` (each row's codes as a little-endian bit stream), per-group `X.scales`. Affine mode (`X.biases`, value = scale * q + bias; embeddings included): 2, 3 and 4-bit weights with groups of a multiple of 32 and f16-exact scales and biases become `q4_1`, 5-bit `q5_1`, both bit-exactly (checked against `mlx.core.dequantize` of the CPU build of MLX, f32, as MLX's quantized matmul accumulates; a BF16 `mx.dequantize` rounds scale * q and the bias add separately and is up to a few BF16 ulps off); 6 and 8 bits (no block type with a per-block offset and 8-bit codes) decode to BF16. The FP modes read bit-exactly: `mxfp4` (E8M0 scale per 32) to `mxfp4`, `nvfp4` (E4M3 scale per 16, no second-level scale) to `nvfp4` (BF16 when K % 64 != 0), `mxfp8` to `f8e4m3_b32` with the power-of-two scale. Other modes are not read.
- Kernels: the f32 matvecs decode through load8 (quant_group.slang); the f16 tensor-core GEMMs (gemm_cm2, the skinny GEMM) take packed-half decoders for `q4z_*` and NVFP4 / `nvfp4_gs` (load16h in matmul.slang; the others go through load8); the integer-dot matvec (`mmvq.slang`, device feature `int_dot`, 8-bit activations) serves `q4z_*`, `q8z_*`, `nvfp4` and `nvfp4_gs` for 1-8 rows (`SHARPLIGHT_MMVQ_BLOCK=0` returns them to the f32 matvec). The GPU has no FP4 cooperative-matrix type (`--devices` lists f16, bf16, fp8 and int8 matrices on Blackwell), so FP4 weights are decoded to f16 for the tensor cores: there is no counterpart to vLLM's CUTLASS W4A4 FP4 GEMM.
- AutoRound (`quant_method auto-round`, `packing_format auto_round:auto_gptq` / `auto_round:auto_awq`; the `auto_gptq` / `auto_awq` exports declare plain GPTQ / AWQ): read through the GPTQ and AWQ GEMM layouts (2/3/4/8-bit GPTQ packing with zero points stored minus one, 4-bit AWQ), checked against AutoRound's own dequantizers (auto-round 0.16.1: `dequantize_gemm`, `qlinear_torch_zp`). ExLlamaV2 / Marlin packings and FP data types are not read.
- PyTorch checkpoints: a directory without `*.safetensors` but with `pytorch_model.bin` or `pytorch_model-XXXXX-of-YYYYY.bin` shards (and `pytorch_model.bin.index.json`) imports like a safetensors checkpoint. The files are read by `Sharplight.Import.PyTorch`, a restricted unpickler: it implements only the opcodes of tensor state dictionaries and resolves only `torch._utils._rebuild_tensor_v2` / `_rebuild_tensor` / `_rebuild_parameter`, `collections.OrderedDict`, `torch.Size`, the `torch.*Storage` classes and `_codecs.encode` (latin1), all as code inside the reader; every other global (`os.system`, `eval`, `torch.load`, ...) and opcode (`INST`, `OBJ`, `NEWOBJ_EX`, ...) is refused with an `UnsafePickleException` and nothing is executed. Both the zip format (stored entries, zip64) and the legacy stream format (up to 2 GB) are read, the storages memory-mapped in place; tensors must be contiguous, dtypes f64/f32/f16/bf16/i64/i32/i16/i8/u8/bool.
- AQLM (`quant_method aqlm`, additive codebook quantization; ISTA-DASLab / BlackSamorez checkpoints): **dequantized at import** to the codebook dtype (F16 or BF16), not kept codebook-quantized: the repository has no codebook-sum decode kernel yet, so a 2-bit checkpoint occupies the memory of a dense 16-bit model. `X.codes` [out groups, in groups, codebooks] (I8 / I16 / I32, masked to the codebook width), `X.codebooks` [codebooks, size, out group size, in group size] and `X.scales` [out groups, 1, 1, 1] become `X.weight` while the weight is read (rows decoded on demand, nothing cached). Bit-identical to aqlm 1.1.7 `_dequantize_weight` (CPU) for F16 codebooks with any number of codebooks, out group sizes above 1, code widths 6 to 16 and for single-codebook BF16; several BF16 codebooks may differ by one ulp on rounding ties (PyTorch's CPU `embedding_bag` rounds BF16 sums half away from zero, the importer to nearest even). Verified on BlackSamorez/TinyLlama-1.1B-Chat AQLM 2-bit 1x16 (greedy "The capital of France is Paris. The capital of Germany is Berlin...") and synthetic 1x8, 2x8, 4x6 (out group 2), 8x8 checkpoints (`tools/mk-tiny-aqlm.py`).
- AWQ `gemv_fast` (AutoAWQ `WQLinear_GEMVFast`, 4 bits): `qweight` I16 [N/4, K] with four output rows interleaved per word and the columns of every 32 permuted for fast dequantization, F16 `scales` and `qzeros` (= -scale * zero), value = scale * code + qzeros: kept as `q4_1` (affine blocks, the f16 pair repeated per 32-value block), bit-exact with the one f16 fma of the kernel. No Python dequantizer exists (CUDA kernel only), so the reference inverts AutoAWQ 0.2.9's `pack_intweight` step by step (checked against the packer on random codes).
- Marlin repacks (GPTQ `checkpoint_format marlin`, AutoAWQ `version marlin` and the original IST-DASLab `X.B` / `X.s` names; symmetric 4 and 8 bits, no act-order): `qweight` in 16 x 16 tiles shuffled by Marlin's tensor-core permutation, scales permuted per 64 (per 32 for one group per column); the importer undoes the repack and keeps `q4z_*` / `q8z_*` with the zero point 2^(bits-1), bit-exact with vLLM 0.33's Marlin GEMM applied to an identity matrix (CUDA; `hf_quant_reference_marlin.py`) for GPTQ-Marlin (`gptq_marlin_repack`) and AutoAWQ's Marlin export. Act-order Marlin checkpoints are rejected.
- AMD Quark integer weights (`int4` / `uint4` / `int8` / `uint8`; per-tensor, per-channel and per-group scheme; pack methods `reorder` and `order`): per-group weights are stored transposed [K, N/8] (AWQ GEMM nibble order when `reorder`), scales [K/group, N], zero points packed like the weights. Signed codes are re-based (+8 / +128) into `q4z_*` / `q8z_*`. Quark writes f32 scales, which are rarely f16 values: those weights **decode to BF16 at import** (the block types hold f16 scales); scales that are f16 values (as in the tiny test checkpoints) stay quantized. Float zero points (`zero_point_type float32`) are rejected. Reference: `quark.torch.utils.pack` unpackers (amd-quark 0.13), (q - zero point) * scale in f32.
- AMD Quark MXFP6 (`fp6_e3m2` / `fp6_e2m3` with an E8M0 `weight_scale` per 32, four 6-bit codes in three bytes) and the in-stream MX variant (`dtype mx`: every 32-value block is a signed exponent byte followed by its codes, MXFP4 or MXFP6): MXFP4 maps to `mxfp4` exactly; **MXFP6 has no block type of its own and is stored as `f8e4m3_b32`** (every FP6 value is an E4M3 value, so the values are exact, in 8 bits instead of 6). Reference: Quark's `Pack_fp6` / `Pack_mxfp6` unpackers.
- optimum-quanto (`quant_method quanto`; `QuantizedModelForCausalLM` state dicts `X.weight._data`, `._scale`, `._shift`; optimum-quanto 0.2.7): `qint8` keeps its codes as `q8z_*` (zero point 128; f16 scales), `qint4` / `qint2` (codes packed along the first dimension of the grouped [R, group size] matrix, scale and a float shift per group) as `q4_1` (value = scale * code - shift, bias = -shift), `qfloat8` (E4M3 + scale) through the FP8 path (`f8e4m3_b128`). Packed weights lose their [N, K] shape in the state dict, so it is recovered from N * K and the config's `hidden_size` (output-side projections `o_proj`, `down_proj`, ... are [H, N K / H], the rest [N K / H, H]); a layer that fits neither is rejected. `qfloat8` with E5M2 data is rejected. Activation scales are dropped. Bit-exact against the library's `dequantize()` with scale and shift cast to f32 (quanto computes in the module dtype, f16, which rounds scale * code and the shift subtraction separately).
- torchao (`quant_method torchao`, torchao 0.18 safetensors from transformers: `X._weight_qdata`, `_weight_scale`, `_weight_zero_point`): `Int8WeightOnlyConfig` / `Int8DynamicActivationInt8WeightConfig` (per row or per group) and `Float8WeightOnlyConfig` / `Float8DynamicActivationFloat8WeightConfig`. Int8 weights go through `q8z_*` when the scales are f16 values (those of BF16 models stored as f32 are: the tiny test checkpoints stay quantized) and **decode to BF16 at import** when not. Float8 weights take the FP8 path (`f8e4m3_b128`, exact). Not read: `Int4WeightOnlyConfig` and the intx / NF4 tensors (the test machine cannot build int4 checkpoints: torchao needs mslk), and the string `quant_type` names of old torchao releases (their layouts are `AffineQuantizedTensor` pickles). Reference: the tensors torchao holds, (qdata - zero point) * scale in f32.
- FP-Quant (`quant_method fp_quant`, ISTA-DASLab MXFP4 / NVFP4 with a Hadamard rotation per group): **dequantized at import** to dense BF16 with the rotation folded into the weight (`HfFpQuantWeights`). `X.qweight` U8 (E2M1 pairs, low nibble first), `X.scales` (row-major E8M0 per 32 for MXFP4, E4M3 per 16 for NVFP4; QuTLASS blocks them only inside its GEMM) and `X.weight_global_scale` give `dq = e2m1 * scale / weight_global_scale` in the rotated domain; the library rotates activations by `forward_hadamard_matrix` H (`x_rot = x @ H` per group, the orthonormal Sylvester matrix), so `W[o,g,i] = sum_j H[i,j] dq[o,g,j]`. Activations are neither rotated nor FP4-quantized here (the library quantizes them too, which costs it more accuracy than the weights). Checked against the real library (transformers 5.23 + fp_quant 0.3.2 + QuTLASS 0.2.0 built for sm_120, `tools/dequantized-parity.py`): random exact-FP4 activations through QuTLASS's FP4 x FP4 GEMM with the checkpoint's codes and scales agree with the un-rotated activations times the importer's BF16 weight to 0.22 % (NVFP4) / 0.23 % (MXFP4) relative error on every projection of layers 0 and 27, which is the BF16 rounding of the weight. On the fixed parity text (176 tokens) Qwen3-1.7B scores perplexity 26.64 in BF16, the RTN checkpoints 68.1 (NVFP4) / 67.6 (MXFP4) through the library and 57.6 / 44.5 with the dense weights and unquantized activations in PyTorch; Sharplight on Vulkan scores 57.11 / 44.33 (within 1 %), and its greedy continuations match PyTorch's with the same weights. The RTN MXFP4 checkpoint answers "The capital of France is the capital of France ..." in the library too: that is the checkpoint, not the import. The importer matches `tools/fpquant-reference-dense.py` to one BF16 ulp on layers 0 and 27 of both. Not kept FP4 because the per-group rotation needs an activation-side op that the canonical graph does not have.
- HIGGS (`quant_method higgs`, IST-DASLab; Hadamard rotation of `hadamard_size`, then vector quantization of input pairs on a 256-point Gaussian-optimal grid, stored in FLUTE's packed layout): **dequantized at import** to the scales' dtype (`HfHiggsWeights`). `X.weight` I16 [N / 4, K] holds FLUTE's tiles (the tile width `tileP` of `tune_metadata[module].template_id`: 64 when id % 48 < 16, else 32), `X.tables2` the grid, `X.scales` one scale per 256 inputs; the dense weight is the orthonormal Sylvester Hadamard matrix applied to each block of grid value * scale (FLUTE rotates activations with it before its GEMM). Only 4-bit, p = 2 checkpoints (the transformers default) are read; 2 / 3-bit and scalar grids are rejected. FLUTE has no CPU dequantizer and is not built for Blackwell here, so `tools/higgs-reference-dense.py` transcribes FLUTE's packer and inverts it (checked on random codes for both tile widths); on galqiwi/Qwen3-1.7B-HIGGS-4bit the decoded weights reconstruct the original Qwen3-1.7B with 8.8 % relative error (the expected 4-bit p = 2 grid error; a wrong layout gives 140 %, a missing 1/sqrt(n) 2100 %), the importer matches the script to one f16 ulp, and the perplexity of the fixed parity text on Vulkan is within 3 % of PyTorch with the same dense weights (38.15 vs 26.64 for the original BF16 model; greedy "The capital of France is Paris. The capital of Italy is Rome...").
- VPTQ (`quant_method vptq`, Microsoft; per-module settings in `config_for_layers`): **dequantized at import** to the centroids' dtype (`HfVptqWeights`). Vector quantization along the output dimension with a main codebook, optional residual codebook, outlier columns with their own codebook, a column permutation and per-column `weight_scale` / `weight_bias`; packed (`is_indice_packed`: little-endian bit stream of index | residual index << index bits per [codebook, vector row]) or 16-bit indices. Every operation rounds to the module dtype like VPTQ's PyTorch `dequant`, so the result is bit-identical to it (`tools/vptq-reference-dense.py`, a transcription of `vptq.ops.quant_gemm.dequant` since the package's CUDA build is not available here), checked on Tiberiw/TinyLlama-1.1B VPTQ v8-k65536-256 (residual) and v8-k4096-0 (no residual), layers 0 and 21. End to end on Vulkan the v8-k65536-256 model is within 3 % of PyTorch's perplexity with the same weights (19.32 vs 16.53 for the original TinyLlama-1.1B-Chat; greedy "The capital of France is Paris. The capital of Germany is Berlin..."); the 1.5-bit v8-k4096-0 checkpoint is unusable in PyTorch too (perplexity 1062).
- ExLlamaV2 / EXL2 (no `quantization_config`; recognized by `X.q_weight`, `X.q_groups`, `X.q_scale`, `X.q_scale_max`, `X.q_invperm`): **dequantized at import** to F16 (`HfExl2Weights`). K rows sorted into groups of 2, 3, 4, 5, 6 or 8 bits (`q_groups`: bits and first packed row), per column one little-endian bit stream of codes per group, 4-bit group scales `q` (eight per word) with scale = f16((q + 1)^2) * f16(`q_scale_max` / 256), value = f16((code - 2^(bits-1)) * scale), dense input j = stored row `q_invperm[j]`; this is exllamav2's `reconstruct`. Not kept quantized: exllamav2 permutes activations instead of weights, and the bit width varies along K. Bit-identical to exllamav2 0.3.2's `reconstruct` (built from source for sm_120; `tools/exl2-reference-dense.py`, which also checks an independent PyTorch decode against it and undoes exllamav2's load-time reordering of gate / up rows by down_proj's permutation) on turboderp/TinyLlama-1B-exl2 4.0 and 2.5 bpw (2 to 8-bit groups), layers 0 and 21 and lm_head; the 4.0 bpw model on Vulkan is within 3 % of PyTorch's perplexity with the same weights (18.66). Output features are taken from q_weight, so EXL2 tensors whose N was padded to a multiple of 32 are not trimmed.
- Not supported: EXL3, SpQR checkpoints, HIGGS other than 4-bit p = 2, torchao int4 / intx, ExLlama-format AWQ and act-order Marlin (HfSupport reports them before download).

FP8 Hugging Face checkpoints (the vLLM formats; `HfFp8Weights` in Sharplight.Import.HuggingFace): an `F8_E4M3` `X.weight` with a separate scale is converted while it is read (nothing cached) into a block-scaled type, before the architecture mapping runs, so fused-projection splits, V-head row reorders and expert stacking work on whole rows unchanged (transposes and column permutations that cut blocks decode to BF16 first).
- DeepSeek-V3 / Qwen FP8 (`quantization_config.quant_method = "fp8"`, `weight_block_size [128, 128]`): BF16/F32 `X.weight_scale_inv` [ceil(N/128), ceil(K/128)], a multiplier despite its name.
- ModelOpt (`hf_quant_config.json`, e.g. Nemotron-3-Nano-30B-A3B-FP8) and compressed-tensors: `X.weight_scale` per tensor, per channel or per block (also a multiplier). `X.input_scale` is ignored (activations are not statically quantized), as are KV-cache scales (`k_scale` / `v_scale`) and `kv_cache_quant_algo`: caches use the `--kv-type` setting.
- Target: `f8e4m3_b128` when K and the scale's K granularity are multiples of 128, else `f8e4m3_b32` (the scale repeated in every block it covers, e.g. Nemotron's expert down projections with K = 1856), else BF16. Modules a checkpoint does not quantize are BF16 in the file and pass through. Scale-less `F8_E4M3` / `F8_E5M2` tensors load as `f8e4m3` / `f8e5m2`.
- Kernels: the f32 matvecs decode through half bits (load8_fp8), the f16 tensor-core GEMMs (gemm_cm2, the MoE grouped GEMM) through packed halves (load16h_fp8 / load8h_fp8). Batches of 2-16 rows run on the skinny f16 tensor-core GEMM (`matmul_skinny.slang`, weights read once for all rows, for matrices with at least 1024 output rows; `SHARPLIGHT_SKINNY_GEMM=0` returns them to the multi-row f32 matvec), batches of 17+ rows (`SHARPLIGHT_FP8_GEMM_MIN_ROWS`) of the block-scaled `f8e4m3_b128` / `f8e4m3_b32` matmuls (K % 128 == 0) on the fp8 tensor-core GEMM (`gemm_fp8.slang`, device feature `matrix_fp8`: activations quantized to E4M3 with one f32 scale per 128 values, 16x16x32 f8E4m3 cooperative matrices with f32 accumulation, one scaling per weight block); `SHARPLIGHT_FP8_GEMM=0` or `--disable-features matrix_fp8` return them to the f16 GEMM. This covers every scaled HF FP8 weight (per-tensor and per-channel scales become block scales on import); unscaled `f8e4m3` / `f8e5m2` tensors and the MoE expert GEMMs stay on the f16 GEMMs. Tested: tests/Sharplight.Cpu.Tests/Fp8DequantTests.cs, Vulkan QuantFormatTests, tests/Sharplight.Import.HuggingFace.Tests/Fp8ImportTests.cs, and Families Fp8CheckpointTests (Qwen3.8-27B-FP8 vs Q8_0, Nemotron-3-Nano-30B-A3B-FP8 vs Q4_K_M; `SHARPLIGHT_TEST_LARGE=1`).

### 2.5 Models that are actually tested

| Test project | Models |
|---|---|
| Sharplight.Families.Tests (HF ids + GGUF text vs transformers/llama.cpp) | gemma-4-E2B-it (HF, BF16, Q8_0), NVIDIA-Nemotron-3-Nano-4B (HF, Q8_0), tiny-random-nemotron-h, NVIDIA-Nemotron-3.5-Bolt-E4B (UD-Q4_K_XL), tiny-random-muse-glimmer, Muse-Glimmer-30B (UD-Q4_K_XL), tiny-random-flint |
| Sharplight.Import.HuggingFace.Tests (HF vs GGUF canonical parity, DecoderParityTests) | Llama-3.2-1B-Instruct, Qwen2.5-0.5B-Instruct (Q8/f16), Qwen3-0.6B, SmolLM2-360M-Instruct, Phi-3.5-mini-instruct, OLMoE-1B-7B-0924-Instruct, gemma-2-2b-it, gemma-3-1b-it, gemma-4-E2B-it, gemma-4-12B-it, Nemotron-3-Nano-4B, Muse-Glimmer-30B, Bonsai-1.7B, Ternary-Bonsai-1.7B/2-2B |
| Qwen35ParityTests, Sharplight.Vision.Tests | Qwen3.5-0.8B (HF + BF16 GGUF + mmproj), Qwen3.8-27B mmproj (BF16) vs llama.cpp |
| Sharplight.Import.Gguf.Tests | metadata/tensor parsing incl. Qwen3.8-27B Q8_0 / Q4_K_M / NVFP4 files, synthetic GGUFs |
| Sharplight.Vulkan.Tests / Cpu.Tests | op-level: SSM (Nemotron-H shapes), MoE, LLM ops, IQ4, Bonsai quant, fusion, submission budget |
| Sharplight.Import.Onnx.Tests | ONNX model zoo |

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

Next in line: the Gemma 3n MobileNetV5 vision tower and audio encoder.

### 3.2 Block catalog and score table

| Code | Block | Archs needing it | Score | Effort | Sharplight today |
|---|---|---|---|---|---|
| `LN` | **LayerNorm (mean-centred, bias) in the generic decoder adapter**. Falcon, GPT-NeoX/J, Phi-2, StarCoder(2), OLMo-1, Command-R/Cohere (no bias), Nemotron-4 (1+w), Bloom, MPT, BERT. | 26 | 34 | S | LayerNorm op exists (used by VisionTowerGraph); decoder adapter has RMS only |
| `RBIAS` | **Aux-loss-free routing: selection bias (e_score_correction_bias), sigmoid scores, optional renorm + scale**. top-k chosen on score+bias, weights taken from the unbiased score. Nemotron-H adapter has it; the generic decoder adapter only has softmax/sigmoid without bias. | 18 | 31 | S | **done** in the shared `MoeOptions`/`MoeRouter`/`MoeBlock` (decoder, hybrid, NemotronH, Qwen3.5 and latent_decoder adapters): `moe.router_activation` softmax / sigmoid / softmax_topk / sqrt_softplus, `mlp.router_bias` selection bias, `moe.normalize_weights`, `moe.weights_scale`, `moe.weights_before_experts` |
| `MLA` | **Multi-head latent attention**. q_lora / kv_lora compression, decoupled rope dims, absorbed form with a latent KV cache (llama.cpp caches kv_lora_rank + rope_dim per token) and, for prefill, the expanded MHA form; yarn mscale on the softmax scale. | 15 | 24 | L | **done** (LatentAttention block in `latent_decoder`: latent cache [kv_lora normed latent, roped k_pe], absorbed queries via per-head MatMulId, CachedAttention `v_dims`); prefill also uses the absorbed form (expanded prefill and an MQA-shaped decode kernel still to do) |
| `RGRP` | **Group-limited top-k routing (n_group / topk_group)**. experts split into groups, best groups chosen by (top-2 sum of) scores, then top-k inside them (DeepSeek-V3, Ling, Kimi-K2, GLM-4.5 n_group=1 degenerates). | 11 | 22 | S | **done** in `MoeRouter` (`moe.group_count`, `moe.group_used_count`, `moe.group_score` top2_sum / max); latent_decoder uses it too |
| `IDX` | **Sparse-attention indexer (DSA / QSA / lightning indexer)**. a small per-layer scoring attention (few heads, head_dim 128, q/k norm) picks top-k visible tokens (budget 2048; QSA pools keys in blocks of compress_ratio) and the main attention only attends to them. | 11 | 20 | L | **done** for DSA: `IndexerTopK` + CachedAttention `selection` (deepseek32, glm-dsa, hy_v4), group selection (per-head block selection, minimax-m3; softmax-gated pooled keys with tail, glm5-next); missing: QSA block selection over pooled keys (qwen4exp) |
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
| `KDA` | **Kimi Delta Attention (per-channel-gated delta rule) / lightning variants**. gated delta rule with a per-key-channel decay instead of a per-head scalar (Kimi-Linear, Ling-3/Ring-linear, GLM5-next, Kimi K3). | 5 | 8 | M | **done**: GatedDeltaRule with a [T, H, Dk] decay (CPU, token-serial Vulkan kernel; the chunked tensor-core form stays scalar-gate only), `ChannelDeltaAttention` |
| `TMP` | **Attention temperature tuning and chunked-local attention**. Llama 4 NoPE layers scale q by log(floor((pos+1)/floor_scale)+1)*attn_scale+1; iRoPE chunked attention mask; Mistral-3/4 use the same scale. | 3 | 6 | S | **done**: per-layer query temperature (`attention.temperature_scale/length/offset/layers`) and chunked attention (`attention.chunk_size/chunk_layers`, CachedAttention `chunk` on CPU and Vulkan) |
| `ALIBI` | **ALiBi / additive score bias in cached attention**. Bloom, MPT, Baichuan-13B, Refact, Jais, Jina-BERT-v2; same plumbing serves T5 relative-position buckets. | 5 | 5 | M | **done** as the provider-neutral attribute `alibi_slopes` (float[Hq]) on CachedAttention / Attention (`attention.alibi_max_bias` in the decoder); the CPU provider and the Vulkan attention kernels (scalar, split-K decode, flash; slopes baked in as a specialization) implement it, for causal and bidirectional (symmetric |i - j|) attention |
| `CONV` | **Gated short-conv layer (LFM2/LFM3)**. in_proj -> (B, C, x); y = C * conv_k3(B * x); out_proj; interleaved with GQA layers and (lfm2moe) MoE. | 3 | 5 | S | CausalConv1d exists; needs adapter + state plumbing |
| `YARNM` | **YaRN attention factor on MLA softmax scale**. mscale^2 folded into the MLA scale (rope.attention_scale exists for non-MLA). | 2 | 5 | S | **done** with MLA (`attention.scale` of latent_decoder) |
| `DSTACK` | **DeepStack vision injection**. Qwen3-VL adds vision-tower intermediate features to the hidden state of the first N decoder layers. ClipGgufMapping throws NotSupported for is_deepstack_layers today. | 2 | 5 | M | **done**: `vision.deepstack_layers` taps in VisionTowerGraph (merged, normed, projected, appended to the embedding row), added after decoder layer k in the multimodal graph |
| `NGRAM` | **PLE n-gram hash embeddings**. Qwen4Exp has one such layer (HF `ple_layer_ids` [2], GGUF `ple.layers` [1]): the last 3 token ids are hashed per head (16 heads = (ngram_size-1) x heads_per_ngram 8, each with its own multiplier, row offset and ~20M-row slice) into one flat [160, ~320M] table (38 GB in Q5_1), then a short causal conv (k=4) over the 4 hyper-connection streams and key/value projections add the result into the stream. Gather + CausalConv1d exist; the 64-bit hash must run host-side (row ids as an int input) or as an int64 op. | 2 | 5 | M | missing: int hash/row-id computation + conv-history state per sequence; the table must be memory-mapped, not materialised |
| `WKV7` | **RWKV-7 time-mix (generalised delta rule)**. token-shift lerps, LoRA-parametrised decay/in-context-lr/gate, WKV7 recurrence, LN, channel-mix ReLU^2. | 2 | 5 | L | missing: Wkv7 op + RWKV adapter |
| `DIFF` | **Masked-diffusion decoding loop**. bidirectional transformer + iterative unmasking sampler (Dream, LLaDA, RND1). | 4 | 4 | L | done: `denoise` graph (bidirectional, logits for every row), `DiffusionGenerator` (timestep / block schedules; confidence, margin, entropy unmasking), `DiffusionGenerationBackend` in Sharplight.Extensions.AI |
| `APOS` | **Learned absolute position embeddings**. GPT-2, StarCoder, BERT family: token + position (+ token-type) embedding sum. | 3 | 4 | S | Gather + Add exist; adapter lacks it |
| `M1` | **Mamba-1 selective scan**. per-channel A [d_inner, d_state], dt_rank low-rank dt projection, x_proj; conv k=4; Mamba, Falcon-Mamba, Jamba, Codestral-Mamba. | 3 | 4 | M | missing: SelectiveScan1 op (CPU + Slang) and its chunked prefill path |
| `SPECDEC` | **Speculative-decoding drafters (EAGLE3, DFlash, MTP assistants)**. draft graphs that consume target hidden states. | 3 | 3 | L | partial: MTP for the qwen3_5 family and NextN heads of DeepSeek-V3 (`latent_decoder`) / GLM-4.5 (`decoder`, uniform layers) from HF and GGUF (`MtpDraftHead` + `MtpSpeculator`, rollback of recurrent state, shared embedding / LM head, device-side verification); verified on tiny random checkpoints only (no real DeepSeek / GLM checkpoint was run); EAGLE3 drafters as the `draft` auxiliary source (`Eagle3DraftHead`, target layer taps, draft vocabulary map), tiny Llama target and drafter, HF and GGUF; DFlash drafters missing |
| `RELPOS` | **Relative-position bias buckets**. T5 style learned bucketed bias added to scores. | 2 | 3 | M | **done**: `position_bias` / `position_bias_range` attention attributes on CPU and Vulkan (`RelativePositionBias`) |
| `WKV6` | **RWKV-6 time-mix**. data-dependent decay with LoRA, WKV6 recurrence, token shift, group norm. | 2 | 3 | M | missing: Wkv6 op (WKV7 kernel can share structure) |
| `LOOP` | **Weight-shared layer loops**. Nanbeige (num_loops), HRM-Text (H/L cycles): the same layer repeated with distinct KV slots. | 2 | 3 | S | graph feature: reuse weights, separate cache keys |
| `QKVZ` | **qwen3next fused projection layouts**. qwen3next stores in_proj_qkvz / in_proj_ba with head-interleaved layout (llama.cpp ssm_in + ssm_beta_alpha, and a per-layer split rule in the loader); Qwen3.5 uses separate qkv/z/b/a. | 1 | 3 | S | mapping + lazy row permutation in Gguf/HF importers |
| `MXFP4` | **MXFP4 / NVFP4 weight formats**. gpt-oss ships MXFP4 experts; many 2026 checkpoints ship NVFP4. CPU and Slang decoders, matvec, GEMM and MatMulId exist and are tested against ggml (quantisation track Q); gpt-oss still needs expert bias and SwiGLU-OAI from the decoder track. | 1 | 3 | M | **done** (kernels; gpt-oss GGUF and HF MXFP4 `*_blocks`/`*_scales` experts import) |
| `EBIAS` | **Per-expert biases in MatMulId**. gpt-oss experts have gate/up/down biases. | 1 | 3 | S | **done** (`experts.{gate,up,gate_up,down}.bias` gathered per selected expert) |
| `LIN` | **Lightning / linear attention with fixed decay**. MiniMax-01/M3 lightning attention: block-wise kv state with exponential decay, softmax layer every 8th. | 2 | 2 | M | **done**: GatedDeltaRule `rule=additive` with [H] gates, `LightningAttention` (minimax-01; minimax-m3 still needs `IDX`) |
| `MOELATENT` | **MoE in a latent space (down/up projection around the experts)**. Nemotron-H MoE latent (supported), Muse-Spark, Maion-Coder. | 2 | 2 | S | exists in NemotronHAdapter |
| `ALTUP` | **Gemma 3n AltUp (4 parallel streams with predict/correct)**. plus LAuReL low-rank residual and activation sparsity; PLE and KV sharing already exist. | 1 | 2 | M | **done** (`AltUp` block over streams [T, S·H], graph code over standard ops; activation sparsity in the dense MLP): gemma3n |
| `LAUREL` | **Gemma 3n LAuReL**. low-rank learned residual. | 1 | 2 | S | **done** (sequential layer, `laurel.*` weights): gemma3n |
| `XATTN` | **Cross-attention encoder-decoder (T5)**. encoder runs once, decoder cross-attends to cached encoder K/V; relative-position bias buckets; no RoPE. | 1 | 2 | L | **done**: `encoder_decoder` adapter (T5, BART, mBART), encoder pass in InferenceSession, Vulkan cross-attention kernels |
| `ADAPTM2` | **Pure Mamba-2 model adapter**. mamba2-130m...: all layers SSM; a degenerate case of NemotronHAdapter once its mapping accepts `mamba2`. | 1 | 2 | S | adapter exists in effect; mapping + hparam names missing |
| `TQ` | **Ternary TQ1_0/TQ2_0/I2_S formats**. BitNet b1.58 official GGUFs. | 1 | 2 | M | **done** for TQ2_0 (bitnet-b1.58-2B-4T runs on CPU and Vulkan) and I2_S (`q2s_g128`, bit-exact with the reference decoder `tools/gguf_i2s_reference.py`) |
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
| `qwen2vl` | Qwen2-VL / Qwen2.5-VL | GQA, M-RoPE, qkv bias; 2-D rotary ViT towers (Qwen2.5: windowed attention, RMS norms, gated SiLU MLP) | **supported** (text and images: sectioned non-interleaved M-RoPE in the decoder; towers from HF `visual.*` and from mmproj `qwen2vl_merger` / `qwen2.5vl_merger` (separate q/k/v, legacy files with swapped MLP names, `n_wa_pattern` windows). Tiny models vs transformers (embeddings, prompt logits, greedy ids) and their mmproj towers vs llama.cpp b11177 (`llama-mtmd-debug`, 1e-4); HF Qwen2-VL-2B (32 greedy ids) and Qwen2.5-VL-3B (prompt logits, 16 greedy ids; at the 17th, transformers' cached generate and its own full forward disagree) with an image vs transformers on CPU and Vulkan; GGUF Q4_K_M + f16 mmproj of both describe the test image. Deviation: the real f16 mmproj towers sit ~0.05 from llama.cpp, which rounds their large activations to f16 and moves by ~0.01 between its own flash-attention settings) | - | qwen2_vl, qwen2_5_vl | hf/tiny-random-qwen2-vl-wide, hf/tiny-random-qwen2-5-vl-wide, gguf/tiny-random-qwen2-vl-wide-f32, gguf/tiny-random-qwen2-5-vl-wide-f32 (+ mmproj), hf/Qwen2-VL-2B-Instruct, hf/Qwen2.5-VL-3B-Instruct, gguf/Qwen2-VL-2B-Instruct (+mmproj), gguf/Qwen2.5-VL-3B-Instruct (+mmproj) |
| `qwen3` | Qwen3 dense, Qwen3-Embedding/Reranker | GQA, QK-norm, RMS, SwiGLU | **supported** | - | qwen3, llava_onevision2 (Mage-VL text decoder; the OneVision tower is not imported) | - |
| `qwen3moe` | Qwen3-30B-A3B, 235B-A22B | QK-norm, MoE softmax norm-topk (no shared) | **supported** | - | qwen3_moe | - |
| `qwen3vl` | Qwen3-VL dense | qwen3 + M-RoPE (interleaved) + DeepStack injection | **supported** (text and images: DeepStack tower vs llama.cpp b11177 (`llama-mtmd-debug`, worst corner 2.7e-3), GGUF + mmproj describes the test image on CPU and Vulkan, HF Qwen3-VL-2B-Instruct with an image vs transformers: prompt logits within 0.007 and 32 greedy ids, CPU and Vulkan. Q4_K_M continuation differs from llama-mtmd-cli at step 24 where the top-2 margin is 0.28) | - | qwen3_vl | gguf/Qwen3-VL-2B-Instruct (+mmproj), hf/Qwen3-VL-2B-Instruct |
| `qwen3vlmoe` | Qwen3-VL-MoE | qwen3moe + M-RoPE + DeepStack | **supported** (HF: fused experts in either order, router, tower with DeepStack through the shared Qwen3-VL path; tiny model vs transformers greedy ids on CPU and Vulkan; GGUF through the qwen3vl mapping with `expert_count`; no real checkpoint under 4 GB) | - | qwen3_vl_moe | hf/tiny-random-qwen3-vl-moe-wide |
| `sensenova_u1` | SenseNova-U1 | qwen3 + temporal-only M-RoPE (image gen) | **supported** (text; image generation and image input out of scope) | - | neo_chat | hf/tiny-random-sensenova-u1 (numpy reference of the llama.cpp graph), gguf/tiny-random-sensenova-u1-f32 |

### Qwen hybrid

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `qwen3next` | Qwen3-Next-80B-A3B, Qwen3-Coder-Next | GDN (3/4 layers) + gated attention (1/4, partial rope 0.25, QK-norm), MoE softmax norm-topk + sigmoid-gated shared expert, conv k=4, MTP | **supported** (MTP not used) | - | qwen3_next | gguf/Qwen3-Coder-Next, hf/Qwen3-Coder-Next (config+tokenizer), hf/tiny-random-qwen3-next-moe |
| `qwen35` | Qwen3.5/3.6/3.8 dense (0.8B...27B) | GDN + gated attention, M-RoPE partial, dense SwiGLU, MTP, vision tower | **supported** | - | qwen3_5 (text) | gguf/Qwen3.5-0.8B (existing), hf/Qwen3.5-0.8B (existing) |
| `qwen35moe` | Qwen3.5-35B-A3B/122B/397B, Qwen3.6-35B-A3B | qwen35 + MoE softmax (256 experts, top-8) + sigmoid-gated shared expert | **supported** | - | qwen3_5_moe | gguf/Qwen3.6-35B-A3B (+mmproj), hf/Qwen3.6-35B-A3B (config+tokenizer), hf/tiny-random-qwen3.5-moe, hf/tiny-random-qwen3.6-moe, hf/tiny-random-qwen3.8-moe |
| `qwen4exp` | Qwen3.8-Flash-Next (hf: qwen4_exp, ~200B total params) | GDN + gated attention, MoE 512 experts top-10 + shared, hyper-connections (4 streams, low-rank), QSA sparse-attention indexer (top-k 2048, pooled keys), PLE n-gram hash embeddings (layer 2), M-RoPE, MTP | **supported** (HC, NGRAM, QSA block selection in the sparse regime; MTP not used) | - | qwen4_exp | gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL (+mmproj), hf/Qwen3.8-Flash-Next (config+tokenizer), hf/tiny-random-qwen3.8-flash-next, hf/tiny-random-qwen3.8-flash-next-sparse (+ f32 GGUF) |
| `clover_moe` | clover-moe (qwen3-next-like) | GDN + attention + MoE shared | **supported** (GGUF through the qwen3_5 adapter: the tiny Qwen3.5-MoE checkpoint retargeted to clover_moe vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan, b11177 predates it; per-layer `attention.recurrent_layers` must be periodic; HF `clover_moe` (round 11): the remote code's causal commit pass, the block-diffusion preview tensors left out, tiny checkpoint vs the remote code and its llama.cpp GGUF vs llama.cpp 8ca5a80; not run on a real model) | - | clover_moe | gguf/tiny-random-clover-moe-f32, hf/tiny-random-clover-moe |

### Gemma

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `gemma` | Gemma 1 | MQA/MHA, RMS(1+w), GeGLU, escale | **supported**| -| gemma, paligemma (text decoder, partial: decoded causally, the bidirectional image + prompt prefix is not modelled) | - |
| `gemma2` | Gemma 2 | GQA, SWA alt, attn+final softcap, sandwich norms, GeGLU | **supported** | - | gemma2, paligemma (PaliGemma 2 text decoder, partial as for gemma) | - |
| `gemma3` | Gemma 3 (+ SigLIP mmproj) | GQA, QK-norm, SWA 5:1 local rope base, sandwich norms | **supported** | - | gemma3_text | hf/tiny-random-gemma-3 |
| `gemma3n` | Gemma 3n E2B/E4B | AltUp, LAuReL, PLE, KV sharing, sparse FFN activation | **supported** (language model on the Gemma 4 path plus the shared `AltUp` block (4 streams, predict / correct, expand / collapse), LAuReL and Gaussian activation sparsity (relu(g − mean − std·Φ⁻¹(s))); HF gemma3n / gemma3n_text and GGUF. Tiny float32 model vs transformers (prompt logits, 24 greedy ids) and its converted GGUF vs the same ids; gemma-3n-E2B-it Q4_K_M vs llama.cpp b11177 greedy ids; CPU and Vulkan. llama.cpp fixes the per-layer input width at 256 and the LAuReL rank at 64, so it cannot load small tiny models. The MobileNetV5 vision and the audio towers are not imported) | `MMPROJ` (mobilenetv5, gemma3n audio) | gemma3n, gemma3n_text | gguf/gemma-3n-E2B-it, hf/tiny-random-gemma-3n-wide, gguf/tiny-random-gemma-3n-wide-f32 |
| `gemma4` | Gemma 4 (E2B/E4B/12B/26B-A4B MoE) | PLE, KV sharing, per-layer head dims, NoPE globals, k=v, MoE next to a dense MLP | **supported** (the 26B-A4B MoE layers run a dense MLP and routed experts side by side on the decoder (`mlp.routed_in_norm` / `dense_out_norm` / `routed_out_norm`), the router on its own weight-less norm times `mlp.router.input_scale`, softmax top-k renormalized and times the per-expert `mlp.experts.output_scale`; HF `enable_moe_block` and GGUF. Tiny float32 MoE model vs transformers (prompt logits, 24 greedy ids) and its converted GGUF vs llama.cpp b11177 (the same ids); gemma-4-26B-A4B-it UD-Q4_K_XL vs llama.cpp b11177 greedy text; CPU and Vulkan) | - | gemma4 | gguf/gemma-4-26B-A4B-it, hf/tiny-random-gemma4-moe, gguf/tiny-random-gemma4-moe-f32 |
| `gemma4-assistant` | Gemma 4 MTP assistants | drafter (centroid head, KV from target) | **supported** (draft head of a Gemma 4 target) | - | gemma4_assistant, gemma4_unified_assistant | hf/tiny-random-gemma4-assistant(-target), gguf/tiny-random-gemma4-assistant(-target)-f32 (drafts vs transformers) |
| `gemma-embedding` | EmbeddingGemma | bidirectional gemma3 + pooling | **supported** (bidirectional decoder layers, symmetric window, Dense projections after pooling; embeddinggemma-300m is gated and was not run) | - | gemma3_text (`use_bidirectional_attention`, sentence-transformers Dense modules) | hf/tiny-random-embeddinggemma (vs transformers), gguf/tiny-random-gemma-embedding-f32 (vs llama.cpp) |
| `gemma-embedding2` | EmbeddingGemma 2 | bidirectional gemma + PLE + pooling | **supported** (HF: bidirectional decoder layers with a symmetric window on the sliding layers, per-layer head sizes and key/value head counts (`per_layer_config`), unscaled attention with q/k/v norms, per-layer embeddings from a projection of the token embeddings alone, `layer_scalar`, the embedding projection after mean pooling; tiny model vs transformers on CPU and Vulkan. No llama.cpp release converts it, so no GGUF mapping; the vision and audio towers are not imported) | - | embedding_gemma2, embedding_gemma2_text | hf/tiny-random-embeddinggemma2 |

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
| `nemotron_h` | Nemotron-H, Nemotron-3-Nano, Nemotron 3.5 Super-VL / Bolt (text) | Mamba2 + attention + MLP hybrid, ReLU2 | **supported** (HF `nemotron_h_omni`, round 11: the language model below `language_model.`, Bolt's Puzzle `block_configs` with per-layer expert and selection counts; the released Super-VL / Bolt checkpoints are modelopt-quantized, which is not read yet) | - | nemotron_h, nemotron_h_puzzle, nemotron_h_omni | hf/tiny-random-nemotron-h-omni(-puzzle) |
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
| `graniteswitch` | granite-switch | granite + MTP/switch heads | **supported** | - | granite_switch | hf/tiny-random-granite-switch, gguf/tiny-random-granite-switch-f32 (numpy reference of the llama.cpp graph, llama.cpp 8ca5a80) |
| `granite_swa` | granite SWA | granite + SWA + sinks | **supported** | - | granite_swa, granitemoe_swa, granitemoeshared | hf/tiny-random-granite-swa, hf/tiny-random-granitemoe-swa, gguf/tiny-random-granitemoe-swa-f32.gguf |

### GLM

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `chatglm` | ChatGLM3, GLM-4-9B-chat | MQA-ish GQA, partial interleaved rope, fused qkv | **supported (GGUF, tiny random vs llama.cpp)**| -| chatglm | gguf/tiny-random-chatglm-f32 |
| `glm4` | GLM-4-0414 (9B/32B), GLM-Z1, GLM-4.1V | GQA, sandwich norms, partial rope, fused gate-up; GLM-4.1V: M-RoPE + vision tower | **supported** (GLM-4.1V: HF glm4v with its tower, GGUF glm4 with `rope.dimension_sections` (q/k in neox order) plus a glm4v mmproj; tiny model vs transformers (text: greedy ids and prompt logits; image prompt: embeddings, logits, greedy ids; CPU and Vulkan) and vs llama.cpp b11177 (text ids, mmproj tower outputs to ~1e-3, llama.cpp's f16 attention); no real checkpoint under ~4 GB) | - | glm4, glm4v | gguf/GLM-4-9B-0414, hf/tiny-random-glm-4, hf/tiny-random-glm4v, gguf/tiny-random-glm4v-f32 (+mmproj) |
| `glm4moe` | GLM-4.5/4.6/4.7, GLM-4.5-Air | QK-norm, partial rope, MoE sigmoid + bias + groups + shared, dense lead, MTP | **supported** | - (MTP layers dropped) | glm4_moe | hf/tiny-random-glm-4-moe |
| `glm-dsa` | GLM-5 / DSA models | MLA + DSA indexer, MoE | **supported** (dense and sparse regime; deviation: llama.cpp b11177's glm-dsa graph leaves the transformers ids on the same weights, the GGUFs are checked against transformers) | - | glm_moe_dsa | hf/tiny-random-glm-moe-dsa(-sparse), gguf/tiny-random-glm-moe-dsa(-sparse)-f32.gguf |
| `glm5-next` | GLM-5.3-Flash | MLA + indexer + HC + linear-attn layers | **supported** (latent decoder: Kimi delta attention with the bounded gate, latent attention without rotary dimensions, the k-pool indexer (softmax-gated pooled keys plus the current tail) with a shared indexer selection, SwiGLU clamped in every MLP, DeepSeek-V4 hyper-connections averaged before the output norm; HF and GGUF, tiny models (dense and sparse regime) vs transformers and llama.cpp 8ca5a80 on CPU and Vulkan; MTP layers and the vision tower not used) | - | glm5_next, glm5_next_text | hf/tiny-random-glm5-next, hf/tiny-random-glm5-next-sparse, gguf/tiny-random-glm5-next{,-sparse}-f32.gguf |

### MLA/DeepSeek-like

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `mistral4` | Mistral Large/Small 4 (MLA) | MLA, MoE, yarn (inherits deepseek2 graph) | **supported** | - | mistral4 | hf/tiny-random-mistral4, gguf/tiny-random-mistral4-f32.gguf, gguf/tiny-random-mistral4-temperature-f32.gguf |
| `mistral-large4` | Mistral Large 4 | MLA + indexer, MoE shared, yarn, attn temp | **supported** (latent decoder without q-LoRA: rope-less layers (`rope.layers`), layers reading the latent cache of an earlier layer (`attention.kv_source_layers`), scalable softmax on the attention and indexer queries, the indexer's LayerNorm'd query and gptj rope, shared indexer layers, learned yarn factors, final logit softcapping; GGUF only, tiny GGUF vs llama.cpp 8ca5a80 (flash attention / token by token) and a float64 numpy reference of its graph, CPU and Vulkan) | `MLA`, `IDX`, `RGRP` | - | gguf/tiny-random-mistral-large4-f32.gguf |
| `minicpm3` | MiniCPM3-4B | MLA, RMS | **supported** (latent_decoder) | - | minicpm3 | gguf/MiniCPM3-4B, hf/tiny-random-minicpm3 |
| `deepseek` | DeepSeek-LLM-MoE (v1) | MHA, MoE softmax + shared, dense lead | **supported** | - | deepseek | gguf/tiny-random-deepseek-f32.gguf |
| `deepseek2` | DeepSeek V2/V2-Lite/V3/R1, Kimi-K2, GLM-4.7-Flash, Moonlight | MLA (absorbed cache), MoE sigmoid/softmax + bias + groups + shared, yarn mscale, MTP | **supported** (latent_decoder; MTP layers loaded but unused) | - | deepseek_v2, deepseek_v3, kimi_k2 | gguf/DeepSeek-V2-Lite-Chat, hf/tiny-random-deepseek-v3.1, hf/tiny-random-glm-4.7-flash |
| `deepseek2-ocr` | DeepSeek-OCR | deepseek2 text decoder + vision | **supported** (text and images: DeepSeek-OCR, Unlimited-OCR, DeepSeek-OCR-2) | - | deepseek_vl_v2 (OCR), deepseek_ocr2, unlimited_ocr | hf/tiny-random-{deepseek,unlimited}-ocr[-sam12], hf/tiny-random-deepseek-ocr2[-sam12, -remote-sam12], gguf/tiny-random-deepseek-ocr[2]-sam12-f32 (+mmproj) |
| `deepseek32` | DeepSeek-V3.2 | MLA + DSA lightning indexer | **supported** (dense and sparse regime) | - | deepseek_v32 | hf/tiny-random-deepseek-v32(-sparse), gguf/tiny-random-deepseek-v32(-sparse)-f32.gguf |
| `deepseek4` | DeepSeek-V4 | hyper-connections (Sinkhorn), compressed/sparse attention, sinks, MoE hash/groups | **supported** (CPU and Vulkan, single-sequence, batched and paged) | - | deepseek_v4 | hf/tiny-random-deepseek-v4, gguf/tiny-random-deepseek-v4-f32 (llama.cpp 8ca5a80 with f32 caches) |
| `deepseek41` | DeepSeek-V4.1 | as v4 + PLE n-gram | **supported** (GGUF; tiny model vs a numpy reference of llama.cpp's graph and llama.cpp 8ca5a80 token by token, f32 caches; CPU and Vulkan, batched decoding matches single-sequence) | - | - | gguf/tiny-random-deepseek-v41.gguf |
| `xing4_0` | xing4.0 | MLA + manifold HC + MoE | **supported** | - | xing4_0 (remote code) | hf/tiny-random-xing4-0[-unclamped] (+ f32 GGUFs; llama.cpp ids on the unclamped model) |
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
| `bailingmoe3` | Ling 3 / Ring-linear | MLA + KDA/GDN hybrid + MoE | **supported** | - | bailing_hybrid | hf/tiny-random-bailing-hybrid, gguf/tiny-random-bailing-hybrid.gguf (MTP layers not mapped) |
| `dots1` | dots.llm1 | QK-norm, MoE sigmoid + bias + shared | **supported** | - (GGUF mapping not verified on a real file) | dots1 | hf/tiny-random-dots1 |
| `dots3note` | dots.3 | MLA + indexer + SWA | **supported** (latent decoder: sliding-window layers with their own MLA geometry, head count and rope base (`attention.sliding.*`, `attention.sliding_layers`), full layers with the DSA indexer, RMS norm on the rope key, per-head sigmoid attention gate, sigmoid MoE with a shared expert; GGUF only, tiny GGUF vs llama.cpp 8ca5a80 on CPU and Vulkan; the vision tower is not imported) | `MLA`, `IDX` | - | gguf/tiny-random-dots3note-f32.gguf |
| `afmoe` | AFM-MoE (Trinity) | SWA 3:1 (rope on the sliding layers only), sigmoid attention gate, sandwich norms, mup embedding scale, MoE sigmoid + bias + shared | **supported** (not run on a real model) | - | afmoe | hf/tiny-random-afmoe, gguf/tiny-random-afmoe-f32 |
| `laguna` | Laguna (poolside) | MoE sigmoid + bias + shared, softplus out gate (per head or per channel), per-layer head counts, own rope base/width on the SWA layers, yarn on the global layers | **supported** (not run on a real model; router logit softcapping not supported) | - | laguna | hf/tiny-random-laguna, hf/tiny-random-laguna-full, gguf/tiny-random-laguna-f32, gguf/tiny-random-laguna-full-f32 |
| `ernie4_5-moe` | ERNIE-4.5-21B-A3B, 300B-A47B | MoE softmax + bias + shared, dense lead | **supported** | - | ernie4_5_moe | gguf/ERNIE-4.5-21B-A3B-PT, hf/tiny-random-ernie-4.5-moe |
| `hunyuan-moe` | Hunyuan-A13B | QK-norm, MoE + shared, dynamic NTK | **supported** | - (GGUF mapping not verified on a real file) | hunyuan_v1_moe | hf/tiny-random-hunyuan-v1-moe |
| `hy_v3` | Hunyuan v3 | MoE sigmoid + bias + shared, MTP | **supported** (MTP layers ignored; not run on a real model) | - | hy_v3 | hf/tiny-random-hy-v3, gguf/tiny-random-hy-v3-f32 |
| `hy_v4` | Hunyuan v4 | MLA-like + indexer + HC | **supported** (HF, latent_decoder: MLA with sigmoid output gate and attention sinks, clamped SwiGLU experts, hyper-connections; tiny model vs transformers on CPU and Vulkan, plus a variant with DeepSeek-V4 mHC modules (Sinkhorn stream mixing) monkeypatched into the reference. The DSA indexer runs in both regimes, shared indexer layers reuse the preceding selection (tiny model with `index_topk` 8 vs transformers). GGUF mapping not done) | - | hy_v4 | hf/tiny-random-hy-v4(-sparse), hf/tiny-random-hy-v4-mhc |
| `gpt-oss` | gpt-oss-20b/120b | MoE top-4 softmax + expert biases, clamped SwiGLU-OAI, sinks, alternating SWA, yarn, MXFP4 weights | **supported** | - | gpt_oss | gguf/gpt-oss-20b (MXFP4), hf/tiny-random-gpt-oss, hf/tiny-random-gpt-oss-mxfp4 |
| `smallthinker` | SmallThinker 4B-A0.6B/21B-A3B | MoE (sigmoid gating), SWA+NoPE | **supported** | - (HF remote code not mapped) | smallthinker | gguf/SmallThinker-4BA0.6B-Instruct |
| `grovemoe` | GroveMoE | adjugate/chunk experts, QK-norm | **supported** | - | GroveMoeForCausalLM (remote code) | hf/tiny-random-grovemoe, gguf/tiny-random-grovemoe-f32 (remote-code MoE block) |
| `minimax-m2` | MiniMax-M2/M2.5 | full-width QK-norm, partial rope, MoE sigmoid + bias | **supported** | - (GGUF mapping not verified on a real file) | minimax_m2 | hf/tiny-random-minimax-m2 |
| `mimo2` | MiMo-V2-Flash | SWA + sinks, value head size < key head size, value scale, MoE sigmoid bias, MTP | **supported** (MTP layers ignored; not run on a real model. HF `mimo_v2` (MiMo-V2.6-Pro, round 11): fused `qkv_proj`, `swa_num_key_value_heads`, vision / audio towers skipped; tiny checkpoint vs the remote code and llama.cpp 8ca5a80. The released Pro checkpoint stores FP8 / MXFP4, see section 11) | - | mimo_v2_flash, mimo_v2 | hf/tiny-random-mimo-v2-flash, gguf/tiny-random-mimo-v2-flash-f32, hf/tiny-random-mimo-v2 |
| `step35` | Step-3.5-Flash | SWA, gated attn, MoE sigmoid + shared | **supported** (clamped SwiGLU, per-layer heads, per-head gate; MTP draft layers not used; not run on a real model, 196B) | - | - | gguf/tiny-random-step35-f32 (tiny GGUF vs llama.cpp b11177 greedy ids) |
| `maion-coder` | Maion-Coder | SWA + NoPE, MoE latent | **supported** (GGUF, tiny random vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan, b11177 predates it; not run on a real model) | - | - | gguf/tiny-random-maion-coder-f32 |
| `mellum` | Mellum (JetBrains) | SWA + MoE, MTP | **supported** | - | mellum | hf/tiny-random-mellum, hf/tiny-random-mellum-llamacpp, gguf/tiny-random-mellum-llamacpp-f32.gguf |
| `k2-horizon` | K2-Horizon | MoVA value experts, MoE sigmoid + shared, yarn | **supported** | - | k2_horizon (remote code) | hf/tiny-random-k2-horizon, gguf/tiny-random-k2-horizon-f32 (llama.cpp ids) |
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
| `chameleon` | Chameleon | QK-LN, swin norm, VQ image tokens | **supported** (text: q/k layer norms with a gain and bias per head (`attention.qk_norm_type` "layer", [heads, head_dim] weights), swin norm as post-norm layers, image-token logits suppressed (`logits.suppressed_ranges`); HF and GGUF, tiny models vs transformers and llama.cpp 8ca5a80 on CPU and Vulkan. Deviation: llama.cpp's swin-norm graph feeds the MLP the normed attention output, so the swin GGUF is checked against transformers. The VQ-GAN image tokenizer is not imported) | `VQ` (image input) | chameleon | hf/tiny-random-chameleon(-mha, -swin), gguf/tiny-random-chameleon-mha-f32.gguf, gguf/tiny-random-chameleon-swin-f32.gguf |
| `arcee` | Arcee AFM-4.5B | ReLU2 ungated MLP, yarn | **supported**| -| arcee | gguf/AFM-4.5B |
| `ernie4_5` | ERNIE-4.5 0.3B-VL, dense | GQA, interleaved rope, RMS | **supported**| -| ernie4_5 | gguf/ERNIE-4.5-0.3B-PT, hf/ERNIE-4.5-0.3B-PT, hf/tiny-random-ernie-4.5 |
| `hunyuan-dense` | Hunyuan 0.5B-7B | QK-norm, GQA, NTK-alpha rope | **supported**| -| hunyuan_v1_dense | gguf/Hunyuan-0.5B-Instruct, hf/Hunyuan-0.5B-Instruct, hf/tiny-random-hunyuan-dense-v1 |
| `hunyuan_vl` | HunyuanOCR / VL | xdrope M-RoPE + mmproj | **supported** (text and images, HF and GGUF + mmproj) | - | hunyuan_vl | hf/tiny-random-hunyuan-vl, gguf/tiny-random-hunyuan-vl-f32 (+mmproj) |
| `smollm3` | SmolLM3-3B | NoPE every 4th layer, GQA | **supported**| -| smollm3 | gguf/SmolLM3-3B, hf/SmolLM3-3B, hf/tiny-random-smollm3 |
| `seed_oss` | Seed-OSS-36B | GQA, qkv bias, RMS | **supported**| -| seed_oss | hf/tiny-random-seed-oss |
| `apertus` | Apertus 8B/70B | xIELU activation, QK-norm | **supported**| -| apertus | gguf/Apertus-8B-Instruct-2509, hf/tiny-random-apertus |
| `cogvlm` | CogVLM | visual-expert qkv/ffn | **supported** (text and images; CogVLM2 not yet) | - | CogVLMForCausalLM (remote code) | hf/tiny-random-cogvlm, gguf/tiny-random-cogvlm-f32 (+mmproj) |
| `pangu-embedded` | openPangu-Embedded | GQA, RMS | **supported** | - | - | gguf/tiny-random-pangu-embedded-f32.gguf |
| `paddleocr` | PaddleOCR-VL | ernie-like + M-RoPE | **supported** (language model: ERNIE-4.5 with sectioned non-interleaved M-RoPE, HF paddleocr_vl and GGUF paddleocr; tiny model vs transformers and its converted GGUF vs llama.cpp 11139 (b11177 lacks the architecture) greedy ids, CPU and Vulkan; image input: the SigLIP tower (learned position grid, 2-D rotary embedding, post-norm) and its `mlp_AR` projector on the shared vision tower graph, HF and mmproj (projector type paddleocr); tiny model with an image prompt vs transformers (merged-token embeddings, prompt logits, greedy ids, CPU and Vulkan), its tower converted by llama.cpp b11177 vs llama-mtmd-debug. Deviations of llama.cpp from the reference: tanh instead of exact GELU in the projector, position grid resampled without align_corners. No real checkpoint run (none under ~/models)) | - | paddleocr_vl | hf/tiny-random-paddleocr-vl, gguf/tiny-random-paddleocr-vl-f32 |
| `spark2_5` | Spark 2.5, Spark-X2.6 | SWA pattern, gated attn, GeGLU | **supported** (HF `spark2_5`, round 11: fused `q_k_v_proj`; tiny checkpoint vs the remote code and llama.cpp 8ca5a80, Spark-X2.6-4B vs transformers) | - | spark2_5 | gguf/tiny-random-spark2_5-f32.gguf, hf/tiny-random-spark2_5 |
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
| `lfm3` | LFM3-1.7B, LFM3-VL/Omni (text part) | gated conv + sliding/full attention (own rope bases, partial rope via `rope_freqs`), QK-norm, per-head sigmoid gate, sandwich norms | **supported** (GGUF: tiny random and real LFM3-1.7B Q8_0 vs llama.cpp 8ca5a80 greedy ids on CPU and Vulkan; b11177 predates lfm3. The non-causal "d1" decision variant (bidirectional full-attention layers, no cache) is not supported; no transformers class. HF `lfm3` (round 11): tiny checkpoint vs the remote code and llama.cpp 8ca5a80, LFM3-1.7B bf16 vs transformers) | - | lfm3 | gguf/LFM3-1.7B, gguf/tiny-random-lfm3-f32, hf/tiny-random-lfm3 |
| `inkling` | Inkling, Inkling-Small (Thinking Machines; text decoder) | rope-less GQA attention, sliding-window layers with their own kv head count, q/k norms with scale 1/head_dim, residual short convolutions of k / v / attention output / MLP output, query-conditioned relative position bias, log-floor query temperature, sigmoid MoE whose shared experts are scored by the router, embedding norm, padded vocabulary | **supported** (text decoder on the hybrid stack: `HybridAttention` relative bias (CachedAttention `relative_bias`, CPU and Vulkan), residual short convolutions and per-layer head counts, `MoeBlock` shared-expert sinks; HF `inkling_mm_model` in the published `model.llm.*` layout and GGUF from unsloth's fork; tiny model vs transformers and the fork's llama.cpp (same 24 ids), CPU and Vulkan, batched decoding matches single-sequence decoding; every Inkling-Small GGUF and the Inkling checkpoint pass the check and import from their headers, none runs here (the smallest file is 75 GB). The vision / audio towers and the MTP layers are not imported. `inkling2` is missing, see section 10) | - | inkling_mm_model | hf/tiny-random-inkling, gguf/tiny-random-inkling-f32.gguf |

### Linear attn

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `minimax-01` | MiniMax-Text-01 / M1 | lightning linear attention + softmax every 8th, MoE | **supported** | - | minimax | hf/tiny-random-minimax, hf/tiny-random-minimax-llamacpp, gguf/tiny-random-minimax-llamacpp.gguf |
| `minimax-m3` | MiniMax-M3 | block-sparse indexer + SwiGLU-OAI MoE | **supported** (decoder: per-head Gemma q/k norms, partial rope, dense and MoE layers with the clamped SwiGLU-OAI, sigmoid routing with selection bias and a shared expert, the block-sparse indexer selecting key blocks per key-value group; HF (published and transformers layouts) and GGUF, tiny models (dense and sparse regime) vs transformers on CPU and Vulkan, the dense one also vs llama.cpp 8ca5a80; the vision tower is not imported) | - | minimax_m3_vl, minimax_m3_vl_text | hf/tiny-random-minimax-m3, hf/tiny-random-minimax-m3-sparse, gguf/tiny-random-minimax-m3{,-sparse}-f32.gguf |
| `kimi-linear` | Kimi-Linear-48B-A3B | KDA (per-channel gated delta) 3:1 + MLA, MoE sigmoid + shared, conv | **supported** | - | kimi_linear | hf/tiny-random-kimi-linear-f32, gguf/tiny-random-kimi-linear-f32.gguf |
| `kimi-k3` | Kimi K3 | KDA + MLA + HC | **supported** | - | kimi_k3 | hf/tiny-random-kimi-k3 (original MXFP4 checkpoint), hf/tiny-random-kimi-k3-f32, gguf/tiny-random-kimi-k3-f32 |
| `kimi-k31` | Kimi K3.1 | KDA + MLA + HC + indexer | **supported** | - | kimi_k3 (with dsa_config) | hf/tiny-random-kimi-k31[-shared]-f32, gguf/tiny-random-kimi-k31[-shared]-f32 |

### Encoder

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `bert` | BERT, MiniLM, bge, e5, rerankers | bidirectional, post-LN, abs pos + token types, GELU, pooling | **supported** | - | bert, roberta, xlm-roberta | all-MiniLM-L6-v2, bge-small-en-v1.5, bge-reranker-base / -v2-m3 (Sharplight.Embeddings.Tests) |
| `modern-bert` | ModernBERT, Granite Embedding R2 | bidirectional, local/global rope, GeGLU, LN | **supported** (canonical decoder with `attention.causal` = false; llama.cpp's converter rejects the masked-LM checkpoints, so the real model was checked through the HF importer only) | - | modernbert | hf/ModernBERT-base (vs transformers), hf/tiny-random-modernbert (vs transformers), gguf/tiny-random-modern-bert-f32 (vs llama.cpp) |
| `nomic-bert` | nomic-embed-text v1/1.5 | bidirectional, rope, SwiGLU | **supported** | - | nomic_bert | nomic-embed-text-v1.5 f16 / Q8_0 GGUF (Sharplight.Embeddings.Tests) |
| `nomic-bert-moe` | nomic-embed-text-v2-moe | bidirectional + MoE | **supported** (GGUF; encoder adapter with the shared `MoeBlock`: softmax routing without renormalization, ungated GELU experts every second layer; tiny GGUF vs llama.cpp b11177 `/embedding`, CPU and Vulkan; the checkpoint's expert output bias is dropped by llama.cpp's converter, so GGUFs lack it; transformers' native `nomic_bert` has no MoE, so the HF checkpoint (remote code) is not mapped; no real checkpoint under ~/models) | - | - | gguf/tiny-random-nomic-bert-moe-f32 |
| `neo-bert` | NeoBERT | bidirectional, RMS, SwiGLU, rope | **supported** (GGUF; NeoBERT's HF code is remote-only) | - | - | gguf/tiny-random-neo-bert-f32 (vs llama.cpp) |
| `jina-bert-v2` | jina-embeddings-v2 | bidirectional, ALiBi | **supported** (GGUF; encoder adapter with ALiBi, q/k projection norms, second attention norm, GeGLU) | - | - | gguf/real-embed/jina-embeddings-v2-small-en-f32 (converted, vs llama.cpp), gguf/tiny-random-jina-bert-v2-f32 (vs llama.cpp) |
| `jina-bert-v3` | jina-embeddings-v3 | bidirectional, rope, LoRA tasks | **supported** (encoder adapter: post-norm layers with rope, fused qkv, GELU MLP; the task LoRA adapters are separate GGUF files and not applied; tiny HF model in transformers' native `jina_embeddings_v3` form vs transformers, tiny GGUF vs llama.cpp b11177 `/embedding`, CPU and Vulkan; no real checkpoint under ~/models) | - | jina_embeddings_v3 | hf/tiny-random-jina-embeddings-v3, gguf/tiny-random-jina-bert-v3-f32 |
| `eurobert` | EuroBERT | bidirectional llama-like | **supported** | - | eurobert | hf/EuroBERT-210m (vs transformers), hf/tiny-random-eurobert (vs transformers), gguf/tiny-random-eurobert-f32 (vs llama.cpp) |

### Encoder-decoder

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `t5` | T5, Flan-T5, UL2 | cross-attention, relative-position buckets, RMS(no bias), GeGLU/ReLU | **supported** (encoder_decoder adapter: encoder pass per request, cross-attention over the encoder output (CPU and Vulkan kernels), relative position bias; flan-t5-small Q4_K_M vs llama.cpp b11177, tiny T5 / T5 v1.1 and flan-t5-small f32 vs transformers, CPU and Vulkan. Deviations: llama.cpp never scales the tied LM head of the original T5 (transformers does; Sharplight follows transformers), and flan-t5-small Q4_K_M departs from the f32 HF greedy ids at step 4) | - | t5, mt5, umt5 | gguf/flan-t5-small, hf/flan-t5-small, hf/tiny-random-t5, hf/tiny-random-t5-v1_1 |
| `t5encoder` | T5 encoder (embeddings) | bidirectional + relative buckets | **supported** (GGUF mapping onto the encoder half of the encoder_decoder adapter with mean pooling; tiny-random-t5's encoder converted as T5EncoderModel vs transformers' T5EncoderModel embeddings on CPU and Vulkan. Deviation from llama.cpp: b11177's llama-server `/embedding` disagrees with transformers for this architecture (its t5 encoder, which Sharplight matches through the t5 cases, does not); GGUF has no `relative_attention_max_distance`, 128 is assumed like llama.cpp) | - | - | gguf/tiny-random-t5encoder-f32 |

### Diffusion

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `dream` | Dream-7B | masked diffusion, bidirectional qwen2-like | supported | `DIFF` | Dream | ✓ |
| `llada` | LLaDA-8B | masked diffusion, llama-like | supported | `DIFF` | llada | ✓ |
| `llada-moe` | LLaDA-MoE | masked diffusion + MoE | supported (GGUF; the HF checkpoints reuse LLaDA's model type `llada`) | `DIFF` | - | ✓ |
| `rnd1` | RND1 (Qwen3-MoE diffusion) | bidirectional qwen3moe | supported | `DIFF` | rnd1 | ✓ |

### Spec-decode

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `eagle3` | EAGLE3 drafters | drafter reading target hidden states | **supported** (`Eagle3DraftHead`: loaded as the `draft` auxiliary source of a `decoder` target, whose decode graphs then export the inputs of the three fused layers as `hidden_taps`; becomes the target's draft graph for `MtpSpeculator`, with the drafter's own vocabulary mapped through `d2t`, its own or the target's embedding; HF (EAGLE / SpecForge `midlayer.*` and speculators `layers.0.*` layouts) and GGUF; tiny Llama target and drafter: drafts equal a transformers + numpy reference, greedy output with speculation equals output without, CPU and Vulkan) | `SPECDEC` | LlamaForCausalLMEagle3 | hf/tiny-random-eagle3, gguf/tiny-random-eagle3-f32.gguf |
| `dflash` | DFlash drafters | diffusion-flash drafter | **partial**: plain DFlash and DFlash2 drafters load as the `draft` auxiliary source of a `decoder` target (`DflashDraftHead`, `BlockDraftSpeculator`): the target exports the inputs of the drafter's `target_layers` as `hidden_taps`, and one pass of the draft graph writes the new context rows (fc over the taps, `enc.output_norm`) to the drafter's KV cache and denoises a block of the pending token and mask tokens with bidirectional `CachedAttention` (`causal` false, also in the multi-sequence form), sliding layers, sinks and value scale; DFlash2 adds the grouped causal convolution of the attention and MLP inputs and the selector lattice (top-k candidates per row, scored by successor · (gate ⊙ predecessor) + logit, walked on the host). Tiny drafters over the tiny Llama target (`tools/mk_tiny_dflash.py`, a float64 numpy reference of llama.cpp's graph that matches llama.cpp 8ca5a80's drafts 24 / 24): drafts equal the reference, greedy output with speculation equals output without, several sequences batched, CPU and Vulkan; Muse-Glimmer-30B UD-Q4_K_XL with its drafter: the same 40 greedy tokens as llama.cpp, 58.5 % of the drafts accepted (llama.cpp 66.1 % over 48 tokens). Missing: the LiLiCorr correction head, DSpark, taps from `qwen35` / `nemotron_h` targets | `SPECDEC` | - | gguf/tiny-random-dflash-f32.gguf, gguf/tiny-random-dflash2-f32.gguf |

### Multimodal

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `clip` | mmproj / vision towers | ViT encoders | **supported** | - | - | - |

### Exotic

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `clef` | clef (decision head, GDN hybrid) | GDN hybrid + decision-order scoring head | **supported** (text, through `SpanScorer`) | - | clef (joint_head_config.json) | hf/tiny-random-clef, gguf/tiny-random-clef-f32 |
| `vllm-decision2` | vllm-decision2 | LayerNorm transformer + decision head | **supported** (through `SpanScorer`) | - | Decision 2.0 package layout | hf/tiny-random-vllm-decision2, gguf/tiny-random-vllm-decision2-f32 |
| `metis16` | metis16 | Mamba2 + HC + memory/route retrieval | **supported** (GGUF) | - | - (remote code) | gguf/tiny-random-metis16-f32 (numpy reference of the llama.cpp graph, llama.cpp 8ca5a80) |
| `hrm_text` | HRM-Text 1B | alternating low/high stacks, H/L cycles | **supported** | - | hrm_text | hf/tiny-random-hrm-text, gguf/tiny-random-hrm-text-f32 |
| `limite` | Limite | XSA, MUDD cross-layer taps, value embeddings | **supported** | - | LimiteForCausalLM (remote code) | hf/tiny-random-limite, gguf/tiny-random-limite[-hf]-f32 (numpy reference of the llama.cpp graph) |

### Out of scope

| Arch | Models | llama.cpp blocks | Status | Missing in Sharplight | HF type | Test assets |
|---|---|---|---|---|---|---|
| `qwen3tts` | Qwen3-TTS | speech codec LM | **missing** | `TTS` | - | - |
| `pockettts` | PocketTTS | TTS flow decoder | **missing** | `TTS` | - | - |
| `wavtokenizer-dec` | WavTokenizer decoder | audio codec decoder | **supported** (as the `vocoder` of OuteTTS, see section 12) | `TTS` | `Sharplight.Import.Gguf` mapping, `WavTokenizerGraph` | `tests/Sharplight.TextToSpeech.Tests` |

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
| Qwen vision | qwen2vl, qwen3vl (DeepStack), qwen3.5 merger (`qwen3vl_merger`) | Qwen2-VL/2.5-VL, Qwen3-VL(-MoE), Qwen3.5/3.6/3.8 | supported: qwen3_5, Qwen3-VL (DeepStack), Qwen2-VL (`qwen2vl_merger`) and Qwen2.5-VL (`qwen2.5vl_merger`, windowed attention) |
| SigLIP / LLaVA-like | siglip, llava, internvl, minicpmv, mobilenetv5 (Gemma 3n), gemma4v/uv, kimivl, kimik25, lfm3vl, step3vl, magevl, mimovl, youtuvl, ling3vl, exaone4_5, nemotron-v2-vl, granite4-vision | Gemma 3/4, SmolVLM, InternVL, MiniCPM-V, GLM-4.xV, Kimi-VL, LFM2-VL, Step3-VL | missing (ViT tower graph exists in VisionTowerGraph; per-family resamplers/mergers and preprocessing missing; **done**: paddleocr (SigLIP tower + mlp_AR projector) and glm4v (patch norm, bicubic position grid, convolution merger, GLM projector) on the shared tower graph) |
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

### 8.3 Regenerating the test assets

The tests skip a case whose model file is missing, so a fresh machine only needs the files below (no network access at test time). Use a
Python venv (torch CPU, transformers 5.23, gguf, safetensors, sentencepiece, huggingface_hub, diffusers, onnx) and a llama.cpp checkout
(`LLAMA_SRC`, a build of 8ca5a80 / b11177 for the llama.cpp references). `MODELS` is `~/models`.

| Asset | How |
|---|---|
| `hf/tiny-random-*` that come from the Hub | `huggingface_hub.snapshot_download` of `hf-internal-testing/tiny-random-<Class>`, `tiny-random/<name>` (the local name is `tiny-random-<name>`), `optimum-internal-testing/` and `katuni4ka/`; `tiny-random-nemotron-h` is `optimum-intel-internal-testing/tiny-random-nemotron-h`. Repositories that ship only `pytorch_model.bin` get a safetensors file from `tools/model-assets/bin_to_safetensors.py` |
| `hf/tiny-random-*` made here | `tests/Sharplight.Families.Tests/tools/mk_tiny_*.py` (the usage line of each script), seeds fixed; they print the transformers greedy ids that the case files hold. The tokenizer-only inputs are Hub files: `Qwen2.5-0.5B-Instruct`, `ERNIE-4.5-0.3B-PT`, `gemma-4-E2B-it`, `Qwen3.5-0.8B`, `Llama-3.2-1B-Instruct`, `DeepSeek-OCR`, `HunyuanOCR`, `google/paligemma-3b-pt-224` (gated: `HF_TOKEN`) |
| remote-code generators (kimi_k3, k2_horizon, xing4_0, clef, vllm_decision2, cogvlm, `mk_tiny_remote_code.py` families) | run the model repository's own code: `tools/model-assets/stage_remote.py` downloads code and configs first, `tools/model-assets/run_remote_sandboxed.sh` then runs them in a venv of their own with an empty environment (no tokens) and offline flags |
| `gguf/tiny-random-*-f32.gguf` of HF checkpoints | `tools/model-assets/convert_all.py` (llama.cpp's converter through `convert_tiny_gguf.py`; `TINY_PRE=falcon` for the Falcon file, `TINY_PRE=auto` for the real tokenizers of glm4v, hunyuan-vl and deepseek-ocr, `TINY_CONVERT_ARGS=--mtp` for the MTP draft file of nemotron-h, `--no-mtp` for its trunk) |
| `gguf/tiny-random-*-f32.gguf` of llama.cpp-only families | `mk_tiny_gguf.py <falcon GGUF> <out dir>`; the Falcon GGUF supplies the tokenizer and **must carry the `falcon` pre-tokenizer** (otherwise every prompt tokenizes differently and the llama.cpp ids change) |
| GGUF references | `tools/model-assets/llamacpp_ids.py <runtime dir> <gguf> <prompt> <n>` (llama-server, CPU, greedy, `return_tokens`) |
| `sd/`, `whisper.cpp/`, `audio/jfk.wav` | `ggerganov/whisper.cpp` (`ggml-*.bin`), `stable-diffusion-v1-5/stable-diffusion-v1-5` (diffusers files, `v1-5-pruned-emaonly.safetensors`), `second-state/stable-diffusion-v1-5-GGUF`, whisper.cpp `samples/jfk.wav`; the tiny SD pipeline is `tools/make-tiny-diffusion-models.py` (wip/diffusers) |
| `onnx/` | `bench/onnx/download_models.py`; the conformance data with `tools/gen-onnx-node-tests.py`, `gen-onnx-ml-tests.py`, `gen-onnx-contrib-tests.py` in a venv with onnx 1.23.2 and onnxruntime 1.31 |

Not reproducible from public sources alone: the `-sparse` variants (their seeds, budgets and prompts are arguments of `mk_tiny_dsa_sparse.py`,
`mk_tiny_group_sparse.py` and `mk_tiny_qwen4exp_sparse.py` that were not recorded).

Converter notes. A GGUF made from a checkpoint with a real tokenizer (lfm3-hf, clef, vllm-decision2, glm4v, hunyuan-vl, deepseek-ocr) needs `TINY_PRE=auto`: with the default
GPT-2 pre-tokenizer llama.cpp splits the prompt differently from transformers (clef: `\n\n` as two tokens) and every id differs; `convert_all.py` sets it per
name. `convert_tiny_gguf.py` copies sub-directories (the `backbone/` of a Decision 2.0 package). The GGUF references of the minimax-m2 file come from
llama.cpp b11177 (CPU) and those of the lfm3 files from 8ca5a80 (b11177 predates lfm3).

Still missing (cannot be rebuilt from what is recorded): `kimi_k3` / `kimi-k31*` and the `-sparse` variants (seeds unknown), `gguf/tiny-random-clover-moe*.gguf`
(the retargeting script `retarget_gguf.py` that the case comment names was never committed), and the cogvlm GGUF + mmproj (the stock converter cannot
read the original cogvlm config).

## 9. Reproducing the references

```bash
D=$LLAMA_ROOT/runtime.llama-b11177-cuda13   # or -vulkan / -cpu
LD_LIBRARY_PATH=$D $D/llama-completion -m MODEL.gguf -p 'The capital of France is' -n 16 \
    --temp 0 --top-k 1 --seed 1 -c 512 -no-cnv --no-display-prompt -t 16 --no-warmup -ngl 99 </dev/null
```

The GPU is shared with other applications: for a model that does not fit, drop `-ngl` (llama.cpp then fits automatically) or use the cpu build. Greedy outputs of 16 tokens are sensitive to quantisation and to the backend's float summation order; use them as a first parity gate (identical text expected for the same file on CPU) and compare logits/token ids for finer checks. Quantised GGUFs from different repackagers of the same model are not interchangeable references.

## 10. Real-checkpoint sweep (bring-up round 10, 2026-11-15)

Method. `bench/ModelSweep` lists every GGUF file (every shard) of a repository, or the config and safetensors headers of a Hugging Face checkpoint, and reads only the headers through HTTP range requests. It writes them into sparse files of the real size and runs two things on them: the header-only compatibility check (`GgufSupport.Validate` / `HfSupport`, as Studio computes it) and the real import (`LlmModel.Load`: importer mapping, adapter graph build, tokenizer, chat template). Tensor data reads as zeros and is never needed for that. The repositories were the most downloaded and trending ones from unsloth, bartowski, ggml-org, the official organizations and some community quantizers. They cover dense, MoE, vision-language, embedding, decision and draft models, and every quantization mix found: UD-*, IQ1–IQ4, TQ1_0/TQ2_0, Q4_0_4_x / Q4_0_8_8, MXFP4_MOE, NVFP4, FP8, Bonsai PQ2_0/PTQ1_0 and the GSQ-RCO allocations. Greedy output was compared with llama.cpp 8ca5a80 (`llama-completion`, 16 tokens, prompt "The capital of France is") on Vulkan and on CPU, for 21 files of 18 checkpoints, each under 20 GB.

Results (after the fixes below):

- **GGUF: 1138 files in 88 repositories.** The check and the import agree on every file.
  - 1051 are accepted and import.
  - 34 are draft models: MTP files, Gemma 4 assistants and EAGLE-3. They are reported as such and load as the `draft` source of their target.
  - 32 have no `general.architecture`: imatrix files, patch files, and MiniMax-H3 pruned files that lack it.
  - 6 use an architecture that is not mapped: four `dflash` drafters, `inkling` and `inkling2`.
  - 9 use ggml type ids this version does not know: 64/65/66 in Kimi-K3 UD-TQ2_0 / UD-TQ1_0 / UD-Q1_0, and 144–148 in agentionai's Gyro TQ*_0.
  - 3 carry NVFP4 per-tensor scales (see below; they import since the HF quantization round: LFM3 NVFP4, Super-VL NVFP4 and Bolt IQ2_XXS).
  - 3 are split uploads that lack a shard.
- **Hugging Face: 72 checkpoints.**
  - 43 import.
  - 19 have a `model_type` that is not supported.
  - 5 are gated for the token used.
  - 2 have an unsupported quantization (modelopt NVFP4, mxfp8). Gemma-4-26B-A4B-NVFP4 imports since the HF quantization round (ModelOpt NVFP4, experts stored one by one); MiniMax-M3.1-Flash's mxfp8 is still unread.
  - 3 passed the check but failed the import, all on quantization formats (open items 6 and 7); Kimi-K3, K3.1-Pro and DeepSeek-V4-Flash import since the HF quantization round.

The first pass, before this round's fixes, had 987 accepted GGUF files, 48 files rejected by the mapping and 9 files the check accepted but the import rejected.

Fixed this round (each with a tiny model or header fixture, CPU and Vulkan):

| Problem found | Repositories | Fix |
|---|---|---|
| A latent-attention `kv_b` of an IQ* / Q2_K / Q3_K / TQ / MXFP4 / NVFP4 type crashed the split at load | bartowski/DeepSeek-Coder-V2-Lite IQ2_XS (and every MLA model in such a quant) | Managed decoders for every ggml block format (`GgmlBlockDecoder.LowBit`), bit-exact against ggml (`GgmlBlockDecoderTests`); greedy output now equals llama.cpp's |
| `ffn_latent_down` / `ffn_latent_up` (HF `fc1_latent_proj` / `fc2_latent_proj`) unmapped | Nemotron 3.5 Super-VL, Bolt, 3 Ultra | Latent experts in `MoeBlock` for `nemotron_h` (`tools/mk_tiny_nemotron_h_latent.py`, transformers and llama.cpp ids) |
| Per-layer `expert_used_count` and expert counts | Nemotron 3.5 Bolt (Puzzle) | Per-layer selection counts (`moe.layer_expert_used_counts`), expert count from the router rows (`tools/mk_tiny_nemotron_h_puzzle.py`, llama.cpp ids, batched decoding matches) |
| The NextN block appended after the trunk (`blk.N.nextn.*`) unmapped | Nemotron 3.5 Lightning (32 files), Super-VL MTP files | Trunk = `block_count - nextn_predict_layers`, NextN block below `mtp.layers.0.`; HF `mtp.layers.0/1` folded the same way (`tools/mk_tiny_nemotron_h_mtp.py`, `tools/mk_tiny_nemotron_h_mtp_hf.py`) |
| MTP-only files accepted by the check, then failing on a missing trunk weight | unsloth Qwen3.8-27B `MTP/`, bartowski Qwen3.6, Lightning / Super-VL `mtp-*`, Gyro `mtp-*-draft` (qwen4exp) | The check reports `IsDraftModel` from the mapped names or, when the family does not import its NextN head, from the block indices; the importer and the Nemotron-H adapter refuse such a file as a draft file (`MtpDraftHead.IsSidecar`) |
| `gemma-embedding2` unmapped | unsloth/embeddinggemma-2-GGUF | GGUF mapping on the Gemma 4 decoder path (tiny GGUF vs transformers) |
| The decision scorer stored as a vector `[W]` | ggml-org/Clef-Flash-GGUF | Reshaped to the canonical `[1, W]` (`tools/mk_clef_vector_scorer.py`) |
| BF16 n-gram convolution, `exp_probs_b_vl`, interleaved Q4_0 type 31 | Qwen3.8-Flash-Next UD-IQ1_M, DeepSeek-V4(.1)-Flash, Llama-3.2-1B Q4_0_4_4 | Header fixtures (`GgufSweepFixtureTests`) |
| A split upload without its last shard accepted by the check | ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (shard 6 of 6 absent on the Hub) | The check compares `split.tensors.count` with the tensors of the shards present |

Greedy identity against llama.cpp 8ca5a80, 16 tokens:

- Identical on both Vulkan and CPU for 9 checkpoints: LFM2.5-1.2B IQ3_XXS, Llama-3.2-1B Q4_0, Phi-4-mini IQ4_NL, gemma-4-E4B QAT q4_0, Ministral-3-3B UD-IQ3_XXS, Qwen3-VL-4B UD-IQ2_XXS, GLM-4.7-Flash UD-TQ1_0, DeepSeek-Coder-V2-Lite IQ2_XS, and gemma-4-26B-A4B UD-Q4_K_XL with the chat template.
- Identical on one device: SmolLM3 IQ3_XS, Qwen3-0.6B IQ2_M, gemma-3-4b UD-IQ2_M and LFM2.5-8B MXFP4_MOE on Vulkan; Qwen3-30B-A3B UD-TQ1_0, MiMo-9B IQ2_M, Qwen3-4B UD-IQ1_S and the interleaved Llama Q4_0_4_4 / Q4_0_8_8 on CPU. llama.cpp read the interleaved files through a deinterleaved copy, since it repacks them only on ARM.
- Different on both: Qwen3-4B UD-Q2_K_XL, LFM2.5-8B UD-IQ2_XXS and LFM3-1.7B QAD-Q4_0, all after the first tokens.

Every divergence was checked against an f32 reference of the same file: Vulkan with `--disable-features int_dot,matrix_*,fp16,bf16`, logits compared at the first differing step. Our logits correlate with the reference at ≥ 0.989 in every case. The runtimes round differently (int8 activation dot products, f16 accumulation) and a near-tie flips. No mapping or kernel error was found.

gemma-4-26B-A4B UD-Q4_K_XL on the raw 6-token prompt is the extreme case. Vulkan continues "CAPITOL,- ,", while llama.cpp and our CPU give a chat-style "thought" answer. The logits correlate with the f32 reference at only 0.91 on prompts of 6 tokens. On prompts of 9 or more tokens they correlate at ~1.0, and the chat-templated prompt is identical on both devices. The model's very short prompts are sensitive to int8 activations: the CPU int-dot path alone gives 0.965. The defaults stay as they are, to keep the existing greedy outputs.

Open, with estimates:

1. **NVFP4 per-tensor scales in GGUF** (done in the HF quantization round: the NVFP4 tensor and its multiplier become one `nvfp4_gs` weight, `GgufNvfp4Scales`, `Nvfp4ScaleTests`; LFM3-1.7B NVFP4 generates coherent text on CPU and Vulkan, logit correlation 0.98 with its Q8_0 file; llama.cpp 8ca5a80 was not at hand to compare with). Three files: Super-VL NVFP4, LFM3 NVFP4, and Bolt IQ2_XXS (whose `ssm_in` / `ssm_out` are NVFP4).
   - Each NVFP4 matrix has a `X.scale` multiplier, per tensor or per expert, that llama.cpp applies to the matmul output (`build_lora_mm(w, x, w_s)`). The `X.input_scale` tensors are ignored.
   - The fix is a canonical `*.weight_scale` companion that `GraphBuilder.Linear` and `MoeBlock`'s MatMulId paths multiply in. It cannot be folded into the E4M3 block scales without loss.
   - About 1 day with tiny NVFP4 fixtures. It is shared with the HF NVFP4 work of the quantization workstream.
2. **dflash drafters.** Four files: Muse-Glimmer, Bonsai DFlash2, Bolt, and the Super-VL LiLiCorr variant.
   - Done (bring-up round 11): the `dflash` canonical architecture, the drafter graph over the target's taps with bidirectional cached attention, the block-draft proposer `BlockDraftSpeculator` and the DFlash2 convolution and selector lattice. All four files are reported as draft models by the check and the import. Muse-Glimmer runs with its target (greedy output equals llama.cpp, 58.5 % acceptance vs llama.cpp's 66.1 %).
   - Open: Bonsai's target is `qwen35`, whose hybrid adapter does not export taps yet (about 0.5 day: the same `hidden_taps` output on the hybrid stack, rollback of the recurrent state is already there). Bolt and Super-VL need the LiLiCorr correction head (a small transformer over the top-k candidates of every block row, with its own heads and embeddings: about 2 days with a numpy reference), plus taps from `nemotron_h` targets (0.5 day) and the NVFP4 per-tensor scales of item 1. Sliding windows of the drafter count from the end of each block run in the multi-sequence form, a deviation from llama.cpp that only matters for contexts longer than the window.
3. **inkling2** (Inkling-2-Small, HF `inkling2` and GGUF `inkling2`). Inkling (v1) is supported since bring-up round 11 (section 4).
   - Inkling 2 changes most of the v1 blocks: block attention residuals (as Kimi K3's `AttentionResiduals`), gated attention with per-token sinks, an L2 clip of q / k / r, an RMS norm on the values, short convolutions whose taps are computed per token, separate q / k / v input norms, softcapped SwiGLU (gate cap 4, up cap 25), a final logit softcap of 30 and per-expert output scales; 384 experts and no shared experts.
   - New pieces: the per-token convolution taps (a CausalConv1d variant taking the kernel as an input, CPU and Vulkan), the per-token attention sinks, the L2 clip and the softcapped SwiGLU; the rest exists. About 3–4 days with a tiny model against transformers.
   - Four of its nine GGUFs (BF16, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q8_K_XL) carry NVFP4 tensors with per-tensor scales, so they also need item 1.
4. **ggml type ids 64–66 and 144–149.** Decoded since bring-up round 11.
   - 64 / 65 / 66 (unsloth's IQ1_XS / IQ1_XXS / IQ1_XXXS, Kimi-K3 UD-TQ2_0 / UD-TQ1_0 / UD-Q1_0) store IQ1_S blocks without some of their fields; they are widened to IQ1_S blocks as they are read (`GgufNarrowIq1Source`), bit-exact against the fork's `dequantize_row`. All nine Kimi-K3 GGUFs now pass the check and import.
   - 144–149 (agentionai's TQ2_T, TQK6, TQK7, TQP5, TQP6, TQP7) are the trellis types `trellis_v{4,2}k{5..8}_g128`, decoded by the managed decoder, the CPU dequantization and the Vulkan `load8` paths, bit-exact against a transcription of the fork's reference decoder. Every Gyro trellis tensor is a Hadamard-rotated expert, so the files import through the rotation unfolding, which dequantizes those experts to F16 at load: correct, but over 200 GB for any of them. Running them in their own size needs the rotation applied to the activations instead (x·H before the expert matmuls, a fast Walsh–Hadamard op on CPU and Vulkan) and the trellis types kept as matmul weights: about 2–3 days. Nothing was run on a real file (30–58 GB).
   - Gyro's `-noPLE` files leave out the n-gram table, which the fork loads from a separate file (`--ple`); the check and the importer report them. Loading the table as an auxiliary source is about 0.5 day.
5. **MiniMax-H3 pruned files** and the DeepSeek-V4.1 `Engram-IQ4_NL` file have no `general.architecture`. They are malformed at the source and are reported as such; nothing to do.
6. **HF compressed-tensors quantization nested in `text_config`**, in Kimi-K3 and K3.1-Pro (done in the HF quantization round: `HfConfig.TryGetQuantization` looks there after the root, for the check and every weight converter; `NestedQuantizationConfigTests`).
   - Their `quantization_config` (MXFP4 pack) sits in `text_config`.
   - `HfSupport` and `HfMxfp4Weights` read only the root config's, so the check calls the checkpoint unquantized and the import does not convert `weight_packed`.
   - The fix is to look up `quantization_config` through the merged config; about 0.5 day. It belongs to the HF quantization workstream (MXFP4), so it is recorded only.
7. **DeepSeek-V4-Flash native checkpoint (HF)** (done in the HF quantization round: F8_E8M0 block scales for the FP8 weights, the I8-packed FP4 experts as `mxfp4`; `E8m0ScaleImportTests`, and `DeepSeekNativeQuantTests` runs the tiny DeepSeek-V4 quantized this way against its own dequantization on CPU and Vulkan; the 284B checkpoint itself was only header-imported).
   - FP8 attention weights plus FP4 experts with `F8_E8M0` scales: the check sees `quant_method fp8` and accepts the checkpoint, and the import stops at the E8M0 scale dtype.
   - The fix is an E8M0 scale reader and E2M1 experts as MXFP4; about 1 day. This is HF quantization workstream territory, so it is recorded only.

Per-repository table (check and import columns count files by outcome; greedy only where compared):

| Repository | Quantizations | Check | Import | Greedy vs llama.cpp |
|---|---|---|---|---|
| unsloth/gemma-4-12b-it-GGUF | mtp-BF16, mtp-F16, mtp-Q8_0, BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, ... (25 files) | 21 supported, 4 draft | 21 ok, 4 draft |  |
| unsloth/gemma-4-26B-A4B-it-GGUF | BF16, mtp-BF16, mtp-F16, mtp-Q8_0, MXFP4_MOE, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, ... (25 files) | 21 supported, 4 draft | 21 ok, 4 draft | UD-Q4_K_XL: chat prompt identical (Vulkan, CPU); raw 6-token prompt: CPU identical, Vulkan differs (int8 activations, see below) |
| unsloth/gemma-4-E2B-it-GGUF | mtp-BF16, mtp-F16, mtp-Q8_0, BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, ... (25 files) | 21 supported, 4 draft | 21 ok, 4 draft |  |
| unsloth/gemma-4-E4B-it-GGUF | mtp-BF16, mtp-F16, mtp-Q8_0, BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, ... (25 files) | 21 supported, 4 draft | 21 ok, 4 draft |  |
| unsloth/gemma-4-31B-it-GGUF | BF16, mtp-BF16, mtp-F16, mtp-Q8_0, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, ... (26 files) | 22 supported, 4 draft | 22 ok, 4 draft |  |
| google/gemma-4-E4B-it-qat-q4_0-gguf | gemma-4-E4B_q4_0-it | supported | ok | q4_0: identical (Vulkan, CPU) |
| google/gemma-4-12B-it-qat-q4_0-gguf | (base) | supported | ok |  |
| unsloth/Qwen3.8-27B-GGUF | BF16, Q4_0, Q4_1, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_S, UD-IQ2_XS, UD-IQ2_XXS, UD-IQ3_S, ... (28 files) | 26 supported, 1 no arch, 1 draft | 26 ok, 1 no arch, 1 draft |  |
| unsloth/Qwen3.6-35B-A3B-GGUF | BF16, MXFP4_MOE, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, ... (24 files) | 24 supported | 24 ok |  |
| unsloth/Qwen3.6-35B-A3B-MTP-GGUF | BF16, MXFP4_MOE, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, ... (22 files) | 22 supported | 22 ok |  |
| unsloth/Qwen3-4B-Instruct-2507-GGUF | F16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (26 files) | 26 supported | 26 ok | UD-IQ1_S: CPU identical, Vulkan differs; UD-Q2_K_XL: both differ late (precision) |
| unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF | BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (27 files) | 27 supported | 27 ok | UD-TQ1_0: CPU identical; Vulkan differs (precision) |
| unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF | BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (27 files) | 27 supported | 27 ok |  |
| unsloth/Qwen3-Coder-Next-GGUF | Q4_K_M, UD-Q4_K_M, UD-IQ2_M, UD-IQ2_XXS | 4 supported | 4 ok |  |
| unsloth/Qwen3.8-Flash-Next-GGUF | UD-Q4_K_XL, UD-IQ1_M, UD-IQ1_S, UD-Q2_K_XL | 4 supported | 4 ok |  |
| unsloth/gpt-oss-20b-GGUF | F16, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, ... (16 files) | 16 supported | 16 ok |  |
| unsloth/gpt-oss-120b-GGUF | F16, Q4_K_M | 2 supported | 2 ok |  |
| unsloth/GLM-4.7-Flash-GGUF | BF16, IQ4_NL, IQ4_XS, MXFP4_MOE, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, ... (28 files) | 28 supported | 28 ok | UD-TQ1_0: identical (Vulkan, CPU) |
| unsloth/GLM-5.3-Flash-GGUF | UD-Q2_K_XL, UD-IQ1_M | 2 supported | 2 ok |  |
| unsloth/DeepSeek-V4-Flash-0731-GGUF | UD-Q2_K_XL | supported | ok |  |
| unsloth/DeepSeek-V4.1-Flash-1017-GGUF | UD-IQ1_S, UD-Q2_K_XL | 2 supported | 2 ok |  |
| unsloth/Kimi-K3-GGUF | UD-IQ1_M, UD-IQ1_S, UD-IQ2_XXS, UD-Q1_0, UD-Q2_K_XL, UD-Q4_K_XL, UD-Q8_K_XL, UD-TQ1_0, UD-TQ2_0 | 9 supported | 9 ok (types 64–66 read as IQ1_S) |  |
| unsloth/LFM2.5-24B-A2B-GGUF | BF16, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL | 10 supported | 10 ok |  |
| unsloth/LFM2.5-8B-A1B-GGUF | BF16, MXFP4_MOE, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, ... (22 files) | 22 supported | 22 ok | MXFP4_MOE: Vulkan identical, CPU differs; UD-IQ2_XXS: both differ late (precision) |
| unsloth/flint-GGUF | BF16, Q8_0, UD-IQ1_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL | 10 supported | 10 ok |  |
| unsloth/Muse-Glimmer-30B-GGUF | BF16, Q8_0, UD-IQ2_M, UD-IQ2_XS, UD-IQ2_XXS, UD-IQ3_M, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, ... (16 files) | 15 supported, 1 draft | 15 ok, 1 draft | dflash-kquant: drafter of UD-Q4_K_XL, greedy identical to llama.cpp with speculation, 58.5 % accepted (llama.cpp 66.1 %) |
| unsloth/North-Mini-Code-1.0-GGUF | BF16, MXFP4_MOE, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, ... (22 files) | 22 supported | 22 ok |  |
| unsloth/Inkling-Small-GGUF | BF16, MXFP4_MOE, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, ... (23 files) | 23 supported | 23 ok | none under 20 GB (smallest 75 GB) |
| unsloth/Inkling-2-Small-GGUF | BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q8_K_XL | 9 arch unsupported (inkling2) | 9 arch unsupported |  |
| unsloth/Mistral-Large-4-1T-A52B-GGUF | UD-IQ1_S | supported | ok |  |
| unsloth/Muse-Spark-Open-816B-A42B-GGUF | UD-IQ1_S | supported | ok |  |
| unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF | BF16, IQ2_XXS, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, ... (14 files) | 13 supported, 1 draft | 13 ok, 1 draft | IQ2_XXS (NVFP4 `ssm_in` / `ssm_out` with scales) imports since the HF quantization round; dflash-NVFP4: draft model with a LiLiCorr head (not supported) |
| unsloth/Llama-3.2-3B-Instruct-GGUF | BF16, F16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, ... (27 files) | 27 supported | 27 ok |  |
| unsloth/Ministral-3-3B-Reasoning-2512-GGUF | BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (26 files) | 26 supported | 26 ok | UD-IQ3_XXS: identical (Vulkan, CPU) |
| unsloth/Qwen3-VL-4B-Instruct-GGUF | BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (26 files) | 26 supported | 26 ok | UD-IQ2_XXS: identical (Vulkan, CPU) |
| unsloth/Qwen2.5-VL-7B-Instruct-GGUF | BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (26 files) | 26 supported | 26 ok |  |
| unsloth/embeddinggemma-2-GGUF | BF16, F16, Q8_0, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL | 6 supported | 6 ok |  |
| unsloth/gemma-3-4b-it-GGUF | BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, ... (26 files) | 26 supported | 26 ok | UD-IQ2_M: Vulkan identical; CPU differs (precision) |
| unsloth/MiniMax-H3-GGUF | minimax_h3_fl2va_pruned-Q2_K, minimax_h3_fl2va_pruned-Q3_K, minimax_h3_fl2va_pruned-Q4_K, ... (16 files) | 16 no arch | 16 no arch |  |
| bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, ... (23 files) | 22 supported, 1 no arch | 22 ok, 1 no arch | IQ2_M: CPU identical; Vulkan differs (precision) |
| bartowski/Qwen2.5-7B-Instruct-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q3_K_XL, Q4_0, ... (24 files) | 24 supported | 24 ok |  |
| bartowski/thomsonreuters_Thomson-1.0-Small-GGUF | IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, ... (28 files) | 27 supported, 1 no arch | 27 ok, 1 no arch |  |
| bartowski/Meta-Llama-3.1-8B-Instruct-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q3_K_XL, ... (24 files) | 24 supported | 24 ok |  |
| bartowski/Qwen_Qwen3.6-35B-A3B-GGUF | IQ1_M, IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, ... (31 files) | 28 supported, 2 draft, 1 no arch | 28 ok, 2 draft, 1 no arch |  |
| bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF | IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, ... (28 files) | 27 supported, 1 no arch | 27 ok, 1 no arch |  |
| bartowski/Qwen_Qwen3-0.6B-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, ... (24 files) | 24 supported | 24 ok | IQ2_M: Vulkan identical; CPU differs (precision) |
| bartowski/Llama-3.2-1B-Instruct-GGUF | IQ3_M, IQ4_XS, Q3_K_L, Q3_K_XL, Q4_0, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, Q4_K_L, Q4_K_M, Q4_K_S, ... (18 files) | 18 supported | 18 ok | Q4_0: identical (Vulkan, CPU); Q4_0_4_4 / Q4_0_8_8: CPU identical, Vulkan differs late (precision) |
| bartowski/Llama-3.2-3B-Instruct-GGUF | IQ3_M, IQ4_XS, Q3_K_L, Q3_K_XL, Q4_0, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, Q4_K_L, Q4_K_M, Q4_K_S, ... (18 files) | 18 supported | 18 ok |  |
| bartowski/Nanbeige_Nanbeige4.3-33B-GGUF | IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q3_K_L, ... (26 files) | 25 supported, 1 no arch | 25 ok, 1 no arch |  |
| bartowski/microsoft_Phi-4-mini-instruct-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, ... (23 files) | 23 supported | 23 ok | IQ4_NL: identical (Vulkan, CPU) |
| bartowski/HuggingFaceTB_SmolLM3-3B-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, ... (24 files) | 24 supported | 24 ok | IQ3_XS: Vulkan identical; CPU differs (precision) |
| bartowski/gemma-2-2b-it-GGUF | IQ3_M, IQ4_XS, Q3_K_L, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, f32 | 11 supported | 11 ok |  |
| bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF | IQ2_M, IQ2_S, IQ2_XS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_K_L, ... (22 files) | 22 supported | 22 ok | IQ2_XS: identical (Vulkan, CPU), after the decoder fix |
| bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF | IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, ... (30 files) | 27 supported, 2 draft, 1 no arch | 27 ok, 2 draft, 1 no arch |  |
| bartowski/LiquidAI_LFM2.5-1.2B-Instruct-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, ... (25 files) | 24 supported, 1 no arch | 24 ok, 1 no arch | IQ3_XXS: identical (Vulkan, CPU) |
| bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF | IQ2_M, IQ3_M, IQ3_XS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q3_K_XL, Q4_0, ... (24 files) | 24 supported | 24 ok |  |
| ggml-org/gemma-3-1b-it-GGUF | Q4_K_M, Q8_0, f16 | 3 supported | 3 ok |  |
| ggml-org/gpt-oss-20b-GGUF | eagle3-BF16, eagle3-Q8_0, MXFP4 | 2 draft, 1 supported | 2 draft, 1 ok |  |
| ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF | BF16, Q4_K_M, Q8_0 | 3 supported | 3 ok |  |
| ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF | BF16, Q4_0, Q8_0, mtp-BF16, mtp-Q4_0, mtp-Q8_0 | 3 supported, 3 draft | 3 ok, 3 draft |  |
| ggml-org/Clef-Flash-GGUF | BF16, Q4_K_M, Q8_0 | 3 supported | 3 ok |  |
| ggml-org/Step-3.5-Flash-GGUF | Q4_K | supported | ok |  |
| ggml-org/GLM-4.6V-GGUF | Q4_K_M | supported | ok |  |
| ggml-org/GLM-OCR-GGUF | Q8_0, f16 | 2 supported | 2 ok |  |
| ggml-org/Qwen2.5-VL-3B-Instruct-GGUF | Q4_K_M, Q8_0, f16 | 3 supported | 3 ok |  |
| ggml-org/Qwen2.5-Omni-3B-GGUF | Q4_K_M, Q8_0, f16 | 3 supported | 3 ok |  |
| ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF | Q4_K_M, Q8_0, bf16 | 3 supported | 3 ok |  |
| ggml-org/LFM3-Omni-2B-GGUF | BF16, Q4_K_M, Q8_0 | 3 supported | 3 ok |  |
| ggml-org/embeddinggemma-300M-GGUF | Q8_0 | supported | ok |  |
| ggml-org/Flint-GGUF | BF16, Q4_K_M, Q8_0 | 3 supported | 3 ok |  |
| ggml-org/SmolLM3-3B-GGUF | Q4_K_M, Q8_0, f16 | 3 supported | 3 ok |  |
| ggml-org/bge-m3-Q8_0-GGUF | (base) | supported | ok |  |
| ggml-org/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF | BF16, NVFP4, Q4_0, Q8_0, lilicorr-NVFP4, mtp-BF16, mtp-Q4_0, mtp-Q8_0 | 4 supported, 4 draft | 4 ok, 4 draft | NVFP4 (per-tensor scales) imports since the HF quantization round; lilicorr-NVFP4: draft model with a LiLiCorr head (not supported) |
| Qwen/Qwen3-VL-8B-Instruct-GGUF | F16, Q4_K_M, Q8_0 | 3 supported | 3 ok |  |
| LiquidAI/LFM3-1.7B-GGUF | BF16, F16, NVFP4, Q4_K_M, Q5_K_M, Q6_K, Q8_0, QAD-Q4_0 | 8 supported | 8 ok | QAD-Q4_0: both differ late (precision); NVFP4 (per-tensor scales) imports since the HF quantization round |
| ornith-ai/Ornith-1.5-9B-GGUF | BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0 | 5 supported | 5 ok |  |
| ornith-ai/Ornith-1.5-35B-A3B-GGUF | BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0 | 5 supported | 5 ok |  |
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF | IQ2_XS, IQ3_S, IQ3_XXS, Q2_0 | 4 supported | 4 ok |  |
| ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF | IQ1_M, IQ2_XS, IQ2_XXS, DeepSeek-V4.1-Flash-Engram-IQ4_NL | 3 shard missing, 1 no arch | 3 shard missing, 1 no arch |  |
| huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF | Huihui-Qwen3.8-Flash-Next-abliterated-patch-all, Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS, ... (11 files) | 9 supported, 2 no arch | 9 ok, 2 no arch |  |
| cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF | (base) | supported | ok |  |
| prism-ml/Ternary-Bonsai-2-9B-gguf | DFlash2-Q8_0, PQ2_0, PQ2_0-MTP-Q8_0, PTQ1_0, Q2_0 | 4 supported, 1 draft | 4 ok, 1 draft | DFlash2-Q8_0: draft model, maps completely; its `qwen35` target does not export taps yet |
| mixedbread-ai/mxbai-embed-large-v1 | f16 | supported | ok |  |
| huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF | ggml-model-Q2_K, ggml-model-Q3_K, ggml-model-Q4_K, ggml-model-Q5_K, ggml-model-Q6_K, ... (12 files) | 7 supported, 5 no arch | 7 ok, 5 no arch |  |
| empero-ai/Qwen3.8-35B-A3B-Distill-GGUF | BF16, IQ2_M, IQ3_M, IQ4_XS, Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 | 10 supported | 10 ok |  |
| agentionai/Qwen3.8-Flash-Next-Gyro-GGUF | L-TQ3_0, L-TQ3_0-noPLE, M-TQ2_0, M-TQ2_0-noPLE, S-TQ1_0, S-TQ1_0-noPLE, mtp-Qwen3.8-Flash-Next-draft | 3 supported, 3 separate n-gram table (-noPLE), 1 draft | 3 ok (trellis experts unfolded to F16), 3 separate n-gram table, 1 draft |  |
| Qwen/Qwen3-0.6B | safetensors | supported | ok |  |
| Qwen/Qwen3-8B | safetensors | supported | ok |  |
| Qwen/Qwen3-14B | safetensors | supported | ok |  |
| Qwen/Qwen2.5-7B-Instruct | safetensors | supported | ok |  |
| Qwen/Qwen2.5-0.5B-Instruct | safetensors | supported | ok |  |
| Qwen/Qwen3.5-0.8B | safetensors | supported | ok |  |
| Qwen/Qwen3.5-9B | safetensors | supported | ok |  |
| Qwen/Qwen3.8-27B | safetensors | supported | ok |  |
| Qwen/Qwen3.8-27B-FP8 | safetensors | supported | ok |  |
| Qwen/Qwen3.6-35B-A3B | safetensors | supported | ok |  |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | safetensors | supported | ok |  |
| Qwen/Qwen3-30B-A3B | safetensors | supported | ok |  |
| Qwen/Qwen3.8-Flash-Next | safetensors | supported | ok |  |
| Qwen/Qwen3.8-Flash-Next-FP8 | safetensors | supported | ok |  |
| Qwen/Qwen3.8-2.4T-A95B | safetensors | supported | ok |  |
| meta-llama/Llama-3.1-8B-Instruct | safetensors | - | gated |  |
| meta-llama/Llama-3.2-1B-Instruct | safetensors | - | gated |  |
| HuggingFaceTB/SmolLM2-135M | safetensors | supported | ok |  |
| openbmb/MiniCPM5-2B | safetensors | supported | ok |  |
| mistralai/Mistral-7B-Instruct-v0.2 | safetensors | supported | ok |  |
| google/gemma-4-31B-it | safetensors | supported | ok |  |
| google/gemma-4-26B-A4B-it | safetensors | supported | ok |  |
| google/medgemma-1.5-4b-it | safetensors | - | gated |  |
| google/paligemma-3b-pt-224 | safetensors | partial (round 11) | ok |  |
| nvidia/Gemma-4-26B-A4B-NVFP4 | safetensors | supported (modelopt NVFP4) | ok | since the HF quantization round |
| deepseek-ai/DeepSeek-V3.2 | safetensors | supported | ok |  |
| deepseek-ai/DeepSeek-V4-Flash-0731 | safetensors | supported | ok | since the HF quantization round (E8M0 scales, FP4 experts) |
| deepseek-ai/DeepSeek-V4.1-Flash | safetensors | model_type unsupported | model_type unsupported |  |
| deepseek-ai/DeepSeek-R1 | safetensors | supported | ok |  |
| zai-org/GLM-5.3-Flash | safetensors | supported | ok |  |
| zai-org/GLM-4.7-Flash | safetensors | supported | ok |  |
| zai-org/GLM-4.5-Air | safetensors | supported | ok |  |
| zai-org/GLM-5.3 | safetensors | supported | ok |  |
| zai-org/GLM-5 | safetensors | supported | ok |  |
| openai/gpt-oss-20b | safetensors | supported | ok |  |
| openai/gpt-oss-120b | safetensors | supported | ok |  |
| moonshotai/Kimi-K3 | safetensors | supported (compressed-tensors mxfp4) | ok | since the HF quantization round (quantization config in `text_config`) |
| moonshotai/Kimi-K3.1-Pro | safetensors | supported (compressed-tensors mxfp4) | ok | since the HF quantization round (quantization config in `text_config`) |
| MiniMaxAI/MiniMax-M2.5 | safetensors | supported | ok |  |
| MiniMaxAI/MiniMax-M3.1-Flash | safetensors | quantization: mxfp8 | unmapped tensors |  |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | safetensors | supported (fp8; round 11) | unsupported: FP8 scale layout of the fused `qkv_proj` (and MXFP4 experts) |  |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | safetensors | supported | ok |  |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | safetensors | supported | ok |  |
| nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | safetensors | supported | ok |  |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | safetensors | quantization: modelopt MIXED_PRECISION (model type mapped in round 11) | unsupported: UnsupportedWeightFormatException |  |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB | safetensors | quantization: modelopt MIXED_PRECISION (model type mapped in round 11) | unsupported: UnsupportedWeightFormatException |  |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | safetensors | model_type unsupported | model_type unsupported |  |
| tencent/Hy4-preview | safetensors | supported | ok |  |
| thinkingmachines/Inkling | safetensors | supported | ok |  |
| thinkingmachines/Inkling-2-Small | safetensors | model_type unsupported (inkling2) | model_type unsupported |  |
| LiquidAI/LFM3-1.7B | safetensors | supported (round 11) | ok | bf16 vs transformers: 16 greedy ids identical, CPU and Vulkan |
| LiquidAI/d1-2B | safetensors | model_type unsupported | model_type unsupported |  |
| baidu/ERNIE-5.5-Flash-Base | safetensors | model_type unsupported | model_type unsupported |  |
| baidu/Unlimited-OCR | safetensors | supported | ok |  |
| baidu/Unlimited-OCR-2 | safetensors | model_type unsupported | model_type unsupported |  |
| PaddlePaddle/PaddleOCR-VL-1.6 | safetensors | supported | ok |  |
| nvidia/Cosmos-Reason2-2B | safetensors | - | gated |  |
| meta-models/Muse-Glimmer-30B | safetensors | supported | ok |  |
| RecursiveSI/flint | safetensors | supported | ok |  |
| XHToken/Spark-X2.6-4B | safetensors | supported (round 11) | ok | bf16 vs transformers: 16 greedy ids identical, CPU and Vulkan |
| ibm-granite/granite-4.1-3b-llmon-preview | safetensors | supported | ok |  |
| openai-community/gpt2 | safetensors | supported | ok |  |
| EleutherAI/pythia-160m | safetensors | supported | ok |  |
| sensenova/SenseNova-Loop-9B-Preview | safetensors | model_type unsupported | model_type unsupported |  |
| KRAFTON/Raon-1-nano | safetensors | model_type unsupported | model_type unsupported |  |
| MinimaLabs/KLM-1.6 | safetensors | model_type unsupported | model_type unsupported |  |
| translated/Lara-4-Mini-Base | safetensors | - | gated |  |
| microsoft/Mage-VL-Cowork | safetensors | supported (round 11) | ok | bf16 vs transformers: 16 greedy ids identical, CPU and Vulkan |
| meituan-longcat/LongCat-2.5 | safetensors | model_type unsupported | model_type unsupported |  |
| Diffuspace/Clover-35B-A3B-Preview | safetensors | supported (round 11) | ok |  |
| dphn/dolphin-2.9.1-yi-1.5-34b | safetensors | supported | ok |  |
| ornith-ai/Ornith-1.0-35B | safetensors | supported | ok |  |
| farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit | Q4_K_M, Q4_K_M_systemone, Q8_0, Q8_0_systemone | 4 supported | 4 ok |  |
| webAI-Official/TwIL-8B | F16-verbose, Q4_K_M-balanced, Q4_K_M-verbose, Q8_0-verbose | 4 supported | 4 ok |  |

## 11. Bring-up round 11 (2026-11-15): the sweep's open rows

Round 11 worked through the open items of section 10: the 19 Hugging Face `model_type` values the sweep could not map (by Hub popularity), the dflash drafters, inkling and the fork-only ggml types. Quantized formats (NVFP4, GPTQ, AWQ, MXFP4, bitsandbytes, modelopt, the FP8 / FP4 layouts of MiMo-V2.6-Pro and DeepSeek-V4-Flash) belong to the quantization workstream and were left alone.

Every new family has a tiny random checkpoint compared with transformers (logits and greedy ids, CPU and Vulkan), a llama.cpp GGUF compared with llama.cpp 8ca5a80 where llama.cpp has the architecture, the header-only check agreeing with the import, and batched multi-sequence decoding token-identical to single-sequence decoding (`HubSweepFamilyCases`, `HubSweepFamilyServingTests`, `InklingFamilyCases`, `InklingFamilyServingTests`, `SpeculativeDecodingTests.Dflash`).

Hugging Face model types added:

| `model_type` | Checkpoint (sweep row) | How | Tests |
|---|---|---|---|
| `lfm3` | LiquidAI/LFM3-1.7B | the `lfm3` hybrid-conv decoder: sliding layers and a zero-tail partial rope | tiny vs the remote code and llama.cpp; LFM3-1.7B bf16 vs transformers, 16 greedy ids identical on CPU and Vulkan |
| `spark2_5` | XHToken/Spark-X2.6-4B | fused `q_k_v_proj` split into rows | tiny vs the remote code and llama.cpp; Spark-X2.6-4B bf16 vs transformers, 16 ids identical on CPU and Vulkan |
| `clover_moe` | Diffuspace/Clover-35B-A3B-Preview | Qwen3.5-MoE layers, the causal commit pass; the block-diffusion preview tensors are not imported | tiny vs the remote code and llama.cpp (24 ids); the real checkpoint (70 GB) only through the header check |
| `nemotron_h_omni` | Nemotron 3.5 Super-VL, Bolt | the language model below `language_model.`; Bolt's Puzzle `block_configs` (per-layer expert and selection counts) | tiny omni and Puzzle checkpoints vs transformers; the released files are modelopt MIXED_PRECISION, which is not read yet |
| `mimo_v2` | XiaomiMiMo/MiMo-V2.6-Pro-RL | the MiMo-V2-Flash layers with a fused `qkv_proj` (split by the per-layer kv head counts), `swa_num_key_value_heads`; MTP, vision and audio towers skipped | tiny vs the remote code and llama.cpp; the real checkpoint stores its fused `qkv_proj` FP8 scales per part, padded to whole blocks ([216, 48]), and MXFP4 experts: quantization workstream |
| `llava_onevision2` | microsoft/Mage-VL-Cowork | the text decoder (here `qwen3`) below `model.language_model.`; the OneVision tower is not imported | tiny vs the remote code and llama.cpp; Mage-VL-Cowork bf16 vs transformers, 16 ids identical on CPU and Vulkan |
| `paligemma` (partial) | google/paligemma-3b-pt-224 | the Gemma / Gemma 2 text decoder, head_dim defaulting to 256, no final logit soft cap (transformers' PaliGemma head does not apply it), decoded causally | tiny PaliGemma and PaliGemma 2 vs transformers' eager path, f32 KV caches |
| `inkling_mm_model` | thinkingmachines/Inkling | the text decoder on the hybrid stack (section 4 row `inkling`) | tiny vs transformers and the fork's llama.cpp; no file under 20 GB |

PaliGemma is partial on purpose: its processor marks the image and prompt tokens as a bidirectional prefix (`token_type_ids`), and transformers' sdpa path attends bidirectionally over a prompt even without them. Sharplight decodes causally, which equals transformers' eager path for text-only input. A prefix-LM prefill (bidirectional attention over the prompt rows, causal afterwards; the multi-sequence `SequenceRunEndRows` form added for dflash is most of it) is about 1–2 days, the SigLIP tower and projector on the multimodal graph another 1–2.

Also in this round:

- **dflash** (section 4 row, section 10 item 2): plain DFlash and DFlash2 drafters run with their target (`DflashDraftHead`, `BlockDraftSpeculator`). Muse-Glimmer's drafter gives output identical to llama.cpp for 40 greedy tokens at 58.5 % acceptance (llama.cpp 66.1 %). Open: taps from `qwen35` targets for Bonsai (0.5 day), the LiLiCorr head of Bolt and Super-VL (about 2 days) plus `nemotron_h` taps (0.5 day) and NVFP4 per-tensor scales.
- **inkling** (GGUF and HF; section 10 item 3): all 23 Inkling-Small GGUFs pass the check and import. inkling2 is surveyed (section 10 item 3: about 3–4 days).
- **ggml types 64–66 and 144–149** (section 10 item 4): all nine Kimi-K3 GGUFs and three Gyro files import; the Gyro `-noPLE` files are reported as needing their separate n-gram table.

Sweep totals after the round: GGUF 1080 of 1138 files import (29 more: Kimi-K3 3, Inkling-Small 23, Gyro 3) and the four dflash drafters are reported as draft models; Hugging Face 49 of 72 checkpoints import (LFM3-1.7B, Spark-X2.6-4B, Mage-VL-Cowork, Clover, PaliGemma and Inkling added). Super-VL and Bolt now stop at their modelopt quantization and MiMo-V2.6-Pro at its FP8 layout, both quantization work.

Findings:

- The PaliGemma tokenizer has an added non-special token `▁▁` (id 139, `normalized: true`) that transformers decodes as two spaces and Sharplight verbatim, so the sweep's tokenizer round trip differs on it. Not fixed.
- Tiny random models with large weights are chaotic enough for f16 rounding to decide greedy ties: the Clover tiny model uses seed 1 and weight std 0.08, the PaliGemma ones keep their KV caches in f32 (their final norm scales the logits to 16).

Model types still unsupported, with estimates (none is a mapping-only job):

| Checkpoint | What it needs | Estimate |
|---|---|---|
| deepseek-ai/DeepSeek-V4.1-Flash | the HF tensor-name mapping onto the existing `deepseek41` graph; its FP8 weights and FP4 experts are quantization work | 1–1.5 days |
| sensenova/SenseNova-Loop-9B-Preview | a layer loop over layers 4–28 and two-stream hyper-connections on the qwen3_5 hybrid stack (both blocks exist separately) | 2–3 days |
| LiquidAI/d1-2B | the non-causal d1 decision variant with set attention over the options | 2–3 days |
| thinkingmachines/Inkling-2-Small | the inkling2 blocks of section 10 item 3 | 3–4 days |
| baidu/Unlimited-OCR-2 | block-sparse router attention (DASH / SR-SWA) with entmax routing and a 2-D page rope | 4–5 days |
| MinimaLabs/KLM-1.6 | GatedDeltaNet-2 channel gates, full attention residuals with a memory source, exact n-gram tables, value residual | 4–5 days |
| KRAFTON/Raon-1-nano | NoPE latent attention with a gate, RDA, four-stream hyper-connections, PLE | 4–6 days |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | an nGPT (normalized) transformer with layer loops; no modeling code is published | not verifiable yet |
| baidu/ERNIE-5.5-Flash-Base | no modeling code is published | not verifiable yet |
| meituan-longcat/LongCat-2.5 | no modeling code is published | not verifiable yet |

## 12. Text to speech (task kind `speech-synthesis`)

Speech synthesizers are `.slm`-storable models of kind `speech-synthesis` run through `ISpeechSynthesizer` (docs/api-overview.md, Tasks). Acoustic models,
codecs and vocoders are graphs of generic Core ops; the host runs only the token loop (the text-generation session and sampler, or Kokoro's length regulator) and the overlap-add.

| Model | Status | Reference | Parity |
|---|---|---|---|
| OuteTTS 0.2 500M + WavTokenizer-Large-75 (GGUF pair) | supported, CPU and Vulkan | llama.cpp 8ca5a80 (`llama-completion`, libllama WavTokenizer graph + numpy iSTFT) | teacher-forced logits: no argmax disagreement beyond a 0.1 margin over 253 positions, worst logit delta 0.32 (CPU) / 0.52 (Vulkan) on logits up to 45 (Q8_0 activation rounding); decoder waveform relative RMS < 2e-3 on both |
| Kokoro-82M v1.0 (hexgrad; StyleTTS 2 style model, ALBERT text encoder, iSTFTNet decoder), all voice packs of the repository (af_heart verified) | supported, CPU and Vulkan | `kokoro` 0.9.4 on torch (CPU), excitation noise of the reference run fed to both | symbols, rounded durations and the F0 curve identical (F0 relative RMS 5e-7 CPU, 4e-7 Vulkan); decoder features before the generator 1.5e-6; conv_post output 5e-4; waveform relative RMS 1.3e-2 (CPU) / 1.3e-2 (Vulkan), which is the reproducibility floor of the sine excitation (the reference itself moves 1.0-1.2 % between an f32 and an f64 phase accumulation) |
| Orpheus 3B fine-tune (Llama decoder over SNAC codes) + SNAC 24 kHz decoder | supported, CPU and Vulkan | transformers (fp32, greedy chain of 140 tokens, teacher forced) and the `snac` package | prompt ids identical; 0 argmax disagreements over 140 positions, worst top-logit delta 0.003 (CPU and Vulkan); SNAC decoder waveform relative RMS 2.1e-6 (CPU) / 2.4e-6 (Vulkan) with the reference's noise |
| CSM-1B + Mimi, Dia-1.6B (+ DAC), qwen3tts, pockettts | not started | - | - |

OuteTTS loading: the language model is the primary file and the WavTokenizer GGUF its `vocoder` auxiliary (`TaskLoadOptions.Auxiliary = [("vocoder", path)]`, `--vocoder` of
`samples/Sharplight.Speak`). `OuteTtsAdapter` is a composite adapter (`ICompositeModelAdapter`, `services.AddCompositeModelAdapter<T>()`): it builds the language model with the adapter of its
architecture, adds the codec weights under `codec.` and the `codec_decode` graph, and describes the model: components `text-model`, `tokenizer`, `codec-decoder`
(config: sample rate, FFT size, hop, vocabulary; symbol bound `T` = 4096 frames). Voices are `voices/<name>.json` attachments of the `text-model` component
(OuteTTS speaker profiles: words with durations and codec tokens); none ship by default.

New generic Core pieces: the `Conv1d` op (time-major `[T, C]`, symbolic `T`, stride / padding / dilation / groups; CPU and a generated Vulkan kernel), `Sharplight.Audio.Window`
and `OverlapAdd` (host side of an inverse STFT; the inverse real DFT and window stay in the graph through the existing `Dft` op), the composite-adapter hook and
`LlmModel.FromTaskModel` made public. Greedy codec chains cannot match llama.cpp token for token: runner-up logits of codec tokens are often within 1e-3 of the winner, so the
check feeds llama.cpp's tokens to both implementations (`OuteTtsTests`). The streaming form decodes each spoken word separately (the decoder attends over all frames, so
this differs slightly from the whole-utterance decode). OuteTTS 0.3 and a default speaker profile are not implemented; llama.cpp 8ca5a80 dropped its OuteTTS demo, so the
reference is the pieces above rather than `llama-tts`.

Kokoro loading: the model is a checkpoint folder (`config.json`, the `.pth` state dictionary, `voices/*.pt`, optionally `lexicon/*.json`); `Sharplight.Import.Kokoro`
(`KokoroImporter`, `services.AddKokoroImporter()`) folds the weight-normalized convolutions (`weight_g` / `weight_v`, a lazily computed effective `weight`), adds the two
LSTM bias vectors of each direction, and stores the vocabulary (`tokenizer/vocab.json`), every voice pack (`voices/{name}.f32`, f32 `[510, 1, 256]`: row `n - 1` is the style
of an utterance of `n` symbols, 128 values for the decoder and 128 for the prosody predictor) and the pronunciation lexicons (`lexicon/{us,gb}_{gold,silver}.json`, the
misaki dictionaries) as attachments, so a `.slm` package carries everything. `KokoroAdapter` (canonical architecture `kokoro`) builds two graphs: `kokoro_text` over the
symbols (ALBERT with one shared layer applied 12 times, duration encoder of bidirectional LSTMs and adaptive layer norms, duration head, convolutional text encoder with an
LSTM) and `kokoro_audio` over the decoder frames (shared LSTM, F0 and noise predictors, the decoder's adaptive residual blocks, the iSTFTNet generator with its
harmonic-plus-noise source, STFT of the source as a convolution, snake residual blocks, and the frequency-domain half of the inverse STFT). The host (`KokoroSynthesizer`)
rounds the predicted durations, expands the features to frames (the length regulator is a row copy), draws the excitation noise (a seeded generator, `Sampling.Seed`) and
overlap-adds the frames (`OverlapAdd`, `center=True` trimming). The English front end (`KokoroPhonemizer`) follows the structure of misaki, the front end the model was
trained with: sentence splitting, the gold and silver lexicons with plural, past tense, progressive and possessive endings, the article before a vowel, numbers read as
words, letter-to-sound rules for unknown words, punctuation passed through; it does not do misaki's part-of-speech dependent choices (heteronyms, the stress of function
words), so unusual sentences differ from misaki slightly (`Language = "phonemes"` bypasses it and takes Kokoro phoneme symbols). British voices (`b*`) use the `gb`
lexicons; other languages need a front end that is not implemented. `SpeechSynthesisOptions.Speed` divides the predicted durations; streaming yields one buffer per sentence
(a sentence of more than 510 symbols is split at word boundaries).

Orpheus loading: the language model is the primary source (a Hugging Face Llama checkpoint, here the `unsloth/orpheus-3b-0.1-ft` mirror of the gated canopylabs release) and
the SNAC folder its `vocoder` auxiliary (`TaskLoadOptions.Auxiliary = [("vocoder", snacFolder)]`, `--vocoder` of `samples/Sharplight.Speak`). `OrpheusAdapter` is a
composite adapter like OuteTTS: the language model is built by the adapter of its architecture, `Sharplight.Import.Snac` (`SnacImporter`, `AddSnacImporter()`; priority above
the Hugging Face importer, which also matches the folder) gives the decoding half of the codec with the weight norms folded, and `SnacGraph` is the decoder as a graph of
generic ops (codebook lookups, 1x1 projections, nearest-neighbor upsampling to the finest rate, depthwise convolutions, snake activation, transposed convolutions, the
noise blocks, three dilated residual units per block, tanh). `OrpheusSynthesizer` builds the prompt `[start of human] "{voice}: {text}" [end of text, end of human]` for the
voices tara, leah, jess, leo, dan, mia, zac and zoe, generates through the text-generation machinery, regroups the audio tokens (seven per frame: one coarse, two medium,
four fine codes, each offset by its slot) and decodes; the streaming form decodes every 8 frames with 2 frames of context. The SNAC noise blocks draw random numbers: they
are inputs of the graph (seeded by default) so a decode is reproducible. The decode of one request covers at most 1024 frames (85 s).

New generic Core ops of this round (CPU and Vulkan, time-major `[T, C]` with a symbolic `T`): `ConvTranspose1d` (stride, crop, output padding, groups), `Interpolate1d`
(nearest and half-pixel linear resampling by a rational factor, with exact integer index arithmetic), `Lstm` (one direction's recurrence over precomputed input
projections; on the device one workgroup walks the steps), `Atan2` (IEEE zero handling on both providers), reflect padding for `Conv1d`, and `CumSum` accumulating in double
on the CPU and with compensated summation on the device. The angle of a complex number uses `Where` so the phase of an exactly-zero imaginary part (DC and Nyquist bins) does
not depend on the sign of that zero. `Sharplight.Import.PyTorch` reads a file written by `torch.save(tensor)` and has the shared weight-norm helpers (`TorchWeightNorm`).
Not done: CSM (the depth decoder over 32 codebooks and the Mimi decoder), Dia (encoder-decoder with delay-pattern codebook heads and the DAC codec), qwen3tts and pockettts
(no GGUFs were looked for), and OuteTTS 0.3.

## 13. Speech to text (task kind `speech-recognition`)

Speech-to-text models are the first non-LLM kind on the task layer (`ModelKinds.SpeechRecognition`, docs/api-overview.md "Tasks").
They run through the same engine as everything else: the encoder and decoder are graphs of generic Core ops built by an architecture
adapter, the model knowledge lives in the importers, and only the feature extraction and the decoding loop run on the host.

| Family | Formats | Status | Verified against | Test assets |
|---|---|---|---|---|
| Whisper tiny, base, small (and `.en`), large-v3-turbo; medium, large-v1/v2/v3 map the same way but are not verified yet | Hugging Face `whisper` checkpoints; whisper.cpp ggml `.bin` (f32, f16, q4_0/q4_1/q5_0/q5_1/q8_0, k-quants); `.slm` | **supported** (CPU and Vulkan) | transformers (log-mel, encoder output, token-exact greedy ids with and without timestamps, language detection) and whisper.cpp 5922450f CPU build (segments and texts) | hf/whisper-{tiny,tiny.en,base,small}, whisper.cpp/ggml-{tiny,tiny.en,base,base.en,small,small.en}[-q5_1].bin, audio/jfk.wav |
| Whisper as GGUF (any size; tested tiny, base, large-v3-turbo) | transcribe.cpp GGUF (`general.architecture = whisper`, `stt.*` keys, `enc.*` / `dec.*` tensors: the handy-computer/whisper-*-gguf repositories, F32, F16, Q8_0, Q4_K_M ...); whisper.cpp ggml files published with a `.gguf` extension (oxide-lab/whisper-large-v3-turbo-GGUF `whisper.cpp/`) | **supported** (CPU and Vulkan) | whisper.cpp (the same model's ggml file): tiny F16 / F32 segments and texts identical, quantized files word for word; imports to the whisper.cpp file's canonical names, shapes, decoding keys, suppress lists and filter bank | whisper-gguf/handy-computer/whisper-{tiny-{F16,F32,Q8_0,Q4_K_M},base-Q4_K_M,large-v3-turbo-{Q8_0,Q4_K_M}}.gguf, whisper-gguf/oxide-lab/whisper.cpp/whisper-large-v3-turbo-q5_0.gguf |
| Whisper as Candle GGUF (oxide-lab root files, xkeyC) | GGUF with Hugging Face tensor names under `model.`, tokenizer and configuration in separate JSON files | **rejected by name** ("Candle whisper GGUF ... not supported; use the whisper.cpp or transcribe.cpp files") | - | whisper-gguf/oxide-lab/whisper-large-v3-turbo-q4_0.gguf, whisper-gguf/xkeyC/model_q4_k.gguf |
| Distil-Whisper (distil-small.en, distil-large-v3.5) | Hugging Face (`whisper` model type; f16 `model.safetensors` beside an `fp32` variant) | **supported** (CPU and Vulkan; same graph, 2-layer decoders) | transformers (encoder output, token-exact greedy ids with and without timestamps, beam search) | hf/distil-small.en, hf/distil-large-v3.5 |
| Moonshine tiny, base | Hugging Face `moonshine`; `.slm` | **supported** (CPU and Vulkan) | transformers (encoder output, token-exact greedy ids) | hf/moonshine-{tiny,base}, audio/jfk.wav, audio/librispeech-1272-128104-0004.wav |
| Parakeet CTC (parakeet-ctc-0.6b; 1.1b maps the same way) and TDT (parakeet-tdt-0.6b-v3); RNN-T maps but is not verified | Hugging Face `parakeet_ctc`, `parakeet_tdt`, `parakeet_rnnt` (transformers format); `.slm` | **supported** (CPU and Vulkan) | transformers (NeMo log-mel features, encoder output, token-exact greedy CTC / TDT ids) | hf/parakeet-ctc-0.6b, hf/parakeet-tdt-0.6b-v3 |
| Canary (canary-1b-v2) | Hugging Face `canary` (transformers format); `.slm` | **supported** (CPU and Vulkan; transcription and translation prompts, no timestamps) | transformers (encoder output, token-exact greedy ids of an English transcription with punctuation) | hf/canary-1b-v2 |
| Voxtral | - | **missing** (needs the tekken tokenizer and audio embeddings in front of the text generator) | - | - |

How it is built (2026-11-15):

- **Model**: canonical architecture `encoder_decoder` with `encoder.input` "audio" (`EncoderDecoderAdapter`): the encode graph takes one
  window of log-mel features [mel bins, 3000] (`InputRole.AudioFeatures`) and runs the `AudioConvStem` (two generic time-major
  `Conv1d` ops, the op the text-to-speech codecs use, with GELU, then learned positions) and pre-norm LayerNorm layers; the decode graph has a KV cache for self-attention and caches the
  cross-attention keys and values per window (`cross_attention.cached`: projected once on the window's first pass, E = 1500, later
  passes carry no source rows). The adapter sets the model description (`SpeechRecognitionDescription`): components `encoder`,
  `decoder` (row and context bound 448), `tokenizer` (the canonical `TokenizerData` JSON as the attachment `tokenizer/tokenizer.json`)
  and `feature-extractor` (log-mel configuration, filter bank as the attachment `feature-extractor/mel-filters`).
- **Importers**: `Sharplight.Import.HuggingFace` (`WhisperHfMapping`: filter bank computed as transformers' feature extractor does,
  speech keys from `generation_config.json`) and the new `Sharplight.Import.WhisperCpp` (`WhisperCppImporter`: the legacy ggml
  container, weights zero-copy from the memory map, the stored filter bank, and the vocabulary rebuilt into a byte-level BPE tokenizer
  whose merges are recovered from the token ranks, plus Whisper's special tokens and non-speech suppression list; the rebuilt tokenizer
  encodes exactly like the transformers tokenizer and yields the same suppress lists).
- **Host**: `LogMelSpectrogram` (feature-extractor type `log-mel`, behind the internal `IAudioFeatureExtractor`; a graph-based
  extractor on the generic STFT op that `wip/onnx-ops` adds can replace it) and `EncoderDecoderSpeechRecognizer`: language detection,
  transcribe / translate, greedy, sampled or beam-search decoding with the timestamp rules (`SpeechRecognitionOptions.BeamSize`;
  `SpeechBeamSearch` follows transformers' `num_beams` selection and length normalization, and evaluates the beams on the single KV
  cache by rewinding it to each beam's shared prefix), no-speech detection, and long-form transcription that advances window by window
  to the last complete timestamp segment (optionally conditioned on the previous text). Not yet: temperature fallback, word timestamps
  (cross-attention alignment heads).
- GGUF (2026-11-15): `WhisperGgufMapping` in `Sharplight.Import.Gguf` reads transcribe.cpp's self-describing layout (the
  `whisper-*-gguf` repositories with the most Whisper downloads on the Hub): hyper-parameters from `stt.whisper.*`, the log-mel
  settings from `stt.frontend.*`, the filter bank tensor `frontend.mel_filterbank` as the `feature-extractor/mel-filters` attachment
  (GGUF mappings can store tensors as attachments, `IGgufArchitectureMapping.MapAttachmentName`), the stored BPE vocabulary and merges
  (every id from end-of-text on typed as control), the special token ids and the transformers suppress lists. whisper.cpp ggml files
  renamed to `.gguf` open through `WhisperCppImporter` (it claims files by their magic, not their extension). Candle's whisper GGUF
  files are rejected with a message that names them.
- large-v3-turbo on jfk.wav: whisper.cpp f16, q5_0 and q8_0 files, the transcribe.cpp Q8_0 / Q4_K_M files and oxide-lab's whisper.cpp
  q5_0 `.gguf` all give whisper.cpp's transcript ("And so, my fellow Americans, ask not ...").
- Hugging Face checkpoints that ship weight variants next to `model.safetensors` (`model.fp32.safetensors`, ...) load the default file
  only, as transformers does.

Round 2 (2026-11-16): the other speech families on the same engine and task client.

- **Moonshine** (`MoonshineHfMapping` → `encoder_decoder`): a variable-length audio encoder over the raw waveform (feature extractor
  `waveform`): the `AudioConvStem` generalized to unpadded convolutions with per-convolution activations and a one-group signal norm, rotary
  attention (`attention.rope.*`, GPT-J pairs on the first 90 % / 62 % of each head) in the encoder and the decoder's self-attention, a
  SwiGLU decoder MLP (the fused `fc1` split), bias-free LayerNorms, and a decoder prompted with the start token only. The checkpoint's
  head-dimension padding only aligns the reference kernels and is not mapped. Windows are up to 30 s and not padded; the decoder's cross
  states hold the rows of the longest window and the decode graph's `RowSpans` input limits the cross-attention to the current window.
- **Parakeet** (`ParakeetHfMapping` → `conformer_asr`, `ConformerAsrAdapter` + `ConformerEncoder`): NeMo's front end on the host
  (`log-mel` with pre-emphasis, a symmetric 400-sample window, ln(mel + 2⁻²⁴), per-channel normalization), the 8× depthwise-separable
  subsampling on the new generic `SequenceConv2d` op (2-D convolution over a time-major feature map whose time axis varies per run),
  conformer blocks with Transformer-XL relative-position attention on the new generic `RelativeAttention` op (CPU and Vulkan), batch norm
  folded into a scale and shift at load, the depthwise convolution on `Conv1d` with groups. CTC models output the logits from the encode
  graph; transducers get the `predict` graph (embedding + LSTM layers from generic ops, the states kept by the host) and the `joint` graph.
  The decoding runs on the host in Core (`FrameSpeechRecognizer`): CTC greedy (merge repeats, drop blanks) or a prefix beam search
  (`BeamSize`), greedy RNN-T and TDT (duration-aware row advance, NeMo's symbols-per-step guard). Each window becomes one segment with the
  times of its first and last token; `TimestampGranularity.Word` gives word times from the token rows (80 ms per encoder row).
- **Canary** (`CanaryHfMapping` → `encoder_decoder` with `encoder.type` "conformer"): the Parakeet encoder in front of the pre-norm
  transformer decoder (NeMo's fixed sinusoidal positions as the position table, an embedding LayerNorm), prompted with the checkpoint's
  template (`speech.prompt_template_ids`: start of context, start of transcript, emotion, source and target language, punctuation, no ITN,
  no timestamps, no diarization); `Translate` sets the target language to English; requests without a language use English
  (`speech.default_language`, Canary has no language detection).
- Recordings longer than a window (30 s) are cut into consecutive windows for the window-length families (Moonshine, Parakeet, Canary)
  without overlap; Whisper keeps its timestamp-driven long-form loop.

Parity of round 2 (fixtures from `tools/make_{moonshine,parakeet,canary}_fixtures.py`; jfk.wav and the 29.4 s LibriSpeech dev-clean
utterance 1272-128104-0004, each one window):

| Check | Result |
|---|---|
| greedy ids vs transformers `generate`: moonshine-tiny, moonshine-base | identical, CPU and Vulkan, both clips |
| greedy ids vs transformers: parakeet-ctc-0.6b (CTC), parakeet-tdt-0.6b-v3 (TDT) | identical, CPU and Vulkan, both clips |
| greedy ids vs transformers: canary-1b-v2 (English transcription with punctuation) | identical, CPU and Vulkan, both clips |
| NeMo log-mel features vs transformers' `ParakeetFeatureExtractor` | max abs error 6e-5 (values up to 3.4) |
| encoder output, Moonshine | ≤ 3.3e-5 relative (CPU: f32 sums of the 127-tap waveform convolution in another order), Vulkan ≤ 1e-5 |
| encoder output, Parakeet and Canary (24 / 32 conformer blocks) | ≤ 2.5e-5 relative, CPU and Vulkan |

Parity on jfk.wav (11 s):

| Check | Result |
|---|---|
| log-mel features vs transformers (`WhisperFeatureExtractor`) | max abs error 3.8e-6 (values up to 1.33) |
| encoder output vs transformers, f32 checkpoints | max abs error 1.1e-5 to 1.4e-4 for values up to 10 to 28 (≤ 5e-6 relative), CPU and Vulkan |
| greedy ids vs transformers `generate` (tiny, tiny.en, base, small; without and with timestamps) | identical, CPU and Vulkan |
| greedy ids vs transformers, Distil-Whisper (distil-small.en, distil-large-v3.5) | identical, CPU and Vulkan; encoder within 2e-5 relative (distil-small.en's f16 weights on Vulkan: within 2e-3, f16 matrix products) |
| 5-beam search vs transformers `generate(num_beams=5)` (tiny, tiny.en, base, small, distil-small.en; without and with timestamps) | identical ids (tiny, tiny.en and distil-small.en pick different sequences than greedy), CPU and Vulkan |
| language detection (tiny, base) | `en`, then identical ids to transformers without a language |
| whisper.cpp f16 files (6 models): segments with timestamps and `-nt` text | identical to whisper.cpp's CPU build, CPU and Vulkan |
| whisper.cpp q5_1 files (6 models) | same words; 3 of 24 runs differ in one punctuation mark. ggml multiplies quantized weights with activations quantized to 8 bits, Sharplight with the dequantized weights; whisper.cpp's own CPU and CUDA builds disagree on the same commas (e.g. small.en-q5_1) |

Tests: `tests/Sharplight.Speech.Tests` (`MoonshineParityTests`, `ParakeetParityTests`, `CanaryParityTests`, `CtcDecodingTests`,
`WhisperTransformersParityTests`, `WhisperCppParityTests`, `WhisperGgufParityTests`, `WhisperCppVocabularyTests`,
`SpeechPackageTests`: conversion to `.slm`, loading through the DI task API with identical tokens, per-component precompilation;
`FourierTransformTests`), fixtures from `tests/Sharplight.Speech.Tests/tools/make_whisper_fixtures.py`; the audio endpoints in
`Sharplight.Serving.Tests` (`AudioApiTests`); the Studio inspector in `PackageInspectorTests.SpeechPackage_ShowsSpeechRecognitionComponents`; recognition, loading and the Transcribe page for the GGUF and whisper.cpp files in `GgufRealModelTests`.

## 14. Vision models (task layer)

Vision models run through the task layer (kinds `image-classification`, `image-embedding`, `object-detection`, `depth-estimation`; see docs/api-overview.md "Tasks"). The host-side
preprocessing is the neutral `ImageProcessorConfig` stored with the model. Reference outputs come from `tools/vision-reference.py`
(transformers, CPU) on one sample image (`--size 640` for the fixed-canvas detectors); checkpoints live in `~/models/vision/<name>`.

| Model | Importer | Kind | Status | Parity vs transformers (relative rms, CPU / Vulkan) |
|---|---|---|---|---|
| ViT (`google/vit-base-patch16-224`) | HF `vit` | image-classification | supported, same top-1 | 1.2e-3 / 1.2e-3 (logits) |
| DINOv2 (`facebook/dinov2-small`, 518 px table resampled to the 224 px processor grid, LayerScale, registers) | HF `dinov2`, `dinov2_with_registers` | image-embedding | supported | 1.2e-3 / 1.2e-3 (CLS embedding) |
| CLIP (`openai/clip-vit-base-patch32`: image tower with projection, causal text tower pooled at the end-of-text token, CLIP byte-level BPE tokenizer) | HF `clip` | image-embedding (+ text) | supported: `EmbedImageAsync`, `EmbedTextAsync`, `ClassifyZeroShotAsync` | image 2.6e-3, text 1e-6, zero-shot logits 4.5e-4 (identical on both backends) |
| SigLIP (`google/siglip-base-patch16-224`: attention-pooling head, bidirectional text tower padded to 64 tokens, unigram tokenizer with SigLIP's lower-casing and punctuation removal, sigmoid scoring) | HF `siglip` | image-embedding (+ text) | supported, same API | image 2.2e-3, text 1e-6, zero-shot logits 6.3e-4 |
| DETR (`facebook/detr-resnet-50`: timm ResNet-50 with folded frozen batch norm, 6+6 layer transformer, 100 queries) | HF `detr` | object-detection | supported: same four detections, boxes within 2 px, scores within 0.02 | CPU = Vulkan (identical detections) |
| RT-DETR (`PekingU/rtdetr_r18vd`: ResNet-D backbone, hybrid encoder with AIFI and CSP-RepVGG FPN/PAN, top-k query selection with anchors, multi-scale deformable attention through `GridSample`, box refinement) | HF `rt_detr` | object-detection | supported: same five detections | CPU = Vulkan |
| Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`: DINOv2 backbone with four tapped layers, DPT reassemble and feature-fusion neck, convolutional head; relative depth) | HF `depth_anything` | depth-estimation | supported: `IDepthEstimator.EstimateAsync` | depth map 1.5e-4 (CPU = Vulkan) |
| SAM 2 / SAM 2.1 (`facebook/sam2.1-hiera-tiny`: Hiera encoder with windowed attention, q-pooling and global blocks, FPN neck, Fourier prompt encoder, two-way mask decoder with high-resolution skips; image path only) | HF `sam2`, `sam2_video` | image-segmentation | supported: `IImageSegmenter` (`EncodeImageAsync` once, `SegmentAsync` per point / box prompt) | point, points with a negative one and box prompts: all three candidate masks IoU > 0.98 and predicted IoU within 0.02 of transformers, CPU and Vulkan |
| TrOCR (`microsoft/trocr-base-printed`: ViT encoder, post-norm decoder via `encoder_decoder` with `encoder.input` "image") | HF `vision-encoder-decoder` | ocr | supported (`ITextRecognizer.RecognizeAsync`, greedy) | text exactly transformers' ("SHARPLIGHT READS PRINTED TEXT FROM IMAGES."), CPU and Vulkan |
| Real-ESRGAN x4plus (RRDBNet, official `.pth` through the restricted unpickler) | `SuperResolutionImporter` | image-upscaling | supported (`IImageUpscaler.UpscaleAsync`, tiles blended) | whole image > 40 dB PSNR from the reference implementation, tiled run > 40 dB (Vulkan) |
| SwinIR classical and real-world x4 (`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`: Swin blocks with relative position bias and shifted windows, sub-pixel convolutions; `nearest+conv` upsampler for the real-world models) | `SuperResolutionImporter` (also reads the RRDBNet files) | image-upscaling | supported (`IImageUpscaler.UpscaleAsync`; sizes are multiples of the window of 8, the client pads edges by repetition where the reference reflects) | whole image 92.9 dB (Vulkan) / 91.6 dB (CPU) PSNR from the authors' network, tiled 61.6 dB; the real-world variant is built but not verified |
| timm ResNet (`timm/resnet18.a1_in1k`, `resnet34`, `resnet50`, `wide_resnet50_2.racm_in1k`: 7x7 stem, basic or bottleneck blocks, batch norm folded) | HF folder with a timm `config.json` (`timm_resnet`) | image-classification | supported (no label names in the repositories: indices only) | logits within 2e-3 relative rms of timm, same top-1 (CPU and Vulkan) |
| Table Transformer (`microsoft/table-transformer-detection`: DETR with a ResNet-18 backbone, pre-norm layers, final norms) | HF `table-transformer` | object-detection | supported | same detection as transformers on a synthetic table, CPU = Vulkan |
| Mask2Former, DPT (other heads), YOLO / YOLOS, Florence-2, SegFormer | - | - | not started (see the sweep below) | - |

### Vision popularity sweep (2026-11-16)

Models taken from the Hub's download ranking per task (image-classification, zero-shot-image-classification, image-feature-extraction, object-detection,
image-segmentation, depth-estimation, image-to-text, image-to-image), distinct architectures first, run through the task layer on CPU and Vulkan against
transformers / timm / the authors' network (tests: `ImageEncoderTests`, `ObjectDetectorTests`, `TextRecognizerTests`, `ImageUpscalerTests` in `tests/Sharplight.Vision.Tests`, references
from `tools/vision-reference.py`, `tools/timm-reference.py`, `tools/ocr-reference.py`, `tools/swinir-reference.py`). Checkpoints in `~/models/vision-sweep/<name>`.

| Model | Task | Result (relative rms of the compared outputs unless stated; CPU = Vulkan) |
|---|---|---|
| SwinIR classical x4 | upscaling | 92.9 dB (Vulkan) / 91.6 dB (CPU) PSNR whole image, 61.6 dB tiled |
| Falconsai/nsfw_image_detection (ViT) | classification | logits < 2e-3, same top-1 |
| timm resnet18 / resnet34 / resnet50 (a1_in1k), wide_resnet50_2 (racm_in1k) | classification | logits < 2e-3, same top-1 |
| openai/clip-vit-base-patch16 (`pytorch_model.bin` only), clip-vit-large-patch14 | embedding, zero-shot | image 2.1e-3 / 2.4e-3, text 1e-6, zero-shot logits 4.6e-4 / 3.3e-4 |
| laion/CLIP-ViT-B-32-laion2B-s34B-b79K (GELU CLIP) | embedding, zero-shot | image 2.7e-3, zero-shot logits 5.0e-4 |
| google/siglip2-base-patch16-224, google/siglip-so400m-patch14-384 | embedding, zero-shot | image 1.3e-3 / 1.6e-3, zero-shot logits 1.5e-4 / 1.8e-4 |
| facebook/dinov2-base, dinov2-with-registers-base, dino-vitb16 (`.bin` only), google/vit-base-patch16-224-in21k | embedding | CLS embedding < 2e-3 |
| PekingU/rtdetr_r50vd, rtdetr_r101vd_coco_o365, rtdetr_v2_r18vd | detection | same detections (R101: 2 cats, sofa, 2 remotes), boxes within 2 px, scores within 0.02 |
| microsoft/table-transformer-detection | detection | same detection as transformers |
| Depth-Anything-V2-Base-hf, Depth-Anything-V2-Large-hf, LiheYoung/depth-anything-base-hf (V1) | depth | depth map 7.4e-5 / 9.8e-5 / 1.2e-4 |
| microsoft/trocr-base-handwritten | OCR | text exactly transformers' |

Fixes of the sweep (each with a regression test or a model in the suites above):

- **SwinIR** is a new adapter (`SwinIrAdapter`, canonical `swinir`: window attention with rebuilt relative position indices and shift masks as constants, window partition and cyclic shift as one row gather,
  sub-pixel or nearest + convolution upsamplers); `SuperResolutionImporter` reads the official `.pth` files (the int64 index and mask buffers are skipped and rebuilt).
- **Vulkan `Gather`** moved rows of 8 values only; activations with other widths (SwinIR's 180 channels) now gather one value per thread (quantized tables keep the 8-value requirement).
- **timm ResNet**: new `resnet` adapter (`ResNetClassifierAdapter`) and `TimmResNetHfMapping` for timm's `config.json` (no `model_type`; `HfConfig.ModelType` reports `timm_resnet`); the processor comes from `pretrained_cfg`
  (shortest edge `floor(size / crop_pct)`, center crop). torchvision's `CenterCrop` rounds its offsets to the nearest integer (ties to even) where Hugging Face's processors truncate, which moved the logits of the `crop_pct` 0.95
  models by 2.5-4 %: `ImageProcessorConfig.CropOffsetNearest` selects it (set for timm models).
- **RT-DETR**: ResNet-50 / 101 backbones (bottleneck blocks; the backbone configurations omit the depths), an encoder width apart from `d_model` (R101: 384 against 256), and the v2 checkpoints (`rt_detr_v2`:
  the `n_points_scale` constants are the same 1 / n per point as v1's; per-level point counts, other offset scales and the discrete sampling method are rejected by `Validate`).
- **Table Transformer**: `DetrAdapter` has a pre-norm form (`detr.pre_norm`) with the encoder's final norm.
- **DINOv2 with registers** resamples its position table with `antialias=True` (`interpolate_antialias`), plain DINOv2 without.
- **SigLIP 2** pads its prompts with the tokenizer's `<pad>` (0) where the text configuration only has the default 1 (the text embeddings were 0.46 off).
- **SigLIP so400m** (384 px, 14 px patches = a 27 x 27 grid and 6 dropped pixels): `ProcessedImage.Patchify` drops the remainder like the patch convolution.
- **laion repositories** hold the checkpoint twice (`model.safetensors` and `open_clip_model.safetensors`); the open_clip export is no longer read as a shard.

Not covered, with what is missing:

- **Florence-2** (OCR, caption, grounding): needs the DaViT vision tower (convolutional embeddings, window and channel attention blocks), the 2D position and temporal embeddings of its projector, image rows
  merged into the BART encoder's input embeddings (the `encoder_decoder` adapter takes either an image or text), the task prompts and the `ITextRecognizer` surface for them. Not started (about a day or two of work).
- **TrOCR small** (`deit` encoder with a distillation token, sinusoidal decoder positions) and **large** (BEiT encoder); BLIP captioning, Donut, manga-ocr, GLM-OCR (other encoder-decoder families); PaddleOCR (Paddle format).
- **Semantic and universal segmentation** (SegFormer b0-b5, Mask2Former, OneFormer, UperNet, CLIPSeg, BiRefNet / RMBG): the segmentation task is prompt based (`IImageSegmenter`, SAM); a per-pixel label task and the
  architectures are missing. SegFormer (mix transformer, all-MLP decoder) would be the first.
- **Detection**: YOLOS, DETR checkpoints without the timm backbone, Ultralytics YOLO (`.pt`; their ONNX exports run through the ONNX importer), RT-DETR v2 variants listed above.
- **Classification**: timm / HF families other than ResNet (MobileNetV3, EfficientNet, ConvNeXt, RepVGG, EfficientViT, Swin / SwinV2, EVA02 `timm_wrapper`, `SiglipForImageClassification`
  such as prithivMLmods' classifiers), HF `resnet` (microsoft/resnet-50), and `open_clip`-only repositories (timm/ViT-B-16-SigLIP, Marqo fashion models).
- **Depth / pose / features**: DPT-hybrid and DPT-large, DepthPro, Depth Anything 3, ZoeDepth, ViTPose, SuperPoint / LightGlue, DINOv3, SigLIP 2 NaFlex.

The residual 1e-3 of the image towers is the resize (Pillow rounds to 8 bits per pass; the sample's pixel difference is not yet bit-exact), identical on both backends. Text towers match to float rounding.

Depth Anything runs on a fixed 518 px square canvas like the detectors (the reference uses `--size 518`, `keep_aspect_ratio=False`); the map is resampled bilinearly to the input size.

Detector limits: the graphs run on a fixed square canvas (640 px, `DetrHfMapping.InputSize`; RT-DETR's own processor size): every image is resized to it without keeping the aspect ratio, where the checkpoints' own processors
resize DETR to 800/1333 px keeping the aspect ratio (a static graph cannot follow; the reference is generated with the same 640 px resize, `vision-reference.py --size 640`). Boxes are returned in pixels of the original image.
DETR needs the timm backbone layout (`use_timm_backbone`, ResNet-18/34/50/101), RT-DETR the ResNet-D backbone with one AIFI level and box refinement; others are rejected by `Validate` with a reason.
SigLIP's tokenizer removes all ASCII punctuation as transformers' `SiglipTokenizer` does (the fast tokenizer.json spares `/ < >`).

## 16. Audio models beyond speech (task layer)

Codecs and voice activity run through the task layer (kinds `audio-codec` and `voice-activity`, see docs/api-overview.md "Audio codecs and voice activity"; tests in
`tests/Sharplight.AudioModels.Tests`). References come from transformers (`EncodecModel`, `DacModel`, `MimiModel`, CPU) and onnxruntime plus the segmentation function of
the `silero-vad` package on `jfk.wav` (resampled to the codec's rate, a whole number of frames); checkpoints live in `~/models/audio-models/<name>`, reference tensors in
`~/models/audio-models/refs`.

| Model | Importer | Kind | Status | Parity |
|---|---|---|---|---|
| EnCodec 24 kHz (`facebook/encodec_24khz`: causal SEANet with reflect padding, 2-layer LSTM, 32 euclidean RVQ stages) | HF `encodec` | audio-codec | supported (the 48 kHz non-causal variant is rejected by `Validate`) | codes token-exact at 6 kbps (CPU and Vulkan); at 24 kbps the test allows 0.1 % of the codes to differ (near ties between entries of the deep stages, which round differently from the reference); the decoder from the reference codes: relative RMS < 1e-3 (CPU and Vulkan) |
| DAC 44 kHz (`descript/dac_44khz`: snake activations, 9 factorized cosine RVQ stages) | HF `dac` | audio-codec | supported | codes token-exact, decoded audio relative RMS < 1e-3 (CPU and Vulkan) |
| Mimi (`kyutai/mimi`: SEANet, 8+8 layer sliding-window transformers, split semantic + acoustic RVQ, 12.5 Hz) | HF `mimi` | audio-codec | supported; 32 stages, 8 by default | codes token-exact for 8 and 32 stages, decoded audio relative RMS < 1e-3 (CPU and Vulkan) |
| Silero VAD v5 (`silero_vad_16k_op15.onnx`: learned STFT, four convolutions, one LSTM step) | `Sharplight.Import.SileroVad` | voice-activity | supported (16 kHz; the 8 kHz branch of the full export is not imported) | frame probabilities within 1e-3 of onnxruntime, segments identical to `get_speech_timestamps` on two clips (CPU and Vulkan) |
| CLAP (`laion/clap-htsat-unfused`: HTS-AT Swin audio tower with a log-mel front end, RoBERTa text tower, projections) | HF `clap` (the fusion variants are rejected) | audio-embedding | supported (10 s clips at 48 kHz; shorter audio is repeated and padded, longer audio is cut or averaged over clips) | audio and text embeddings within 2e-3 relative RMS of transformers, zero-shot logits within 0.05 (CPU and Vulkan) |
| MusicGen small (`facebook/musicgen-small`: T5 text encoder, 24-layer decoder over four EnCodec 32 kHz codebooks with the delay pattern, classifier-free guidance, non-causal EnCodec decoder) | HF `musicgen` (mono models; one canonical source holds the T5 encoder, the decoder and the codec decoder) | music-generation | supported (greedy and sampling, up to 40 s, streaming in windows) | greedy codes with guidance 3 token-exact against `generate` (60 steps, CPU and Vulkan), decoded audio within 2e-3 relative RMS |
| Demucs (`htdemucs`: spectral and waveform U-Nets with dilated residual branches, a cross-domain transformer, complex-as-channels spectrogram output) | HF layout `htdemucs` (a safetensors conversion of the state dict with the constructor arguments as `config.json`, e.g. KeighBee/demucs; the original `.th` pickle is not read) | audio-separation | supported (4 stems, 7.8 s segments with 25 % overlap and triangular cross-fade; no Wiener filtering, no shifts, no multi-band models) | stems of a 10 s stereo mix against the demucs package's `apply_model` with the same weights: SDR above 35 dB on every stem and channel (CPU and Vulkan) |

CLAP, MusicGen and Demucs have their references in `tools/audio-reference` (`reference_clap.py`, `reference_musicgen.py`, `reference_demucs.py`). The CLAP checkpoint was converted from `pytorch_model.bin` to safetensors with `torch.load(weights_only=True)` (no pickled code runs), the MusicGen decoder reuses the EnCodec decoder graph with the non-causal padding of the 32 kHz codec (`EncodecParameters.Causal`), and its delay pattern and guidance are written out in `MusicGeneratorClient` (checked first against transformers' decoder stepped by hand). Demucs' host side (spectrogram, normalization, the cross-fade of overlapping segments and the centered padding of the last one) follows `demucs.apply` and `demucs.htdemucs._spec` / `_ispec`.

Limits: the codecs run on one channel; the encoder pads the audio at the end with zeros to a whole frame (the references pad per layer, so the last frame of audio whose
length is not a frame multiple can differ), audio longer than the component's bound runs in independent segments (no overlap, a seam can be audible in the non-causal DAC).
EnCodec's streaming chunking of the 48 kHz model is not implemented.

## 15. Image generation (2026-11-15)

Image generators are regular models of kind `image-generation` (see [api-overview.md](api-overview.md) "Image generation"): the
importer `Sharplight.Import.Diffusers` translates checkpoints into the canonical `latent_diffusion` source, `LatentDiffusionAdapter`
builds the CLIP text encoder, UNet and VAE as graphs of generic Core ops (Conv, GroupNorm, LayerNorm, Resize, batched Attention;
no new ops were needed), and Core's latent-diffusion pipeline runs them with host-side samplers and classifier-free guidance.

| family | formats | status | verified |
|---|---|---|---|
| Stable Diffusion 1.x (SD 1.5) | diffusers folder (with or without `model_index.json`), original single-file `.safetensors` or `.ckpt` (restricted unpickler, nothing executed), stable-diffusion.cpp GGUF (f16, Q8_0, Q4_0, any ggml type for linear weights) | **supported** (txt2img, negative prompts, img2img, mask-blend inpainting, any size that is a multiple of 64, `.slm` with per-component precompilation) | below |
| Stable Diffusion 2.x (2.1 768-v, 2.1 base) | diffusers folder, original single-file `.safetensors` / `.ckpt` (OpenCLIP ViT-H tower; 768 files default to v-prediction at 768×768, files named like the 512 / base releases to epsilon at 512×512) | **supported** (as SD 1.x) | below |
| SDXL (base 1.0) | diffusers folder, original (SGM) single-file `.safetensors` (CLIP-L + OpenCLIP bigG with the text projection), stable-diffusion.cpp GGUF in both naming schemes (sd.cpp's own `cond_stage_model.1.transformer.*`, e.g. gpustack/stable-diffusion-xl-base-1.0-GGUF Q8_0 / Q4_0; the SGM names, e.g. kostakoff/stable-diffusion-xl-base-1.0-GGUF Q4_K) | **supported** (two text encoders, pooled + size conditioning, `force_zeros_for_empty_prompt`) | below |
| SDXL-Turbo, SDXL-Lightning (distilled SDXL) | the SDXL formats; stable-diffusion.cpp GGUF (OlegSkutte/sdxl-turbo-GGUF, mzwing/SDXL-Lightning-GGUF) | **supported** (single files are recognized by their published names: Turbo 512×512, Euler ancestral on trailing timesteps, 4 steps, no guidance; Lightning 1024×1024, Euler trailing, the file's step count, no guidance, the 1-step "x0" file predicts the sample) | Turbo below; Lightning maps the same way, not verified |
| FLUX.1-schnell, FLUX.1-dev (and fine-tunes with the same layout) | diffusers folder; original single-file `.safetensors` (Black Forest Labs names, also with ComfyUI's `model.diffusion_model.` prefix); transformer-only GGUF (city96 / stable-diffusion.cpp: any ggml type) plus companion files: CLIP-L (`clip_l.safetensors`), T5-XXL (encoder GGUF with its tokenizer, or `t5xxl_*.safetensors` with the embedded T5 vocabulary), VAE (`ae.safetensors`), given as auxiliary sources or found beside the transformer | **supported** (txt2img, img2img, inpainting; flow-matching Euler with FLUX's static / resolution-dependent shift; dev's guidance embedding; any size that is a multiple of 16; `.slm`) | [below](#flux) |
| SD 1.5 inpainting UNet (9 input channels), SDXL refiner, SD3, Chroma, Qwen-Image, video (Wan, LTX) | - | missing (recognized in GGUF and safetensors single files and rejected as "Not supported yet: ... (diffusion transformer)") | the graph contract (`DiffusionGraphs`), the description, the flow sampler and the companion-file loading are kind-generic; see the [FLUX notes](#flux) for SD3 / Chroma |
| LoRA (kohya / diffusers), ControlNet, TAESD | - | missing | |

Verification (tools/make-diffusion-references.py writes the diffusers references; tests in `tests/Sharplight.ImageGeneration.Tests`):

- Tiny random SD 1.x pipeline (`~/models/sd/tiny-random-sd`, tools/make-tiny-diffusion-models.py): text encoder, UNet, VAE decode
  and encode within 1e-4 relative RMS of transformers / diffusers on CPU and Vulkan; whole txt2img runs from the same initial
  noise with Euler, Euler Karras, DDIM, PNDM, DPM++ 2M and DPM++ 2M Karras above 40 dB PSNR on both; the single-file checkpoint
  is bit-identical to the diffusers folder after import; the CLIP tokenizer (rebuilt from the embedded OpenAI merges) gives
  transformers' ids.
- SD 1.5 (fp32 diffusers folder) on Vulkan, 512×512, 20 steps, guidance 7.5, same initial noise as diffusers (fp32, CUDA):
  Euler 76.3 dB, Euler Karras 76.7 dB, DDIM 76.2 dB, PNDM 58.8 dB, DPM++ 2M 60.5 dB, DPM++ 2M Karras 73.8 dB PSNR; text encoder,
  UNet and VAE decode within 2e-3 / 5e-3 / 5e-3 relative RMS. The single-file checkpoint, the original `.ckpt` and the three GGUF
  files map to exactly the diffusers folder's canonical names and shapes; the `.ckpt` weights are byte-identical to the `.safetensors` ones.
- stable-diffusion.cpp 5dcfa79 (CUDA and Vulkan builds), Euler, 20 steps, `--rng cpu` (torch CPU noise, the same as the
  diffusers references): on the f16 GGUF Sharplight (Vulkan) is 48.7 dB from diffusers fp32 and 44.3 dB from sd.cpp CUDA (sd.cpp
  CUDA 43.5 dB from diffusers, sd.cpp Vulkan 28 dB from both); on Q8_0 all implementations give visually the same image but drift
  apart along the trajectory (Sharplight vs sd.cpp Vulkan 30.0 dB, vs CUDA 20.2 dB); the published Q4_0 file produces a degenerate
  image in every implementation (Sharplight vs sd.cpp CUDA / Vulkan 37.8 / 41.0 dB).
- Tiny random SD 2.x pipeline (`~/models/sd/tiny-random-sd2`, `make-tiny-diffusion-models.py --family sd2`: linear projections,
  per-level head counts, upcast attention, GELU text encoder, v-prediction, DDIM): graphs within 1e-4 and txt2img above 40 dB for
  all six samplers on CPU and Vulkan; its OpenCLIP single file (`.safetensors` and `.ckpt`, with an extra last block that the
  importer drops) imports to exactly the folder's weights. Sharplight's `pndm` is diffusers' PLMS form (`skip_prk_steps`), also for
  DDIM-configured models.
- SD 2.1 768-v (sd2-community mirror) on Vulkan, 768×768, 20 steps, same initial noise as diffusers: fp32 folder Euler 70.2 dB,
  DDIM 64.1 dB, DPM++ 2M Karras 51.8 dB, graphs within the SD 1.5 tolerances; with the fp16 files (computed in f16 on Vulkan) the
  images are visually identical but 37.8 / 34.1 / 28.1 dB from diffusers' fp32-upcast run (UNet 2.2 % relative RMS on random
  inputs). The 768-v single file maps to the diffusers folder's names, shapes and configuration.
- Tiny random SDXL pipeline (`~/models/sd/tiny-random-sdxl`, tools/make-tiny-sdxl.py): both text encoders (penultimate layers),
  the pooled projection, the UNet with added conditioning and the VAE within 1e-4, txt2img above 40 dB (Euler, DDIM, DPM++ 2M
  Karras; without a negative prompt the zero unconditional row) on CPU and Vulkan. The SDXL base single file maps to the diffusers
  folder's names, shapes and configuration. When a scheduler config omits `set_alpha_to_one`, DDIM / PNDM follow diffusers'
  per-scheduler defaults.
- SDXL base 1.0 on Vulkan (published fp16 files), 1024×1024, 20 steps, guidance 7.5, same initial noise as diffusers (fp16 files
  upcast to fp32, CUDA): UNet 5.7e-5, VAE decode 1.5e-5, second text encoder 6e-4 relative RMS; Euler 42.4 dB, DPM++ 2M Karras
  30.0 dB (visually identical; the multistep sampler drifts further). Unlike SD 2.1 (attention upcast in its config, 2.2 % UNet
  error in f16), SDXL keeps its accuracy with f16 weights.
- SDXL as stable-diffusion.cpp GGUF (2026-11-15; `SdxlGgufTests`, `ImporterTests.SingleFileCheckpointsMapToTheDiffusersNames`): the
  gpustack Q8_0 / Q4_0, kostakoff Q4_K and OlegSkutte SDXL-Turbo Q8_0 files map to exactly the SDXL diffusers folder's names and
  shapes. Against diffusers (fp32-upcast published files, same initial noise), on Vulkan: gpustack Q8_0 text encoders 1.3e-4 /
  6e-4, pooled 4e-4, UNet 2.4e-3 relative RMS, Euler 1024×1024 27.7 dB; kostakoff Q4_K (norm scales quantized too) bigG hidden
  states 20 % off, the image visually equivalent (17.3 dB); SDXL-Turbo Q8_0 against stabilityai/sdxl-turbo (512×512, 4 Euler steps
  on trailing timesteps, no guidance; ~/models/sd/sdxl-turbo-reference): text encoders 3.5e-3 / 4e-2 (every tensor Q8_0, embeddings
  and norms included), pooled 2.8e-2, UNet 1.0e-2 relative RMS, image 28.5 dB (stable-diffusion.cpp 5dcfa79 CUDA on the same file,
  Euler, 4 steps, `--rng cpu`: 18.9 dB from diffusers; it uses the file's projection as stored, which for this file is OpenCLIP's
  orientation, and its own sigma spacing). Three file quirks: sd.cpp renames the bigG projection
  without transposing it, so its orientation follows the source (diffusers-converted files such as gpustack's: transformers'
  layout; files converted from an original checkpoint such as OlegSkutte's: OpenCLIP's), told apart by the diffusers conversion's
  mark (gpustack's files keep the upsampler of an attention level at `output_blocks.N.1`, beside the attention block, which the
  importer also maps); the Q4_K files quantize the projection and the fused q/k/v biases, which are decoded before they are
  transposed or split. An SDXL UNet without its second encoder fails on import naming it, and a package converted before this fix
  fails on load ("Import the original checkpoint again") instead of at the first image ("The denoiser needs a pooled text
  embedding"). UNet-only files (no VAE / text encoder) fail naming the missing parts; separate VAE / text-encoder files are not
  supported yet.
- Indicative speed on the shared RTX PRO 4500 (GPU 0, other agents active, not a benchmark): Sharplight denoises at about 0.1 s per
  step (CFG batch of 2, 512×512, f16 GGUF), sd.cpp sampled 20 steps in 17.1 s (Vulkan) and 22.8 s (CUDA) in the same session.
  Without a precompiled `.slm` the first load spends 50–130 s compiling the UNet kernels (Slang, CPU-bound); `sharplight-convert
  precompile` stores them in the package.

### FLUX

<a id="flux"></a>
FLUX.1 runs on the same pipeline as the UNet models (2026-11-15). Design:

- Canonical source `flux` (`Sharplight.Import.Diffusers`): `transformer.*` in the original Black Forest Labs layout (fused q/k/v,
  `img_*` / `txt_*` streams of the dual-stream blocks, `linear1` / `linear2` of the single-stream blocks; diffusers' separate
  projections are joined lazily, zero-copy for GGUF / single files), the CLIP-L encoder under `text_encoder.*`, the T5 encoder under
  `text_encoder_2.*` in the encoder vocabulary of the `encoder_decoder` architecture (so a T5 encoder GGUF read by the GGUF importer
  maps 1:1), the VAE under `vae.*`, the CLIP tokenizer under the canonical `tokenizer.*` keys and the T5 tokenizer under
  `flux.tokenizer_2.tokenizer.*`, the flow schedule (`flux.scheduler.*`: shift, dynamic shift, base / max shift and sequence lengths)
  and the defaults (schnell: 4 steps, no guidance, T5 prompts of 256 tokens; dev: 28 steps, guidance 3.5, 512 tokens).
- `FluxAdapter` (Sharplight.Architectures) builds five graphs of generic ops: the CLIP graph with only the pooled output
  (`ClipTextGraphBuilder`, final-normed end token, no projection), the T5 graph (`T5TextGraphBuilder`: the bucketed relative
  position bias gathered into a [heads, L, L] tensor and added to batched per-head score products (`BatchMatMul`, `Softmax`):
  T5-XXL's 64 × 257 bias table as the attention op's constant `position_bias` stalls the Vulkan attention kernel; projections keep
  f32 activations in the integer-dot kernels of quantized weights, gated tanh-GELU, no attention mask as in diffusers), the transformer
  (`FluxTransformerGraphBuilder`: patchify / unpatchify as reshape + permute, sinusoidal timestep / guidance embeddings, modulation
  chunks as slices of one projection, per-head RMS-normed q/k, RoPE over (0, row, column) positions through the `Rope` op's
  multi-axis sections plus per-frequency divisors that give each axis its own θ^(−2j/d) series, joint bidirectional `Attention`
  with the new `f16_products` attribute (`StandardOps.F16ProductsAttribute`: f16 products with f32 accumulation for that node, as
  `AllowReducedPrecision` would allow; FLUX is trained and run in bf16, and the exact f32 kernel for 128-wide heads is the scalar
  one because the split-precision flash tiles exceed 48 KB of workgroup memory: a 1024×1024 step took 18.6 s, 96 % in attention,
  and takes 1.04 s with it) and the existing 16-channel KL decoder / encoder (scaling 0.3611, shift 0.1159). No new op was needed.
- Core: `NoiseSchedule` describes rectified-flow schedules (`predictionType: flow`), `FlowMatchEulerSampler` follows diffusers'
  `FlowMatchEulerDiscreteScheduler` as the FLUX pipeline drives it (σ = linspace(1, 1/n), static or μ-shifted by the image token
  count, timestep = σ·1000, img2img start int(n − n·strength)); `LatentDiffusionGenerator` gained a second text-encoder component
  with its own tokenizer (`text-encoder-2`, `tokenizer-2`), per-token joining of the hidden states (encoders without them only give
  the pooled embedding), the `guidance` denoiser input, the `classifierFreeGuidance` model switch, row-by-row runs for denoisers built
  for one row, and the release of the text encoders after encoding when the weights take more than 80 % of the device memory.
- Companion files: `IModelImporter.FindCompanions` (Core) lets an importer name the files a model needs; `ModelImporterCatalog.OpenSet`
  opens them for the auxiliary roles the caller leaves out. The diffusers importer recognizes stand-alone component files by their
  tensors (CLIP text encoders, KL autoencoders in original or diffusers names, T5 encoders; configurations from a `config.json`
  beside them or from the shapes; T5 files without a tokenizer get the embedded T5 v1.1 vocabulary) and looks for them beside the
  transformer and in the sibling `clip`, `text_encoders`, `vae` folders of ComfyUI's layout. `FluxCompanionAdapter` merges the
  auxiliary sources (`text-encoder`: `clip_text_encoder`, `text-encoder-2`: any `encoder_decoder` T5, `vae`: `kl_autoencoder`) into
  the canonical source, keeping the family's VAE scaling. Studio's load panel shows the three companion fields for transformer-only
  files, filled with what was found; `sharplight-generate-image` takes `--text-encoder`, `--text-encoder-2` and `--vae`.
Verification (tools/make-tiny-flux.py, tools/make-flux-references.py; tests `TinyFluxTests`, `FluxSchnellTests`, `FluxDevTests` in
`tests/Sharplight.ImageGeneration.Tests`, `FluxImageTests` in `tests/Sharplight.Studio.Tests`; GPU 1 of the 3× RTX PRO 4500 machine,
32 GB, dedicated):

- Tiny random FLUX pipelines (`~/models/flux/tiny-random-flux/{dev,schnell}`: guidance embedding + dynamic shift / neither): CLIP
  pooled embedding, T5 hidden states, transformer (64×96 image, non-square RoPE grid), VAE decode and encode within 1e-4 relative RMS
  of diffusers / transformers on CPU and Vulkan, from the diffusers folder, the original-name single file and the Q8_0 GGUF with
  companions (the GGUF transformer within 5e-3 of diffusers on the dequantized weights: both backends quantize the activations of
  Q8_0 products); 4-step txt2img from the same latents above 40 dB PSNR in all 12 combinations; the tokenizers give transformers'
  ids; bare companion files beside the GGUF (no configuration or tokenizer) are found, configured from their shapes (the T5 one with
  the embedded vocabulary) and give the same image; a `.slm` converted from the transformer-only file holds the companions and
  generates the same image after the source files are deleted.
- FLUX.1-schnell, diffusers folder (bf16 files) on Vulkan against diffusers in f32 on the CPU: CLIP pooled 2.5–4.0e-4, T5 4–13e-6,
  transformer 1.2e-4 (512×512), VAE decode 8.6e-6 relative RMS; 4-step txt2img from the same latents 32.5 dB (512×512) and 38.8 dB
  (1024×1024) PSNR, visually identical (the attention's f16 products; without them the scalar f32 kernel makes 1024×1024 18× slower).
- FLUX.1-schnell, city96's Q8_0 GGUF with `t5-v1_1-xxl-encoder-Q8_0.gguf`, `clip_l.safetensors` and `ae.safetensors` (found beside
  it): the transformer is 1.3e-3 from the bf16 reference, the T5 GGUF 0.11–0.21 (relative RMS of the hidden states; transformers
  running the dequantized GGUF weights gives the same 0.205, so it is the file's, not the implementation's), `clip_l.safetensors`
  (f16) 3–6e-3. Images: 22.6 dB (512) / 28.9 dB (1024) from the bf16 diffusers pipeline, visually the same picture with different
  fine texture; stable-diffusion.cpp (CUDA, same files, its flux scheduler with the shift disabled to match diffusers' schnell
  schedule, `--rng cpu` = the same latents) is 21.7 / 26.3 dB from diffusers and 22.0 / 25.8 dB from Sharplight at 512 / 1024.
  Against diffusers running the very same files (dequantized GGUF transformer and T5, `clip_l.safetensors`;
  `make-flux-references.py --transformer-gguf --t5-gguf --clip-file`): Sharplight 33.4 dB (512) / 39.7 dB (1024),
  stable-diffusion.cpp 21.8 / 25.8 dB.
- FLUX.1-dev (city96 Q8_0 GGUF): the transformer with the guidance embedding is 3.3e-4 from diffusers running the dequantized GGUF;
  20 steps at 512×512 (guidance 3.5, dynamic shift) 31.7 dB from diffusers running the same files (Q8_0 transformer and T5,
  `clip_l.safetensors`); 1024×1024 28 steps in 33.2 s (below).
- Speed and memory (Vulkan, Q8_0 transformer and T5, warm runs with the conditioning reused): 512×512 0.26 s per step, decode
  0.14 s, 4 steps 1.16 s; 1024×1024 1.04 s per step (attention 26 %, Q8_0 products 55 %), decode 1.05 s, 4 steps 5.2 s; FLUX.1-dev
  1024×1024 1.15 s per step, 28 steps 33.2 s, peak 23.5 GB of device memory. stable-diffusion.cpp CUDA on the same files samples 4
  steps in 2.24 s (512) / 4.51 s (1024). The bf16 diffusers folder (33.7 GB of weights) runs on the 32 GB GPU by alternating the
  text encoders and the denoiser (4 steps at 1024×1024 5.7 s with the conditioning reused). Without a precompiled `.slm` the first
  run compiles the kernels for 70–150 s.
- Studio: the load panel recognizes the transformer-only GGUF, fills in the three companions found beside it and loads; the Image
  page generates through the loaded model (tiny pipeline and FLUX.1-schnell GGUF at 512×512).
- Later families: SD3 / SD3.5 are MMDiTs with joint attention as well (different block layout: separate q/k/v with optional
  RMS norms, 2D sin-cos position embedding instead of RoPE, three text encoders whose CLIP hidden states are padded and concatenated
  with T5's along the sequence, CFG): a second transformer builder, a sequence-joining mode for the text conditioning, the existing
  flow sampler with shift 3. Chroma is a FLUX.1-schnell derivative without CLIP whose modulation comes from a small "distilled
  guidance" MLP instead of the per-block modulation projections, with a T5 attention mask and real CFG: a FluxAdapter variant (T5 as
  the first text encoder, a mask input built from the padding), CFG row by row through the existing fallback. Qwen-Image and the
  video DiTs (Wan, LTX: a frame axis in the latents and 3D RoPE through the same multi-axis sections) reuse the flow sampler and the
  companion-file loading.

## 16. Video generation (2026-11-16)

Video generators are models of kind `video-generation` (see [api-overview.md](api-overview.md) "Video generation"): the importer
`Sharplight.Import.Diffusers` translates Wan pipeline folders into the canonical `wan` source, `WanAdapter` builds the umT5 text
encoder, the video DiT and the causal 3-D VAE as graphs of generic Core ops (no new op: a causal 3-D convolution is one 2-D
convolution over the frame windows stacked along the channels, the importer folding the kernel's time taps into its input
channels), and Core's `VideoDiffusionGenerator` runs them with the UniPC or Euler flow sampler and classifier-free guidance on the
host, decoding latent frame by latent frame with the causal caches carried between chunks.

| family | formats | status | verified |
|---|---|---|---|
| Wan 2.1 T2V (1.3B, 14B) | diffusers folder | **supported** (text-to-video, negative prompts, any size that is a multiple of 16, frames 1 + 4k, UniPC (default) and Euler) | tiny pipeline; 1.3B below |
| Wan 2.2 TI2V-5B | diffusers folder | **supported** (text- and image-to-video: per-token timesteps, the residual VAE with 2×2 pixel patching, first-frame conditioning; multiples of 32) | tiny pipeline; 5B below |
| Wan 2.1 I2V / FLF2V (CLIP image embedding), Wan 2.2 A14B (two experts), Wan GGUF (city96 / QuantStack) | - | missing (the I2V and A14B folders are rejected by name) | |
| LTX-Video, HunyuanVideo, CogVideoX, Mochi | - | missing | |

Design notes:

- Graph contract `VideoDiffusionGraphs`: `text_encode` (tokens and a key mask, so padding is never attended to; the pipeline zeroes
  the padding rows as the Wan pipelines do), `denoise` (latents [C, T, h, w], timestep [1] or per token, context [512, D]),
  `vae_decode_first` / `vae_decode_next` (one latent frame in, 1 or 4 frames out, cache inputs `cache_in.*` and outputs
  `cache_out.*` holding each causal convolution's last two input frames) and `vae_encode` (one image to the first latent frame).
- Transformer canonical names follow the original Wan checkpoints (`patch_embedding`, `blocks.{i}.self_attn.*`, `norm3`,
  `cross_attn.*`, `ffn.0|2`, `modulation`, `head.*`), so original-name files and GGUF can map 1:1 later. RoPE over (frame, row,
  column) uses the `Rope` op's multi-axis sections with per-axis frequency divisors (`AxialRope`, shared with FLUX).
- The first decoded chunk skips the temporal upsampling (as the reference decoder does), so a clip of T latent frames has
  1 + 4 (T − 1) frames; decoding memory does not grow with the clip length.
- Tokenizer fix: Hugging Face templates that start with the sequence (T5 / umT5 `$A </s>`) no longer add a BOS token when the
  tokenizer configuration names one.

Verification (tools/make-tiny-wan.py, tools/make-wan-references.py; tests `TinyWanTests`, `Wan21Tests` in
`tests/Sharplight.ImageGeneration.Tests`):

- Tiny random Wan pipelines (`~/models/video/tiny-random-wan/{t2v,ti2v}`: Wan 2.1 layout; Wan 2.2 TI2V layout with the residual
  VAE, 2×2 patching and per-token timesteps): umT5 conditioning, transformer, the streamed VAE decode of 9 frames and the image
  encoder within 1e-4 relative RMS of diffusers / transformers on the CPU and on Vulkan; whole runs from the same initial latents
  (4 steps, guidance 5, negative prompt) above 40 dB PSNR in every frame: text-to-video with UniPC and with Euler for both
  layouts, image-to-video for TI2V.
- Real models on Vulkan (one RTX PRO 4500, 32 GB) against diffusers in f32 on CUDA, 832×480, 17 frames, 20 UniPC steps,
  guidance 5, the same initial latents: umT5-XXL (bf16 files) 3.0e-6 relative RMS for both models; the transformer 2.4e-4 (Wan 2.1
  1.3B) and 2.8e-4 (Wan 2.2 TI2V-5B, f32 files) relative RMS; Wan 2.2 TI2V-5B image-to-video worst frame 55.1 dB PSNR (468.7 s
  for the whole run including the first-time kernel compilation of every graph). The Wan 2.1 folder's f32 umT5 (22.7 GB) is the
  same model as Wan 2.2's bf16 one (1.7e-3 relative weight difference = bf16 rounding); the references use the bf16 files.
