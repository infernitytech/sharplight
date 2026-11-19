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
| `decoder` | DecoderAdapter | llama (incl. Mixtral), qwen2, qwen3, gemma, gemma2, gemma3, gemma4, phi2, phi3, olmoe, qwen3moe, muse-glimmer, flint, smollm3, olmo, olmo2, glm4, chatglm, ernie4_5, hunyuan-dense, exaone, exaone4, seed_oss, internlm2, qwen, xverse, openelm, granite, granitemoe, minicpm, starcoder2, command-r, cohere2, falcon, gptneox, gpt2, nemotron, stablelm, apertus, arcee, qwen2moe, starcoder, refact, jais, jais2, orion, codeshell, plamo, plamo3, pangu-embedded, maincoder, spark2_5, baichuan, granite_swa, mellum, dbrx, deepseek, bailingmoe, cohere2moe, qwen3vl, qwen3vlmoe, qwen2vl (and the families listed in GgufArchitectureMappings) | llama, mistral, mixtral, qwen2, qwen3, gemma, gemma2, gemma3_text, gemma3 (text decoder of Gemma3ForConditionalGeneration), gemma4(_text/_unified(_text)), phi, phi3, olmoe, qwen3_moe, muse_glimmer, flint, smollm3, olmo, olmo2, olmo3, glm4, ernie4_5, hunyuan_v1_dense, exaone, exaone4, seed_oss, internlm2, granite, granitemoe, minicpm, starcoder2, cohere, cohere2, falcon, gpt_neox, gptj, gpt2, nemotron, stablelm, apertus, arcee, qwen2_moe, gpt_bigcode, jais2, granite_swa, granitemoe_swa, granitemoeshared, mellum, dbrx, cohere2_moe, qwen3_vl, qwen3_vl_moe, qwen2_vl, qwen2_5_vl (and the families listed in HfArchitectureMappings) |
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
| `SPECDEC` | **Speculative-decoding drafters (EAGLE3, DFlash, MTP assistants)**. draft graphs that consume target hidden states. | 3 | 3 | L | partial: MTP for the qwen3_5 family (incl. the hyper-connected head of Qwen3.8-Flash-Next / `qwen4exp`, `MtpDraftHead.BuildHyperConnected`, HF, GGUF and the `mtp-*.gguf` sidecar) and NextN heads of DeepSeek-V3 (`latent_decoder`) / GLM-4.5 / Qwen3-Next / MiMo-7B (`mimo`, HF: Qwen2 layers + `model.mtp_layers.0`, drafts equal a transformers-module reference on a tiny model) (`decoder`, uniform layers) from HF and GGUF (`MtpDraftHead` + `MtpSpeculator`, rollback of recurrent state, shared embedding / LM head, device-side verification); verified on tiny random checkpoints only (no real DeepSeek / GLM checkpoint was run); EAGLE3 drafters as the `draft` auxiliary source (`Eagle3DraftHead`, target layer taps, draft vocabulary map), tiny Llama target and drafter, HF and GGUF; DFlash drafters missing |
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
| `qwen4exp` | Qwen3.8-Flash-Next (hf: qwen4_exp, ~200B total params) | GDN + gated attention, MoE 512 experts top-10 + shared, hyper-connections (4 streams, low-rank), QSA sparse-attention indexer (top-k 2048, pooled keys), PLE n-gram hash embeddings (layer 2), M-RoPE, MTP | **supported** (HC, NGRAM, QSA block selection in the sparse regime; MTP head as draft head: HF, GGUF and the `mtp-*.gguf` sidecar, drafts equal a transformers-module reference on the tiny model, acceptance 0.78 / 0.71 / 0.58 at k = 1 / 2 / 3 on the real model) | - | qwen4_exp | gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL (+mmproj), hf/Qwen3.8-Flash-Next (config+tokenizer), hf/tiny-random-qwen3.8-flash-next, hf/tiny-random-qwen3.8-flash-next-sparse (+ f32 GGUF) |
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
| MTP block | `eh_proj`, `enorm`, `hnorm` per stream, own HC head | done: `MtpDraftHead.BuildHyperConnected` (reads the trunk's residual streams before the final mixer, decode output `hidden_streams`) |
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
| Qwen3-TTS 12Hz 0.6B CustomVoice (Qwen; talker: 28-layer Qwen3 decoder over the codec's first codebook fed with embedding rows; code predictor: 5-layer Qwen3 decoder for codebooks 1-15; 12 Hz codec in `speech_tokenizer/`), 9 built-in speakers, 10 languages + 2 dialects | supported, CPU and Vulkan | official `qwen_tts` 0.1.1 on transformers 4.57.3 (CPU, f32; greedy talker and code predictor, non-streaming text; `tests/Sharplight.TextToSpeech.Tests/tools/make_qwen3_tts_reference.py`) | text ids identical, talker prompt rows relative RMS 8e-8; teacher forced over 47 frames: 0 argmax disagreements of the talker and 0 of the code predictor (705 codes), worst top-logit delta 0.016 (CPU) / 0.015 (Vulkan); greedy frames on CPU identical for the first 21 frames (the code predictor's top two logits are often within 0.01, the reference's first such tie is at frame 1), 46 vs 47 frames; on Vulkan (bf16 GEMMs with f16 activations) greedy decoding leaves the reference at that first tie; codec decode of the reference frames relative RMS 2.6e-6 (CPU) / 2.4e-6 (Vulkan) |
| OmniVoice (k2-fsa; 600+ languages; Qwen3-0.6B backbone with bidirectional attention over text tokens and 8 audio codebooks, masked-diffusion decoding in 32 steps with classifier-free guidance; Higgs Audio v2 tokenizer in `audio_tokenizer/`), voice of a recording with its transcript, designed voices (instructions) | supported, CPU and Vulkan; licenses: weights CC-BY-NC (non-commercial), audio tokenizer Boson Higgs Audio 2 Community License (expanded license above 100k annual active users), code Apache-2.0 | the official `omnivoice` 0.2.1 on transformers 5.23 (CPU, f32; deterministic: position and class temperature 0; `tests/Sharplight.TextToSpeech.Tests/tools/make_omnivoice_reference.py`; jfk.wav as the cloning reference, plus a voice-design case) | prompt token ids and audio masks identical, unmasking schedule identical; language-model logits (conditional and unconditional) at the first and last of the 32 steps relative RMS 5.7e-7 / 6.9e-7 (CPU) and 3.1e-7 / 5.3e-7 (Vulkan), greedy codes identical over all 32 steps (0 of 1080 differ cloning, 0 of 616 designed; CPU and Vulkan); reference codes of the tokenizer's encoder 0 of 2200 differ (CPU and Vulkan); decoded audio relative RMS 2.5e-6 (CPU) / 2.8e-6 (Vulkan) cloning, 1.7e-6 designed; real-time factor 6.0 (CPU, 4 threads on a loaded machine) / 0.87 (Vulkan, shared RTX PRO 4500) for a 5.6 s cloned sentence including the 11 s reference encode (32 steps over a 457-row prompt, uncached: every step runs the whole sequence) |
| Chatterbox (ResembleAI/chatterbox, English; T3 Llama 520M over S3 speech tokens, S3Gen flow matching, HiFT vocoder), built-in voice | supported, CPU and Vulkan | the `chatterbox-tts` package 0.1.7 on torch (CPU, float32; `tests/Sharplight.TextToSpeech.Tests/tools/make_chatterbox_reference.py`) | text tokens identical; conditioning rows 7e-7 (CPU) / 4e-7 (Vulkan) relative RMS; T3 guided greedy chain teacher forced: 0 argmax disagreements over 92 steps (reference margins down to 1.5e-3); flow mel for the reference tokens and noise 4.4e-7 / 5.9e-7; HiFT F0 7.4e-7 / 7.8e-7 and waveform 5.6e-6 / 7.0e-6 with the reference's source draws; voice cloning (`tools/make_chatterbox_clone_reference.py`, LibriSpeech 1272-128104-0004): voice encoder partials 9e-7 / 1.2e-6, S3 tokens 150 of 150 (T3 prompt) and 250 of 250 (flow prompt), CAM++ x-vector 3e-6 / 2e-6, flow prompt log-mel 1.9e-7; a cloned sentence transcribed word for word by Qwen3-ASR-1.7B, its speaker similarity to the recording 0.88 against 0.71 for the built-in voice |
| F5-TTS v1 Base (SWivid; flow-matching diffusion transformer over characters) + Vocos mel 24 kHz, voice of a reference recording | supported, CPU and Vulkan | the `f5-tts` package 1.1.22 on torch (CPU, float32; `tools/make_f5tts_reference.py`, jfk.wav as the reference) | character ids and duration identical; sampled mel from the reference's log-mel and noise (32 guided Euler steps) 4.2e-6 (CPU) / 3.5e-5 (Vulkan) relative RMS; Vocos waveform 2.4e-5 / 1.8e-5; the reference log-mel within 5e-3 on a 24 kHz copy of the recording (from 16 kHz the resamplers differ in near-silent frames, which the log floor magnifies) |
| XTTS v2 (Coqui; GPT-2 over audio codes + HiFi-GAN decoder), the 58 built-in speakers, English | supported, CPU and Vulkan | coqui-tts 0.28.0 on torch (CPU, float32; `tools/make_xtts_reference.py`, speaker "Claribel Dervla") | text ids identical (abbreviations, numbers as num2words spells them, symbols); greedy codes 90 of 90 identical (repetition penalty 10); latents 6.4e-5 (CPU) / 6.8e-5 (Vulkan) relative RMS; HiFi-GAN waveform of the reference latents 1.6e-4 / 1.8e-4; sampled with the model's settings, a four-sentence passage transcribed word for word by Qwen3-ASR-1.7B; Vulkan real-time factor 0.24 (4.2 s of audio in 1.0 s, RTX PRO 4500), CPU 1.2; voice cloning: style latents 3.7e-6 / 2.7e-6, speaker embedding 2.8e-6 / 1.3e-6 |
| CosyVoice 3 (FunAudioLLM Fun-CosyVoice3-0.5B; Qwen2 0.5B over S3 speech tokens v3, DiT flow, causal HiFT), the voice of a recording (zero-shot, cross-lingual, instruct) | supported, CPU and Vulkan | FunAudioLLM's CosyVoice code (dd9105b) on torch (CPU, float32) and onnxruntime 1.32 (`tools/make_cosyvoice3_reference.py`, jfk.wav as the recording) | S3 tokenizer v3 275 of 275 tokens identical and CAM++ x-vector 3.0e-6 (CPU) / 1.8e-6 (Vulkan) vs onnxruntime; prompt log-mel 7.5e-6; text and prompt ids identical; language model teacher forced along the greedy chain: 0 argmax disagreements over 300 steps, worst logit delta 1.1e-2 / 1.2e-2; flow mel for the sampled tokens and the reference's noise 4.1e-5 / 3.3e-5; causal HiFT F0 1.5e-6 / 1.7e-6, waveform 2.2e-5 / 2.1e-5 on the reference's F0 and source noise (3.9e-3 / 1.0e-2 on its own F0, see below); a zero-shot sentence transcribed word for word by Qwen3-ASR-1.7B, CAM++ speaker similarity to the recording 0.85; Vulkan real-time factor 0.37 (9.2 s of audio in 3.4 s, RTX PRO 4500), CPU 2.8 |
| CSM-1B (Sesame; Llama 1B backbone over text tokens and frame sums, 4-layer depth decoder over codebooks 1-31, Mimi codec bundled in the checkpoint), speaker numbers 0 and 1, no audio context | supported, CPU and Vulkan | transformers 4.57.3 `CsmForConditionalGeneration` (CPU, float32; `tools/make_csm_reference.py`, greedy backbone and depth decoder) | text ids identical; teacher forced over 31 frames (992 codes): 0 argmax disagreements of the backbone and of the depth decoder, worst top-logit delta 6.3e-3 (CPU) / 5.7e-3 (Vulkan), backbone hidden state relative RMS 2.1e-4 / 3.2e-4; greedy frames identical to the reference for the first 25 (CPU) / 9 (Vulkan) of 30 (near ties); Mimi decode of the reference frames relative RMS 1.4e-6 (CPU) / 2.1e-4 (Vulkan); `.slm` round trip generates the same frames and audio |
| VibeVoice-1.5B (Microsoft; Qwen2 1.5B over a speaker-turn script, diffusion head over 64-wide continuous acoustic latents at 7.5 Hz, acoustic tokenizer decoder, semantic tokenizer encoder feeding back), multi-speaker scripts (`Speaker N: ...`), voices from reference recordings | supported, CPU and Vulkan | transformers 5.23 `VibeVoiceForConditionalGeneration` (CPU, float32, weights of the original Microsoft checkpoint renamed to transformers' layout; `tools/make_vibevoice_reference.py`) and diffusers' `DPMSolverMultistepScheduler`; greedy language model, 10 DPM-Solver steps, guidance 1.3, the reference's noise | tokenizers (section 12.1): acoustic latents 3.0e-6, semantic latents 6.9e-6, decoded waveform 1.7e-6 relative RMS (CPU); prompt ids identical (single speaker, two speakers, with a voice prompt); teacher forced along the reference's tokens and latents over 34 steps (32 frames): 0 argmax disagreements of the constrained next token, worst logit delta 1.2e-2 (CPU) / 1.2e-2 (Vulkan), language model hidden state 1.7e-3 / 1.9e-3 and context-free (CFG) run's 7.2e-4 / 8.9e-4 relative RMS, diffusion latents (own LM state, same noise and steps) 2.7e-3 / 2.5e-3, waveform of the reference's latents 3.0e-6 / 5.0e-4; two speakers (36 frames, an audio-end/audio-start pair between the turns): 1.3e-3 hidden, 1.1e-3 context-free, 3.6e-3 latents (CPU); voice prompt (4 s recording, 30 frames): acoustic latents 2.9e-6, generation 2.0e-3 hidden, 1.5e-3 latents (CPU) / 2.0e-3 / 2.1e-3 (Vulkan); free running with the reference's noise: the reference's 34 tokens, waveform 5.4e-2 relative RMS (feedback amplifies the 1e-3 latent differences); `.slm` round trip generates the same latents and audio; streamed in buffers of 4 frames |
| Dia-1.6B (Nari Labs; Apache 2.0; encoder-decoder transformer over the bytes of `[S1]`/`[S2]` dialogue text and nine delayed 44.1 kHz DAC codebooks: 12 encoder layers of width 1024, 18 decoder layers of width 2048 with grouped-query self-attention and cross-attention, one head over all codebooks; the DAC codec as the vocoder companion), dialogue with two speakers and non-verbal sounds, voices from an audio prompt with its transcript | supported, CPU and Vulkan | transformers 5.23 `DiaForConditionalGeneration` and `DiaProcessor` (CPU, float32, the original checkpoint renamed to its layout; `tools/make_dia_reference.py`; greedy with classifier-free guidance 3.0 and top-k 50; one case without and one with a 5.5 s audio prompt) | text ids and the delay table identical; teacher forced over 160 steps (no prompt) and 100 steps (audio prompt), conditional and unconditional logits of all nine codebooks: 0 argmax disagreements beyond a 0.05 margin, worst logits relative RMS 1.9e-4 (CPU) / 2.0e-4 (Vulkan) without and 2.2e-4 (CPU) with the prompt; greedy sequences identical to the reference (161 rows CPU and Vulkan, 575 rows with the prompt on the CPU), including the end tokens forced by the delay pattern; audio prompt codes 0 of 4266 differ; DAC decode of the reference frames relative RMS 3.8e-5 (CPU) / 5.9e-5 (Vulkan), 8.6e-7 with the prompt's frames; `.slm` round trip generates the same rows and audio |
| qwen3tts, pockettts | not started | - | - |

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

Qwen3-TTS loading: the model folder is enough; the Hugging Face importer reports its `speech_tokenizer/` folder (the 12 Hz codec, section 16) as the `vocoder`
companion (`IHfArchitectureMapping.FindCompanions`), and `Qwen3TtsAdapter` (composite) builds the talker with `DecoderAdapter` (embedding-row input), the code
predictor as a second decoder whose weights and KV caches live under `code_predictor.` (kept in f32: it reads the talker's hidden state, whose outliers near 100
lose precision in the 16-bit-weight paths on Vulkan), and the codec's decoder graph under `codec.`. `Qwen3TtsSynthesizer` follows the reference's non-streaming
`generate_custom_voice`: the projected text of `<|im_start|>assistant\n`, the codec control rows (think / no-think with the language or a speaker's dialect, the
speaker, pad, bos) under text pad / bos rows, the whole text and its end over codec pad rows; per step the talker's first codebook (repetition penalty 1.05, the
top 1024 codec ids except end suppressed, end not before the third frame), the code predictor's 15 codebooks, and the sum of the 16 codebook embeddings plus the
text pad row as the next input. Sampling defaults come from `generation_config.json` (temperature 0.9, top-k 50 for both); a request with temperature 0 is greedy
for both. Audio is decoded in chunks of 300 frames with 25 frames of context, as the reference; streaming decodes every 25 frames. Voices are the speakers
(`ryan`, `vivian`, ...), `SpeechSynthesisOptions.Language` one of the languages (or none / `auto`).

OmniVoice loading: the model folder is enough; the Hugging Face importer reports its `audio_tokenizer/` folder (the Higgs Audio v2 tokenizer, section 16) as the `vocoder`
companion, and `OmniVoiceAdapter` (composite) builds the backbone with `DecoderAdapter` as a bidirectional masked-diffusion model whose `denoise` graph takes embedding rows
(`decoder.embedding_input`; the 8 audio heads are the output projection) and the tokenizer's encoder and decoder graphs under `codec.`. `OmniVoiceSynthesizer` follows the
official `generate`: the style tokens (`<|denoise|>` with a reference, `<|lang_start|>..`, `<|instruct_start|>..`), the text (the reference's transcript first; non-verbal tags such
as `[laughter]` are tokenized on their own), the reference's codes and as many mask frames as the rule-based length estimate gives (`SpeechSynthesisOptions.Speed` divides it); each
of the 32 steps runs the conditional and the unconditional (target only) sequences packed in one pass, mixes their log-probabilities with guidance 2, reveals the scheduled number of
the most confident (code, position) pairs (confidence minus 5 per codebook index, optionally with Gumbel noise of temperature 5) and keeps the argmax of the unmasked codes. A
request with temperature 0 is deterministic (the default adds seeded position noise, which cannot reproduce torch's random stream, so the parity runs are deterministic). Text
longer than 30 s of speech is cut at punctuation and spoken in chunks of about 15 s (the first chunk's voice is the reference of the others when there is no recording) and
cross-faded. The audio is trimmed of long silences, matched to the reference's loudness and faded in and out like the package does (a port of its pydub-based silence removal).
`SpeechSynthesisOptions.Language` takes a language name or id (646 names), `Instructions` a voice design such as `female, british accent` (gender, age, pitch, whisper, an English
accent or a Chinese dialect; validated like the package). The cloning transcript is required (`ReferenceText`); there is no automatic transcription, no text normalization of the
package's optional kind beyond `NormalizeText`, and no built-in speakers; a voice can be saved (`IVoiceCloner`: the reference codes, transcript and loudness). The checkpoint is f32 (3.3 GB with the tokenizer).

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
Not done: qwen3tts and pockettts
(no GGUFs were looked for), and OuteTTS 0.3.

CSM loading: the checkpoint folder (`Sharplight.Import.HuggingFace`, `CsmHfMapping`, model type `csm`) maps the backbone and the depth decoder like Llama checkpoints (llama3 rope
scalings become `rope.freq_factors`; the decoders take host-composed embedding rows, with the first codebook head as stand-in output), keeps the text embedding, the codebook
embedding table (tied between the backbone's input and the depth decoder's, 32 x 2051 rows), the depth decoder's input projector and its 31 codebook heads under `csm.`, and maps the
bundled Mimi codec (`codec_model.*`, through `MimiHfMapping`) under `codec.`; the Hub tokenizer is sized by `text_vocab_size`, not the 2051-entry audio `vocab_size` (a general
change of the BPE translator: `text_vocab_size` wins when a config has it). `CsmAdapter` (canonical `csm`) builds the backbone through `DecoderAdapter` (embedding input), the depth
decoder as a second decoder under `depth.` and the Mimi decode graph; `CsmSynthesizer` embeds `<|begin_of_text|>[speaker]text<|end_of_text|>`, then per frame samples codebook 0 from
the backbone (temperature 0.9, top-k 50 from the generation config; 0 is greedy), runs the depth decoder over [projected backbone hidden state, code 0, ...] for codebooks 1-31
(positions 0-31; the heads are applied on the host), feeds the sum of the 32 codebook embeddings back, and stops on a frame whose first 31 codes are zero; audio is decoded in chunks
of 250 frames with 25 frames of context, streaming every 25 frames. Not done: audio context (earlier turns with their recordings, which need the Mimi encoder to feed frame
sums into the prompt) and so voice prompting; `ReferenceAudio` is ignored. The catalog entry is documentation only: the repository is gated and its `model.safetensors` was replaced after the
verification (the verified file is no longer at a retrievable revision).


Chatterbox loading: the checkpoint folder (`Sharplight.Import.Chatterbox`, `ChatterboxImporter`, `services.AddChatterboxImporter()`): T3's Llama is
mapped like a Hugging Face Llama checkpoint to the canonical decoder weights (the speech-token embedding is the `token_embedding`, the speech head
the `lm_head`; the llama3 rope scaling becomes `rope.freq_factors`), the text embedding, the learned text and speech positions and the conditioning
encoder keep their names under `t3.`, S3Gen under `s3gen.` (weight norms folded), the voice encoder under `ve.`; the text tokenizer (a character
BPE behind tokenizers' `Whitespace` pre-tokenizer, which the Hugging Face tokenizer translator now converts generically to the byte-level form with
the new `whitespace` pre-tokenizer preset) becomes canonical tokenizer metadata, and `conds.pt` the voice attachments `voices/default/*` (its
`prompt_feat` is a transposed view: the PyTorch reader accepts dense permuted tensors with `TorchCheckpointOptions.AllowPermuted`).
`ChatterboxAdapter` (canonical `chatterbox`) builds the decoder graphs through `DecoderAdapter` (`decoder.embedding_input` adds the multimodal graph:
every T3 row is a host-computed embedding) and four graphs of generic ops: `t3_condition` (speaker projection, the perceiver resampler over the voice's
150 prompt tokens, the emotion row), `s3gen_encode` (token embedding, pre-lookahead convolutions, six conformer layers with relative-position
attention, nearest ×2 upsampling, four more layers, the projection to 80 mel bins, and the speaker condition), `s3gen_estimate` (the causal U-Net
vector field: causal residual blocks with Mish and the time embedding, 56 transformer blocks) and `hift` (F0 predictor, nine-harmonic source with
its random draws as inputs, the source STFT as a convolution, the upsampling generator with snake residual blocks, the inverse-DFT half).
`ChatterboxSynthesizer` runs T3 twice side by side for classifier-free guidance (weight 0.5; the unconditional run has zero text embeddings), applies
the repetition penalty (1.2) to the guided logits and samples (temperature 0.8, min-p 0.05 by default), encodes the voice's 157 flow prompt tokens
plus the generated ones, runs ten Euler steps of the guided flow (cosine time schedule, guidance 0.7) from Gaussian noise, and vocodes the frames after
the prompt with a 20 ms fade-in. Not done: the
multilingual and Turbo checkpoints, the watermark (the reference adds Perth's implicit watermark; Sharplight does not).

Chatterbox voice cloning: with `ve.safetensors` in the folder, `SpeechSynthesisOptions.ReferenceAudio` (`--reference` of
`samples/Sharplight.Speak`) speaks with the voice of a recording, following `prepare_conditionals`. The recording goes to 24 kHz and then
16 kHz, and four encoders, all graphs of generic ops, derive the voice from it:
- `voice_encoder` (three LSTM layers and the projection): the speaker embedding of the 16 kHz signal (librosa's 20 dB trim, 40 mel energies via the
  new `logBase: none` of `LogMelSpectrogram`, 160-frame partials at 1.3 per second, mean of the normalized partials);
- `s3_tokenizer` (the S3 tokenizer v2: strided convolutions, six blocks with rotate-half rotary attention plus the FSMN memory, the FSQ down
  projection; the host takes tanh, scales by 0.999, rounds half to even and reads the ternary digits): T3's prompt tokens from the first 6 s (at most
  150, any length now: the condition graph's prompt is symbolic) and the flow tokens of the first 10 s;
- `campplus` (CAM++: the FCM head over `SequenceConv2d`, the dense TDNN blocks with segment-pooled context masks via `SequencePool`, statistics
  pooling): the x-vector, over the new `kaldi-fbank` feature extractor (torchaudio's `compliance.kaldi.fbank` defaults, mean subtracted);
- the flow prompt's Matcha log-mel at 24 kHz (`LogMelSpectrogram` with the new `edgePadding` and `magnitudeGuard`).
The importer adds the three filter banks as attachments; the components appear only when the voice encoder's weights are present.
`ChatterboxSynthesizer.CloneVoice` / `LoadVoice` return a `ChatterboxVoice`. The last recording's voice is kept for the next request with the same
`AudioBuffer`.

F5-TTS loading: the model is the checkpoint folder (`Sharplight.Import.F5Tts`, `F5TtsImporter`: `model_*.safetensors` with the EMA transformer under
`ema_model.transformer.`, `vocab.txt`; hyper-parameters read from the weight shapes; the character vocabulary as `tokenizer/characters.json`; the
reference front end as a log-mel component: magnitude STFT, torchaudio's HTK mel bank without normalization, ln with a 1e-5 floor, the new
`magnitude` option of `LogMelSpectrogram`) and the Vocos folder its `vocoder` auxiliary (`VocosImporter`; `TaskLoadOptions.Auxiliary =
[("vocoder", vocosFolder)]`, `--vocoder` of `samples/Sharplight.Speak`). `F5Adapter` (composite, canonical `f5tts`) builds `f5_transformer` (the
text embedding with ConvNeXt V2 blocks and filler masking, the input projection with the grouped convolutional position embedding, the time MLP, 22
DiT blocks with adaptive LayerNorm modulation and interleaved-pair rotary attention, the final adaptive norm) and `vocos` (ConvNeXt backbone and the
ISTFT head up to the inverse-DFT frames). `F5Synthesizer` follows `infer_process`: the reference recording (`SpeechSynthesisOptions.ReferenceAudio`
and `ReferenceText`, new) RMS-raised to 0.1 when quieter, the characters of reference text plus text (the reference's segmenter's space insertion
between punctuation and words reproduced for text without Chinese characters), the duration from the reference's frames and the UTF-8 byte ratio,
32 Euler steps over sway-sampled times (coefficient −1) with guidance 2 (a second evaluation without reference audio and text, keeping the filler
mask), Vocos over the frames after the reference. Not done: Chinese text (pinyin conversion), automatic transcription of the reference, chunking of
long texts.

XTTS v2 loading: the model is Coqui's checkpoint folder (`Sharplight.Import.Xtts`, `XttsImporter`: `config.json` with `model_args`, Python's bare
`Infinity` read as null; `model.pth`, a training checkpoint whose config object is built inert by the restricted unpickler (`InertModulePrefixes`),
the state dictionary under `model`; `vocab.json`, a tokenizers file whose merges list repeats pairs: the Hugging Face translator now keeps each
pair's last occurrence, as tokenizers ranks them; `speakers_xtts.pth`, the built-in speakers as `voices/{name}/latents.f32` and `speaker.f32`,
permuted tensors reordered by the new `TorchPermutation`). The GPT-2 maps to canonical decoder weights (`token_embedding` the audio-code
embedding, `lm_head` with bias, XTTS's final norm as the new optional `output_norm_2` after `output_norm`), the text and position tables stay on the
host. `XttsAdapter` (canonical `xtts`) builds the decoder graphs with `decoder.embedding_input` and `xtts_vocoder` (the HiFi-GAN with the speaker
embedding's 1x1 conditioning per upsampling stage). `XttsSynthesizer` follows `Xtts.inference`: English cleaning (`XttsTextNormalizer`), the
language token and `[SPACE]`, start 261 / stop 0; prompt rows = the speaker's 32 conditioning latents, text embedding plus learned position, the
start-of-audio row; then code by code with the model's sampling settings from `config.json` (temperature 0.75, top-k 50, top-p 0.85, repetition
penalty 5 over the codes and the reference's two placeholder ids); the final hidden states (`InferenceSession.CaptureHidden`, new) at the start row
and every code are the latents, linearly interpolated ×4 and ×24000/22050 (torch `align_corners=False`) and decoded with the speaker embedding.
Text is spoken sentence by sentence (wrapped at 250 characters). Sampling: a request whose `SpeechSynthesisOptions.Sampling` is left at its defaults
(only a seed set) now gets the model's own settings in all sampling synthesizers (XTTS, Chatterbox, Orpheus, OuteTTS); `samples/Sharplight.Speak`
passes only the seed unless `--greedy` (with its former fixed settings XTTS ran to the 605-code limit on most seeds; coqui and Sharplight give
4.5 to 6.2 s for the same sentence over six seeds). Not done: the cleaners of the 16 other languages (they are lowercased only; Chinese, Japanese and Korean need their
romanizers), decimal numbers, ordinals and currency amounts as num2words spells them, the speed option.

XTTS v2 voice cloning: `SpeechSynthesisOptions.ReferenceAudio` (`--reference`) speaks with the voice of a recording, following
`get_conditioning_latents` with the model's settings (at most 30 s at 22.05 kHz):
- `xtts_conditioning`: the style encoder (the log-mel divided by `mel_stats.pth`, the conditioning encoder's group-normalized attention blocks,
  the Perceiver resampler's 32 latents with GEGLU feed forwards and the RMS norm), averaged over 4 s chunks;
- `xtts_speaker_encoder`: the decoder's speaker embedding (Coqui's ResNet with squeeze-and-excitation blocks and attentive statistics pooling) of
  the 16 kHz resampling.
Both run over `LogMelSpectrogram` front ends (new options: the Hamming `window` and the reflect `preemphasisEdge`; the torchaudio HTK filter banks
come from the new `HtkMelFilterBank`, which F5-TTS now uses too). Parity vs coqui-tts (`tools/make_xtts_clone_reference.py`, LibriSpeech
1272-128104-0004, with the real torchaudio's Python part, `SL_TORCHAUDIO_DIR`), CPU / Vulkan:
- style latents: worst of 8 chunks 3.7e-6 / 2.7e-6
- speaker embedding: 2.8e-6 / 1.3e-6
- from the reference's signals: conditioning latents 1.9e-6, speaker embedding 3.1e-6
A cloned sentence is transcribed word for word by Qwen3-ASR-1.7B; its speaker similarity to the recording is 0.62, against 0.19 for a built-in
speaker. The Perceiver's latents grow to 1e10 before its norm, beyond f16's range; the split-f16 flash attention on Vulkan scales each query row,
key row and V tile by a power of two into f16's range before splitting (docs/benchmarks.md section 39), so it runs on the tensor cores like
the other attention.

CosyVoice 3 loading: the release folder (`Sharplight.Import.CosyVoice`, `CosyVoiceImporter`, `services.AddCosyVoiceImporter()`:
`cosyvoice3.yaml`, `llm.pt`, `flow.pt`, `hift.pt`, the Qwen tokenizer in `CosyVoice-BlankEN/`, `speech_tokenizer_v3.onnx`, `campplus.onnx`). The
language model's Qwen2 backbone maps to canonical decoder weights with the speech embedding (6561 codes plus 200 control rows) as `token_embedding`
and the speech head as `lm_head`; the text embedding (`cosyvoice3.text_embedding`) stays on the host, rows read per token. The flow keeps its names
under `flow.`, HiFT under `hift.` (weight norms folded). The two ONNX models become weights of generic graphs: the exporter names folded and
transposed tensors `onnx::MatMul_1234`, so the importer names them by the module path of the node that reads them (the new
`OnnxImporter.NodeParameters`, `/blocks.0/attn/query/MatMul` → `blocks.0.attn.query.weight`, MatMul weights transposed); the S3 tokenizer v3 is
the v2 graph (`S3TokenizerGraph`) with 12 blocks, CAM++ is `CamPlusGraph`, which now takes convolutions with an exporter-folded batch norm (a bias,
no statistics). The tokenizer gets CosyVoice 3's 280 added special tokens (`<|endofprompt|>`, `[breath]`, `[laughter]`, ARPAbet and pinyin
tokens) in the release's order. `CosyVoiceAdapter` (canonical `cosyvoice3`) builds the decoder graphs with `decoder.embedding_input`,
`cosyvoice_flow_encode` (token embedding, the pre-lookahead convolutions, ×2 repetition, the speaker projection), `cosyvoice_flow_estimator` (the
DiT: the F5-TTS blocks of `F5TransformerGraph` with a causal convolutional position embedding; CosyVoice applies the rotary embedding before the
head split, which rotates only the first head's channels, `ropeHeads: 1`), `cosyvoice_hift_f0` (the causal F0 predictor) and `cosyvoice_hift` (the
causal generator: left-padded convolutions, nearest upsampling plus convolution, strided causal downsampling of the source, the look-ahead input
convolution). `CosyVoiceSynthesizer` follows `inference_zero_shot` (`ReferenceAudio` and `ReferenceText`: the system prompt, the reference text
and `<|endofprompt|>` before the text, the recording's speech tokens after the task row), `inference_cross_lingual` (`ReferenceAudio` only) and
`inference_instruct2` (the new `SpeechSynthesisOptions.Instructions`, e.g. "Please speak as fast as possible."); the voice is the recording's
speech tokens, its 24 kHz Matcha log-mel (fmax at Nyquist) and its CAM++ x-vector (`S3SpeechEncoders`, now shared with Chatterbox's cloning).
Tokens are drawn by repetition-aware sampling (`CosyVoiceRasSampler`: top-p 0.8 within the top 25, a token seen in the last 10 is redrawn from the
whole distribution), the length between 2 and 20 times the text tokens (as the reference, only the first control row is masked below the
minimum); ten Euler steps over the cosine schedule with guidance 0.7 (full attention, the non-streaming path); the causal HiFT with
`finalize=True`. The text front end is the reference's without its optional normalizers: digits spelled as inflect spells them, the text split
into utterances of up to 80 tokens; markup (`<|…|>`) passes through. Precision notes: the reference predicts F0 in float64 and accumulates the
source phase to about 1e5 rad (`2π · 480 · Σ F0 · k / rate`), where a float32 F0 rounding of 1e-6 moves a harmonic's phase by up to a radian, so
the waveform on Sharplight's own F0 differs by 4e-3 (CPU) to 1e-2 (Vulkan) relative RMS while it matches to 2e-5 on the reference's F0; the
harmonics are computed on the host (`CosyVoiceVocoderGraph.Harmonics`, the running sum in double as torch's CPU cumsum) because a device's
parallel scan and fast sine of such phases are a hundredth off. The release's random tables (the flow's starting noise, the source's uniform
noise) are seeded draws here, inputs of the graphs. Not done: built-in voices (the release has no `spk2info.pt`), the streaming path (chunked
flow with the static 50-frame chunk mask and the cached HiFT), the ttsfrd / wetext normalizers, Japanese kana conversion. The reference script
needs a transformers 5 fix: `Qwen2Encoder.forward_one_step` passes a one-element attention mask with the cache, which transformers 4.51 ignored
and transformers 5 applies (attending to the new row only); the script drops it.

### Streaming, saved voices and text normalization (speech round 4)

Streaming. `CosyVoiceSynthesizer.MelChunked` is the flow of the reference's `streaming=True`: the estimator graph `cosyvoice_flow_estimator_chunked` takes row spans (key rows
`[0, end of the query's 50-frame chunk)`, the `Attention` op's span input) so attention is block-causal over chunks of 25 tokens; the encoder sees its three look-ahead tokens
(the last three tokens of a partial run only serve as look-ahead and produce no frames); the vocoder runs over the whole mel so far and the last eight frames are held back
(`mel_cache_len`), so the chunks add up to the one-shot chunked output (bit-identical on the CPU, relative RMS 9e-2 on Vulkan where the growing prefixes run other kernel shapes and
the harmonic source amplifies rounding). XTTS emits per sentence and, within it, from the vocoder over the GPT latents so far (after 12 codes, then every 20, four rows held back;
identical to the one-shot audio). Chatterbox decodes chunks of speech tokens (15, then 25 per chunk, 8 tokens of context before and 3 after) and cross-fades them (the S3Gen flow
is not causal, so this is an approximation of one-shot synthesis rather than an equivalent). Measured on Vulkan (RTX PRO 4500, a shared and busy machine): CosyVoice 3 first audio after 1.5 s for 9.2 s of speech (whole utterance 20.3 s), XTTS 0.19 s for 6.3 s (whole 5.2 s), Chatterbox 1.2 s for 7.4 s (whole 11.9 s).

Saved voices (`IVoiceCloner`, `VoiceProfile`, `VoiceLibrary`). Tensors per family: XTTS `latents` [rows, 1024] and `speaker`; Chatterbox `speaker`, `prompt_tokens`, `flow_tokens`,
`flow_mel`, `flow_speaker`; CosyVoice 3 `speech_tokens`, `prompt_mel`, `xvector` (and the reference text); F5-TTS `reference_audio` (24 kHz mono) and the reference text. Synthesizing
with the saved voice gives exactly the samples of synthesizing with the recording (checked for XTTS, Chatterbox and CosyVoice 3 on Vulkan after a save and reload).

Text normalization (`Sharplight.Speech.Normalization`): rule-based number, ordinal, date, clock time, currency, percent and abbreviation expansion for EN, DE, FR, ES, IT, PT, ZH and
JA with the number words of each language (English without "and"; German compounds and "ein" before nouns; French soixante-dix/quatre-vingts/et un; Spanish apocope and
"mil millones"; Italian elision and the accent of tré; Portuguese "e" rules and Brazilian scales; Chinese 零 and 两; Japanese kanji numerals with 万/億). Grammatical gender and case are not
modelled (21 is always the masculine form), and these are the essentials of wetext and num2words, not their complete rule sets.

### 12.1 VibeVoice (2026-11-18)

Microsoft's VibeVoice family shares two speech tokenizers (24 kHz audio, one latent per 3200 samples = 7.5 Hz): an acoustic one (a VAE: encoder and decoder, 64-wide
continuous latents) and a semantic one (encoder only, 128-wide). Both are the same causal stack: a stem convolution (kernel 7), seven stages of ConvNeXt-style blocks (RMS norm,
depthwise convolution of kernel 7, layer scale, RMS norm, a 4x feed-forward with GELU, layer scale; 3, 3, 3, 3, 3, 3 and 8 blocks, 32 channels doubling at every stage) and
strided convolutions of kernel 2r between the stages (r = 2, 2, 4, 5, 5, 8), a head convolution; the decoder mirrors it with transposed convolutions. `VibeVoiceTokenizerGraph`
emits them as generic ops; every causal convolution reads its left context from a graph input (`acoustic.cache.N`) and writes the next run's to an output, so a stream of runs
(one frame per run in generation: `VibeVoiceConvCaches` carries the contexts) equals one run over the whole signal, the way transformers' padding cache works; the graphs
cover a fixed number of frames (1 for generation, 64 for encoding whole recordings). Verified against transformers' `VibeVoiceAcousticTokenizerModel` and the semantic encoder on 6 s
of speech (jfk.wav, normalized to -25 dBFS): acoustic latents 3.0e-6, semantic latents 6.9e-6 and the acoustic decoder's waveform 1.7e-6 relative RMS on the CPU; Vulkan 2.5e-4, 3.4e-3
(the semantic encoder's 26 blocks over 2048 channels amplify the GPU's 16-bit matrix products; its latents only feed the connectors) and 2.0e-4.

- **Text to speech** (`vibevoice`, canonical `vibevoice`, `VibeVoiceAdapter`; `Sharplight.Import.HuggingFace`, `VibeVoiceHfMapping`): the Qwen2 language model is built by
  `DecoderAdapter` over embedding rows (like CSM and Qwen3-TTS); the head (`VibeVoiceDiffusionGraph`: latent and condition projections, a timestep MLP, four adaptive-RMS-norm gated
  feed-forward layers, a final modulated layer), the tokenizers and the two connectors (linear, RMS norm, linear) are graphs of their own components; the host keeps the token
  embedding rows, the latent scale and bias, the DPM-Solver (`VibeVoiceDpmSolver`: a port of diffusers' `DPMSolverMultistepScheduler`, cosine schedule, `v_prediction`, order 2,
  midpoint, first order at the end, timesteps and sigmas identical to diffusers for 10 and 20 steps) and the classifier-free guidance (scale 1.3 by default,
  `SpeechSynthesisOptions.GuidanceScale`; steps `DiffusionSteps`, 10 by default). The prompt is the system text, one `Speaker N:` line with the speaker's acoustic latents between the audio
  start and end tokens per speaker with a voice (`SpeakerVoices`, `VoiceProfile`, `ReferenceAudio`: the recording is normalized to -25 dBFS, encoded, sampled with the
  encoder's noise (the request's seed), scaled by the model's latent scale and bias and projected by the acoustic connector), the script (`Speaker N: text` lines; lines without
  a label continue the turn before; a script without labels is speaker 0) and `Speech output:` with the audio start token. Each step takes the best of four tokens: audio start,
  audio end, audio and end of text; an audio token runs the context-free language model (one run that has seen only the audio start token and the frames so far; an audio start
  token mid-speech, as between two speakers, starts it over) and the diffusion head for the next latent, the acoustic decoder turns it into 3200 samples, and the sum of the
  acoustic connector's image of the latent and the semantic connector's image of the semantic encoder's feature of those samples is the next input. `ISpeechSynthesizer`
  (`VibeVoiceSynthesizer`), `IVoiceCloner` (`CreateVoiceAsync`), streaming in buffers of 4 frames (about half a second).
  Both layouts import to the same canonical model (`VibeVoiceHfShared.CanonicalName` maps the tensor names of either; the configurations give the same hyper-parameters, checked by
  `VibeVoiceLayoutTests` against both repositories' configurations and safetensors indexes): transformers' conversion `vibevoice/VibeVoice-1.5B-hf` (carries the Qwen2.5 tokenizer and the
  generation config with the guidance scale and steps; the catalog's variant; the tests give the same numbers as with the original) and Microsoft's `microsoft/VibeVoice-1.5B` (MIT; ships no
  tokenizer: the Qwen2.5 `tokenizer.json` and `tokenizer_config.json` must sit next to it).
- **Speech to text** (section 13.4): `VibeVoiceAsrAdapter` reuses the tokenizers' encoders and the connectors with a Qwen2 7B language model.



### 12.2 Dia (2026-11-19)

Nari Labs' Dia is a dialogue model: the text is bytes (`[S1]` and `[S2]` are the bytes 1 and 2, non-verbal sounds such as `(laughs)` are plain text the model learned), the audio
is nine codebooks of the 44.1 kHz Descript Audio Codec at 86 frames per second, and the decoder predicts all nine codebooks of a frame at once with the codebooks delayed by
`[0, 8, 9, 10, 11, 12, 13, 14, 15]` frames (a frame's codebook c is the token `delay[c]` rows after its first codebook).

- **Loading** (`dia`, canonical `dia`, `DiaAdapter`, a composite adapter; `Sharplight.Import.HuggingFace`, `DiaHfMapping`): both layouts of the network import to the same canonical
  weights: the original checkpoint of the `dia` package (Dia-1.6B: a `config.json` with `data` and `model` sections and no model type, projections as `[in, heads, head dim]`, a
  fused `wi_fused` `[in, 2, hidden]`, one embedding table per codebook; the tensors are transposed on first read, the nine embedding tables concatenated) and the transformers port
  (`model_type` `dia`, as in Dia-1.6B-0626: fused `gate_up_proj`, one embedding table; zero-copy views). The model needs no tokenizer. The DAC codec (Hugging Face `DacModel`, 44.1 kHz,
  9 codebooks of 1024 entries; the existing `DacHfMapping` and `DacGraph`) is the vocoder companion, found in a `dac/` (or `audio_tokenizer/`) folder of the model folder; the catalog
  installs `descript/dac_44khz` as the `vocoder` companion. The importer takes the `dia` package's sampling defaults (guidance 3.0, temperature 1.3, top-k 35, top-p 0.95) or the
  port's `generation_config.json`.
- **Graphs** (`DiaGraph`): the encoder (rotary attention without score scaling, RMS norm, SwiGLU, bidirectional), the decoder step (the sum of the nine codebook embeddings, grouped-query
  self-attention over a KV cache, cross-attention over the encoder rows whose keys and values are projected once per text into states and read through row spans, SwiGLU, one
  head `[9 · 1028, 2048]`), and the DAC encoder and decoder graphs (`codec.`). Text of up to 1024 bytes; the decoder takes up to 3072 + 15 rows (about 35 s).
- **Synthesis** (`DiaSynthesizer`: `ISpeechSynthesizer`, `IVoiceCloner`, streaming): two decoder streams, the conditional one on the text and the unconditional one on a text of zeros
  (classifier-free guidance, `SpeechSynthesisOptions.GuidanceScale`, 3.0 by default; `Sampling`'s temperature, top-k and top-p; temperature 0 is greedy). `DiaTokenSampler` is the
  reference's chain of logits processors: the guided scores `cond + scale · (cond - uncond)` (restricted to the top-k of the guided scores), the temperature (not below 1), only the first
  codebook may end the audio (its end token is forced when it is the best choice, suppressed otherwise), top-k and top-p, and the end tokens of the later codebooks forced after their
  delays; a codebook that has ended continues with padding. A voice is an audio prompt: the DAC codes of the recording (`CreateVoiceAsync`, or `ReferenceAudio` with
  `ReferenceText`) go into the decoder's prompt rows after the start row (delayed like generated frames), and the transcript, written with the same `[S1]`/`[S2]` tags, goes in front of the
  text. One recording gives the voices of both speakers: there is no separate voice per speaker, and the Speak page shows the tag syntax instead of the speaker list. Text that does not
  start with a speaker tag gets `[S1] `; the text is not normalized (the model was trained on raw transcripts). Streaming decodes every 86 completed frames (a frame is complete when the last
  codebook's delay has passed) with 16 frames of context before and after, since the codec's convolutions are not causal.
- Not done: the transformers port's Dia-1.6B-0626 checkpoint was not downloaded (6.4 GB; its mapping is covered by the shared canonical weights, not by a parity run), and long texts are not
  split into chunks (a request over 1024 bytes is rejected).

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
| Qwen3-ASR (1.7B), Voxtral Mini 3B: audio language models | Hugging Face (`qwen3_asr` original and `-hf` layouts, `voxtral` with `tekken.json`); transcribe.cpp `qwen3_asr` GGUF | **supported** (CPU and Vulkan), section 13.2 | transformers (token-exact greedy ids) | hf/Qwen3-ASR-1.7B[-hf], speech-gguf/Qwen3-ASR-1.7B-F16.gguf, hf/Voxtral-Mini-3B-2507 |
| Nemotron 3.5 ASR streaming 0.6B (`nemotron3_5_asr`) | Hugging Face (transformers format); `.slm` | **supported** (CPU and Vulkan; offline, default lookahead of 3 rows, 128 language prompts); round 5, section 13.3 | transformers (log-mel, encoder output, token-exact greedy RNN-T ids) | hf/nemotron-3.5-asr-streaming-0.6b |
| Voxtral Mini 4B Realtime (`voxtral_realtime`) | Hugging Face (`tekken.json`; the `consolidated.safetensors` copy is ignored); `.slm` | **supported** (CPU and Vulkan; offline mode, one window of up to about 55 s per request); round 5, section 13.3 | transformers (projected audio embeddings, token-exact greedy ids incl. pad / word markers) | hf/Voxtral-Mini-4B-Realtime-2602 |
| VibeVoice-ASR 7B (`vibevoice` original layout with `VibeVoiceForASRTraining`; two speech tokenizers and connectors feeding a Qwen2 7B decoder), speaker-diarized long-form transcripts with timestamps | Hugging Face (Microsoft's original layout; the tokenizer files of `microsoft/VibeVoice-ASR-HF` next to it); `.slm` | **supported** (CPU and Vulkan); 2026-11-18, section 13.4 | transformers 5.23 `VibeVoiceAsrForConditionalGeneration` (audio rows, token-exact greedy ids) | vibevoice/VibeVoice-ASR, jfk.wav, a 17 s two-speaker conversation |

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

### 13.1 Speech popularity sweep (2026-11-16)

The most downloaded `automatic-speech-recognition` and `text-to-speech` models of the Hub (sorted by downloads, cross-checked with the
`text-to-audio` list), tried in the formats they are distributed in. References: transformers 5.23 (greedy, float32, CPU) in a venv; clips: jfk.wav,
LibriSpeech 1272-128104-0004, and FLEURS test clips in German, Spanish and Japanese. "Same text" compares case- and punctuation-insensitively.

| Model | Format | Result |
|---|---|---|
| openai/whisper-large-v3 | HF | same text as transformers on all 5 clips (CPU of the reference, Vulkan) |
| handy-computer/whisper-large-v3-gguf | GGUF Q5_K_M | same text as transformers on all 5 clips |
| facebook/wav2vec2-base-960h | HF | **new**: CTC ids and text identical to transformers on jfk and LibriSpeech, CPU and Vulkan (`Wav2Vec2ParityTests`) |
| jonatasgrosman/wav2vec2-large-xlsr-53-japanese | HF `pytorch_model.bin` | **new**: layer-normed feature encoder and pre-norm layers; text identical to transformers on the Japanese clip |
| handy-computer/parakeet-tdt-0.6b-v3-gguf | transcribe.cpp GGUF F16, Q8_0 | **new**: F16 text identical to the HF checkpoint on 5 clips, tokens identical on jfk and LibriSpeech, CPU and Vulkan (`ParakeetGgufParityTests`); Q8_0 differs in one capital letter (German) and a few Japanese kana |
| handy-computer/parakeet-ultra-gguf | GGUF Q8_0 | **new**: the `vad.*` head is ignored; English and Spanish transcripts correct (no reference run) |
| handy-computer/parakeet-unified-en-0.6b-gguf | GGUF Q8_0 | **new**, partial: RNN-T head with biases and input scaling; transcripts match the reference text, no NeMo reference |
| handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf | GGUF | rejected with a reason: layer-norm convolution module, causal convolutions, chunked-limited attention (cache-aware streaming FastConformer); the Hugging Face checkpoint is supported since round 5 (section 13.3), the GGUF mapping is not written |
| Qwen/Qwen3-ASR-1.7B (+ transcribe.cpp GGUF) | HF, GGUF | **supported since 2026-11-16** (section 13.2): greedy ids identical to transformers |
| mistralai/Voxtral-Mini-4B-Realtime-2602 | HF | **supported since round 5** (section 13.3); the offline mistralai/Voxtral-Mini-3B-2507 since 2026-11-16 (section 13.2) |
| handy-computer/cohere-transcribe-03-2026-gguf | GGUF | unsupported architecture `cohere_asr` |
| Systran/faster-whisper-* | CTranslate2 | unsupported format (no importer) |
| Xenova/whisper-* | ONNX | graphs run through the ONNX runner (section 9), no speech-recognition task on ONNX graphs |
| coqui/XTTS-v2, ResembleAI/chatterbox, SWivid/F5-TTS | PyTorch / safetensors folders | **supported since 2026-11-16** (section 12) |
| hexgrad/Kokoro-82M | PyTorch folder | re-checked: synthesized sentence transcribed by Whisper small with one name misspelled ("Sharplugget") |
| VoxCPM2, Qwen3-TTS, CosyVoice 3, MeloTTS, SpeechT5, Indic Parler-TTS, KittenTTS | various | unsupported architectures (metadata only, not downloaded); each needs its own acoustic model plus codec / flow-matching decoder / vocoder |

Fixes of the sweep: wav2vec2 CTC import and `Wav2Vec2CtcAdapter` (weight-normed grouped positional convolution folded at import, both
`weight_g`/`weight_v` and parametrization names; character CTC vocabularies become a Unigram tokenizer with `|` as the space); a `parakeet`
GGUF mapping (transcribe.cpp layout, the Slaney filter bank computed from the front-end parameters since the file stores none:
`IGgufArchitectureMapping.SyntheticAttachments`, `IgnoresTensor`); the Slaney mel filter bank moved from the Hugging Face importer to
`Sharplight.Audio.SlaneyMelFilterBank` in Core. Not run: Studio Transcribe page for the new formats, `.slm` of the new models.

### 13.2 Audio language models (2026-11-16)

Audio language models recognize speech with a language model: an audio tower turns the log-mel features into embedding rows, which replace
the audio placeholder tokens of a transcription prompt in the decoder's multimodal graph, the same path the image embeddings of
vision-language models take (`decode_multimodal`, rows with mask 1 take the embedding instead of the token embedding). They load as
speech-recognition models through `ISpeechRecognizer` (`AudioLanguageSpeechRecognizer`, chosen when the encoder component is an audio tower and
the decoder component carries an audio prompt), so the Transcribe page, the audio endpoints and `samples/Sharplight.Transcribe` use them unchanged.

- **Model**: canonical `decoder` with `decoder.audio.*` keys (`AudioTowerParameters`) and `audio.*` weights. Two towers: `chunked-conv2d`
  (Qwen3-ASR: 100-frame chunks through three strided 3×3 convolutions over mel bins × frames with GELU, laid out back to back in one time-major
  map with each chunk padded to 104 rows and row masks between the convolutions so a single `SequenceConv2d` per layer keeps the chunks
  independent; the flattened channels projected, the in-chunk sinusoid positions added, the valid rows gathered; pre-norm layers whose attention
  stays within 8-chunk windows (`Attention` with row spans); final LayerNorm; GELU projector) and `whisper` (Voxtral: Whisper's encoder over one
  30 s window, every four rows concatenated before the projector). `DecoderAdapter` adds the tower's `audio_encode` graph, the multimodal graph
  and the speech-recognition description (components `encoder`, `decoder`, `tokenizer`, `feature-extractor`; `AudioLanguageModelDescription`).
- **Prompt** (`AudioPromptSettings`, written by the importer and kept in the decoder component so packages carry it): a template with
  `{context}` (the request's `Prompt`), `{audio}` (one placeholder token per embedding row) and `{language}`; the language prefix
  (`language {language}<asr_text>` for Qwen3-ASR, `lang:{code}` for Voxtral) when the request names a language; the transcript marker
  (`<asr_text>`): without a language Qwen3-ASR writes `language English<asr_text>…`, the recognizer returns the text after the marker and the
  language before it (`SpeechLanguages`, ISO 639 codes to English names, in Core).
- **Host**: Whisper's log-mel front end (128 bins; 30 s of trailing silence as the reference pads to 30 s), recordings cut into 30 s windows,
  greedy decoding (or the request's sampling) until the end-of-turn token; one segment per window.
- **Importers**: `Qwen3AsrHfMapping` (both layouts of Qwen3-ASR: the original release with `thinker.*` tensors and `thinker_config`, which
  `HfConfig` now flattens, and transformers' `-hf` conversion), `Qwen3AsrGgufMapping` (transcribe.cpp's `qwen3_asr` GGUF files: `stt.qwen3_asr.*`
  keys, `enc.*` / `dec.*` tensors), `VoxtralHfMapping` (Voxtral; the tokenizer comes from `tekken.json`, translated to a byte-level BPE whose
  merges are recovered from the ranks, special tokens first, pre-tokenizer `tekken`).

| Check (fixtures from `tools/make_qwen3_asr_fixtures.py`, transformers 5.23, float32) | Result |
|---|---|
| Qwen3-ASR 1.7B, greedy ids vs transformers `generate`, language detected and English forced, jfk.wav and LibriSpeech 1272-128104-0004 | identical, CPU and Vulkan (8 of 8) |
| Qwen3-ASR 1.7B, projected audio embeddings | LibriSpeech within 2e-3 of the largest value; jfk max abs error 4.8e-4 of max 0.099 (CPU and Vulkan agree; the log-mel of jfk is within 6e-4 of transformers' near the recording's peak, LibriSpeech within 1.4e-4) |
| Qwen3-ASR 1.7B, `-hf` layout and transcribe.cpp F16 GGUF (jfk, Vulkan, language detected) | ids identical to transformers (the GGUF file types `<asr_text>` as a normal token; the mapping makes the added tokens user-defined) |
| Voxtral Mini 3B, greedy ids vs transformers (mistral-common transcription template), without a language and with English, jfk and LibriSpeech | identical, CPU and Vulkan (8 of 8) |
| Voxtral Mini 3B, projected audio embeddings (375 rows per 30 s window) | CPU within 2e-3 of the largest value; Vulkan within 1.2e-2 (bf16 weights) |

Tests: `Qwen3AsrParityTests` (features, embeddings, greedy ids, the `-hf` layout and the GGUF file), `VoxtralParityTests` (embeddings, greedy ids);
fixtures from `tools/make_qwen3_asr_fixtures.py` and `tools/make_voxtral_fixtures.py`. Not done: the Qwen3-ASR forced aligner (timestamps),
Qwen3-ASR 0.6B and Voxtral Small runs, long-form context across windows, a `.slm` round trip of these models.

### 13.3 Streaming speech models (round 5, 2026-11-18)

Two models of the hot list that are built for streaming, run here in transformers' offline mode (the whole recording as one input, which is what
the reference produces; the cache-aware chunked form computes the same values).

**Voxtral Mini 4B Realtime** (`voxtral_realtime`, `VoxtralRealtimeHfMapping`): decoder architecture with the new `causal` audio tower
(`CausalAudioTowerGraph`, `AudioTowerParameters.Causal`): two causally padded `Conv1d` layers (strides 1 and 2, GELU), 32 rotary RMS-norm
layers (SwiGLU, biases on q, v, o and the MLP's down projection, no key bias, head size 64 while the width is 1280) whose attention sees the
last 750 rows (row spans), the output RMS norm, four rows concatenated and a bias-free two-layer projector. The decoder is Mistral with
`audio.additive`: the multimodal graph adds the audio row of the step to the token embedding instead of replacing it. The model conditions its
layers on the transcription delay through adaptive RMS norms whose scale is a constant of the model (the sinusoid of the delay in tokens
through `linear2(gelu(linear1(.)))`), so the importer folds `1 + scale` into the post-attention norm weights and the decoder is an ordinary
Mistral graph. `CausalAudioSpeechRecognizer` pads the recording as mistral-common does (32 silent steps before, whole 80 ms steps, delay + 11
steps after; log-mel with the global peak `logPeak` 1.5), encodes it in one pass (one row per 80 ms step) and runs one decoder step per row:
the prompt is the start token and 38 pad tokens, every later step feeds the previous token. The output interleaves pad and word-marker tokens
with the words; the transcript is the decoded text. Recordings longer than the window (`window_frames` 6000, about 55 s of audio) are cut into
consecutive windows, one segment each, without carrying the encoder state across (the encoder is causal, so a stateful form is possible, not
done); the delay is fixed at the checkpoint's default (6 steps, 480 ms), and the importer ignores the `consolidated.safetensors` copy.

**Nemotron 3.5 ASR streaming 0.6B** (`nemotron3_5_asr`, `NemotronAsrHfMapping` onto `conformer_asr`): the Parakeet encoder with the streaming
changes as keys of `ConformerEncoder`: causal subsampling (padding of kernel - 1 before and stride - 1 after on both axes, `SequenceConv2d`
already pads asymmetrically), causal depthwise convolutions with a LayerNorm in the convolution module (`encoder.conv.norm` "layer"), and
chunk-limited attention: `RelativeAttention` got an optional sixth input of row spans (i32 [rows, 2]; CPU and Vulkan kernels), the host
(`SpeechEncoder`) fills it from the chunk of lookahead + 1 rows and 14 chunks of left context. NeMo's front end without per-channel
normalization (128 mel bins), the RNN-T prediction and joint networks, and the language prompt: the one-hot of the request's language (locale or
code, "auto" by default) goes through `prompt.linear_1` (split into the part over the encoder rows and the part over the one-hot) and
`prompt.linear_2` before the joint projection. Lookahead of 3 rows (the checkpoint's default; 0, 6 and 13 are trained too, not exposed yet).
Not done: NVIDIA's own q8_0 GGUF, the other lookaheads, feeding audio incrementally (windows are 30 s as for the other transducers).

| Check (jfk.wav 11 s, LibriSpeech 1272-128104-0004 29 s) | Result |
|---|---|
| Voxtral Realtime greedy ids vs transformers (`generate`, float32; 187 / 417 steps) | identical, CPU and Vulkan (4 of 4) |
| Voxtral Realtime projected audio embeddings | max abs error 1.8e-5 (CPU) and 1.2e-3 (Vulkan) of values up to 10.4 |
| Nemotron 3.5 encoder rows vs transformers | max abs error 2e-6 of 0.9, CPU and Vulkan |
| Nemotron 3.5 log-mel features | max abs error 1.3e-4 of values up to 16.6 |
| Nemotron 3.5 greedy RNN-T ids and text | identical, CPU and Vulkan (4 of 4) |

Tests: `VoxtralRealtimeParityTests`, `NemotronAsrParityTests`, and the two models in `SpeechPackageTests` (a `.slm` round trip transcribes like
the source); fixtures from `tools/make_voxtral_realtime_fixtures.py` and `tools/make_nemotron_fixtures.py`. CohereLabs/cohere-transcribe-03-2026 was
not reached: the Hub repository is gated and the machine's token has not been approved for it (the reference could not be downloaded).

### 13.4 VibeVoice-ASR (2026-11-18)

Microsoft's VibeVoice-ASR transcribes long recordings in one pass and says who speaks when: the acoustic and the semantic speech tokenizer of section 12.1 (the encoders only; the
acoustic decoder in the checkpoint is not imported) turn 24 kHz audio into 64- and 128-wide latents at 7.5 per second, the two connectors (linear, RMS norm, linear, summed) project them to the
width of a Qwen2 7B language model, and the rows replace the `<|box_start|>` placeholders of a chat prompt (`<|object_ref_start|>` ... `<|object_ref_end|>`, "This is a N seconds audio, please
transcribe it with these keys: Start time, End time, Speaker ID, Content"); the model writes `assistant` and a JSON array of `{"Start":..,"End":..,"Speaker":..,"Content":".."}` objects.

- **Model**: `VibeVoiceAsrAdapter` (canonical `vibevoice_asr`, `VibeVoiceAsrHfMapping`; config class `VibeVoiceForASRTraining` has `model_type` `vibevoice` like the synthesis model, so the importer
  reports `vibevoice_asr` for it): the language model through `DecoderAdapter` over embedding rows, the audio encoder graph `asr.encode` (64 frames per run with the convolutions'
  contexts carried from run to run, the whole of transformers' 60 s chunking with a padding cache, exactly; the recording is normalized to -25 dBFS first) as a component of its own.
  The acoustic latents are the encoder's mean: transformers adds the training-time sampling noise by default (`sample=True`), which makes transcripts vary between runs; the reference
  runs with `sample=False`.
- **Task client**: `VibeVoiceRecognizer` (`ISpeechRecognizer`, so also `ISpeechToTextClient` through `SharplightSpeechToTextClient`): greedy by default, segments as soon as each JSON object is
  complete (`TranscribeStreamingAsync`), `TranscriptionSegment.Speaker` (new: the model's speaker number as text) and, with the Microsoft.Extensions.AI client, the `speaker` additional property
  of each update; `SpeechRecognitionOptions.Prompt` is the "extra info" of the prompt. One pass covers as much audio as the session's context holds (the audio rows take at most 40 % of it: 7.5
  rows per second, so about 7 minutes at the default 8192 context; `TaskLoadOptions.ContextLength` raises it); a longer recording runs in windows whose speaker numbers are independent.
  Not done: voice activity pre-segmentation (`UseVoiceActivityDetection` is ignored), translation, word timestamps.
- **Verification** (transformers on CPU in float32 with the original checkpoint renamed to its layout, `tools/make_vibevoice_asr_reference.py`, greedy, `sample=False`): the prompt ids are identical;
  the audio rows 4.9e-6 (CPU) / 1.3e-3 (Vulkan) relative RMS on jfk.wav (83 rows) and 6.0e-6 / 2.2e-3 on a 17.2 s conversation of two speakers (129 rows; the reference run of the
  synthesis model of section 12.1 spoke it); the greedy tokens are identical (51 of 51 on jfk.wav, 142 of 142 on the conversation, CPU and Vulkan), teacher forced along them there are no
  argmax disagreements beyond a 0.05 margin and the worst top-logit delta is 4.2e-2 / 4.5e-2 (jfk.wav) and 1.5e-1 / 1.6e-1 (conversation; logits up to about 40, bf16 weights); jfk.wav
  is one segment of speaker 0 (0.0-11.0 s), the conversation four segments of speakers 0, 1, 0, 1 (0.0-5.41, 5.41-8.98, 8.98-11.86, 12.0-17.2 s) with the script's words; the model converted to a
  `.slm` package transcribes identically. End to end: a four-turn two-speaker script spoken by this repository's VibeVoice-1.5B synthesizer on Vulkan (11.9 s, no voice prompts) is
  transcribed with two speakers and a word error rate of 3.4 % ("I'm doing well" for "I am doing well").
- Layouts and catalog: both import to the same canonical model (`VibeVoiceLayoutTests`); the catalog's variant is `microsoft/VibeVoice-ASR-HF` (transformers' conversion with the Qwen2.5 tokenizer, 16 GB); the original `microsoft/VibeVoice-ASR` ships no tokenizer files (copy `tokenizer.json` and `tokenizer_config.json` next to it).

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
| SegFormer (`nvidia/segformer-b0-finetuned-ade-512-512`, b5 ADE 640: mix-transformer encoder with spatial-reduction attention and depthwise Mix-FFN, all-MLP decoder with the batch norm folded into the fusion convolution) | HF `segformer` | image-segmentation-semantic | supported: `ISemanticSegmenter.SegmentAsync` (label map at the input size, class names, per-class masks, optional logits; a single-channel model returns an alpha matte) | b0: logits 5.1e-4, label map 99.81 % identical; b5: logits 1.5e-6 (CPU) / 5.3e-7 (Vulkan), label map 99.998 % / 100 % |
| BiRefNet / RMBG-2.0 (`ZhengPeng7/BiRefNet`, `briaai/RMBG-2.0`: Swin encoder on the image and its half-size copy, deformable-ASPP decoder with the image's own patches injected per scale, gradient attention; batch norms folded) | HF remote-code class `BiRefNet` (`birefnet`) | image-segmentation-semantic (single-channel matte, alpha in `SemanticSegmentation.Alpha`) | supported: Studio's Semantic mode exports the alpha cut-out | matte logits 1.9e-6 (BiRefNet) / 2.3e-6 (RMBG-2.0) relative rms, CPU and Vulkan |
| Florence-2 base (`florence-community/Florence-2-base`: DaViT tower with window and channel attention, 2-D position and temporal embeddings in the projector, image rows merged into the BART encoder input) | HF `florence2` (`encoder_decoder` with `encoder.input` "image_text") | ocr (`ITextRecognizer.RecognizeAsync` with `TextRecognitionOptions.Task` / `Input`; boxes from `<loc_N>` tokens in `Regions`) | supported: `<OCR>`, `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>` and the other prompt tasks as text | image features 6.3e-4, encoder 1.1e-3; greedy text and boxes identical to transformers (num_beams=1) |
| Ultralytics YOLOv8 / YOLO11 / YOLOv10 detection and instance segmentation (`yolov8n`, `yolov8n-seg`, `yolov10n` ONNX exports: the exported graph runs as the detect graph; letterbox preprocessing, score threshold, class-aware NMS (IoU 0.7 by default) and instance masks from the prototypes in the task layer) | `UltralyticsImporter` (`Sharplight.Import.Ultralytics`, recognizes the export by its initializer names, class names from the `names` metadata, COCO names for 80-class exports without it) | object-detection (`DetectedObject.Mask` for segmentation models) | supported: `IObjectDetector.DetectAsync`; the `.pt` checkpoints pickle Ultralytics classes and are not read (export with `yolo export format=onnx`); Ultralytics weights are AGPL-3.0 licensed (enterprise license available), check the license of a checkpoint before shipping it | same detections as onnxruntime plus Ultralytics' decoding (classes, scores within 0.01, boxes within 2 px), masks IoU 0.994-1.000, CPU = Vulkan, also from an `.slm` package |
| YOLOS (`hustvl/yolos-tiny`, `yolos-small`: ViT over class, patch and 100 detection tokens, DETR heads; the small/base models add per-layer mid position tables) | HF `yolos` (`YolosHfMapping`, `YolosAdapter`) | object-detection | supported on the fixed 640 px canvas (position tables resampled bicubically at import) | same detections as transformers (boxes within 2 px, scores within 0.02), CPU = Vulkan |
| HF ResNet (`microsoft/resnet-50`, `resnet-18`: embedder, basic or bottleneck stages, ImageNet label names) | HF `resnet` (`ResNetHfMapping`, same canonical `resnet` adapter as timm) | image-classification | supported | logits within 2e-3 of transformers, same top-1 (CPU and Vulkan) |
| ConvNeXt V1 (`timm/convnext_tiny.fb_in1k`, `facebook/convnext-tiny-224`: 4x4 stem, 7x7 depthwise blocks with layer scale, layer-norm downsamplers) | HF `convnext` and timm `convnext_*` (`ConvNextHfMapping`, `ConvNextClassifierAdapter`; atto to xlarge by name) | image-classification | supported (only tiny was run; other sizes share the code) | logits within 2e-3 of timm / transformers, same top-1 (CPU and Vulkan) |
| MobileNetV3 (`timm/mobilenetv3_small_100.lamb_in1k`, `mobilenetv3_large_100.ra_in1k`: hard-swish, squeeze-excite with hard-sigmoid gates, depthwise 3x3 / 5x5 blocks, head convolution after the pooling; batch norms folded) | timm folder (`timm_mbconv`, `MbConvHfMapping`; canonical `mbconvnet`, `MbConvClassifierAdapter`; the 050 / 075 widths map the same way, not run) | image-classification | supported (no label names in the repositories) | logits 7.5e-5 (small) / 1.0e-4 (large) of timm, same top-1 (CPU and Vulkan; both also from an `.slm`) |
| EfficientNet B0 and EfficientNetV2-S (`timm/efficientnet_b0.ra_in1k`, `timm/tf_efficientnetv2_s.in21k_ft_in1k`: MBConv, fused-MBConv (edge residual) and plain 3x3 blocks, SiLU, squeeze-excite, TensorFlow "same" padding computed for the fixed 300 px input of the `tf_` models) | same mapping (b1 to b4 map by the same code, not run) | image-classification | supported | logits 8.3e-5 (B0) / 1.4e-4 (V2-S) of timm, same top-1 (CPU and Vulkan) |
| DINOv3 ViT (`facebook/dinov3-vits16-pretrain-lvd1689m`, `vitb16`: 2-D rotary positions on the patch rows (patch centers in [-1, 1], tables built into the graph), class token and 4 registers, LayerScale, optional gated MLP) | HF `dinov3_vit` (`VitHfMapping`, canonical `vit` with `vit.rope_theta`; gated repositories: accept the terms, `HF_TOKEN`) | image-embedding | supported (CLS embedding after the final norm; the larger variants (`vitl16`, `vith16plus`, `vit7b`) map the same way, not run) | CLS embedding 1.9e-6 (S) / 2.9e-6 (B) on a 224 px input (CPU), 1.4e-6 / 1.5e-6 (Vulkan); on a 640x480 photo 6.9e-3 / 8.6e-3 because the processor resizes in float where the host resize rounds to 8 bits per pass (a 1 / 255 pixel step moves this model's embedding that much; the same effect measured with PIL gives 6.8e-3) |
| Depth Anything 3 (`depth-anything/DA3-SMALL`, `DA3-LARGE`; `DA3-BASE` maps the same way, not run: DINOv2 backbone whose layers from `alt_start` alternate local and global attention with 2-D rotary positions, per-head query / key layer norms and a camera token in place of the class token; four tapped layers with local and final features concatenated; DualDPT head, depth branch only) | `Da3HfMapping` (`Sharplight.Import.HuggingFace`: the authors' `config.json` with a `config` object and no `model_type`; the 37 x 37 position table is resampled to the 504 px canvas as DINOv2 does), canonical `da3_depth` (`Da3DepthAdapter`) | depth-estimation | supported: `IDepthEstimator.EstimateAsync` on a fixed 504 px square canvas; single image (the camera encoder / decoder and the ray branch are not read); the metric and monocular variants (`DA3METRIC-LARGE`, `DA3MONO-LARGE`: plain DPT head with a sky branch) are not supported; LARGE is CC-BY-NC-4.0 | depth map 3.1e-5 (small) / 1.8e-5 (large) of the authors' network (`tools/da3-reference.py`), CPU = Vulkan, also from an `.slm` |
| GLM-OCR (`zai-org/GLM-OCR`: CogViT-style tower of 24 blocks with per-head query / key RMS norms, axial rotary positions, 2x2 merge convolution and the GLM-4V projector; GLM decoder with M-RoPE) | HF `glm_ocr` through `Glm4vHfMapping` (the multi-token-prediction layer `layers.16` becomes the NextN draft head; `eos_token_id` lists `<|endoftext|>` and `<|user|>`, the second ends the turn) | image-to-text through the chat / generator path, and `ITextRecognizer.RecognizeAsync` (round 11: `LanguageModelTextRecognizerClient`, tasks `text`, `formula`, `table` = the prompts "Text Recognition:", "Formula Recognition:", "Table Recognition:", from the canonical `recognition.tasks`; greedy, page-level text) | supported, with multi-token prediction on the serving engine (see below) | greedy text of a three-line document image identical to transformers (`tools/glm-ocr-reference.py`), and the formula and table tasks identical too (`GlmOcrTests`), CPU and Vulkan, also from an `.slm` |
| RF-DETR (`Roboflow/rf-detr-large`: DINOv2 backbone with 2x2 windows (a class token per window, window and global layers), C2f projector over four layers, two-stage top-k query selection, 4 decoder layers with single-level deformable cross-attention and one box refinement; first of 13 query groups) | HF `rf_detr` (`RfDetrHfMapping`: the repository keeps the authors' tensor names and a fused `in_proj`), canonical `rf_detr` (`RfDetrAdapter`) | object-detection | supported on the fixed 704 px canvas; the instance-segmentation models and several feature levels are not supported | same five detections as transformers, boxes within 0.28 px, scores within 3.3e-3, CPU = Vulkan, also from an `.slm` |
| PP-DocLayoutV3 (`PaddlePaddle/PP-DocLayoutV3_safetensors`: HGNetV2-L backbone (stem with 2x2 convolutions and a max-pooled branch, light blocks with depthwise 5x5 in stages 3 and 4), RT-DETR hybrid encoder (AIFI on the last level, CSP-RepVGG FPN/PAN) plus a mask-feature pyramid merged with the stride-4 backbone features (32 prototypes), top-300 query selection whose initial boxes are the enclosing boxes of the thresholded query masks, 6 decoder layers with box heads shared across layers, a global-pointer reading-order head on the last layer's queries, per-query mask embeddings) | HF `pp_doclayout_v3` (`PpDocLayoutHfMapping`, `PpDocLayoutAdapter`, `HgNetV2Backbone`; the adapter reuses the RT-DETR builder blocks) | object-detection (`IObjectDetector`: `DetectedObject.ReadingOrder`, `DetectedObject.Mask`; objects returned in reading order; `ObjectDetectionOptions.IncludeMasks`) | supported on the fixed 800 px canvas | same ten detections in the same reading order as transformers on a synthetic page (`tools/layout-reference.py`, `LayoutDetectorTests`), boxes within 1 px, scores within 0.02, masks present; CPU, Vulkan, `.slm` |
| Mask2Former, DPT (other heads) | - | - | not started (see the sweep below) | - |

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

The Studio Vision page loads and runs SwinIR (upscale), timm ResNet-50 (classify) and RT-DETR R50 (detect) headlessly on the real engine runtime (`VisionRealModelTests`); the library recognizes timm folders by their `architecture` / `pretrained_cfg`.

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

- **Florence-2** (round 6): done for the tasks that return text or boxes (`DaViTGraph`, `encoder.input` "image_text", prompts and generation settings stored with the model, `LocationTextParser`). Missing: beam search (the
  checkpoint's default is 3 beams; the client is greedy with the no-repeat n-gram rule), the polygon / segmentation tasks (`<REFERRING_EXPRESSION_SEGMENTATION>`, `<REGION_TO_SEGMENTATION>`) and region inputs given as `<loc_N>` tokens are passed through as text only.
- **GLM-OCR multi-token prediction** (done 2026-11-19 with the multimodal serving path, kv-cache.md section 2.7): the checkpoint's `layers.16` (eh_proj, enorm, hnorm, shared norm) is the NextN draft head, compiled from `mtp_draft_multimodal` so that rows before an image row see its embedding, with the target's M-RoPE positions. On the serving engine with an image prompt the output is token-identical to the single-session path, 96 % of the drafts are accepted and decoding is 1.84x faster on Vulkan, 1.70x on the CPU (one stream; `BatchedMultimodalTests`, benchmarks.md section 52). The single-session `Generator` path has no speculation.
- **TrOCR small** (`deit` encoder with a distillation token, sinusoidal decoder positions) and **large** (BEiT encoder); BLIP captioning, Donut, manga-ocr, GLM-OCR (other encoder-decoder families); PaddleOCR (Paddle format).
- **Semantic segmentation** (round 5): the per-pixel task exists now (`ISemanticSegmenter`, kind `image-segmentation-semantic`, graph `segment_semantic`: `pixels` [1, 3, H, W] in, class scores [1, K, h, w] out; the client resamples
  the scores bilinearly to the input size and takes the arg-max, as transformers' post-processing does; K = 1 is read as a matte with a sigmoid). SegFormer (`SegformerAdapter`, `SegformerHfMapping`) is the first architecture; Studio's
  Vision page has a `Semantic` mode (colored overlay with a legend of the classes by pixel share, PNG export; a matte model exports the cut-out as PNG with alpha, `PngCodec.EncodeRgba`). The graph runs on the processor's fixed square
  size (512 / 640 / 1024 px, no aspect ratio kept). Still missing: Mask2Former / OneFormer (universal: instance and panoptic), UperNet, CLIPSeg and (BiRefNet / RMBG-2.0 matting is done, see the table).
- **Detection**: DETR checkpoints without the timm backbone, Ultralytics YOLO `.pt` files and the pose / oriented-box / classification heads (the detection and segmentation ONNX exports are done, see the table), RT-DETR v2 variants listed above. Studio's Detect mode draws the boxes; the instance masks of segmentation models are in the API result only.
- **Classification**: timm / HF families other than ResNet, ConvNeXt V1, MobileNetV3 and EfficientNet B0-B4 / V2-S (EfficientNetV2 M / L and the b5+ / `tf_` variants with other stage tables, MobileNetV2 / V4, ConvNeXt V2, RepVGG, EfficientViT, Swin / SwinV2, EVA02 `timm_wrapper`, `SiglipForImageClassification`
  such as prithivMLmods' classifiers), and `open_clip`-only repositories (timm/ViT-B-16-SigLIP, Marqo fashion models).
- **Depth / pose / features**: DPT-hybrid and DPT-large, DepthPro, Depth Anything 3, ZoeDepth, ViTPose, SuperPoint / LightGlue, SigLIP 2 NaFlex (DINOv3 embeddings are done, see the table; the dense-feature / backbone outputs are not exposed).

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
| Qwen3-TTS Tokenizer 12Hz (`Qwen/Qwen3-TTS-Tokenizer-12Hz`: Mimi encoder over 16 codebooks; decoder: split RVQ, 8-layer sliding-window (72) transformer, two ConvNeXt upsampling stages, SnakeBeta vocoder, 24 kHz) | HF `qwen3_tts_tokenizer_12hz` | audio-codec | supported; 16 codebooks | codes token-exact (2000 of 2000), decoded audio relative RMS 6e-6 (CPU and Vulkan) against the official qwen_tts modeling code |
| Higgs Audio v2 tokenizer (`audio_tokenizer/` of `k2-fsa/OmniVoice`: DAC-style acoustic encoder and decoder with snake activations, HuBERT semantic encoder (layer-averaged hidden states of the audio resampled to 16 kHz) with a convolutional adapter, 8 euclidean RVQ stages with linear projections, 24 kHz at 25 Hz; license: Boson Higgs Audio 2 Community License) | HF `higgs_audio_v2_tokenizer` | audio-codec | supported; 8 codebooks | codes token-exact (2000 of 2000), decoded audio relative RMS 2.5e-6 (CPU) / 2.9e-6 (Vulkan) against transformers' `HiggsAudioV2TokenizerModel` (`tools/audio-reference/reference_higgs_audio_tokenizer.py`) |
| Silero VAD v5 (`silero_vad_16k_op15.onnx`: learned STFT, four convolutions, one LSTM step) | `Sharplight.Import.SileroVad` | voice-activity | supported (16 kHz; the 8 kHz branch of the full export is not imported) | frame probabilities within 1e-3 of onnxruntime, segments identical to `get_speech_timestamps` on two clips (CPU and Vulkan) |
| CLAP (`laion/clap-htsat-unfused`: HTS-AT Swin audio tower with a log-mel front end, RoBERTa text tower, projections) | HF `clap` (the fusion variants are rejected) | audio-embedding | supported (10 s clips at 48 kHz; shorter audio is repeated and padded, longer audio is cut or averaged over clips) | audio and text embeddings within 2e-3 relative RMS of transformers, zero-shot logits within 0.05 (CPU and Vulkan) |
| MusicGen small (`facebook/musicgen-small`: T5 text encoder, 24-layer decoder over four EnCodec 32 kHz codebooks with the delay pattern, classifier-free guidance, non-causal EnCodec decoder) | HF `musicgen` (mono models; one canonical source holds the T5 encoder, the decoder and the codec decoder) | music-generation | supported (greedy and sampling, up to 40 s, streaming in windows) | greedy codes with guidance 3 token-exact against `generate` (60 steps, CPU and Vulkan), decoded audio within 2e-3 relative RMS |
| ACE-Step 1.5 turbo (`ACE-Step/Ace-Step1.5`: Qwen3 caption encoder (0.6 B, last hidden states), lyric embeddings, condition encoder (8-layer lyric encoder, 4-layer timbre encoder, caption projection), 24-layer diffusion transformer (2048 wide, GQA 16/8, windowed and full attention alternating, adaptive-norm blocks with cross-attention, two timestep embeddings, patch 2) trained as a flow, 48 kHz stereo Oobleck autoencoder at 25 Hz; license: MIT) | `Sharplight.Import.AceStep` (the release's folders `acestep-v15-turbo`, `vae`, `Qwen3-Embedding-0.6B`; only the distilled transformer) | audio-generation | supported: text and lyrics to music, up to 120 s per request, 8 Euler steps by default (steps and shift adjustable), `IAudioGenerator`; no reference audio, cover, repaint or lego task, no 5 Hz language-model planner, no guidance (the base and SFT transformers are rejected) | with the same noise and steps against the checkpoint's own modeling code (fp32, CPU; `tools/audio-reference/reference_acestep.py`) for a 10 s instrumental: latents within 8.1e-6 (CPU) / 2.8e-4 (Vulkan) relative RMS, decoded audio (diffusers `AutoencoderOobleck`) within 4.3e-5 (CPU) / 8.3e-4 (Vulkan); the same audio from the `.slm` package |
| Demucs (`htdemucs`: spectral and waveform U-Nets with dilated residual branches, a cross-domain transformer, complex-as-channels spectrogram output) | HF layout `htdemucs` (a safetensors conversion of the state dict with the constructor arguments as `config.json`, e.g. KeighBee/demucs; the original `.th` pickle is not read) | audio-separation | supported (4 stems, 7.8 s segments with 25 % overlap and triangular cross-fade; no Wiener filtering, no shifts, no multi-band models) | stems of a 10 s stereo mix against the demucs package's `apply_model` with the same weights: SDR above 35 dB on every stem and channel (CPU and Vulkan) |

ACE-Step's checkpoint is read as it is released (three folders: the `acestep` transformer and condition encoder in bf16, the diffusers-format autoencoder with its legacy weight-norm names, the Qwen3 embedding model). The reference run uses the checkpoint's modeling code
for everything but the text encoder (transformers' `Qwen3Model`) and the autoencoder (diffusers' `AutoencoderOobleck`); a text-only generation uses the model's silence latent as source audio and the chunk mask 2.0 (the "auto" mode of the official pipeline),
the caption prompt of the official `SFT_GEN_PROMPT` and the lyric prompt with the language header. The Oobleck decoder graph (snake activations with log-scale parameters, residual units with dilations 1, 3, 9, transposed convolutions) is the one
Stable Audio Open's autoencoder shares. Stable Audio Open 1.0 and Stable Audio 3 are not implemented: both repositories are gated and the Hugging Face account used for this round has not accepted their terms (the model card and license files are readable, the weights are not).

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
| InstructPix2Pix (timbrooks/instruct-pix2pix), SD 1.5 / SDXL inpainting UNets (stable-diffusion-v1-5/stable-diffusion-inpainting, diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | diffusers folder, the original single files (the UNet's `input_blocks.0.0` takes 8 or 9 channels) | **supported since 2026-11-18** (an editing instruction with text and image guidance; inpainting with the mask and the masked picture as UNet inputs), see [below](#sd-editing) | below |
| Stable Diffusion 2.x (2.1 768-v, 2.1 base) | diffusers folder, original single-file `.safetensors` / `.ckpt` (OpenCLIP ViT-H tower; 768 files default to v-prediction at 768×768, files named like the 512 / base releases to epsilon at 512×512) | **supported** (as SD 1.x) | below |
| SDXL (base 1.0) | diffusers folder, original (SGM) single-file `.safetensors` (CLIP-L + OpenCLIP bigG with the text projection), stable-diffusion.cpp GGUF in both naming schemes (sd.cpp's own `cond_stage_model.1.transformer.*`, e.g. gpustack/stable-diffusion-xl-base-1.0-GGUF Q8_0 / Q4_0; the SGM names, e.g. kostakoff/stable-diffusion-xl-base-1.0-GGUF Q4_K) | **supported** (two text encoders, pooled + size conditioning, `force_zeros_for_empty_prompt`) | below |
| SDXL-Turbo, SDXL-Lightning (distilled SDXL) | the SDXL formats; stable-diffusion.cpp GGUF (OlegSkutte/sdxl-turbo-GGUF, mzwing/SDXL-Lightning-GGUF) | **supported** (single files are recognized by their published names: Turbo 512×512, Euler ancestral on trailing timesteps, 4 steps, no guidance; Lightning 1024×1024, Euler trailing, the file's step count, no guidance, the 1-step "x0" file predicts the sample) | Turbo below; Lightning maps the same way, not verified |
| FLUX.1-schnell, FLUX.1-dev, FLUX.1 Kontext dev (and fine-tunes with the same layout) | diffusers folder; original single-file `.safetensors` (Black Forest Labs names, also with ComfyUI's `model.diffusion_model.` prefix); transformer-only GGUF (city96 / stable-diffusion.cpp: any ggml type) plus companion files: CLIP-L (`clip_l.safetensors`), T5-XXL (encoder GGUF with its tokenizer, or `t5xxl_*.safetensors` with the embedded T5 vocabulary), VAE (`ae.safetensors`), given as auxiliary sources or found beside the transformer | **supported** (txt2img, img2img, inpainting; Kontext: editing with one reference image, recognized by `FluxKontextPipeline` or "kontext" in the file name; flow-matching Euler with FLUX's static / resolution-dependent shift; dev's guidance embedding; any size that is a multiple of 16; `.slm`) | [below](#flux) |
| SD3 / SD3.5 (SD3 Medium, SD3.5 Medium / Large and fine-tunes) | diffusers folder; Stability's single file (`model.diffusion_model.*`, the VAE under `first_stage_model.*`); transformer-only GGUF (city96 / stable-diffusion.cpp `sd3`, any ggml type) plus companions: CLIP-L, CLIP-G (`clip_g.safetensors`), T5-XXL (encoder GGUF, `t5xxl_*.safetensors` or a sharded `text_encoder_3` folder named by a shard), the 16-channel VAE | **supported** (txt2img, img2img, inpainting; classifier-free guidance; three text encoders; any size that is a multiple of 16 up to the position grid; `.slm`) | [below](#sd3) |
| Chroma (Chroma1-HD / Base / Flash and fine-tunes) | diffusers folder; original single file (FLUX names plus `distilled_guidance_layer.*`); transformer-only GGUF (silveroxides, any ggml type) plus companions: T5-XXL, the FLUX VAE | **supported** (txt2img, img2img, inpainting; classifier-free guidance with negative prompts; masked T5 and text attention) | [below](#chroma) |
| Z-Image (Z-Image-Turbo and fine-tunes; the undistilled Z-Image since 2026-11-18, section 21) | diffusers folder; original-name single file (ComfyUI); transformer-only GGUF (unsloth / stable-diffusion.cpp, `lumina2` architecture id, any ggml type) plus companions: the Qwen3-4B text encoder (a Qwen3-4B GGUF or the pipeline's `text_encoder` folder), the FLUX VAE | **supported** (txt2img, img2img, inpainting; 9 steps without guidance by default; `.slm`) | [below](#z-image) |
| Qwen-Image-Edit (-2509, -2511, the first Qwen-Image-Edit, FireRed-Image-Edit) | diffusers folder (`QwenImageEditPlusPipeline` / `QwenImageEditPipeline`; `transformer/` weights or a transformer-only GGUF with the folder's `text_encoder/` (Qwen2.5-VL with its vision tower) and `vae/` as companions); unsloth / QuantStack GGUF files (the -2511 files hold a `__index_timestep_zero__` marker tensor, the editing variant otherwise follows the file name: "edit", with a 25xx release for the multi-image pipeline) | **supported since 2026-11-18** (editing with up to three reference images (the first Edit: one), see [below](#qwen-image-edit)) | below |
| Qwen-Image (Qwen-Image and fine-tunes with the same layout, e.g. Qwen-Image-2512) | diffusers folder; transformer-only GGUF (city96 / QuantStack / unsloth, `qwen_image` architecture id, any ggml type) and ComfyUI single files plus companions: the Qwen2.5-VL-7B text encoder (the pipeline's `text_encoder` folder or a `qwen2vl` GGUF), the causal VAE (the pipeline's `vae` file or ComfyUI's `qwen_image_vae.safetensors` in the original Wan names) | **supported** (txt2img, img2img, inpainting; true CFG with diffusers' norm rescaling; 50 steps, guidance 4 by default) | [below](#qwen-image) |
| Qwen-Image 2.1 (and its Turbo) | diffusers folder; transformer-only GGUF (unsloth: no metadata, ComfyUI names) and ComfyUI single files plus companions: the Qwen3-VL-8B text encoder (the pipeline's `text_encoder` folder or a `qwen3vl` GGUF), the pipeline's VAE file | **supported** (text-to-image; 40 steps without guidance by default; Turbo: its 8-step schedule from the pipeline's `sample_sigmas` or the file name) | [below](#qwen-image-21) |
| FLUX.2 Klein (4B; 9B and the base models share the layout), FLUX.2 Dev (32B) | diffusers folder; original-name single file (the one in the model repository, ComfyUI's); transformer-only GGUF (unsloth / leejet, the `flux` architecture id, told apart by the shared modulations) plus companions: the Qwen3 text encoder (Klein; the pipeline's `text_encoder` folder or a `qwen3` GGUF) or Mistral Small 3.2 24B (Dev; a `llama` GGUF of Mistral-Small-3.2-24B-Instruct-2506 or the pipeline's folder), the pipeline's VAE file | **supported** (txt2img and editing with up to 10 reference images; Klein distilled: 4 steps without guidance; Klein base: 50 steps, guidance 4; Dev: 50 steps with the embedded guidance 4) | [below](#flux-2) |
| Krea 2 (Turbo, Raw and fine-tunes) | diffusers folder; the release's original-name single files (`turbo.safetensors`, `raw.safetensors`, ComfyUI's bf16 files); transformer-only GGUF (realrebelai, vantagewithai, ...: architecture `krea2`, some labelled `qwen_image`) plus companions: the Qwen3-VL-4B text encoder (the pipeline's `text_encoder` folder or a `qwen3vl` GGUF), the Qwen-Image VAE (the pipeline's file or ComfyUI's `qwen_image_vae.safetensors`) | **supported** (text-to-image; Turbo: 8 steps without guidance; Raw (file names with `raw` / `base`): 28 steps, guidance 5.5) | [below](#krea-2) |
| SDXL refiner (stabilityai/stable-diffusion-xl-refiner-1.0: the OpenCLIP bigG encoder alone, aesthetic-score conditioning; as the second stage after the SDXL base or for image to image) | diffusers folder (`text_encoder_2/` and `tokenizer_2/` only) | **supported** since 2026-11-18 ("SDXL refiner" below): base + refiner 48.3 dB against diffusers | |
| Qwen-Image 2.1 image editing, Lumina-Image-2.0, HiDream, AuraFlow, MiniMax-H3-Image | - | missing (recognized in GGUF and safetensors single files and rejected as "Not supported yet: ... (diffusion transformer)") | the graph contract (`DiffusionGraphs`), the description, the flow sampler and the companion-file loading are kind-generic |
| LoRA | kohya-ss (SD 1.x diffusers names, SDXL SGM names, FLUX original names), diffusers / PEFT, ComfyUI `.safetensors` | **supported** for SD 1.x / 2.x / XL, FLUX, Chroma, Qwen-Image, Krea 2 (merged at load, role `lora[@scale]`, several at once); LyCORIS / DoRA rejected | [below](#lora) |
| ControlNet, TAESD | - | missing | |

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
- Later families: SD3 / SD3.5, Chroma and Z-Image below; Qwen-Image reuses the flow sampler, the companion-file loading and the
  decoder graph as its text encoder (as Z-Image does).

### SD3

<a id="sd3"></a>
SD3 / SD3.5 (2026-11-16). Design:

- Canonical source `sd3` (`Sharplight.Import.Diffusers`): `transformer.*` in the original MMDiT layout (fused q/k/v per stream;
  diffusers' separate projections are joined lazily, its (scale, shift) `norm_out` and last-block `norm1_context` halves swapped; the
  patch convolution becomes a linear weight [D, C·p²], the position grid `pos_embed` [G², D]), CLIP-L / CLIP-G under `text_encoder.` /
  `text_encoder_2.` (each with its projection), the T5 encoder under `text_encoder_3.`, the VAE, the CLIP tokenizer under the canonical
  keys (CLIP-G prompts reuse its tokens, padded with `!`) and the T5 tokenizer under `sd3.tokenizer_3.`; the family keys (both CLIP
  encoders' penultimate layer without the final norm, T5 prompts of 256 tokens, the VAE's 1.5305 / 0.0609 scaling, 28 steps,
  guidance 7, the scheduler's own noise levels with shift 3) are written by the importer, so transformer-only files and their
  companions get them too. SD3 files without q/k norms do not tell the head width: 64, as every release.
- `Sd3Adapter` builds the two CLIP graphs (hidden states and projected pooled embeddings), the T5 graph and the transformer
  (`Sd3TransformerGraphBuilder`: patch rows plus the centre crop of the position grid, timestep + pooled embedding as the
  modulation vector, joint blocks over [image, text] with the second image-only attention of SD3.5's first blocks and the
  text-stream-free last block, f16 attention products only with RMS-normed q/k (SD3.5): SD3's unnormed scores lose ~1 % in f16).
- Core: a third text encoder (`text-encoder-3`, `tokenizer-3`) whose hidden states are appended along the sequence
  (`DiffusionConfigKeys.JoinSequence`: the CLIP rows [77, 768 + 1280] zero-padded to T5's 4096 and followed by its 256 rows); pooled
  embeddings of all encoders concatenated in order; the flow sampler's `scheduler` spacing (diffusers' `set_timesteps` without
  sigmas: linspace from 1 to the shifted smallest training level, shifted again, as the scheduler's f32 / numpy arithmetic does);
  CFG row by row through the fixed-batch fallback.
- Companions: `Sd3CompanionAdapter` (roles `text-encoder` CLIP-L, `text-encoder-2` CLIP-G, `text-encoder-3` T5, `vae`) on the generic
  `DiffusionCompanionAdapter` (FLUX, SD3, Chroma and Z-Image declare their slots); stand-alone CLIP files are told apart by width (768 /
  1280, the latter with GELU); a T5 named by one shard of a sharded folder opens the whole checkpoint; a diffusers `text_encoder{N}`
  folder finds its tokenizer in `tokenizer{N}`.

Verification (tools/make-tiny-sd3.py, tools/make-sd3-references.py; `TinySd3Tests`, `Sd35MediumTests`; GPU 0 of the 3× RTX PRO 4500
machine, shared):

- Tiny random SD3 and SD3.5 pipelines (`~/models/sd3/tiny-random-sd3/{sd3,sd35}`: without q/k norms; with them and the second
  attention in blocks 0–1): both CLIP encoders (hidden states and pooled), T5, transformer, VAE decode and encode within 1e-4 of
  diffusers / transformers on CPU and Vulkan, from the diffusers folder, the original-name single file and the Q8_0 GGUF with
  companions (GGUF within 5e-3 of diffusers on the dequantized weights); guided 4-step txt2img above 40 dB in all 12 combinations;
  a `.slm` converted from the GGUF with companions holds them and generates the same image.
- SD3.5 Medium (diffusers folder, fp16 files, an ungated mirror of stabilityai/stable-diffusion-3.5-medium) on Vulkan against diffusers
  in f32 on CUDA, 28 steps, guidance 7, same latents: CLIP-L / CLIP-G hidden 1.3e-4 / 6e-4, pooled 2–4e-4, T5 5–11e-6, transformer
  1.4e-4, VAE 2.2e-4 relative RMS; images 46.6 dB (512) / 38.8 dB (1024), visually identical. city96's Q8_0 GGUF: transformer 1.5e-3
  from the fp16 reference; against diffusers running the dequantized GGUF transformer with the folder's encoders: 42.5 dB (512) / 29.4 dB (1024), visually identical (the Q8_0 products quantize the activations; the 28-step 1024 trajectory drifts).
- Speed (Vulkan, fp16 folder, warm, conditioning reused): 512×512 28 guided steps 2.8 s, 1024×1024 12.6 s (0.45 s per step for both
  rows); diffusers f32 on the same GPU ~1.1 s per step at 1024. First run compiles for 100–180 s without a precompiled `.slm`.

### Chroma

<a id="chroma"></a>
Chroma (2026-11-16): FLUX.1-schnell's layout without CLIP, with the "distilled guidance" approximator in place of the modulation
projections and embedders. Design:

- Canonical source `chroma`: the FLUX transformer vocabulary without `*_mod`, `time_in`, `vector_in`, `guidance_in`, plus
  `transformer.approximator.{in_proj, layers.{i}.in_layer|out_layer, norms.{i}.weight, out_proj}`; T5 under `text_encoder.` (its only
  text encoder, tokenizer under the canonical keys); family keys: T5 prompts of 512 tokens, the FLUX VAE scaling, static shift 3
  (Chroma1-HD's scheduler), 35 steps, guidance 5.
- `ChromaAdapter` reuses `FluxTransformerGraphBuilder`: with the approximator the modulation rows [3·single + 12·double + 2, D] come from
  the MLP over the sinusoidal timestep (16), guidance-0 (16) and row-index · 1000 (32, computed in f32 as diffusers does) channels, and
  the blocks slice their (shift, scale, gate) rows instead of projecting the conditioning vector; the T5 graph takes the prompt's key
  mask (`DiffusionGraphs.TokenMask`), the transformer a `DiffusionGraphs.ContextMask` (the prompt's tokens and the first padding token
  visible, `DiffusionConfigKeys.ContextMaskPadding` = 1) as a key bias of the joint attention.
- Masking follows the original Chroma code (and ComfyUI / stable-diffusion.cpp): padding keys excluded. diffusers' ChromaPipeline
  builds a float 0/1 mask, which `scaled_dot_product_attention` adds to the scores instead of masking (checked: none vs float mask
  differ, the boolean mask equals dropping the padding); the references use the boolean mask (`make-tiny-chroma.py`,
  `make-chroma-references.py`).

Verification (tools/make-tiny-chroma.py, tools/make-chroma-references.py; `TinyChromaTests`, `ChromaHdTests`):

- Tiny random Chroma (`~/models/chroma/tiny-random-chroma`): masked T5 (prompt rows), transformer, VAE within 1e-4 on CPU and
  Vulkan from the diffusers folder, the original-name single file and the Q8_0 GGUF with companions (5e-3); guided txt2img with a
  negative prompt above 40 dB in all 6 combinations.
- Chroma1-HD, silveroxides' Q8_0 GGUF with FLUX.1-schnell's T5 (sharded folder) and VAE as companions, on Vulkan against diffusers
  running the same files (dequantized GGUF transformer, f32 T5 / VAE, CUDA), 20 steps, guidance 4, negative prompt, same latents:
  T5 ≤ 3e-5, transformer 2.1e-3 relative RMS; images 46.3 dB (512) / 37.8 dB (1024).
- Speed (Vulkan, Q8_0, warm): 512×512 20 guided steps 12.8 s, 1024×1024 48 s (1.2 s per step and row); diffusers (GGUF dequantized on
  the fly, CUDA f32) 2.3 s / 7.7 s per guided step.

### Z-Image

<a id="z-image"></a>
Z-Image-Turbo (2026-11-16): a ~6B single-stream DiT conditioned on Qwen3-4B hidden states. Design:

- Canonical source `zimage`: `transformer.*` (fused `attention.qkv` per block of the `noise_refiner`, `context_refiner` and `layers`
  stacks, sandwich RMS norms, SwiGLU `ffn.w1|w2|w3`, the 4D-wide `mod` projections of the timestep embedding, `x_pad_token`,
  `cap_pad_token`, `cap_embedder`, `final.mod|linear`; diffusers' separate q/k/v joined lazily, the original / GGUF names mapped
  directly); the text encoder kept whole: its canonical decoder weights under `text_encoder.` and its own canonical metadata under
  `zimage.text_encoder_source.` (with its architecture), its tokenizer under the canonical keys; family keys: the Qwen3 chat template
  (`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`, `enable_thinking=True` adds nothing), 512 prompt tokens, the
  second-to-last hidden states, shift 3, 9 steps, no guidance, the FLUX VAE scaling. Lumina-Image-2.0 GGUF files share the `lumina2`
  architecture id; Z-Image is told apart by its bare block modulation (`adaLN_modulation.0`).
- `ZImageAdapter`: the text graph is the existing decoder graph (`DecoderGraphBuilder.BuildLayerHiddenStates`: one padded prompt,
  causal packed attention, the hidden state entering a given layer; Qwen3 runs as it does for text generation). The transformer
  (`ZImageTransformerGraphBuilder`): image patches (features (row, column, channel)) padded to a multiple of 32 with the pad token;
  the caption rows (the fixed 512) with the prompt's tokens visible through `DiffusionGraphs.ContextMask` and the rest replaced by
  the caption pad token; the padded caption length c = 32·⌈n / 32⌉ is computed in the graph (ReduceSum, Ceil) and gives the key
  mask of the rows ≥ c and the image tokens' RoPE position (c + 1, row, column), so one compiled graph serves every prompt; 3-axis
  RoPE (32 + 48 + 48, θ 256) through the `Rope` op's sections and per-frequency divisors as for FLUX; tanh-gated modulation; the
  model's timestep 1 − σ and its negated output stay inside the graph. The down projections keep f32 activations in the integer-dot
  kernels of quantized weights: their inputs reach 10⁵ (the int8 path produced NaN).
- Core: prompt templates (`DiffusionConfigKeys.PromptTemplate`, the template's special tokens parsed, the prompt's not); `Reshape`
  keeps a fixed outer dimension with 0 (decoder code written for a symbolic token axis); a generator that runs out of device memory
  compiling the denoiser releases the text encoders and keeps releasing them after encoding.
- Companions: `ZImageCompanionAdapter` (roles `text-encoder`: any `decoder` source, a Qwen3-4B GGUF or the pipeline's `text_encoder`
  folder; `vae`); found beside the transformer (Qwen3 GGUF, 16-channel VAE) and in diffusers pipeline folders next to the
  transformer's folder (their `text_encoder` and `vae`). The Hugging Face importer finds a diffusers `text_encoder{N}` folder's
  tokenizer in `tokenizer{N}`.

Verification (tools/make-tiny-zimage.py, tools/make-zimage-references.py; `TinyZImageTests`, `ZImageTurboTests`,
`ZImageImageTests` in Studio.Tests):

- Tiny random Z-Image (`~/models/zimage/tiny-random-zimage`, 64×96: 24 image tokens padded to 32): prompt tokens equal transformers'
  chat-templated ids; the text encoder's hidden states (prompt rows), the transformer and the VAE within 1e-4 on the CPU (Vulkan
  2.5e-4: f16 attention products) from the diffusers folder and the original-name single file, the Q8_0 GGUF within 5e-3 (Vulkan) /
  3e-2 (CPU) of diffusers on the dequantized weights; 4-step txt2img above 40 dB in all 6 combinations.
- Z-Image-Turbo, unsloth's Q8_0 GGUF with the pipeline's Qwen3-4B `text_encoder` folder (bf16) and VAE as companions, against
  diffusers running the same files (dequantized GGUF transformer on CUDA, f32 text encoder): text encoder 1.4e-4, transformer 1.5e-3
  (Vulkan) / 9.5e-3 (CPU) relative RMS; images 49.6 dB (512) / 46.4 dB (1024), 9 steps from the same latents.
- Speed (Vulkan, Q8_0, warm): 512×512 9 steps 4.3 s, 1024×1024 14.6 s (1.6 s per step; the text encoder 8 GB bf16 + transformer 7.3 GB on the device); diffusers (GGUF dequantized on the fly, CUDA f32) 0.65 s / 2.5 s per step. Studio: the load panel shows the Qwen3-4B and VAE slots, fills them from the pipeline folder beside the GGUF folder, and the Image page generates (tiny pipeline and Z-Image-Turbo Q8_0 at 512×512).

### Qwen-Image

<a id="qwen-image"></a>
Qwen-Image (2026-11-16): a 20B dual-stream MMDiT conditioned on Qwen2.5-VL-7B hidden states, with Wan 2.1's causal VAE. Design:

- Canonical source `qwen_image`: the transformer in the FLUX transformer vocabulary (`transformer.double.{i}.{img,txt}_{mod, attn.q|k|v|proj|
  norm_q|norm_k, mlp.up|down}`, `img_in`, `txt_norm`, `txt_in`, `time_in`, `final.mod` with diffusers' (scale, shift) halves swapped,
  `final.linear`); the q / k / v projections stay separate (city96's Q4_K_M quantizes them with different types). The text encoder is kept
  whole as in Z-Image (`text_encoder.`, its metadata under `qwen_image.text_encoder_source.`; Qwen2.5-VL's vision tower is not used). The
  VAE is the Wan 2.1 layout (`vae.*` as for Wan, `qwen_image.vae.*` with the per-channel latent normalization); stand-alone VAE files open
  as `wan_autoencoder` components, in diffusers or original Wan names (`WanVaeOriginalNames`, ComfyUI's `qwen_image_vae.safetensors`).
  Family keys: the system-prompt template, 34 skipped template tokens, 512 prompt tokens, the exponential dynamic shift (0.5 to 0.9 between
  256 and 8192 image tokens) with the terminal stretch to 0.02, 50 steps, guidance 4.
- `QwenImageAdapter`: the text graph is the decoder graph with the final norm (`BuildLayerHiddenStates(..., finalNorm: true)`, transformers'
  `hidden_states[-1]`) over 34 + 512 rows; the transformer is `FluxTransformerGraphBuilder` without single-stream blocks, pooled embedding or
  guidance, with the RMS-normed text input, separate projections, the text key mask and centred RoPE positions (image rows and columns
  around 0, text token j at max(rows / 2, columns / 2) + j on all three axes, (16, 56, 56) dimensions, θ 10000); the VAE graphs are
  `WanVaeGraphBuilder`'s first-chunk graphs as image graphs (`BuildImageDecoder` / `BuildImageEncoder`).
- Core: `DiffusionConfigKeys.PromptSkip` (the generator drops the template's rows from the conditioning and the mask) and
  `DiffusionConfigKeys.GuidanceRenormalization` (true CFG rescaled per token to the conditional prediction's norm over channels and patch
  pixels, as diffusers' QwenImagePipeline). Attention takes f16 products only where the RMS norm scales bound the scores
  (max |norm_q| · max |norm_k| ≤ 50): Qwen-Image's first block scales queries and keys by up to 63 and 48 and lost 2 % of the transformer
  output in f16 (8.4e-3 with that block exact); FLUX and Chroma stay ≤ 5 everywhere. Qwen2.5-VL's slow tokenizer files (vocab.json and
  merges.txt only) are converted as transformers' Qwen2Converter does (NFC, the Qwen2 split, the added tokens of tokenizer_config.json).

Verification (tools/make-tiny-qwen-image.py, tools/make-qwen-image-references.py; `TinyQwenImageTests`, `QwenImageTests`,
`QwenImageImageTests` in Studio.Tests):

- Tiny random Qwen-Image (`~/models/qwen-image/tiny-random-qwen-image`, 64×96, non-square RoPE grid, random latent normalization): prompt
  tokens equal transformers'; text encoder (after the skipped tokens), transformer (masked padding), VAE decode and encode within 1e-4 / 1e-3
  (Vulkan) of diffusers from the diffusers folder, the ComfyUI-named single file and the Q8_0 GGUF with companions (GGUF within 5e-3 / 3e-2);
  guided 4-step txt2img above 40 dB in all 6 combinations.
- Qwen-Image, city96's Q4_K_M GGUF with the pipeline's Qwen2.5-VL text encoder (bf16) and VAE as companions, on Vulkan against diffusers running
  the same files (dequantized GGUF transformer on CUDA, f32 text encoder), 20 steps, true CFG 4, negative prompt " ", same latents: text
  encoder 4.6–6.4e-4, transformer 8.4e-3 relative RMS; images 30.9 dB (512) / 29.8 dB (1024). Qwen-Image-2512 (unsloth's Q4_K_M, same
  companions; its blocks 0, 52 and 58 exceed the f16 bound): 25.5 dB (1024), visually identical. Studio: the load panel shows the Qwen2.5-VL and
  VAE slots and fills them (with the Qwen2.5-VL-7B Q8_0 GGUF and ComfyUI's VAE beside the transformer), and the Image page generates.
- Speed (Vulkan, Q4_K_M, warm, conditioning reused; the 13 GB transformer and the 16 GB bf16 text encoder alternate on the 32 GB GPU):
  512×512 20 guided steps 34 s, 1024×1024 131 s (3.3 s per step and both rows); diffusers (GGUF dequantized on the fly, CUDA f32) 8.2 s per
  step at 1024.

### Qwen-Image 2.1

<a id="qwen-image-21"></a>
Qwen-Image 2.1 (2026-11-16, the most downloaded text-to-image family on the Hub at the time): a 7B single-stream DiT over [text, image]
conditioned on Qwen3-VL-8B, with a 64-channel 16× residual VAE that reads and writes RGBA. Design:

- Canonical source `qwen_image21`: `transformer.{img_in, txt_in.norm (the zero-centred scale + 1), txt_in.in_layer|out_layer, time_in.*,
  modulation, blocks.{i}.attn.q|k|v|o|norm_q|norm_k, blocks.{i}.mlp.gate|up|down, final.mod, final.linear}` (ComfyUI's / unsloth's fused
  `img_mlp.gate_up` split into row blocks, gate first); the Qwen3-VL text model kept whole (`text_encoder.`, the tokenizer from the
  pipeline's `processor/` folder); the VAE in the Wan names (`WanVaeConfig`, residual, 5 levels, separate decoder width). Family keys: the
  system-prompt template with its 14-token system turn skipped, 512 prompt tokens, an empty prompt read as " ", the Qwen-Image flow schedule,
  40 steps, no guidance; Turbo: the pipeline's `sample_sigmas` (8 steps; transformer-only files named `*turbo*`).
- `QwenImage21TransformerGraphBuilder`: one token per latent pixel; the text through the RMS norm and a GELU MLP; blocks with LayerNorm ·
  (1 + scale) and tanh gates from one shared modulation projection (text rows from t = 0, image rows from the step's t); block-causal
  attention as two attention ops: text rows causal over the text (row spans), image rows over all visible text and the whole image (key
  bias); SwiGLU with f32 activations in the down projection; the scale-only final norm. RoPE: text j at (j, j, j), image tokens at
  (n, centred row, centred column) with n the visible text tokens, computed in the graph from the context mask. The text graph is the decoder
  graph's last layer output before the final norm. The VAE graphs are the Wan image graphs, which now take and drop the alpha channel of
  RGBA VAEs (opaque alpha in, colour channels out).
- Core: `DiffusionConfigKeys.EmptyPrompt` (Qwen-Image 2.1: " "), `DiffusionConfigKeys.SampleSigmas` (pipeline noise levels before the shift
  for the default step count).

Verification (tools/make-tiny-qwen-image21.py, tools/make-qwen-image21-references.py; `TinyQwenImage21Tests`, `QwenImage21Tests`):

- Tiny random Qwen-Image 2.1 (64×96, 5-level RGBA VAE): prompt tokens equal transformers'; text encoder, transformer, VAE decode and encode
  within 1e-4 / 1e-3 (Vulkan) of diffusers from the diffusers folder, the ComfyUI single file and a Q8_0 GGUF without metadata (GGUF within
  5e-3 / 3e-2); guided 4-step txt2img above 40 dB in all 6 combinations.
- Qwen-Image 2.1, unsloth's Q8_0 GGUF with the pipeline's Qwen3-VL text encoder (bf16) and VAE as companions, on Vulkan against diffusers on
  the same files (dequantized GGUF transformer on CUDA, f32 text encoder; diffusers keeps the GGUF's BF16 norm scales undecoded, so the
  reference script dequantizes them first), 20 steps without guidance, same latents: text encoder 7.3–9.6e-4, transformer 1.4e-2 (random
  inputs); images 42.2 dB (512) / 37.4 dB (1024). Speed (Vulkan, warm): 1024×1024 20 steps 35.8 s (1.8 s per step), 512×512 10.6 s;
  diffusers (GGUF dequantized on the fly, CUDA f32) 2.8 s per step at 1024. Qwen-Image 2.1 Turbo (unsloth's Q8_0, its 8-step schedule from
  the file name): 34.5 dB (1024) against diffusers with the pipeline's `sample_sigmas`, 8 steps in 14.8 s.

### FLUX.2

<a id="flux-2"></a>
FLUX.2 Klein (2026-11-17): FLUX.1's MMDiT with shared modulations, SwiGLU, 4-axis RoPE and a Qwen3 text encoder, on 128-channel latents.

- Canonical source `flux2`: the FLUX transformer vocabulary (original names map 1:1; diffusers' q / k / v joined) plus `transformer.mod_img`,
  `mod_txt`, `mod_single` (the shared `double_stream_modulation_img|txt` and `single_stream_modulation` projections); the Qwen3 text model kept
  whole; the 32-channel KL VAE with `vae.bn.running_mean|running_var`. Family keys: the Qwen3 chat template with thinking disabled
  (`<think>\n\n</think>\n\n` after the assistant turn), 512 prompt tokens, the hidden states entering layers 9, 18 and 27 concatenated per
  token, FLUX.2's empirical flow shift (`DiffusionConfigKeys.EmpiricalShift`), distilled (`is_distilled` / file names without `base`): 4 steps
  without guidance.
- `Flux2Adapter`: `FluxTransformerGraphBuilder` with `SharedModulation`, `SwiGlu` and `TextPositionAxis` 3 (text j at (0, 0, 0, j), image
  tokens at (0, row, column, 0), 4 × 32 dimensions, θ 2000; `AxialRope` takes four axes); the text graph reads every one of the 512 rows (the
  pipeline passes no mask to the transformer) and gets transformers' padding mask as row spans from `DiffusionGraphs.TokenMask`, so the padding
  rows attend to the prompt only; the VAE graphs unpatchify / patchify the latents 2×2 and apply the batch-norm statistics
  (`AutoencoderHyperParameters.LatentPatch`).
- Root-cause fix: Vulkan row reductions of a rank-1 tensor launched one row per element and wrote past the one-element output (the
  mask sum above); a rank-1 input is now one row (regression test `RowOpTests.RowReduceOfAVector`).

Verification (tools/make-tiny-flux2.py, tools/make-flux2-references.py; `TinyFlux2Tests`, `Flux2KleinTests`):

- Tiny random FLUX.2 Klein (64×96): prompt tokens equal transformers'; text encoder (all 512 rows), transformer, VAE decode and encode within
  1e-4 / 1e-3 (Vulkan) of diffusers from the diffusers folder, the original-name single file and a Q8_0 GGUF (5e-3 / 3e-2); 4-step txt2img
  above 40 dB in all 6 combinations.
- FLUX.2 Klein 4B, unsloth's Q8_0 GGUF with the pipeline's Qwen3-4B text encoder (bf16) and VAE as companions, on Vulkan against diffusers on
  the same files (dequantized GGUF transformer on CUDA, f32 text encoder), 4 steps, same latents: text encoder 4.5e-4 / 1.3e-3, transformer
  1.6e-3; images 40.1 dB (512) / 41.3 dB (1024). Speed (Vulkan, warm): 1024×1024 4 steps in 3.4 s, 512×512 in 0.8 s.

FLUX.2 Dev (2026-11-17): the same transformer (8 dual- and 48 single-stream blocks, 6144 wide) with the guidance embedding of a
guidance-distilled model, conditioned on Mistral Small 3.2 24B. FLUX.2 Dev's `text_encoder` folder holds exactly Mistral-Small-3.2-24B-Instruct-2506
(the tensors hash the same; Mistral Small 3.1 differs), so any GGUF of that model serves as the companion.

- Recognized by the guidance embedding (`transformer.guidance_in`; Klein has none): the Mistral template of the pipeline
  (`<s>[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]{prompt}[/INST]`, its system message, right padding with `<pad>` to 512 tokens), the hidden states
  entering layers 10, 20 and 30, 50 steps with the embedded guidance 4 and no classifier-free guidance; companions: a `llama` / `mistral3`
  GGUF 5120 wide or a `mistral3` folder.
- Mistral's hidden states carry outlier channels: with int8-quantized activations in the Q8_0 matrix products the text encoder was 5.7e-2 away
  (the padding rows, which the transformer reads, 10× worse than the prompt's), and the 1024 image 21 dB. The diffusion text graph now keeps
  f32 activations where the family asks for it (`GraphBuilder.F32MatMulActivations`, `flux2.text_encoder.f32_activations`): 4.8e-5.
- Verification (tools/make-flux2-dev-references.py; `Flux2DevTests`): unsloth's FLUX.2-dev Q4_K_M GGUF (20 GB) with unsloth's
  Mistral-Small-3.2-24B-Instruct-2506 Q8_0 GGUF and the pipeline's VAE as companions, on Vulkan against diffusers on the same files (the
  dequantized GGUF transformer on CUDA, the text encoder dequantized to f32 by transformers on the CPU), 20 steps, guidance 4, same latents:
  prompt tokens equal the pipeline's processor; text encoder 3.1–4.8e-5 (all 512 rows), transformer 1.1e-3; images 40.0 dB (512) / 49.7 dB
  (1024). Speed (Vulkan, one 32 GB GPU; the 20 GB transformer and the 25 GB text encoder alternate): 1024×1024 20 steps in 73 s (3.6 s per step),
  512×512 in 21 s; diffusers (GGUF dequantized on the fly, CUDA f32) 14 s per step at 1024.

Editing (2026-11-17): FLUX.2 (Klein and Dev) and FLUX.1 Kontext take reference images (`ImageGenerationRequest.ReferenceImages`).

- Core contract: the denoiser component's `DiffusionConfigKeys.ReferenceImages` (how many), the graph dimension `DiffusionGraphs.References`
  and the input `DiffusionGraphs.ReferenceLatent` [R, C, h, w]; `LatentDiffusionGenerator` resizes each reference to the output size (the output
  size defaults to the first reference's, rounded to the token size), encodes it with the VAE and passes the latents to every denoiser
  evaluation (both rows of a guided step). `ImageGenerationDefaults.ReferenceImages` tells front ends that a model edits.
- `FluxTransformerGraphBuilder`: the reference tokens follow the image's in the image stream (same `img_in`, no noise, the step's modulation),
  at the first position axis `reference_index_base` + `reference_index_step` · i (FLUX.2: 10 + 10 i as `Flux2Pipeline._prepare_image_ids`;
  Kontext: 1); only the image's tokens are projected out.
- The pipelines resize references by their own rules (FLUX.2: to at most 1024² and multiples of 16; Kontext: to the preferred resolution of the
  aspect ratio); Sharplight resizes them to the output size, which is the same when the reference has the output's size.
- Verification: the tiny FLUX.2 Klein pipeline with one reference against `Flux2KleinPipeline(image=[…])` from the same noise above 40 dB on CPU
  and Vulkan (`TinyFlux2Tests.EditMatchesDiffusers`). FLUX.1 Kontext dev (QuantStack's Q8_0 GGUF, recognized by its name, with the FLUX.1
  companions CLIP-L, T5-XXL Q8_0 and the VAE; tools/make-flux-kontext-references.py, `FluxKontextTests`): the transformer with a reference
  3.0e-4 relative RMS from diffusers; an edit at 1024 ("turn the horse into a zebra", 20 steps, embedded guidance 2.5, same latents) 23.9 dB
  from `FluxKontextPipeline` on the same files, the same picture (the zebra's stripe edges drift apart; 14 dB from the input), 45 s warm.
- Studio: the Image page shows an "Image to edit" input when the loaded model edits (`ImageGenerationDefaults.ReferenceImages`), hides image to
  image and inpainting while a picture is set, and keeps the picture's size (`ImageEditModeTests` with the tiny FLUX.2 pipeline).
- GGUF text encoders (2026-11-17): the encoders of Z-Image and FLUX.2 Klein 4B are exactly Qwen/Qwen3-4B, Krea 2's exactly
  Qwen/Qwen3-VL-4B-Instruct (tensor hashes), so unsloth's Q8_0 GGUFs serve as companions: against the pipelines' encoders 5.7e-3 (Z-Image),
  1.2e-2 / 2.0e-2 (Klein, all 512 rows), 1.3e-2 (Krea 2), Qwen-Image with Qwen2.5-VL-7B Q8_0 1.3e-2 (`GgufTextEncoderMatches*`). Keeping f32
  activations changed none of these materially (it does for Mistral Small 3.2 under FLUX.2 Dev, above), so only FLUX.2 Dev keeps them;
  FLUX's T5 graph always did.

### Krea 2

<a id="krea-2"></a>
Krea 2 (2026-11-17, the most downloaded text-to-image release on the Hub at the time; Krea 2 Community License, open weights): a 13B
single-stream DiT over [text, image] conditioned on twelve tapped layers of Qwen3-VL-4B, with the Qwen-Image (Wan 2.1) VAE.

- Canonical source `krea2`: `transformer.{img_in, time_in.in_layer|out_layer, time_mod, txt_in.norm|in_layer|out_layer, text_fusion.{layerwise|refiner}.{i}.*,
  text_fusion.projector, blocks.{i}.{mod [6, D], norm1, norm2, attn.q|k|v|gate|o|norm_q|norm_k, mlp.gate|up|down}, final.mod [2, D], final.norm, final.linear}`
  from the original Krea names (`first`, `tmlp`, `tproj`, `txtmlp`, `txtfusion`, `blocks.{i}.attn.wq|...|qknorm`, `prenorm` / `postnorm`, `mod.lin`, `last.*`)
  or diffusers' `Krea2Transformer2DModel` names; every norm scale is zero-centred and stored + 1. The Qwen3-VL text model kept whole (`Qwen3VLModel`'s bare
  `language_model.` names are now mapped too). Family keys: the system-prompt template with its 34-token system turn skipped, 512 prompt tokens, the
  pipeline's `text_encoder_select_layers` (2, 5, …, 35; transformer-only files: every (layers / 12)-th layer ending one before the last), the
  exponential shift (μ from the image tokens between 0.5 at 256 and 1.15 at 6400; the distilled Turbo's fixed μ = 1.15 as a dynamic shift with equal
  bounds), Turbo: 8 steps without guidance, Raw: 28 steps with guidance 5.5 (diffusers' Krea convention cond + 4.5 · (cond − uncond) is the usual
  scale 1 + 4.5). vantagewithai's GGUF files carry two extra 6144 × 6144 tensors (`last.up` / `last.down`, an embedded picture described by `egg_*`
  metadata) that no implementation reads; they are skipped.
- `Krea2TransformerGraphBuilder`: the text fusion first (two blocks across the 12 tapped layers of each token as one batched `Attention` with
  `sequence_rows` = `sequence_keys` = 12, the projector over the layer axis as a linear map, two refiner blocks across the tokens with the padding as a
  key bias), then the zero-centred RMS norm and the tanh-GELU MLP into the model width; 28 blocks over [text, image] with RMS norms modulated by the shared
  time modulation plus each block's table, grouped-query attention (48 query / 12 key-value heads) with q / k RMS norms, interleaved 3-axis RoPE (32 + 48 +
  48 dims, θ 1000; text at the origin, image tokens at (0, row, column)) and a sigmoid output gate, SwiGLU; the final norm modulated by the time embedding
  plus a table. Image tokens are 2×2 patches of the 16-channel latents. The text graph is the decoder graph's hidden states entering the 12 layers,
  concatenated per token.
- Prompt layout: the pipeline pads the system turn and the prompt and appends the assistant turn after the padding (its positions continue after the
  prompt); Sharplight tokenizes the assistant turn right after the prompt and pads after it. The valid rows get the same hidden states, and the
  transformer gives text rows no positions and masks the padding, so the result is the same; only prompts longer than 507 tokens differ (the
  pipeline keeps the assistant turn, Sharplight cuts at 546 tokens).

Verification (tools/make-tiny-krea2.py, tools/make-krea2-references.py; `TinyKrea2Tests`, `Krea2Tests`, `Krea2ImageTests` in Studio.Tests):

- Tiny random Krea 2 (64×96, GQA in both the blocks and the text fusion, 6-layer Qwen3-VL tapped at 1, 3, 5): prompt tokens equal the pipeline's valid
  tokens; text encoder, transformer (masked padding), VAE decode and encode within 1e-4 / 1e-3 (Vulkan) of diffusers from the diffusers folder, the
  original-name single file and a Q8_0 GGUF (GGUF within 5e-3 / 3e-2); guided 4-step txt2img above 40 dB in all 6 combinations.
- Krea 2 Turbo, realrebelai's Q8_0 GGUF (13.6 GB) with the Turbo pipeline's Qwen3-VL-4B text encoder (bf16) and VAE as companions, on Vulkan against
  diffusers on the same files (dequantized GGUF transformer on CUDA, f32 text encoder), 8 distilled steps, same latents: text encoder 3.3–3.8e-4,
  transformer 5.7e-3 relative RMS (random inputs at 512); images 26.9 dB (512) / 34.1 dB (1024), visually identical. Speed (Vulkan, warm, conditioning
  reused): 1024×1024 8 steps in 29.9 s (about 3.6 s per step), 512×512 in 8.6 s; diffusers (GGUF dequantized on the fly, CUDA f32) 5.7 s per step at
  1024. Studio: the load panel shows the Qwen3-VL-4B and VAE slots, fills them from the pipeline folder beside the GGUF's folder, and the Image page
  generates 1024×1024.

### SD editing (InstructPix2Pix and the inpainting UNets)

<a id="sd-editing"></a>
InstructPix2Pix, the SD 1.5 inpainting UNet and the SDXL inpainting UNet (2026-11-18) are the SD UNets with more input channels than they
predict: 8 (noisy latents + the input image's latents), 9 (noisy latents, the latent-resolution mask, the masked image's latents).

- Core: `UNetHyperParameters.ConditionChannels` (input minus output channels); the UNet graph takes the extra channels as the optional
  `condition_latent` input (`DiffusionGraphs.ConditionLatent`) and concatenates them to `latent` inside the graph, so the `.slm` keeps the
  contract; the denoiser component says what they carry (`DiffusionConfigKeys.ConditionMode`: `image` for 4 extra channels, `inpaint` for 5)
  and the model carries the default image guidance scale 1.5 (`imageGuidanceScale`) and the VAE's scaling factor.
- InstructPix2Pix: the request's `Image` is the picture and the prompt the instruction (no mask); the run starts from noise (as diffusers: the
  strength does not apply), the image's latents enter unscaled (the VAE mode, not its scaled mean), and a guided step evaluates three rows
  (instruction and image, empty prompt and image, empty prompt and zeros) combined as `uncond + gs · (text − image) + igs · (image − uncond)`.
  `ImageGenerationOptions.ImageGuidanceScale` (default 1.5; below 1 disables guidance) and `ImageGenerationDefaults.ImageGuidanceScale` expose it.
  A three-row step is one denoiser run (the denoiser's batch bound is 3 for image-conditioned UNets). Vulkan once gave wrong results for
  batches of three and more (found while verifying): the convs' split-K workspace was sized over the batches that split only (one and two
  rows for the SD UNets), so a larger batch running unsplit through it wrote past it and overwrote the time embedding; it now holds every
  batch's partials (`ConvMatrixTests.SplitKAcrossBatches`, `TinyUNetBatchTests`, the three-row case of `SdEditTests` on Vulkan).
- Inpainting UNets: `Image` and `Mask` as for the 4-channel models, but the UNet sees the mask and the masked image (grey where the mask is
  set, as diffusers' `image * (mask < 0.5)` in [-1, 1]), so there is no re-imposing of the known part after each step (diffusers does that
  only for 4-channel UNets) and a strength of 1 starts from pure noise.

Verification (tools/make-sd-edit-references.py: float32 diffusers on CUDA, Euler for InstructPix2Pix (its ancestral scheduler draws new noise
every step) and DDIM for the SD 1.5 inpainting model, shared initial noise, 512×512, 20 steps, the VAE posterior taken at its mode;
`SdEditTests`): the UNet with the extra channels 2.2e-4 relative RMS on the CPU and on Vulkan, for InstructPix2Pix and both inpainting
models; the VAE encoder against the mode of the posterior (InstructPix2Pix's image latents) within 5e-3; whole edits on Vulkan: InstructPix2Pix
**59.7 dB**, SD 1.5 inpainting **56.3 dB** and SDXL inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1, Euler, 512×512) **40.5 dB**. The `.slm` round trip (256×256,
Vulkan) generates the same bytes as the folder for both. Studio's Image page shows the edit input and an "Image guidance" parameter for
InstructPix2Pix-style models (`ImageEditModeTests`).

### SDXL refiner

<a id="sdxl-refiner"></a>
stabilityai/stable-diffusion-xl-refiner-1.0 (2026-11-18) is an SDXL UNet (384 / 768 / 1536 channels, four transformer blocks per level) conditioned
on OpenCLIP bigG alone and on an aesthetic score instead of the target size.

- Import: a diffusers folder without `text_encoder/` but with `text_encoder_2/` and `tokenizer_2/` puts the bigG encoder in the first encoder's
  place (`text_encoder.*`, with its `projection.weight`), as SDXL's second encoder: the penultimate layer without the final norm, the pooled
  embedding from the projection. The text-encode graph then outputs the pooled embedding itself.
- Size conditions: the UNet's added conditioning takes `class_embedding_input` = pooled width + conditions × `addition_time_dim`, so the adapter
  derives the number of conditions from the pooled projection's width (SDXL 2816 = 1280 + 6 · 256, the refiner 2560 = 1280 + 5 · 256;
  `UNetHyperParameters.SizeConditions`, nothing new stored in the `.slm`). With five the generator passes `[height, width, 0, 0, score]` per row:
  the negative prompt's row 2.5, the prompt's 6 (diffusers' `negative_aesthetic_score` / `aesthetic_score`; `ImageGenerationOptions.AestheticScore`,
  `NegativeAestheticScore`, `ImageGenerationDefaults.AestheticScore`).
- Ensemble: `ImageGenerationOptions.Refiner` (another latent-diffusion generator, e.g. the refiner's) and `RefinerStart` (default 0.8). The
  first generator runs its schedule down to the switch timestep `round(T − start · T)` (diffusers' `denoising_end`: the timesteps at or above it),
  hands its latents to the refiner, which runs the timesteps of its own schedule below it (`denoising_start`) on its own text conditioning and
  decodes with its VAE. The steps of both stages are reported as one run. The refiner alone does image to image as any SDXL model (diffusers'
  default strength 0.3).
- Studio: the Image page offers every other resident image-generation model as the refiner of the chosen one, with the "Refiner switch"
  parameter (`ImageRefinerTests`).

Verification (tools/make-sdxl-refiner-references.py: diffusers' StableDiffusionXLPipeline with `denoising_end` and StableDiffusionXLImg2ImgPipeline
with `denoising_start`, the fp16 files upcast to f32 on CUDA, Euler, 1024×1024, 20 steps cut at 0.8, guidance 5, the same initial noise;
`SdxlRefinerTests`): bigG hidden states 5.7e-4 and pooled embedding 1.9e-4 relative RMS, the refiner UNet (two rows, both scores) 4.4e-5 on
Vulkan, the whole base + refiner image on Vulkan **48.3 dB**. Not checked: the original single file (`sd_xl_refiner_1.0.safetensors`).

### Qwen-Image-Edit

<a id="qwen-image-edit"></a>
Qwen-Image-Edit (2026-11-18; `QwenImageEditPlusPipeline` for -2509, -2511 and FireRed-Image-Edit, `QwenImageEditPipeline` for the first
Qwen-Image-Edit) runs the Qwen-Image transformer (section above) with the reference images in two places:

- The language model reads them. The Qwen2.5-VL text encoder keeps its vision tower (`vision_encode` graph in the text-encoder component; the
  weights were always kept with the encoder, they were just unused) and its prompt is the edit template (64 skipped tokens) followed by
  `Picture n: <|vision_start|><|image_pad|><|vision_end|>` per image and the instruction; each image is resized to about 384 · 384 pixels at
  its aspect ratio (sides multiples of 32, Lanczos as diffusers' `VaeImageProcessor`; the first Edit: 1024 · 1024), preprocessed by the vision
  preprocessor (smart resize to multiples of 28, windowed attention) and its merged-token embeddings replace the placeholders' rows;
  the M-RoPE positions come from `Qwen35Multimodal` (text positions advance by one, an image by max(rows, columns) of its token grid).
  `DecoderGraphBuilder.BuildLayerHiddenStates(imageRows: true)` takes `positions`, `embeddings` and `embedding_mask`; the generator pads the
  prompt to 64 + 1024 rows (the first Edit 1536) and masks the padding as for text-to-image. The negative prompt (true CFG) sees the images too.
- The transformer sees their latents. The VAE-encoded references (normalized, resized to the output's size) follow the image's tokens with
  frame index 1, 2, 3 on the first RoPE axis (the existing reference mechanism of FLUX.1 Kontext and FLUX.2); -2511's `zero_cond_t`
  modulates the reference tokens' image stream at timestep 0 (`FluxHyperParameters.ZeroConditionTimestep`: a second evaluation of the timestep
  embedding at 0 for the image modulations, not the final one, and the reference rows split off for each modulation and gate). References
  of other sizes than the output are resized to it; diffusers encodes each at about one megapixel with its own aspect ratio, so the two agree
  when the references share the output's aspect ratio.
- Output size: when the request gives none, about one megapixel at the last reference image's aspect ratio (`DiffusionConfigKeys.ReferenceArea`,
  diffusers' `calculate_dimensions`). True CFG with the norm rescaling, 50 steps, guidance 4, the exponential dynamic shift (max length 8192).
- Importers: the diffusers folder's pipeline class and `zero_cond_t` pick the variant; a transformer-only file takes it from the marker tensor
  of the -2511 GGUF files (`__index_timestep_zero__`, skipped as a weight) or its file name. Metadata `qwen_image.edit.*` and
  `qwen_image.transformer.{max_references, reference_index_base, reference_index_step, zero_cond_t}` carry it into the `.slm`.

Verification (tools/make-tiny-qwen-image-edit.py, tools/make-qwen-image-edit-references.py; `TinyQwenImageEditTests`, `QwenImageEditTests`):

- Tiny random pipelines (output 64×96, references at the output size, condition pictures at the real 384 · 384 area; "edit-2509" and
  "edit-2511"): the transformer with one and two references against diffusers 1e-4 (CPU) / 1e-3 (Vulkan) relative RMS, the Q8_0 GGUF
  (with the -2511 marker tensor) within 3e-2; whole edits with one and two images **58 to 59 dB** (diffusers folder) and 56 to 59 dB (Q8_0 GGUF),
  on the CPU and on Vulkan.
- Qwen-Image-Edit-2511, unsloth's Q4_K_M GGUF with the pipeline's text encoder (bf16) and VAE as companions, on Vulkan against diffusers running
  the same files (dequantized GGUF transformer on CUDA, f32 text encoder), 512×384, 20 steps, true CFG 4, negative prompt " ", same latents,
  640×480 input pictures: one picture **40.3 dB**, two pictures **26.4 dB**. The language model's rows are sensitive to the pixels it reads: the
  vision tower alone matches transformers to 2.5e-5 relative RMS on the CPU (4.4e-3 on Vulkan) on identical pixels, but a few gray levels of
  resampling difference (PIL's Lanczos for the pipeline's resize, torch's bicubic for the processor's) move its embeddings by about 4 % and the
  text encoder's image rows by far more; diffusers itself, with PIL's bicubic instead of Lanczos for the pictures the language model reads, moves
  its own result by 45.6 dB (one picture) and 24.3 dB (two), the same size as the difference above, which is why the thresholds are 35 / 22 dB.
  Speed on Vulkan (Q4_K_M, the 13 GB transformer and 16 GB text encoder alternate on the 32 GB GPU): 74 s for one picture and 97 s for two
  (including loading, conditioning and 40 transformer evaluations at 512×384).
- Not verified numerically: the first Qwen-Image-Edit (`QwenImageEditPipeline`: one picture read at 1024 · 1024 pixels, no "Picture n" label)
  and FireRed-Image-Edit (the same pipeline as -2509 per its model index); both go through the code above (the first Edit with 1536 prompt rows),
  and -2509 through the tiny pipeline and the same transformer.

### LoRA

<a id="lora"></a>
LoRAs are merged at load (2026-11-16): a LoRA file is an auxiliary source (role `lora`, `lora:2`, ...; `lora@0.8` for strength 0.8) whose
`FormatData` is an `IWeightPatch`; `WeightPatches.Merge` (Core, before any adapter) replaces the targeted canonical weights with
W + s · α / r · up · down, computed when the weight is read (quantized weights are decoded and re-encoded as Q8_0 where the rows are multiples of 32, f16 otherwise). `DiffusionLoraPatch`
(Import.Diffusers) maps modules through the importer's own name mappings: kohya names are re-dotted by known multi-word names
(`KohyaNames`), SDXL's SGM UNet names mapped as diffusers' `_maybe_map_sgm_blocks_to_diffusers`; diffusers / PEFT names go through
`DiffusersNames` (UNet, CLIP), `FluxNames` (q / k / v and the single blocks' MLP input as row blocks of the fused projections) or
`QwenImageNames`. A module that matches no weight fails naming the module and the model family.

Verification (tools/make-tiny-lora.py, `TinyLoraTests`): random rank-4 LoRAs (α 8, strength 0.8) on the tiny SD 1.5, SDXL and FLUX.1-dev
pipelines against diffusers' `load_lora_weights` from the same noise: SD 1.5 kohya and PEFT (UNet attention / MLP, text encoder) 82 dB
(CPU) / 76 dB (Vulkan); SDXL kohya (SGM names, both text encoders) and PEFT 44 / 41 dB; FLUX kohya (original names, fused q/k/v) and PEFT
81–85 / 75 dB; the unadapted image is 11–14 dB away.

Krea 2 LoRAs (2026-11-17): diffusers / PEFT names (`transformer.transformer_blocks.{i}.attn.to_q`, `text_fusion.*`, as ComfyUI's Krea 2 style
LoRAs) and the original Krea names map through `Krea2Names`. A LoRA on every linear of a quantized transformer used to turn the merged weights
into f16 (Krea 2 Q8_0: 26 GB, more than the GPU); merged quantized weights are now re-encoded as Q8_0 (`Q8Blocks`) where the rows are
multiples of 32. ComfyUI's `krea2_darkbrush.safetensors` (rank 32, every linear incl. the text fusion) on realrebelai's Q8_0 GGUF at 1024, 8 steps,
against diffusers' `load_lora_weights` on the same files: 23.3 dB, the same picture (the brush texture keeps the PSNR low); 7.8 dB from the
image without the LoRA (`Krea2Tests.LoraMatchesDiffusers`).

## 18. Video generation (2026-11-16)

Video generators are models of kind `video-generation` (see [api-overview.md](api-overview.md) "Video generation"): the importer
`Sharplight.Import.Diffusers` translates Wan pipeline folders into the canonical `wan` source, `WanAdapter` builds the umT5 text
encoder, the video DiT and the causal 3-D VAE as graphs of generic Core ops (no new op: a causal 3-D convolution is one 2-D
convolution over the frame windows stacked along the channels, the importer folding the kernel's time taps into its input
channels), and Core's `VideoDiffusionGenerator` runs them with the UniPC or Euler flow sampler and classifier-free guidance on the
host, decoding latent frame by latent frame with the causal caches carried between chunks.

| family | formats | status | verified |
|---|---|---|---|
| Wan 2.1 T2V (1.3B, 14B) | diffusers folder; transformer-only GGUF (city96 / QuantStack, any ggml type) and original-name `.safetensors` (also ComfyUI's `model.diffusion_model.` prefix) with a Wan diffusers folder as companion for the umT5 encoder, tokenizer and VAE | **supported** (text-to-video, negative prompts, any size that is a multiple of 16, frames 1 + 4k, UniPC (default) and Euler) | tiny pipeline; 1.3B below |
| Wan 2.2 TI2V-5B | diffusers folder; transformer-only GGUF / safetensors with companions as for Wan 2.1 | **supported** (text- and image-to-video: per-token timesteps, the residual VAE with 2×2 pixel patching, first-frame conditioning; multiples of 32) | tiny pipeline; 5B below |
| Wan 2.2 T2V-A14B, I2V-A14B (two experts) | diffusers folder with transformer_2; high- and low-noise transformer GGUFs (QuantStack / city96) with `AuxiliaryRoles.SecondTransformer` and a Wan 2.1 folder as companion | **supported** (expert switch at the boundary timestep (T2V 0.875, I2V 0.9) · 1000, `GuidanceScale2`; I2V by concatenating a first-frame mask and the encoded start-image video, streamed causal VAE encoder; optional last frame) | tiny pipelines; T2V-A14B and I2V-A14B Q4_K_M below |
| Wan 2.1 I2V-14B, FLF2V-14B (CLIP image embedding) | diffusers folder (with `image_encoder/` and `image_processor/`); transformer-only GGUF (city96) with a Wan 2.1 folder for umT5 / VAE and the I2V / FLF2V configuration folder as `AuxiliaryRoles.ImageEncoder` | **supported** (CLIP ViT-H/14 penultimate states, the image embedding and the added image key/value cross-attention; FLF2V: `VideoGenerationRequest.LastImage`, the two images' states with the learned position table, the mask 1 on the last frame) | tiny pipelines; I2V and FLF2V Q4_K_M below |
| Stand-alone umT5 / Wan VAE files (ComfyUI layout) as companions | - | missing | |
| LTX-Video 0.9.0 (2B) | diffusers folder (the T5-XXL text encoder is the one of FLUX.1 / PixArt) | **supported** (text-to-video, multiples of 32, frames 1 + 8k, flow Euler with the resolution-dependent shift and the terminal stretch) | tiny pipeline; 2B below |
| HunyuanVideo (13B, text-to-video) | diffusers folder (hunyuanvideo-community/HunyuanVideo); transformer-only GGUF (city96, any ggml type) / original-name safetensors with a HunyuanVideo folder as companion (`AuxiliaryRoles.TextEncoder` for the Llama and CLIP encoders, `AuxiliaryRoles.Vae`) | **supported** (LLaVA-Llama-3 states two layers before the last over the templated prompt, the template's 95 tokens cropped; CLIP-L pooled; guidance embedded (no classifier-free pass); flow Euler, shift 7; the causal 3-D VAE decoded in diffusers' temporal and spatial tiles; frames 1 + 4k, multiples of 16) | tiny pipeline; Q4_K_M below |
| LTX-Video 0.9.5 and later (2B) | diffusers folder | **supported** (the timestep-conditioned decoder at decode timestep 0 (diffusers' default), residual upsamplers that halve the channels, static shift 1 without the terminal stretch) | tiny 0.9.5 pipeline; 0.9.5 below |
| LTX-Video 13B (0.9.7 / 0.9.8, dev and distilled) as single files | transformer GGUF (QuantStack) or original-name safetensors, the stand-alone VAE file (`AuxiliaryRoles.Vae`), T5 from an LTX-Video diffusers folder (`AuxiliaryRoles.TextEncoder`) | **supported** (original names mapped as diffusers' single-file conversion; Lightricks' causal RoPE frame coordinates and masked T5, as diffusers' condition pipeline; distilled files (by name): the 8-step distilled schedule without guidance) | 0.9.8 distilled Q4_K_M below |
| HunyuanVideo image-to-video, HunyuanVideo 1.5, LTX-Video 0.9.1 (noise-injecting decoder), LTX-Video image / video conditioning and the latent upsamplers, CogVideoX, Mochi | - | missing (LTX 0.9.1's decoder is rejected by name) | |

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
- Transformer-only files: `WanCheckpoint` configures the transformer from its tensor shapes (128-wide heads; 48 latent channels =
  Wan 2.2 TI2V with per-token timesteps and first-frame conditioning); `WanCompanionAdapter` takes the text encoder, tokenizer and
  VAE from complete `wan` sources given as auxiliary sources (`AuxiliaryRoles.TextEncoder`, `AuxiliaryRoles.Vae`; one diffusers folder
  serves both), and a file without them fails naming what is missing.
- LTX-Video: every latent pixel is a token; its RoPE rotates pairs of the whole width by angles linear in fractional (frame, row,
  column) coordinates (the frame axis scaled by the VAE's temporal compression over the frame rate, 25 fps), computed in the graph as
  a constant coordinate × frequency product; the text encoder runs without a mask and the cross-attention masks the padding
  (`VideoDiffusionGraphs.ContextMask`); the non-causal VAE decoder runs over the whole latent video (`vae_decode`: replicated first
  and last frames in time, depth-to-space 2×2×2 upsamplers, 4×4 unpatching).
- Two experts (Wan 2.2 A14B): a second transformer component (`transformer-2`, graph `denoise_2`) runs for the timesteps below
  `boundaryRatio` · train steps with `guidanceScale2`; the first expert is released at the switch when the model holds one stage at a time.
- Image-to-video by channel concatenation (Wan 2.2 I2V-A14B): the transformer takes more channels than the VAE gives; the pipeline
  encodes the start image followed by blank frames with the streamed causal encoder (`vae_encode`, then `vae_encode_next` per
  temporal-compression frames, the temporal downsampling over the previous chunk's last frame) and concatenates a mask that is 1 on the
  first latent frame and the encoded latents to the noisy latents of every transformer input.
- HunyuanVideo: the language model (LLaVA-Llama-3-8B) is an embedded decoder source (`hunyuan_video.text_encoder_source.*`, weights under
  `text_encoder.`) whose `text_encode` graph returns the hidden states two layers before the last for the template plus 256 prompt tokens;
  the pipeline drops the template's states (`VideoConfigKeys.PromptPrefixTokens`) and passes the key mask to the denoiser. CLIP-L
  (`text_encode_2`, component `text-encoder-2`, tokenizer `tokenizer-2`) gives the pooled vector; the guidance scale is a denoiser input
  (`VideoConfigKeys.EmbeddedGuidance`). The transformer keeps the original names (diffusers' split q/k/v are concatenated at import, so GGUF
  files map 1:1). The VAE decoder groups frames inside its causal convolutions so only one group's stacked windows exist at a time, runs
  its frame-causal mid attention frame by frame, and the pipeline decodes in diffusers' temporal tiles (5 latent frames, 3 apart, blended
  over 4 frames) and spatial tiles (32 latent pixels, 24 apart, blended over 64 pixels; `VideoConfigKeys.DecodeTileFrames`,
  `DecodeSpatialTile`), so decoding memory does not grow with the video size.
- Tokenizer fix: Hugging Face templates that start with the sequence (T5 / umT5 `$A </s>`) no longer add a BOS token when the
  tokenizer configuration names one.

Verification (tools/make-tiny-wan.py, tools/make-wan-references.py, tools/make-tiny-hunyuan-video.py; tests `TinyWanTests`, `WanTests`,
`WanImageToVideoTests`, `TinyHunyuanVideoTests`, `HunyuanVideoTests` in
`tests/Sharplight.ImageGeneration.Tests`):

- Tiny random Wan pipelines (`~/models/video/tiny-random-wan/{t2v,ti2v}`: Wan 2.1 layout; Wan 2.2 TI2V layout with the residual
  VAE, 2×2 patching and per-token timesteps): umT5 conditioning, transformer, the streamed VAE decode of 9 frames and the image
  encoder within 1e-4 relative RMS of diffusers / transformers on the CPU and on Vulkan; whole runs from the same initial latents
  (4 steps, guidance 5, negative prompt) above 40 dB PSNR in every frame: text-to-video with UniPC and with Euler for both
  layouts, image-to-video for TI2V.
- Tiny random LTX-Video pipeline (`~/models/video/tiny-random-ltx`, tools/make-tiny-ltx.py): T5 states, transformer (with a key mask)
  and the whole-video VAE decode of 17 frames within 1e-4 relative RMS, 4-step text-to-video above 40 dB in every frame, CPU and
  Vulkan.
- Tiny two-expert pipelines (`tiny-random-wan/a14b`: text-to-video; `tiny-random-wan/i2v`: concatenated image-to-video): both experts
  within 1e-4, whole runs above 40 dB in every frame, CPU and Vulkan.
- Real models on Vulkan (one RTX PRO 4500, 32 GB) against diffusers in f32 on CUDA, 832×480, 17 frames, 20 UniPC steps,
  guidance 5, the same initial latents: umT5-XXL (bf16 files) 3.0e-6 relative RMS; the transformer 2.4e-4 (Wan 2.1 1.3B, f32
  files), 2.8e-4 (Wan 2.2 TI2V-5B, f32 files) and 2.0e-3 (QuantStack's Wan2.2-TI2V-5B-Q8_0.gguf with the 2.2 folder as companion)
  relative RMS. Whole runs, worst frame PSNR: Wan 2.1 text-to-video 50.8 dB, Wan 2.2 TI2V image-to-video 55.1 dB, the Q8_0 GGUF
  51.1 dB (first runs including the kernel compilation of every graph: 202 s, 305 s, 460 s). The Wan 2.1 folder's f32 umT5
  (22.7 GB) is the same model as Wan 2.2's bf16 one (1.7e-3 relative weight difference = bf16 rounding); the references use the
  bf16 files.
- LTX-Video 0.9.0 (2B, f32 transformer files; T5-XXL bf16) on Vulkan against diffusers f32 on CUDA, 704×480, 25 frames, 20 Euler
  steps, guidance 3, same initial latents: T5 states 1.4e-5, transformer 1.8e-5 relative RMS, text-to-video worst frame 57.5 dB
  (134 s for the first run including kernel compilation; diffusers 1.0 s per step).
- Wan 2.2 T2V-A14B (QuantStack's HighNoise and LowNoise Q4_K_M GGUFs, the Wan 2.1 1.3B folder as companion for umT5 and the VAE;
  tools/make-wan-a14b-references.py: diffusers running the same GGUFs dequantized to f32 on CUDA, 832×480, 17 frames, 20 UniPC steps,
  guidance 4 / 3): high-noise transformer 3.4e-4, low-noise transformer 2.3e-4, umT5 3.0e-6 relative RMS; the whole run 21.3–24.9 dB per
  frame, the same scene with diverging fine detail. The two quantized 14B experts amplify any difference: moving the initial noise by
  ±1e-3 changes Sharplight's own result by 22–26 dB per frame, as much as it differs from diffusers. 14.6 s per step (both guidance
  passes) at 832×480×33 frames, decode 26 s, peak 16.1 GB (one expert resident at a time), 334 s in all; diffusers took 23 s per step at
  17 frames.
- Wan 2.1 I2V-14B-480P and FLF2V-14B-720P (city96 Q4_K_M GGUFs; umT5 and the VAE from the Wan 2.1 1.3B folder, CLIP ViT-H/14 from
  the I2V / FLF2V configuration folders; tools/make-wan-i2v-references.py: diffusers on the same GGUFs, f32 on CUDA without TF32,
  832×480, 17 frames, 20 UniPC steps, guidance 5, a start image (and a zoomed crop of it as the last image)): CLIP states 3.5e-3 / 3.4e-3
  relative RMS against transformers on the CPU (one-level pixel differences of the bicubic resize; transformers' own CUDA run with TF32
  convolutions is 1e-2 away), transformer 3.9e-4 / 1.7e-4; image-to-video 30.6–48.1 dB per frame, first-and-last-frame 50.8–60.2 dB.
- Wan 2.2 I2V-A14B (QuantStack's HighNoise and LowNoise Q4_K_M GGUFs, the Wan 2.1 folder as companion; same reference script, guidance
  3.5 / 3.5, switch at 0.9): high- / low-noise transformer 3.0e-4 / 6.2e-4 relative RMS at each expert's first step, image-to-video
  35.0–52.5 dB per frame.
- HunyuanVideo (city96's hunyuan-video-t2v-720p-Q4_K_M.gguf, the HunyuanVideo folder as companion; tools/make-hunyuan-video-references.py:
  diffusers on the same GGUF in f32 on CUDA without TF32, the Llama encoder and the VAE in f32 on the CPU with `enable_tiling()`, 832×480,
  33 frames, 20 Euler steps, embedded guidance 6): Llama states 2.5e-4, CLIP pooled 3.4e-4 (f16 files), transformer 4.1e-3 relative RMS;
  text-to-video 28.2–33.0 dB per frame, visually the same video (a ±1e-3 nudge of the initial noise gives 30–34 dB against the
  reference). 5.7 s per step (one pass, guidance embedded) at 832×480×33 frames, decoding about 45 s (45 tiles), peak 23.4 GB, 163 s in all;
  diffusers took 16.6 s per step.
- LTX-Video 0.9.5 (Lightricks/LTX-Video-0.9.5, bf16 transformer, the T5-XXL of the 0.9.0 folder) on Vulkan against diffusers f32 on CUDA
  (TF32 off), 704×480, 25 frames, 20 Euler steps, guidance 3: T5 states 1.4e-5, transformer 2.9e-4 relative RMS, text-to-video worst
  frame 45.0 dB (8.2 s including the kernel compilation; diffusers 33.5 s). At 768×512×97 frames with 40 steps: 0.75 s per step (both
  guidance passes), decode 8 s, peak 10.0 GB, 49 s in all.
- LTX-Video 13B 0.9.8 distilled (QuantStack's Q4_K_M GGUF and its VAE file, T5 from the LTX-Video folder; tools/make-ltx-single-file-references.py:
  diffusers' LTXConditionPipeline on the same files in f32 on CUDA, 704×480, 25 frames, timesteps 1000, 993, 987, 981, 975, 909, 725, 0.03,
  no guidance): transformer 2.7e-3 relative RMS, text-to-video 43.1–46.7 dB per frame; the reference's final latents decoded by Sharplight's
  VAE: 74–76 dB. At 768×512×121 frames: 2.4 s per step (one pass), decode 7 s, peak 12.3 GB, 34 s in all. diffusers' `LTXPipeline` runs T5
  without a mask and uses frame coordinates f · 8; the diffusers folders keep that behaviour (0.9.0 / 0.9.5 verification), single files follow
  Lightricks' inference.
- Speed and memory on the 32 GB GPU (Vulkan, `sharplight-generate-video`, 832×480, 33 frames = 9 latent frames, 20 UniPC steps
  with guidance, each run a new process: the conditioning time includes loading and compiling the umT5 encoder from the kernel
  cache): Wan 2.1 1.3B (f32 transformer) 4.2 s per step (both guidance passes), decode 19 s, peak 16.3 GB; Wan 2.2 TI2V-5B (f32
  transformer, one stage resident at a time) 3.3–3.7 s per step, decode 27 s, peak 30.7 GB; Wan 2.2 TI2V-5B Q8_0 GGUF 1.6 s per step,
  decode 23 s, peak 16.4 GB. diffusers in f32 on CUDA took 3.1 s per step for Wan 2.1 at 17 frames. The decoders are not tuned yet
  (the frame windows are stacked into the channels, 3× the activation of a native 3-D convolution).

## 17. Top-20 sweep (2026-11-16)

The most downloaded text-generation models on the Hugging Face Hub (`pipeline_tag=text-generation`, sorted by downloads, cross-checked
with the trending and most-liked lists on 2026-11-16), one per model family and size, safetensors below 30 GB, skipping quantized
re-uploads (AWQ, NVFP4, FP8 copies) and non-LLM repositories. Each model was tried twice: the original safetensors through the Hugging
Face importer and the most downloaded GGUF of the same model through the GGUF importer. On request, popular models above 30 GB were
added as GGUF (and as an FP8 checkpoint where one exists): Qwen3-32B, Qwen3-30B-A3B, GLM-4.7-Flash, Qwen3.6-35B-A3B,
Llama-3.3-70B-Instruct, gpt-oss-120b, Qwen3.8-27B and Qwen3.8-Flash-Next. The meta-llama repositories are gated for the token used, so
their official unsloth mirrors were taken (same files).

Method (`bench/ParitySweep`, `tools/parity/`):

- Tokenizer: encode and decode of seven fixed texts (English, German / French / Spanish / Russian / Greek / Polish, Japanese / Chinese /
  Korean / Arabic / Hindi / Thai / Vietnamese, code with tabs and runs of spaces, emoji and symbols, leading and trailing spaces,
  contractions and digit groups) against transformers 5.23 (`AutoTokenizer`, no remote code).
- Chat template: four conversations (user only; system + multi-turn; no generation prompt; a tool definition, an assistant tool call
  and the tool result) rendered by Sharplight's Jinja engine against transformers' `apply_chat_template`, compared as text and as
  prompt ids. For a GGUF whose embedded template differs from the checkpoint's, transformers renders the GGUF's own template as the
  reference.
- Greedy: 48 tokens for two raw prompts and the rendered `user only` chat prompt. Safetensors against transformers running the
  checkpoint in fp32 on the CPU (raw logits, no logits processors: the checkpoints' `generation_config` would otherwise add a
  repetition penalty), with teacher forcing on the reference's tokens (our logits at its top-5 ids at every step). GGUF against
  llama.cpp b11177 (`llama-completion`) on Vulkan and, when the outputs differ, on the CPU as well.
- Near-ties: a first differing token whose reference logit gap is at most 0.1 counts as a near-tie. For GGUF a prompt also passes
  when one of our outputs (Vulkan, CPU) equals one of llama.cpp's backends (they split on the same near-ties, often crosswise), or
  when ours agree with llama.cpp at least as long as llama.cpp's own backends agree with each other. Remaining differences are checked
  by teacher forcing against transformers running the same GGUF dequantized (`--gguf-file`), with a tie tolerance of 0.5 for the
  int8-activation rounding of quantized matmuls.
- Studio: `TopSweepRealModelTests` loads three of the models through Studio's engine runtime (library recognition, load, greedy chat
  turn). `.slm`: four models converted to packages and every check rerun on the package (identical results).

Models above one 32 GB GPU ran over two GPUs with Sharplight's multi-device placement (`--devices 2`), gpt-oss-120b and
Qwen3.8-Flash-Next with the CPU as fallback device for the layers that did not fit; llama.cpp ran with the same two GPUs.

Results (HF = safetensors vs transformers, GGUF = vs llama.cpp unless stated):

| # | Model (downloads rank) | Architecture | Safetensors | HF result | GGUF tested | GGUF result | Issue found | Fix |
|---|---|---|---|---|---|---|---|---|
| 1 | openai-community/gpt2 | gpt2 | 0.5 GB F32 | pass: tokenizer, ids identical (3 prompts) | mradermacher/gpt2-GGUF Q8_0 | pass: Vulkan identical to llama.cpp Vulkan, CPU to llama.cpp CPU (their backends split at a near-tie) | none | |
| 2 | Qwen/Qwen3-0.6B | qwen3 | 1.5 GB | pass: tokenizer, 4 templates incl. tools, ids identical | Qwen/Qwen3-0.6B-GGUF Q8_0 | pass (cross-device near-tie, as gpt2) | none | |
| 3 | Qwen/Qwen3-14B | qwen3 | 29.5 GB | pass: ids identical (Vulkan, one whole GPU) | Qwen/Qwen3-14B-GGUF Q4_K_M | pass: identical to llama.cpp | none | |
| 4 | Qwen/Qwen2.5-7B-Instruct | qwen2 | 15.2 GB | pass: ids identical; chat identical up to the end of turn | Qwen/Qwen2.5-7B-Instruct-GGUF Q4_K_M (2 shards) | pass (cross-device near-tie); tool-call template: the GGUF's older template, rendered identically to transformers | none | |
| 5 | Qwen/Qwen3-8B | qwen3 | 16.4 GB | pass: all identical | Qwen/Qwen3-8B-GGUF Q4_K_M | pass: identical to llama.cpp | none | |
| 6 | ornith-ai/Ornith-1.5-9B-GGUF (GGUF only; Ornith-1.0-9B for safetensors) | qwen3_5 (hybrid gated delta net) | 18.8 GB | pass after fix: ids identical; tokenizer differed on Devanagari | Ornith-1.5-9B Q4_K_M | pass (cross-device near-tie) | tokenizer_config.json's `pretokenize_regex` (transformers 5 applies it over tokenizer.json) was ignored | 925ec39f |
| 7 | openai/gpt-oss-20b | gpt_oss (MXFP4 experts) | 13.8 GB | pass: harmony template, ids identical | unsloth/gpt-oss-20b-GGUF Q4_K_M | pass (cross-device near-tie) | none | |
| 8 | Qwen/Qwen2.5-1.5B-Instruct | qwen2 | 3.1 GB | pass | Qwen/Qwen2.5-1.5B-Instruct-GGUF Q8_0 | pass (template as #4) | none | |
| 9 | facebook/opt-125m | opt (pytorch .bin, slow tokenizer) | 0.3 GB | pass after fix: tokenizer and ids identical | none published (llama.cpp has no OPT) | - | `opt` unmapped; tokenizers without tokenizer.json (vocab.json + merges.txt) unsupported | 09132005 |
| 10 | meta-llama/Llama-3.1-8B-Instruct (unsloth mirror) | llama | 16.1 GB | pass: ids identical, tools template identical | bartowski Meta-Llama-3.1-8B-Instruct Q4_K_M | pass: first token is a 0.03 near-tie (" a" / " Paris"); identical to transformers running the same GGUF | none | |
| 11 | Qwen/Qwen2.5-0.5B-Instruct | qwen2 | 1.0 GB | pass | Qwen/Qwen2.5-0.5B-Instruct-GGUF Q8_0 | pass: teacher forcing vs transformers on the same GGUF, mean logit difference 0.005–0.02, no flip | none | |
| 12 | EleutherAI/pythia-160m | gpt_neox | 0.4 GB | pass | mradermacher/pythia-160m-GGUF Q8_0 | pass: ours agree with llama.cpp longer than llama.cpp's backends agree with each other | none | |
| 13 | Qwen/Qwen3-4B | qwen3 | 8.0 GB | pass | Qwen/Qwen3-4B-GGUF Q4_K_M | pass: identical | none | |
| 14 | Qwen/Qwen3-1.7B | qwen3 | 4.1 GB | pass | unsloth/Qwen3-1.7B-GGUF Q8_0 | pass (cross-device near-tie) | none | |
| 15 | HuggingFaceTB/SmolLM2-135M | llama | 0.3 GB | pass | prithivMLmods/SmolLM2-135M-GGUF Q8_0 | pass: identical | none | |
| 16 | Qwen/Qwen2.5-3B-Instruct | qwen2 | 6.2 GB | pass | Qwen/Qwen2.5-3B-Instruct-GGUF Q4_K_M | pass: identical (template as #4) | none | |
| 17 | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | nemotron_h (Mamba-2 hybrid) | 7.9 GB | pass | lmstudio-community NVIDIA-Nemotron-3-Nano-4B Q4_K_M | pass: identical | none | |
| 18 | meta-llama/Llama-3.2-1B-Instruct (unsloth mirror) | llama | 2.5 GB | pass | bartowski/Llama-3.2-1B-Instruct-GGUF Q8_0 | pass: identical | none | |
| 19 | mistralai/Mistral-7B-Instruct-v0.2 | mistral (SentencePiece) | 14.5 GB | pass after fix: tokenizer and chat prompt ids identical, ids identical | TheBloke Mistral-7B-Instruct-v0.2 Q4_K_M | pass: teacher forcing vs transformers on the same GGUF (max 0.21, no flip); prompt ids follow llama.cpp's space-prefix placement by design | non-legacy Metaspace `prepend_scheme: first` not honoured (`▁` after `<s>`, double `▁` before a leading space) | 1f207421 |
| 20 | openbmb/MiniCPM5-2B | llama (MiniCPM 5 pre-tokenizer) | 5.0 GB | pass | bartowski/MiniCPM5-2B-GGUF Q8_0 | pass: identical | none | |
| 21 | LiquidAI/LFM3-1.7B (trending) | lfm3 | 3.5 GB | pass | LiquidAI/LFM3-1.7B-GGUF Q8_0 | pass vs transformers by teacher forcing (llama.cpp b11177 cannot load `lfm3`): mean logit difference 0.09–0.19, only exact ties flip | none | |

Larger models (GGUF vs llama.cpp b11177; tokenizer and templates against transformers with the original repository's tokenizer files):

| Model | Architecture | Tested | Placement | Result |
|---|---|---|---|---|
| Qwen/Qwen3-32B | qwen3 | Qwen/Qwen3-32B-GGUF Q4_K_M (19.8 GB) | one GPU | pass: tokenizer and templates; greedy near-tie (ours agree with llama.cpp longer than its own backends do) |
| Qwen/Qwen3-30B-A3B | qwen3moe | Qwen/Qwen3-30B-A3B-GGUF Q4_K_M (18.6 GB) | one GPU | pass: tokenizer and templates; greedy identical to llama.cpp's CPU build (its Vulkan build did not fit beside the other users of the shared GPU) |
| zai-org/GLM-4.7-Flash | deepseek2-style MLA MoE | unsloth/GLM-4.7-Flash-GGUF Q4_K_M (18.3 GB) | one GPU | pass: tokenizer and templates; greedy cross-device near-tie |
| Qwen/Qwen3.6-35B-A3B | qwen35moe (hybrid MoE) | unsloth/Qwen3.6-35B-A3B-GGUF UD-Q4_K_M (22.1 GB) | one GPU | pass: tokenizer, templates and greedy text identical to llama.cpp |
| unsloth/Llama-3.3-70B-Instruct | llama | bartowski Q4_K_M (42.5 GB) | 2 GPUs | pass: identical to llama.cpp CUDA; llama.cpp Vulkan differs at a near-tie in one prompt |
| openai/gpt-oss-120b | gpt_oss | ggml-org MXFP4 (63.4 GB) | 2 GPUs + CPU | pass: harmony templates; one prompt identical, the other a spacing near-tie ("a,b = 0,1" / "a,b=0,1") with identical text after |
| Qwen/Qwen3.8-27B | qwen3_5 | lmstudio-community Q8_0 (29 GB); Qwen/Qwen3.8-27B-FP8 (30.9 GB) | 2 GPUs | Q8_0: identical to llama.cpp. FP8 safetensors vs transformers on the bf16 checkpoint (fp32, CPU): raw prompts identical, chat identical to the end of turn, teacher-forced logit difference mean 0.16–0.19 |
| Qwen/Qwen3.8-Flash-Next | qwen4exp | unsloth UD-IQ1_S (66.4 GB, 3 shards) | 2 GPUs + CPU | pass: tokenizer, templates and greedy text identical to llama.cpp. The MTP draft file (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`) loads as the draft source (CLI `--draft`): acceptance 0.78 / 0.71 / 0.58 at k = 1 / 2 / 3 (benchmarks section 17.5) |

Fixed in this sweep (each with a regression test):

| Problem | Model | Layer | Fix | Test |
|---|---|---|---|---|
| `opt` not mapped | facebook/opt-125m | HF importer | `OptHfMapping` (learned positions minus their two-row offset through the new `TryMapSpecialGlobalTensor` hook, ReLU MLP, biases; post-norm and projected-embedding OPT refused with a reason) | tiny-random-opt Families case (`tools/mk_tiny_opt.py`) |
| Tokenizers without tokenizer.json (vocab.json + merges.txt) refused | facebook/opt-125m (and every GPT2Tokenizer / RobertaTokenizer-style checkpoint) | HF tokenizer translation | the tokenizer.json transformers' GPT2Converter builds is synthesized (ByteLevel split, `add_prefix_space`, special tokens, BOS template for `add_bos_token`) | `SlowBpeTokenizerTests` |
| Metaspace `prepend_scheme: first` treated as llama.cpp's prefix: `▁` after `<s>` in every chat prompt, `▁▁` before a leading space | mistralai/Mistral-7B-Instruct-v0.2 (non-legacy Llama-2-style tokenizer.json) | Core tokenizer + HF translation | neutral `tokenizer.space_prefix_scheme` ("first" / "always"; absent = llama.cpp's placement, so GGUF keeps matching llama.cpp) | `SentencePieceTests` |
| tokenizer_config.json's `pretokenize_regex` ignored (transformers 5 applies it over tokenizer.json's split): Devanagari vowel signs split differently | ornith-ai/Ornith-1.0-9B (Qwen3.5 derivatives) | HF tokenizer translation | a known `pretokenize_regex` selects the preset | `PretokenizeRegexTests` |

Not Sharplight bugs, recorded for reference: the official Qwen2.5 GGUFs embed an older chat template whose tool instructions print a
literal `{{"name": ...}}` (transformers renders that template identically); Mistral GGUF prompt ids follow llama.cpp's space-prefix
placement and so differ from transformers' after special tokens (intended: llama.cpp is the reference for GGUF); LFM3 GGUF could not be
compared with llama.cpp b11177, which lacks `lfm3` (checked against transformers by teacher forcing instead).

Open:

1. **Row invariance of speculative verification on qwen4exp.** A verification step (2-4 rows) must compute each row exactly as a
   one-row step does. Fixed: CPU matmuls (up to 4 rows run the single-row arithmetic), the Vulkan expert matmul (row-invariant
   steps stay on the expert matvec). `SpeculativeRowInvarianceTests` pass on Vulkan for tiny Llama, Qwen3.5, Qwen3.8-MoE,
   Qwen3-Next-MoE and Qwen3.8-Flash-Next. Open: (a) tiny Qwen3.8-Flash-Next on the CPU still differs in the last bits; it is not the
   n-gram embedding, the QSA indexer, attention or the matmuls (bisected with config variants and a one-layer model; dense and MoE
   Qwen pass), which leaves the hyper-connection mixers' CPU ops (Tile, the permuted ReduceSum) and the sigmoid-gated delta rule;
   (b) the real Qwen3.8-Flash-Next (UD-IQ1_S, two GPUs, n-gram table on the CPU) still meets plain greedy output at one near-tie
   with MTP (benchmarks section 17.5.2), so some op of the real model (or its CPU table gathers) is not row-invariant yet.
2. **OPT variants.** Post-norm OPT (`do_layer_norm_before = false`, OPT-350m) and projected embeddings (`word_embed_proj_dim` other than the
   hidden size) are refused by the mapping; both are small additions to the decoder (a post-norm layer flag and in/out projections).
3. **Larger safetensors.** bf16 checkpoints above ~30 GB were not run through the HF importer (only GGUF, and FP8 for Qwen3.8-27B), by
   the sweep's disk rule.

## 19. Hot-model sweep 2026-11-17

The product owner's request: the 20 most popular models per category, current rather than all-time, because older models run but many hot new
ones were untested. Selection rules:

- **Ranking.** Hub API, `GET /api/models?pipeline_tag=<tag>&sort=trendingScore|downloads&limit=50`, queried on 2026-11-17 with the tags named in
  each section. `downloads` is the Hub's 30-day count, `trendingScore` its current trending score; the tables give the 30-day downloads (dl30)
  and likes. Models created or updated in the last ~6 months come first; where a category has few new releases (vision) the list is filled with
  the most downloaded of the last 30 days.
- **Official checkpoints only.** The developer's own repository (openai, nvidia, Qwen, mistralai, microsoft, google, facebook, ...), and
  conversions or quantizations only from the developer itself or a trusted provider (unsloth, ggml-org, onnx-community, Xenova where it is the
  de-facto official ONNX). A hot model that exists only as community conversions is listed as "community only" and not tested. The publisher is
  recorded in the catalog entry (`publisher`) and in the tables as "Publisher".
- One entry per family and size; fine-tunes and re-uploads skipped. Licenses that restrict use (non-commercial, AGPL, gated) are marked.
- Already verified models (docs/compatible-models.md, `catalog/verified-models.json`) are listed with their earlier result.
- Downloads go to `~/models/hot-b`; references run in the sandboxed venvs (transformers 5.23, float32, CPU).

Architecture descriptions in the "needs arch" cells are read from the Hub metadata (`model_type`, class names, file lists, model cards), not from
the modeling code; each needs its own read before it is estimated. "n/a" in the release / downloads cells: not queried individually.

Status legend: **verified** (this sweep or earlier: output compared with the reference), **needs arch** (the architecture is not implemented; the
column says what is missing), **community only**, **not run** (selected, not reached in this round).

### 19.1 Speech to text (`automatic-speech-recognition`, trending + downloads)

| # | Model | Publisher | Released | dl30 / likes | Format | Status |
|---|---|---|---|---|---|---|
| 1 | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B | NVIDIA | 2026-10-07 | 85k / 3.4k | safetensors (`nemo_speechlm`, 33.7 B) | needs arch: speech-LM wrapper (`Nemotron3_5TranscribeForConditionalGeneration`); non-commercial "other" license; no GGUF/ONNX from NVIDIA |
| 2 | nvidia/nemotron-3.5-asr-streaming-0.6b | NVIDIA | 2026-05-15 | 481k / 1.3k | `.nemo`, NVIDIA's own `q8_0.gguf`, HF `nemotron3_5_asr` | **verified (round 5)**: cache-aware streaming FastConformer, token-exact vs transformers on CPU and Vulkan (section 13.3); "other" license |
| 3 | openai/whisper-large-v3 | OpenAI | 2023-11 | 3.3M / 6.8k | HF, whisper.cpp, GGUF | verified (section 13.1) |
| 4 | openai/whisper-large-v3-turbo | OpenAI | 2024-10 | 4.8M / 3.6k | HF, whisper.cpp ggml | verified |
| 5 | Qwen/Qwen3-ASR-1.7B | Alibaba (Qwen) | 2026-01-28 | 1.6M / 1.3k | HF (both layouts), GGUF | verified (section 13.2) |
| 6 | Qwen/Qwen3-ASR-0.6B | Alibaba (Qwen) | 2026-01-28 | 306k (`-hf`: 527k) / 400 | HF `-hf` | **verified this sweep**: CPU token-exact vs transformers on jfk and LibriSpeech (auto language and English forced); Vulkan: jfk identical, LibriSpeech differs at one comma (section 19.1.1) |
| 7 | nvidia/parakeet-tdt-0.6b-v3 | NVIDIA | 2025-08-04 | 529k / 1.3k | HF `parakeet_tdt`, GGUF | verified (section 13) |
| 8 | microsoft/VibeVoice-ASR | Microsoft | 2026-01-21 | 719k / 1.3k | safetensors (`vibevoice`, 8.7 B) | **verified (2026-11-18)**: acoustic + semantic tokenizers (conv encoders) feeding a Qwen2 decoder with speaker-diarized, timestamped long-form output; token-exact greedy vs transformers on jfk.wav and a two-speaker conversation, CPU and Vulkan (section 13.4); MIT |
| 9 | mistralai/Voxtral-Mini-4B-Realtime-2602 | Mistral AI | 2026-01-21 | 1.5M / 1.1k | safetensors (`voxtral_realtime`, 4.4 B) | **verified (round 5)**: causal streaming audio tower and delayed-token decoder, token-exact vs transformers on CPU and Vulkan (section 13.3) |
| 10 | CohereLabs/cohere-transcribe-03-2026 | Cohere Labs | 2026-03-24 | 94k / 1.2k | HF `cohere_asr` (gated, auto), handy-computer GGUF | needs arch `cohere_asr` (conformer encoder + autoregressive decoder, close to the Canary path); not attempted in round 5: the gate needs accepted terms and the machine's token is not approved for it |
| 11 | ibm-granite/granite-speech-5.0-2b | IBM | 2026-10-07 | 11k / 85 | HF `granite_speech` (2.3 B) | needs arch: conformer encoder + window Q-former projector + Granite LLM (the Granite text model is supported) |
| 12 | google/medasr | Google | 2025-12-18 | 7k / 389 | HF `lasr_ctc` (105 M, gated, auto) | needs arch: LASR conformer CTC; Health AI license |
| 13 | ai-sage/GigaAM-v3 | Sber | 2025-11-19 | 342k / 210 | `pytorch_model.bin` (`gigaam`, MIT) | needs arch: GigaAM conformer (RNN-T / CTC heads, custom code) |
| 14 | nvidia/parakeet-ctc-1.1b | NVIDIA | 2023-12-28 | 47k / 67 | HF `parakeet_ctc`, NVIDIA's own `q8_0.gguf` | not run (maps like the verified parakeet-ctc-0.6b) |
| 15 | moonshine-ai/moonshine-streaming-medium | Moonshine AI | 2026-01-06 | 44k / 35 | HF `moonshine_streaming` | needs arch: sliding-window streaming encoder |
| 16 | kyutai/stt-2.6b-en | Kyutai | 2025-06-06 | 0 / 128 | safetensors (moshi) | needs arch: Mimi-token delayed-stream LM (the Mimi codec is supported) |
| 17 | facebook/omniASR-LLM-7B | Meta | 2025-11-27 | 0 / 49 | fairseq2 checkpoint | needs arch: wav2vec2 encoder + LLM decoder in fairseq2 layout |
| 18 | nvidia/canary-qwen-2.5b | NVIDIA | 2025-06-26 | 25k / 474 | `.nemo` | needs a `.nemo` (tar of NeMo checkpoint) reader plus the FastConformer + Qwen3 SALM wrapper |
| 19 | openai/whisper-small | OpenAI | 2022-09 | 4.5M / 657 | HF, whisper.cpp | verified |
| 20 | facebook/wav2vec2-base-960h | Meta | 2022-03 | 1.8M / 418 | HF | verified (section 13.1) |

Community-only hot models, not tested: fluid-speech-pro (FluidVoice-Labs), SOV-Transcribe-1, moondream/parakeet-ultra and
handy-computer/*-gguf re-uploads of the NVIDIA models (the transcribe.cpp GGUF files were already run in section 13.1), indic and language
fine-tunes. Systran/faster-whisper-* (CTranslate2, 1.7-6.3M dl30) is a trusted conversion of the OpenAI models but the CTranslate2 `model.bin`
format has no importer; the same models import from the OpenAI checkpoints.

#### 19.1.1 Qwen3-ASR 0.6B

`Qwen/Qwen3-ASR-0.6B-hf` through the existing `qwen3_asr` mapping, no code change: the fixtures come from
`tests/Sharplight.Speech.Tests/tools/make_qwen3_asr_fixtures.py` (transformers 5.23, float32; `Fixtures/qwen3-asr-0.6b.json`), the test is
`Qwen3AsrParityTests.SmallModelTokensMatchTransformers`.

- CPU: greedy ids identical on jfk.wav and LibriSpeech 1272-128104-0004, language auto-detected ("English") and forced.
- Vulkan: jfk identical; LibriSpeech drops one comma ("a sort of, up guards" against "a sort of up guards"). It is a near-tie: after "a sort of" the reference
  logits of "," and "up" are 29.405 and 29.402 (gap 0.004), which the GPU path's f16 compute resolves the other way. The test leaves that one clip
  out on Vulkan.

### 19.2 Text to speech incl. voice cloning (`text-to-speech`, trending + downloads)

| # | Model | Publisher | Released | dl30 / likes | Format | Status |
|---|---|---|---|---|---|---|
| 1 | Zyphra/ZONOS3-Mini | Zyphra | 2026-11-09 | 94k / 501 (trending #1) | Zyphra repository (no `config.json`; `-hf` conversion and a GGUF by AtomicChat, community) | not run: needs inspection of the architecture; voice cloning |
| 2 | hexgrad/Kokoro-82M | hexgrad | 2024-12-26 | 13.0M / 7.4k | PyTorch folder; onnx-community ONNX (877k) | verified (section 12) |
| 3 | k2-fsa/OmniVoice | k2-fsa | 2026-03-30 | 4.2M / 1.7k | safetensors (`omnivoice`, 613 M) | **verified, voice cloning and voice design (section 12)**: masked-diffusion acoustic model (Qwen3-0.6B over 8 audio codebooks, bidirectional, 32 unmasking steps with guidance) + the Higgs Audio v2 tokenizer (section 16); logits 6e-7, greedy codes identical, audio 2.5e-6 against the official package (CPU and Vulkan). Licenses (approved by the owner, shown in Studio's Discover as the catalog entry's license and notes, and installed with the model): code Apache 2.0, pre-trained weights CC-BY-NC (non-commercial, from its Emilia training data), bundled audio tokenizer the Boson Higgs Audio 2 community license (a Llama-3-style agreement, an expanded license from Boson AI needed above 100k annual active users). Like XTTS (CPML), Studio has no accept step: the notice is the catalog entry's license badge and notes |
| 4 | coqui/XTTS-v2 | Coqui | 2023-10-31 | 5.2M / 4.0k | PyTorch folder | verified, voice cloning (section 12); CPML non-commercial license |
| 5 | ResembleAI/chatterbox | Resemble AI | 2025-04-24 | 2.5M / 1.9k | safetensors folder | verified, voice cloning |
| 6 | SWivid/F5-TTS | SWivid | 2024-10-07 | 1.4M / 1.2k | safetensors | verified, voice cloning |
| 7 | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (+ 0.6B-Base, 0.6B-CustomVoice, 1.7B-VoiceDesign, Qwen3-TTS-Tokenizer-12Hz) | Alibaba (Qwen) | 2026-01-21 | 1.17M (family 3.4M) / 2.2k | HF `qwen3_tts` (0.9-1.7 B), tokenizer `qwen3_tts_tokenizer_12hz` | 0.6B-CustomVoice verified (section 12; 1.7B-CustomVoice maps the same way, not run); Base (voice cloning: ECAPA speaker encoder + reference codes) and VoiceDesign (instructions) not implemented, rejected on import |
| 8 | openbmb/VoxCPM2 | OpenBMB | 2026-04-03 | 980k / 1.8k | safetensors (2.3 B, BF16; `audiovae.pth` 377 MB) | needs arch (inspected 2026-11-19, not implemented): license Apache 2.0 (commercial use allowed, no `licenseNotice` needed). Parts of `config.json` / the checkpoint: MiniCPM4 backbone (28 layers, width 2048, 16 query / 2 KV heads, head dim 128, LongRoPE with short and long factors, muP scaling `scale_emb` 12 and `scale_depth` 1.4, vocabulary 73448), an 8-layer residual LM without rope, a 12-layer feature encoder and a 12-layer local DiT (width 1024, flow matching with an Euler solver, classifier-free guidance 2.0, log-norm timestep schedule) over 4-frame patches of 64-wide AudioVAE latents, a finite scalar quantizer (512), a stop head, and the AudioVAE V2 (16 kHz in, 48 kHz out through its built-in super-resolution; PyTorch `audiovae.pth`); the official package is `voxcpm` 2.0.3 (`model/voxcpm2.py`, `modules/audiovae/audio_vae_v2.py`) |
| 9 | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | Alibaba (FunAudioLLM) | 2025-12 | n/a | PyTorch folder | verified, voice cloning |
| 10 | fishaudio/drama-3 | Fish Audio | 2026-10-01 | 44k / 1.3k (Jmica GGUF community: 559k) | safetensors | not run: architecture not inspected; license to check |
| 11 | fishaudio/s2-pro | Fish Audio | 2026-03-09 | 46k / 1.5k | safetensors | not run (Fish "dual-AR" with DAC codec; the DAC codec is supported) |
| 12 | microsoft/VibeVoice-1.5B | Microsoft | 2025-08-25 | 216k / 2.5k | safetensors (`vibevoice`, 2.7 B) | **verified (2026-11-18)**: Qwen2 LM + diffusion head over continuous acoustic latents (7.5 Hz tokenizers), multi-speaker scripts, voice prompts, streaming; teacher forced against transformers on CPU and Vulkan (section 12.1); MIT |
| 13 | OpenMOSS-Team/MOSS-TTS-Local-Transformer | OpenMOSS | 2026-02-08 | 204k / 32 | safetensors | not run |
| 14 | gradium/phonon-31m-en | Gradium | 2026-10-19 | 321k / 167 | safetensors (31 M) | not run (small; candidate for a quick bring-up) |
| 15 | KittenML/kitten-tts-nano-0.8, kitten-tts-2-mini | KittenML | 2026-02-19 / 2026-11-13 | 261k / 2.8k | ONNX (Kokoro-style StyleTTS 2) | not run (ONNX graph; the espeak phonemizer is the open part) |
| 16 | ai4bharat/indic-parler-tts | AI4Bharat | 2024-10-28 | 176k / 581 | safetensors (Parler: T5 + DAC decoder LM) | needs arch `parler_tts` (DAC codec supported) |
| 17 | microsoft/speecht5_tts | Microsoft | 2023-02-02 | 172k / 852 | safetensors | needs arch `speecht5` (+ HiFi-GAN vocoder, x-vector speaker) |
| 18 | canopylabs/orpheus-3b-0.1-ft | Canopy Labs | 2025 | n/a | safetensors (gated, auto); unsloth mirror | verified through the unsloth mirror (section 12) |
| 19 | OuteAI/OuteTTS-0.2-500M | OuteAI | 2024 | n/a | GGUF | verified (section 12) |
| 20 | sesame/csm-1b | Sesame | 2025 | n/a | safetensors (gated, auto) | verified, CPU and Vulkan (section 12; no audio context) |
| 21 | nari-labs/Dia-1.6B | Nari Labs | 2025 | n/a | safetensors (original layout, 6.4 GB) + descript/dac_44khz | **verified (2026-11-19, section 12.2)**: dialogue TTS with `[S1]`/`[S2]` tags, delay-pattern DAC codebooks, classifier-free guidance, audio-prompt voices; teacher-forced logits 2e-4, greedy sequences identical, DAC audio 4e-5 against transformers on CPU and Vulkan; Apache 2.0 |
| 22 | 2Noise/ChatTTS | 2Noise | 2024 | n/a | safetensors (`asset/`: GPT, DVAE, Vocos) | not run: needs arch (GPT + DVAE + Vocos); CC-BY-NC-4.0 (non-commercial; a `licenseNotice` would be needed) |

Community only: ampixa/sanoTTS, remsky/kokoro-inno-clone-tuner, pnnbao-ump VieNeu-TTS (a fine-tune family), drbaph/Higgs-Audio-v3-Studio,
HoppouAI/Breeze-TTS-2.cpp, audio-cpp/* GGUF collections (a third-party runtime's conversions; not tested).

### 19.3 Audio: codecs, music generation, separation, embeddings and classification, VAD (`audio-to-audio`, `audio-classification`, `voice-activity-detection`, `text-to-audio`)

| # | Model | Publisher | Released | dl30 / likes | Format | Status |
|---|---|---|---|---|---|---|
| 1 | facebook/encodec_24khz | Meta | 2023 | n/a | HF | verified (section 16) |
| 2 | descript/dac_44khz | Descript | 2023 | n/a | HF | verified |
| 3 | kyutai/mimi | Kyutai | 2024 | n/a | HF | verified |
| 4 | Qwen/Qwen3-TTS-Tokenizer-12Hz | Alibaba (Qwen) | 2026-01-21 | 57k / 103 | HF `qwen3_tts_tokenizer_12hz` (171 M) | verified: codes token-exact, decode relative RMS 6e-6 (CPU and Vulkan; section 16) |
| 5 | neuphonic/neucodec | Neuphonic | 2025-06-19 | 49k / 194 | safetensors (630 M, gated auto) | not run: FSQ codec with a Wav2Vec2-BERT semantic encoder |
| 6 | nvidia/bigvgan_v2_22khz_80band_256x (+ 44khz 128band) | NVIDIA | 2024-07-15 | 1.8M, 1.7M / 43, 85 | PyTorch (`bigvgan` package) | not run: vocoder (snake activations, anti-aliased upsampling); mel to wave |
| 7 | ACE-Step/Ace-Step1.5 | ACE-Step | 2026-01-23 | 105k / 976 | safetensors + 5 Hz LM (`acestep`) | turbo transformer verified (section 16: latents 8.1e-6 CPU / 2.8e-4 Vulkan, audio 4.3e-5 / 8.3e-4); the autoencoder is Oobleck (not a DCAE), the text encoder Qwen3-Embedding; not done: the 5 Hz LM planner (Qwen3-style, audio codes through the FSQ tokenizer and detokenizer for cover and structured generation), the base and SFT transformers (guidance), reference audio |
| 8 | stabilityai/stable-audio-3-medium (small-sfx, small-music) | Stability AI | 2026-05-17 | 134k / 620 | safetensors (gated auto, community license) | not run: gated, the account has not accepted the terms (weights return 403; the model card says a T5Gemma conditioner and a "semantic-acoustic" autoencoder, so it is not the Open 1.0 architecture); the Stability AI Community License allows research and non-commercial use and commercial use below 1 M USD annual revenue with registration, requires the notice file and "Powered by Stability AI", forbids using outputs to train other foundation models; the weights also carry the Gemma Terms of Use |
| 9 | stabilityai/stable-audio-open-1.0 | Stability AI | 2024-05-24 | 23k / 2.0k | safetensors (gated) | not run: gated (403 until the terms are accepted on the model page); diffusers layout (`StableAudioDiTModel`, `AutoencoderOobleck` (decoder graph exists since ACE-Step), T5 text encoder, projection model, seconds conditioning, v-objective sampler); same Community License as row 8 |
| 10 | facebook/musicgen-small, -medium | Meta | 2023-06-08 | 194k, 2.0M / 533, 179 | HF `musicgen` | small verified (section 16); medium (CC-BY-NC) maps the same way, not run |
| 11 | m-a-p/YuE2-3B | M-A-P | 2026-09-09 | 21k / 1.3k | safetensors | needs arch: two-stage LM (Llama-style, supported) + xcodec-style audio codec |
| 12 | MiniMaxAI/MiniMax-Music3 | MiniMax | 2026-08-07 | 5k / 1.5k | safetensors | not run: size and architecture not inspected |
| 13 | google/magenta-realtime-2 | Google | 2026-05-28 | 18k / 267 | not inspected | not run |
| 14 | MIT/ast-finetuned-audioset-10-10-0.4593 | MIT CSAIL | 2022-11-14 | 1.9M / 392 | HF `audio-spectrogram-transformer` (87 M) | not run: ViT over fbank patches; needs an audio-classification task or reuses the CLAP audio path |
| 15 | laion/clap-htsat-fused | LAION | 2023-02-16 | 9.4M / 179 | HF `clap` | rejected on purpose: the fusion variant is not implemented (the unfused laion/clap-htsat-unfused is verified) |
| 16 | pyannote/segmentation-3.0, pyannote/speaker-diarization-community-1 | pyannote | 2023-09-22 / 2025-04-15 | 4.7M, 4.9M / 2.7k, 4.6k | PyTorch (gated auto) | not run: PyanNet (SincNet + LSTM) segmentation; powerset decoding |
| 17 | nvidia/Nemotron-3-Diarization | NVIDIA | 2026-09-01 | 232k / 842 | safetensors (`nemotron3_diarization`, 99 M); onnx-community ONNX | not run: streaming Sortformer-style diarizer |
| 18 | snakers4/silero-vad v5 / v6 | Silero | 2025 | n/a | ONNX | v5 verified (section 16); v6.2 only as handy-computer GGUF (community) |
| 19 | nvidia/NVIDIA-NemotronLabs-RE-USE | NVIDIA | 2026-07-22 | 48k / 230 | safetensors | not run: speech enhancement (ReUSE) |
| 20 | facebook/htdemucs family, m-a-p/MERT-v1-95M, audeering/wav2vec2-large-robust-24-ft-age-gender (2.4M), facebook/audiobox-aesthetics (367k) | Meta, M-A-P, audeering | various | | HF | htdemucs verified (KeighBee conversion, section 16); the rest not run (wav2vec2-family encoders with heads: reuse the wav2vec2 graph) |

Community only: audio-cpp/* GGUF collections, JusperLee/TIGER-DnR (separation research code, community of the paper's authors, not tested),
chenmozhijin/BSRoformer-GGUF (500k-ish), handy-computer silero / sortformer GGUF files, RVC voice models.

### 19.4 Vision (`image-classification`, `image-feature-extraction`, `object-detection`, `image-segmentation`, `mask-generation`, `depth-estimation`, `image-to-text`, `image-to-image`, `keypoint-detection`)

Few vision models are new, so the lists mix new releases (DINOv3, SAM 3, RF-DETR, YOLO26, GLM-OCR, DA3, RMBG-3.0) with the most downloaded of the
last 30 days.

| # | Task | Model | Publisher | Released | dl30 / likes | Status |
|---|---|---|---|---|---|---|
| 1 | classification | google/vit-base-patch16-224 | Google | 2022 | 15.7M / 1.1k | verified (section 14) |
| 2 | classification | timm/mobilenetv3_small_100.lamb_in1k | Ross Wightman (timm) | 2022-12 | 8.4M / 160 | **verified round 9** (section 14): small and large, logits 7.5e-5 / 1.0e-4 of timm, CPU and Vulkan |
| 3 | classification | timm/resnet50.a1_in1k, resnet18, resnet34, wide_resnet50_2 | timm | 2023 | 0.6-1.9M | verified |
| 4 | classification | timm/efficientnet_b0.ra_in1k, tf_efficientnetv2_s | timm | 2022-12 | 0.7M, 3.6M | **verified round 9** (section 14): B0 8.3e-5, V2-S (`tf_`, same padding, 300 px) 1.4e-4, CPU and Vulkan |
| 5 | classification | Falconsai/nsfw_image_detection | Falconsai | 2023-10 | 1.6M / 1.2k | verified (ViT) |
| 6 | embedding | facebook/dinov3-vitl16 / vitb16 / vits16 / vith16plus (lvd1689m) | Meta | 2025-08 | 697k / 344k / 174k / 60k | **vits16 and vitb16 verified round 9** (section 14: CLS embedding 2e-6 on a 224 px input, CPU and Vulkan); vitl16 / vith16plus (gated MLP) map the same way, not run; Meta license, gated (`HF_TOKEN`) |
| 7 | embedding | facebook/dinov2-base, -small, -large, dinov2-with-registers | Meta | 2023 | 3.6M / 2.5M / 0.6M | base, small, with-registers verified; large maps the same, not run |
| 8 | embedding | openai/clip-vit-*, google/siglip*, siglip2 | OpenAI, Google | 2021-2025 | | verified (several) |
| 9 | detection | Ultralytics/YOLO26 | Ultralytics | 2026-01-26 | 28k / 269 | needs work: YOLO26 NMS-free end-to-end head (ONNX export import exists for v8 / v10 / 11); AGPL-3.0 |
| 10 | detection | Roboflow/rf-detr-large | Roboflow | 2026-05-11 | 2.8k / 27 | **verified round 10** (section 14: `rf_detr`, same five detections as transformers, boxes within 0.3 px, CPU and Vulkan, also from an `.slm`); Apache-2.0 |
| 11 | detection | PekingU/rtdetr_r50vd, r101vd, v2_r18vd, r18vd | PekingU / Baidu | 2024-25 | 1.2M, 232k, 173k | verified |
| 12 | detection | microsoft/table-transformer-structure-recognition | Microsoft | 2022-10 | 1.24M / 245 | **verified this sweep**: 13 detections (rows, columns, column header, table) as transformers, CPU and Vulkan |
| 13 | detection | microsoft/table-transformer-detection, hustvl/yolos-small, facebook/detr-resnet-50 | Microsoft, HUST, Meta | 2022 | 270k, 756k, 84k | verified |
| 14 | layout | PaddlePaddle/PP-DocLayoutV3_safetensors | Baidu | 2026-01-20 | 817k / 59 | needs arch `pp_doclayout_v3` (RT-DETR family with a PP-HGNetV2 backbone and mask head) | **Verified round 11** (section 14): `pp_doclayout_v3`, same ten detections in the same reading order as transformers on a synthetic 800 px page, boxes within 1 px, CPU, Vulkan and `.slm`; masks and reading order in `DetectedObject`; Apache-2.0 |
| 15 | segmentation | facebook/sam3, sam3.1 | Meta | 2025-11-07 / 2026-03-26 | 2.0M, 155k / 6.8k, 1.5k | needs arch `sam3` (gated: manual; Meta SAM license); the SAM 2 image path is verified |
| 16 | segmentation | facebook/sam2.1-hiera-large, -small, -tiny, -base-plus | Meta | 2024-09 | 272k, 171k, 163k | tiny verified; the others map the same way, not run |
| 17 | matting | ZhengPeng7/BiRefNet, briaai/RMBG-2.0 | ZhengPeng, Bria | 2024 | 1.0M, 395k | verified; briaai/RMBG-3.0 (2026-11-10, 8.1 B, "other" license) and RMBG-1.4 (SegFormer-style IS-Net, 235k) not run |
| 18 | segmentation | nvidia/segformer-b0 / b2 / b5, facebook/mask2former-swin-large-ade-semantic, CIDAS/clipseg-rd64-refined | NVIDIA, Meta, CIDAS | 2022-23 | 285k, 362k, 387k | segformer verified; Mask2Former and CLIPSeg need arch |
| 19 | depth | depth-anything/Depth-Anything-V2-Small / Base / Large-hf | TikTok / HKU | 2024-06 | 1.3M, 81k, 206k | verified |
| 20 | depth | depth-anything/DA3-SMALL / DA3-LARGE / DA3METRIC-LARGE / DA3MONO-LARGE | ByteDance | 2025-11-13 | 174k, 473k, 458k, 200k | **SMALL and LARGE verified round 10** (section 14: `da3_depth`, depth 3.1e-5 / 1.8e-5 of the authors' network, CPU and Vulkan, also from an `.slm`); BASE maps the same, not run; the METRIC and MONO variants (plain DPT head with a sky branch) need a second head; LARGE is CC-BY-NC |
| 21 | depth | apple/DepthPro | Apple | 2024-10 | 6k / 565 | needs arch (two-scale DINOv2 patches + FOV head); Apple research license |
| 22 | OCR | zai-org/GLM-OCR | Z.ai | 2026-01-30 | 1.8M / 2.2k | **verified round 10** (greedy "Text Recognition:" text identical to transformers, CPU, Vulkan and `.slm`) and **behind `ITextRecognizer` since round 11** (section 14: tasks text, formula, table identical to transformers; `DocumentReader` pairs it with PP-DocLayoutV3); no multi-token prediction yet (the engine has no image rows) |
| 23 | OCR | microsoft/trocr-base-printed, -handwritten, -small-handwritten | Microsoft | 2022 | 349k, 132k, 245k | verified |
| 24 | OCR | PaddlePaddle/PP-OCRv5_server_det, _rec, UVDoc, PP-LCNet doc-orientation | Baidu | 2025 | 432k, 156k, 452k, 330k | needs a Paddle importer (inference graphs) |
| 25 | OCR | naver-clova-ix/donut-base, facebook/nougat-base, tiiuae/Falcon-OCR | NAVER, Meta, TII | 2022-2026 | 58k, 2k, 6k | needs arch (Donut / Nougat: Swin encoder + mBART decoder; Falcon-OCR not inspected) |
| 26 | upscaling | Real-ESRGAN x4plus, SwinIR classical x4 | xinntao, JingyunLiang | 2021 | | verified |
| 27 | pose | usyd-community/vitpose-base-simple (+ plus-base, plus-large, plus-huge) | USYD (ViTPose authors) | 2025-01-08 | 317k, 55k, 28k, 47k | needs a keypoint task (ViT backbone with a deconvolution / simple head; heat-map decode) |
| 28 | pose | facebook/sapiens2-pose-0.4b / 1b / 5b | Meta | 2026-04-23 | 10k, 51k, 15k | needs arch `sapiens2` (Meta license) |

Community only: Comfy-Org/* repacks of official models (BiRefNet, Depth-Anything-3), qualcomm/* on-device exports, Arm/*, pixai taggers.

### 19.5 Results of this round (2026-11-17)

New verifications (catalog entries added, publisher recorded):

| Model | Publisher | Format | Check | Result |
|---|---|---|---|---|
| Qwen/Qwen3-ASR-0.6B-hf | Alibaba (Qwen) | safetensors | greedy ids vs transformers (fp32), jfk and LibriSpeech, auto and English | CPU identical (4 of 4); Vulkan jfk identical, LibriSpeech one comma differs (19.1.1) |
| microsoft/table-transformer-structure-recognition | Microsoft | safetensors | detections, boxes, scores vs transformers on a table image (640 px canvas) | identical detections, boxes within 2 px, scores within 0.02; CPU and Vulkan |

Not reached in this round, in the order they would be taken next (each is a new architecture, none is a quick check): DINOv3 (ViT with RoPE: the
DINOv2 path plus a rotary position op), Cohere Transcribe and Granite Speech (conformer encoder, section 13's `ConformerEncoder`, plus an
autoregressive decoder), MobileNetV3 / EfficientNet (depthwise + squeeze-excite blocks on the existing `Conv` ops), GLM-OCR, ViTPose (needs a
`keypoint-detection` task kind), Moonshine streaming (the other streaming ASR models, Nemotron 3.5 ASR streaming and Voxtral
Realtime, were done in round 5, section 13.3).

## 20. Hot-model sweep 2026-11-17 (text generation, vision-language, embeddings and rerankers)

The sweep of section 17 ranked models by all-time downloads. This one takes the currently hot models: Hugging Face `trendingScore`,
downloads of the last 30 days (`downloads` of the Hub API) and likes, released or updated in roughly the last six months, one entry per
family and size. Only official checkpoints are tested: the developer's own repositories, and quantized variants only from the developer
or trusted providers (unsloth, ggml-org, bartowski, lmstudio-community, RedHatAI / neuralmagic for FP8). A hot model that exists only as
community conversions is listed with that note and not tested. Models that fit one 32 GB GPU run as the original checkpoint (up to
about 30 GB) or GGUF Q4_K_M; larger ones as GGUF with the CPU as fallback device, otherwise they are listed with the reason.

Sources (queried 2026-11-17, `limit=100`, with `expand[]=downloads,likes,trendingScore,createdAt,safetensors,gguf`):

```
GET https://huggingface.co/api/models?pipeline_tag=text-generation&sort=trendingScore
GET https://huggingface.co/api/models?pipeline_tag=image-text-to-text&sort=trendingScore
GET https://huggingface.co/api/models?pipeline_tag=feature-extraction&sort=trendingScore      (and sentence-similarity)
GET https://huggingface.co/api/models?pipeline_tag=text-ranking&sort=trendingScore
GET https://huggingface.co/api/models?filter=base_model:quantized:<repo>&sort=downloads        (the most used quantization of a model)
```

Method: `bench/ParitySweep` and `tools/parity/` as in section 17 (tokenizer: 7 texts; chat template: 4 conversations incl. tools; greedy
48 tokens against transformers fp32 for safetensors, against llama.cpp b11177 on Vulkan and CPU for GGUF with the near-tie policy). Where
a GGUF embeds a template that differs from the original repository's, transformers renders the GGUF's template. Image inputs: the CLI
(`--image`, `--mmproj`) against `llama-mtmd-cli` b11177, greedy, one synthetic image (a red square, a blue circle, a green triangle and
the text "HELLO 42") with the same prompt and the same thinking setting. Embeddings and rerankers: `Sharplight.Embeddings.Tests`
`ReferenceTests` (cosine against sentence-transformers, rerank logits against the model's own transformers code).


### 20.1 Hot-model sweep 2026-11-17: text generation

Selection (trending rank within `pipeline_tag=text-generation` and, where marked VL, `image-text-to-text`; "dl30" = downloads in the last 30
days, "trend" = Hub trendingScore at query time):

| # | Model | Released | dl30 / likes / trend | Size and format tested | Publisher | Status |
|---|---|---|---|---|---|---|
| 1 | Qwen/Qwen3.8-27B (VL) | 2026-08-05 | 6.9 M / 20.4 k / 481 | Q8_0 and FP8 (section 17) | Alibaba Cloud (Qwen) | verified in section 17 |
| 2 | google/gemma-4-31B-it (VL) | 2026-03-11 | 6.3 M / 4.4 k / 35 | unsloth Q4_K_M 18.3 GB + mmproj | Google | see 20.1.1 |
| 3 | Qwen/Qwen3.5-9B (VL) | 2026-02-27 | 6.2 M / 2.4 k / 32 | unsloth Q4_K_M 5.7 GB + mmproj | Alibaba Cloud (Qwen) | see 20.1.1 |
| 4 | Qwen/Qwen3.5-4B (VL) | 2026-02-27 | 4.8 M / 1.2 k / 26 | BF16 safetensors 9 GB; unsloth Q4_K_M + mmproj | Alibaba Cloud (Qwen) | see 20.1.1 |
| 5 | Qwen/Qwen3.6-35B-A3B (VL) | 2026-04-15 | 3.9 M / 3.1 k / 30 | unsloth UD-Q4_K_M (section 17) | Alibaba Cloud (Qwen) | verified in section 17 |
| 6 | Qwen/Qwen3.8-Flash-Next (VL, 180 GB) | 2026-08-24 | 3.0 M / 7.3 k / 64 | unsloth UD-IQ1_S, 2 GPUs + CPU (section 17) | Alibaba Cloud (Qwen) | verified in section 17 |
| 7 | NousResearch/Arete-27B (VL) | 2026-11-06 | 10.7 k / 679 / 95 | bartowski Q4_K_M 17.2 GB + mmproj | Nous Research | see 20.1.1 |
| 8 | Nanbeige/Nanbeige4.3-33B | 2026-10-24 | 23.8 k / 1.1 k / 18 | bartowski Q4_K_M 20.8 GB | Nanbeige | see 20.1.1 |
| 9 | RecursiveSI/flint | 2026-10-21 | 59.3 k / 1.4 k / 28 | unsloth UD-Q4_K_XL 25 GB | RecursiveSI | see 20.1.1 |
| 10 | KRAFTON/Raon-1-nano | 2026-10-29 | 1.9 k / 148 / 49 | not run | KRAFTON | unsupported architecture (20.1.2) |
| 11 | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B | 2026-10-16 | 15.2 k / 265 / 18 | not run | NVIDIA | unsupported architecture (20.1.2) |
| 12 | maxwell-model/Maxwell-V1-Flash | 2026-10-13 | 287 / 124 / 122 | not run (36.8 GB bf16, remote code) | Maxwell | unsupported architecture (20.1.2) |
| 13 | LiquidAI/LFM3-1.7B | 2026-10-23 | 110 k / 1.9 k / 49 | section 17 | Liquid AI | verified in section 17 |
| 14 | nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B | 2026-10-05 | 158 k / 463 / 47 | catalog entry (unsloth GGUF) | NVIDIA | verified earlier (catalog) |
| 15 | openai/gpt-oss-120b | 2025-08-04 | 2.4 M / 5.5 k / 22 | ggml-org MXFP4, 2 GPUs + CPU (section 17) | OpenAI | verified in section 17 |
| 16 | meta-llama/Llama-3.1-8B-Instruct | 2024-07-18 | 4.2 M / 9.0 k / 57 | section 17 | Meta | verified in section 17 (older, still trending) |
| 17 | ornith-ai/Ornith-1.5-35B-A3B | 2026-08-18 | 327 k / 800 / 17 | only community GGUF (mudler, peculiar-ragdoll) for the 36 GB checkpoint | Ornith | not tested: community conversions only; the architecture is qwen3_5_moe (Qwen3.6-35B-A3B passes) |
| 18 | ibm-granite/granite-4.1-8b-llmon-preview, webAI-Official/TwIL-8B | 2026-10 | 0.7 k, 62 k / 20, 436 / 20, 28 | no trusted GGUF (mradermacher, liodon-ai only); granite arch (mapped) | IBM, webAI | not tested: community conversions only |
| 19 | whittleOrg/Whittle-Next-27B-A3B | 2026-09-03 | 1.2 k / 64 / 34 | qwen4_exp; 137-file repository, only community GGUF | Whittle | not tested: community conversions only |
| 20 | zai-org/GLM-5.3-Flash (321 GB), deepseek-ai/DeepSeek-V4.1-Flash(-1017) (763 GB), MiniMaxAI/MiniMax-M3.1-Flash (220 GB), thinkingmachines/Inkling-2-Small (594 GB), moonshotai/Kimi-K3(.1-Pro) (2.8 TB) | 2026-06..11 | 0.07-6.4 M | unsloth GGUF: GLM-5.3-Flash UD-IQ1_S 93 GB, MiniMax-M3.1-Flash UD-IQ1_M 129 GB, DeepSeek-V4.1-Flash-1017 UD-IQ1_S 233 GB (not downloaded yet) | Z.ai, DeepSeek, MiniMax, Thinking Machines, Moonshot | open: the original checkpoints are far beyond one 32 GB GPU; the unsloth IQ1 GGUFs of GLM-5.3-Flash and MiniMax-M3.1-Flash are feasible on GPU 0 plus CPU offload (251 GB RAM) but were not run in this increment (machine load, local disk 56 GB free); their architectures (glm5_next, minimax_m3, deepseek_v41) are verified on tiny models (section 4) |

Hub entries that are fine-tunes, abliterations, "uncensored" or quantization re-uploads of the same families (DavidAU, huihui-ai, HauhauCS,
ISTA-DASLab GSQ-RCO, xtools-at, Jackrong, ...) were skipped by the one-per-family rule.

#### 20.1.1 Results (increment 1)

Greedy "pass" for GGUF follows section 17's near-tie policy (ours on Vulkan or CPU equals one of llama.cpp's Vulkan / CPU outputs).

| Model | Release | Popularity | Format | Result | Fix |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 2026-02-27 | 4.8 M dl30 | BF16 safetensors (Qwen) | pass: 7 tokenizer texts, 4 chat templates (text and ids) and 3 x 48 greedy ids identical to transformers fp32 | none |
| unsloth/Qwen3.5-4B-GGUF Q4_K_M | | 1.6 M | GGUF + mmproj F16 | pass: ours on the CPU identical to llama.cpp on the CPU; Vulkan differs from llama.cpp Vulkan only at a spacing near-tie (`n-1` / `n - 1`); image prompt: same description as llama-mtmd-cli (thinking trace identical for 35+ tokens, then a near-tie) | none |
| Qwen/Qwen3.5-9B (unsloth GGUF Q4_K_M) | 2026-02-27 | 6.2 M | GGUF + mmproj F16 | pass: greedy identical to llama.cpp on both backends (both prompts, 4 device pairs); tokenizer identical; the GGUF's embedded template differs from the repository's (thinking off by default), compared with transformers rendering the GGUF's template | none |
| NousResearch/Arete-27B (bartowski GGUF Q4_K_M 17.2 GB) | 2026-11-06 | 10.7 k dl30, trend 95 | GGUF + mmproj | pass: greedy identical to llama.cpp on all four device pairs; tokenizer and 4 templates identical to transformers (22 checks) | none |
| google/gemma-4-31B-it (unsloth GGUF Q4_K_M 18.3 GB) | 2026-03-11 | 6.3 M | GGUF + mmproj | text pass: tokenizer and 4 templates identical to transformers; greedy identical to llama.cpp up to the end-of-turn token (llama.cpp's `--ignore-eos` masks `<turn|>`, ours emits it; the second prompt identical). Image input: the gemma4v tower (20.1.3) matches transformers' `Gemma4VisionModel` + embedder on 266 tokens (worst row cosine 1.000000, worst relative error 1.1e-4, CPU; `Gemma4TowerTests`); end to end (Vulkan, greedy, the synthetic test image) our thinking trace and llama-mtmd-cli's (b11177, `--jinja`) list the same three shapes, colors and the text, and differ in wording from the first line ("Light grey/off-white" vs "Off-white/beige"), i.e. a near-tie from the preprocessing resize, not a tower fault. Needs a free 19 GB on the GPU (a run beside other agents failed with a memory error before the shared GPU was free) | none for the 31B; the 26B-A4B, 12B and E4B mmproj files were not run |
| Nanbeige/Nanbeige4.3-33B (bartowski GGUF Q4_K_M 20.8 GB) | 2026-10-24 | 23.8 k | GGUF | partial: first prompt identical to llama.cpp; the code prompt differs at token 13, and llama.cpp's own backends give two different continuations there (Vulkan `if n <= 1`, CPU `a, b = 0, 1`), ours (Vulkan = CPU) a third, so a near-tie that cannot be settled by teacher forcing (33 B in fp32 does not fit); tokenizer differs from the repository's tokenizer.json only by llama.cpp's space-prefix placement (leading space, ids after BOS), as for Mistral in section 17 | none; not added to the catalog |

#### 20.1.2 Unsupported or open

| Model | Reason | What is missing |
|---|---|---|
| KRAFTON/Raon-1-nano (`raon1_nano`, remote code; GGUF `raon1nano` has no mapping, llama.cpp b11177 lacks it too) | new architecture: 18 recurrent-delta-attention (RDA) layers with a short conv and a full-rank gate lower bound, 6 MLA layers with output gate, 4-stream manifold hyper-connections (Helmert basis), per-layer input embeddings (dim 256), tied head, MTP head | RDA mixer, the mHC variant (`StreamHyperConnections` has the pieces but not this mixing), per-layer inputs (exist for Gemma 3n) |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B (`dory`, no transformers class; only community GGUF) | new architecture: looped transformer (16 input, 40 recurrent layers run twice with per-loop KV caches, 16 output layers), nGPT-style normalized weights (`ngpt_alpha_init`), attention output gate, sliding and full attention layers, MLP-only layers | `HierarchicalCycles` (hrm_text) covers looping with per-pass caches; the nGPT scaling and the layer pattern are new |
| maxwell-model/Maxwell-V1-Flash (`maxwell_moe`, 36.8 GB bf16, 256 experts, remote code; one GGUF with 0 downloads, community) | new architecture and above one GPU in bf16 | mapping from `modeling_maxwell_moe.py`; not analysed further |
| flint (RecursiveSI, `flint`, unsloth UD-Q4_K_XL 25 GB downloaded) | not run in this increment | |
| Ornith-1.5-35B-A3B, granite-4.1-8b-llmon-preview, TwIL-8B, Whittle-Next-27B-A3B | only community GGUF conversions exist (policy: not tested) | |

#### 20.1.3 Fixes

Gemma 4 vision tower (mmproj projector type `gemma4v`, which the importer now also reads from `clip.vision.projector_type`): axial position
tables, split-halves 2-D rope, q/k/v norms, sandwich norms, average pooling with standardization, a fit-budget resize and the image framing
tokens, on the shared tower graph. Verified for gemma-4-31B-it: tower and embedder vs transformers (float32, CPU) at 1.1e-4 relative error,
and end to end against llama-mtmd-cli as described in 20.1.1. Not verified: the other Gemma 4 mmproj files (26B-A4B, 12B, E4B; the same
projector type) and the Hugging Face `gemma4_vision` import (the tower loads from the mmproj only).

The other findings above are open items.

### 20.2 Hot-model sweep 2026-11-17: embeddings and rerankers

Checked with `Sharplight.Embeddings.Tests` `ReferenceTests` (8 texts incl. non-Latin text and a long one, token ids, optional prompt; references
written by `Fixtures/make_reference.py` with sentence-transformers, and for the reranker the model's own transformers code). Float weights
from the Hugging Face checkpoints; every tokenization matches the reference.

| Model | Architecture | Device | Result |
|---|---|---|---|
| ibm-granite/granite-embedding-311m-multilingual-r2 | modernbert | CPU, Vulkan | tokens identical, min cosine 1.000000 |
| microsoft/harrier-oss-v1-0.6b | qwen3 (last-token pooling) | CPU, Vulkan | tokens identical, min cosine 1.000000, also with the `web_search_query` prompt |
| Qwen/Qwen3-Embedding-4B | qwen3 | Vulkan | tokens identical, min cosine 1.000000, also with the query prompt |
| Qwen/Qwen3-Reranker-4B | qwen3 (yes / no logits) | Vulkan | tokens identical, max logit error 4.3e-3 (logits 1.58 / -8.05 / -1.48 / -9.58 / 2.65 / -5.59) |

The 4B models were run on Vulkan only (CPU float32 would take minutes per test). All four are in the verified catalog (`kind` embeddings).

## 21. Hot-model sweep 2026-11-18: image generation

Same method and selection rules as section 19 (Hub API `GET /api/models?pipeline_tag=<tag>&sort=trendingScore|downloads&limit=50`, queried on
2026-11-18; official checkpoints and trusted conversions only; one entry per family and size; restrictive licenses marked), for the tags
`text-to-image` and `image-to-image` (which carries the editing models), including their GGUF variants (city96, unsloth, QuantStack, the
developer's own). The Hub lists below were taken from the union of both tags and both sort orders (160 repositories); fine-tune and LoRA
re-uploads, "uncensored" / "NSFW" variants, pruned or re-quantized community packages (PrunaAI, chatrlabs, AtomicChat, Hearthscale, ...) were
skipped as community only. dl30 is the Hub's 30-day download count. Families verified in sections 15 and 18 keep their earlier result;
the models that exist in a family but not yet as a checkpoint are run now ("verified now": output compared with diffusers from the same latents,
PSNR as in section 15). Downloads and references went to local disk and were deleted afterwards.

### 21.1 Text to image

| # | Model | Publisher / license | Released | dl30 / likes | Format | Status |
|---|---|---|---|---|---|---|
| 1 | MiniMaxAI/MiniMax-H3-Image (Comfy-Org repack 2.7M; unsloth / Abiray / Hearthscale GGUF, community conversions) | MiniMax; MiniMax H3 Image community license (other) | 2026-11-03 | 727k / 1.5k | diffusers-style folder (`minimax-h3`), ComfyUI single file, GGUF | **needs arch**: `MiniMaxH3DiTModel` (50 blocks, width 5376, 56 heads of 128, token refiner, 24 latent channels, adaLN-modulated), Qwen3-VL text encoder over 5120-wide features, its own VAE; also does editing (`tasks: t2i, edit`). Section 15 already lists it as "missing" |
| 2 | Qwen/Qwen-Image-2.1 (+ Turbo: unsloth GGUF) | Alibaba (Qwen); apache-2.0 | 2026-09-14 | 287k / 3.7k (Turbo GGUF 1.7M) | diffusers, unsloth GGUF | verified earlier (section 15, catalog `qwen-image-2.1`, `qwen-image-2.1-turbo`) |
| 3 | black-forest-labs/FLUX.1-dev, FLUX.1-schnell | Black Forest Labs; FLUX.1 non-commercial (dev, gated), apache-2.0 (schnell) | 2024-07-31 | 460k / 16k, 286k / 6.8k | diffusers, city96 GGUF | verified earlier (section 15) |
| 4 | krea/Krea-2-Turbo, krea/Krea-2-Raw | Krea; Krea 2 Community License (other, gated) | 2026-06-18 | 85k / 2.0k, 107k / 1.4k | diffusers, ComfyUI files, GGUF | Turbo verified earlier (section 15); Raw uses the same graph with classifier-free guidance (family keys and a tiny model are tested), the 26 GB Raw checkpoint was not run |
| 5 | OpenMOSS-Team/MOSS-Image-SFT, MOSS-Image-Base (lodestones/Kromoss is a fine-tune of it; unsloth GGUF is a trusted conversion) | OpenMOSS; apache-2.0 | 2026-10-11 | 22k (GGUF 237k) / 973 | diffusers-style folder, unsloth GGUF | **needs arch**: `MOSSImageTransformer` (custom pipeline code, `MOSSImageScheduler`), Qwen3.5 text encoder (`Qwen3_5Model`, a hybrid linear-attention decoder), FLUX.2 VAE (supported) |
| 6 | Tongyi-MAI/Z-Image-Turbo | Alibaba (Tongyi); apache-2.0 | 2025-11-25 | 800k / 5.6k | diffusers, unsloth / leejet GGUF | verified earlier (section 15) |
| 7 | Tongyi-MAI/Z-Image (undistilled; unsloth GGUF 251k) | Alibaba (Tongyi); apache-2.0 | 2026-01-27 | n/a (GGUF 251k) | diffusers, unsloth GGUF | **verified now** (21.3): unsloth Q8_0 GGUF, 512, 28 steps, 31.5 dB; 8 steps 28.7 dB; catalog `z-image` |
| 8 | Qwen/Qwen-Image, Qwen-Image-2512 | Alibaba (Qwen); apache-2.0 | 2025-08-02 | 368k / 2.7k, 472k (GGUF) | diffusers, city96 / unsloth GGUF | verified earlier (section 15) |
| 9 | black-forest-labs/FLUX.2-dev, FLUX.2-klein-4B | Black Forest Labs; FLUX Dev non-commercial (gated), apache-2.0 | 2025-11-22 / 2026-01-14 | 287k / 2.6k, 436k / 1.1k | diffusers, unsloth / city96 GGUF | verified earlier (section 15) |
| 10 | black-forest-labs/FLUX.2-klein-9B (unsloth: ungated mirror and GGUF 492k) | Black Forest Labs; FLUX non-commercial | 2026-01-14 | 112k / 1.9k | diffusers, unsloth GGUF | **verified now**: Q8_0 GGUF, 512, 4 steps, 29.8 dB, visually identical (21.3) |
| 11 | black-forest-labs/FLUX.2-klein-base-4B (unsloth GGUF) | Black Forest Labs; apache-2.0 | 2026-01-14 | 723k / 198 | diffusers, unsloth GGUF | **verified now**: Q8_0 GGUF, 512, 28 steps with guidance 4, 36.6 dB (21.3) |
| 12 | black-forest-labs/FLUX.2-klein-base-9B (unsloth GGUF) | Black Forest Labs; FLUX non-commercial | 2026-01-14 | 66k / 454 | unsloth GGUF | not run: the 9B transformer of 21.1 #10 with guidance; the same code path as #10 and #11 |
| 13 | black-forest-labs/flux-3-image-klein (+ `-base`, unsloth GGUF 437k, Comfy repack 1.1M) | Black Forest Labs; other, gated | 2026-10-13 | 161k / 1.0k (trending) | diffusers (gated), single file, unsloth GGUF | **needs arch**: `Flux3Transformer2DModel` (5 dual-stream + 28 single-stream blocks, 24 heads of 128, `flow_map: true` (a flow-map sampler: the transformer is conditioned on the target time as well), 128-channel conditioning `cond_in_channels`, 20480-wide joint features = 8 tapped layers of a Qwen3-VL-4B text encoder), FLUX.2 VAE. The Comfy / unsloth files are not gated, the official folder is |
| 14 | stabilityai/stable-diffusion-xl-base-1.0 and fine-tunes (RealVisXL, Juggernaut, animagine-4.0, Pony, noobai, WAI-Illustrious, Illustrious) | Stability AI and the fine-tune authors; OpenRAIL++ | 2023-07-25 | 4.8M / 8.5k | diffusers, single file, stable-diffusion.cpp GGUF | verified earlier (section 15; animagine-xl-3.1 and DreamShaper 7 are the same SDXL / SD 1.x graphs as the verified 4.0 and 8) |
| 15 | stabilityai/sdxl-turbo, SDXL-Lightning | Stability AI, ByteDance; OpenRAIL / non-commercial research | 2023-11-27 | 587k / 2.7k | diffusers, GGUF | verified earlier (section 15) |
| 16 | stabilityai/sd-turbo | Stability AI; Stability community / non-commercial (other) | 2023-11-27 | 384k / 471 | diffusers, single file | **verified now** (catalog level was "loads"): fp16 folder, 512, 1 step, no guidance, 54.0 dB; the folder defaults were wrong (21.2) |
| 17 | stabilityai/stable-diffusion-3.5-medium | Stability AI; Stability community | 2024-10-29 | 79k / 1.3k | diffusers, city96 GGUF | verified earlier (section 15) |
| 18 | stable-diffusion-v1-5/stable-diffusion-v1-5, CompVis/stable-diffusion-v1-4 | Runway / CompVis; CreativeML OpenRAIL-M | 2022 / 2024 | 1.2M / 1.4k, 438k / 7.1k | diffusers, single file | verified earlier (v1-5; v1-4 is the same graph) |
| 19 | Comfy-Org/Fibo-2.0, briaai/Fibo-2.0-edit-turbo | Bria; CC-BY-NC-4.0 / other, gated | 2026-10-19 | 457k / 238, 3.7k / 186 | ComfyUI single files, diffusers (gated) | needs arch (not inspected beyond the file lists: a Bria DiT conditioned on a structured-prompt language model; non-commercial) |
| 20 | PrunaAI P-H3-Image / P-Q-Image-2.1-*, chatrlabs Qwen-Image-2.1 Omni, AtomicChat / abenzerps / KasugaiSakura Qwen-Image-2.1 GGUF, Viggle Qwen-Image-2.1 turbo, lightx2v Lightning LoRAs, anima / HaDeS / MossYume | community | | | | community only (re-quantizations, fine-tunes and LoRAs of the models above; the Lightning LoRAs would take the LoRA path of section 15) |

### 21.2 Image to image and editing

| # | Model | Publisher / license | Released | dl30 / likes | Format | Status |
|---|---|---|---|---|---|---|
| 1 | black-forest-labs/FLUX.1-Kontext-dev (QuantStack GGUF) | Black Forest Labs; FLUX.1 non-commercial | 2025-05-28 | 186k / 3.0k (GGUF 99k) | diffusers, QuantStack GGUF | verified earlier (section 15, editing) |
| 2 | FLUX.2 Klein 4B / 9B, FLUX.2 Dev (reference-image editing) | Black Forest Labs | 2025-11 / 2026-01 | see 21.1 | | editing verified earlier on the tiny pipeline and Kontext numbers (section 15); the 9B model shares the code |
| 3 | Qwen/Qwen-Image-Edit-2511, -2509, Qwen-Image-Edit (unsloth GGUF 1.0M, QuantStack GGUF 192k, lightx2v Lightning 881k) | Alibaba (Qwen); apache-2.0 | 2025-08 to 12 | 237k / 1.6k, 595k / 1.3k, 148k / 2.6k | diffusers, GGUF | **supported since the editing round of 2026-11-18** (branch `imagegen/edit-r1`; section 15 "Qwen-Image-Edit"): `QwenImageEditPlusPipeline` runs the Qwen-Image transformer, the Qwen2.5-VL text encoder reads the reference images through its vision tower (M-RoPE image rows in the text graph) and the reference latents follow the image tokens (-2511's zero-timestep modulation); -2511 Q4_K_M verified against diffusers (40.3 dB with one picture, 26.4 dB with two, which is the size of diffusers' own sensitivity to the resampling of the pictures the language model reads), -2509 through the tiny pipeline; the first Qwen-Image-Edit shares the code (1024 · 1024 reading size, one picture) and is not verified numerically |
| 4 | FireRedTeam/FireRed-Image-Edit-1.0 | FireRed; apache-2.0 | 2026-02-12 | 45k / 358 | diffusers (`QwenImageEditPlusPipeline`) | supported with #3 (same pipeline class as -2509; its checkpoint was not run) |
| 5 | jdopensource/JoyAI-Image-Edit-Diffusers | JD; apache-2.0 | 2026-04-10 | 105k / 23 | diffusers | needs arch: `JoyImageEditTransformer3DModel` over a Wan VAE with a Qwen3-VL text encoder |
| 6 | timbrooks/instruct-pix2pix | Tim Brooks; MIT | 2023-01-20 | 80k / 1.2k | diffusers | **supported since the editing round of 2026-11-18** (section 15 "SD editing"): 8-channel UNet, text and image guidance on three rows; 59.7 dB against diffusers (Euler, 20 steps, f32, same noise) on Vulkan |
| 7 | stable-diffusion-v1-5/stable-diffusion-inpainting, stabilityai/stable-diffusion-xl-refiner-1.0 | Runway, Stability AI | 2022 / 2023 | 271k / 122, 113k / 2.1k | diffusers | the inpainting UNet is **supported since the editing round of 2026-11-18** (section 15 "SD editing": 56.3 dB against diffusers; the SDXL inpainting UNet diffusers/stable-diffusion-xl-1.0-inpainting-0.1 40.5 dB); the SDXL refiner is **supported since 2026-11-18** (section 15 "SDXL refiner": base + refiner 48.3 dB against diffusers) |
| 8 | MiniMax-H3-Image (editing mode), flux-3-image-klein (editing), Fibo-2.0-edit-turbo | | | | | needs the arch of 21.1 #1, #13, #19 |
| 9 | LoRA and adapter repositories (Qwen-Image-Edit multiple-angles, AnyPose, Anything2Real, BFS face swap, MOSS control LoRAs, krea2 anypaint / pose controlnet) | community | | | | community only |

### 21.3 Findings and fixes

- **SD-Turbo ran with guidance 7.5.** The diffusers folder carries the distillation only in its scheduler (Euler on trailing timesteps); the
  importer gave it the generic defaults, so an image without an explicit guidance came out oversaturated (10 dB from diffusers). Folders whose
  name (or `_name_or_path`) says "turbo" with such a scheduler now default to 4 steps without guidance
  (`SchedulerSettings.WithDistilledDefaults`; SDXL-Turbo single files already had it by file name). Tests: `SchedulerDistilledDefaultsTests`.
- **The undistilled Z-Image ran as the Turbo** (9 steps, no guidance, shift 3) and its prediction was NaN on Vulkan. Three fixes (`ZImageConfig`,
  `ZImageHyperParameters`): the base model is recognized by its scheduler shift (6, folders) or its file name (`z-image`, `z_image`
  without "turbo", transformer-only files) and gets classifier-free guidance, 50 steps and shift 6; its default guidance is 5 because
  diffusers' Z-Image pipeline combines `pos + g * (pos - neg)` (its default `guidance_scale` 4), the usual scale `1 + g` (as for Krea 2; with 4
  the 8-step comparison was 15.6 dB, with 5 it is 28.7 dB); and its transformer keeps f32 activations in every matrix product and f32 attention
  products (`zimage.transformer.f32_activations`): with either of the two f16 shortcuts of the Turbo path alone the prediction at the noisy
  timesteps (sigma >= 0.75) overflowed to NaN on the real prompt features (random features passed, which is why the graph test did not see it).
  The f32 paths are slower than the Turbo's (not measured). Tests: `ZImageBaseConfigTests`; the whole-image comparisons of 21.4.
- A harness for any further checkpoint: `tools/make-hot-imagegen-references.py` (diffusers reference with fixed latents for FLUX.2 Klein,
  Z-Image and SD folders) and `HotSweepTests` (manifest per model under `~/models/hot-imagegen`, `SHARPLIGHT_HOT_ROOT`,
  `SHARPLIGHT_HOT_FILTER`, `SHARPLIGHT_HOT_DEVICE`).

### 21.4 Results of this round (2026-11-18)

All on Vulkan (GPU 0, one RTX PRO 4500), 512x512, the same initial latents as the diffusers reference (f32, the GGUF transformer dequantized
on the fly where there is one), PSNR of the 8-bit images:

| Model | Files | Steps / guidance | PSNR |
|---|---|---|---|
| SD-Turbo | stabilityai/sd-turbo fp16 folder | 1 / none | 54.0 dB |
| FLUX.2 Klein base 4B | unsloth Q8_0 GGUF, text encoder and VAE of the Klein 4B folder | 28 / 4 | 36.6 dB |
| FLUX.2 Klein 9B | unsloth Q8_0 GGUF, the unsloth mirror's Qwen3-8B text encoder and VAE | 4 / none | 29.8 dB (visually identical; the 4B distilled model measured 40 dB, the 9B is 2x the transformer to accumulate the Q8_0 / bf16 differences over) |
| Z-Image (undistilled) | unsloth Q8_0 GGUF, the pipeline's text encoder and VAE | 28 and 8 / 4 (scale 5 here) with negative prompt | 31.5 dB and 28.7 dB (before the fixes the output was a gray NaN image, 8.5 dB) |

Needs a new architecture (none implemented in this branch): MiniMax-H3-Image, MOSS-Image (and Kromoss), FLUX.3 Klein (`Flux3Transformer2DModel`),
Fibo 2.0, JoyAI-Image-Edit. The pipeline work on supported transformers (Qwen-Image-Edit / -2509 / -2511, FireRed-Image-Edit, InstructPix2Pix, the
inpainting UNets) was done in the editing round of the same day (section 15 "SD editing" and "Qwen-Image-Edit"); the SDXL refiner followed later that day (section 15 "SDXL refiner").

### 21.5 Editing round (2026-11-18, branch `imagegen/edit-r1`)

Vulkan, GPU 0, the same initial latents and prompts as the diffusers reference (f32; the GGUF transformer dequantized on the fly), PSNR of the 8-bit images
(details in section 15):

| Model | Files | Steps / guidance | Result |
|---|---|---|---|
| InstructPix2Pix | timbrooks/instruct-pix2pix fp32 diffusers folder | 20 / 7.5 text, 1.5 image, Euler | 59.7 dB |
| SD 1.5 inpainting | stable-diffusion-v1-5/stable-diffusion-inpainting fp16 diffusers folder | 20 / 7.5, DDIM | 56.3 dB |
| SDXL inpainting | diffusers/stable-diffusion-xl-1.0-inpainting-0.1 fp16 diffusers folder | 20 / 7.5, Euler, 512×512 | 40.5 dB |
| Qwen-Image-Edit-2511, one picture | unsloth Q4_K_M GGUF, the pipeline's text encoder and VAE, 640×480 picture, 512×384 | 20 / true CFG 4 | 40.3 dB (diffusers against itself with PIL's bicubic for the language model's pictures: 45.6 dB) |
| Qwen-Image-Edit-2511, two pictures | as above | 20 / true CFG 4 | 26.4 dB (24.3 dB) |
| tiny Qwen-Image-Edit-2509 / -2511 (random) | diffusers folder and Q8_0 GGUF | 4 / true CFG 4 | 56 to 59 dB, CPU and Vulkan |

Not done in this round: the SDXL refiner (done afterwards in branch `fix/vk-unet-batch3`, section 15 "SDXL refiner"). Downloads and references
were deleted afterwards except the diffusers references and the tiny pipelines.
