# Compatible models

Last updated: 2026-11-16.

This is the list of models that have been tried with Sharplight and work. It only contains what was actually tested; a model that is not
listed has not been tried yet, which does not mean it fails. Technical detail (architecture coverage, quantization formats, test
methodology, numbers) is in [model-support.md](model-support.md), ONNX details in [onnx-support.md](onnx-support.md).

## What "compatible" means

A model is listed when it **imports** (the file format is read, the architecture is recognized, tokenizer and chat template or feature
extractor come along) and **runs** on the CPU provider, the Vulkan provider, or both. The Providers column says which were tested. Each row
also says how the output was checked:

| Level | Meaning |
|---|---|
| **Verified** | Output compared with a reference implementation on real weights: greedy tokens identical to transformers or llama.cpp, text identical to whisper.cpp, images or audio within a stated PSNR or relative error of diffusers, stable-diffusion.cpp, the model's own package, or onnxruntime. |
| **Verified (tiny)** | The architecture is verified against its reference on a small random checkpoint of the same structure; no real checkpoint was compared. |
| **Loads** | The real checkpoint passes the compatibility check and the import (weights mapped, tokenizer built). Generation was not compared with a reference. |
| **Partial** | Works with a stated limitation. |

Formats: safetensors and GGUF (every ggml quantization type, including the IQ, TQ, MXFP4, NVFP4 and Bonsai types), Hugging Face
quantization (FP8, NVFP4, MXFP4, GPTQ, AWQ and others, see model-support.md), whisper.cpp `.bin`, diffusers folders, single-file
checkpoints, stable-diffusion.cpp GGUF, ONNX, and Sharplight's own `.slm` package (every model above can be stored in one).

Differences from a reference that are not errors: quantized files round differently between runtimes (int8 activation dot products,
f16 accumulation), so greedy text from a 2-3 bit file can diverge from llama.cpp after some tokens even when the logits agree (correlation
above 0.98 in every such case checked).

**Requesting a model:** open an issue on the GitHub repository with the Hub repository id (and the file or quantization you use). Models that
fail to import are reported with a reason; those reasons are collected in the last section.

---

## 1. Text generation

### 1.1 Language models, verified against a reference (real checkpoints)

Greedy generation compared token by token. "llama.cpp" = llama.cpp, "transformers" = Hugging Face transformers (fp32 compute).

| Model | Hub repository | Format | Providers | Verification | Notes | Verified |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | Qwen/Qwen3-0.6B, Qwen/Qwen3-0.6B-GGUF, unsloth/Qwen3-0.6B-GGUF, bartowski/Qwen_Qwen3-0.6B-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | HF and GGUF import to identical weights; greedy ids identical | bartowski IQ2_M: Vulkan identical to llama.cpp, CPU differs late (precision) | 2026-11-15 |
| Qwen2.5 0.5B Instruct | Qwen/Qwen2.5-0.5B-Instruct | safetensors, GGUF Q8_0, f16 | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| Qwen3-VL 2B Instruct | Qwen/Qwen3-VL-2B-Instruct | safetensors, GGUF Q4_K_M | CPU, Vulkan | greedy ids identical to transformers; text identical to llama.cpp | image input via the mmproj file, see section 1.3 | 2026-11-14 |
| Qwen3-VL 4B Instruct | unsloth/Qwen3-VL-4B-Instruct-GGUF | GGUF UD-IQ2_XXS | CPU, Vulkan | greedy identical to llama.cpp | | 2026-11-15 |
| Qwen3.5 0.8B | Qwen/Qwen3.5-0.8B | safetensors, GGUF BF16 | CPU, Vulkan | HF vs GGUF parity; mmproj vs llama.cpp | hybrid (gated delta net) architecture | 2026-11-14 |
| Qwen3.6 35B-A3B (MoE) | Qwen/Qwen3.6-35B-A3B, unsloth/Qwen3.6-35B-A3B-GGUF | GGUF UD-Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Qwen3-Coder-Next | unsloth/Qwen3-Coder-Next-GGUF | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | 48 GB file | 2026-11-14 |
| Qwen3.8-Flash-Next | unsloth/Qwen3.8-Flash-Next-GGUF | GGUF UD-Q4_K_XL | CPU, Vulkan | text identical to llama.cpp | 111 GB file | 2026-11-14 |
| Qwen3-4B | unsloth/Qwen3-4B-Instruct-2507-GGUF | GGUF UD-IQ1_S | CPU | identical to llama.cpp on CPU; Vulkan differs (precision) | UD-Q2_K_XL differs on both late in the text | 2026-11-15 |
| Qwen3-30B-A3B | unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF | GGUF UD-TQ1_0 | CPU | identical to llama.cpp on CPU; Vulkan differs (precision) | | 2026-11-15 |
| Llama 3.2 1B Instruct | meta-llama/Llama-3.2-1B-Instruct, bartowski/Llama-3.2-1B-Instruct-GGUF | safetensors, GGUF Q4_0, Q4_0_4_4, Q4_0_8_8 | CPU, Vulkan | Q4_0 identical to llama.cpp on both; interleaved types identical on CPU | | 2026-11-15 |
| Llama 3.2 3B Instruct | meta-llama/Llama-3.2-3B-Instruct, unsloth/Llama-3.2-3B-Instruct-GGUF | safetensors, GGUF | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| Mistral 7B Instruct v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Ministral 3 3B | unsloth/Ministral-3-3B-Reasoning-2512-GGUF | GGUF UD-IQ3_XXS | CPU, Vulkan | greedy identical to llama.cpp | also verified with its Pixtral mmproj | 2026-11-15 |
| Gemma 4 E2B-it | google/gemma-4-E2B-it | safetensors, GGUF BF16, Q8_0 | CPU, Vulkan | ids identical to transformers; text identical to llama.cpp | | 2026-11-14 |
| Gemma 4 E4B-it (QAT q4_0) | google/gemma-4-E4B-it-qat-q4_0-gguf | GGUF q4_0 | CPU, Vulkan | greedy identical to llama.cpp | | 2026-11-15 |
| Gemma 4 26B-A4B-it (MoE) | google/gemma-4-26B-A4B-it, unsloth/gemma-4-26B-A4B-it-GGUF | GGUF UD-Q4_K_XL | CPU, Vulkan | chat prompt identical to llama.cpp on both | very short raw prompts: Vulkan differs (int8 activations) | 2026-11-15 |
| Gemma 4 12B-it | google/gemma-4-12B-it | safetensors, GGUF | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| Gemma 3 1B / 4B it | google/gemma-3-1b-it, ggml-org/gemma-3-4b-it-GGUF, unsloth/gemma-3-4b-it-GGUF | safetensors, GGUF Q4_K_M, UD-IQ2_M | CPU, Vulkan | HF vs GGUF parity; 4B Q4_K_M text identical to llama.cpp | 4B UD-IQ2_M: Vulkan identical, CPU differs late | 2026-11-15 |
| Gemma 2 2B it | google/gemma-2-2b-it, bartowski/gemma-2-2b-it-GGUF | safetensors, GGUF | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| Gemma 3n E2B it | google/gemma-3n-E2B-it | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Phi-4 mini instruct | microsoft/Phi-4-mini-instruct, bartowski/microsoft_Phi-4-mini-instruct-GGUF | GGUF Q4_K_M, IQ4_NL | CPU, Vulkan | text / greedy identical to llama.cpp | | 2026-11-15 |
| Phi-3.5 mini instruct | microsoft/Phi-3.5-mini-instruct | safetensors, GGUF | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| Phi-2 | microsoft/phi-2 | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| SmolLM2 360M Instruct | HuggingFaceTB/SmolLM2-360M-Instruct | safetensors, GGUF | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| SmolLM3 3B | HuggingFaceTB/SmolLM3-3B, bartowski/HuggingFaceTB_SmolLM3-3B-GGUF | GGUF Q4_K_M, IQ3_XS | CPU, Vulkan | Q4_K_M identical to llama.cpp | IQ3_XS: Vulkan identical, CPU differs late | 2026-11-15 |
| OLMoE 1B-7B Instruct (MoE) | allenai/OLMoE-1B-7B-0924-Instruct | safetensors, GGUF | CPU, Vulkan | HF vs GGUF parity | | 2026-11-14 |
| OLMo 2 1B Instruct | allenai/OLMo-2-0425-1B-Instruct | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| gpt-oss 20B (MoE) | openai/gpt-oss-20b, ggml-org/gpt-oss-20b-GGUF | safetensors, GGUF MXFP4 | CPU, Vulkan | text identical to llama.cpp (GGUF) | | 2026-11-14 |
| GLM-4 9B 0414 | zai-org/GLM-4-9B-0414 | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| GLM-4.7-Flash (MLA, MoE) | zai-org/GLM-4.7-Flash, unsloth/GLM-4.7-Flash-GGUF | GGUF UD-TQ1_0 | CPU, Vulkan | greedy identical to llama.cpp | | 2026-11-15 |
| DeepSeek-V2-Lite Chat (MLA, MoE) | deepseek-ai/DeepSeek-V2-Lite-Chat | GGUF Q4_K_M | CPU | text identical to llama.cpp | | 2026-11-14 |
| DeepSeek-Coder-V2-Lite Instruct | bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF | GGUF IQ2_XS | CPU, Vulkan | greedy identical to llama.cpp | | 2026-11-15 |
| MiniCPM3 4B | openbmb/MiniCPM3-4B | GGUF Q4_K_M | CPU | text identical to llama.cpp | | 2026-11-14 |
| MiniCPM4.1 8B | openbmb/MiniCPM4.1-8B | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| MiMo 9B (distill) | bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF | GGUF IQ2_M | CPU | identical to llama.cpp on CPU; Vulkan differs (precision) | | 2026-11-15 |
| LFM2.5 1.2B Instruct | LiquidAI/LFM2.5-1.2B-Instruct, bartowski/LiquidAI_LFM2.5-1.2B-Instruct-GGUF | safetensors, GGUF Q4_K_M, IQ3_XXS | CPU, Vulkan | ids identical to transformers; text / greedy identical to llama.cpp | hybrid short-convolution architecture | 2026-11-15 |
| LFM2.5 8B-A1B (MoE) | LiquidAI/LFM2.5-8B-A1B, unsloth/LFM2.5-8B-A1B-GGUF | GGUF Q4_K_M, MXFP4_MOE | CPU, Vulkan | text identical to llama.cpp (Vulkan for MXFP4_MOE) | UD-IQ2_XXS differs late | 2026-11-15 |
| LFM3 1.7B | LiquidAI/LFM3-1.7B, LiquidAI/LFM3-1.7B-GGUF | safetensors (bf16), GGUF incl. NVFP4 | CPU, Vulkan | 16 greedy ids identical to transformers | QAD-Q4_0 differs late | 2026-11-15 |
| Spark-X2.6 4B | XHToken/Spark-X2.6-4B | safetensors (bf16) | CPU, Vulkan | 16 greedy ids identical to transformers | | 2026-11-15 |
| Mage-VL-Cowork | microsoft/Mage-VL-Cowork | safetensors (bf16) | CPU, Vulkan | 16 greedy ids identical to transformers | text decoder only; the OneVision image tower is not imported | 2026-11-15 |
| NVIDIA Nemotron 3 Nano 4B | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | safetensors, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; text identical to llama.cpp | hybrid Mamba-2 / attention | 2026-11-14 |
| NVIDIA Nemotron 3.5 Bolt E4B | NVIDIA-Nemotron-3.5-Bolt-E4B (GGUF) | GGUF UD-Q4_K_XL | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Muse-Glimmer 30B | meta-models/Muse-Glimmer-30B, unsloth/Muse-Glimmer-30B-GGUF | safetensors, GGUF UD-Q4_K_XL | CPU, Vulkan | HF vs GGUF parity; text identical to llama.cpp; DFlash drafter output identical | speculative decoding with the DFlash drafter | 2026-11-15 |
| Falcon-H1 0.5B Instruct | tiiuae/Falcon-H1-0.5B-Instruct | safetensors, GGUF Q4_K_M | CPU, Vulkan | ids identical to transformers; text identical to llama.cpp | | 2026-11-14 |
| Granite 4.0 H 350M | ibm-granite/granite-4.0-h-350m | safetensors, GGUF Q4_K_M | CPU, Vulkan | ids identical to transformers; text identical to llama.cpp | | 2026-11-14 |
| Jamba Reasoning 3B | ai21labs/AI21-Jamba-Reasoning-3B | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Mamba 130M / Mamba-2 130M | state-spaces/mamba-130m-hf, AntonV/mamba2-130m-hf | safetensors | CPU, Vulkan | ids identical to transformers | | 2026-11-14 |
| RWKV7 G1 1.5B | rwkv7-g1h-1.5b (GGUF) | GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| GPT-2 | openai-community/gpt2 | safetensors, GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Pythia 160M | EleutherAI/pythia-160m | safetensors, GGUF Q4_K_M | CPU, Vulkan | text identical to llama.cpp | | 2026-11-14 |
| Bonsai 1.7B, Ternary-Bonsai 1.7B / 2-2B | prism-ml (Bonsai GGUF) | GGUF Q1_0, Q2_0, PQ2_0, PTQ1_0 | CPU, Vulkan | HF vs GGUF parity | 1-2 bit formats | 2026-11-14 |

More text models were verified against llama.cpp on GGUF files and are summarized by family in model-support.md section 8.1 (all Q4_K_M or
Q8_0, text identical to llama.cpp on CPU and Vulkan): Granite 3.x dense and MoE (granite-3.1-1b-a400m, granite-3.3-2b), Command-R7B and Aya Expanse 8B,
StarCoder2 3B, InternLM2.5 1.8B chat, ERNIE-4.5 0.3B and 21B-A3B, EXAONE 3.5 2.4B and 4.0 1.2B, StableLM 2 1.6B, BLOOM 560M, FLAN-T5 small,
Nemotron-Mini 4B, Apertus 8B, AFM-4.5B, SmallThinker 4B-A0.6B, OpenELM 270M, Nanbeige 4.2 3B, BitNet b1.58 2B (TQ2_0), Hunyuan 0.5B, LFM2-VL 450M, Falcon-H1,
Mamba and RWKV6 World 1.6B, Qwen3-VL 2B. Two of them generate degenerate text in llama.cpp itself (Hunyuan 0.5B, EXAONE 4.0 1.2B); Sharplight matches that output.

### 1.1.1 Hub top-20 sweep (2026-11-16)

The most downloaded text-generation models on the Hub plus popular larger ones, each checked as safetensors against transformers and as
the most downloaded GGUF against llama.cpp b11177: tokenizer on seven multilingual texts, chat template with and without system prompt and
with tool calls, 48 greedy tokens on three prompts (method and per-model notes in model-support.md section 17). "near-tie" = the first
differing token is a near tie of the reference itself (llama.cpp's own CPU and Vulkan builds differ there too, or the logit gap is below 0.1).

| Model | Hub repository | Format | Providers | Verification | Notes | Verified |
|---|---|---|---|---|---|---|
| GPT-2 | openai-community/gpt2, mradermacher/gpt2-GGUF | safetensors F32, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; GGUF identical to llama.cpp per backend | | 2026-11-16 |
| Qwen3 0.6B | Qwen/Qwen3-0.6B, Qwen/Qwen3-0.6B-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | ids and templates (tools) identical to transformers; GGUF near-tie vs llama.cpp | also as .slm | 2026-11-16 |
| Qwen3 1.7B / 4B / 8B | Qwen/Qwen3-1.7B, -4B, -8B; unsloth/Qwen3-1.7B-GGUF, Qwen/Qwen3-4B-GGUF, Qwen/Qwen3-8B-GGUF | safetensors, GGUF Q8_0 / Q4_K_M | CPU, Vulkan | ids identical to transformers; 4B / 8B GGUF identical to llama.cpp, 1.7B near-tie | | 2026-11-16 |
| Qwen3 14B | Qwen/Qwen3-14B, Qwen/Qwen3-14B-GGUF | safetensors (29.5 GB), GGUF Q4_K_M | Vulkan | ids identical to transformers; GGUF identical to llama.cpp | safetensors on one whole 32 GB GPU | 2026-11-16 |
| Qwen3 32B | Qwen/Qwen3-32B-GGUF | GGUF Q4_K_M | CPU, Vulkan | tokenizer and templates identical to transformers; near-tie vs llama.cpp | | 2026-11-16 |
| Qwen2.5 0.5B / 1.5B / 3B / 7B Instruct | Qwen/Qwen2.5-*-Instruct, Qwen/Qwen2.5-*-Instruct-GGUF | safetensors, GGUF Q8_0 / Q4_K_M | CPU, Vulkan | ids identical to transformers; GGUF identical or near-tie vs llama.cpp | the official GGUFs embed an older tool-call template (literal `{{"name": ...}}`), rendered exactly as transformers renders it | 2026-11-16 |
| Ornith 1.0 9B / 1.5 9B | ornith-ai/Ornith-1.0-9B, ornith-ai/Ornith-1.5-9B-GGUF | safetensors, GGUF Q4_K_M | CPU, Vulkan | ids identical to transformers; GGUF near-tie vs llama.cpp | qwen3_5 hybrid | 2026-11-16 |
| Qwen3.8 27B | Qwen/Qwen3.8-27B-FP8, lmstudio-community/Qwen3.8-27B-GGUF | safetensors FP8, GGUF Q8_0 | Vulkan (2 GPUs) | FP8: ids identical to transformers (bf16 checkpoint) on raw prompts, teacher-forced logit difference 0.16-0.19; Q8_0 identical to llama.cpp | | 2026-11-16 |
| Qwen3.8-Flash-Next | unsloth/Qwen3.8-Flash-Next-GGUF | GGUF UD-IQ1_S (66 GB) | Vulkan (2 GPUs) + CPU | tokenizer, templates and greedy text identical to llama.cpp | its MTP draft file (`mtp-*-Q8_0.gguf`, CLI `--draft`) works as the draft head: acceptance 0.87 / 0.79 / 0.64 at k = 1 / 2 / 3 on one GPU + CPU | 2026-11-16 |
| MiMo 7B RL | XiaomiMiMo/MiMo-7B-RL | safetensors BF16 | Vulkan | MTP speculative greedy output identical to plain decoding (k = 1-3); the trunk was not compared with a reference (remote code) | built-in MTP layer as draft head: k = 1 1.38x (56.0 → 77.4 tok/s), acceptance 0.70 | 2026-11-16 |
| gpt-oss 20B / 120B | openai/gpt-oss-20b, unsloth/gpt-oss-20b-GGUF, ggml-org/gpt-oss-120b-GGUF | safetensors MXFP4 (20B), GGUF Q4_K_M, MXFP4 (120B) | CPU, Vulkan (120B: 2 GPUs + CPU) | 20B ids identical to transformers; GGUF near-tie vs llama.cpp | | 2026-11-16 |
| Llama 3.1 8B Instruct | unsloth/Llama-3.1-8B-Instruct, bartowski/Meta-Llama-3.1-8B-Instruct-GGUF | safetensors, GGUF Q4_K_M | CPU, Vulkan | ids and tool template identical to transformers; GGUF first token a 0.03 near-tie, identical to transformers on the same GGUF | | 2026-11-16 |
| Llama 3.2 1B Instruct | unsloth/Llama-3.2-1B-Instruct, bartowski/Llama-3.2-1B-Instruct-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; GGUF identical to llama.cpp | also in Studio and as .slm | 2026-11-16 |
| Llama 3.3 70B Instruct | bartowski/Llama-3.3-70B-Instruct-GGUF | GGUF Q4_K_M | Vulkan (2 GPUs) | identical to llama.cpp CUDA; llama.cpp Vulkan differs at a near-tie | | 2026-11-16 |
| Mistral 7B Instruct v0.2 | mistralai/Mistral-7B-Instruct-v0.2, TheBloke/Mistral-7B-Instruct-v0.2-GGUF | safetensors, GGUF Q4_K_M | CPU, Vulkan | tokenizer, chat prompt ids and greedy ids identical to transformers; GGUF identical to transformers on the same GGUF | GGUF prompt ids follow llama.cpp's space-prefix placement | 2026-11-16 |
| OPT 125M | facebook/opt-125m | PyTorch .bin | CPU, Vulkan | ids identical to transformers | no GGUF (llama.cpp has no OPT); post-norm OPT-350m not supported | 2026-11-16 |
| Pythia 160M | EleutherAI/pythia-160m, mradermacher/pythia-160m-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; GGUF near-tie vs llama.cpp | | 2026-11-16 |
| SmolLM2 135M | HuggingFaceTB/SmolLM2-135M, prithivMLmods/SmolLM2-135M-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; GGUF identical to llama.cpp | | 2026-11-16 |
| Nemotron 3 Nano 4B | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF | safetensors, GGUF Q4_K_M | CPU, Vulkan | ids identical to transformers; GGUF identical to llama.cpp | Mamba-2 hybrid; also as .slm | 2026-11-16 |
| MiniCPM5 2B | openbmb/MiniCPM5-2B, bartowski/MiniCPM5-2B-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; GGUF identical to llama.cpp | | 2026-11-16 |
| LFM3 1.7B | LiquidAI/LFM3-1.7B, LiquidAI/LFM3-1.7B-GGUF | safetensors, GGUF Q8_0 | CPU, Vulkan | ids identical to transformers; GGUF teacher-forced vs transformers (llama.cpp b11177 lacks lfm3) | also in Studio | 2026-11-16 |
| Qwen3 30B-A3B (MoE) | Qwen/Qwen3-30B-A3B-GGUF | GGUF Q4_K_M | CPU, Vulkan | tokenizer and templates identical to transformers; greedy identical to llama.cpp (CPU build) | | 2026-11-16 |
| Qwen3.6 35B-A3B (MoE) | unsloth/Qwen3.6-35B-A3B-GGUF | GGUF UD-Q4_K_M | Vulkan | tokenizer, templates and greedy text identical to llama.cpp | | 2026-11-16 |
| GLM-4.7-Flash (MLA, MoE) | unsloth/GLM-4.7-Flash-GGUF | GGUF Q4_K_M | CPU, Vulkan | tokenizer and templates identical to transformers; greedy near-tie vs llama.cpp | | 2026-11-16 |

### 1.2 Language models that load (header check and import, output not compared)

These real checkpoints, and every quantization file listed, were imported through the full load path (mapping, graph build, tokenizer, chat
template) during the round-10 sweep of popular Hub repositories. Weights were not downloaded; output has not been compared with a reference.
Counts are files per repository.

| Model family | Hub repositories | Formats | Notes | Checked |
|---|---|---|---|---|
| Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8 dense and MoE | Qwen/Qwen3-8B, Qwen3-14B, Qwen3-30B-A3B, Qwen3-Coder-30B-A3B-Instruct, Qwen2.5-7B-Instruct, Qwen3.5-9B, Qwen3.8-27B, Qwen3.8-27B-FP8, Qwen3.6-35B-A3B, Qwen3.8-Flash-Next (+ FP8), Qwen3.8-2.4T-A95B; unsloth and bartowski GGUF repositories of the same models (about 20 repositories, 24 to 31 files each); Qwen/Qwen3-VL-8B-Instruct-GGUF; unsloth/Qwen2.5-VL-7B-Instruct-GGUF; ggml-org Qwen2.5-VL-3B, Qwen2.5-Omni-3B, Qwen3-Omni-30B-A3B | safetensors, GGUF BF16 to IQ1_S, FP8 | Omni models: text decoder only | 2026-11-15 |
| Llama and Mistral | bartowski/Meta-Llama-3.1-8B-Instruct-GGUF, bartowski/Llama-3.2-3B-Instruct-GGUF, mistralai/Mistral-7B-Instruct-v0.2, unsloth/Mistral-Large-4-1T-A52B-GGUF | safetensors, GGUF | | 2026-11-15 |
| Gemma | unsloth/gemma-4-{E2B,E4B,12b,26B-A4B,31B}-it-GGUF (21 to 22 files each), google/gemma-4-31B-it, google/gemma-4-26B-A4B-it, nvidia/Gemma-4-26B-A4B-NVFP4, unsloth/embeddinggemma-2-GGUF | safetensors, GGUF, ModelOpt NVFP4 | | 2026-11-15 |
| DeepSeek | deepseek-ai/DeepSeek-V3.2, DeepSeek-R1, DeepSeek-V4-Flash-0731; unsloth DeepSeek-V4-Flash and V4.1-Flash GGUF; bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF | safetensors (FP8, FP4), GGUF | 284B V4-Flash was imported from headers only | 2026-11-15 |
| GLM | zai-org/GLM-5, GLM-5.3, GLM-5.3-Flash, GLM-4.5-Air; unsloth GLM-5.3-Flash GGUF; ggml-org GLM-4.6V, GLM-OCR | safetensors, GGUF | | 2026-11-15 |
| gpt-oss | openai/gpt-oss-120b, unsloth gpt-oss-20b / 120b GGUF (16 files) | safetensors, GGUF | | 2026-11-15 |
| Kimi, MiniMax, Step | moonshotai/Kimi-K3, Kimi-K3.1-Pro (MXFP4); unsloth/Kimi-K3-GGUF (9 files); MiniMaxAI/MiniMax-M2.5; ggml-org/Step-3.5-Flash-GGUF | safetensors, GGUF | | 2026-11-15 |
| NVIDIA Nemotron | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16; unsloth and ggml-org Nemotron 3.5 Bolt, Lightning, Super-VL GGUF | safetensors, GGUF incl. NVFP4 | MTP and draft files load as drafts of their target | 2026-11-15 |
| Liquid, Flint, Muse, Inkling, Hy4, Clover, Ornith, Granite, Yi, others | LiquidAI/LFM2.5-24B-A2B (GGUF), RecursiveSI/flint, unsloth/Muse-Spark-Open-816B-A42B-GGUF, thinkingmachines/Inkling and unsloth/Inkling-Small-GGUF (23 files), tencent/Hy4-preview, Diffuspace/Clover-35B-A3B-Preview, ornith-ai/Ornith-1.0-35B and Ornith-1.5 GGUF, ibm-granite/granite-4.1-3b-llmon-preview, dphn/dolphin-2.9.1-yi-1.5-34b, openbmb/MiniCPM5-2B, HuggingFaceTB/SmolLM2-135M, bartowski Nanbeige4.3-33B, Thomson-1.0-Small, KAT-Coder-V2.5-Dev, farbodtavakkoli OTel-2.0-Decision, webAI TwIL-8B, ggml-org Clef-Flash | safetensors, GGUF | Clover: block-diffusion preview tensors not imported | 2026-11-15 |
| PaliGemma | google/paligemma-3b-pt-224 | safetensors | Partial: text decoder decodes causally, which matches transformers for text-only input; image prefix attention is not implemented | 2026-11-15 |

In all, 1,080 of 1,138 GGUF files across 88 repositories and 49 of 72 Hub checkpoints passed this sweep; the remainder are in section 9.

### 1.3 Vision-language models

| Model | Hub repository | Format | Providers | Verification | Notes | Verified |
|---|---|---|---|---|---|---|
| Qwen3-VL 2B Instruct | Qwen/Qwen3-VL-2B-Instruct | safetensors, GGUF + mmproj | CPU, Vulkan | greedy text identical to transformers / llama.cpp; mmproj vs llama.cpp | | 2026-11-14 |
| Qwen3.5 0.8B | Qwen/Qwen3.5-0.8B | GGUF BF16 + mmproj | CPU, Vulkan | mmproj vs llama.cpp | | 2026-11-14 |
| Qwen3.8 27B | Qwen/Qwen3.8-27B | GGUF mmproj (BF16) | CPU, Vulkan | projector output vs llama.cpp | | 2026-11-14 |
| Ministral 3 3B, Gemma 3 4B, LFM2-VL 450M | mistralai / google / LiquidAI | GGUF + mmproj | CPU, Vulkan | text vs llama.cpp (see 1.1); projector files load | | 2026-11-14 |
| PaddleOCR-VL 1.6, Unlimited-OCR | PaddlePaddle/PaddleOCR-VL-1.6, baidu/Unlimited-OCR | safetensors | CPU | Loads | | 2026-11-15 |

### 1.4 Diffusion language models

| Model | Hub repository | Verification | Notes |
|---|---|---|---|
| Dream, LLaDA, RND1, LLaDA-MoE architectures | Dream-7B, LLaDA-8B (and tiny random checkpoints of each) | Verified (tiny): masked-diffusion unmasking loop against the reference implementations on CPU and Vulkan | No real checkpoint compared yet |

### 1.5 Embedding and reranking models

Cosine similarity above 0.999 per embedding and identical token ids against the reference; rerankers within about 1e-3 of the reference logits.

| Model | Hub repository | Format | Providers | Verification | Verified |
|---|---|---|---|---|---|
| bge-small-en-v1.5 | BAAI/bge-small-en-v1.5 | safetensors, GGUF f16, Q8_0 | CPU, Vulkan | cosine > 0.999 vs reference | 2026-11-14 |
| all-MiniLM-L6-v2 | sentence-transformers/all-MiniLM-L6-v2 | safetensors, GGUF f16 | CPU, Vulkan | cosine > 0.999 | 2026-11-14 |
| e5-small-v2, multilingual-e5-small | intfloat/e5-small-v2, intfloat/multilingual-e5-small | safetensors | CPU, Vulkan | cosine > 0.999 | 2026-11-14 |
| nomic-embed-text v1.5 | nomic-ai/nomic-embed-text-v1.5 | safetensors, GGUF f16, Q8_0 | CPU, Vulkan | cosine > 0.999 | 2026-11-14 |
| ms-marco-MiniLM-L6-v2 (reranker) | cross-encoder/ms-marco-MiniLM-L6-v2 | safetensors | CPU, Vulkan | logits within ~1e-3 | 2026-11-14 |
| bge-reranker-base, bge-reranker-v2-m3 | BAAI/bge-reranker-base, BAAI/bge-reranker-v2-m3 | safetensors | CPU, Vulkan (v2-m3: Vulkan) | logits within ~1e-3 | 2026-11-14 |
| EmbeddingGemma 300M and 2 | ggml-org/embeddinggemma-300M-GGUF, unsloth/embeddinggemma-2-GGUF | GGUF | CPU, Vulkan | Loads | 2026-11-15 |
| bge-m3, mxbai-embed-large-v1 | ggml-org/bge-m3-Q8_0-GGUF, mixedbread-ai/mxbai-embed-large-v1 | GGUF | CPU, Vulkan | Loads | 2026-11-15 |

---

## 2. Speech to text

Verified on CPU and Vulkan. Reference: transformers (token-exact greedy ids), whisper.cpp (segments and text).

| Model | Hub repository | Formats | Verification | Notes | Verified |
|---|---|---|---|---|---|
| Whisper tiny, tiny.en, base, base.en, small, small.en | openai/whisper-*, ggerganov/whisper.cpp | HF, whisper.cpp `.bin` (f16, q5_1), `.slm` | ids identical to transformers (with and without timestamps, 5-beam search, language detection); whisper.cpp f16 identical, q5_1 same words (3 of 24 runs differ in one punctuation mark) | | 2026-11-15 |
| Whisper large-v3-turbo | openai/whisper-large-v3-turbo, oxide-lab/whisper-large-v3-turbo-GGUF | HF, whisper.cpp `.bin` f16 / q5_0 / q8_0, whisper GGUF Q8_0 / Q4_K_M | transcript identical to whisper.cpp | medium and large-v1/v2/v3 map the same way, not verified | 2026-11-15 |
| Whisper as GGUF (tiny, base) | handy-computer/whisper-*-gguf | GGUF F32, F16, Q8_0, Q4_K_M | tiny F16 / F32 identical to whisper.cpp, quantized word for word | Candle-style whisper GGUF files are rejected with a message | 2026-11-15 |
| Distil-Whisper small.en, large-v3.5 | distil-whisper/distil-small.en, distil-whisper/distil-large-v3.5 | HF | ids identical to transformers, beam search identical | | 2026-11-15 |
| Moonshine tiny, base | moonshine-ai/moonshine-tiny, moonshine-ai/moonshine-base | HF, `.slm` | ids identical to transformers | | 2026-11-16 |
| Parakeet CTC 0.6B | nvidia/parakeet-ctc-0.6b | HF (transformers format), `.slm` | ids identical to transformers | 1.1B maps the same way, not verified | 2026-11-16 |
| Parakeet TDT 0.6B v3 | nvidia/parakeet-tdt-0.6b-v3 | HF, `.slm` | ids identical to transformers | | 2026-11-16 |
| Canary 1B v2 | nvidia/canary-1b-v2 | HF, `.slm` | ids identical to transformers (English transcription with punctuation) | translation prompt works; no timestamps | 2026-11-16 |
| Whisper large-v3 | openai/whisper-large-v3, handy-computer/whisper-large-v3-gguf | HF, whisper GGUF Q5_K_M | transcript identical to transformers greedy (English, LibriSpeech, German, Spanish, Japanese); the Q5_K_M file gives the same text | | 2026-11-16 |
| wav2vec2 CTC (base-960h, XLSR-53 fine-tunes) | facebook/wav2vec2-base-960h, jonatasgrosman/wav2vec2-large-xlsr-53-japanese | HF (`.safetensors` and `pytorch_model.bin`) | base-960h: CTC ids and text identical to transformers on jfk and LibriSpeech, CPU and Vulkan (encoder test); XLSR-53 Japanese (layer-normed, pre-norm variant): text identical to transformers | the other `jonatasgrosman/wav2vec2-large-xlsr-53-*` language models share the layout, not run | 2026-11-16 |
| Parakeet as GGUF (TDT 0.6B v3) | handy-computer/parakeet-tdt-0.6b-v3-gguf | transcribe.cpp GGUF F16, Q8_0 | F16: text identical to the Hugging Face checkpoint on 5 clips (English, LibriSpeech, German, Spanish, Japanese); Q8_0: same text except one capital letter in German and a few kana in Japanese | Q4_K_M, Q5_K_M, Q6_K map the same way, not run | 2026-11-16 |
| Parakeet unified (RNN-T, English) | handy-computer/parakeet-unified-en-0.6b-gguf | transcribe.cpp GGUF Q8_0 | Partial: transcripts match the reference text on jfk and LibriSpeech, no NeMo reference at hand to compare tokens | streaming chunked-attention settings read, but run with full context | 2026-11-16 |
| Parakeet RNN-T | nvidia/parakeet-rnnt-* | HF | Partial: maps, not verified | | |
| Qwen3-ASR 1.7B (audio language model) | Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-1.7B-hf, handy-computer/Qwen3-ASR-1.7B-gguf | HF (original and transformers layouts), transcribe.cpp GGUF | greedy ids identical to transformers on jfk and LibriSpeech, language detected and English forced, CPU and Vulkan; the `-hf` layout and the transcribe.cpp F16 GGUF give the same ids; the language the model names is returned | 30 languages; recordings in 30 s windows; Qwen3-ASR 0.6B maps the same way, not run | 2026-11-16 |
| Voxtral Mini 3B (audio language model) | mistralai/Voxtral-Mini-3B-2507 | HF (transformers layout, `tekken.json` tokenizer) | greedy ids identical to transformers on jfk and LibriSpeech, without a language and with English given, CPU and Vulkan | 30 s per window (375 audio rows); Voxtral Small 24B maps the same way, not run; the realtime 4B (streaming decoder) is not supported | 2026-11-16 |

Long recordings are cut into 30 s windows (Whisper uses its own timestamp-driven loop). Not yet: temperature fallback, Whisper word timestamps.

## 3. Text to speech

| Model | Hub repository | Formats | Providers | Verification | Notes | Verified |
|---|---|---|---|---|---|---|
| Kokoro 82M v1.0 | hexgrad/Kokoro-82M | PyTorch folder, `.slm` | CPU, Vulkan | symbols, durations and F0 identical to the `kokoro` package; waveform within 1.3e-2 relative RMS (the reference's own noise floor) | all voice packs load; English front end only | 2026-11-15 |
| Orpheus 3B + SNAC 24 kHz | canopylabs/orpheus-3b-0.1-ft (unsloth mirror), hubertsiuzdak/snac_24khz | safetensors | CPU, Vulkan | 0 argmax disagreements over 140 tokens vs transformers; SNAC decoder within 2.4e-6 | voices tara, leah, jess, leo, dan, mia, zac, zoe | 2026-11-15 |
| OuteTTS 0.2 500M + WavTokenizer-Large-75 | OuteAI/OuteTTS-0.2-500M, WavTokenizer GGUF | GGUF pair | CPU, Vulkan | teacher-forced logits vs llama.cpp; decoder waveform within 2e-3 | no default speaker profile; OuteTTS 0.3 not supported | 2026-11-15 |
| Chatterbox (English) | ResembleAI/chatterbox | safetensors folder (`t3_cfg`, `s3gen`, `ve`, `tokenizer.json`, `conds.pt`) | CPU, Vulkan | vs the chatterbox-tts package: T3 conditioning within 7e-7, guided greedy chain teacher forced with 0 argmax disagreements over 92 tokens, S3Gen flow mel within 6e-7, HiFT waveform within 7e-6 (relative RMS, the reference's noise) | built-in voice `default`, or the voice of a recording (`SpeechSynthesisOptions.ReferenceAudio`; voice encoder within 1.2e-6, S3 prompt and flow tokens identical, CAM++ x-vector within 8e-6 vs the package); multilingual and Turbo variants not run | 2026-11-16 |
| F5-TTS v1 Base + Vocos | SWivid/F5-TTS (F5TTS_v1_Base), charactr/vocos-mel-24khz | safetensors folder + Vocos folder (`vocoder` auxiliary) | CPU, Vulkan | vs the f5-tts package: sampled mel within 3.5e-5 and Vocos waveform within 2.4e-5 (relative RMS, the reference's noise); character ids and duration identical | speaks with the voice of a reference recording (`SpeechSynthesisOptions.ReferenceAudio` and `ReferenceText`); no Chinese pinyin conversion, no automatic transcription of the reference | 2026-11-16 |
| XTTS v2 | coqui/XTTS-v2 | Coqui checkpoint folder (`model.pth`, `config.json`, `vocab.json`, `speakers_xtts.pth`) | CPU, Vulkan | vs coqui-tts: text ids and 90 of 90 greedy codes identical; latents within 6.8e-5, HiFi-GAN waveform within 1.8e-4 (relative RMS); Vulkan real-time factor 0.24 | the 58 built-in speakers or the voice of a recording (`SpeechSynthesisOptions.ReferenceAudio`; style latents within 3.7e-6, speaker embedding within 2.8e-6 vs coqui-tts); English text cleaning (other languages lowercased only) | 2026-11-16 |

## 4. Image generation

Reference: diffusers (same initial noise) and stable-diffusion.cpp. PSNR values are image-vs-image on a fixed seed; above about 30 dB the
images are visually identical. All run on CPU and Vulkan (Vulkan used for the real checkpoints). Supported: txt2img, img2img, inpainting by mask blend, `.slm`.

| Model | Hub repository | Formats | Verification | Notes | Verified |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | SD 1.5 checkpoints (diffusers folder, original single file, GGUF) | diffusers folder, single file `.safetensors` / `.ckpt`, GGUF f16, Q8_0, Q4_0 | Euler 76.3 dB, DDIM 76.2 dB, DPM++ 2M Karras 73.8 dB, PNDM 58.8 dB vs diffusers fp32; GGUF f16 48.7 dB | Q4_0 file produces a degenerate image in every implementation | 2026-11-15 |
| Stable Diffusion 2.1 (768-v) | sd2-community/stable-diffusion-2-1 | diffusers folder, single file | Euler 70.2 dB, DDIM 64.1 dB, DPM++ 2M Karras 51.8 dB | fp16 files: visually identical, 28 to 38 dB | 2026-11-15 |
| SDXL base 1.0 | stabilityai/stable-diffusion-xl-base-1.0, gpustack/stable-diffusion-xl-base-1.0-GGUF, kostakoff/stable-diffusion-xl-base-1.0-GGUF | diffusers folder, single file, GGUF Q8_0 / Q4_0 / Q4_K | Euler 42.4 dB, DPM++ 2M Karras 30.0 dB (fp16 files); GGUF Q8_0 27.7 dB | Q4_K: visually equivalent (17.3 dB); separate VAE / text-encoder files not supported | 2026-11-15 |
| SDXL-Turbo | stabilityai/sdxl-turbo, OlegSkutte/sdxl-turbo-GGUF | diffusers, GGUF Q8_0 | Q8_0 28.5 dB vs stabilityai/sdxl-turbo | 4 steps, no guidance | 2026-11-15 |
| SDXL-Lightning | mzwing/SDXL-Lightning-GGUF | GGUF | Partial: maps the same way as SDXL, not verified | | |
| SD-Turbo | stabilityai/sd-turbo | diffusers | Loads | | |
| FLUX.1-schnell | black-forest-labs/FLUX.1-schnell, city96/FLUX.1-schnell-gguf | diffusers (bf16), GGUF Q8_0 + CLIP-L, T5-XXL, VAE companions | bf16: 32.5 dB (512) / 38.8 dB (1024) vs diffusers; Q8_0: 33.4 dB / 39.7 dB vs diffusers running the same files | companion files found beside the GGUF; 1024x1024 step about 1 s on a 32 GB GPU | 2026-11-15 |
| FLUX.1-dev | black-forest-labs/FLUX.1-dev, city96/FLUX.1-dev-gguf | GGUF Q8_0 + companions | 31.7 dB (512, 20 steps) vs diffusers on the same files | 28 steps at 1024x1024 in 33 s, 23.5 GB device memory | 2026-11-15 |
| SD3.5 Medium | stabilityai/stable-diffusion-3.5-medium (ungated mirror adamo1139/stable-diffusion-3.5-medium-ungated), city96/stable-diffusion-3.5-medium-gguf | diffusers (fp16), GGUF Q8_0 + CLIP-L, CLIP-G, T5-XXL, VAE companions | fp16: 46.6 dB (512) / 38.8 dB (1024) vs diffusers f32; transformer 1.4e-4; Q8_0: 42.5 dB (512) / 29.4 dB (1024) vs diffusers on the same files | 28 steps, guidance 7; 1024x1024 in 12.6 s | 2026-11-16 |
| Chroma1-HD | lodestones/Chroma1-HD, silveroxides/Chroma1-HD-GGUF | GGUF Q8_0 + T5-XXL, VAE companions | 46.3 dB (512) / 37.8 dB (1024) vs diffusers on the same files (boolean text mask, as the original Chroma) | 20 guided steps at 1024x1024 in 48 s | 2026-11-16 |
| Qwen-Image | Qwen/Qwen-Image, city96/Qwen-Image-gguf, unsloth/Qwen2.5-VL-7B-Instruct-GGUF (text encoder) | GGUF Q4_K_M + Qwen2.5-VL-7B text encoder (the pipeline's `text_encoder` folder or a Qwen2.5-VL GGUF) and VAE (the pipeline's or ComfyUI's `qwen_image_vae.safetensors`) companions | Q4_K_M: 30.9 dB (512) / 29.8 dB (1024) vs diffusers on the same files (20 steps, true CFG 4); text encoder 4.6-6.4e-4, transformer 8.4e-3 | 1024x1024 20 guided steps in 131 s on a 32 GB GPU (13 GB transformer + 16 GB bf16 text encoder, alternating); Studio loads the GGUF with its companions; the diffusers folder loads (verified on the tiny pipeline; its 41 GB bf16 transformer was not run) | 2026-11-16 |
| FLUX.2 Klein 4B | black-forest-labs/FLUX.2-klein-4B, unsloth/FLUX.2-klein-4B-GGUF | GGUF Q8_0 + Qwen3-4B text encoder and VAE companions (the pipeline's folders) | 40.1 dB (512) / 41.3 dB (1024) vs diffusers on the same files (4 distilled steps); text encoder ≤ 1.3e-3, transformer 1.6e-3 | 1024x1024 in 3.4 s | 2026-11-17 |
| Qwen-Image 2.1 Turbo | Qwen/Qwen-Image-2.1-Turbo, unsloth/Qwen-Image-2.1-Turbo-GGUF | GGUF Q8_0 + the Qwen-Image 2.1 companions | 34.5 dB (1024, its 8-step `sample_sigmas`) vs diffusers on the same files | 8 steps at 1024x1024 in 14.8 s; the schedule comes from `model_index.json` or the `turbo` file name | 2026-11-16 |
| Qwen-Image 2.1 | Qwen/Qwen-Image-2.1, unsloth/Qwen-Image-2.1-GGUF | GGUF Q8_0 + Qwen3-VL-8B text encoder and VAE companions (the pipeline's folders) | 42.2 dB (512) / 37.4 dB (1024) vs diffusers on the same files (20 steps, no guidance); text encoder ≤ 9.6e-4 | 1024x1024 20 steps in 36 s; text-to-image only (the editing path is not implemented) | 2026-11-16 |
| Z-Image-Turbo | Tongyi-MAI/Z-Image-Turbo, unsloth/Z-Image-Turbo-GGUF | diffusers, GGUF Q8_0 + Qwen3-4B text encoder, VAE companions | 49.6 dB (512) / 46.4 dB (1024) vs diffusers on the same files; text encoder 1.4e-4, transformer 1.5e-3 | 9 steps, no guidance; 1024x1024 in 14.6 s; companions found in the pipeline folder beside the GGUF folder | 2026-11-16 |

### 4.1 Popularity sweep (2026-11-16)

The most downloaded text-to-image checkpoints on the Hub that use a supported architecture, loaded as distributed (single-file
`.safetensors`; tools/make-sweep-references.py, `SweepTests`): Vulkan against diffusers' `from_single_file` (f32, CUDA) from the same latents,
Euler, 20 steps, guidance 7 with a negative prompt (SDXL-Lightning: 4 steps on trailing timesteps, no guidance, as its model card runs it).
All images are visually the same picture; the lower values of the anime merges come from their high-frequency line work.

| Model | Hub repository | File | PSNR vs diffusers | Load + first image |
|---|---|---|---|---|
| Realistic Vision 5.1 (SD 1.5) | SG161222/Realistic_Vision_V5.1_noVAE | Realistic_Vision_V5.1_fp16-no-ema.safetensors | 46.1 dB (512) | 19 s |
| DreamShaper 8 (SD 1.5) | Lykon/DreamShaper | DreamShaper_8_pruned.safetensors | 44.3 dB (512) | 54 s |
| RealVisXL 5.0 | SG161222/RealVisXL_V5.0 | RealVisXL_V5.0_fp16.safetensors | 40.2 dB (1024) | 107 s |
| Juggernaut XL v9 | RunDiffusion/Juggernaut-XL-v9 | Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors | 44.1 dB | 77 s |
| Animagine XL 4.0 | cagliostrolab/animagine-xl-4.0 | animagine-xl-4.0.safetensors | 40.1 dB | 90 s |
| Pony Diffusion V6 XL | LyliaEngine/Pony_Diffusion_V6_XL | ponyDiffusionV6XL_v6StartWithThisOne.safetensors | 40.4 dB | 111 s |
| WAI-Illustrious-SDXL 17.0 | LyliaEngine/waiIllustriousSDXL_v170 | waiIllustriousSDXL_v170.safetensors | 31.8 dB | 52 s |
| Illustrious XL 0.1 | OnomaAIResearch/Illustrious-xl-early-release-v0 | Illustrious-XL-v0.1.safetensors | 36.2 dB | 118 s |
| NoobAI XL 1.1 | Laxhar/noobai-XL-1.1 | NoobAI-XL-v1.1.safetensors | 28.5 dB | 89 s |
| SDXL-Lightning 4-step | ByteDance/SDXL-Lightning | sdxl_lightning_4step.safetensors | 45.9 dB | 92 s |
| Qwen-Image-2512 | unsloth/Qwen-Image-2512-GGUF | qwen-image-2512-Q4_K_M.gguf + the Qwen-Image companions | 25.5 dB (1024, 20 steps, true CFG 4; vs diffusers on the same files; visually identical, three blocks run exact attention) | 150 s |

Real LoRAs on SDXL base 1.0 (single file), `RealLoraTests` (diffusers `load_lora_weights`, Euler, same latents, no guidance): ByteDance's
`sdxl_lightning_4step_lora.safetensors` (kohya, SGM names) 45.0 dB, nerijs/pixel-art-xl (kohya, strength 0.9) 64.8 dB; the unadapted
images are 21 / 29 dB away.

The Diffusers repositories of these (RealVisXL, Juggernaut, Animagine, NoobAI, Illustrious: `StableDiffusionXLPipeline`) are offered for
download in Studio (`ClassifyPipeline`), the single files load in Studio's library like the SDXL base file. Not covered, because their
architectures are not implemented yet: FLUX.2 Dev, Krea-2, MiniMax-H3-Image (Qwen-Image 2.1 and FLUX.2 Klein: section 4).

LoRA (merged at load, role `lora@scale`; tools/make-tiny-lora.py, `TinyLoraTests`): random LoRAs on the tiny SD 1.5, SDXL and
FLUX.1-dev pipelines against diffusers' `load_lora_weights` from the same noise, strength 0.8: SD 1.5 kohya and PEFT 82 dB (CPU) / 76 dB
(Vulkan), SDXL kohya (SGM names, both text encoders) and PEFT 44 / 41 dB, FLUX kohya (original names, fused q/k/v) and PEFT (q/k/v and
the single blocks' MLP input as row blocks of the fused projections) 81-85 / 75 dB.

## 5. Vision

Reference: transformers (relative RMS of the outputs, CPU and Vulkan identical unless stated). One sample image per model.

| Model | Hub repository | Task | Verification | Notes | Verified |
|---|---|---|---|---|---|
| ViT base | google/vit-base-patch16-224 | classification | same top-1, logits 1.2e-3 | | 2026-11-15 |
| DINOv2 small | facebook/dinov2-small | image embedding | CLS embedding 1.2e-3 | registers variant supported | 2026-11-15 |
| CLIP ViT-B/32 | openai/clip-vit-base-patch32 | image and text embedding, zero-shot | image 2.6e-3, text 1e-6, zero-shot logits 4.5e-4 | | 2026-11-15 |
| SigLIP base | google/siglip-base-patch16-224 | image and text embedding, zero-shot | image 2.2e-3, text 1e-6, logits 6.3e-4 | | 2026-11-15 |
| DETR ResNet-50 | facebook/detr-resnet-50 | detection | same four detections, boxes within 2 px | fixed 640 px square canvas | 2026-11-15 |
| RT-DETR R18vd | PekingU/rtdetr_r18vd | detection | same five detections | fixed square canvas | 2026-11-15 |
| Depth Anything V2 small | depth-anything/Depth-Anything-V2-Small-hf | depth | depth map 1.5e-4 | fixed 518 px canvas | 2026-11-15 |
| SAM 2.1 Hiera tiny | facebook/sam2.1-hiera-tiny | segmentation (point, box prompts) | mask IoU > 0.98 vs transformers | image path only, no video | 2026-11-16 |
| TrOCR base printed | microsoft/trocr-base-printed | OCR | text identical to transformers | greedy | 2026-11-16 |
| Real-ESRGAN x4plus | xinntao/Real-ESRGAN (official `.pth`) | 4x upscaling | > 40 dB PSNR vs the reference, also tiled | | 2026-11-16 |
| SwinIR classical x4 | JingyunLiang/SwinIR (`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`) | 4x upscaling | 92.9 dB (Vulkan) / 91.6 dB (CPU) PSNR vs the authors' network, tiled 61.6 dB | sizes in multiples of 8; real-world x4 variant built, not verified | 2026-11-16 |
| ResNet 18 / 34 / 50, Wide ResNet 50-2 | timm/resnet18.a1_in1k, resnet34.a1_in1k, resnet50.a1_in1k, wide_resnet50_2.racm_in1k | classification | logits < 2e-3 vs timm, same top-1 | no label names in timm repositories; other timm families not supported | 2026-11-16 |
| NSFW image detection (ViT) | Falconsai/nsfw_image_detection | classification | logits < 2e-3, same top-1 | | 2026-11-16 |
| CLIP ViT-B/16, ViT-L/14 | openai/clip-vit-base-patch16, openai/clip-vit-large-patch14 | image and text embedding, zero-shot | image 2.1e-3 / 2.4e-3, zero-shot logits 4.6e-4 / 3.3e-4 | B/16 ships only `pytorch_model.bin` | 2026-11-16 |
| CLIP ViT-B/32 LAION-2B | laion/CLIP-ViT-B-32-laion2B-s34B-b79K | image and text embedding, zero-shot | image 2.7e-3, zero-shot logits 5.0e-4 | the `open_clip_*` files of the repository are ignored | 2026-11-16 |
| SigLIP 2 base, SigLIP so400m | google/siglip2-base-patch16-224, google/siglip-so400m-patch14-384 | image and text embedding, zero-shot | image 1.3e-3 / 1.6e-3, zero-shot logits 1.5e-4 / 1.8e-4 | SigLIP 2 NaFlex and giant-opt not tried | 2026-11-16 |
| DINOv2 base, DINOv2 with registers base, DINO ViT-B/16 | facebook/dinov2-base, facebook/dinov2-with-registers-base, facebook/dino-vitb16 | image embedding | CLS embedding < 2e-3 | | 2026-11-16 |
| RT-DETR R50vd, R101vd (COCO + O365), v2 R18vd | PekingU/rtdetr_r50vd, PekingU/rtdetr_r101vd_coco_o365, PekingU/rtdetr_v2_r18vd | detection | same detections as transformers, boxes within 2 px | fixed square canvas | 2026-11-16 |
| Table Transformer detection | microsoft/table-transformer-detection | detection | same detection as transformers | fixed square canvas | 2026-11-16 |
| Depth Anything V2 base / large, Depth Anything V1 base | depth-anything/Depth-Anything-V2-Base-hf, Depth-Anything-V2-Large-hf, LiheYoung/depth-anything-base-hf | depth | depth map 7.4e-5 / 9.8e-5 / 1.2e-4 | fixed 518 px canvas | 2026-11-16 |
| SegFormer b0 ADE20K, b5 ADE20K (640 px) | nvidia/segformer-b0-finetuned-ade-512-512, nvidia/segformer-b5-finetuned-ade-640-640 | image-segmentation-semantic | logits 5.1e-4 / 1.5e-6, label map 99.81 % / 99.998 % identical (CPU = Vulkan) | fixed square input; `pytorch_model.bin` read for b5 | 2026-11-16 |
| TrOCR base handwritten | microsoft/trocr-base-handwritten | OCR | text identical to transformers | greedy; TrOCR small (DeiT encoder, sinusoidal positions) and large (BEiT) not supported | 2026-11-16 |
| BiRefNet, RMBG-2.0 (background removal) | ZhengPeng7/BiRefNet, briaai/RMBG-2.0 (CC BY-NC 4.0, gated: accept the terms) | image-segmentation-semantic (matte) | matte logits 1.9e-6 / 2.3e-6 relative rms vs the authors' code, alpha within 1.4e-4 (CPU 3.2e-4), foreground identical | 1024 px square; the network's settings are read from the checkpoint's `birefnet.py` (never run): Swin t/s/b/l backbone, deformable-ASPP decoder; both code revisions' input-patch channel orders | 2026-11-16 |
| Florence-2 base | florence-community/Florence-2-base | OCR / caption / detection (`ITextRecognizer` tasks `<OCR>`, `<CAPTION>`, `<DETAILED_CAPTION>`, `<OD>`, ...) | image features 6.3e-4 and encoder output 1.1e-3 vs transformers; greedy text and `<OD>` boxes identical on two images (CPU and Vulkan) | num_beams=1 with the checkpoint's no-repeat n-gram (3) and forced BOS; beam search, region-to-X polygon tasks not done | 2026-11-16 |
| YOLOv8n, YOLOv8n-seg, YOLOv10n (Ultralytics ONNX exports; also YOLO11 and the larger sizes by the same graph) | `yolo export format=onnx` output (AGPL-3.0: check the license) | object-detection (instance masks for `-seg`) | same classes, scores within 0.01 and boxes within 2 px as onnxruntime with Ultralytics' decoding; mask IoU 0.994-1.000; CPU = Vulkan | detection and segmentation heads only (pose, obb and classification heads are not read); `.pt` checkpoints are not read (they pickle Ultralytics code), export to ONNX first; the export's fixed input size is used (letterboxed) | 2026-11-17 |
| YOLOS tiny, small | hustvl/yolos-tiny, hustvl/yolos-small (revision 3d8f7130d3ce4907cb206fe1c8485dc8fe8703de) | object-detection | same detections as transformers, boxes within 2 px, scores within 0.02; CPU = Vulkan | fixed 640 px square canvas; small has the mid position tables | 2026-11-17 |

## 6. Audio

Reference: transformers (EnCodec, DAC, Mimi), onnxruntime and the `silero-vad` package (VAD). CPU and Vulkan.

| Model | Hub repository | Kind | Verification | Notes | Verified |
|---|---|---|---|---|---|
| EnCodec 24 kHz | facebook/encodec_24khz | codec | codes token-exact at 6 kbps; decoder < 1e-3 relative RMS | 48 kHz model rejected; at 24 kbps up to 0.1 % of codes differ (near ties) | 2026-11-16 |
| DAC 44 kHz | descript/dac_44khz | codec | codes token-exact, decoded audio < 1e-3 | | 2026-11-16 |
| Mimi | kyutai/mimi | codec | codes token-exact for 8 and 32 stages | | 2026-11-16 |
| Silero VAD v5 | snakers4/silero-vad (16 kHz ONNX) | voice activity | probabilities within 1e-3 of onnxruntime, segments identical | 8 kHz branch not imported; also used to pre-segment long speech recognition | 2026-11-16 |

## 7. ONNX models

130 ONNX models were run against onnxruntime 1.3x on CPU and Vulkan (round 8 added the operators several of the failing ones needed; see
onnx-support.md for the current state of each). Statuses below are from the survey and the later rounds; "OK" means outputs within the strict
tolerance of onnxruntime (typically relative RMS below 1e-5).

| Group | Models (Hub repositories: onnxmodelzoo/*, Xenova/*, onnx-community/*, others) | Result |
|---|---|---|
| Image classification | ResNet-18/50/101-DUC, DenseNet-9, ShuffleNet-v2, SqueezeNet 1.1, MobileNetV2, EfficientNet-Lite4, ConvNeXt-tiny, Swin-tiny, DeiT-tiny, ViT-base, MobileViT, MobileViTv2, SegFormer-b0 | Verified (CPU, Vulkan) |
| Quantized (int8) | MobileNetV2-int8, ResNet50-int8, EfficientNet-Lite4-int8, ConvNeXt-tiny-q, CLIP-vision-int8 | Verified within relative RMS 5e-2 to 2e-1 (int8 requantization noise) |
| Detection and pose | DETR (ResNet-50, panoptic), YOLOS-tiny, RT-DETR-r18, YOLOv8n, YOLOv8n-seg, YOLOv8n-pose, YOLO11n, YOLO11n-face, YOLOv10n, YOLO26n, RetinaNet-9, TinyYOLOv2, RFB-320, SCRFD-10g, ViTPose-base, RTMPose-m | Verified |
| SSD, YOLOv3, Faster R-CNN, Mask R-CNN | onnxmodelzoo ssd-10, yolov3-12, FasterRCNN-12, MaskRCNN-12 | Operators NonMaxSuppression, RoiAlign, Loop implemented in round 8; matched onnxruntime on ssd-10, yolov3-12, fasterrcnn-12, maskrcnn-12 |
| Depth, segmentation, matting | Depth-Anything-small / V2-small / V2-base, FCN-ResNet50, MobileSAM encoder and decoder, SAM2 encoder, U2-Netp, MODNet, RAFT-large | Verified |
| Super-resolution and style | Real-ESRGAN x4v3, Swin2SR, super-resolution-10, candy-9, mosaic-9, AnimeGAN (hayao, v2), SDXL VAE, UVDoc | Verified |
| Text encoders | BERT base (fp32, q4, fp16), DistilBERT, MiniLM-L6, RoBERTa, DeBERTa-v3-small, ModernBERT-base, nomic-moe, CLIP text, SigLIP text and vision, T5-small encoder, distilBART, BERT-SQuAD-12 | Verified; BERT fp16 and dynamic-int8 within 5e-2 |
| Decoders | GPT-2 (plain, past, merged), gpt2-lm-head-10, T5-small decoder, distilBART decoder, Whisper tiny encoder and decoders, Moonshine encoder, TrOCR-small-printed, ViT-GPT2 captioning, Florence-2-base (4 parts), Qwen2.5-0.5B q4, Gemma-3-270m, LFM2-350M, Phi-3.5-mini q4f16 | Verified (decoders run in their no-past branch); Gemma-3, LFM2 and Phi-3.5 match onnxruntime since round 8 |
| Speech and audio | wav2vec2-base, Silero VAD, OpenUnmix vocals, Zipformer streaming encoder, PP-OCR detection and recognition (v2, v3, v5) | Verified |
| Other | tiny-mamba, tiny Stable Diffusion text / UNet / VAE, arcface (ResNet100, w600k-r50), emotion-FerPlus, mnist-12 | Verified |
| Partial or not working | Qwen2.5-0.5B fp16 and dynamic-int8 (numerically off), ssd-12-int8 (differs from onnxruntime 1.32; matched 1.31), BERT-SQuAD-12-int8 (runtime-B MatMulInteger), GLiNER-small (dynamic Expand), Grounding-DINO-tiny, DPT-hybrid-MiDaS (convolution variant), Kokoro ONNX (the STFT operator is implemented since round 8; not re-run) | see onnx-support.md |

## 8. Video generation

Reference: diffusers in f32 on CUDA (same initial latents), 832×480 (LTX: 704×480), 17 frames (LTX: 25), 20 steps; PSNR is the worst
frame. Vulkan on a 32 GB GPU. Supported: text-to-video, image-to-video where the model has it, streaming decode, export to WebM / GIF /
APNG (MP4 with a system ffmpeg), `.slm`.

| Model | Hub repository | Formats | Verification | Notes | Verified |
|---|---|---|---|---|---|
| Wan 2.1 T2V 1.3B | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | diffusers folder; transformer-only GGUF / safetensors with the folder as companion | text-to-video 50.8 dB, transformer 2.4e-4, umT5 3e-6 relative RMS | 4.2 s per step at 832×480×33 frames (f32), 16.3 GB | 2026-11-16 |
| Wan 2.2 TI2V 5B | Wan-AI/Wan2.2-TI2V-5B-Diffusers | diffusers folder | image-to-video 55.1 dB, transformer 2.8e-4 | text- and image-to-video; 3.5 s per step (f32), 30.7 GB | 2026-11-16 |
| LTX-Video 0.9.0 (2B) | Lightricks/LTX-Video (diffusers folder; T5-XXL of FLUX.1 / PixArt) | diffusers folder | text-to-video 57.5 dB, transformer 1.8e-5, T5 1.4e-5 relative RMS (704×480, 25 frames) | LTX-Video 0.9.1+ (causal / timestep-conditioned VAE) not yet | 2026-11-16 |
| Wan 2.2 TI2V 5B GGUF | QuantStack/Wan2.2-TI2V-5B-GGUF (Q8_0) | GGUF + the Wan 2.2 diffusers folder as companion | image-to-video 51.1 dB vs the f32 reference, transformer 2.0e-3 | 1.6 s per step, 16.4 GB | 2026-11-16 |
| Wan 2.2 T2V A14B GGUF (two experts) | QuantStack/Wan2.2-T2V-A14B-GGUF (HighNoise + LowNoise Q4_K_M) | the two GGUFs (`--transformer-2` / `AuxiliaryRoles.SecondTransformer`) + a Wan 2.1 diffusers folder as companion | against diffusers running the same GGUFs (f32 on CUDA): high- / low-noise transformer 3.4e-4 / 2.3e-4, umT5 3e-6 relative RMS; text-to-video 21-25 dB per frame (832×480, 17 frames, 20 steps), the same scene; a 1e-3 nudge of the initial noise moves Sharplight's own result as far | 14.6 s per step at 832×480×33 frames, 16.1 GB peak, 334 s total (diffusers: 23 s per step at 17 frames) | 2026-11-16 |
| Wan 2.1 I2V 14B 480P GGUF | city96/Wan2.1-I2V-14B-480P-gguf (Q4_K_M) | GGUF + a Wan 2.1 diffusers folder (umT5, VAE) + Wan-AI/Wan2.1-I2V-14B-480P-Diffusers' `image_encoder/` and `image_processor/` (`--image-encoder`) | against diffusers on the same GGUF: transformer 3.9e-4, CLIP 3.5e-3 relative RMS; image-to-video 30.6-48.1 dB per frame (832×480, 17 frames, 20 steps) | image-to-video | 2026-11-16 |
| Wan 2.1 FLF2V 14B 720P GGUF | city96/Wan2.1-FLF2V-14B-720P-gguf (Q4_K_M) | as I2V, with the FLF2V diffusers folder's image encoder | transformer 1.7e-4; first-and-last-frame video 50.8-60.2 dB per frame | `--image` and `--last-image` | 2026-11-16 |

## 9. Known unsupported or not working

Popular models tried that do not work today, with the reason. These are tracked; most are a matter of days of work, not of principle.

| Model | Hub repository | Reason |
|---|---|---|
| Inkling 2 Small | thinkingmachines/Inkling-2-Small, unsloth/Inkling-2-Small-GGUF | new `inkling2` blocks not implemented |
| DeepSeek V4.1 Flash (HF) | deepseek-ai/DeepSeek-V4.1-Flash | Hugging Face tensor-name mapping missing (the GGUF files import) |
| Voxtral Mini 4B Realtime | mistralai/Voxtral-Mini-4B-Realtime-2602 | streaming (delayed-token) decoder; the offline Voxtral Mini 3B is supported (section 2) |
| Cohere Transcribe 03-2026 | handy-computer/cohere-transcribe-03-2026-gguf | `cohere_asr` (conformer encoder with an autoregressive decoder) not implemented |
| Nemotron 3.5 ASR streaming | handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf | cache-aware streaming FastConformer: causal convolutions with layer-norm, chunked-limited attention context; the file loads as `parakeet` and is rejected with that reason |
| faster-whisper (CTranslate2) | Systran/faster-whisper-* | CTranslate2 `model.bin` format has no importer (the same models import from the OpenAI checkpoints or whisper.cpp / transcribe.cpp files) |
| Whisper ONNX (Transformers.js layout) | Xenova/whisper-* | the encoder and decoder graphs run through the ONNX runner, but there is no speech-recognition task on top of ONNX graphs |
| OmniVoice, VoxCPM2, VibeVoice, Qwen3-TTS, CosyVoice 3, MeloTTS, SpeechT5, Indic Parler-TTS, KittenTTS (ONNX) | k2-fsa/OmniVoice, openbmb/VoxCPM2, microsoft/VibeVoice-1.5B, Qwen/Qwen3-TTS-*, FunAudioLLM/Fun-CosyVoice3-0.5B-2512, myshell-ai/MeloTTS-English, microsoft/speecht5_tts, ai4bharat/indic-parler-tts, KittenML/kitten-tts-nano-0.8-fp32 | text-to-speech architectures without an adapter (autoregressive acoustic LMs with codec or flow-matching decoders, SpeechT5's encoder-decoder with HiFi-GAN, StyleTTS-style ONNX graphs); not downloaded, the Hub metadata names architectures the registry does not know |
| SenseNova-Loop, LiquidAI d1 | sensenova/SenseNova-Loop-9B-Preview, LiquidAI/d1-2B | layer loops and hyper-connections; non-causal decision variant |
| Unlimited-OCR-2, KLM-1.6, Raon-1-nano | baidu/Unlimited-OCR-2, MinimaLabs/KLM-1.6, KRAFTON/Raon-1-nano | new attention variants not implemented |
| Dory 1.0, ERNIE-5.5-Flash-Base, LongCat-2.5 | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, baidu/ERNIE-5.5-Flash-Base, meituan-longcat/LongCat-2.5 | no modeling code published, cannot be verified |
| MiniMax M3.1 Flash | MiniMaxAI/MiniMax-M3.1-Flash | MXFP8 weights not read |
| MiMo V2.6 Pro RL | XiaomiMiMo/MiMo-V2.6-Pro-RL | FP8 scale layout of the fused `qkv_proj` not read |
| Nemotron 3.5 Super-VL and Bolt (ModelOpt NVFP4 / mixed-precision safetensors) | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4, nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB | ModelOpt mixed-precision layout not read (GGUF versions import) |
| DFlash drafters with LiLiCorr head | Bolt and Super-VL drafters | correction head not implemented (reported as draft models) |
| Gated repositories | meta-llama/Llama-3.1-8B-Instruct, google/medgemma-1.5-4b-it, nvidia/Cosmos-Reason2-2B | could not be downloaded without a token; not tested |
| SD 1.5 inpainting UNet, SDXL refiner, Qwen-Image-Edit, Qwen-Image 2.1 editing, Lumina-Image-2.0, HiDream, AuraFlow, FLUX.2 Dev, Krea-2, MiniMax-H3 | various | not implemented yet (rejected with a clear message) |
| ControlNet, TAESD, LyCORIS / DoRA LoRAs | | not implemented (plain LoRAs: section 4) |
| CSM-1B, Dia-1.6B, Qwen3-TTS, PocketTTS, OuteTTS 0.3 | | text to speech, not implemented |
| Mask2Former | | vision model not implemented (Florence-2, SegFormer, BiRefNet, YOLO detection and segmentation are native task models, see model-support.md section 14) |
| Demucs, CLAP, MusicGen | | audio models not implemented |
| Mamba / RWKV real checkpoints beyond those above | | not covered by a model test |
| Candle-format whisper GGUF | oxide-lab/whisper-large-v3-turbo-q4_0.gguf, xkeyC/model_q4_k.gguf | rejected on purpose; use the whisper.cpp or transcribe.cpp files |
| Files without `general.architecture`, shards missing, `-noPLE` Gyro files | MiniMax-H3 pruned, some imatrix / patch files | malformed or incomplete at the source |

Detailed numbers and the full per-repository sweep table: [model-support.md](model-support.md).
