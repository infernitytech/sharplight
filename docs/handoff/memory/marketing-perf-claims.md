---
name: marketing-perf-claims
description: Owner's marketing positioning for performance (2026-11-21): claim parity with vLLM, promote the large leads only
metadata:
  type: project
---

Owner decision 2026-11-21: marketing claims **speed parity with vLLM** and promotes only the **large, measured leads**. There is no "10% faster than vLLM across the board" claim.

The leads as of 2026-11-21 (benchmarks §66–§72):
- Big MoE with CPU expert offload vs llama.cpp: gpt-oss-120b decode 1.45–1.5x and prefill 2.7–2.8x; Qwen3.5-122B on Vulkan 1.2–1.4x; Qwen3-Coder-Next on Vulkan 1.5–1.7x; 122B on CUDA 1.04–1.10x (§71). vLLM has no comparable mode.
- Long context vs vLLM: roughly even. Like-for-like (§73), gpt-oss decode at 32K is 1.06x / 1.05x and at 128K 1.00x / 0.93x (1 / 4 streams). It is not a large lead. §78 (r26, depth 130,048 both engines): 4 streams 1.00–1.03x, 1 stream 1.05x, 32K 1.08/1.10x (1/4 streams); the old 0.93x was a noisy-method artifact.
- Time to first token vs vLLM: 266 vs 413 ms at 64 streams.
- Full-GPU GGUF Q4_K_M vs llama.cpp b11177 (§74, 2026-11-21): Vulkan batched decode 1.7–2.5x (Gemma 4 31B, Qwen3.8 27B, Qwen3-8B); Vulkan prefill 1.10–1.27x; Qwen3.8 27B prefill 1.11–1.19x on both providers, CUDA batched decode 1.17/1.55x (16/64 seq). Single-stream decode is parity, not a lead. §77 (feed-r2): CUDA batched decode vs llama.cpp Qwen3-8B 1.37–1.41x (16–64 seq), Qwen3.8 27B 1.67/2.04x (16/64), Gemma 4 31B 1.11x (16); Gemma 4 31B 32K fits a 32 GB card (21.1 GiB, 128K 28.2 GiB) with 32K prefill 1.04x CUDA / 1.21x Vulkan. §79 (feed-r3): CUDA batched decode 16–64 seq Qwen3-8B 1.37–1.43x, 27B 1.70–2.05x; 48 seq Vulkan Qwen3-8B 1.81x. §81 (feed-r4): long-context decode on Vulkan vs llama.cpp Vulkan: Qwen3.8 27B 1.14x @128K / 1.12x @32K, Gemma 4 31B 1.14x @32K / 1.06x @128K; Gemma Vulkan pp512 1.19x, 16 seq 2.06x. CUDA decode at depth 1.01–1.05x = parity. §83 (feed-r5): Vulkan batched decode vs llama.cpp Vulkan Qwen3-8B 2.93/2.30/2.02/1.79x (16/32/48/64 seq), Qwen3.8 27B 3.45–2.62x, Gemma 4 31B 2.65x @16; q8_0 KV 27B @32K 1.39x CUDA / 1.15x Vulkan. vs vLLM on GGUF-equivalent int4 checkpoints we trail (0.62–0.91x): do not claim.

- Huge models streamed from disk (owner, 2026-11-22): once a huge MoE (DeepSeek-V3.1, Kimi K3, GLM-5.x) is verified correct against a reference, promote it, e.g. "runs DeepSeek / Kimi / GLM on a single 32 GB GPU + 62 GB RAM". Quote measured tok/s and the llama.cpp comparison under the same limits (DeepSeek-V3.1 ~1.0 vs 0.62 tok/s, §85). Do not promote it before the correctness check passes ([[experts-on-disk]]). §87 verified: DeepSeek-V3.1 Q4_K_M 405 GB 0.97 vs 0.62 tok/s; GLM-5.2 UD-Q2_K_XL 254 GB median 0.99 (cold 0.41) vs 0.55 tok/s. DeepSeek-V4-Flash UD-IQ1_S verified on CUDA 2026-11-23 (expert-cache double-registration fix): 7.2 tok/s cold vs llama.cpp 7.15, warm median 12.1 (1.7x); first prompt ~3x slower than llama.cpp (multi-row experts on CPU, open item); Kimi K3 is unverified (truncated only, 0.08 tok/s).

**Why:** close to vLLM on its best formats (0.92–0.99x) with diminishing returns on a 200 W-capped card. Parity is credible; a broad lead is not.

**How to apply:**
- Every published number must be like-for-like and reproducible: same hardware, `--feedback`/`--text` decode, same-session vLLM, scripts in bench/. See [[benchmarking-shared-gpu]].
- Recheck the gpt-oss-120b lead without page-cache overflow before publishing it, since llama.cpp was slowed by it.
- Perf work still targets ≥1.0x everywhere ([[project-goals-parity]]).
- 2026-11-23 state (§95–§98, new defaults): serving vs vLLM 0.33 — gpt-oss-20b 0.97–1.02x (16) / 1.00–1.03x (64), Qwen3-8B-FP8 0.99–1.00x both; TTFT ~half of vLLM's. ONNX CUDA beats ORT CUDA EP on every row (phi-3.5 0.93x); mixed vs TensorRT. ONNX CPU 0.62–1.53x ORT CPU.

2026-11-24 (round 41): from r29 to r40, batched Qwen3-8B-FP8 on CUDA generated garbage text (gemm_fp8_decode read activations with __ldg while launched as a dependent kernel). Fixed in r41 by reading through L2 (__ldcg). Serving speed with correct text is unchanged (1100/2731 tok/s vs vLLM 1119/2699 at 16/64 streams). No release tag contains the bug. Rule from now on: a serving perf number counts only when the batched text check passed (added in r42).
