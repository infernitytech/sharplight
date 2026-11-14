# Roadmap

Goals: run every architecture llama.cpp supports, match or beat llama.cpp and vLLM on the same hardware, and cover the
full ONNX operator set. Optimizations stay generic (op and kernel selection, not per model); hardware paths are device
features with a fallback and a disable switch; anything consumers see goes through the provider abstraction.

Numbers below refer to sections of [benchmarks.md](benchmarks.md).

## In progress

- **Small-batch FP8 / bf16 decode (2–16 rows)**: read each weight once for all rows, as the Q8_0 path does. The 27B FP8
  step costs 24/39/62/118 ms at 1/2/4/8 rows against Q8_0's 24/20/24/29 ms; this is the 0.38x serving gap at 4 streams
  and why MTP does not pay on FP8 yet (§16, §17).
- **9–64-row GEMM at full memory bandwidth**: 0.7–0.93 TB/s today against 1.61 TB/s peak, for 16 and 64 streams (§16).
- **Fewer barriers and tiny launches per batched step**: about 2000 dispatches and 1650 barriers per step; f16
  activation copies and split-K reductions take 7–19 % of GPU time. Fuse them into neighbouring kernels (§16).
- **MTP follow-ups** (§17): share the embedding / LM-head buffers between the target and the draft head (2.5 GB on the
  27B); verify drafts on the device instead of reading back 1 MB of logits per row; count rollback records and the draft
  head in `EstimateFootprint`; wire up DeepSeek-V3 and GLM-4.5 NextN heads.
- **Chunked delta rule in rollback-form graphs**: with speculation on, prompt prefill uses the token-serial kernels and
  loses the 5–25 % prefill gain of §18. Replay the committed rows, then hand long runs to the chunked kernel.
- **ONNX operator coverage**: control flow (Loop, If, Scan), sequences, NonMaxSuppression and the rest of the opset.
- **Sharplight Studio**: desktop app (Avalonia) for model search, download, conversion, chat and serving.

## Performance ideas

GPU (Vulkan):

- **Faster FP8 prefill GEMM**: 27B FP8 prefill is at about 0.55x of vLLM (§13, §16).
- **K-quant MMQ**: the integer-dot path for Q4_K / Q5_K / Q6_K only ties the f16 path today and stays opt-in (§13).
  27B Q4_K_M prefill is still 2–4 % behind llama.cpp CUDA (§18).
- **MoE expert GEMMs**: gpt-oss prefill is at 11079 tok/s against a 17432 target (§14).
- **Chunked delta rule, split form**: the sequential part now waits between steps rather than on tensor cores. FLA's
  split (short sequential state pass plus a parallel output kernel) needs 100–400 MB scratch for an estimated 1–2 % of
  prefill (§18).
- **Chunked scans for other recurrences**: Mamba-2 is a simplified case of the chunked delta-rule kernel; RWKV-7 needs a
  formulation for its per-channel decay (§18).
- **ONNX `LinearAttention`**: lower to `GatedDeltaRule` with g = 0, which gives it the chunked kernel.
- **Device-side speculative verification for sampled decoding**, not only greedy.

CPU:

- Relu fusion into the preceding GEMM, single-pass LayerNorm, polynomial `exp`.
- 128-bit GEMM kernels for CPUs without AVX-512; ARM64 (NEON / SVE) kernels.

## Precision options

Defaults keep near-f32 accuracy. `--reduced-precision` (`LlmLoadOptions.AllowReducedPrecision`,
`CompileOptions.AllowReducedPrecision`) opts in to faster f16 paths, e.g. one f16 product instead of two in the chunked
delta rule: about 3 % of 27B prefill, 10 % at FP8 with 512-token prompts, and no difference in greedy output in the tests
of §18.

## Multi-node inference

- **Pipeline parallel across a cluster**: run one model too large for a single machine across several nodes. Placement
  already splits a graph into per-device segments in pipeline order (`PlacementOptions.Devices`); a remote segment
  worker on each node loads only its own layers from its local `.slm` copy, keeps the KV cache of those layers, and
  exchanges only the boundary activations (`d_model` x 2 bytes per token, about 14 KB for DeepSeek-V3; well under 1 ms
  per token over 10 GbE for 4 nodes). Exposed as a provider whose devices are remote segments, so other providers can
  implement it. Single-stream decode gains capacity, not speed; throughput scales with nodes by keeping several
  micro-batches in flight through the multi-sequence decoder.
- Later: tensor and expert parallel across nodes for RDMA networks (InfiniBand / RoCE) and large MoE models; over TCP the
  per-layer all-reduces cost several milliseconds per token.

## Other

- `optimize` command for `.slm` packages.
- Device-profile open issues (offline compilation for devices not present).
- Known flake: an occasional `vkCreateDevice` failure when tests create devices in parallel.
