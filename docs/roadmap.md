# Roadmap

Goals: run every architecture llama.cpp supports, match or beat llama.cpp and vLLM on the same hardware, and cover the
full ONNX operator set. Optimizations stay generic (op and kernel selection, not per model); hardware paths are device
features with a fallback and a disable switch; anything consumers see goes through the provider abstraction.

Numbers below refer to sections of [benchmarks.md](benchmarks.md).

## In progress

- **Per-row prefill cost (serving at 16–64 streams)**: a serving step costs the decode step plus ~0.24 ms per prompt row
  (vLLM ~0.17 ms), so throughput and TTFT at 16–64 streams follow the prefill kernels, not the chunk / budget settings.
  The 27B FP8 serves 16 / 64 streams at 0.63x / 0.68x of vLLM (§20.1, §20.5). The dual gate/up fp8 GEMM now also runs
  above 64 rows where it is bit-identical to the separate launches (pp2048 +3 %, §22.2). Next: a fetch-efficient fp8
  GEMM (~280 of ~520 TFLOPS without global loads, §20.2, §22.10).
- **Skinny GEMM (2–16 rows)**: retuned to 1.2–1.55 TB/s at 4 rows and 0.95–1.1 TB/s at 16 rows (§22.1); FP8 steps of
  2–8 rows are 6–11 % faster. The 16-row step (33.1 ms against vLLM's 24 ms ITL at 16 streams) is dominated by the dual
  gate/up GEMM; next is keeping x in registers across fragments, or a cross-workgroup K split with the reduction fused
  into the consumer.
- **Reduced-precision delta-rule state** (opt-in, `--reduced-precision`): an f16 / bf16 recurrent state for the gated
  delta rule, about a quarter of the 64-row decode step (§19.6). Needs the state type through the op, its CPU
  reference, both kernels, rollback records and snapshots (§22.10).
- **Fewer glue launches per batched step**: f16 activation copies, fp8 quantization and split-K reductions are 3–7 % of
  GPU time. Produce the copy in the preceding norm / elementwise epilogue (§19.6).
- **MTP follow-ups** (§17): share the embedding / LM-head buffers between the target and the draft head (2.5 GB on the
  27B); verify drafts on the device instead of reading back 1 MB of logits per row; count rollback records and the draft
  head in `EstimateFootprint`; wire up DeepSeek-V3 and GLM-4.5 NextN heads.
- **Chunked delta rule in rollback-form graphs**: with speculation on, prompt prefill uses the token-serial kernels and
  loses the 5–25 % prefill gain of §18. Replay the committed rows, then hand long runs to the chunked kernel.
- **ONNX operator coverage**: control flow (Loop, If, Scan), sequences, NonMaxSuppression and the rest of the opset.
- **ONNX latency parity with onnxruntime** (§21): conv / matmul epilogues, prepacked weights, decomposed LayerNorm and
  GELU translation and block-quantized MatMulNBits are in. Open gaps:
  - CPU convolution GEMM efficiency (resnet50, segformer, mobilesam);
  - Vulkan f32 GEMMs at 200-1600 rows against ORT's TF32 (vit, whisper);
  - NonMaxSuppression and RoiAlign on the GPU (CPU fallback segments today);
  - reduced-precision mode, which is slower than the default at batch 1 and loses the device on two encoders.
- **Sharplight Studio**: desktop app (Avalonia) for model search, download, conversion, chat and serving.

## Performance ideas

GPU (Vulkan):

- **Faster FP8 prefill GEMM**: 27B FP8 prefill is at 0.83x / 0.72x of vLLM at 512 / 2048 tokens (§22.2).
- **K-quant MMQ**: the integer-dot path for Q4_K / Q5_K / Q6_K only ties the f16 path today and stays opt-in (§13).
  27B Q4_K_M prefill is still 2–4 % behind llama.cpp CUDA (§18).
- **MoE expert GEMMs**: gpt-oss prefill is at 11079 tok/s against a 17432 target (§14). gpt-oss decode is at 0.93x of
  llama.cpp CUDA after the matmul-bias fusion (§22.3); a fused MoE gate/up for decode is next.
- **Chunked delta rule, split form**: the sequential part now waits between steps rather than on tensor cores. FLA's
  split (short sequential state pass plus a parallel output kernel) needs 100–400 MB scratch for an estimated 1–2 % of
  prefill (§18).
- **Chunked scans for other recurrences**: Mamba-2 is a simplified case of the chunked delta-rule kernel; RWKV-7 needs a
  formulation for its per-channel decay (§18).
- **ONNX `LinearAttention`**: lower to `GatedDeltaRule` with g = 0, which gives it the chunked kernel.
- **Device-side speculative verification for sampled decoding**, not only greedy.

CPU:

- Single-pass LayerNorm, polynomial `exp`; fold the attention scale (`Div` after Q·Kᵀ) into the BatchMatMul alpha.
  Conv / MatMul bias, residual and activation epilogues are done (§21).
- Conv GEMM: the 8x32 microkernel reaches roughly 10-15 % of the 9950X's f32 peak inside ONNX CNNs. Next: pack A (the
  weights) once per conv, block over output channels per thread, and run batch > 1 as one GEMM over N·H·W (§21).
- 128-bit GEMM kernels for CPUs without AVX-512; ARM64 (NEON / SVE) kernels.

## Precision options

Defaults keep near-f32 accuracy, except that matmuls over 2+ activation rows multiply f16 activations on tensor cores
(E4M3 with block scales for FP8 weights from 17 rows, as vLLM does); single-row decode stays on the f32 matvec (§19).
`--reduced-precision` (`LlmLoadOptions.AllowReducedPrecision`,
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
