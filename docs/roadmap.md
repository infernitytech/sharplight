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
  GEMM (~280 of ~520 TFLOPS without global loads, §20.2, §22.10). The aligned split FP8 layout gives the GEMM 2–7 %
  (274–297 TFLOPS on the 27B shapes at 512 rows, §23.2); that is within the noise end to end.
- **Skinny GEMM (2–16 rows)**: on NVIDIA (`matrix_mma_layout`), operands now load straight into the MMA fragments
  (§23.1). FP8 gate/up runs at 1.38 TB/s and [17408→5120] at 1.24 TB/s at 16 rows. The 16-row step dropped from 34.3 to
  29.6 ms (FP8) and from 33.7 to 29.3 ms (Q8_0), against vLLM's 26.2 ms mean ITL at 16 streams. Next:
  - close the remaining gap to f16's ~1.65 TB/s;
  - fold the small [5120→48] split-K matmuls (~0.9 ms per step);
  - make Q8_0 at 5–8 rows use the direct GEMM (10–13 % faster) without changing greedy MTP output (§23.3);
  - re-measure serving against vLLM.
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
- **ONNX operator coverage**: control flow (Loop, If, Scan), sequences and the rest of the opset.
- **ONNX latency parity with onnxruntime** (§21, §25). Done:
  - conv / matmul epilogues on CPU and Vulkan, prepacked weights, decomposed LayerNorm / GELU and attention-scale
    translation, block-quantized MatMulNBits;
  - the 128x128 SIMT GEMM, bmm split-K, the CPU conv as one implicit GEMM (with a shifted stride-1 form);
  - Vulkan NMS / RoiAlign / MaxPool indices / NonZero;
  - the reduced-precision device loss and its batch-1 slowness.

  Open gaps:
  - Vulkan f32 GEMMs against ORT's TF32: vit, whisper, bert b8 and raft at 1.6-2.6x;
  - no flash-style fusion of the decomposed attention (MatMul → Softmax → MatMul) on either provider;
  - CPU convolution at 1.4-2x behind ORT on resnet50: NHWC / blocked layouts with propagation, Winograd;
  - CPU GEMM at ~200 rows (vit) and CPU attention for batched encoders;
  - detection models round-trip each data-dependent count through the host (R-CNN: 100 ms wall, 34 ms GPU).
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

- Single-pass LayerNorm, polynomial `exp`. Conv / MatMul epilogues (§21) and the attention-scale fold (§25) are done.
- Conv GEMM: one implicit GEMM over N·H·W with in-task packing and a shifted stride-1 form reaches 1.6-2.2 TFLOPS per
  conv (§25). Next: NHWC / blocked layouts chosen per op with layout propagation, Winograd F(2,3) for 3x3 stride 1 behind
  an accuracy check, and cheaper b1 pooling / global-mean ops.
- 128-bit GEMM kernels for CPUs without AVX-512; ARM64 (NEON / SVE) kernels.

## Precision options

Defaults keep near-f32 accuracy, except that matmuls over 2+ activation rows multiply f16 activations on tensor cores
(E4M3 with block scales for FP8 weights from 17 rows, as vLLM does); single-row decode stays on the f32 matvec (§19).
`--reduced-precision` (`LlmLoadOptions.AllowReducedPrecision`,
`CompileOptions.AllowReducedPrecision`) opts in to faster f16 paths, e.g. one f16 product instead of two in the chunked
delta rule: about 3 % of 27B prefill, 10 % at FP8 with 512-token prompts, and no difference in greedy output in the tests
of §18. It also stores the gated delta rule's recurrent state in bf16 (`LlmLoadOptions.RecurrentStateType`, diagnostic
override `SHARPLIGHT_RECURRENT_STATE=f32|bf16|f16`): the token-by-token kernel rounds the state after every row (so MTP
verification still matches plain decoding), the chunked prefill once per run. On the 27B FP8 it halves the `delta_rule`
kernel (226 → 108 us per layer at 64 decode rows) and cuts the step GPU time by about 3 % at 16 rows and 9 % at 64. f16 is
more accurate but its integer round-to-nearest-even conversion eats most of the gain (−1.4 % at 64 rows). Open: rounding the
chunked form per chunk instead of per run, and 16-bit states for the other recurrences (`SelectiveScan`, `Wkv6/7`).

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
