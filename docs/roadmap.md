# Roadmap

Goals: run every architecture llama.cpp supports, match or beat llama.cpp and vLLM on the same hardware, and cover the
full ONNX operator set. Optimizations stay generic (op and kernel selection, not per model); hardware paths are device
features with a fallback and a disable switch; anything consumers see goes through the provider abstraction.

Numbers below refer to sections of [benchmarks.md](benchmarks.md).

## In progress

- **Prefill against vLLM on identical weights** (§38-§40, same HF bf16 / FP8 checkpoints on both sides): prompts prefill in
  passes of up to 2048 rows and the server prefills a prompt in whole passes; round 4 (§39) added a grouped tile order for
  large-weight GEMMs, a dual bf16 gate / up GEMM and a register softmax in the prefill attention; round 5 (§40) converts bf16
  weights after the MMAs instead of at the fetch (bf16 GEMMs +10-14 %) and lets the dual GEMM store the down projection's f16
  input. pp2048 is now at 1.02x (Qwen3-0.6B), 1.02x (Llama-1B), 0.90x (Qwen3-8B bf16) and 0.74x (FP8) of
  vLLM; pp8192 at 0.91x / 0.93x / 0.89x / 0.74x (same statistic on both sides, median of 5: pp2048 1.05x / 0.98x / 0.87x /
  0.72x; GGUF against llama.cpp 1.65x / 1.02x / 0.93x for Qwen3-0.6B Q8_0 / Llama-1B Q8_0 / Qwen3-8B Q4_K_M, §40.9). Decode is level.
  Round 6 (§42): per-model gap table with the identical procedure (Qwen3-8B bf16 0.87-0.89x, Llama-3.1-8B bf16 0.86-0.89x,
  Qwen3-8B-FP8 0.71-0.74x of vLLM; Llama-3.1-8B Q4_K_M 0.91x → 0.97x and Qwen3.8-27B Q8_0 0.94-1.00x of llama.cpp; decode
  1.00-1.05x); the dual gate / up GEMM now serves quantized weights; stream-K in one launch was built and measured (no gain
  end to end, not kept); the FP8 GEMM is bound by L2 fetch bandwidth against MMAs at ~335 TFLOPS (§42.3). Next: the FP8 and
  bf16 GEMM fetch (bigger tiles with less L2 traffic per FLOP), norms writing the f16 / E4M3 activations at prefill sizes,
  the K/V staging of the attention kernel, Gemma-3 from Hugging Face.
  Round 7 (§44): row norms write the GEMMs' f16 / E4M3 activations from 128 rows and skip their f32 rows when every reader takes
  the copy (Qwen3-8B-FP8 pp2048 +3 %, bf16 +1.6 %, Q4_K_M +0.8 %, greedy output byte-identical); against §42's vLLM figures
  Qwen3-8B bf16 0.91-0.92x, Llama-3.1-8B 0.91-0.92x, Qwen3-8B-FP8 0.76-0.78x. Gemma 3 vision-language checkpoints import as their
  text decoder (gemma-3-12b-it matches transformers). Separate K/V staging in `fe` was slower (it runs at 255 registers, one
  workgroup per SM; staging is 23 % of its time); bigger FP8 / bf16 GEMM tiles do not fit 48 KB of shared memory and 255
  registers (§44.3). Next: attention at 8192 tokens (20 % of the FP8 pass), the down projection's E4M3 input, a vLLM rerun.
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
  - serving against vLLM (§28.6): 0.81x / 0.82x / 0.69x / 0.77x at 1 / 4 / 16 / 64 streams (bf16 state opt-in: 0.83x /
    0.85x / 0.71x / 0.80x); mixed decode + prompt steps are 70 % fp8 GEMM at 100-500 rows and 20 % delta rule (§28.4).
  - sustained serving steps on the 300 W Max-Q card run at the power cap (SM clock 1.6-1.8 GHz, §29.2): narrow fp8
    tiles that were 5-24 % faster in micro-benchmarks gave no end-to-end gain (§29.1). Prefer changes that move fewer
    bytes or do fewer operations over wave-fill and occupancy tuning, and confirm every kernel win on full steps.
  - the f32 delta-rule state already streams at the memory bandwidth (§29.4); only the opt-in bf16 state makes it cheaper.
- **Reduced-precision delta-rule state**: done as an opt-in (`--reduced-precision` stores it as bf16; 64-row step
  -8.6 %, 16-row -2.7 % on the 27B FP8, §24.6). Next: round the chunked prefill per chunk like the token-by-token
  kernel, and 16-bit states for Mamba / RWKV.
- **Fewer glue launches per batched step**: row norms now write the f16 activation copy for 2-16-row steps (1334 → 1158
  dispatches, 917 → 789 barriers at 16 rows on the 27B FP8; Blackwell GPU time unchanged, §28.3). Left: the copies after
  elementwise and matmul epilogues (attention gate, SwiGLU output), fp8 quantization and split-K reductions.
- **New card baseline** (§34, RTX PRO 4500 Blackwell 32 GB): decode 0.88-0.99x of llama.cpp CUDA, dense prefill
  0.71-0.85x, MoE / hybrid prefill 0.95-1.12x, serving 1.1-4.4x llama-server and ahead of vLLM except Llama-1B at 64
  streams (0.75x). Two prefill changes that alter greedy output are now the default (§34.5, decided 2026-11-15): K splits
  from the SM count (pp2048 +2-17 %, `SHARPLIGHT_GEMM_WAVE=fixed` for the old splits) and the causal `fe` flash kernel
  (pp2048 +6-38 %, `SHARPLIGHT_ATTN_FE_CAUSAL=0` for `fa`); together Qwen3-0.6B goes 0.71x → 1.09x of llama.cpp CUDA.
  Round 2 (§36, output unchanged): 4-warp GEMM tiles at every grid size and cheaper K-quant fetches (pp2048 Qwen3-8B
  0.82x → 0.93x, Mistral-7B 0.88x → 0.98x of llama.cpp CUDA), split-K decode attention merged in the last workgroup (one
  launch and barrier less per layer for narrow head groups). Next: a deeper GEMM fetch pipeline (the Q4_K GEMM is bound by
  weight-load latency at two workgroups per SM), rope / q-k norm / cache writes inside decode attention (Qwen3-0.6B decode
  0.80x; barriers are ~30 % of its step), batched expert matmuls at 16-64 rows.
- **Single-stream gaps** (§28.5): 27B Q8_0 decode (0.90x of llama.cpp; single matvecs at 1.34-1.42 TB/s) and Mistral-7B
  Q4_K_M pp512 (0.91x). Both need a different summation order, i.e. a deliberate change of greedy output.
- **Row-invariant MTP verification** (§28.1): verification rows now reproduce plain decoding bit for bit. Next: a
  row-invariant skinny GEMM so k = 3 on FP8 regains its 5-8 %, and bit-exact int-dot variants for K-quant weights.
- **MTP follow-ups** (§17): share the embedding / LM-head buffers between the target and the draft head (2.5 GB on the
  27B); verify drafts on the device instead of reading back 1 MB of logits per row; count rollback records and the draft
  head in `EstimateFootprint`; wire up DeepSeek-V3 and GLM-4.5 NextN heads.
- **Chunked delta rule in rollback-form graphs**: with speculation on, prompt prefill uses the token-serial kernels and
  loses the 5–25 % prefill gain of §18. Replay the committed rows, then hand long runs to the chunked kernel.
- **ONNX operator coverage**: control flow (Loop, If, Scan), sequences and the rest of the opset.
- **ONNX latency parity with onnxruntime** (§21, §25-§27, §30, §31, §33). Done:
  - conv / matmul epilogues on CPU and Vulkan, prepacked weights, decomposed LayerNorm / GELU and attention-scale
    translation, block-quantized MatMulNBits;
  - the 128x128 SIMT GEMM, bmm split-K, the CPU conv as one implicit GEMM (with a shifted stride-1 form);
  - Vulkan NMS / RoiAlign / MaxPool indices / NonZero, CumProd;
  - the reduced-precision device loss and its batch-1 slowness;
  - fused encoder attention (decomposed MatMul → Softmax → MatMul and com.microsoft Attention / MultiHeadAttention)
    on the CPU tiled kernel and a Vulkan flash kernel with split-f16 products in strict precision;
  - the split-f16 tensor-core GEMM and conv at f32 accuracy (row / block scales, promoted tensor-core sums) as the
    strict default for f32 weights (`SplitHalfMatMul`, §30);
  - CPU conv task planning by operand traffic, subsampled strided 1x1 convs, row-form max pooling, core-affine
    parallel loops; fused sibling (Q / K / V) projections; Vulkan Det / Inverse / volumetric GridSample / int64
    CumSum (§27);
  - CPU channels-last (NHWC) convs, depthwise convs and pooling with layout propagation; Vulkan attention over strided
    K / V; data-dependent counts grouped by level with device-side forwarding (R-CNN wall 119 → 12 ms);
    LinearAttention / DeformConv / Unique / TfIdf / tree ensembles / SVM on Vulkan; AggressiveOptimization on the
    CPU kernels for package consumers (§30);
  - elementwise groups in the split-f16 GEMM's store, 64-token split tiles for few hundred rows, a reworked encoder
    flash kernel, in-place LayerNorm over channel-strided views, conv_narrow for few output channels, sub-millisecond
    re-recording for new detection counts; for devices without matrix units an f32 encoder flash kernel and tiles
    sized by `DeviceInfo.ConcurrentWorkgroups` (integrated GPU: whisper-tiny 16x, vit-base 3.2x) (§31);
  - opt-in fast matrix precision (TF32 class) for GEMMs, convs and attention: 0.60-1.00x of ORT's TF32 default on the
    encoders and convnets (raft 1.04x); ScatterND over a symbolic axis on Vulkan (qwen2.5 q4 in one segment); phi-3.5 parity against
    an f32-copy reference; conformance 1863 / 1884 node and 171 / 172 extra tests (§33).
  - on the RTX PRO 4500 against onnxruntime 1.32 (§35): fast precision 0.46-0.94x of ORT's TF32 default on every
    model but the phi-3.5 decoder step, strict 0.06-0.95x of ORT strict; decomposed attention that round-trips its
    probabilities through Reshapes is fused; concatenation inputs are computed in their output slices (SegFormer b8
    GPU time -7 %); the ssd-12-int8 reference follows the spec (ORT's u8 x s8 kernels saturate on AVX2 without VNNI).
  - convs against TensorRT (§37): tap-major implicit-GEMM reduction (one address per 8-channel item), im2col scaling after
    the previous step's products, fast-precision block scales kept across K steps (products accumulate on the tensor
    cores), narrow convs (K, channels from 16) on the tensor cores, 4 outputs per thread in wide depthwise convs; Q4_0
    group decode with 16-bit loads (phi-3.5 q4f16 decode step at ORT CUDA's level); Vulkan Unique along an axis (the
    three remaining Vulkan conformance failures); ssd-10's strict parity gap verified as tie order (all 200 detections
    match, two near-tied boxes swap).

  Open gaps:
  - strict Vulkan against ORT's TF32 on the RTX PRO 4500 (§37): vit 1.7-1.8x, convnext b8 1.19x, whisper 1.03x,
    resnet50 b8 1.04x (fast precision closes all of them). The split GEMM's three products cost about 1.5x by
    construction; the activation split is a pre-pass that the producing LayerNorm / GELU could write (needs a row max);
  - TensorRT against fast precision (§37): convnext 1.38x (LayerNorm, the split pre-pass, a residual add through a
    transpose), mobilesam 1.31x, ssd-10 1.20x, yolov10n 1.13x, mobilenetv2 b8 1.09x. The fast 3x3 conv reaches 45-55
    TFLOPS against the GEMM's 96: next are a double-buffered conv pipeline and NHWC activations;
  - phi-3.5 q4f16 decode at 1.04x of ORT CUDA: the f32-activation Q4_0 matvec reads about 520 GB/s; f16
    MultiHeadAttention does not take the fused attention (f32 only), rotary runs as four elementwise launches;
  - batch-strided concatenation slices only from Resize (convs / GEMMs / elementwise producers would need an output
    batch stride);
  - convnext b8 1.5x strict (low-K GEMMs, split-K reductions); raft 1.25x strict / 1.04x fast (SIMT bmm for the
    correlation volume, about 290 small concats); Faster / Mask R-CNN on the CPU backend 2.1-2.3x of ORT CPU;
  - `fes` for head sizes above 64; a compute-unit query (VK_NV_shader_sm_builtins / VK_AMD_shader_core_properties)
    instead of the per-device-type `ConcurrentWorkgroups`; qwen2.5 fp16 on Vulkan 6x the CPU backend's error;
  - CPU Winograd on the channels-last layout for 64 / 128-channel 3x3 convs; encoder GEMMs stream each layer's weights
    from DRAM (vit b1 1.1x, resnet50 b8 1.05x).
- **Sharplight Studio**: desktop app (Avalonia) for model search, download, conversion, chat and serving.

## Performance ideas

GPU (Vulkan):

- **Faster FP8 prefill GEMM**: 27B FP8 prefill is at 0.83x / 0.72x of vLLM at 512 / 2048 tokens (§22.2). A
  workgroup-scope matrix (`VK_NV_cooperative_matrix2`) version was 1.6x slower than `gemm_fp8` (§24.4); the fetch
  needs a deeper kernel-managed pipeline within the 48 KB of shared memory Vulkan exposes.
- **K-quant MMQ**: the integer-dot path for Q4_K / Q5_K / Q6_K is still 6 % slower than the f16 path on the 27B
  Q4_K_M (§24.4) and stays opt-in; any int8 path changes greedy output against the f16 default. 27B Q4_K_M prefill
  is still 2–4 % behind llama.cpp CUDA (§18).
- **MoE expert GEMMs**: gpt-oss is at 0.95x / 1.10x of llama.cpp CUDA at pp512 / pp2048 and 0.97x in decode (§24);
  the two split-tile launches no longer wait on a barrier (pp512 +10 %, §28.2).
  Left: the expert GEMM's weight fetch (64-row tiles still about 65 % full before the split launch), and the
  dual gate/up and slot-sum fusions for the batched expert GEMM (they cover the int-dot matvec only).
  Serving (§43): 16-64 sequences now take the grouped GEMM from half the expert count of items (16 streams +48-79 %); against
  vLLM on gpt-oss-20b HF Sharplight leads at 1 stream and is at 0.62-0.76x from 4 streams on. Next: MoE steps that mix
  prefill chunks with decode rows (29 ms per 171-row step against 18.5 ms for 64 decode rows), a dual gate / up expert GEMM
  with the activation epilogue, and Qwen3.8-Flash-Next with CPU offload against llama.cpp (not measured, §43.5).
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

`CompileOptions.MatrixPrecision = MatrixPrecision.Fast` (`SHARPLIGHT_MATRIX_PRECISION=fast`, OnnxBench `--fast-precision`)
is the opt-in TF32-class mode for f32 GEMMs, convs and encoder attention on Vulkan (§33). It uses one tensor-core product
of row-scaled 11-bit inputs per multiply-add instead of the strict three-product split. Its error against an f64
reference is 5.6-6.8e-4 (ORT TF32: 5.5-7.9e-4). It is 1.1-2.2x faster than strict on encoders, and matches or beats ORT's
TF32 default on vit, whisper, bert, convnext and resnet50. `AllowReducedPrecision` takes precedence when both are set.
Devices without f16 matrix units keep strict. Open: the q4 decoders and the SIMT bmm (raft) do not use it yet.

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
