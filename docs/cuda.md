# CUDA provider

`Sharplight.Cuda` is a native execution provider for NVIDIA GPUs next to the Vulkan provider. It exists to close the performance gaps
to vLLM that the Vulkan path cannot close on NVIDIA hardware: Vulkan compute on the NVIDIA driver is limited to 48 KB of shared
memory per workgroup, exposes neither the Tensor Memory Accelerator (`cp.async.bulk`, TMA) nor `cp.async`, and has no thread-block
clusters (see [benchmarks.md §44.3](benchmarks.md#443-tried-and-not-kept) and [§46](benchmarks.md#46-llm-round-8-fresh-vllm-baseline-expert-grouped-skinny-gemm-for-moe-decode-moe-range-guard-2026-11-18)).
The FP8 GEMM tiles that would cut L2 traffic, the flash-attention tiles that would hide K/V staging and the MoE GEMMs all need one of
these.

Phase 1 (branch `cuda/provider-r1`) is the provider itself: every op of the shared kernel library runs on CUDA, correct and
driver-only, with the same lowering as Vulkan. Phase 2 (branch `cuda/provider-r2`) brings the tensor cores to CUDA (CUDA forms of the
cooperative-matrix fragments on `ldmatrix` / `mma.sync`), removes the active-mask tracking of Slang's CUDA output from the wave
operations, lowers the tile kernels for the opt-in shared memory above 48 KB, and adds cuBLASLt as an optional library GEMM. Phase 3
(branch `cuda/provider-r3`) adds kernels written in CUDA C++ that use what the Slang kernels cannot express
on CUDA ([kernels written in CUDA C++](#kernels-written-in-cuda-c)): an f16 GEMM that replaces cuBLASLt (bf16 prefill at vLLM's
level without the 541 MB library) and an FP8 GEMM with `cp.async` pipelines in 96 KB of dynamic shared memory, plus CUDA graphs for
repeated steps and a per-provider decode attention batch. CUDA is now ahead of Vulkan on every measured prefill of a dense model
(8B bf16 1.10-1.17x, FP8 1.05-1.08x) and on the decode of the 1B models (1.03-1.16x), at vLLM's level for bf16 (0.98-1.06x) without
cuBLASLt; FP8 prefill stays at 0.78-0.79x of vLLM ([phase 3 numbers](#phase-3-numbers)). Phase 4 (branch `cuda/provider-r4`) gives
the FP8 GEMM scale layouts it can stream densely (weights in 128-row groups repacked at load, a dense copy of the activation scales)
and 64-row tiles from 65 rows: FP8 prefill 0.85-0.87x of vLLM (+7-12 %), mixed serving steps 13-20 % faster, and measures the
batched decode step kernel by kernel against vLLM ([phase 4 numbers](#phase-4-numbers)).

## Requirements

- **Only the NVIDIA driver** on the user's machine: no CUDA toolkit, no `CUDA_PATH`, nothing on `LD_LIBRARY_PATH`. The provider talks
  to the driver API (`libcuda.so.1` / `nvcuda.dll`, installed with every NVIDIA driver) through `LibraryImport`; the CUDA libraries it
  needs beyond that are redistributables that Sharplight ships ([packaging](#packaging-and-licensing)).
- **Driver of the CUDA 13 branch, R580 or later** (CUDA 13.0, August 2025). The shipped NVRTC is 13.0; it emits PTX ISA 9.0, which
  every CUDA 13 driver compiles. A driver older than the PTX fails at kernel load with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, and the
  provider's message says to update the driver.
- **GPU of compute capability 7.5 or later** (Turing, RTX 20 / T4 and newer: the oldest architecture CUDA 13 compiles for). Older GPUs
  are not listed; they keep the Vulkan provider. Warp size 32 (every NVIDIA GPU).
- Platforms: linux-x64, linux-arm64 (SBSA: Grace, GB10 / DGX Spark, Jetson Thor), win-x64. Not macOS (no NVIDIA driver).

## Design

The provider follows the Vulkan and Metal providers exactly; only launch, memory and the last compilation step differ.

```
graph segment ──GraphLowerer──▶ KernelProgram (launches, buffers, args)       shared with Vulkan / Metal
      Slang kernel library ──Slang──▶ CUDA C++ ──NVRTC──▶ PTX  (portable code, code tier, kernel/<hash>)
                                                        PTX ──driver JIT (cuLink)──▶ cubin (device code, binary tier, cubin/<hash>)
                                                        cubin ──cuModuleLoadData──▶ CUfunction ──cuLaunchKernel on a stream
```

- **Kernels.** The Slang kernels compile to CUDA C++ unchanged (`ShaderTarget.CudaSource`); the kernel conventions were written with
  CUDA in mind (one `uniform` parameter struct: all pointers first, then 32-bit scalars, which becomes the kernel's single by-value
  parameter). CUDA C++ has no block-size attribute, so `SlangCompiler` writes the entry point's `[numthreads]` into a header line
  (`// sharplight-threads: x y z`, as for Metal) that the launch reads. A `KernelTranslator` hook in `SlangCompiler` (new, provider
  neutral, used only by CUDA) runs NVRTC on the Slang output before it is cached, so the cached and exported code of a kernel is its
  PTX: `NvrtcTranslator` compiles for the device's virtual architecture (`-arch=compute_XY`), with the floating-point headers passed
  in memory. Cache keys of SPIR-V and MSL kernels are unchanged.
- **Lowering.** `CudaCodeGenerator` calls the shared `GraphLowerer` with warp size 32, the device's opt-in shared memory (see
  [shared memory](#shared-memory)), the SM count as compute units and the device's features, so the CUDA provider selects the same
  kernels as the Vulkan provider on the same NVIDIA GPU where both have the features. That keeps Vulkan the reference for parity.
- **Memory.** Device memory per buffer (`cuMemAlloc`, zeroed), weights uploaded through two pinned staging buffers of up to 64 MB that the CPU
  fills (reading mapped files, repacking rows with the shared `WeightCopier`) while the GPU copies the other one. When the system
  refuses page-locked memory the staging halves down to 1 MB, then falls back to one pageable buffer with synchronous copies; an
  allocation that fails throws `DeviceOutOfMemoryException` with the driver's free/total figures ([hardware.md](hardware.md)). Shared weights,
  per-graph states, device forwarding between segments and data-dependent symbols work as on Vulkan.
- **Execution.** Each segment owns a CUDA stream and two timing events. Launches are recorded once per binding of the symbols (kernel,
  grid, packed arguments) and replayed with `cuLaunchKernel`; launches on one stream run in order, so the hazard tracker's barriers are
  not needed. Runtime scalars go through a pinned buffer with one asynchronous copy per step; small inputs and outputs through pinned
  staging buffers on the segment's stream, larger ones are copied on the segment's stream and waited for. Nothing runs on the legacy
  (NULL) stream: a copy there waits for every blocking stream of the context and fails while another thread captures a graph
  (`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`, two models on one GPU), so the segments' and the device's streams are created non-blocking and
  every ordering between them is explicit (an event, or a host wait for the other segment's pending execution).
  Every driver call first makes the device's primary context current on the calling thread. Copies of state rows
  (`ICompiledSegment.CopyStateRows`: recurrent-state snapshots for prefix caching, copy-on-write of a forked slot's KV block) are
  `cuMemcpyDtoDAsync` on the segment's stream, waited for, so hybrid prefix caching and `n` > 1 forks work as on Vulkan.
- **Module loading.** `CudaDevice` loads each kernel once: its cubin from the model-level cache (`cubin/<kernel hash>`) or the per-user
  artifact store when present, otherwise it links the PTX into a cubin with the driver's JIT (`cuLinkCreate` / `cuLinkAddData` /
  `cuLinkComplete`), records it in the cache (so the next load skips the JIT) and loads it with `cuModuleLoadData`.
- **Device features** use the neutral names: `fp16`, `subgroup_ops`, `int_dot` (`dp4a`) and `subgroup_size_control` (the warp size
  is fixed at 32) on every listed GPU; from compute capability 8.0 the tensor-core features `matrix_f16`, `matrix_bf16`, `matrix_int8`,
  `matrix_mma_layout`, the `bf16` type, `native_gemm_f16` and `native_gemm_kquant`; from 8.9 (Ada, Hopper, Blackwell) `matrix_fp8`, `fp8` and
  `native_gemm_fp8` ([kernels written in CUDA C++](#kernels-written-in-cuda-c)); from 9.0 `tensor_copies`; on 12.x with a CUDA 12.9
  driver or later `block_scaled_fp8` (the block-scaled E4M3 MMA, used only in the opt-in block-scaled FP8 mode), `block_scaled_fp4` (the
  block-scaled E2M1 MMA: the FP4 GEMM of NVFP4 weights) and `register_reallocation`
  (`setmaxnreg`: the FP8 GEMM's copy warpgroup); `library_gemm` when cuBLASLt is installed
  ([library GEMM](#library-gemm-cublaslt)); `launch_graphs` (not a code-generation feature, like `kernel_binaries`). The CUDA provider then selects the same tensor-core kernels as the Vulkan
  provider on the same GPU, except the `matrix_workgroup` ones (`VK_NV_cooperative_matrix2` has no CUDA counterpart in
  `mma.sync`). `SHARPLIGHT_DISABLE_FEATURES` and `DeviceOptions.DisabledFeatures` work as for every provider, and `kernel_binaries`
  turns off the cubin tier (kernels then load from PTX, JIT-compiled by the driver on every load).
- **Alignment.** CUDA requires vector loads to be aligned to their size; SPIR-V and MSL pointer accesses only assume the element
  alignment. `LoweringOptions.AlignedVectorAccess` (set by the CUDA provider only) keeps kernels whose row strides would misalign their
  16-byte loads off such shapes: the int-dot matvec then needs K to be a multiple of 64 (its quantized activation rows are 1.25 K
  bytes), other K take the f32 matvec. Found by `MatMulTests.OddShapes` (Q8_0, K = 96: "misaligned address").
- **Format-agnostic core.** Nothing outside `src/Sharplight.Cuda` (and the native packages `Sharplight.Cuda.Runtime` and
  `Sharplight.Cuda.Blas`) knows about CUDA. The shared changes are provider-neutral: the `KernelTranslator` hook, the block-size header
  and the `SL_TARGET_CUDA` target macro in `Sharplight.Shaders`, the `library_gemm` feature with its `LibraryKernel` descriptors, the
  CUDA forms inside the kernel library (`cuda_mma.slang`, `#if SL_TARGET_CUDA` blocks), which SPIR-V and MSL do not see, and the
  CUDA C++ kernels (`*.cu` in the kernel library, `SlangCompiler.CompileCuda`), which only a compiler with a CUDA C++ translator takes.

### Tensor cores

Slang's own CUDA cooperative matrices were not usable for the tensor-core kernels: code generation of `gemm_cm2` aborts ("internal
error E99999: unexpected IR opcode during code emit"), and the prelude's fragment loads are element-wise loads with up to 16 warp
shuffles per fragment instead of `ldmatrix`. The kernels therefore name their fragment types through typealiases, and on the CUDA C++
target (`SL_TARGET_CUDA`, a macro `SlangCompiler` sets for `ShaderTarget.CudaSource` only) those select the structs of
`Kernels/cuda_mma.slang` instead of Slang's `CoopMat`:

- `CuA16` / `CuB16` (f16), `CuA16bf` / `CuB16bf` (bf16): 16x16 operands, `CuA8i` / `CuB8i` (int8), `CuA8f` / `CuB8f` (E4M3): 16x32 /
  32x16 operands, `CuAccF`, `CuAccI`, `CuAccH`: 16x16 accumulators. They hold their registers in the layout of `mma.sync.m16n8k16`
  (16-bit) and `mma.sync.m16n8k32` (8-bit), and number their elements like the NVIDIA Vulkan driver (`matrix_mma_layout`: lane
  `4g + c` holds row `g + 8 (e >> 2)`, column `2c + (e & 1) + 8 ((e >> 1) & 1)`), so the kernels that address fragment elements
  (the register softmax of `fe`, the direct skinny GEMM, the fp8 and int8 GEMMs' scale epilogues) work unchanged; the kernels
  verify the layout at run time as on Vulkan.
- The same API as `CoopMat`: `T.Load<layout>(array or pointer, element, stride)`, `Store`, `coopMatMulAdd<float, false>` (overloads
  for the fragment structs), construction from a scalar or from an f32 accumulator, element access and accumulator arithmetic.
- Loads from shared memory are one `ldmatrix.x4` per fragment (`.trans` for the transposed layouts), from device memory element by
  element. The PTX helpers come with the kernel through `__requirePrelude`, so the CUDA C++ of a kernel is self-contained.
- The shared arrays are declared 16-byte aligned (`NvrtcTranslator` adds `__align__(16)`), so `ldmatrix` needs no run-time
  alignment check for the packed-half tiles (`uint4` arrays); arrays of narrower elements are checked and fall back to loads.
- Where the Vulkan driver merges repeated fragment loads and NVVM cannot (the inline `ldmatrix` is volatile), the CUDA form loads
  once: the fp8 GEMM's B fragments of a weight block (`gemm_fp8.slang`, `#if SL_TARGET_CUDA`; its prefill GEMMs went from 1.5x
  slower than Vulkan to slightly faster).

Nothing changes for SPIR-V and MSL: the `#else` branches are the former typealiases. The SPIR-V of the 6472 kernels the Vulkan
kernel suite compiles, and of the 778 kernels of six models (Qwen3-0.6B Q8_0, Llama-3.2-1B Q8_0, gemma-3-1b Q8_0, Qwen3-0.6B HF,
Qwen3.5-0.8B, OLMoE), is byte-identical between this branch and `develop` (`SHARPLIGHT_DUMP_KERNELS`, then compiled with the kernel
libraries of both commits); the models' MSL differs only in `#line` directives.

**Wave operations.** Slang's CUDA backend makes the active mask of every wave operation explicit: a function with one
`WaveReadLaneAt` gets a `__ballot_sync` per branch and loop iteration (664 in the decode attention), which made the decode
attention 3.5x slower than its SPIR-V form. On CUDA `common.slang` maps the wave operations the kernels use (`WaveReadLaneAt`,
`WaveActiveSum` / `Max` / `Min` / `Product` / `AllTrue` / `AnyTrue`, `WaveIsFirstLane`, `WavePrefixSum`) to their `WaveMask*` forms
over `__activemask()`: the lanes executing the call, which is the semantics of subgroup operations on SPIR-V and MSL. The kernels
call them in subgroup-uniform control flow (or in branches whose lanes are meant to take part), as SPIR-V requires.

**Alignment.** CUDA faults on vector loads that are not aligned to their size; SPIR-V and MSL only assume the element alignment.
Besides the lowering rule of phase 1, `common.slang` has `LOAD_U4_ALIGNED` / `LOAD_U2_ALIGNED` with a compile-time alignment
(`ALIGN_OF` of the row pitch): on CUDA they split into the widest aligned loads, elsewhere they are the plain vector load. Used for
the Q8_0R code rows (K = 1152: 1224-byte rows, gemma-3), the FP8 GEMM's weight rows (264-byte rows at K = 256) and activation rows,
and the int8 GEMM's activation rows.

### Two-stage precompilation

The provider implements the contract of [model-format.md](model-format.md#two-stage-precompilation):

| stage | CUDA | artifact | needs |
| --- | --- | --- | --- |
| portable (`CompilePortableAsync`, profile) | Slang → CUDA C++ → NVRTC → PTX for `compute_XY` | code tier, `provider=cuda kind=ptx target=ptx_XY`, entries `kernel/<hash>` | NVRTC, no GPU |
| device (`CompileDeviceSpecificAsync`, opened device) | PTX → cubin with the driver's JIT | binary tier, `kind=cubin`, matched on GPU name, CUDA driver version and kernel ABI, entries `cubin/<hash>` | the driver |

`PortableCodeFormat` is `ptx`; a device profile's `CodeTarget` is `ptx_<major><minor>` (`ptx_120` for compute capability 12.0), so one
portable set serves every GPU of that compute capability and, through the driver's PTX JIT, newer ones. A model precompiled with both
stages (`sharplight-convert precompile`, or the CLI's `--save`) loads on a machine without NVRTC: the kernels come from the package
as PTX and cubins, and nothing is compiled.

### Shared memory

The Slang kernels declare their shared arrays statically, and CUDA limits static shared memory to 48 KB per block, the same limit
the Vulkan driver reports. `SharedMemoryRewriter` (in `NvrtcTranslator`) moves the shared arrays of a kernel that needs more into one
`extern __shared__` buffer (16-byte aligned offsets, a `#define` per array), writes the size into the PTX header
(`// sharplight-shared: <bytes>`), and the device opts the function into it (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`) and
launches it with that much dynamic shared memory. The lowering gets the device's opt-in maximum (99 KB on the RTX PRO 4500
Blackwell, `max_shared_memory_optin`; 227 KB on H100 / B200), `SHARPLIGHT_CUDA_MAX_SHARED=49152` keeps the static limit.

Measured on Qwen3-8B (bf16) at 2048 rows: the larger budget changes no default tile choice of the LLM kernels, and double-buffering
gemm_cm2's 64-value K blocks (80 KB, `SHARPLIGHT_GEMM_MACROS=GNBUF=2`) is 21 % slower (one workgroup per SM instead of two), as are
more register prefetch sets (`GPF=2`). What the budget changes is which forms pass their shared-memory fit checks (the lowering
compares against `max_shared_memory`): the MQA attention over latent KV rows, the general flash attention, the two-row convolution
tiles, the delta-rule chunk width, top-k and NMS sorting in shared memory; the kernel suite runs on CUDA with the 99 KB budget. New tile shapes
for the GEMMs belong to the hand-written kernels of the roadmap.

### Library GEMM (cuBLASLt)

`DeviceFeatures.LibraryGemm` (`library_gemm`) lets a provider run matrix products on its vendor's GEMM library. The lowering emits
them like kernels whose code is a one-line descriptor (`LibraryKernel`: `// sharplight-library: gemm a=bf16 n=4096 k=12288`) with
the arguments x, w, y and the row count; a provider without the library rejects the descriptor at load (`KernelCompilationException`)
and hosts fall back as for any kernel. The CUDA provider reports the feature from compute capability 8.0 when cuBLASLt can be loaded
and runs the descriptors with `cublasLtMatmul` on the segment's stream (`CudaBlas`: one handle and a 32 MB workspace per device, a
plan per shape, row count and operand alignment, the heuristic's first algorithm).

Used where it beats the generated GEMMs on this GPU ([benchmarks.md §51](benchmarks.md)): 16-bit weights from 256 rows
(`SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS`). cuBLASLt takes both operands in one type, so the operands are those of the generated tensor-core
GEMMs: the f16 copy of x they read (row-ranged where x may exceed f16's range, with the same `f16_range_unscale` fix-up of the output
rows), and the weights in f16: bf16 weights whose every consumer is a MatMul are stored as f16 (`WeightRepack.Bf16ToF16` at upload),
which is the rounding the generated GEMMs apply when they load bf16 weights. The products therefore have the generated GEMMs'
precision (an earlier form with bf16 activations failed the massive-activation tests of `F16RangeTests`). A gate / up pair with its
SiLU·mul group runs as two library GEMMs and the group's own launch instead of the dual gemm_cm2. cuBLASLt measured 145-165 TFLOPS on the Qwen3-8B shapes against 112-128 for gemm_cm2 on CUDA (and
118-125 on Vulkan); FP8 with per-tensor scales 250-336 TFLOPS, but the block-scaled FP8 of the checkpoints in use (128x128 weight
blocks, `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F`) is not supported on compute capability 12.0 (`CUBLAS_STATUS_NOT_SUPPORTED`),
so FP8 stays on the generated `gemm_fp8`.

Since phase 3 the same products run on the hand-written f16 GEMM (`native_gemm_f16`) wherever the GPU has it, which matches
cuBLASLt per shape ([benchmarks.md §53.3](benchmarks.md)); the library is used only when `native_gemm_f16` is disabled. It stays an
optional package that neither `Sharplight.Cuda` nor Studio depends on.

### Kernels written in CUDA C++

Some kernels need what the Slang kernels cannot express on CUDA: asynchronous copies (`cp.async`, `cp.async.bulk`), shared-memory
layouts sized at run time, inline PTX sequences. Those are written in CUDA C++ as `.cu` files of the kernel library
(`src/Sharplight.Shaders/Kernels`) and compiled with `SlangCompiler.CompileCuda`: the file, preceded by the lowering's macros
(`#define K 4096`, ...) and generated source (an epilogue), goes to the provider's translator directly, without Slang. On CUDA that is
`NvrtcTranslator`, the same as for the Slang output, so the kernels are PTX in the code tier and cubins in the binary tier exactly like
every other kernel: cached per user, exported by `precompile`, loaded from a precompiled model without NVRTC. The source states the
block size and its dynamic shared memory in its header lines (`// sharplight-threads:`, `// sharplight-shared:`).

Why NVRTC rather than `nvcc` at build time: the lowering specializes these kernels like the Slang ones (K, N, row strides, tile
configuration, the fused epilogue of the dual GEMMs, which is generated from the graph's elementwise group), so the set of variants is
not known when the provider is built; NVRTC ships with the provider anyway, and the two-stage contract already delivers PTX and cubins
to machines without it. Nothing needs a toolkit at build time either. `-arch=compute_XY` PTX (no `sm_XYa` features) keeps them
portable to newer GPUs through the driver's JIT.

A lowering selects them only through device features, and every one has the generated kernel as its fallback: disabling the feature
(or a GPU without it, or any provider other than CUDA) gives exactly the phase-2 kernels. Vulkan never sees them: its SPIR-V is
byte-identical to `develop` ([benchmarks.md §53.7](benchmarks.md)).

- **`gemm_f16_native.cu`** (`native_gemm_f16`, compute capability 8.0+): y = x · Wᵀ on the operands of the library GEMM (the f16 copy of
  x, bf16 weights stored as f16), f32 output. `cp.async` copies of 64-value K steps (128-byte rows, XOR swizzle so the copies and
  `ldmatrix` are free of bank conflicts) into dynamic shared memory, `mma.sync.m16n8k16`, two tile configurations: 128 tokens x 256
  weight rows (8 warps of 64 x 64, two stages, 96 KB) and 128 x 128 (4 warps, three stages, 96 KB), chosen per row count by the
  estimated number of waves on the GPU's SMs. Blocks are rasterized in groups of 16 token tiles that walk all weight tiles, so a group's
  activations stay in L2 while the weights stream (8192-row prefill: 138 → 154 TFLOPS on [4096 → 4096]). The dual form computes a gate /
  up pair with the SiLU·mul group (any group of plain elementwise operations) as its epilogue, which the library path ran as two GEMMs
  and a separate launch; a row-ranged f16 copy of x (massive activations) keeps the separate path, as its scaling must be undone before
  a nonlinear epilogue. Used from `SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS` (256) rows, like the library.
  With `H_BIAS` the store adds a bias row, and with `H_XRS` it first undoes the row scales of a row-ranged copy (exact), so projections
  with biases (gpt-oss, Qwen2) take it from 1024 rows instead of gemm_cm2 (gpt-oss-20b prefill +2 % at 2048 / 8192 tokens,
  [benchmarks.md §61.5](benchmarks.md#61-llm-round-15-decode-step-projections-and-expert-gemms-the-native-f16-gemm-with-a-bias-store-2026-11-20)).
  A decode form (`H_CFG` 2: 64 tokens x 64 or 32 weight rows, the narrower where 64-row tiles would leave a quarter of the SMs idle,
  four stages, no K split) serves 33-64 rows where gemm_cm2 split K and reduced, with the bias store, and a multi-output form of it
  (`MULTI`) runs q / k / v as one launch, each weight tile in one member's matrix: gpt-oss-20b at 64 sequences q / k / v + o 102 → 79 µs
  per layer, step -0.5 ms ([benchmarks.md §68](benchmarks.md)); `SHARPLIGHT_NATIVE_F16_DECODE_BN=0` turns it off.
  Prompt-sized q / k / v groups (from 256 rows, 1024 with biases) run as one launch of the 128-row tiles over the members' weight rows
  (`MULTI` with the prompt tile configurations; 128 x 256 tiles where every member boundary is a multiple of 256, else 128 x 128):
  gpt-oss-20b at 2048 rows 378 + 2 x 51.4 → 406 µs per layer ([benchmarks.md §82](benchmarks.md); `SHARPLIGHT_MULTI_GEMM_TALL=0`
  keeps the members' own launches). The store undoes row-range scales (`H_XRS`) also without a bias, so no `f16_range_unscale` launch
  follows. Products narrower than 256 outputs (MoE routers: gpt-oss's [2880 → 32]) stay on gemm_cm2's K split (one column tile of
  128-row token tiles is 16 blocks at 2048 rows on 82 SMs): 39.6 → 16.9 + 3.9 µs per layer.
- **`gemm_fp8_native.cu`** (`native_gemm_fp8`, compute capability 8.9+): the block-scaled E4M3 GEMM of `gemm_fp8.slang` on operands
  whose scales it can copy densely: the weights in the 128-row groups of `WeightRepack.Fp8B128Grouped` (`f8e4m3_b128g`: [128 rows of
  K codes][K / 128 blocks of the 128 rows' f32 scales], the same bytes as the split rows, repacked at upload, so a precompiled package
  keeps the stored layout and every load produces the provider's one) and the `quantize_fp8` rows followed by a dense copy of their
  scales ([K / 128][capacity] f32, written by `quantize_fp8` with `DENSE_XCAP` and by the norms that write E4M3 rows). A tile's scales
  of a K block are then 64 16-byte copies instead of 256 4-byte loads from separate lines. A `cp.async` ring of 128-value K blocks in
  96-99 KB: 128 x 128 tiles with three stages or 64 x 128 tiles with four, chosen per batch by estimated block waves; 2 x 4 warps, per K
  block the B fragments loaded once and each 16-row fragment's MMAs summed in a fresh accumulator and added with one fused
  multiply-add scaled by `sx · sw`; dual form with the generated epilogue; grouped rasterization. From
  `SHARPLIGHT_NATIVE_GEMM_MIN_ROWS` (65) rows, K a multiple of 512, N of 128; from 9 rows in a decode form (64-row weight tiles; 64-row
  token tiles with four stages above 32 rows, 32-row token tiles with three stages, two blocks per SM, up to 32) for single matmuls,
  dual gate / up pairs and multi-output q / k / v (Qwen3-8B-FP8 at 64 sequences 17.2 → 16.4 ms, [benchmarks.md §63.3](benchmarks.md);
  at 12-32 sequences -5-7 %, [§68](benchmarks.md); `SHARPLIGHT_NATIVE_FP8_DECODE_BN=128` turns it off). 225-240 TFLOPS on the Qwen3-8B shapes at 2048 rows
  in the model (phase 3: 187-227). The generated kernels that read FP8 weights (decode matvecs, skinny and fp8 GEMMs) decode the
  grouped layout too (`FP8_GROUPED`, `F8_GROUPED`), so the layout costs decode nothing measurable; on Vulkan it is off
  (`SHARPLIGHT_FP8_GROUPED=1` turns it on anywhere, [benchmarks.md §54.7](benchmarks.md)). The dual gate / up pair takes the decode
  form from 9 rows (the skinny GEMM keeps 1-8), and the first stages' weight copies are issued before `griddepcontrol.wait`, so a
  programmatic dependent starts its weight stream while the launch that writes x still runs ([benchmarks.md §84](benchmarks.md)).

- **`gemm_fp4_native.cu`** and **`quantize_fp4.cu`** (`block_scaled_fp4`, compute capability 12.x with a CUDA 12.9+ driver, compiled
  for `compute_120f` with NVRTC machine code): matmuls of NVFP4 weights (`nvfp4`, and `nvfp4_gs` with its second-level scale, which is
  the same in every block of a row) from 9 activation rows (`SHARPLIGHT_NATIVE_FP4_MIN_ROWS`); below, the integer-dot matvec. The
  activations are quantized to FP4 per row (`quantize_fp4.cu`: E2M1 codes, an E4M3 scale per 16 values under the row's f32 scale
  amax / 2688, 160-byte records per 256 values, a numerics change like the W4A4 mode of the checkpoints' producers: about 9 % relative
  error per activation; so weights of a weight-only format, `WeightInfo.Activations` unquantized as the importer reads it from
  the checkpoint's config, e.g. ModelOpt `W4A16_NVFP4`, never take it, see below); the weights are read in their stored 36- / 40-byte blocks (cp.async into shared memory, 32-byte sectors kept in
  one granule, [benchmarks.md §97](benchmarks.md)) and multiplied by `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X` with
  both operands' 16-value scales applied in the MMA and f32 accumulation over all of K; the store multiplies by the activation row's
  and the weight row's f32 scales. Tiles: 16 / 32 / 64 token rows over 32-64 weight rows at decode batches, 64 x 128 and 128 x 128 above
  (`NativeFp4Tile`); the gate / up pair runs one dual launch with the group in its store; `F4_MOE` is the expert-grouped form over
  `moe_route` tiles (16 / 32 / 64 rows) for NVFP4 expert stacks, whose gate / up products run as two launches and the group.
  `quantize_fp4` is a file of its own so that it gets the dependency wait at its entry (a source that issues
  `griddepcontrol.wait` itself gets none). `SHARPLIGHT_NATIVE_FP4=0` (or disabling `block_scaled_fp4`) keeps NVFP4 matmuls on the
  integer-dot matvec, the skinny GEMM and the GEMMs on dequantized weights; `SHARPLIGHT_NATIVE_FP4_MOE=0` only the expert products.

- **Weight-only checkpoints** (`WeightInfo.Activations` unquantized: ModelOpt `W4A16_*` / `*_WO`, compressed-tensors groups without
  `input_activations`, torchao `Float8WeightOnlyConfig`, ...): their products never quantize activations to E2M1 or E4M3, whatever
  the device offers (`LowPrecisionActivationsAllowed` in the lowering; `CompileOptions.AllowReducedPrecision` lifts it). NVFP4 weights
  then keep the integer-dot matvec (q8 blocks), the skinny GEMM and the GEMMs on decoded weights from 9 rows; block-scaled E4M3
  weights keep their stored layout (no 128-row groups, so no hand-written FP8 GEMM and no power-of-two requantization) and run the
  f16 tensor-core GEMM on decoded weights. Formats that declare activation quantization (W4A4, W8A8, `quant_method fp8`) and
  sources that say nothing (GGUF) keep the native paths. `WeightOnlyActivationTests` checks the selection on a Blackwell profile.

- **`gemm_fp8_decode.cu`** (`native_gemm_fp8`): single block-scaled E4M3 matmuls at 9-16 rows in the grouped layout. No shared-memory
  stages: every warp loads 16 bytes per lane of its 32 weight rows and of the 16 activation rows straight into registers (the same
  permutation of K inside each 64-value chunk for both operands, so the words fill the m16n8k32 fragments with the weights as A), the K
  range split over 1-8 warps of a block (a divisor of the K blocks giving ~512 warps) and summed through shared memory in a fixed order;
  blocks walk their tiles, and the first K block's weights are loaded before `griddepcontrol.wait`. Qwen3-8B-FP8 down [12288 → 4096] at
  16 rows 69.8 → 67.4 µs, 16-sequence step -1.0 % ([benchmarks.md §84](benchmarks.md)); the dual pair is slower on it (141.8 against
  131.5 µs) and stays on `gemm_fp8_native.cu`. `SHARPLIGHT_FP8_DECODE_KERNEL=0` / `=1`: off / also for dual pairs.

- **`gemm_kq_native.cu`** (`native_gemm_kquant`, compute capability 8.0+): K-quant (Q4_K, Q5_K, Q6_K, the GGUF block layout) projections
  at decode batches of 9-64 rows (single matmuls from the first batch past the matvec, `SHARPLIGHT_NATIVE_KQ_MIN_ROWS`), the q / k / v
  multi-output form, and a dual gate / up form (`KQ_DUAL`: the tiles interleave the rows of both matrices, 32 outputs of each per block, and
  the store applies the activation product; `SHARPLIGHT_NATIVE_KQ_DUAL=0` keeps two launches and the elementwise launch); batches whose
  generated split would be 1 are routed to it too. 16-, 32-, 48- or 64-row token tiles by batch (`SHARPLIGHT_NATIVE_KQ_BM`; 48-row tiles
  for 33-48 rows since [§79](benchmarks.md)) x 64 weight rows, eight warps of 8 weight rows each; a 2-4 stage `cp.async` ring of 256-value superblocks
  holds the swizzled f16 activation rows and the raw blocks (16-byte windows, so Q6_K's 210-byte blocks may start at any even byte);
  each thread decodes its `mma.sync.m16n8k16` B fragments from the raw codes in registers. K split only for small grids or a nearly
  empty last wave (`SHARPLIGHT_NATIVE_KQ_SPLIT`), reduced in the last block per tile. 1.6-2.1x gemm_cm2 per kernel at 32 rows, Qwen3-8B
  Q4_K_M decode steps 1.32-1.79x at 9-64 sequences ([benchmarks.md §77](benchmarks.md)). At 64 rows the kernel is bound by the
  `mma.sync` issue rate (about 46 µs of a [4096 → 12288] Q4_K product's 82 µs with no copies at all, §79), not by its copy ring.

- **`gemm_moe_native.cu`** (`native_gemm_moe`, compute capability 8.0+): the expert-grouped product of a MatMulId with MXFP4 experts
  (the 16-row groups of `WeightRepack.Mxfp4Split`) from 1 routed row per expert on average (`SHARPLIGHT_NATIVE_MOE_MIN_ROWS`;
  below, the expert-grouped skinny GEMM serves; [benchmarks.md §64.2](benchmarks.md)). One block per routing tile of 128 rows (`moe_route`) and 256 (dual) or 192 weight
  rows (up to 12 routed items per expert: 32-row tiles of four stages, up to 24: 64-row tiles of three, both two blocks per SM where they fit, dual tiles narrowed to 128
  weight rows for it, [benchmarks.md §63.2](benchmarks.md); the 64-row dual tile takes four stages without the factor table, `NM_TABLE=0`,
  [§65.2](benchmarks.md); `SHARPLIGHT_NATIVE_MOE_SHORT`, `SHARPLIGHT_NATIVE_MOE_BLOCKS`, `SHARPLIGHT_NATIVE_MOE_TABLE=1`): a four-stage `cp.async` ring of 64-value K steps holds the expert-sorted f16 activation rows (swizzled 128-byte rows) and
  the raw codes and E8M0 scales (4.25 bits per weight; codes stored [row][block], swapped in rows with bit 2 set (`nm_cpos`), so each
  row's two 16-byte copies of one 32-byte sector land in one 32-byte granule and a fragment's 8 rows still cover all banks: copies
  split across granules held the launches at ~1.7 GHz under the software power cap, [§97](benchmarks.md); with
  an `L2::256B` prefetch hint so the 32 bytes a stage takes of each row still stream from memory in whole bursts). Eight warps side
  by side along the weight rows (warp tile 128 x 32) decode the 4-bit codes of the fragment they multiply with the arithmetic of
  gemm_cm2's decode (doubled code in half bits, two half factors of the scale from a per-block table) right before
  `mma.sync.m16n8k16`, and the next block's decode is issued between the current block's MMAs. A partial tile's empty 16-row
  fragments are skipped by a warp-uniform branch (predicated-off MMAs still take the tensor pipe's time; this alone took 2048-token
  gate / up from 2.42 to 2.29 ms per layer), taken once per block: each live fragment count runs its own copy of the pipelined K loop
  (a switch inside the loop cost 1-8 %, [benchmarks.md §75.2](benchmarks.md)). The store applies the expert group's generated epilogue in CUDA C++ (dual: gpt-oss's
  halves with their bias rows and clamped SwiGLU, or a gate / up pair; single: bias rows, router weight; `MoeEpilogueWriter`
  writes both the Slang and the CUDA form), and multiplies row-ranged rows by their inverse scale. The decode and the MMAs' K
  order are gemm_cm2's, so outputs are identical with the feature on and off (gpt-oss-20b greedy outputs byte-identical).
  gpt-oss-20b, 2048 tokens: gate / up 2.63 → 2.26 ms, down 1.43 → 1.20 ms per layer ([benchmarks.md §56](benchmarks.md)).
  The gate / up (dual) form stores the down projection's input directly (`NM_Y16`): f16 rows at their expert-sorted position, the
  layout of `moe_gather_x16`, so the down projection's gather and the f32 round trip go. On a row-ranged copy (`NM_Y16_RS`) a value
  of 2^15 or more is stored as an f16 infinity with its f32 value spilled to the f32 output and the row's scale word set to 1;
  `moe_range_fixup` (moe.slang, 256 rows per workgroup) then stores flagged rows scaled and clears every other row's word (it may hold
  the previous layer's). gpt-oss's clamped SwiGLU bounds the values at ~56, so on its text no row is ever flagged and the fix-up is a
  2-4 µs scan per layer; outputs are bit-identical to the gather path. `SHARPLIGHT_MOE_SORTED_STORE=0` keeps the f32 store and the
  gather (gpt-oss-20b 2048-row prompt step 119.1 → 115.0 ms, 64-row decode -1.0 %, [benchmarks.md §82](benchmarks.md)). Where the
  router's f16 copy of x serves (the token rows of a prompt step), the gate / up GEMM reads its items' token rows from it
  (`NM_X_TOKEN`) instead of a gathered copy (`SHARPLIGHT_MOE_TOKEN_ROWS=0` gathers).
- **`gemm_fp8_tma.cu`** (`native_gemm_fp8` and `tensor_copies`, compute capability 9.0+): the 128-row tiles of the FP8 GEMM in a
  warp-specialized persistent form. At most one block per SM walks its share of the tiles (the copies of a tile's first K blocks
  overlap the previous tile's last blocks and stores); one lane of a copy warp issues tensor copies (`cp.async.bulk.tensor`, 128-byte
  swizzle) of the activation and weight tiles and bulk copies of their scales into a ring of three 32 KB stages and two scale slots;
  full and empty barriers replace the block barrier per K block; the eight computing warps do the MMAs and the promotion of
  `gemm_fp8_native.cu`, bit for bit the same results. Qwen3-8B-FP8 prefill +9-11 % (0.86-0.99x of vLLM, [benchmarks.md
  §57](benchmarks.md)). `SHARPLIGHT_FP8_TMA=0` or disabling `tensor_copies` keeps `gemm_fp8_native.cu`. Two forms
  ([benchmarks.md §86](benchmarks.md)): `QOUT` (dual gate / up pairs from 256 rows in the block-scaled mode; with the promotion it
  spills at the 168 registers of a 288-thread block) stores the down projection's E4M3 rows
  and scales instead of f32 rows, bit for bit what `quantize_fp8` writes from them (a block walks the two 64-output tiles of a
  128-value scale group back to back and combines the rows' maxima in shared memory), so the f32 round trip and the quantization
  launch go (`SHARPLIGHT_FP8_QSTORE_MIN_ROWS=0` turns it off); `F8N_BLOCK_SCALED` (`block_scaled_fp8`, compute capability 12.x,
  compiled for `compute_120f`; on by default since 2026-11-23, see "Block-scaled FP8 by default" below) runs the block-scaled MMA on power-of-two activation
  scales (`E4M3_UE8M0` in e4m3.slang) and weights requantized to power-of-two scales at load (`WeightRepack.Fp8B128GroupedUe8m0`),
  accumulating all of K in the tensor cores (gate / up 1,612 → 1,463 µs per layer at 2,048 rows; a numerics change).
  Three more forms, all bit for bit the default path's results ([benchmarks.md §89](benchmarks.md)): `F8N_WG`
  (`register_reallocation`, compiled for `compute_120f`; the default, `SHARPLIGHT_FP8_WG=0` keeps the 9-warp form) makes the copy warp a whole warpgroup that gives its registers to the computing warps (`setmaxnreg`: 40
  for the copy warps, 232 instead of 168 for the computing ones), which leaves room for `QOUT` with the f32 scales too, so the dual
  gate / up pair stores the down projection's E4M3 rows. The CUDA 13.2 driver's PTX compiler drops `setmaxnreg` (its cubins target
  `sm_120`), so through the driver the form ran as 12 warps of 168 registers; with NVRTC's machine code (below) the reallocation is
  real, so the device reports `register_reallocation` only while that machine code is on (not with
  `SHARPLIGHT_CUDA_NVRTC_SASS=0`). The computing warps release a stage after `fence.proxy.async.shared::cta`: their reads of it
  (`ldmatrix`, `ld.shared`) are the generic proxy's and the next tensor copy into it the async proxy's, which the barrier's release
  alone does not order; without it a copy could land before a late warp's read (the run-to-run differences of §92, [§93](benchmarks.md)). `F8N_WUNI` (`SHARPLIGHT_FP8_UNIFORM_SCALES=0` turns it off) reads one weight scale per 128-row group and
  K block for weights whose groups share each block's scale (`WeightInfo.ScaleRowBlock` a multiple of 128: checkpoints with
  128 x 128 scale blocks, which the Hugging Face importer marks and packages store in their `WLAY` section), so the promotion
  multiplies once per row instead of once per element; `MULTI` runs a q / k / v group (the same E4M3 rows, one K) as one launch over
  the concatenated weight rows, each tile in one member's matrix (`SHARPLIGHT_MULTI_GEMM_TALL=0` keeps the members' launches).
  On compute capability 12.x the kernel also gets machine code from NVRTC (it defines `SL_NVRTC_SASS`): NVRTC compiles it for the
  device's real architecture next to the PTX, the code carries the cubin (a `// sharplight-cubin:` header line), and the device loads
  it instead of the driver's compilation of the PTX (Qwen3-8B-FP8 prompt steps 2-5 % faster on the RTX PRO 4500 with driver 595;
  [benchmarks.md §92](benchmarks.md)). `SHARPLIGHT_CUDA_NVRTC_SASS=0` turns it off, `all` applies it to every kernel and
  `if:<text>` to the kernels whose source contains the text (diagnostics).

- **`attn_prefill_native.cu`** (`native_attention`, compute capability 8.0+): causal attention of many queries against an f16 KV cache
  (prefill, prompt runs of serving steps, mixed steps' long runs) in the FlashAttention-2 form. A block takes 16 query rows per warp
  and fragment (4 warps of two 16-row fragments for head size 64, so each K / V fragment feeds twice the MMAs: +7-12 % from 16K keys,
  [benchmarks.md §65.3](benchmarks.md), `SHARPLIGHT_ATTN_MR`; 8 warps of one for 128) of one head; each warp keeps its Q · SCALE fragments (f16), its scores, the online-softmax
  statistics and its output accumulators in registers for the whole key loop, and the probabilities become the A operands of P · V in
  the accumulator layout. The block shares 64-key K / V tiles copied with `cp.async` into a two-stage ring (three at head size 128, where the
  block has the SM to itself: +0.5-1 % at 16K-128K, `SHARPLIGHT_ATTN_STAGES`; 16-byte chunks, XOR swizzle; `ldmatrix` for K, `ldmatrix.trans` for V), one block barrier per tile. Masks are applied only on tiles that cross a row's
  limits (diagonal, window start), and tiles a warp's rows cannot see are skipped by a warp-uniform branch. Blocks run the longest query
  tiles first. With multi-sequence runs grid y enumerates the query tiles of all runs (ceil(T / tile rows) + runs blocks, each finds
  its run by a warp prefix sum), so short prompts launch no empty blocks ([benchmarks.md §80.2](benchmarks.md)). For multi-sequence runs (serving's prompt and mixed steps) a block takes the query heads of one KV head
  (`FN_HG`, a multiple of the warps per block; each K / V tile feeds all of them): gpt-oss-20b, 8 prompts of 256 rows, 72.3 → 69.2 µs
  per layer ([benchmarks.md §82](benchmarks.md); `SHARPLIGHT_ATTN_HG=1` / `=0` everywhere / nowhere). It serves sliding windows, chunked attention, soft caps, sinks, grouped-query heads, multi-sequence runs (also paged)
  and the mixed steps' `MIX_RUN`, at head sizes that are multiples of 64 up to 256 (32-key tiles above 128) and of 128 up to 512 (Gemma 4's
  global layers: two warps share each 16-row group, each with half the key and value dimensions, and exchange their partial scores
  through shared memory; 16-key tiles), [benchmarks.md §74](benchmarks.md); score biases, spans, masks, selections and other head sizes
  keep `fa` / `fe`. gpt-oss-20b prefill attention at 2048 tokens 16.2 → 3.5 ms (FlashInfer 4.4), Qwen3-8B 12.7 → 9.6 ms against `fe`
  ([benchmarks.md §58](benchmarks.md)). Results differ from `fa` / `fe` by f16 rounding of the probabilities in another summation
  order (as `fa` and `fe` differ from each other). `SHARPLIGHT_NATIVE_ATTN=2` keeps plain causal attention on `fe`; `=0`, or
  disabling the feature, keeps every attention on the generated kernels.

- **`attn_decode_native.cu`** (`native_attention`, compute capability 8.0+): the split-K decode attention (`decode` in attention.slang:
  same grid, key ranges and partials, so `decode_combine` merges them) on tensor cores for f16 caches at head sizes 64 and 128. The
  up to 8 query heads of a KV group are the rows of `mma.sync.m16n8k16`; each of a block's 4 warps takes every fourth tile of 32 keys
  (16 at head size 128) and runs its own online softmax in registers, with no shared memory or barriers in the key loop: a lane group
  loads its key and value rows straight from the cache in 16-byte chunks, which are the B operands of Q · Kᵀ as loaded (the head
  dimensions permuted alike in Q and K), the score accumulators become the A operands of P · V (f16), and the value words become B
  operands by an in-register transpose (`movmatrix`). The warps' states merge in shared memory at the end. It serves windows, chunks,
  soft caps, sinks, multi-sequence and paged caches and mixed steps' `MIX_RUN` (a step whose runs average fewer than 64 rows: its
  decode rows run here and the prompt runs on `attn_prefill_native.cu`, instead of a whole query tile per decode row there,
  [benchmarks.md §90](benchmarks.md)); groups narrow enough for the fused combine (heads per
  block times head size up to 256) keep `decode`. gpt-oss-20b decode attention at 64 sequences 1.32 → 1.03 ms per step (vLLM 1.20,
  [benchmarks.md §72](benchmarks.md)). Q · SCALE and the probabilities are rounded to f16, as in the prefill kernel.
  Each warp prefetches the key and value rows of its next tile into L2 (`prefetch.global.L2`, one 128-byte line per lane) before
  loading the current one (key ranges of four tiles per warp or more), so its loads wait on L2 rather than DRAM: 4 x 130,048 keys of gpt-oss-20b 1.44 → 1.31 ms per layer
  (739 → 815 GB/s) in the harness, 2-9 % at every shape measured, gpt-oss-20b 4 x 128K decode step -2 %
  ([benchmarks.md §78](benchmarks.md)); `SHARPLIGHT_ATTN_DECODE_PF=0` turns it off (`=n`: n tiles ahead).
  `SHARPLIGHT_NATIVE_DECODE=0` keeps `decode`.

**Tensor maps.** A CUDA C++ kernel can take tensor maps (`CUtensorMap`) after its scalars: the lowering states them in a header line
(`// sharplight-tensormaps: <pointer argument> <rank> <dims> <strides> <box>; ...`, `SlangCompiler.CudaTensorMapsHeader`), the
translator keeps it in front of the PTX, and `CudaSegment` encodes each map over the pointer argument it names
(`cuTensorMapEncodeTiled`, bytes, 128-byte swizzle) when it records the launch, 64-byte aligned after the scalars. Recordings, CUDA
graphs and precompiled packages need nothing else. Two compiler details matter for such kernels under NVRTC (benchmarks.md §57.3):
`__launch_bounds__` without a block minimum (with one, NVRTC 13.0's PTX spilled inside the MMA loop), and `shared::cta` copy
destinations (the `shared::cluster` form makes ptxas add a run-time check with a fallback call that needs local memory, whose first
launch can fail with "out of memory" once the KV cache has filled the GPU).

Measured limits of the FP8 kernel (prototypes, [benchmarks.md §53.4 and §54.3](benchmarks.md)): with the MMAs and `ldmatrix` alone
the loop runs at ~300 TFLOPS (the `mma.sync` E4M3 peak of this GPU is 474); with the dense layouts the scale copies cost nothing, the
per-block promotion ~6 % and the code copies that the MMAs do not hide ~15 %. CUTLASS's block-scaled sm120 kernel (vLLM with
`VLLM_USE_DEEP_GEMM=0`) reaches ~290 TFLOPS with the same f32 scales, so a different main loop can still gain ~20 %. Tried and not
kept: TMA (phase 3), a producer warp with mbarrier pipelines, 16-warp 256 x 128 tiles (spills), 64 x 128 / 128 x 64 tiles for
large batches, register double-buffering of A, L2 prefetch hints; a promotion with one weight scale per tile is 6 % faster but only
valid for 128 x 128 block-scaled checkpoints, which the lowering cannot tell from per-channel ones. Blackwell's block-scaled
`mma.sync ... kind::mxf8f6f4.block_scale` needs UE8M0 (power-of-two) scales per 32 values and `sm_120a` / `sm_120f`; on this GPU it
issues at the same rate as the plain E4M3 `mma.sync` (470 against 473 TFLOPS in a register-only loop, [benchmarks.md §86](benchmarks.md)),
so it saves the promotion's instructions and energy, not MMA time. vLLM's default DeepGEMM path rounds the checkpoints' scales up to
powers of two and requantizes the weights for it; this provider does the same per row and 128-value block in its block-scaled mode
(`block_scaled_fp8`: Qwen3-8B-FP8 perplexity +0.8 %, vLLM's own E8M0 path +0.9 %), the default since 2026-11-23.

**Block-scaled FP8 by default** (owner's decision 2026-11-23, [benchmarks.md §86.3, §89.4, §95](benchmarks.md)). Where it applies, the
block-scaled mode is the default: a device with `block_scaled_fp8` (compute capability 12.x with a CUDA 12.9+ driver), `native_gemm_fp8`
and `tensor_copies`, and a compiled segment with matrices of block-scaled E4M3 weights (`f8e4m3_b128`, the 128 x 128 scale-block and the
per-channel / per-tensor FP8 checkpoints after import) that the hand-written FP8 GEMM reads in its 128-row groups. Segments without such
weights (every non-FP8 model) compile exactly as before. Measured: about +1.5-2.5 % serving throughput at 64 streams, perplexity +0.77 %
(Qwen3-8B-FP8) and +0.26 % (Qwen3.8-27B-FP8) against f32 scales; vLLM's own E8M0 path, its default, is +0.95 %. The opt-out keeps the
checkpoints' f32 scales and the promoting GEMM, bit for bit the earlier default:

| where | setting |
|---|---|
| code | `TaskLoadOptions.BlockScaledFp8 = false` (`LlmLoadOptions`, ...), or `block_scaled_fp8` in `DisabledFeatures` |
| CLI (`sharplight`, `run`, `serve`, `segment-worker`) | `--fp8-block-scaled off` (same as `--disable-features block_scaled_fp8`) |
| hosting JSON (`sharplight serve --config`, Sharplight.Hosting) | `"blockScaledFp8": false` in `defaults` or per model |
| environment (process-wide override) | `SHARPLIGHT_FP8_BLOCK_SCALED=0` (off) / `=1` (on even where a load's options turned it off) |

The Vulkan provider has no block-scaled MMA (no `block_scaled_fp8`), so FP8 models on Vulkan keep the f32 scales and their results do
not change; the setting has no effect there.
Round 13 ([benchmarks.md §57.2](benchmarks.md)) revised these limits: a model's prefill and a sustained GEMM run at the board's 200 W
limit, so burst timings of ablations mostly measure clock (the "no code loads" ablation kept 2.7 GHz instead of 2.2 because the
tensor cores multiplied stale shared memory). Per SM cycle `gemm_fp8_native.cu` reaches 65 % of the MMA peak, `gemm_fp8_tma.cu` 76 %
and CUTLASS ~90 %; the block barrier per K block cost 13 points and the per-output promotion 9. Measure FP8 kernels in FLOPs per SM
cycle or sustained (`bench/s57_gemm`), not in bursts. Round 33 ([benchmarks.md §91](benchmarks.md)) on sm_120: the copies alone and the
MMAs alone each reach the 200 W cap, yet no form with fewer bytes per tile pays: 256-row tiles leave room for two 48 KB stages only
(19 % slower), TMA multicast within a 2-CTA cluster is correct but ~10^4 x slow (ptxas warns that it is meant for sm_90a / sm_100a),
and forwarding half a weight tile to the peer block over DSMEM is 60 % slower. On identical operands CUTLASS's f32-scale blockwise
kernel is 8-13 % faster than `gemm_fp8_tma.cu` and as fast as ours with the outputs dead (stores and promotion dropped), so the gap is
the promotion's and epilogue's schedule, which ptxas 13.2 does not take from the source order.

**CUDA graphs** (`launch_graphs`). A segment records its launches once per symbol binding (kernel, grid, arguments); positions and
lengths are runtime scalars uploaded before the launches, so every decode step of a sequence repeats one recording. When a recording
runs a second time, `CudaSegment` captures its launches into a CUDA graph (`cuStreamBeginCapture` / `cuGraphInstantiateWithFlags`)
and from then on replays it with one `cuGraphLaunch`; recordings that run once (prefill chunks of new lengths) are never captured.
Qwen3-0.6B decode: 622 → 645 tok/s (GPU time per step 1.51 → 1.46 ms: the launch gaps between ~300 kernels shrink); the 8B models
gain under 1 % (19.38 → 19.27 ms per step). The host time per step (wall minus GPU time) was already ~0.1 ms with the recordings.
Disabled with `launch_graphs` in `SHARPLIGHT_DISABLE_FEATURES` / `DeviceOptions.DisabledFeatures`, and in profile and per-launch sync
modes.

**Parallel branches in captured graphs** ([benchmarks.md §74](benchmarks.md)). The recording takes the Vulkan recorder's launch order and
hazard batches (`HazardTracker.Order`: independent launches moved into one batch, a batch boundary where a launch reads or writes what an
earlier launch of the batch touches), and the capture runs a batch's launches side by side: the first on the segment's stream, the others
on three side streams forked from it by an event and joined back before the next batch (`CudaConcurrentLanes`), so the graph has parallel
branches where the program has independent launches (sibling matmuls and their activation copies, split-K parts). Launches of a batch of
several are not programmatic dependents. Uncaptured runs keep one stream. Qwen3-8B Q4_K_M decode steps at 1 / 16 / 32 / 64 sequences
-2.6 / -2 / -4 / -6.7 % (64: 24.69 → 23.03 ms), Qwen3.8 27B -1.5-2 %, Gemma 4 31B -1.2-2.7 %; greedy text byte-identical.
`SHARPLIGHT_CUDA_CONCURRENT=0` keeps the captured launches on one stream.

**Dependent launches** (`dependent_launches`, compute capability 9.0+). From 9.0 the translator starts every kernel (Slang output
and CUDA C++) with `griddepcontrol.wait; griddepcontrol.launch_dependents;` and marks its PTX with a header line
(`// sharplight-dependent`). The segment launches such kernels with `cuLaunchKernelEx` and the programmatic-stream-serialization
attribute, so a kernel's launch and block setup overlap the previous kernel's last blocks. Each kernel then waits for that kernel's
completion and memory at its entry. CUDA graph capture keeps the programmatic edges. Without the attribute both instructions are
no-ops, so the code serves plain launches too. Kernels compiled before (no header line), and drivers without `cuLaunchKernelEx`,
fall back to plain launches. A launch is a dependent only when it and the previous launch each fit in one wave (grid within the
resident blocks of the driver's occupancy query). Launching every kernel early cost Qwen3-8B-FP8 up to 2 % at 16-64 sequences:
waiting blocks take SM slots. Decode steps change by -0.7 to +0.2 % ([benchmarks.md §58.3](benchmarks.md)), because CUDA graphs had
already left few gaps. Disabling the feature launches every kernel after the previous one.

**Launch bounds** of generated kernels. The translator adds `__launch_bounds__(threads, blocks)` to the kernels a rule names (source
file and entry; `NvrtcTranslator.DefaultMinBlocks`): by default one block per SM for the expert-grouped skinny GEMMs (`gemm_sk` of the
MoE sources). Without the bounds ptxas schedules them for full occupancy and their weight stream stalls; with them gpt-oss-20b's decode
steps at 16 / 64 sequences went from 12.0 / 17.7 to 10.1 / 16.1 ms (the Vulkan provider's: 9.9 / 16.7-17.3 ms). The rules are part of the translator's cache key (`SHARPLIGHT_CUDA_MIN_BLOCKS` replaces them).

**Decode attention batch** (`DEC_R`). `develop` round 11 set the key rows per lane group and batch of the decode attention to 1 for
batched decode on Vulkan. The lowering now picks it per target: 1 on SPIR-V and MSL, 4 on PTX, where the CUDA form keeps more loads in
flight (Qwen3-0.6B single-sequence decode 1.68 → 1.46 ms per step with graphs; `SHARPLIGHT_ATTN_DECODE_CFG=...,rows` overrides it).
The split configuration differs per target too: 1024 target workgroups, 128 minimum keys per range and 64 ranges on SPIR-V and
MSL, 1024 / 256 / 128 on PTX up to head size 64 (512 / 128 / 64 above: the larger configuration cost Qwen3-8B-FP8 1.5-4.4 %), where the 256-key minimum keeps short batched contexts on one range per row (no combine) while long
contexts spread over twice the workgroups of the 512-workgroup target before (gpt-oss-20b at 128K keys, 1 / 4 streams 9.99 / 25.52
→ 9.42 / 24.69 ms per step, [benchmarks.md §65.4](benchmarks.md)).

## Provider selection

- **Registration.** `services.AddCudaProvider(o => ...)` (`CudaProviderOptions`). `AddSharplight()` does **not** add it: the entry
  package `Sharplight` stays free of the 110 MB NVRTC, and applications opt in by referencing `Sharplight.Cuda`. The CLI and the test
  suites register it after the Vulkan provider.
- **Default stays Vulkan.** Providers are discovered in registration order and the samples pick the first available GPU provider, so
  with CUDA registered after Vulkan, Vulkan remains the default on NVIDIA GPUs. CUDA runs when a host picks it explicitly
  (`sharplight --provider cuda`, `ExecutionProviderCatalog.TryFind("cuda", ...)`), and becomes the GPU provider when Vulkan is
  unavailable (no Vulkan loader, e.g. a headless container with only the CUDA driver libraries mounted).
- **Per device, later.** Both providers list the same GPU; the CUDA device reports the same UUID as the Vulkan one (`DeviceInfo.Uuid`),
  so a host can match them (Vulkan writes it as 32 hex digits, CUDA as a dashed GUID: compare them without dashes and case, as Studio does). The plan for making CUDA the default is per device and per measured win: a provider-preference policy in
  the catalog keyed by vendor and compute capability (e.g. "cuda first for compute capability ≥ 8.9 once the FP8 GEMMs land"), applied
  by the samples and Studio, with `--provider` and a Studio setting as overrides. It flips only for a device class after CUDA is shown
  faster and the Families and kernel suites are green on it. Phase 1 changes no default. Studio has the setting and the hook
  ([below](#studio)): a provider per GPU, `IProviderPerformanceHints` for the policy.
- **Disable switch.** Not registering the provider, `CudaProviderOptions`, or hiding GPUs with `CUDA_VISIBLE_DEVICES=""`. Features
  are switched off per device like everywhere (`SHARPLIGHT_DISABLE_FEATURES`).
- **Fallback.** A kernel NVRTC or the driver rejects fails the segment's compilation with a `KernelCompilationException`, as on Vulkan;
  hosts fall back to the Vulkan provider or the CPU as they do for any provider error.

## Packaging and licensing

| package | contents | size |
| --- | --- | --- |
| `Sharplight.Cuda` | the managed provider (`Sharplight.Cuda.dll`), depends on `Sharplight.Shaders` and `Sharplight.Cuda.Runtime` | 72 KB |
| `Sharplight.Cuda.Runtime` | native assets per RID: NVRTC with its builtins library and the six floating-point type headers, plus NVIDIA's license | see [sizes](#package-sizes) |
| `Sharplight.Cuda.Blas.<rid>` (optional) | cuBLASLt (`libcublasLt.so.13` / `cublasLt64_13.dll`) for the [library GEMM](#library-gemm-cublaslt), plus NVIDIA's license | 541 MB unpacked (linux-x64) |

`Sharplight.Cuda.Runtime` (`src/Sharplight.Cuda.Runtime`) follows the pattern of `Sharplight.Vulkan.MoltenVK`: nothing is committed;
MSBuild downloads the pinned archives from NVIDIA's redistributable server
(`https://developer.download.nvidia.com/compute/cuda/redist/`, manifest `redistrib_13.0.2.json`), checks each against the SHA-256 of
the manifest, and extracts only the files it ships. A build fetches the archive of the build machine's runtime when the machine has an
NVIDIA driver (or with `-p:SharplightBundleCudaRuntime=true`) and copies the files next to the application; `dotnet pack` fetches all
three runtimes. To update, change `CudaRedistVersion`, the component versions and the SHA-256 values together.

What we ship, and under which clause of the **NVIDIA CUDA Toolkit EULA** (the license file of the archives, shipped as
`LICENSE-NVIDIA-CUDA.txt` and set as the package's license, which NuGet makes the consumer accept):

| file (per RID) | Attachment A component | archive |
| --- | --- | --- |
| `libnvrtc.so.13`, `libnvrtc-builtins.so.13.0` (linux-x64, linux-arm64) | NVIDIA Runtime Compilation Library and Header (NVRTC): `libnvrtc.so`, `libnvrtc-builtins.so` | `cuda_nvrtc` 13.0.88 (`linux-x86_64`, `linux-sbsa`) |
| `nvrtc64_130_0.dll`, `nvrtc-builtins64_130.dll` (win-x64) | NVRTC: `nvrtc.dll`, `nvrtc-builtins.dll` | `cuda_nvrtc` 13.0.88 (`windows-x86_64`) |
| `cuda_fp16.h`, `cuda_fp16.hpp`, `cuda_bf16.h`, `cuda_bf16.hpp`, `cuda_fp8.h`, `cuda_fp8.hpp` | CUDA Floating Point Type Headers | `cuda_cudart` 13.0.96 |

The EULA (v13.5 text, January 2026) grants distribution in section 1.1.1 item 3 ("Distribute those portions of the SDK that are
identified in this Agreement as distributable"), lists the distributable files in **Attachment A** (section 2.6, referenced from 2.2
"Distribution"), and sets the conditions in **1.1.2 "Distribution Requirements"**: the application must have material additional
functionality beyond the SDK portions, the distributed portions may only be accessed by the application, and the license terms must
flow down to the users. Section 2.3 allows Linux files to be redistributed provided they are not modified (we copy them byte for byte,
only renamed to their SONAME). `THIRD-PARTY-NOTICES.md` in the package repeats this table.

Not shipped, and why:

- **The driver** (`libcuda.so.1`, `nvcuda.dll`, `libnvidia-ptxjitcompiler.so`): part of the user's driver installation. Attachment A
  lists them, but a driver library from another driver version must never be loaded.
- **cudart**: the provider uses the driver API only, so the runtime library is not needed.
- **cuBLAS** (`libcublas.so.13`, 54 MB): the library GEMM calls cuBLASLt directly, which does not need it. cuBLASLt itself is the
  optional `Sharplight.Cuda.Blas` package (`src/Sharplight.Cuda.Blas`, same pattern as the runtime package: the pinned `libcublas`
  13.1.0.3 archive of the CUDA 13.0 Update 2 manifest, SHA-256 checked, only `libcublasLt` extracted; "CUDA BLAS Library" in
  Attachment A). `libcublasLt.so.13` is 541 MB (SASS for every architecture; the archive is 839 MB with the static libraries) and
  375 MB deflated, so it is packed per runtime (`dotnet pack -p:CudaBlasRid=linux-x64` gives `Sharplight.Cuda.Blas.linux-x64`;
  likewise `linux-arm64`, `win-x64`), and even then exceeds nuget.org's 250 MB per package: it can be published to a feed without that
  limit, or fetched on first use from NVIDIA's server into the storage location (the same pinned archive and SHA-256), which is the
  plan for Studio. Builds of this repository copy it next to the application only
  with `-p:SharplightBundleCudaBlas=true`; `SHARPLIGHT_CUDA_BLAS_DIR` names another directory with it. Neither `Sharplight.Cuda` nor
  Studio depends on it.
- **nvJitLink** is not needed: the driver's `cuLink*` API links PTX into a cubin.

### Why NVRTC ships

The kernels are specialized at load time: the lowering picks variants and injects shapes, strides, tile sizes, epilogues and generated
elementwise code as macros or source, so the set of kernels a model needs is only known when it is lowered (a typical LLM uses 60-150
distinct kernels out of an unbounded family). Compiling "every kernel at build time" is therefore not possible; what replaces it is the
two-stage contract above: a model precompiled for a profile carries its PTX, a model precompiled on a device its cubins, and either loads
with the driver alone. NVRTC is what lets an arbitrary model, or a precompile step on the user's machine, produce that PTX without a
toolkit, so it ships like PyTorch's wheels ship it (`nvidia-cuda-nvrtc`). Hand-written CUDA kernels of later phases with a fixed set of
specializations will be compiled at build time with `nvcc` into cubins for the target SMs plus PTX, and embedded.

`Sharplight.Cuda` depends on `Sharplight.Cuda.Runtime`, so a `PackageReference` to the provider brings NVRTC and the headers along
(`runtimes/<rid>/native/` of the output, where the provider looks for them). The provider project references the runtime project as a
plain `ProjectReference`: NuGet leaves references with `ReferenceOutputAssembly="false"` out of the restore graph and therefore out of
the package's dependencies (the runtime project's assembly is empty, nothing references it at run time, and its package has no `lib/`).
In this repository that reference puts NVRTC next to the executables of the CLI, the samples and the test projects on machines with
an NVIDIA driver. An application that only loads models precompiled for its GPUs (PTX and cubins in the `.slm`) needs no NVRTC and can
leave the runtime package out with a direct `<PackageReference Include="Sharplight.Cuda.Runtime" ExcludeAssets="all" />`.
`Sharplight.Cuda.Blas.<rid>` stays optional (`PrivateAssets="all"` on its reference).

### Package sizes

Measured on `cuda/provider-r1` (`dotnet pack`, CUDA 13.0 Update 2 redistributables):

| | size |
| --- | --- |
| `Sharplight.Cuda` nupkg | 72 KB (+ 26 KB symbols) |
| `Sharplight.Cuda.Runtime` nupkg (all three RIDs) | 130.5 MB compressed, 315.5 MB unpacked (nuget.org allows 250 MB per package) |
| linux-x64 files | `libnvrtc.so.13` 109.3 MB, `libnvrtc-builtins.so.13.0` 4.4 MB, headers 0.8 MB: 114.5 MB |
| linux-arm64 files | 99.5 + 4.4 + 0.8 = 104.7 MB |
| win-x64 files | `nvrtc64_130_0.dll` 91.0 MB, `nvrtc-builtins64_130.dll` 4.5 MB, headers 0.8 MB: 96.2 MB |
| linux-x64 files compressed | 44.9 MB with deflate (zip, Velopack), 34.3 MB with zstd -19 (AppImage squashfs) |

A self-contained linux-x64 publish of the CLI with the CUDA provider is 278 MB, of which 114.5 MB are the CUDA runtime files.

### Studio

Studio ships with the CUDA provider (decision of 2026-11-19; the package cost below is accepted): `Sharplight.Cuda` and the NVRTC redistributables
of `Sharplight.Cuda.Runtime` are part of the Linux packages (AppImage and tar.gz, x64 and ARM64) and the Windows x64 packages (Velopack Setup.exe
and the portable zip), with `THIRD-PARTY-NOTICES-Cuda.md` and NVIDIA's `LICENSE-NVIDIA-CUDA.txt` next to the executable. macOS and Windows on
ARM have no NVIDIA driver and carry none of it (`StudioCuda` in `Sharplight.Studio.csproj`; `packaging/check-cuda.sh` fails the packaging when a
runtime has the wrong set). The package cost is about 115 MB installed per platform, about 34 MB in the AppImage and about 45 MB in each full
Velopack package (deltas carry it only when NVRTC changes); measured sizes are in [studio.md](studio.md#installing-and-updating).

Studio registers the provider after Vulkan (`AddStudioProviders`), so Vulkan stays the default on every NVIDIA GPU. Settings > Engine > Compute
devices has a provider per GPU (Automatic, Vulkan, CUDA); Automatic is Vulkan until `IProviderPerformanceHints` (the hook for the per-device policy
above: a catalog keyed by vendor and compute capability) names CUDA for the GPU. A driver older than CUDA 13.0, a CUDA that does not initialize or
a load that fails on CUDA hands the GPU to Vulkan, with a note in the device list and a line in `studio.log`. Precompilation covers CUDA devices
(PTX, then a cubin for the device), and the package inspector lists both stages. `sharplight-studio --smoke --smoke-provider cuda` runs the smoke
test on a CUDA device. Details in [studio.md](studio.md#cuda-and-the-provider-per-gpu).

## Tests

- `tests/Sharplight.Cuda.Tests` (new): Slang → CUDA C++ with the block-size header; NVRTC → PTX; the compile-only device of a profile
  compiles every kernel of a transformer block, attention, experts and a convolution for compute capabilities 7.5, 8.9 and 12.0 (NVRTC
  only, no GPU: runs on CI); with a GPU, device listing, an MLP block against the CPU provider, and the device stage (cubins from a
  portable set serve every kernel of a later load without linking).
- The GPU kernel suite `tests/Sharplight.Vulkan.Tests` is provider-neutral (graphs against the CPU oracle) and runs on the CUDA device
  with `SHARPLIGHT_TEST_PROVIDER=cuda` (`CUDA_VISIBLE_DEVICES` selects the GPU); tests of Vulkan internals skip.
- `tests/Sharplight.Families.Tests`: `FamilyGenerationTests.Generation_matches_reference_on_cuda` (the golden outputs of the reference
  implementations) and `CudaParityTests` (greedy tokens on CUDA against Vulkan on the same GPU), opt-in with `SHARPLIGHT_TEST_CUDA=1`.

Results on `cuda/provider-r3` after merging `develop` at `79fccbd1` (RTX PRO 4500 Blackwell, compute capability 12.0, driver
595.112.07, NVRTC 13.0.88, cuBLASLt 13.1.0.3 through `SHARPLIGHT_CUDA_BLAS_DIR`):

| suite | result |
| --- | --- |
| `Sharplight.Vulkan.Tests` on CUDA (`SHARPLIGHT_TEST_PROVIDER=cuda`, one process per class, hand-written GEMMs, 99 KB shared memory, library installed) | 1624 tests: 1578 pass, 0 fail, 46 skip (Vulkan internals, 5 workgroup-matrix GEMMs, 2 opt-in large tests, benchmarks) |
| `Sharplight.Vulkan.Tests` on Vulkan (GPU 1, unchanged provider) | 1624 tests: 1598 pass, 0 fail, 26 skip (as on `develop`) |
| `Sharplight.Cuda.Tests` (library GEMM and the hand-written f16 GEMM, single and dual, against the CPU; profiles of 7.5 / 8.9 / 12.0) | 13 / 13 pass |
| `CudaParityTests` (greedy, 32 tokens, CUDA against Vulkan, SIMT kernels on both) | 4 / 4 identical |
| `FamilyGenerationTests.Generation_matches_reference_on_cuda` (reference outputs) | 315 cases: 215 pass, 0 fail, 100 skip (models not on the machine) |

`CudaParityTests` switches the tensor cores off on both providers: with them, Llama-3.2-1B-Instruct Q8_0 diverged at token 6 of 32, at
a near tie (top-2 logits 0.0009 apart, the providers' logits differ by up to 0.007), since the CUDA fragments and the Vulkan driver's
cooperative matrices round and order the products differently. The tensor-core kernels are checked against the CPU on CUDA by the
kernel suite and the reference generations.

A kernel fault leaves a CUDA context unusable ("sticky" errors such as `misaligned address`), so every later test of the same process
fails at its first driver call; running the suite one class per process isolates a fault to its class. Faults found: the int-dot
alignment (phase 1), and the Q8_0R, FP8 and int8 GEMM row alignments of phase 2 ([tensor cores](#tensor-cores)).

### ONNX on CUDA

ONNX models run on CUDA through the same importer, IR and `GraphLowerer` as on Vulkan; nothing in the CUDA provider is ONNX
specific. The ONNX suites and the vision and audio suites take the kernel suite's switch through `tests/Shared/TestGpu.cs`:
`SHARPLIGHT_TEST_PROVIDER=vulkan|cuda|metal` (default `vulkan`), `SHARPLIGHT_TEST_DEVICE` for the device index. The tests name their
devices `cpu` and `gpu` (`ZooTests.Zoo_Gpu`, `NodeConformanceTests.Gpu`, ...); the conformance skip list and report use the
provider's name. `bench/onnx/SurveyRunner` takes `cuda` as provider.

```sh
SHARPLIGHT_TEST_PROVIDER=cuda CUDA_VISIBLE_DEVICES=0 dotnet run --project tests/Sharplight.Import.Onnx.Conformance -c Release --no-build \
  -- -method "*NodeConformanceTests.Gpu" -method "*ExtraConformanceTests.Gpu"
SHARPLIGHT_TEST_PROVIDER=cuda dotnet run --project tests/Sharplight.Import.Onnx.Tests -c Release --no-build -- -class "*.ZooTests"
dotnet bench/onnx/SurveyRunner/bin/Release/net11.0/SurveyRunner.dll ~/models/onnx/ref/zoo/convnext-tiny.json cuda
```

Results (2026-11-19, branch `cuda/onnx-parity`, RTX PRO 4500 Blackwell, driver 595.112.07; one class per process; cases whose
model or onnxruntime reference is not on the machine skip, the same on both providers):

| suite | Vulkan | CUDA |
| --- | --- | --- |
| `Import.Onnx.Conformance` (1884 backend node tests + 214 extra tests, GPU theories) | 2077 pass, 0 fail, 21 skip | 2077 pass, 0 fail, 21 skip |
| conformance report: tests passing without the CPU fallback | 2037 | 2037 (the same ops fall back) |
| `Import.Onnx.Tests` (16 classes, 181 tests incl. the CPU ones) | 153 pass, 0 fail, 28 skip | 153 pass, 0 fail, 28 skip |
| `Vision.Tests` `YoloDetectorTests` (Ultralytics ONNX exports, also through a package) | 5 / 5 | 5 / 5 |
| `AudioModels.Tests` `SileroVadTests`; `TextToSpeech.Tests` `CosyVoiceTests.Onnx_encoders_match_onnxruntime` | 4 / 4; 2 / 2 | 4 / 4; 2 / 2 |
| `SurveyRunner` over the 23 reference manifests on the machine (ref, ref/zoo, ref/survey) | 22 run, 1 import failure | the same 22 run, the same import failure |

The 21 conformance skips are the reviewed list in `conformance-skips.txt` (image decoder, Bernoulli and training Dropout, 64-bit
BitCast), identical on both providers; the report's statuses (2077 pass, 6 wrong, 15 import failures, i.e. the skipped tests) and
the CPU fallback placement are identical test by test. On every survey model the CUDA output's relative RMS against onnxruntime is
within a few percent of the Vulkan one (resnet50 1.7e-6 / 1.7e-6, ViT-B/16 1.2e-6 / 1.2e-6, BERT-base 2.1e-6 / 2.3e-6, whisper-tiny
encoder 3.0e-6 / 2.8e-6, convnext-tiny 3.3e-6 / 3.2e-6, fasterrcnn-12 1.9e-7 / 1.9e-7, phi-3.5-mini q4f16 1.2e-4 / 1.2e-4), so CUDA
meets the same tolerances; no test needed a looser one. `ssd-10` differs from the reference by the same 2.1e-2 on both providers in
`SurveyRunner`, which compares near-tied detections in order; `SurveyModelTests` aligns tied scores and passes on both.
`gliner-small` fails at import on both (a runtime `ConstantOfShape` extent, [onnx-support.md](onnx-support.md)).

Latency (batch 1, minimum of `Run` plus output read-back, same GPU, no ONNX-specific tuning; Vulkan / CUDA in ms): resnet50
0.72 / 0.81, mobilenetv2 0.32 / 0.34, efficientnet-lite4 0.64 / 0.76, ViT-B/16 2.80 / 2.29, BERT-base (16 tokens) 1.38 / 1.18,
whisper-tiny encoder 2.13 / 1.92, convnext-tiny 1.41 / 1.15, yolov10n 0.99 / 1.07, yolov8n 1.09 / 1.18, segformer-b0 2.96 / 2.87,
mobilesam encoder 7.68 / 7.55, fasterrcnn-12 10.6 / 12.3, maskrcnn-12 11.1 / 12.7, ssd-10 15.3 / 15.2, gemma3-270m 3.49 / 3.17,
qwen2.5-0.5b q4 3.53 / 3.00, phi-3.5-mini q4f16 4.90 / 4.65. Transformers and convnext are 5-18 % faster on CUDA; the small CNNs
and the two-stage detectors are 5-18 % slower (not profiled: this round was correctness only). Compile times are lower on CUDA once
the kernel cache is warm (convnext-tiny 0.7 s against 1.5 s).

Gaps: none specific to CUDA in these suites. Not covered on this machine: the ZooTests / ReferenceTests / SurveyModelTests cases
whose onnxruntime references are not present (18 + 10 skips, the same on Vulkan). Running the suites on a GPU without resizable BAR
found a Vulkan bug: staging blocks did not fall back to host memory when the 256 MiB BAR heap was full, so on Vulkan 724 of the 2098
conformance tests (every model with weights) failed to load; `MemoryAllocator` now retries such a block in host memory.

## Driver-only proof

`packaging/smoke/cuda-driver-only.sh <published app> <model> [arguments]` runs a self-contained published CLI so that nothing of a
CUDA toolkit can be found: an empty environment (`env -i`: no `PATH` entries of a toolkit, no `LD_LIBRARY_PATH`, no `CUDA_PATH`), a new
empty `HOME` (no per-user kernel cache, so every kernel is compiled in the run: Slang, NVRTC, the driver's JIT) and the dynamic loader
without its cache (`ld.so --inhibit-cache`, through the small `hostfxr-launch.c` launcher, since under `ld.so` the apphost cannot find
its own directory). Without the cache, `/etc/ld.so.conf` entries such as `/usr/local/cuda/lib64` or a toolkit's forward-compatibility
driver are not searched; only the app directory and the loader's built-in system directories, which hold glibc and the driver's
user-space libraries. The script then lists every CUDA library the process mapped and fails if one came from a toolkit directory.

This machine (RunPod container, CUDA 13.2 toolkit installed at `/usr/local/cuda` and registered in `/etc/ld.so.conf.d`, including its
`compat/libcuda.so`) allows neither `mknod` nor mount namespaces, so a chroot with only the driver libraries was not possible; the
loader isolation above is the equivalent. Run on GPU 0 (`CUDA_VISIBLE_DEVICES=0`), Qwen3-0.6B Q8_0, `--provider cuda -n 16`:

```
Compiling 110 / 110 (NVIDIA RTX PRO 4500 Blackwell)
Compiled 'decode' in 31121 ms
The capital of France is Paris. The capital of France is also the capital of the Republic of France.
CUDA / NVIDIA libraries the process mapped:
  <app>/libnvrtc-builtins.so.13.0
  <app>/libnvrtc.so.13
  /usr/lib/x86_64-linux-gnu/libcuda.so.595.112.07
  /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.595.112.07
  /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.595.112.07
OK: NVRTC from the app directory, the driver API from the driver installation
```

The same run with the loader cache (a plain start of the app) maps `libcuda.so` from `/usr/local/cuda-13.2/compat/`: on a machine
whose administrator registered the toolkit's forward-compatibility driver, the provider uses it, as every CUDA application does
(`dlopen("libcuda.so.1")`). Where containers are available, the strict form is
`docker run --gpus all -v <publish dir>:/app ubuntu:24.04 /app/Sharplight.Cli ...`: the NVIDIA container toolkit mounts only the
driver's libraries and device nodes into a plain image.

## First numbers

Phase 1 (SIMT kernels on CUDA); see [phase 2 numbers](#phase-2-numbers) for the current state.

GPU 1 of the RunPod machine (RTX PRO 4500 Blackwell, 82 SMs, nothing else running), `sharplight <model> --bench 512` (prefill of 512
tokens: best of 5; decode: 64 steps after the prefill), three runs each, medians of the three shown (they agree within 2 %). Raw output
of the runs in `/workspace/work/cuda-r1/logs/bench` on the machine. "Vulkan SIMT" is the Vulkan provider with the features the CUDA
provider does not report switched off (`--disable-features matrix_*,bf16,fp8,subgroup_size_control`), i.e. the same kernels as CUDA.

| model | provider | prefill 512 tok/s | decode tok/s (median) | GPU ms per decode step |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | Vulkan (default: tensor cores) | 61,565 | 577 | 1.64 |
| | Vulkan SIMT | 9,804 | 549 | 1.63 |
| | **CUDA** | **12,159** | **331** | 2.93 |
| Llama-3.2-1B-Instruct Q8_0 | Vulkan (default) | 42,611 | 465 | 2.04 |
| | Vulkan SIMT | 6,525 | 468 | 2.03 |
| | **CUDA** | **7,729** | **364** | 2.66 |

- **Prefill.** With the same SIMT kernels, CUDA is 18-24 % faster than Vulkan (12.2k against 9.8k, 7.7k against 6.5k tok/s); against
  Vulkan's tensor-core GEMMs it is 5x slower, which is what phase 2 (tensor cores on CUDA) closes.
- **Decode** is 1.3-1.8x slower on CUDA. It is GPU time (the event-timed step), not launch overhead (~300 launches per step, host time
  per step equals GPU time within 0.1 ms on both providers). The per-kernel profile (`SHARPLIGHT_CUDA_PROFILE=1` against
  `SHARPLIGHT_VK_PROFILE=1`, Qwen3-0.6B, 64 steps) puts most of the gap in the decode attention: 92.2 ms on CUDA against 25.0 ms on
  Vulkan (50 against 14 µs per layer and step) of a 104 ms total gap (236.6 against 132.6 ms). The kernel is the same Slang source; its
  CUDA form has no spills (93 registers, `ptxas -v`), but Slang's CUDA backend makes the active mask of every wave operation explicit:
  the decode attention's CUDA source carries 664 `__ballot_sync` calls that track the mask through every branch and loop iteration,
  which SPIR-V does not need. Removing that (a Slang option, uniform-control-flow annotations, or a CUDA form of the decode attention)
  is the first decode item of phase 2.

Load: the first load compiles every kernel (Slang, then NVRTC: about 0.3-0.6 s per kernel on one thread, in parallel); Qwen3-0.6B's
110-126 kernels took 31-67 s depending on the machine load (the per-user artifact store then serves PTX and cubins, and the next load
compiles nothing: 3.3 s). Precompiled packages skip it entirely.

## Phase 2 numbers

Same GPU and procedure ([benchmarks.md §51](benchmarks.md): GPU 1, prefill median of 5, decode median of 64 steps, identical weights
for every engine, vLLM 0.33.0 fresh on the same GPU). tok/s, prefill / decode:

| model | rows | Vulkan | CUDA | CUDA / Vulkan | vLLM | CUDA / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | 512 | 61,120 / 575 | 58,889 / 623 | 0.96 / 1.08 | | |
| Llama-3.2-1B Q8_0 | 512 | 42,373 / 448 | 44,544 / 483 | 1.05 / 1.08 | | |
| gemma-3-1b Q8_0 | 2048 | 56,892 / 451 | 58,022 / 453 | 1.02 / 1.00 | | |
| Qwen3-8B (bf16) | 512 | 7,286 / 52.7 | 8,194 / 51.2 | 1.12 / 0.97 | 8,086 / 51.7 | 1.01 / 0.99 |
| | 2048 | 7,718 | 7,907 | 1.02 | 8,807 | 0.90 |
| Qwen3-8B-FP8 | 2048 | 10,993 / 86.9 | 11,458 / 85.3 | 1.04 / 0.98 | 14,948 / 87.3 | 0.77 / 0.98 |
| Llama-3.1-8B (bf16) | 2048 | 7,824 / 53.4 | 7,902 / 51.7 | 1.01 / 0.97 | 8,736 / 52.5 | 0.90 / 0.98 |

Prefill went from 0.18-0.20x of Vulkan (phase 1) to 0.96-1.12x, decode of the 1B models from 0.57-0.78x to 1.00-1.08x. Where CUDA
still trails: Qwen3-0.6B's prefill (4 %), the 8B decode matvecs (2-3 %, weights streamed at 848 against 875 GB/s), and the prefill
attention `fe` (5-8 %). Against vLLM, FP8 prefill (0.77x) is the large gap; bf16 is at 0.90-1.01x.

## Phase 3 numbers

Same GPU and procedure ([benchmarks.md §53](benchmarks.md): GPU 1, prefill median of 5, decode median of 64 steps, identical weights
for every engine). vLLM 0.33.0: the section 51 runs for the 8B models, gpt-oss-20b fresh on the same GPU. "CUDA" is the default
(hand-written GEMMs, no cuBLASLt needed), "CUDA + cuBLASLt" the same build with `--disable-features native_gemm_f16,native_gemm_fp8`
and the library installed. tok/s, prefill / decode:

| model | rows | Vulkan | CUDA | CUDA / Vulkan | CUDA + cuBLASLt | vLLM | CUDA / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | 512 | 61,169 / 554 | 60,416 / 645 | 0.99 / 1.16 | | | |
| | 2048 | 77,676 / 530 | 81,615 / 589 | 1.05 / 1.11 | | | |
| Llama-3.2-1B Q8_0 | 512 | 42,574 / 444 | 44,764 / 490 | 1.05 / 1.10 | | | |
| gemma-3-1b Q8_0 | 2048 | 56,994 / 450 | 57,310 / 464 | 1.01 / 1.03 | | | |
| Qwen3-8B (bf16) | 512 | 7,317 / 52.3 | 8,581 / 51.6 | 1.17 / 0.99 | 8,096 | 8,086 / 51.7 | 1.06 / 1.00 |
| | 2048 | 7,720 | 8,682 | 1.12 | 7,890 | 8,807 | 0.99 |
| | 8192 | 6,830 | 7,528 | 1.10 | 6,940 | 7,678 | 0.98 |
| Qwen3-8B-FP8 | 512 | 10,844 / 87.2 | 11,348 / 87.7 | 1.05 / 1.01 | 11,155 | 14,546 / 87.3 | 0.78 / 1.00 |
| | 2048 | 10,983 | 11,808 | 1.08 | 11,437 | 14,948 | 0.79 |
| | 8192 | 9,288 | 9,791 | 1.05 | 9,556 | 12,316 | 0.79 |
| Llama-3.1-8B (bf16) | 512 | 7,498 / 53.4 | 8,434 / 52.3 | 1.12 / 0.98 | 8,014 | 8,513 / 52.5 | 0.99 / 1.00 |
| | 2048 | 7,805 | 8,721 | 1.12 | 7,916 | 8,736 | 1.00 |
| | 8192 | 6,978 | 7,692 | 1.10 | 7,053 | 7,730 | 1.00 |
| gpt-oss-20b (MXFP4) | 512 | 10,430 / 178.1 | 9,276 / 182.9 | 0.89 / 1.03 | | 12,159 / 172.9 | 0.76 / 1.06 |
| | 2048 | 12,269 | 12,205 | 0.99 | | 16,554 | 0.74 |
| | 8192 | 10,056 | 10,013 | 1.00 | | 15,918 | 0.63 |

* **bf16 without cuBLASLt**: the hand-written f16 GEMM puts bf16 prefill 6-9 % above CUDA with cuBLASLt (whose gate / up pair needs a
  separate SiLU·mul launch) and at vLLM's level (0.98-1.06x).
* **FP8**: 3-4 % faster than phase 2, 1.05-1.08x of Vulkan, but still 0.78-0.79x of vLLM (see the limits in
  [kernels written in CUDA C++](#kernels-written-in-cuda-c)).
* **Decode**: the 1B models 3-16 % above Vulkan (CUDA graphs and `DEC_R` = 4); the 8B models 1-2 % below Vulkan (bf16 GPU time per
  step 19.27 against 18.97 ms) and at vLLM's level.
* **gpt-oss-20b** prefill (MoE, no part of this phase) is at Vulkan's level from 2048 rows and 11 % behind it at 512; against vLLM it is
  the largest gap (0.63-0.76x).

## Phase 4 numbers

Same GPU and procedure ([benchmarks.md §54](benchmarks.md)); only the FP8 paths changed. Qwen3-8B-FP8, tok/s, prefill / decode:

| rows | Vulkan | CUDA phase 3 | CUDA | CUDA / Vulkan | vLLM | CUDA / vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 10,817 / 87.0 | 11,348 / 87.7 | 12,688 / 87.4 | 1.17 / 1.00 | 13,946-14,546 / 87.3 | 0.87-0.91 / 1.00 |
| 2048 | 10,972 | 11,808 | 12,819 | 1.17 | 14,948-14,971 | 0.86 |
| 8192 | 9,282 / 77.0 | 9,791 / 78.1 | 10,494 / 77.7 | 1.13 / 1.01 | 12,316-12,391 / 77.2 | 0.85 |

* **FP8 prefill** +7-12 % over phase 3, 0.85-0.87x of vLLM (DeepGEMM with power-of-two scales, or CUTLASS with the f32 scales: the
  same speed). The GEMMs are the whole remaining gap: 122.7 ms of a 155.3 ms 2048-row prefill against ~98 ms at vLLM's ~290 TFLOPS.
* **Batched decode** (64 sequences, GPU ms per step): CUDA 17.5, Vulkan 16.9, vLLM 14.1 + 1.55 for the LM head outside its
  measured range. Per kernel: the weight GEMMs stream at 680-700 GB/s against vLLM's ~766 (read limit ~780), the decode attention is
  within 7 % of vLLM's split-KV FlashAttention, and vLLM fuses q / k / v into one GEMM.
* **Mixed steps** (64 decode rows plus a prompt chunk of 16-256 rows) 6-20 % faster through the 64-row FP8 tiles from 65 rows; serving
  at 64 streams is unchanged at 0.78x of vLLM (0.84x at 16, 0.92x at 4, 1.00x single), since its steps are pure decode steps or
  prompt batches of 1-2k rows.

## Phase 4 status

State of `cuda/provider-r4` (merged with `develop` at `47f9af79`, builds clean) when the session was interrupted:

- **Done and measured:** dense FP8 scale layouts (128-row weight groups `f8e4m3_b128g` repacked at upload; dense activation-scale
  copy from `quantize_fp8` and the E4M3-writing norms), 64-row FP8 tiles chosen by waves, hand-written FP8 GEMM from 65 rows,
  `SHARPLIGHT_FP8_GROUPED`, `BatchBench --provider cuda`, `bench/s54_vllm_decode_profile.py`. Numbers in
  [phase 4 numbers](#phase-4-numbers) and [benchmarks.md §54](benchmarks.md): Qwen3-8B-FP8 prefill 12,688 / 12,819 / 10,494 tok/s at
  512 / 2048 / 8192 rows (0.85-0.87x vLLM, phase 3 0.79x), decode unchanged (1.00x), serving 0.78x at 64 streams (unchanged).
- **Verified (one GPU, after merging `develop` with phase 3):** `Sharplight.Cuda.Tests` 17 / 17; `Sharplight.Vulkan.Tests` on CUDA,
  one process per class, all 62 classes: 1624 tests, 0 failed, 46 skipped (as in phase 3); the Vulkan SPIR-V of the 225 kernels of
  Qwen3-8B-FP8 and Qwen3-0.6B Q8_0 is byte-identical to `develop`; long-prompt logits of Qwen3-8B-FP8 against Vulkan within the
  provider spread of phase 3. `Sharplight.Cuda` now depends on `Sharplight.Cuda.Runtime` as a package (see
  [packaging](#packaging-and-licensing)).
- **Also verified (after the Vulkan allocator fallback `43367d0e`):** `CudaParityTests` 4 / 4 identical;
  `FamilyGenerationTests.Generation_matches_reference_on_cuda` 315 cases, 0 failed, 100 skipped (models not on the machine); the
  Vulkan suite on Vulkan, one process per class, all 62 classes: 1624 tests, 0 failed, 26 skipped (as on `develop`).
- **Next steps:** the open items of the roadmap's phase 4 (FP8 main loop
  toward CUTLASS's ~290 TFLOPS; decode step: fused q / k / v GEMM, weight streaming, split-KV decode attention; FlashAttention-2
  prefill kernel; MXFP4 expert GEMM). Ideas measured and rejected are listed in §54.3 so they are not retried.

## Diagnostics

`DiagnosticSwitches` of `Sharplight.Cuda`: `SHARPLIGHT_CUDA_RUNTIME_DIR` (directory with NVRTC and the headers, searched first),
`SHARPLIGHT_CUDA_DUMP_SOURCE=dir` (CUDA C++ and PTX of every kernel compiled), `SHARPLIGHT_CUDA_TRACE=1` (module loads, recordings),
`SHARPLIGHT_CUDA_PROFILE=1` (per-launch event timing in `--bench`, summed per launch kind like the Vulkan profile),
`SHARPLIGHT_CUDA_SYNC=1` (synchronize after every launch, so a faulting kernel is named), `SHARPLIGHT_CUDA_MAX_SHARED=bytes` (shared
memory the kernels are lowered for; 49152 keeps the static limit), `SHARPLIGHT_CUDA_BLAS_DIR` (directory with cuBLASLt, searched
first); of `Sharplight.Shaders`: `SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS` (rows from which matmuls take the library GEMM or the hand-written
f16 GEMM, default 256), `SHARPLIGHT_NATIVE_GEMM_MIN_ROWS` (rows from which the hand-written FP8 GEMM serves, default 65),
`SHARPLIGHT_FP8_GROUPED=1|0` (FP8 weights in 128-row groups on every target or none; default: where the hand-written FP8 GEMM runs),
`SHARPLIGHT_NATIVE_MOE_MIN_ROWS` (average routed rows per expert from which the hand-written grouped MXFP4 GEMM serves, default 12),
`SHARPLIGHT_FP8_TMA=0` (128-row FP8 tiles on `gemm_fp8_native.cu` instead of `gemm_fp8_tma.cu`), `SHARPLIGHT_FP8_BLOCK_SCALED=1|0`
(the block-scaled FP8 mode where `block_scaled_fp8` exists: off / on, overriding `TaskLoadOptions.BlockScaledFp8`; default on where it applies), `SHARPLIGHT_FP8_QSTORE_MIN_ROWS=n` (block-scaled mode: rows from
which the dual gate / up GEMM stores E4M3 rows; 0: never, default 256), `SHARPLIGHT_FP8_UE8M0_WEIGHTS=1` (diagnostic: the block-scaled mode's weight
requantization alone), `SHARPLIGHT_FP8_DECODE_KERNEL=0|1` (FP8 matmuls at 9-16 rows on `gemm_fp8_decode.cu`: off / dual pairs too), `SHARPLIGHT_FP8_L2_PREFETCH=n`
(K blocks of its weight rows each FP8 decode block asks L2 for before the dependency wait; slower in every setting measured), `SHARPLIGHT_FP8_TALL_COST` (relative
cost of a 128-row FP8 block in the tile choice), `SHARPLIGHT_ATTN_DECODE_FUSED_WIDTH` (widest head group whose decode attention merges
its split partials in the last workgroup),
`SHARPLIGHT_DISABLE_FEATURES=native_gemm_f16,native_gemm_fp8,native_gemm_moe,tensor_copies,launch_graphs` (back to the phase-2 kernels and per-launch
submission),
`SHARPLIGHT_GEMM_MACROS` (e.g. `GNBUF=2` for double-buffered 64-value K blocks of gemm_cm2), `SHARPLIGHT_CUDA_MIN_BLOCKS=file:entry:blocks[;...]`
(`__launch_bounds__` for matching generated kernels, replacing the default one block per SM for the MoE skinny GEMMs; `none`: no bounds),
`SHARPLIGHT_MULTI_GEMM=0` (no multi-output q / k / v GEMM; `SHARPLIGHT_MULTI_DEBUG=1` lists the groups emitted and skipped),
`SHARPLIGHT_NO_INT_DOT_MULTI_BIAS=1` (the q / k / v biases of int-dot weights as their own launch after the multi-output GEMM instead of in
its store, benchmarks.md §78), `SHARPLIGHT_NO_ROPE_QKV=1` (a layer's query rope, key rope + cache write and value cache write as three
launches instead of `rope_qkv`; `SHARPLIGHT_ROPE_QKV_DEBUG=1` names the layers that keep them and why), `SHARPLIGHT_NORM_WIDE=0` (fused row norms of 4096+ values keep 256-thread blocks at decode-size steps instead of 1024, benchmarks.md §76.3), `SHARPLIGHT_NATIVE_GEMM_BIAS=0` (biased matmuls stay on the generated GEMMs),
`SHARPLIGHT_SPLITK_FUSED=1` / `SHARPLIGHT_MULTI_FUSED=1` (split-K reductions in the GEMM's last workgroup, slower, benchmarks.md §61.4),
`SHARPLIGHT_MULTI_GEMM_TALL=0`, `SHARPLIGHT_MOE_SORTED_STORE=0`, `SHARPLIGHT_MOE_TOKEN_ROWS=0`, `SHARPLIGHT_ATTN_HG=0|1` (benchmarks.md §82);
`SHARPLIGHT_ROPE_QKV_WARP=0|n` (`rope_qkv`'s 64-thread workgroups per row instead of a warp per row, or n rows per warp) and
`SHARPLIGHT_ATTN_MIXED_RUN_ROWS=n` (mean rows per run below which a multi-sequence step takes the mixed attention form, default 64)
(benchmarks.md §90); plus `SHARPLIGHT_LOAD_TIMING`, `SHARPLIGHT_SUBMISSION_BUDGET_MS`, `SHARPLIGHT_CONCURRENT_WORKGROUPS` and
`SHARPLIGHT_DEVICE_FORWARDING` as for Vulkan. `compute-sanitizer` works on the provider as on any CUDA application.

## Roadmap

Phase 1 is correctness on the shared kernels. The phases after it each add CUDA-only fast paths behind neutral device features with the
phase-1 kernels as the fallback, and each flips the default for a device class only when measured faster and green.

1. **Phase 2: tensor cores and decode** (done, branch `cuda/provider-r2`): tensor-core fragments for CUDA (`cuda_mma.slang`), the
   `matrix_*` features from compute capability 8.0 / 8.9, wave operations over `__activemask()`, `exp` as `__expf`, aligned vector
   loads, dynamic shared memory above 48 KB, and cuBLASLt as the optional library GEMM for 16-bit weights. Open from it:
   - the gate / up pair on the library GEMM with its SiLU·mul group and the f16 copy of its output in one launch (the dual gemm_cm2
     fuses them; ~3 % of a bf16 prefill), or one library GEMM over the concatenated gate / up weights;
   - FP8 at vLLM's level (0.77x): cuBLASLt has no 128x128 block-scaled FP8 on compute capability 12.0, so this needs a hand-written
     FP8 GEMM with larger tiles (the dynamic shared memory is in place) or the Blackwell block-scaled `mma.sync ... kind::mxf8f6f4`;
   - the prefill attention `fe` is 5-8 % slower than its SPIR-V form on the NVIDIA Vulkan driver (same kernel);
   - the decode attention's rows per batch (`DEC_R`): `develop` round 11 set it to 1 for batched decode, which costs CUDA's
     single-sequence decode 10 % (Qwen3-0.6B: 1.68 against 1.51 ms per step) and Vulkan about 2 %.
2. **Phase 3: kernels written in CUDA C++** (done, branch `cuda/provider-r3`): `SlangCompiler.CompileCuda` (NVRTC, cached and
   precompiled like every kernel), the hand-written f16 GEMM replacing cuBLASLt (single and dual with the fused SiLU·mul), the FP8
   GEMM with a `cp.async` ring in 96 KB, CUDA graphs for repeated steps (`launch_graphs`), `DEC_R` per target. Open from it:
   - FP8 at vLLM's level (0.79x): a scale layout the GEMM streams densely (the per-row scale loads are its largest cost), then a
     promotion that overlaps the MMAs; the down projection's activation quantization in the dual GEMM's store (needs 128-column
     output tiles);
   - the prefill attention: a flash-attention kernel in CUDA C++ with `cp.async` K / V staging, larger tiles and split-KV for long
     contexts (`fe` was 5-8 % slower on CUDA than on Vulkan in phase 2; it is ~5 % of an 8B prefill at 2048 rows);
   - the 8B decode matvecs (1-2 % behind Vulkan) and the f16 copy of the dual GEMM's output written in its store;
   - gpt-oss-20b prefill (0.63-0.76x of vLLM; MoE).
3. **Phase 4: dense FP8 scales and the batched decode step** (done, branch `cuda/provider-r4`): the 128-row FP8 weight
   groups and the dense activation scale copies, 64-row FP8 tiles from 65 rows, the decode step and vLLM's measured kernel by kernel
   (`BatchBench --provider cuda`, `bench/s54_vllm_decode_profile.py`). Open from it:
   - FP8 GEMM at CUTLASS's ~290 TFLOPS (ours 225-240 in the model): a main loop that overlaps the promotion and the copies with
     the MMAs (TMA and ping-pong consumer groups);
   - the decode step at vLLM's level: weight streaming at the read limit (~10 % on the GEMMs), a fused q / k / v GEMM, a split-KV
     decode attention with `cp.async`;
   - a FlashAttention-2-style prefill kernel in CUDA C++ (`fe` is a quarter of an 8192-token FP8 prefill at 105 TFLOPS), split-K for
     the native f16 GEMM's small grids, the dual GEMM writing the f16 copy, the 8B decode matvecs;
   - gpt-oss-20b prefill (MoE, 0.63-0.76x of vLLM): a grouped MXFP4 expert GEMM in CUDA C++.
   Round 13 (branch `perf/llm-r13`, [benchmarks.md §57](benchmarks.md)): tensor maps in kernel parameters, `tensor_copies`, the
   warp-specialized persistent FP8 GEMM (`gemm_fp8_tma.cu`, default from 9.0): Qwen3-8B-FP8 prefill +9-11 %, 0.99 / 0.88 / 0.86x of
   vLLM at 512 / 2048 / 8192 rows. Open from it:
   - FP8 main loop at CUTLASS's ~90 % of the MMA peak per cycle (ours 76 %): the promotion (64 scale products per thread and K
     block; one per row where a warp's weight scales agree, which needs registers below the 168-register cap of 288 threads) and the
     tile-end stores;
   - the batched decode step (CUDA 17.9, Vulkan 17.3, vLLM 15.6 ms at 64 sequences): the CUDA forms of `gemm_fp8.slang` at decode
     sizes (0.7 ms behind their SPIR-V forms), weight streaming at 520-670 GB/s, a multi-output q / k / v GEMM, the split-K reductions in the GEMM's last
     workgroup or overlapped by programmatic dependent launch.
   Round 14 (branch `perf/llm-r14`, [benchmarks.md §58](benchmarks.md)): the FlashAttention-2-style prefill kernel
   (`attn_prefill_native.cu`, `native_attention`) and programmatic dependent launches (`dependent_launches`). Open from it: the
   multi-output q / k / v GEMM (each weight is its own buffer with one source, so one GEMM over the concatenated rows needs buffers
   holding several weights in every provider's uploader, or a kernel with several weight pointers); the split-K reductions in the
   GEMM's last workgroup; the f32 Q / O of the attention (a window layer's 52 µs at 2048 tokens is mostly their 66 MB).
4. **Phase 5: MoE.** Started on `perf/llm-r12` ([benchmarks.md §56](benchmarks.md)): the hand-written grouped MXFP4 expert GEMM
   (`gemm_moe_native.cu`, raw codes in a `cp.async` ring, decoded per MMA); gpt-oss-20b prefill 0.81-1.07x / 0.74-0.87x / 0.63-0.71x of
   vLLM at 512 / 2048 / 8192 tokens before / after. Open: the prefill attention (`fa` is 16 of 140 ms at 2048 tokens against
   FlashInfer's 4.4 ms, 30 % of an 8192-token prefill), the bf16 projections (q / k / v as one GEMM), the 64-sequence decode step
   (19.8 against vLLM's 16.2 ms incl. LM head), TMA-gathered rows, fused routing / gather, NVFP4 experts, and the Blackwell
   block-scaled instructions (they need UE8M0 activation scales, a numerics change).
5. **Phase 6: launch overhead.** Programmatic dependent launch and asynchronous host copies (CUDA graphs are in phase 3).
6. **Phase 7: FP8 / NVFP4 / MXFP4 tensor cores everywhere** (KV cache in FP8, attention in FP8), and **multi-GPU**: NCCL-free
   peer-to-peer copies (`cuMemcpyPeer`, P2P over NVLink / PCIe) for pipeline and tensor parallelism across the three GPUs.

Throughout, Vulkan keeps its behaviour and speed: shared Slang changes must keep the SPIR-V and MSL output working (Vulkan suite here,
Metal in CI).
