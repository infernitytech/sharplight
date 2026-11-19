# CUDA provider

`Sharplight.Cuda` is a native execution provider for NVIDIA GPUs next to the Vulkan provider. It exists to close the performance gaps
to vLLM that the Vulkan path cannot close on NVIDIA hardware: Vulkan compute on the NVIDIA driver is limited to 48 KB of shared
memory per workgroup, exposes neither the Tensor Memory Accelerator (`cp.async.bulk`, TMA) nor `cp.async`, and has no thread-block
clusters (see [benchmarks.md §44.3](benchmarks.md#443-tried-and-not-kept) and [§46](benchmarks.md#46-llm-round-8-fresh-vllm-baseline-expert-grouped-skinny-gemm-for-moe-decode-moe-range-guard-2026-11-18)).
The FP8 GEMM tiles that would cut L2 traffic, the flash-attention tiles that would hide K/V staging and the MoE GEMMs all need one of
these.

Phase 1 (this document's status, branch `cuda/provider-r1`) is the provider itself: every op of the shared kernel library runs on CUDA,
correct and driver-only, with the same lowering as Vulkan. It is not faster than Vulkan yet ([first numbers](#first-numbers)); the
speed comes with the CUDA-only kernels of the [roadmap](#roadmap).

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
- **Lowering.** `CudaCodeGenerator` calls the shared `GraphLowerer` with warp size 32, 48 KB of shared memory (the static limit, see
  below), the SM count as compute units and the device's features, so the CUDA provider selects the same kernels as the Vulkan provider
  on the same NVIDIA GPU. That makes Vulkan the reference for parity and keeps phase 1 free of new tuning decisions.
- **Memory.** Device memory per buffer (`cuMemAlloc`, zeroed), weights uploaded through two 64 MB pinned staging buffers that the CPU
  fills (reading mapped files, repacking rows with the shared `WeightCopier`) while the GPU copies the other one. Shared weights,
  per-graph states, device forwarding between segments and data-dependent symbols work as on Vulkan.
- **Execution.** Each segment owns a CUDA stream and two timing events. Launches are recorded once per binding of the symbols (kernel,
  grid, packed arguments) and replayed with `cuLaunchKernel`; launches on one stream run in order, so the hazard tracker's barriers are
  not needed. Runtime scalars go through a pinned buffer with one asynchronous copy per step; inputs and outputs are synchronous copies.
  Every driver call first makes the device's primary context current on the calling thread.
- **Module loading.** `CudaDevice` loads each kernel once: its cubin from the model-level cache (`cubin/<kernel hash>`) or the per-user
  artifact store when present, otherwise it links the PTX into a cubin with the driver's JIT (`cuLinkCreate` / `cuLinkAddData` /
  `cuLinkComplete`), records it in the cache (so the next load skips the JIT) and loads it with `cuModuleLoadData`.
- **Device features** use the neutral names: `fp16`, `subgroup_ops` and `int_dot` (`dp4a`) on every listed GPU. The tensor-core
  features (`matrix_f16`, `matrix_bf16`, `matrix_int8`, `matrix_fp8`, `matrix_mma_layout`) and the `bf16` / `fp8` types are not reported
  in phase 1, so the kernels written for them take their SIMT fallbacks, exactly like the Vulkan provider on a GPU without cooperative
  matrices. Slang's CUDA prelude does implement cooperative matrices (16x16x16 fragments on `mma.sync.m16n8k16`: f16 and bf16 with f32
  accumulators, int8, fp8; the register layout is the MMA layout `matrix_mma_layout` describes), and reporting those features from
  compute capability 8.0 was tried: Slang aborts the code generation of the tensor-core GEMM `gemm_cm2` ("internal error E99999:
  unexpected IR opcode during code emit", the same failure as its Metal backend for fragment loads from `uint4` shared arrays, see
  [macos.md](macos.md#slang-8x8-fragment-loads-from-uint4-shared-arrays-on-metal)). Giving those kernels CUDA forms is phase 2.
  `SHARPLIGHT_DISABLE_FEATURES` and `DeviceOptions.DisabledFeatures` work as for every provider, and `kernel_binaries` turns off the
  cubin tier (kernels then load from PTX, JIT-compiled by the driver on every load).
- **Alignment.** CUDA requires vector loads to be aligned to their size; SPIR-V and MSL pointer accesses only assume the element
  alignment. `LoweringOptions.AlignedVectorAccess` (set by the CUDA provider only) keeps kernels whose row strides would misalign their
  16-byte loads off such shapes: the int-dot matvec then needs K to be a multiple of 64 (its quantized activation rows are 1.25 K
  bytes), other K take the f32 matvec. Found by `MatMulTests.OddShapes` (Q8_0, K = 96: "misaligned address").
- **Format-agnostic core.** Nothing outside `src/Sharplight.Cuda` (and the native package `Sharplight.Cuda.Runtime`) knows about CUDA.
  The only shared change is the `KernelTranslator` hook and the CUDA block-size header in `Sharplight.Shaders`.

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

The Slang kernels declare their shared arrays statically, and CUDA limits static shared memory to 48 KB per block, the same limit the
Vulkan driver reports. Phase 1 therefore lowers with 48 KB. Larger tiles (up to 99 KB per block on Ada / Hopper-class consumer parts,
227 KB on H100 / B200, 99 KB on RTX PRO Blackwell: `max_shared_memory_optin` in `sharplight --devices`) need dynamic shared memory
(`extern __shared__`) and `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)` per kernel; that comes with the
hand-written kernels of phase 2, which are where the larger tiles pay off.

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
| `Sharplight.Cuda` | the managed provider (`Sharplight.Cuda.dll`), depends on `Sharplight.Shaders` | 72 KB |
| `Sharplight.Cuda.Runtime` | native assets per RID: NVRTC with its builtins library and the six floating-point type headers, plus NVIDIA's license | see [sizes](#package-sizes) |

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
- **cuBLAS / cuBLASLt** (phase 2): `libcublasLt.so.13` alone is 505 MB (all SASS for every architecture), `libcublas.so.13` 54 MB. They
  are redistributable ("CUDA BLAS Library" in Attachment A) and will go into a separate optional package, `Sharplight.Cuda.Blas`, so
  that neither `Sharplight.Cuda` nor the Studio download carries them unless the GEMM path is wanted.
- **nvJitLink** is not needed: the driver's `cuLink*` API links PTX into a cubin.

### Why NVRTC ships

The kernels are specialized at load time: the lowering picks variants and injects shapes, strides, tile sizes, epilogues and generated
elementwise code as macros or source, so the set of kernels a model needs is only known when it is lowered (a typical LLM uses 60-150
distinct kernels out of an unbounded family). Compiling "every kernel at build time" is therefore not possible; what replaces it is the
two-stage contract above: a model precompiled for a profile carries its PTX, a model precompiled on a device its cubins, and either loads
with the driver alone. NVRTC is what lets an arbitrary model, or a precompile step on the user's machine, produce that PTX without a
toolkit, so it ships like PyTorch's wheels ship it (`nvidia-cuda-nvrtc`). Hand-written CUDA kernels of later phases with a fixed set of
specializations will be compiled at build time with `nvcc` into cubins for the target SMs plus PTX, and embedded.

`Sharplight.Cuda` does not depend on `Sharplight.Cuda.Runtime`: an application that only loads models precompiled for its GPUs (PTX and
cubins in the `.slm`) needs no NVRTC, so it adds the runtime package only when it compiles kernels itself, which is the normal case. In
this repository the provider project references the runtime project (`ReferenceOutputAssembly="false"`), so the CLI, the samples and
the test projects get NVRTC next to their executables on machines with an NVIDIA driver.

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

Results on `cuda/provider-r1` (RTX PRO 4500 Blackwell, compute capability 12.0, driver 595.112.07 / CUDA 13.2, NVRTC 13.0.88):

| suite | result |
| --- | --- |
| `Sharplight.Vulkan.Tests` on CUDA (`SHARPLIGHT_TEST_PROVIDER=cuda`, one process per class) | 1610 tests: 1564 pass, 0 fail, 46 skip (38 Vulkan internals, 5 workgroup-matrix GEMMs, 2 opt-in large tests, 1 diagnostic) |
| `Sharplight.Cuda.Tests` | 8 / 8 pass |
| `CudaParityTests` (greedy, 32 tokens, CUDA against Vulkan with the same feature set) | 4 / 4 identical: Qwen3-0.6B Q8_0, Llama-3.2-1B-Instruct Q8_0, gemma-3-1b-it Q8_0 (GGUF), Qwen3-0.6B (HF) |
| `FamilyGenerationTests.Generation_matches_reference_on_cuda` (reference outputs) | 9 / 9 pass: gemma-4-E2B-it HF (Gemma), tiny Mixtral (MoE), tiny gemma4 MoE HF + GGUF, tiny Qwen2.5-VL HF + GGUF and tiny GLM-4V HF + GGUF (vision-language), tiny Nemotron-H (hybrid Mamba / MoE, with CPU fallback) |
| `Sharplight.Metal.Tests` (MSL generation, Linux) | 7 tests: 4 pass, 3 skip (need macOS) |

A kernel fault leaves a CUDA context unusable ("sticky" errors such as `misaligned address`), so every later test of the same process
fails at its first driver call; running the suite one class per process isolates a fault to its class. The only fault found was the
int-dot alignment above.

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

## Diagnostics

`DiagnosticSwitches` of `Sharplight.Cuda`: `SHARPLIGHT_CUDA_RUNTIME_DIR` (directory with NVRTC and the headers, searched first),
`SHARPLIGHT_CUDA_DUMP_SOURCE=dir` (CUDA C++ and PTX of every kernel compiled), `SHARPLIGHT_CUDA_TRACE=1` (module loads, recordings),
`SHARPLIGHT_CUDA_PROFILE=1` (per-launch event timing in `--bench`), `SHARPLIGHT_CUDA_SYNC=1` (synchronize after every launch, so a
faulting kernel is named), plus `SHARPLIGHT_LOAD_TIMING`, `SHARPLIGHT_SUBMISSION_BUDGET_MS`, `SHARPLIGHT_CONCURRENT_WORKGROUPS` and
`SHARPLIGHT_DEVICE_FORWARDING` as for Vulkan. `compute-sanitizer` works on the provider as on any CUDA application.

## Roadmap

Phase 1 is correctness on the shared kernels. The phases after it each add CUDA-only fast paths behind neutral device features with the
phase-1 kernels as the fallback, and each flips the default for a device class only when measured faster and green.

1. **Phase 2: tensor cores and decode.**
   - `matrix_f16` / `matrix_bf16` / `matrix_int8` / `matrix_fp8` / `matrix_mma_layout` on CUDA. Slang's CUDA prelude already maps
     16x16x16 cooperative matrices to `mma.sync.m16n8k16`; what fails is the code generation of some kernels (`gemm_cm2`: E99999).
     Find the construct (most likely the fragment loads from `uint4` shared arrays, as on Metal), give those kernels a form Slang's
     CUDA backend emits (as `*_FRAG8` did for Metal) without changing their SPIR-V, and report the features from compute
     capability 8.0 (8.9 for FP8). That alone brings the prefill GEMMs to the Vulkan level.
   - Decode: remove the explicit active-mask tracking of Slang's CUDA output from the hot kernels (decode attention first), see
     [first numbers](#first-numbers).
   - cuBLASLt for the large prefill GEMMs (bf16, FP8 E4M3 with per-tensor and per-block scales, MXFP8 / NVFP4 block-scaled on
     Blackwell), in the optional `Sharplight.Cuda.Blas` package, chosen per shape against the Slang GEMMs with a small autotune cache.
   - Dynamic shared memory above 48 KB for the existing tile kernels (larger GEMM tiles, two K/V tiles in flight in `fe`).
2. **Phase 3: attention.** A flash-attention kernel for prefill with large shared memory, `cp.async` / TMA loads and warp-specialized
   producer / consumer pipelines (Hopper / Blackwell: `wgmma` / `tcgen05`), paged decode attention with split-K, written in CUDA C++ and
   compiled at build time to cubins for sm_80, sm_89, sm_90a, sm_100a, sm_120a plus PTX.
3. **Phase 4: MoE.** Grouped GEMMs over sorted experts with TMA-gathered rows, fused routing / top-k / gather, MXFP4 and NVFP4 expert
   weights on the Blackwell block-scaled tensor-core instructions.
4. **Phase 5: launch overhead.** CUDA graphs for the decode step (one `cuGraphLaunch` per token instead of ~300 launches; the recording
   per symbol binding already gives the structure), programmatic dependent launch, and asynchronous host copies.
5. **Phase 6: FP8 / NVFP4 / MXFP4 tensor cores everywhere** (KV cache in FP8, attention in FP8), and **multi-GPU**: NCCL-free
   peer-to-peer copies (`cuMemcpyPeer`, P2P over NVLink / PCIe) for pipeline and tensor parallelism across the three GPUs.

Throughout, Vulkan keeps its behaviour and speed: shared Slang changes must keep the SPIR-V and MSL output working (Vulkan suite here,
Metal in CI).
