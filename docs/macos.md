# macOS and Apple Silicon

Status of macOS round 1 (merged into `develop`) and round 2 (branch `macos/round-2`: the MoltenVK prefill regression, see
[below](#moltenvk-prefill-regression-round-2)). Sharplight runs on Apple Silicon Macs with the
CPU provider (NEON kernels) and with the Vulkan provider through a bundled MoltenVK. A native Metal provider is planned
(see [the plan](#native-metal-provider-plan)); it is not part of this round.

Nothing here changes Linux or Windows: the MoltenVK loading, the portability-subset handling and the Arm64 kernels are
only active on macOS or Arm64 hosts. The x64 kernels, their greedy outputs and the x64 test suites are unchanged.

## Status

| Area | macOS arm64 (CI: `macos-15`, Apple M1 VM) | Linux arm64 (CI: `ubuntu-24.04-arm`, Neoverse N2) |
| --- | --- | --- |
| Build (`osx-arm64` / `linux-arm64`) | ✓ | ✓ |
| CPU provider, `Sharplight.Cpu.Tests` | ✓ (1287 tests: 1251 pass, 36 skip) | ✓ (same counts; kernel tests also at the `neon` tier) |
| Vulkan provider via MoltenVK, `Sharplight.Vulkan.Tests` | ✓ (all 1454 tests in four parts: 1425 pass, 29 skip: benchmarks, opt-in large tests, the workgroup-matrix GEMM) | — (no GPU) |
| LLM smoke test, Qwen2.5-0.5B-Instruct Q4_K_M, CPU | ✓ | ✓ |
| LLM smoke test, Vulkan via MoltenVK | ✓ (same text as the CPU) | — |
| `Sharplight Studio.app` (unsigned, ad-hoc signed zip artifact) | ✓ | — |
| Studio Velopack package (`studio.yml`, `osx-arm64`) with MoltenVK | ✓ (`pack-studio.sh` requires `libMoltenVK.dylib`) | — |

The workflow is [`.github/workflows/macos.yml`](../.github/workflows/macos.yml). It runs only for `macos/**`
branches and on manual dispatch. Round 1's last full run: 43736304439 (all three jobs green). The MoltenVK LLM benchmark
runs in its own job (`macos-bench`, in parallel with the suites): the defaults and a set of switches at 128 and 512 prompt
tokens, then the prefill's kernel list.

The macOS job runs the Vulkan suite in four time-boxed parts (`-class` filters in the workflow's `env`):

| Part | Classes | Tests (pass / skip) | Time on `macos-15` |
| --- | --- | --- | --- |
| core | elementwise, norms, RoPE, row ops, reductions, gathers, views, data ops, casts, matmuls, fusion, SIMT / skinny / split-half GEMMs, row invariance, stateful | 612 / 0 | 31 min |
| MMQ GEMMs | `MmqTests` | 48 / 0 | 6 min |
| quantized formats and data kernels | quant formats, Bonsai, IQ4, FP8 GEMM, MoE, top-k, pad, resize | 433 / 0 | 10 min |
| remaining classes | everything else (attention, SSM, convolutions, vision and audio kernels, precompilation...) | 332 / 29 | 7 min |

Compiled kernels persist between runs: the workflow points `SHARPLIGHT_COMPILATION_CACHE` (the per-user artifact store,
`FileCompilationCache`) at `.kernel-cache` in the workspace and restores / saves it with `actions/cache`, keyed by the kernel
sources. Stale entries never hit, so an older cache is only a partial speed-up.

## Vulkan through MoltenVK

macOS has no system Vulkan driver. MoltenVK is a Vulkan implementation layered on Metal (it translates SPIR-V to the Metal
Shading Language with SPIRV-Cross). Sharplight ships it in a separate NuGet package, so `Sharplight.Vulkan` stays small:

- **`Sharplight.Vulkan.MoltenVK`** (`src/Sharplight.Vulkan.MoltenVK`): `runtimes/osx-arm64/native/libMoltenVK.dylib` from the
  pinned KhronosGroup/MoltenVK release **v1.4.2** (`MoltenVK-macos.tar`, SHA-256
  `f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e`), plus its Apache 2.0 license and a notice. The binary is
  not committed: the project downloads the release and checks the checksum the first time it builds on macOS (or with
  `-p:SharplightBundleMoltenVK=true`) or packs. To update it, change `MoltenVKVersion` and `MoltenVKSha256` together.
- In this repository, `Sharplight.Vulkan` references the package project only when building on macOS (or with
  `SharplightBundleMoltenVK=true`, for cross-publishing), so every app and test project built on a Mac gets the dylib next to
  its executable. The published `Sharplight.Vulkan` package does not depend on it; Mac applications add
  `Sharplight.Vulkan.MoltenVK` themselves.
- Loading (`VulkanLibrary`): on macOS the provider tries, in order, `SHARPLIGHT_VULKAN_LIBRARY` (e.g. the Vulkan SDK's
  `libvulkan.1.dylib` to use the loader and validation layers), `libMoltenVK.dylib` next to the app, in
  `runtimes/osx-arm64/native/`, in `../Frameworks` (app bundles) and in the host's native search directories, and then
  Vortice.Vulkan's default probing (`libvulkan.dylib`, `libMoltenVK.dylib`, the frameworks). Other platforms always use the
  system loader as before.
- The instance enables `VK_KHR_portability_enumeration`. Devices that report **`VK_KHR_portability_subset`** get the extension
  enabled at device creation, as the specification requires. `VkPhysicalDevicePortabilitySubsetFeaturesKHR` is queried and its
  unsupported members are listed in the device capabilities (`portability_restrictions`). The compute kernels use none of the
  restricted features (events, image views, samplers, tessellation, triangle fans, vertex input...); barriers and timeline
  semaphores are all the provider synchronizes with.
- **SPIR-V rewrite for SPIRV-Cross** (`SpirvPortability`). MoltenVK's SPIRV-Cross mistranslates two constructs the Slang
  kernels use, so the MSL fails to compile (the first CI run failed about 780 Vulkan tests this way) or computes wrong
  results:
  - `OpCopyLogical` of the push-constant parameter struct to its logical twin is emitted as a dereference of every pointer
    member (`*_41.x = entryPointParams.p.x;`, "assigning to 'float' from 'device float *'"). It is rewritten into an
    `OpCompositeExtract` per member and one `OpCompositeConstruct`.
  - An `OpPtrAccessChain` whose result is used as a pointer value (cast, passed, returned, stored, selected) is emitted as the
    element value (`reinterpret_cast<device float4*>(_70.o[i])`). It is rewritten into address arithmetic,
    `ConvertUToPtr(ConvertPtrToU(base) + index * ArrayStride)`.

  - A subgroup operation whose only use is in a later, divergent block is moved into that block
    (`if (lane == 0) y[row] = simd_sum(acc);`), where only the active lanes take part: every subgroup reduction of the
    matvecs and attention came out wrong (the smoke test produced garbage tokens). Each `OpGroupNonUniform*` result gets a
    second, unused `OpCopyObject` use, which makes SPIRV-Cross evaluate it into a temporary where the kernel computes it.

  The rewrite runs in `PipelineBuilder.CreateModule`, so every path that creates a shader module (pipelines on demand,
  stored pipeline binaries, precompilation captures for `VK_KHR_pipeline_binary` and the pipeline cache) gets the rewritten
  kernels; only on devices that report `VK_KHR_portability_subset`.
  `SHARPLIGHT_VK_SPIRV_PORTABILITY=0` turns it off; `=all` applies it on every device, which is how it is checked on a native
  driver: the full Vulkan suite (1454 tests) passes with it on an x64 GPU, and all 600 cached kernels still pass `spirv-val` and translate
  to MSL without any of the patterns. All three are upstream SPIRV-Cross bugs (the access-chain and subgroup ones are still
  in its current main); once a MoltenVK release fixes them, the rewrite can go.
- **Metal fast math off.** MoltenVK compiles its MSL with Metal fast math by default, which may reassociate sums (the
  row-invariant kernels then differ between batch sizes) and assume finite values (masked softmax scores are `-inf`). On
  macOS the instance sets `MVK_CONFIG_FAST_MATH_ENABLED` to *never* through `VK_EXT_layer_settings`;
  `SHARPLIGHT_VK_PORTABILITY_FAST_MATH=1` keeps MoltenVK's default.
- The Vulkan 1.3 features (`subgroupSizeControl`, `computeFullSubgroups`, `shaderIntegerDotProduct`) only count when the
  device reports Vulkan 1.3, since a 1.2 device (an older MoltenVK) could not enable them.

On the CI runner (paravirtual GPU of an M1 VM; numbers are indicative only), Qwen2.5-0.5B-Instruct Q4_K_M with
`--bench 128` runs 544 tok/s prefill and 53.6 tok/s decode through MoltenVK (485 dispatches and 340 barriers per decode
step), against 158 / 13.7 tok/s on the VM's 3 CPU cores. After the merge of `develop` the prefill fell to about 100 tok/s;
round 2 found and fixed the cause ([below](#moltenvk-prefill-regression-round-2)). Creating the 104 pipelines of the decode graph takes about 50 s
the first time (SPIR-V to MSL, then Metal compilation). That is also why CI runs the Vulkan suite in four time-boxed parts (see
[Status](#status)).

### MoltenVK prefill regression (round 2)

**Symptom.** The `--bench 128` prefill of Qwen2.5-0.5B-Instruct Q4_K_M on the CI's Apple GPU fell from 557 tok/s (run
43659665611, `c1b3c706`) to 101 tok/s (run 43680525445, `d3b6fa00`, the first merge of `develop` into round 1) and stayed
there (87.8 in run 43736304439, a noisier runner); decode did not move. The suspects named at the time (the causal `fe`
attention kernel, SM-sized GEMM K splits, the two-row split-K matvec, pipeline binaries) all came later than `d3b6fa00`, and
switching each off changes nothing (run 43762256231, about 99-101 tok/s each).

**Cause.** ONNX performance round 6 (in that merge) gave `DeviceInfo.ConcurrentWorkgroups` a value of 64 for integrated GPUs
and CPU implementations. MoltenVK reports the Apple GPU as `IntegratedGpu`, so the SIMT GEMM (the prefill path of every
device without matrix units; the int8 MMQ GEMM needs `matrix_int8`) started to choose the 128 x 128 `gemm_wide` tile from
128 rows instead of 512, which helped the AMD iGPU it was tuned on. `gemm_wide` stages 2 x 16 x 128 f32 values of
activations and weights, double-buffered: 32 KB of workgroup memory, which is the whole `maxComputeSharedMemorySize` of the
Apple GPU, plus 64 accumulators per thread. Each core then runs one workgroup with nothing to hide its loads behind. Run
43762256231 isolates it:

| `--bench 128`, MoltenVK, CI Apple GPU | prefill tok/s | prefill kernels of the large matmuls |
| --- | --- | --- |
| defaults of `develop` | 99.8 | `gemm_wide` (split-K) |
| `SHARPLIGHT_CONCURRENT_WORKGROUPS=0` (sized as a large GPU) | 552.3 | `gemm`, deep K splits |
| `SHARPLIGHT_CONCURRENT_WORKGROUPS=256` | 101.3 | `gemm_wide` |
| `SHARPLIGHT_GEMM_WIDE=0` (64 x 64 tile, small-device K splits) | 629.0 | `gemm`, small-device K splits |
| `SHARPLIGHT_ATTN_FE_CAUSAL=0` / `SHARPLIGHT_GEMM_WAVE=fixed` / `SHARPLIGHT_MV_ROWS=1` / `SHARPLIGHT_DISABLE_FEATURES=kernel_binaries` | 99.2 / 99.9 / 100.6 / 98.5 | `gemm_wide` |

**Fix.** The SIMT lowering uses `gemm_wide` only on devices whose workgroup memory limit is larger than the tile's 32 KB
(`GraphLowerer.SimtWideFits`), a device property every provider reports (`LoweringOptions.MaxSharedMemory`). NVIDIA (48 KB) and
AMD (64 KB) keep their selection and outputs unchanged; on 32 KB devices (Apple GPUs through MoltenVK) large SIMT
matmuls take the 64 x 64 `gemm` tile, which keeps the small-device K splits. `SHARPLIGHT_GEMM_WIDE=force` restores the old
choice for comparison. `ConcurrentWorkgroups` itself stays 64 for the Apple GPU: with the 64 x 64 tile the small-device K
splits are faster than the large-GPU ones (629 against 552 tok/s).

RESULTS_PLACEHOLDER

**Diagnostics added.** `SHARPLIGHT_CONCURRENT_WORKGROUPS=n` (or the device setting `concurrent_workgroups`) overrides the
sizing hint per device, like `SHARPLIGHT_SUBMISSION_BUDGET_MS`; `sharplight --devices` prints `concurrent_workgroups` and
`submission_budget_ms`. `SHARPLIGHT_VK_PROFILE=1` lists the kernels on MoltenVK, but its per-dispatch times read 0 there:
Metal samples GPU timestamps only at encoder boundaries, so per-kernel timing needs Xcode's Metal tools.

### Device features MoltenVK exposes

From the CI runner (`sharplight --devices`, MoltenVK 1.4.2 on the `macos-15` paravirtual Apple GPU, Metal 3):

```
vulkan:0 Apple Paravirtual device (IntegratedGpu, 7168 MiB)
  features: fp16 int_dot subgroup_ops subgroup_size_control
  api_version=1.3.357, subgroup_size=32, subgroup_size_range=4-32, fp16=True, bf16=False, fp8=False,
  cooperative_matrix=False, integer_dot_product=True, max_shared_memory=32768, max_push_constants=4096,
  memory_budget=True, portability_subset=True,
  portability_restrictions=pointPolygons samplerMipLodBias tessellationIsolines tessellationPointMode
```

`int_dot` is reported because MoltenVK advertises `shaderIntegerDotProduct`; Metal has no packed int8 dot instruction, so
these are emulated. No `matrix_*` feature (no `VK_KHR_cooperative_matrix`), no `bf16`, no `fp8`.

Every feature without support takes its existing fallback, the same paths that run on the AMD iGPU and llvmpipe locally:
no cooperative matrices (no `matrix_*` features, so the SIMT / subgroup GEMMs), and the kernels without int8 dot products
when `int_dot` is absent. `SHARPLIGHT_DISABLE_FEATURES` and `DeviceOptions.DisabledFeatures` work as everywhere else.

### Gaps versus native Metal

MoltenVK runs the same SPIR-V kernels as every other Vulkan device, so it is correct but leaves Apple-specific performance
on the table:

- **Matrix units.** Apple GPUs have `simdgroup_matrix` (8x8 f16/f32/bf16 tiles, M1 and later) and the M5 adds Metal 4 tensor
  operations, but MoltenVK does not expose `VK_KHR_cooperative_matrix`, so every matmul runs on the SIMT kernels.
- **Integer dot products.** Metal has no packed int8 dot instruction; MoltenVK emulates `shaderIntegerDotProduct` (when
  reported) with scalar code, which is no faster than the fallback.
- **Memory.** Unified memory is exposed as a device-local, host-visible heap, but weights are still copied into Vulkan
  buffers. Metal can wrap page-aligned mmapped weights without a copy (`newBufferWithBytesNoCopy`).
- **Pipeline creation.** SPIR-V → MSL translation and Metal compilation happen at pipeline creation, which makes the first
  load slower than on other drivers; MoltenVK's pipeline cache keeps MSL, not GPU binaries.
- **Submission.** Each Vulkan submit becomes Metal command buffers with MoltenVK's own encoding overhead; argument buffers
  and buffer device addresses are emulated on top of Metal 3 GPU addresses.
- **Diagnostics.** `VK_EXT_memory_budget` is reported, but no pipeline statistics or GPU counters (`VK_KHR_pipeline_executable_properties`), so kernel tuning needs Xcode's Metal tools.

## ARM64 CPU kernels

The CPU provider picks an instruction-set tier at startup (`CpuIsa`); `SHARPLIGHT_CPU_ISA` caps it for testing. Two Arm64
tiers sit between the portable 128-bit tier and the x86 tiers (a host supports only one family):

| Tier | Requires | Used by |
| --- | --- | --- |
| `neon` | `AdvSimd.Arm64` | f32 dot / dot4 / dot4x2, sums, max, exp, scale, axpy; f16 and bf16 → f32 conversion, f16 dot and axpy; 8-bit dequantization (`StoreU8x16` / `StoreI8x16`); the 6 x 16 GEMM micro-kernel with by-lane FMLA (also the gathered form used by the shifted convolution); the activation quantizer of the integer-dot matvec; tiled flash attention (8 lanes as two 128-bit vectors) and the decode attention's softmax |
| `neondot` | plus `Dp` (FEAT_DotProd, `sdot`) | the integer-dot matvec for every `IntDot` format (Q4_0 ... Q6_K, IQ4, MXFP4): codes enter as `u - 128`, products are summed exactly in int32 lanes and the shift returns through the offsets; reports the `int_dot` device feature |

The tiers keep the structure of the x86 ones (same tile shapes, same task splits), and the integer part of the dot is exact
on every tier, so the reference tests (`IntDotTests` runs every supported tier against the scalar one) cover them. On x64
`AdvSimd.Arm64.IsSupported` is a JIT-time constant `false` and the paths disappear.

`i8mm` (`smmla`, 2x8 by 8x2 int8 matrix multiply) is available on the Neoverse N2 runner and on M2 and later, but .NET 11
exposes no intrinsic for it (only `Dp`, `Rdm` and SVE/SVE2 without the matrix instructions), so it is not used yet.

Measured on CI (Qwen2.5-0.5B-Instruct Q4_K_M, `--cpu --bench 128`, prefill of 128 tokens and decode, best run):

| Host | `scalar` | `v128` | `neon` | `neondot` |
| --- | --- | --- | --- | --- |
| Linux arm64, Neoverse N2, 4 vCPU | 12.5 tok/s prefill, 1.41 tok/s decode | 12.5 / 1.43 | 171.8 / 14.4 | 171.2 / 18.2 |
| macOS arm64, Apple M1 (virtual), 3 vCPU | 8.4 / 4.98 | 11.8 / 5.04 | 219.4 / 15.4 | 216.2 / 17.5 |

Before this round an Arm64 host ran the `v128` tier, which is effectively scalar for these kernels: on the Neoverse N2 the
NEON tiers are 13.7x faster in prefill and 13x faster in decode, and `sdot` adds another 26 % in decode; on the M1 VM
prefill is 18.6x faster and decode 3.5x (the VM's decode is memory bound and noisy).

Not covered yet: the elementwise kernels of `ScalarMath` / `BinaryOp` / `ReduceOp` and the depthwise convolution still
only have AVX tiers (they run their scalar or portable code on Arm64), and `Simd.Prefetch` is a no-op (.NET has no Arm
prefetch intrinsic).

## Sharplight Studio.app

`packaging/macos/build-app.sh <version> [output directory]` (run on macOS) builds `Sharplight Studio.app`:

- a self-contained, ReadyToRun `osx-arm64` publish in `Contents/MacOS`;
- `libMoltenVK.dylib` in `Contents/Frameworks` (the provider finds it as `../Frameworks`), with the MIT, MoltenVK and
  third-party license files in `Contents/Resources`;
- `Info.plist` from `packaging/macos/Info.plist` (bundle id `io.github.infernitytech.sharplight.studio`, macOS 14+);
- `sharplight-studio.icns`, made with `sips` and `iconutil` from `packaging/linux/sharplight-studio.png` (256 px, so the
  icon set stops at 256 px; a 1024 px source is needed for a full set);
- an **ad-hoc** signature (`codesign --sign -`), since Apple Silicon only runs signed code, and a zip
  (`Sharplight-Studio-<version>-macos-arm64.zip`, made with `ditto` to keep the bundle intact).

The CI job `studio-app` uploads the zip as an artifact.

Releases use Velopack instead (`.github/workflows/studio.yml`, `osx-arm64` on `macos-15`, through
`packaging/pack-studio.sh`): the publish on macOS brings `libMoltenVK.dylib` along next to the executable, where the
Vulkan provider finds it, and the script copies the MoltenVK license and notice into the package. It fails if the dylib is
missing, so a macOS Studio package always has the Vulkan provider.

### Opening the unsigned app

The app has no Developer ID signature and is not notarized, so Gatekeeper blocks the first launch of a downloaded copy
("cannot be opened because Apple cannot check it for malicious software"). To approve it once:

1. Unzip and move `Sharplight Studio.app` to `/Applications`.
2. Open it; macOS refuses. Open **System Settings → Privacy & Security**, scroll to *Security* and click **Open Anyway** for
   Sharplight Studio, then confirm with **Open**. (On macOS 14, Control-click the app in Finder and choose **Open** instead.)
3. Alternatively, remove the quarantine attribute in Terminal: `xattr -dr com.apple.quarantine "/Applications/Sharplight Studio.app"`.

### Signing and notarization (to do)

Distributing without these steps needs:

- An **Apple Developer Program** membership (99 USD per year) and a **Developer ID Application** certificate, exported as a
  `.p12`, stored as repository secrets (`MACOS_CERTIFICATE_P12` base64, `MACOS_CERTIFICATE_PASSWORD`) and imported into a
  temporary keychain in CI.
- Signing every Mach-O inside out (`libMoltenVK.dylib`, the .NET native libraries, `createdump`, then the app) with
  `codesign --options runtime --timestamp --sign "Developer ID Application: ..."` and an entitlements file for the hardened
  runtime. .NET needs `com.apple.security.cs.allow-jit` (the JIT) and `com.apple.security.cs.allow-unsigned-executable-memory`
  is not required with ReadyToRun; MoltenVK needs none.
- **Notarization** with `xcrun notarytool submit --wait` using an App Store Connect API key (secrets `APPLE_API_KEY_ID`,
  `APPLE_API_ISSUER_ID`, `APPLE_API_KEY_P8`) or an app-specific password with the team id, then
  `xcrun stapler staple "Sharplight Studio.app"` so the ticket works offline.
- Optionally a `.dmg` (`hdiutil create`), signed and notarized the same way.

## Native Metal provider plan

The next step for Macs is a `Sharplight.Metal` execution provider next to `Sharplight.Vulkan`, with the same graph lowering
and kernel library. Planned design (not started):

- **Kernels from Slang.** `Sharplight.Shaders` already has `ShaderTarget.Metal`; the Slang kernels compile to MSL with the
  same lowering (`KernelProgram`), so the provider mostly needs launch, memory and synchronization code. Feature-specific
  variants get Metal mechanisms behind the provider-neutral names (`matrix_f16`, `matrix_bf16` via `simdgroup_matrix`).
- **Thin Objective-C interop on plain `net11.0`.** No Xamarin / `net11.0-macos` workload: P/Invoke into
  `libobjc.dylib` (`objc_msgSend`, `sel_registerName`, `objc_getClass`) and `Metal.framework` (`MTLCreateSystemDefaultDevice`)
  with a small set of hand-written wrappers for `MTLDevice`, `MTLCommandQueue`, `MTLCommandBuffer`,
  `MTLComputeCommandEncoder`, `MTLBuffer`, `MTLLibrary`, `MTLComputePipelineState`, `MTLSharedEvent`. AOT-compatible, no
  runtime registrar.
- **No-copy weights.** On unified memory, page-aligned mmapped weight files become `MTLBuffer`s with
  `newBufferWithBytesNoCopy:length:options:deallocator:` (storage mode shared), so loading a model maps it instead of copying;
  the GGUF / `.slm` loaders already produce page-aligned tensor data.
- **Matrix units.** `simdgroup_matrix<half|float|bfloat, 8, 8>` (`simdgroup_multiply_accumulate`) for the GEMM and attention
  kernels on M1 and later; **Metal 4 tensor operations** (`MTLTensor`, `mpp::tensor_ops::matmul2d`) on M5 where the neural
  accelerators in the GPU cores are exposed, detected per device and with the simdgroup path as fallback.
- **Two-stage code generation** (the provider contract of [model-format.md](model-format.md#two-stage-precompilation); no
  change to the package format, only new `ArtifactTargets`):
  - *Portable stage* (`PortableCodeFormat` = `air`, `CompilePortableAsync`): Slang's Metal output (MSL) for the request's
    `DeviceProfile`, compiled offline to AIR and linked into a **metallib** (`xcrun metal -c` / `metallib`, or
    `MTLDevice.newLibraryWithSource` + `MTLLibrary` serialization where no toolchain is installed). A metallib is
    GPU-independent within a Metal language version and target OS, so it plays SPIR-V's role: keyed by content hash, stored in
    the `code` tier and shared by every Apple GPU of a profile (Metal version, SIMD width 32, threadgroup memory, features). It
    needs no device, so `sharplight-convert precompile --profile` can produce it on any Mac.
  - *Device stage* (`CompileDeviceSpecificAsync`): an **`MTLBinaryArchive`** with one compute pipeline descriptor per kernel
    (`addComputePipelineFunctionsWithDescriptor`, `serializeToURL`), i.e. the GPU machine code of exactly this GPU family and OS
    build, stored in the `binary` tier and keyed like the Vulkan pipeline binaries (device identity = GPU family / registry id
    class, driver = OS build). Loads pass the archive in `MTLComputePipelineDescriptor.binaryArchives`, so pipeline creation
    skips the backend compiler; a stale or foreign archive is simply a miss (Metal falls back to compiling the metallib). The
    feature name `kernel_binaries` disables the stage, as on Vulkan.
  - Precedence and the per-user `ArtifactStore` stay as they are (package sets, then the store, then compilation).
- **Registration** (DI, like every provider, see [api-overview.md](api-overview.md#registration)): a `Sharplight.Metal`
  assembly with `services.AddMetalProvider(o => ...)` (`MetalProviderOptions`: validation via `MTL_DEBUG_LAYER`, shader
  validation, capture directory), registering `MetalProvider` as an `IExecutionProvider` in the `ExecutionProviderCatalog`.
  `AddSharplight()` adds it on macOS ahead of `AddVulkanProvider()` so discovery prefers it; MoltenVK stays available
  (hosts that do not register the Metal provider, or pick the Vulkan device explicitly) as the fallback and the reference for
  kernel parity.
- **Device properties**: `ConcurrentWorkgroups` and compute units from the GPU core count (IOKit `gpu-core-count`) and
  `maxThreadsPerThreadgroup`, not from the device type (see [the prefill regression](#moltenvk-prefill-regression-round-2)).
- **Argument passing** through `setBytes` (≤ 4 KB, the push-constant equivalent) and GPU addresses (`MTLBuffer.gpuAddress`,
  Metal 3) for the pointer-based kernels; residency sets (`MTLResidencySet`, macOS 15) for large models.
- **Tests:** the same suites as Vulkan (`Sharplight.Vulkan.Tests` structure) on the `macos-15` runners, whose paravirtual
  GPU supports Metal 3 but not every Apple GPU family feature.
