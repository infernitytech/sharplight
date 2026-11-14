# macOS and Apple Silicon

Status of macOS round 1 (branch `macos/round-1`, not merged into `main`). Sharplight runs on Apple Silicon Macs with the
CPU provider (NEON kernels) and with the Vulkan provider through a bundled MoltenVK. A native Metal provider is planned
(see [the plan](#native-metal-provider-plan)); it is not part of this round.

Nothing here changes Linux or Windows: the MoltenVK loading, the portability-subset handling and the Arm64 kernels are
only active on macOS or Arm64 hosts. The x64 kernels, their greedy outputs and the x64 test suites are unchanged.

## Status

| Area | macOS arm64 (CI: `macos-15`, Apple M1 VM) | Linux arm64 (CI: `ubuntu-24.04-arm`, Neoverse N2) |
| --- | --- | --- |
| Build (`osx-arm64` / `linux-arm64`) | ✓ | ✓ |
| CPU provider, `Sharplight.Cpu.Tests` | ✓ | ✓ (also at the `neon` tier) |
| Vulkan provider via MoltenVK, `Sharplight.Vulkan.Tests` | ✓ (all 1277 tests: 1247 pass, 30 skip: benchmarks, an opt-in large test, the matrix-unit kernels) | — (no GPU) |
| LLM smoke test, Qwen2.5-0.5B-Instruct Q4_K_M, CPU | ✓ | ✓ |
| LLM smoke test, Vulkan via MoltenVK | ✓ (same text as the CPU) | — |
| `Sharplight Studio.app` (unsigned, ad-hoc signed zip artifact) | ✓ | — |

The workflow is [`.github/workflows/macos.yml`](../.github/workflows/macos.yml). It runs only for the `macos/round-1`
branch and on manual dispatch.

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

  The rewrite runs at pipeline creation, only on devices that report `VK_KHR_portability_subset`.
  `SHARPLIGHT_VK_SPIRV_PORTABILITY=0` turns it off; `=all` applies it on every device, which is how it is checked on a native
  driver: the full Vulkan suite passes with it on an x64 GPU, and all 600 cached kernels still pass `spirv-val` and translate
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
step), against 158 / 13.7 tok/s on the VM's 3 CPU cores. Creating the 104 pipelines of the decode graph takes about 50 s
the first time (SPIR-V to MSL, then Metal compilation). That is also why CI runs the Vulkan suite in two time-boxed parts
(core kernels about 17 min, the rest about 25 min).

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
- **Precompiled artifacts.** Metal binary archives (`MTLBinaryArchive`) as the binary tier of `.slm` packages, keyed by GPU
  family and OS build like the Vulkan pipeline cache (`ArtifactTargets`), with MSL source as the portable code tier.
- **Argument passing** through `setBytes` (≤ 4 KB, the push-constant equivalent) and GPU addresses (`MTLBuffer.gpuAddress`,
  Metal 3) for the pointer-based kernels; residency sets (`MTLResidencySet`, macOS 15) for large models.
- **Tests:** the same suites as Vulkan (`Sharplight.Vulkan.Tests` structure) on the `macos-15` runners, whose paravirtual
  GPU supports Metal 3 but not every Apple GPU family feature.
