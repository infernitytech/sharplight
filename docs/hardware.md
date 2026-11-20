# Hardware requirements and fallbacks

Sharplight runs on whatever the machine has: discrete GPUs with and without resizable BAR, integrated GPUs whose memory is system
memory, CPU implementations of Vulkan (llvmpipe), GPUs shared with other applications. No provider assumes a memory topology, a
heap size or a limit; each one detects what the device offers, uses the best path, and falls back where a feature or a heap is
missing. Only when nothing works does an operation fail, with an error that names what was tried.

Device features have provider-neutral names (`DeviceFeatures` in `Sharplight.Core`), and every feature can be switched off per
device (`DeviceOptions.DisabledFeatures`), per compilation, or process-wide (`SHARPLIGHT_DISABLE_FEATURES=a,b` or `all`), which
exercises the fallbacks on a capable GPU.

## What each provider needs

| Provider | Hard requirements (a device without them is not listed, with the reason) |
|---|---|
| Vulkan | Vulkan 1.2; `bufferDeviceAddress`; `storageBuffer8BitAccess`, `storageBuffer16BitAccess`; `shaderInt8`, `shaderInt16`, `shaderInt64`; `scalarBlockLayout`; subgroup arithmetic and shuffle with subgroups of at least 8 lanes; workgroups of 256 invocations; a queue family with compute |
| CUDA | NVIDIA driver of the CUDA 13 branch (R580 or later); nothing else installed ([cuda.md](cuda.md)) |
| Metal | macOS with an Apple GPU ([macos.md](macos.md)) |
| CPU | any x64 or arm64 CPU; wider SIMD (AVX2 / AVX-512, NEON) is used when present |

A device the provider rejects is reported by `CheckAvailability()` / `GetDevices()` with what it lacks; placement then uses the
other devices, and the CPU always remains.

## Optional features and their fallbacks

| Feature | Provider mechanism | Without it |
|---|---|---|
| `matrix_f16`, `matrix_bf16`, `matrix_int8`, `matrix_fp8`, `matrix_workgroup` | Vulkan cooperative matrices (KHR, NV2), CUDA `mma.sync`, Metal simdgroup matrices (`matrix_f16_8x8`) | SIMT GEMMs and attention |
| `fp16`, `bf16`, `fp8` | native 16/8-bit float arithmetic | f32 arithmetic on unpacked values |
| `int_dot` | packed int8 dot products | scalar int8 products |
| `subgroup_size_control` | pinned subgroup size | kernels sized for the device's default subgroup |
| `memory_budget` | `VK_EXT_memory_budget`; CUDA `cuMemGetInfo` | memory figures from Sharplight's own tally (other processes are not seen) |
| `kernel_binaries` | `VK_KHR_pipeline_binary`, cubins, Metal binary archives | the driver's whole pipeline cache blob |
| `paged_kv` | paged KV kernel forms | the contiguous per-slot KV cache |
| `window_kv` | window-sized caches for sliding-window layers with the paged cache (host side only) | those caches hold a row per position of the pool |
| Vulkan sparse binding (not a named feature) | `sparseBinding` on the compute queue | a buffer larger than one allocation fails with a clear error |
| Vulkan timestamps (not a named feature) | `timestampValidBits` of the compute queue | no GPU step time (`LastGpuMs` is 0); the submission-budget calibration times launches on the host |
| Vulkan transfer queue (not a named feature) | a transfer-only queue family | downloads on the compute queue |

## Memory: what falls back to what

**Vulkan** (`MemoryAllocator`, `WeightUploader`):

- Memory types are tried in order of preference, and the next type serves when a heap is full. Weights, state and activations live
  in device-local memory. Upload staging prefers device-local host-visible memory (the PCIe BAR window, all of VRAM with resizable
  BAR, 256 MB without it, shared by every process on the GPU) and falls back to host memory; readback staging prefers host-cached
  memory, then any host-coherent memory. Where `VK_EXT_memory_budget` shows that a staging heap's budget is used up, that heap is
  skipped rather than taken from other processes.
- Integrated GPUs and CPU implementations have only device-local host-visible memory (unified memory): weights are written straight
  into their buffers, without staging buffers or copies on the device.
- Sub-allocation blocks are 256 MB for device memory and 16 MB for staging, never larger than the driver's largest allocation
  (`maxMemoryAllocationSize`; 2 GB on llvmpipe and some integrated GPUs). A block that cannot be allocated whole is replaced by an
  allocation just large enough.
- A buffer larger than the largest allocation (a big embedding table, a long KV cache) is one buffer bound to several allocations
  (sparse binding): kernels still see one contiguous address range. A buffer beyond `maxBufferSize`, or beyond the largest allocation
  on a device without sparse binding, fails clearly.
- The weight upload's two staging buffers are sized to the need: at most 128 MB, no more than the model's weights, at most half the
  free budget of the staging heap, halved on every failed allocation down to 1 MB, and a single buffer when a second does not fit.

**CUDA**: device buffers come from `cuMemAlloc`. Weight staging is two page-locked buffers of up to 64 MB; when the system refuses
page-locked memory they halve down to 1 MB, then one pageable buffer with synchronous copies serves. The cuBLASLt workspace is
optional: without room for it the products stay on the generated GEMMs.

**Metal**: buffers are shared (unified) memory; a buffer beyond `maxBufferLength` fails clearly.

**Host memory** is a limited resource too. The CPU device's figures (`Info.TotalMemory`, `AvailableMemory`) are bounded by the
process's cgroup memory limit (v2, or v1) minus its usage plus reclaimable file cache: in a container `/proc/meminfo` and `free`
show the whole machine while the OOM killer enforces the limit, so placement must not offload layers to memory the container does not
have. The GPU providers' fallbacks into host memory are bounded: staging buffers (at most two of 128 MB on Vulkan, 64 MB on CUDA) and
per-input/output staging; weights and state never move to host memory behind the caller's back (offloading is placement's decision,
on the CPU device).

**When nothing works**, the allocation throws `DeviceOutOfMemoryException` (an `OutOfMemoryException`): which allocation (e.g. "weight
buffer blk.0.attn_q.weight", "weight upload staging"), its size, every attempt (memory type and result), every heap's size, budget,
process usage and Sharplight's share, the largest allocation and the active limits, followed by what to do. Placement refuses a model
that cannot fit before allocating anything (`InsufficientMemoryException`), and a failed compilation leaves nothing allocated.

## Will it fit

Every load decision uses the live figures of the device, not its total memory. `IComputeDevice.GetMemoryInfo()` returns
`DeviceMemoryInfo`: `Available` is what this process can still allocate (with `VK_EXT_memory_budget` or CUDA, other processes' use
is already taken off), `Capacity` is what this process could hold (its own use plus `Available`). Placement
(`PlacementPlanner`), the automatic rows per pass and KV sizing, and the hosting layer use `Available`; the diffusion and video
generators decide whether to keep stages resident from `Capacity`; Studio's load dialog uses `Available`, and its "fits this machine"
filters and badges (verified catalog, Hub search) use `DeviceMemory.UsableBytes` (the capacity). `DeviceOptions.MemoryBudget` caps
what placement plans for on a device.

## Simulating smaller or different hardware

`DeviceOptions.MemoryLimits` (`DeviceMemoryLimits`) caps a device below the hardware, the same way on every provider; the
environment sets process-wide caps (sizes in bytes or with K/M/G/T):

| Cap | Environment | Effect |
|---|---|---|
| `DeviceMemory` | `SHARPLIGHT_LIMIT_DEVICE_MEMORY` | device-local memory beyond it fails like a full heap; the device reports it as its total, so placement offloads or refuses |
| `HostVisibleDeviceMemory` | `SHARPLIGHT_LIMIT_HOST_VISIBLE_DEVICE_MEMORY` | caps the BAR window (Vulkan, discrete GPUs); `0` hides it, as on GPUs without one |
| `MaxAllocation` | `SHARPLIGHT_LIMIT_MAX_ALLOCATION` | the largest single allocation (Vulkan: smaller blocks, sparse buffers; all providers: smaller staging) |
| `PinnedHostMemory` | `SHARPLIGHT_LIMIT_PINNED_HOST_MEMORY` | page-locked staging (CUDA); `0` forces pageable copies |

Unified memory and the limits of a CPU implementation are best tested on llvmpipe itself: extract Mesa's `mesa-vulkan-drivers`
package into a folder (no installation needed), set `"library_path"` in its `lvp_icd.json` to the extracted `libvulkan_lvp.so`, and
run with `VK_DRIVER_FILES=<folder>/lvp_icd.json` (only llvmpipe is then visible; it has one 2 GB-allocation, unified heap, sparse
binding and 8-lane subgroups). Missing features are simulated with `SHARPLIGHT_DISABLE_FEATURES`.

Tests: `MemoryLimitsTests` (Vulkan kernel suite: no or a tiny BAR window, a 1 MB largest allocation with sparse buffers, a 2 MB
device), `CudaMemoryLimitsTests` (pageable and small pinned staging, a small device), and `HardwareLimitsRealTests` (Serving tests:
SmolLM2-135M generates the same greedy tokens without a BAR window, with a 6 MB one, with an 8 MB largest allocation and without
pinned memory; a 48 MB device refuses it clearly and a 96 MB one offloads layers to the CPU), on the GPU and on llvmpipe.

## Known gaps

- Vulkan requires 8- and 16-bit storage, `shaderInt16` and `shaderInt64`; GPUs without them (some mobile GPUs) are not listed.
- Upload and readback staging of a single input or output larger than the largest allocation is not split.
- On Vulkan devices without `VK_EXT_memory_budget`, memory used by other processes is invisible; the default budget is 92 % of the
  device-local heap.
- CUDA honours `MaxAllocation` only for staging (CUDA has no per-allocation limit of its own).
- Vulkan's host-heap budget (`VK_EXT_memory_budget`) comes from the driver, which may not see a cgroup limit; host staging is small
  enough that this has not mattered.
- Studio's first-run starter recommendation still sizes from the GPU's total memory (with its configured fraction).
