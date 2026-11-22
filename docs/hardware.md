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
| `window_kv` | window-sized caches for sliding-window layers with the paged cache, and a ring of window rows in single-sequence sessions (host side only) | those caches hold a row per position of the pool or context |
| Vulkan sparse binding (not a named feature) | `sparseBinding` on the compute queue | a buffer larger than one allocation fails with a clear error |
| Vulkan timestamps (not a named feature) | `timestampValidBits` of the compute queue | no GPU step time (`LastGpuMs` is 0); the submission-budget calibration times launches on the host |
| Vulkan transfer queue (not a named feature) | a transfer-only queue family | downloads on the compute queue |

The fused decode kernels of the generated path take subgroup operations, and for the int-dot matvecs `int_dot`: the router top-k that keeps
its row in registers (rows of a multiple of the subgroup size, up to 32 values per lane), row norms that write the q8_1 rows of the int-dot
matvecs reading them (subgroups of at least 8 lanes), elementwise groups in the int-dot matvec's store, the delta rule reading repeated key
heads in place. Without them, or past those sizes, the separate launches run (per-lane top-k lists, a quantization launch per activation
row set, the group's own launch, a materialized repeat).

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

**Mixture-of-experts models with CPU offload** (`PlacementOptions.Fallback` on the CPU, `--cpu-offload`): the routed experts of a layer
form a placement group of their own (`<layer>#experts`), and when the model does not fit, the experts of the first layers go to the CPU,
as few as needed, while attention, shared experts, dense layers and the output head stay on the GPU (llama.cpp's `--n-cpu-moe`; a token
reads its k experts but every dense row, so the CPU then reads the fewest bytes per token). The CPU reads those experts in place from the
memory-mapped file. Every moved group keeps room on the GPU for the values it exchanges with its layer and for its extra segment
(`PlacementOptions.ReservePerSegment`), and the segments of one model on a device share their activation buffers. From
`StreamExpertsMinRows` rows per pass (default 32: prefill, large serving steps) the CPU experts run on the GPU instead, their weights
copied in for each pass into one layer's worth of memory the placement keeps free; on CUDA the mapped pages of those weights are
page-locked read-only (`cuMemHostRegister`) within `DeviceMemoryLimits.PinnedHostMemory`, or else half of the process memory (locked
file pages cannot be reclaimed; they are the experts the CPU reads at every decode step, which must stay in memory anyway), and copied
asynchronously while the GPU runs the layer before. Vulkan imports the same pages where the driver allows it (VK_EXT_external_memory_host,
feature `host_memory_import`, same budget; drivers that pin through the locked-memory limit refuse imports beyond `RLIMIT_MEMLOCK`, e.g. in
containers without `CAP_IPC_LOCK`) and copies from them on the device queue; the rest goes through one pair of staging buffers per device.
Streamed compilations keep the stored weight layout (no repack on the host, so the copies need no CPU). The streaming memory keeps the last streamed layer between prefills, so that layer
runs on the GPU in decode as well. Small passes (decode, up to `SHARPLIGHT_SPARSE_EXPERT_ROWS` rows, default 4) run each CPU layer's experts
split on CUDA, and on Vulkan where it imports the mapped pages: the GPU copies the few experts its share of the slots selects (from the
page-locked mapped file, into banks of 32 expert slots shared by the layers, stored layout; gathered per-expert tables such as gpt-oss's
biases are kept whole) while the CPU computes the other slots (slots of the other side get zero weights where the layer adds per-expert
biases); the share per layer and row count is the one with the fastest measured passes, from 0 (the CPU alone, when the bus is slower
than the CPU for these experts) to all slots, the CPU alone probed every 24 passes. Layers whose banks are not page-locked or imported are not split. On CUDA a split pass's copies from page-locked memory go in one `cuMemcpyBatchAsync` (drivers from 13.0), and every segment's small inputs and outputs (up to 1 MiB) go through page-locked staging buffers: the uploads are queued on the segment's stream, the downloads, the runtime scalars and the forwarded inputs are part of the step's CUDA graph, so handing a layer to the CPU and back costs one graph launch and one wait. The CPU computes a layer's experts in two passes: gate and up with the gated activation, then the down product with the weighted slot sum. A compilation that runs
out of device memory is placed again with the measured shortfall kept free (more experts on the CPU), up to three times; a streamed
compilation's failure takes the same path while retries remain, then leaves its experts on the CPU for every batch size. With a fallback
device the default reserve is 512 MiB (1 GiB without). Vulkan takes the driver's memory budget (VK_EXT_memory_budget) as the limit where it
exists, else 92 % of the device-local heap. Switches: `SHARPLIGHT_SPLIT_EXPERTS=0` (whole layers to the CPU, as before),
`SHARPLIGHT_STREAM_EXPERTS_MIN_ROWS=0` (no streaming), `SHARPLIGHT_STREAM_RESIDENT=0` (no resident layer), `SHARPLIGHT_SPARSE_EXPERT_ROWS=0`
(no split), `SHARPLIGHT_STREAM_REGISTER=0` (no page locking), `SHARPLIGHT_SHARED_ARENA=0` (one activation buffer per segment),
`SHARPLIGHT_DEVICE_RESERVE_MIB` / `SHARPLIGHT_SEGMENT_RESERVE_MIB` (reserves).

**Routed experts on disk** (opt-in: `TaskLoadOptions.ExpertsOnDisk`, CLI `--cpu-offload --experts-on-disk [--expert-cache 24G]`, hosting
`ExpertsOnDisk` / `ExpertCacheGiB` per model or `HostedModelDefaults.ExpertsOnDisk`, Studio's load toggle "Read experts from disk"): for
mixture-of-experts models larger than RAM and VRAM together. When the routed experts the placement puts on the CPU fallback do not fit the
host memory beside the CPU's other weights and state (`DeviceSlot.MemoryLimit` of the fallback, else the free memory of the process,
cgroup-aware, minus 4 GiB, `SHARPLIGHT_EXPERT_HOST_RESERVE_MIB`), or the cache asked for is smaller than they are, they stay in the model
file: the placement neither counts them as resident nor page-locks or imports them, and makes no streamed or split GPU compilations of them
(placed again without the GPU memory streaming keeps free, so more expert layers stay on the GPU), except that small passes still split
with a GPU that copies from the expert cache (`IComputeDevice.CopiesFromExpertCache`, CUDA): the store page-locks its slot pools for that
device (`cuMemHostRegister`, within `DeviceMemoryLimits.PinnedHostMemory`, `ExpertStore.PinFor`), the sparse compilation holds the experts
its share of the slots selects in the cache and copies each one asynchronously straight from its slot, the same memory the disk read filled
with `O_DIRECT`: no CPU copy between the disk and the GPU. Vulkan and Metal run the disk-backed layers on the CPU alone for now. The CPU's expert products read the experts each
pass selects into an `ExpertStore` (one per loaded model, shared by its graphs): fixed-size slots of one expert of one weight (gate, up or
down of one layer), one pool per slot size with the budget split by bytes, least recently used evicted first among the slots that no pass
holds and no read fills; `ExpertStore.Statistics` has hits, misses, prefetches, evictions, bytes read and the time passes waited
(`LlmModel.DiskExperts`, `TaskModel.DiskExperts`; the CLI prints them after a run and per phase of `--bench`). A layer's first product (gate /
up) queues the same experts of the layer's later product (down) behind its own reads, so those overlap its compute; a pass that selects more
experts than a quarter of a pool (prefill) runs in chunks of experts in id order, the next chunk read while the current one computes. The
bytes are used as stored and an expert's items always run together, so the outputs equal those of resident experts bit for bit
(`DiskExpertTests`, `StreamedExpertsTests.DiskExpertsMatchTheResidentExperts`). Expert weights must be one contiguous range of a file
(`IWeightSource.TryGetFileRange`: GGUF, `.slm` and safetensors weights), or stored in another block encoding of the same values
(`IWeightSource.TryGetEncodedFileRange` with an `IStoredBlockEncoding`: the narrow IQ1 types 64-66 of Kimi-K3's UD-TQ*/UD-Q1_0 GGUFs, read as
IQ1_S): the store then reads the stored blocks and decodes them in the slot, behind the read, in parallel once the read completed and before the slot counts as filled.
Others stay resident. At the first use the store reads a few experts
to measure the disk and logs the expected decode speed.

Reads go through `IExpertReader`: a batch of (file, offset, length, destination) reads is submitted at once and awaited, the files opened with
`O_DIRECT` on Linux (aligned reads that bypass the page cache, so expert reads do not evict other memory; file systems that refuse it are read
buffered) and read in pieces of up to 8 MiB. On Linux the reader is io_uring (raw `io_uring_setup` / `io_uring_enter`, no native dependency,
32 reads in flight) where the kernel allows it; container seccomp profiles (Docker's default since 25.0) and `kernel.io_uring_disabled` refuse
it with EPERM, old kernels with ENOSYS, and the store then reads with positional reads on 8 threads (logged; `SHARPLIGHT_EXPERT_READER=threads`
forces them, `SHARPLIGHT_EXPERT_READ_DEPTH` sets the depth). Windows and macOS use the threads. A destination may also name a device buffer
(`ExpertReadDestination.DeviceBuffer`): the extension point for a reader that reads straight into GPU memory (DirectStorage on Windows), which
reports `SupportsDeviceDestinations` and plugs in without changes to the cache or the placement; the same holds for NVIDIA GPUDirect Storage
(cuFile, which needs the `nvidia-fs` kernel module; not on the development pod), both to be enabled as a detected device feature. The hop
through host memory costs little today: an NVMe drive reads 3.5-14 GB/s (7.4 GB/s on the PCIe 4.0 x4 link of the development pod), while a
GPU reads page-locked host memory at 25 GB/s (PCIe 4.0 x16) to 50+ GB/s (PCIe 5.0 x16), so copying an expert from its slot takes a fraction
of the time its read took (an 8 MiB DeepSeek-V3 expert: ~1.1 ms to read at 7.4 GB/s, ~0.2-0.3 ms to copy), and it overlaps the reads of
the next experts. A direct device read would save that copy and the host memory of the slots, not disk time. The io_uring reader uses
plain (not registered) buffers: the reads are `O_DIRECT` DMA into the slots either way, and pieces of 8 MiB make the per-read page pinning
that fixed buffers avoid negligible.

Hosts see it in `HostedModelStatus` (`ExpertsFromDisk`, `DiskExpertBytes`, `ExpertCache` with the statistics, `ExpertReadBytesPerSecond`;
`LoadedModel.DiskExperts`), plan with `ExpertStore.EstimateDecode` (the same estimate as the load log, from the bytes a token selects, the read
speed and the cache), and feed manifests list the routed experts (`requirements.memory[].expertWeights` / `expertBytesPerToken`, model-feed.md).
The fit planner (`FitQuery.ExpertsOnDisk`) reports such a load as `NeedsOffload` with `ExpertsFromDisk` instead of `DoesNotFit`. Decode then
runs at the disk's speed: a step reads the selected experts it does not find in the cache (DeepSeek-V3 at Q4_K_M: up to 12 GB per token).

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
filters and badges (verified catalog, Hub search) use `DeviceMemory.UsableBytes` (the capacity); the feed's badges plan with
`Available` and the resident models like the load dialog. Studio reads these figures again when a picker opens, after loads and
unloads and every 5 s while one is visible (studio.md "Memory figures"). `DeviceOptions.MemoryBudget` caps
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
- Window-sized KV caches for sliding-window layers (`window_kv`) also serve single-sequence sessions since benchmarks.md §77 (the one-slot
  paged form with a ring of window rows, used where the ring at most halves those caches): Gemma 4 31B Q4_K_M at 32K takes 21.1 GiB
  on the session path (was 44.8 GiB), 128K fits a 32 GB card (28.2 GiB, f16 caches). The one-slot paged form costs ~2 % of a CUDA
  decode step against the contiguous cache (none on Vulkan).
