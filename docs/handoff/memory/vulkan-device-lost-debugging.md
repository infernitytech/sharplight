---
name: vulkan-device-lost-debugging
description: "How to localize Vulkan VK_ERROR_DEVICE_LOST in Sharplight (GPU-AV, launch bisect, Xid in journal)"
metadata:
  node_type: memory
  type: reference
  modified: 2026-11-13T03:07:16.910Z
---

Device-lost that depends on test order / memory history is usually an out-of-bounds BDA access that only faults when the buffer ends at an allocation edge.

- GPU-assisted validation: `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_KHRONOS_VALIDATION_GPUAV_ENABLE=true` reports "Out of bounds access" with dispatch index, invocation id and load size (SHARPLIGHT_VK_VALIDATION=1 alone did not report it).
- `SHARPLIGHT_VK_LIST_LAUNCHES=1` prints each recorded launch (name, kernel key, grid); `SHARPLIGHT_VK_MAX_LAUNCHES=n` records only the first n launches, for bisecting.
- `journalctl -k | grep Xid`: NVIDIA Xid 109 (ctx switch timeout) / 31/32 identify hangs vs faults.

Found this way (2026-11-13): gemm_cm2 prefetched a tile at K for empty split-K slices. Related: [[benchmarking-shared-gpu]], [[device-features-fallbacks]].

Also found (2026-11-13): Slang `ptr + a - b` (pointer minus integer, e.g. `blk + 144 - 128`) miscompiles into a GPU hang (Xid 109, no GPU-AV report); sum the offset first: `blk + (a - b)`.

Also found (2026-11-13): this NVIDIA driver drops the high bits of 64-bit pointer offsets ≥ 4 GiB inside one buffer (wrong data, no fault). Kernels must compute big offsets on integers via `rowStart`/`byteAt` in common.slang; test with LargeTableGatherTests (SHARPLIGHT_TEST_LARGE=1).
