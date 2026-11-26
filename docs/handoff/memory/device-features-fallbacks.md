---
name: device-features-fallbacks
description: "Run on lots of hardware: optional GPU features detected+disableable with fallbacks; never assume memory topology (small BAR, low VRAM, unified memory) — every allocation path falls back"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-11-12T21:07:47.534Z
---

Device extensions/features (tensor cores, fp16, int8 dot, subgroup control, ...) must be detected per device and only used when detected AND enabled; hosts can disable them when opening a device or compiling a model. Names must be provider-neutral (Core `DeviceFeatures`: matrix_f16, matrix_bf16, matrix_int8, fp16, bf16, int_dot, subgroup_ops, subgroup_size_control), not Vulkan-specific. Every feature-specific kernel needs a portable fallback.

**Why:** User (2026-11-12): "We need to run on other gpus as well so if the features are not there or not enabled, the extensions must not be used and we need fallbacks", "use common flag names not vulkan specific names".

**How to apply:** Gate new fast paths on `LoweringOptions.Has(DeviceFeatures.X)` (effective = detected − DeviceOptions.DisabledFeatures − CompileOptions.DisabledFeatures − SHARPLIGHT_DISABLE_FEATURES env); don't enable driver extensions for disabled features. Test portability on the local AMD iGPU (vulkan:1, subgroup 64, no coopmat) and llvmpipe (vulkan:2, subgroup 8). Related: [[format-agnostic-core]].

**Resilience across hardware (user, 2026-11-19, after the small-BAR staging OOM):** "we need to be resilient and work on lots of hardware". Beyond features: never assume memory topology or sizes — small/absent resizable BAR (~256 MiB host-visible device-local heap shared by all processes), low VRAM, no host-cached type, unified memory (iGPU/Apple), heaps shared with other processes. Every allocation path needs a fallback (other memory type, smaller chunks, host memory, CPU offload) and a clear error only when nothing works. Treat a hardware-specific failure found on any machine as a resilience bug to fix generally, and add a test that simulates the constraint (env caps, llvmpipe).
