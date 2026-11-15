---
name: generic-optimizations-provider-apis
description: Optimizations must be generic (benefit other models/architectures), and any consumer-facing API must go through the provider abstraction so other providers can implement it
metadata:
  type: feedback
---

Every optimization should be generic: implement it at the op/IR/lowering level so every architecture using that op benefits (e.g. a chunked delta rule helps all gated-delta-net models; MTP/speculative decoding should work for any model with draft heads), not as a model-specific hack. Any new API that consumers of the engine see must be exposed through the provider/device abstraction (interfaces like IComputeDevice, provider registry, device features) so other execution providers (CPU, future CUDA, ...) can implement it.

**Why:** User, 2026-11-13, when starting the vLLM-gap work (benchmark, chunked delta rule, MTP): "Make sure that all the optimizations are generic and apply to other models as well if possible. Also any new apis must be exposed over the provider ... so that other providers can implement them."

**How to apply:** Put this requirement in every agent brief for performance/feature work; when reviewing, reject model-name special cases in kernels/lowering and Vulkan-only public APIs. Related: [[device-features-fallbacks]], [[format-agnostic-core]].
