---
name: cuda-provider
description: "User 2026-11-18 (reversing 'skip cuda'): build a CUDA execution provider to close vLLM gaps; must need nothing but the NVIDIA driver (ship CUDA runtime libs ourselves like PyTorch; license allows); Vulkan must not lag behind"
metadata:
  type: project
---

User (2026-11-18): "We need a cuda execution provider to close more performance gaps. So start working on this, but keep our roadmap in mind and do not let vulkan lag behind... It is especially important that the cuda provider does not need anything on a machine except a nvidia driver. So we need to ship the cuda dlls ourselves like pytorch does. This is allowed by the license."

**Why:** Vulkan on NVIDIA lacks >48 KB shared memory, TMA/async bulk copies, thread-block clusters (benchmarks §44.3/§46); FP8 prefill ~0.73x and MoE serving ~0.61x of vLLM.
**How to apply:** CUDA provider behind the same provider abstraction / device features with fallback (Vulkan stays default elsewhere, provider order decided per device). Runtime dependency = driver only (libcuda / nvcuda.dll from the driver); redistributable CUDA libs (cudart, cuBLAS/cuBLASLt, NVRTC if needed) shipped in our own native runtime packages per RID. Keep a Vulkan perf agent running in parallel; report gaps per provider. Related: [[project-goals-parity]], [[device-features-fallbacks]], [[generic-optimizations-provider-apis]].
User 2026-11-19: ship Studio WITH the CUDA provider (~45 MB measured; user: "the size difference does not matter") — Linux + Windows packages include Sharplight.Cuda + Sharplight.Cuda.Runtime; Vulkan stays default until CUDA is proven faster per device.
User 2026-11-19: cuBLASLt (541 MB) stays OPTIONAL and out of Studio — "Try to match it with our own kernels first".
User 2026-11-19: Sharplight.Cuda NuGet package SHOULD depend on Sharplight.Cuda.Runtime automatically.
