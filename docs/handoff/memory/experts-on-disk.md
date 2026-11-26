---
name: experts-on-disk
description: Owner-approved feature (2026-11-22): MoE experts streamed from disk with a RAM/VRAM expert cache, io_uring on Linux
metadata:
  type: project
---

On 2026-11-22 the owner approved an "experts on disk" mode as a competitive edge: MoE models larger than RAM + VRAM keep their routed experts on disk, read them on demand into an expert cache (RAM, optionally VRAM), and target ~1 tok/s decode on huge models (DeepSeek-V3 / Kimi-K2 class) from a fast NVMe.

**Why:** demos of compatibility with huge models; llama.cpp's mmap paging is slow (page faults, no queue depth).

**How to apply:**
- On Linux use io_uring (owner's explicit requirement). Fall back to threaded pread where io_uring is blocked (seccomp, old kernels). Other platforms use normal async file I/O.
- Later, not now: a DirectStorage backend on Windows (owner, 2026-11-22). The reader sits behind a batch-read interface so it can plug in.
- The mode is opt-in. Disk-backed experts must not be page-locked.
- Output must match the resident run.
- Related: [[marketing-perf-claims]] (promote it as a lead once measured), [[device-features-fallbacks]].

Status 2026-11-22: built in perf/offload-r6 (merged 10b5ac10). DeepSeek-V3.1 Q4_K_M (405 GB) ran at ~1 tok/s on 62 GB RAM + 32 GB VRAM, against llama.cpp's 0.62 tok/s. The container used at the time (a RunPod pod) blocked io_uring (seccomp EPERM), so the pread fallback served the reads (7.4 GB/s). Its DeepSeek text diverged from llama.cpp at token 4, unresolved; resolve it before promoting the demo.

Update 2026-11-23: §87 resolved the DeepSeek-V3.1 divergence (verified). The GLM-5.2 demo files and the truncated checkpoints were deleted for disk (owner). DeepSeek-V4-Flash is being re-downloaded to local disk as the new demo while its CUDA multi-row bug is fixed (fix/dsv4-cuda-disk).
