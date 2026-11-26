---
name: machine-runpod-3gpu
description: "HISTORY: the single RunPod container used 2026-11-15..26 (3 GPUs, then 1 GPU + ~10 cores after 2026-11-19); lessons that carry over to any machine"
metadata:
  type: project
---

**Previous setup (2026-11-15 to 2026-11-26).** All development ran on one RunPod container (Ubuntu 24.04, root). On
2026-11-26 the owner moved development to several machines running their own agent platform; this note is kept for the
lessons, not as a description of the current machine.

- 2026-11-15 to 2026-11-19: 3x NVIDIA RTX PRO 4500 Blackwell (32 GB each), AMD EPYC 7443P, ~30-core CPU quota, 251 GB RAM.
  All GPUs dedicated to Sharplight: one GPU-heavy agent per GPU, suites in parallel, one GPU reserved for merge verification.
- From the 2026-11-19 relaunch (RunPod capacity shortage, confirmed intentional by the owner): 1 GPU (RTX PRO 4500, no
  resizable BAR: 256 MiB BAR1), ~10-core CPU quota, 62 GB cgroup memory limit (`free` showed 251 GB, misleading), 1 TB local
  disk. Caps then: ~3 agents (1 engine on the GPU, 1 Studio, 1 Server), 2 concurrent builds, verification on the same GPU
  behind a lock.
- Persistent network volume (FUSE) for repos, toolchains, models and logs; everything else was wiped on a pod relaunch.
  The network FS could drop below 10 MB/s and stall processes in D state for hours, so working clones, worktrees, venvs
  and repeatedly read models lived on local disk.

**Lessons that apply to every machine:**
- Read real limits from cgroups (cpu.max, memory.max, memory.stat `anon`), not nproc/free. Whole-suite Vulkan test
  processes reach ~28 GB RSS; with a tight memory limit run big GPU suites one class per process (tools/agent/run-suites.sh does).
- Vulkan in containers needs `apt-get install libegl1 libgl1` (the NVIDIA ICD fails with ErrorIncompatibleDriver without
  glvnd's libEGL.so.1).
- Pin a process to a GPU with `SHARPLIGHT_VK_DEVICES=<index>` and, for llama.cpp references, `GGML_VK_VISIBLE_DEVICES=<index>`
  (llama.cpp Vulkan grabs every GPU otherwise); `CUDA_VISIBLE_DEVICES` for torch/vLLM.
- Small-BAR GPUs exposed real resilience bugs (staging OOM): see [[device-features-fallbacks]].
- io_uring may be blocked by seccomp in containers; experts-on-disk falls back to pread ([[experts-on-disk]]).
- HF token: read-only, user-provided, kept in a mode-600 file outside every repo and exported as HF_TOKEN by the env
  script. Never print or commit it. Gated repos need it.
- User 2026-11-15: trust_remote_code generators for tiny test models may run, sandboxed (own venv, offline, no credentials).
- The user wants subagents on cheaper models (sonnet/haiku) where possible.

**How to apply:** on a new machine, size agent caps, build slots and GPU assignment from its real limits; keep the
rules of [[keep-machine-responsive]] and [[work-dirs-cleanup]]. Related: [[benchmarking-shared-gpu]].
