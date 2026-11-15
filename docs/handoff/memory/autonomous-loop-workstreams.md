---
name: autonomous-loop-workstreams
description: "Since 2026-11-14 the user asked for an autonomous loop: model bring-up, LLM perf (single + multi-user) vs vLLM/llama.cpp, ONNX coverage + perf parity with onnxruntime (CPU and Vulkan)"
metadata:
  node_type: memory
  type: project
  modified: 2026-11-14T05:43:58.877Z
---

On 2026-11-14 the user asked to "work in an autonomous loop": bring up more models, improve single-user and multi-user
inference performance to close gaps to vLLM and llama.cpp, implement more of ONNX, and optimize ONNX operators/inference
to parity with onnxruntime on CPU and Vulkan.

**Why:** long-running goal; see [[project-goals-parity]].

**How to apply:** keep ~3 worktree agents running (GPU is shared, see [[benchmarking-shared-gpu]]): LLM performance
rounds (benchmarks.md sections 16/19/20...), model bring-up (docs/model-support.md), ONNX coverage then ONNX perf vs
onnxruntime. On each completion: privacy scan, merge to main, build, suites, remove worktree, start the next round.
Push only when the user says so ([[public-repo-privacy]]). Studio build for the user lives in ~/Apps/SharplightStudio
(dotnet publish self-contained linux-x64). Follow [[generic-optimizations-provider-apis]].
