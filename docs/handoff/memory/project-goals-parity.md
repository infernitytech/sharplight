---
name: project-goals-parity
description: "Sharplight long-term goals - support every llama.cpp architecture, be competitive with llama.cpp/vLLM, production serving utilities"
metadata:
  node_type: memory
  type: project
  modified: 2026-11-13T04:07:27.192Z
---

User (2026-11-13): Sharplight should support every architecture llama.cpp supports (Qwen3.8-Next explicitly named as important), stay modular/separated per format and architecture, be performance-competitive with llama.cpp or even vLLM (verify with benchmarks), add production scalable-inference utilities in the relevant projects, keep trying more ONNX models with benchmarks, and keep the code clean ([[code-style-rules]]).

**Why:** the goal is a production-grade .NET inference runtime, not a demo.

**How to apply:** prioritize breadth of architectures plus measured performance against llama.cpp. llama.cpp binaries live at $LLAMA_ROOT/runtime.llama-b11177-{cuda13,vulkan,cpu}/. Use feature agents in worktrees ([[git-worktrees-parallel]]); benchmark per [[benchmarking-shared-gpu]].
