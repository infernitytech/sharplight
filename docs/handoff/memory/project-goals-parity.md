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

Owner 2026-11-20: vLLM parity is the minimum ("at least"); beating it ("a bit more performance") is wanted for marketing. Target ≥1.0x on every vLLM comparison, aim about 1.1x on the headline numbers (gpt-oss-20b and Qwen3-8B-FP8 serving at 16/64 streams, prefill at 2048/8192). Keep a perf agent on this continuously until it is reached ([[three-workstreams]]).
Owner 2026-11-20: long prompts are common (256K+ context models); every perf comparison must include long contexts (prefill and decode at 16K–128K+, TTFT at 64K/128K), not just ≤8192. The gap grows with prompt length, so long context is a primary target.
Owner 2026-11-22, restated while feature rounds were added (experts on disk, multi-node, huge models): "The first goal is still performance parity or better with vllm". Perf rounds (perf/llm-*) get GPU priority over feature, bring-up, Studio and server agents. Always keep one perf round running.
Owner 2026-11-25, end goal restated: parity on ALL models and benchmarks with vLLM and llama.cpp, and with onnxruntime for ONNX models and the reference implementations for other model types (image/video/audio, e.g. PyTorch for SeedVR2), plus some areas where Sharplight is clearly faster. Track every known gap (website "Where Sharplight is behind" + the work queue in docs/handoff/README.md) and keep perf rounds closing them, behind-list first.
