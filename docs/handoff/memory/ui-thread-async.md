---
name: ui-thread-async
description: "Standing rule (user 2026-11-16): inference and all heavy work must never run on the UI thread; every task (all model kinds) async off the main thread, in Studio and in the engine's async APIs"
metadata:
  type: feedback
---

User (2026-11-16) after testing image models in Studio: "it works but blocks the UI... any inference should always be
async of the main thread... this is for all tasks".

- Studio: model loading/compilation, generation, transcription, synthesis, vision, precompile, conversion, downloads,
  library scans run off the UI thread; only state updates are marshaled back via the dispatcher; progress throttled.
- Engine: `...Async` task-client methods must not do heavy synchronous work on the caller's thread before their first
  real await (offload compute to a worker / dedicated thread, honor CancellationToken); streaming callbacks never block.
- Tests: a headless Studio test asserts the UI thread stays responsive during a run (e.g. dispatcher heartbeat).

**Why:** a frozen UI during multi-second image generation is unacceptable for a desktop app.
**How to apply:** check every new page/task client for this; put it in Studio/task-layer agent briefs. Related:
[[core-task-utilities]], [[code-style-rules]].
