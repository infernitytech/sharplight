---
name: scope-audio-image-generation
description: "Since 2026-11-15 audio (whisper.cpp STT, llama.cpp TTS archs) and image generation (stable-diffusion.cpp models) are in scope"
metadata:
  node_type: memory
  type: project
  modified: 2026-11-15T03:30:02.149Z
---

On 2026-11-15 the user asked why audio decoders were "out of scope" (that was an unflagged scoping call from the
model-support survey, not the user's) and asked to also support the models of whisper.cpp
(github.com/ggml-org/whisper.cpp) and stable-diffusion.cpp (github.com/leejet/stable-diffusion.cpp).

**Why:** parity goal extends beyond text LLMs; see [[project-goals-parity]].

**Architecture (user, 2026-11-15):** "the core engine runs them and the rest becomes import" — no separate audio or
diffusion engines; models are graphs of generic Core ops run by the existing runtime, model knowledge only in importers
and architecture adapters; only I/O and the sampling/decoding loops on the host.

**How to apply:** treat audio (STT + TTS) and image/video generation as regular workstreams in the autonomous loop
([[autonomous-loop-workstreams]]), with the same rules (importers in own assemblies per [[format-agnostic-core]],
provider-abstraction APIs per [[generic-optimizations-provider-apis]]). Don't declare whole model classes out of scope
without asking the user.
