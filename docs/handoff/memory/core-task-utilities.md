---
name: core-task-utilities
description: "LONG-TERM GOAL (user, 2026-11-15): every model type (LLM, STT, TTS, image/video gen, vision, audio, time series, retrieval, 3D, bio, VLA) on one core; task utilities in Core; all storable in .slm; Studio UIs per type"
metadata:
  node_type: memory
  type: project
  modified: 2026-11-15T04:10:36.273Z
---

User's long-term goal, stated 2026-11-15 ("Remember this as the long term goal"):

1. Support all model types: LLMs plus speech-to-text (Whisper, Moonshine, Parakeet/Canary, Voxtral...), text-to-speech
   (llama.cpp TTS archs, Kokoro, Orpheus/SNAC, CSM/Mimi, Dia), image/video generation (stable-diffusion.cpp set: SD1/2,
   SDXL, SD3, FLUX, Chroma, Qwen-Image, Wan, LTX...), vision (ViT/DINO/CLIP/SigLIP, DETR/YOLO, SAM 2, depth, OCR,
   upscalers), audio beyond speech (MusicGen, Stable Audio, CLAP, VAD, Demucs, codecs EnCodec/DAC/SNAC/Mimi),
   speech-to-speech (Moshi, Omni talkers), time series (Chronos, TimesFM, Moirai), retrieval (ColBERT/ColPali, SPLADE),
   3D (TripoSR, Hunyuan3D, VGGT), bio (ESM, Evo, DNA models), robotics VLA (OpenVLA, pi0) — and all the extra weight
   formats / containers (HF quant methods, EXL2/3, AQLM, PyTorch pickle, whisper.cpp ggml, SD single-file...).
2. Layering: the core engine stays as it is; only add operators etc. to Core to make models work. Sharplight.Llm folds
   into Core as higher-level inference utilities on top of compiled models; similar generic task utilities (speech
   recognition, synthesis, image generation, ...) go into Core. Importers and execution-graph generation (architecture
   adapters) stay separate assemblies.
3. Everything must be storable in the .slm package format (multi-component models, configs, tokenizers, schedulers).
4. Overhaul Sharplight Studio: inference, downloading etc. for all model types, with their own non-chat UIs where needed.

**Why:** extends [[scope-audio-image-generation]] and [[project-goals-parity]]; one engine, model knowledge only in
importers/adapters ([[format-agnostic-core]]).

**How to apply:** plan autonomous-loop rounds ([[autonomous-loop-workstreams]]) around these model classes; never scope
a class out without asking. On 2026-11-15 the user also asked to keep GPU/RAM use low and do feature work first; no
benchmark rounds until asked ([[benchmarking-shared-gpu]]).

Owner 2026-11-24 (SeedVR2 bring-up): "Make sure that the core runtime only gains generic operations for this.... and that seedvr2 specific handling stays in architectures / adapters etc..." **How to apply:** every bring-up brief states it; before merging, grep the Core/Cpu/Shaders/Cuda diff for model/vendor names and require a generic justification for each new op.
