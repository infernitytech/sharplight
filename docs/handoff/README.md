# Handoff: moving the autonomous work to a new machine (2026-11-15)

This branch (`handoff/new-machine`) is `main` as of 2026-11-15 plus this folder. Everything the next session needs to
continue without interruption is here or on the `wip/*` branches.

## 1. Restore on the new machine

```sh
git clone https://github.com/infernitytech/sharplight.git && cd sharplight
git fetch origin 'refs/heads/*:refs/remotes/origin/*'
git checkout main && git merge --ff-only origin/handoff/new-machine   # main = handoff state
for b in core-restructure onnx-ops whisper diffusers vulkan-matvec-rows; do git branch wip/$b origin/wip/$b; done
git config user.email "826745+BernhardGlueck@users.noreply.github.com"    # noreply only, see memory/public-repo-privacy.md
```

Claude Code memory: copy `docs/handoff/memory/*.md` into the new machine's project memory directory
(`~/.claude/projects/<path-of-the-clone-with-slashes-as-dashes>/memory/`). `MEMORY.md` is the index. Then tell Claude
"continue the autonomous loop from docs/handoff/README.md".

Toolchain: .NET SDK 11.0.100 (installed with dotnetup), Vulkan driver + loader (`glslc` is used by some tools), Python 3
with per-task venvs (transformers, torch CPU, onnxruntime, gguf, diffusers, mlx CPU... created on demand in a scratch
dir, never into system Python). llama.cpp reference binaries `runtime.llama-b11177-{cuda13,vulkan,cpu}` and a build of
llama.cpp 8ca5a80 go under `$LLAMA_ROOT` (set it in the shell profile; the docs refer to `$LLAMA_ROOT`).

Model assets are not in git (~900 GB on the old machine). Layout expected by tests and docs:

| dir | content | size (old machine) | priority to copy |
|---|---|---|---|
| `~/models/hf/tiny-*`, `~/models/gguf/tiny-*` | tiny random test models (also regenerable with `tools/mk_tiny_*.py`, `tools/convert_tiny_gguf.py`) | 7 GB | **must** (tests skip without them) |
| `~/models/audio`, `~/models/whisper.cpp`, `~/models/sd` | jfk.wav, whisper.cpp ggml models, SD 1.5 (gguf + diffusers), tiny-random-sd | 18 GB | **must** for the speech / image workstreams |
| `~/models/onnx` | ONNX survey models (`bench/onnx/download_models.py` re-downloads) | 29 GB | should |
| `~/models/gguf`, `~/models/hf`, `~/models/hf-vllm`, `~/models/sharplight` | real checkpoints for parity / sweeps (docs/model-support.md section 8) | ~800 GB | optional, re-download from the Hub as needed |

`rsync -a --info=progress2 oldbox:models/ ~/models/` is the simplest route if both machines are reachable.

## 2. Long-term goal (user, 2026-11-15) — see memory/core-task-utilities.md

1. Support **all model types**: LLMs (all llama.cpp archs, done 168/172 + inkling), speech-to-text, text-to-speech,
   image/video generation, vision (classification/embeddings, detection, segmentation, depth, OCR, upscalers), audio
   (codecs, music, VAD, separation, CLAP), speech-to-speech, time series, retrieval (ColBERT/ColPali, SPLADE), 3D, bio,
   robotics VLA; plus all extra weight formats and containers.
2. **Layering**: the core engine stays as it is; only operators etc. are added to Core to make models work.
   Sharplight.Llm folds into Core as higher-level inference utilities on compiled models; generic task utilities for the
   other model types (speech recognition, synthesis, image generation, ...) also live in Core. Importers and
   execution-graph generation (architecture adapters) stay separate assemblies.
3. Everything storable in the **.slm** package format (multi-component models, configs, tokenizers, schedulers, task
   metadata).
4. **Studio overhaul**: inference, downloading etc. for all model types, with dedicated non-chat UIs where needed.
5. For now: feature work first; no benchmark rounds until the user asks again. Performance parity (vLLM, llama.cpp,
   onnxruntime) remains a goal for later.

## 3. State of `main` (5f16fc09 + this folder)

Merged on 2026-11-15: quantized HF checkpoints (NVFP4, MXFP4, GPTQ, AWQ, bitsandbytes NF4, MLX 4-bit, DeepSeek-V4
native, GGUF NVFP4 multipliers); ONNX StringSplit / MurmurHash3 / WordConvEmbedding, opt-in fast matrix precision,
parity round 7 (benchmarks section 33); the importer rename to `Sharplight.Import.Gguf / .HuggingFace / .Onnx` and
`Sharplight.Package.Import` folded into `Sharplight.Package`; bring-up round 11. Build: 0 errors; suites run green after
the merges: Cpu 1272, Import.HuggingFace 197, Import.Gguf 66, Onnx 448, Package 32, Hub 90, Tokenizers 456, Vulkan
QuantFormatTests 179. Not run after the last merge: full Vulkan, Families, Vision, Serving, Embeddings, Studio — run them
first on the new machine.

## 4. Work in progress (`wip/*` branches, all based on main or its parent; merge `main` into them first)

| branch | what | state | next step |
|---|---|---|---|
| `wip/core-restructure` | Llm + Tokenizers + Templating folded into Core, BuiltinAdapters moved to Architectures, `Sharplight.Defaults` one-call registration (commit e3c86c19); task-layer skeleton started (`src/Sharplight.Core/{Tasks,Speech,Audio,Imaging}`, last commit WIP) | restructure commit done; skeleton unfinished, unverified | finish the skeleton per section 2 (open task abstraction, .slm model kind + components description, docs/api-overview.md + docs/model-format.md), build, run all suites, merge. **Do this first**: the speech/image/Studio work builds on it |
| `wip/onnx-ops` | ONNX missing ops round: Slice with runtime starts/ends, MatMul with leading unit dims, LFM2 imports (37ff6e99) + uncommitted translator work (WIP commit) | partial | continue the list: com.microsoft GroupQueryAttention / RotaryEmbedding / SimplifiedLayerNormalization (+Skip), NonMaxSuppression, RoiAlign, Loop, STFT (generic Core DSP op, reused by speech), runtime Range, bertsquad CPU i32 Scale; docs/onnx-support.md |
| `wip/whisper` | Whisper: HF mapping, mel filter bank, log-mel spectrogram, SpeechKeys (WIP) | early, unverified | rebase onto the restructure; add whisper.cpp ggml `.bin` importer; speech-recognition task utility; verify vs transformers + whisper.cpp on `~/models/audio/jfk.wav` |
| `wip/diffusers` | `Sharplight.Import.Diffusers` (diffusers + LDM single-file names, UNet/VAE/CLIP configs), CLIP tokenizer (WIP) | early, unverified | rebase onto the restructure; graph generation for UNet/VAE/CLIP in Architectures; image-generation task utility (schedulers, CFG); SD1.5 vs diffusers/stable-diffusion.cpp; then SDXL, SD3, FLUX, Chroma, Qwen-Image, Wan |
| `wip/vulkan-matvec-rows` | perf: two output rows per workgroup in split-K matvec (FP8/bf16) | perf experiment | parked until performance work resumes |
| `macos/round-1` (already on GitHub) | macOS / MoltenVK / NEON / Studio .app | separate until the user decides | — |

Not started (planned): extra weight formats (Marlin GPTQ/AWQ, AWQ gemv_fast, AutoRound, HQQ, EETQ, quanto, torchao,
Quark int/MXFP6, EXL2/EXL3, AQLM, VPTQ, HIGGS, SpQR, FP-Quant, MLX FP4 + other bit widths, bitsandbytes FP4/int8 kept
quantized, GGUF I2_S, safe PyTorch pickle `.bin/.pt/.ckpt` loader); the earlier agent produced nothing committable.

## 5. Plan / queue

1. Finish `wip/core-restructure` and merge (alone; it touches everything).
2. In parallel after that (the new machine allows more agents; still use worktrees, see memory/git-worktrees-parallel.md):
   ONNX ops (`wip/onnx-ops`), weight formats, Whisper STT, image generation (SD family), TTS (qwen3tts, pockettts,
   wavtokenizer-dec / OuteTTS, Kokoro, Orpheus+SNAC, CSM+Mimi, Dia).
3. Then: vision tasks, audio (codecs, MusicGen, Stable Audio, CLAP, VAD, Demucs), video generation, speech-to-speech
   (Moshi, Omni talkers), time series, retrieval, 3D, bio, VLA.
4. Studio overhaul once the task layer is on main.
5. Per round: privacy scan of the diff (no personal email, no home paths), merge to main, build, suites, remove the
   worktree. Push to GitHub only when the user says so.

Agent briefs: docs/bringup-brief.md (architecture rules; note it still says `master` — the branch is `main`, and the
importer projects are now `Sharplight.Import.*`), docs/cleanup-brief.md, docs/code-style.md. Tests are xunit v3
executables: `dotnet run --project tests/X -c Release --no-build -- -class "*Name"`.
