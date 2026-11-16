# Handoff: moving the autonomous work to a new machine (2026-11-15)

`main` (and the `handoff/new-machine` branch) as of 2026-11-15 include this folder. Everything the next session needs to
continue without interruption is here or on the `wip/*` branches.

## 1. Restore on the new machine

```sh
git clone https://github.com/infernitytech/sharplight.git && cd sharplight   # main contains this handoff
for b in core-restructure onnx-ops whisper diffusers vulkan-matvec-rows; do git branch wip/$b origin/wip/$b; done
git config user.email "826745+BernhardGlueck@users.noreply.github.com"    # noreply only, see memory/public-repo-privacy.md
```

Claude Code memory: run `docs/handoff/restore.sh`, which copies `docs/handoff/memory/*.md` into the clone's project
memory directory (`~/.claude/projects/<clone path, non-alphanumerics as dashes>/memory/`; `MEMORY.md` is the index).
`CLAUDE.md` at the repo root tells a new session to do this itself. Then tell Claude "continue the autonomous loop from
docs/handoff/README.md".

Toolchain: .NET SDK 11.0.100 (installed with dotnetup), Vulkan driver + loader (`glslc` is used by some tools), Python 3
with per-task venvs (transformers, torch CPU, onnxruntime, gguf, diffusers, mlx CPU... created on demand in a scratch
dir, never into system Python). llama.cpp reference binaries `runtime.llama-b11177-{cuda13,vulkan,cpu}` and a build of
llama.cpp 8ca5a80 go under `$LLAMA_ROOT` (set it in the shell profile; the docs refer to `$LLAMA_ROOT`).

Multi-GPU machines: `SHARPLIGHT_VK_DEVICES=<i>[,<j>...]` makes the Vulkan provider list only the devices with those
enumeration indices (see `sharplight --devices`), so every test, sample or app that picks "the first discrete GPU" runs on
the pinned one. Give each GPU-heavy agent or test run its own index. `DeviceInfo.Index` keeps the real enumeration index.
Test-only overrides that index into the filtered list also exist (`SHARPLIGHT_TEST_DEVICE` in the Vulkan tests,
`SHARPLIGHT_TEST_VK_DEVICE` in the ONNX/Vision/Families tests); prefer `SHARPLIGHT_VK_DEVICES`.

Containers (e.g. RunPod with the NVIDIA runtime): the NVIDIA Vulkan ICD needs glvnd's `libEGL.so.1`. Without it the
loader reports `ErrorIncompatibleDriver`. Fix: `apt-get install libegl1 libgl1`.

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

## 3a. State of `develop` (2026-11-15 evening, new machine)

`develop` (pushed) = `main` + everything merged on the new machine: SHARPLIGHT_VK_DEVICES; release engineering (CI for
develop/main/release/*, NuGet from release/*, Velopack Studio packages for 5 RIDs x stable/unstable, update UI; unstable
releases live on download.sharplight.dev); Core restructure (Llm/Tokenizers/Templating in Core, DI task layer, `.slm`
model description + attachments, `services.AddSharplight()`); safe PyTorch pickle loader; weight formats AQLM, MLX
2/3/5-bit + mxfp4/nvfp4/mxfp8, AutoRound; ONNX ops round 8 and ONNX perf (benchmarks section 35); LLM perf (section 34;
split-K matvec rows, SM-sized GEMM splits and the faster causal prefill attention kernel on by default); speech-to-text
(Whisper HF + whisper.cpp, Distil-Whisper, beam search, /v1/audio/transcriptions); text-to-speech round 1 (OuteTTS +
WavTokenizer); image generation (SD 1.5 / 2.x / SDXL; diffusers, single file, .ckpt, sd.cpp GGUF; /v1/images/generations);
vision (ViT, DINOv2, CLIP/SigLIP zero-shot, DETR, RT-DETR, Depth Anything V2); two-stage precompilation in the provider
contract + per-user artifact store + Studio precompile; Studio S0/S1 (kinds, Discover filters), S2 (shared inference page
blocks, Transcribe and Speak pages); regenerated tiny test models and goldens (tools/model-assets).
Suites on develop: all green except 49 ONNX conformance (known) and 1 Hub test (tests run as root).

In flight (branches/worktrees): public API pass (static registries -> DI; final develop merge), macOS round 1 (CI green,
final merge pending), weight formats A2 (bitsandbytes FP4/int8, I2_S, HQQ, EETQ) and A3 (Marlin, gemv_fast, quanto,
torchao, Quark), weight formats B leftovers (FP-Quant unvalidated, EXL2/3, VPTQ, HIGGS, SpQR, AQLM kernel), Studio S2b
(Image + Vision pages). Next: cleanup pass, speech round 2 (Moonshine, Parakeet, Canary, Voxtral), TTS round 2 (Kokoro,
Orpheus+SNAC, CSM+Mimi, Dia, qwen3tts, pockettts), audio (codecs, MusicGen, Stable Audio, CLAP, VAD, Demucs), SAM 2,
video generation, then the rest of waves 2-3.

Machine notes: tests read ~/models through a local mirror (the machine scripts `sl-test`, `run-suites.sh`,
`sync-local-models.sh`); keep builds niced (`nice -n 15 ionice -c3`, `-m:4`) and at most ~6 agents.

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
| `wip/core-restructure` | Llm + Tokenizers + Templating folded into Core, BuiltinAdapters moved to Architectures, `Sharplight` one-call registration (commit e3c86c19); task-layer skeleton started (`src/Sharplight.Core/{Tasks,Speech,Audio,Imaging}`, last commit WIP) | restructure commit done; skeleton unfinished, unverified | finish the skeleton per section 2 (open task abstraction, .slm model kind + components description, docs/api-overview.md + docs/model-format.md), build, run all suites, merge. **Do this first**: the speech/image/Studio work builds on it |
| `wip/onnx-ops` | ONNX missing ops round: Slice with runtime starts/ends, MatMul with leading unit dims, LFM2 imports (37ff6e99) + uncommitted translator work (WIP commit) | partial | continue the list: com.microsoft GroupQueryAttention / RotaryEmbedding / SimplifiedLayerNormalization (+Skip), NonMaxSuppression, RoiAlign, Loop, STFT (generic Core DSP op, reused by speech), runtime Range, bertsquad CPU i32 Scale; docs/onnx-support.md |
| `wip/whisper` | Speech-to-text on the task layer: Whisper from HF checkpoints and whisper.cpp `.bin` (new `Sharplight.Import.WhisperCpp`), Core `ISpeechRecognizer` implementation (host log-mel, decoding loop, long-form), `.slm` description, `sharplight-transcribe`, `/v1/audio/transcriptions` | Whisper tiny/base/small (+ .en, q5_1) and Distil-Whisper verified (greedy and 5-beam search) vs transformers (token-exact) and whisper.cpp, CPU and Vulkan (docs/model-support.md section 13) | merge to develop; then Moonshine, Parakeet (CTC/TDT), Canary, Voxtral (Moonshine needs a raw-waveform, variable-length encoder with rotary attention and an unprompted decoding loop); temperature fallback, word timestamps; switch the feature extractor to the generic STFT op once `wip/onnx-ops` lands it |
| `wip/diffusers` | `Sharplight.Import.Diffusers` (diffusers + LDM single-file names, UNet/VAE/CLIP configs), CLIP tokenizer (WIP) | early, unverified | rebase onto the restructure; graph generation for UNet/VAE/CLIP in Architectures; image-generation task utility (schedulers, CFG); SD1.5 vs diffusers/stable-diffusion.cpp; then SDXL, SD3, FLUX, Chroma, Qwen-Image, Wan |
| `wip/vulkan-matvec-rows` | perf: two output rows per workgroup in split-K matvec (FP8/bf16) | perf experiment | parked until performance work resumes |
| `macos/round-1` (already on GitHub) | macOS / MoltenVK / NEON / Studio .app | separate until the user decides | — |

Not started (planned): extra weight formats (Marlin GPTQ/AWQ, AWQ gemv_fast, AutoRound, HQQ, EETQ, quanto, torchao,
Quark int/MXFP6, EXL2/EXL3, AQLM, VPTQ, HIGGS, SpQR, FP-Quant, MLX FP4 + other bit widths, bitsandbytes FP4/int8 kept
quantized, GGUF I2_S, safe PyTorch pickle `.bin/.pt/.ckpt` loader); the earlier agent produced nothing committable.

## 5. Plan / queue (roadmap, 2026-11-15, new 3-GPU machine)

Every model class of the long-term goal (section 2) is on this list; none is out of scope without asking the user.
Each wave runs as parallel worktree agents (about 6-8 at a time, GPU-heavy agents pinned with `SHARPLIGHT_VK_DEVICES`).
A model class is done when it imports, runs through a Core task utility, converts to `.slm` and loads from it on the
fast path (page-aligned, mmap-able weights, per-component precompilation), shows correctly in the Studio `.slm`
inspector, matches its reference implementation, and has a Studio page. The `.slm` description stays generic: open
kind and component-role strings, per-component JSON config and attachments, nothing LLM-specific.

**Step 1 – Core restructure** (`wip/core-restructure`, alone): Llm/Tokenizers/Templating in Core, task layer (text
generation, speech recognition, speech synthesis, image generation; open for new kinds), audio/image I/O, versioned JSON
"model kind + components" description in `.slm`, DI registration (`services.AddSharplight()`), docs/packaging/CI.

**Step 2 – Public API pass** (alone, right after step 1): Microsoft design guidelines across the public API; every static
registry (`XxxImporter.Register()`, provider registration) becomes `IServiceCollection` extensions plus options; task
utilities shaped like Microsoft.Extensions.AI (IChatClient, ISpeechToTextClient, IImageGenerator, IEmbeddingGenerator);
samples, Server, Studio and docs follow. Rules in docs/code-style.md.

**Step 3 – Cleanup / dead-code removal / refactoring pass** (scheduled by the user 2026-11-15): after the public API
pass and the two-stage precompilation refactor are merged, run parallel cleanup agents per project area (Sonnet,
docs/cleanup-brief.md, non-overlapping project lists, behavior- and performance-preserving): dead code, duplication,
single-line scopes, one type per file, extension blocks, visibility, XML docs, simplification of complex code, shared
patterns across model kinds. Feature agents merge develop right after it lands. Repeat a smaller cleanup round after
every wave (and whenever a merge round leaves warnings or style debt).

**Wave 1** (in parallel after step 2; ONNX ops, weight formats and performance may start during step 2):
- ONNX ops (`wip/onnx-ops`): GroupQueryAttention, RotaryEmbedding, (Skip)SimplifiedLayerNormalization, NonMaxSuppression,
  RoiAlign, Loop, STFT (generic Core DSP op), runtime Range, bertsquad CPU i32 Scale; vs onnxruntime.
- Weight formats and containers: Marlin GPTQ/AWQ, AWQ gemv_fast, AutoRound, HQQ, EETQ, quanto, torchao, Quark int/MXFP6,
  EXL2/EXL3, AQLM, VPTQ, HIGGS, SpQR, FP-Quant, MLX FP4 + other bit widths, bitsandbytes FP4/int8 kept quantized, GGUF I2_S,
  safe restricted PyTorch pickle loader (.bin/.pt/.ckpt); bit-exact vs the reference libraries.
- Speech-to-text (`wip/whisper`): Whisper (HF + whisper.cpp ggml .bin) vs transformers and whisper.cpp on jfk.wav; then
  Distil-Whisper, Moonshine, Parakeet (CTC/TDT), Canary, Voxtral; long-audio chunking with VAD.
- Image generation (`wip/diffusers`): diffusers + single-file LDM importer, UNet/VAE/CLIP/T5 graphs, schedulers, CFG;
  SD1.5 vs diffusers and stable-diffusion.cpp, then SD2.x, SDXL, SD3/3.5, FLUX.1, Chroma, Qwen-Image; img2img, inpainting,
  LoRA, ControlNet, TAESD (the stable-diffusion.cpp feature set).
- Text-to-speech: llama.cpp TTS archs (qwen3tts, pockettts, OuteTTS with wavtokenizer-dec), Kokoro, Orpheus + SNAC,
  CSM + Mimi, Dia.
- Performance (re-enabled 2026-11-15): new baseline on this hardware vs llama.cpp CUDA/Vulkan, vLLM and onnxruntime
  (docs/benchmarks.md model set); resume `wip/vulkan-matvec-rows`, then the open gaps in docs/roadmap.md "In progress";
  later the new model classes (Whisper, SD) vs whisper.cpp / stable-diffusion.cpp / diffusers.
- Studio S0/S1 (see the Studio track below).

**Wave 2:**
- Vision: classification / image embeddings (ViT, DINOv2/v3, CLIP, SigLIP), detection (DETR, RT-DETR, YOLO), segmentation
  (SAM 2, Mask2Former), depth (Depth Anything, DPT), OCR (TrOCR, PaddleOCR, Florence-2 / GOT-OCR), upscalers (Real-ESRGAN,
  SwinIR).
- Audio: codecs (EnCodec, DAC, SNAC, Mimi), MusicGen, Stable Audio Open, CLAP, VAD (Silero), source separation (Demucs).
- Video generation: Wan 2.x, LTX-Video.
- Speech-to-speech: Moshi, Omni models with talkers (Qwen-Omni).
- LLM breadth: the remaining llama.cpp archs (168/172 + inkling) and new archs as llama.cpp adds them; mmproj follow-ups.

**Wave 3:**
- Time series: Chronos (incl. Bolt), TimesFM, Moirai.
- Retrieval: ColBERT, ColPali, SPLADE (next to the existing embeddings / rerankers).
- 3D: TripoSR, Hunyuan3D, VGGT.
- Bio: ESM-2/ESM-3, Evo, DNA models (Nucleotide Transformer, DNABERT, Caduceus).
- Robotics VLA: OpenVLA, pi0.

**Unified (user, 2026-11-15: "Everything should be unified"):** one system for all model types, no per-type silos.
Engine: one model concept (kind + components), one load path (`.slm`, importers convert into it), one task API pattern
(DI-registered clients with options, async, MEAI-shaped), one registration (`services.AddSharplight()`), one serving
surface. Studio: one library, one Discover/download flow with type filters, one convert / inspect / load / unload /
residency model, and inference pages built from shared components (model picker, parameter panel, presets, run queue,
history, outputs gallery, export) so every type looks and behaves the same; chat is one of these pages.

**Studio track** (one agent continuously from wave 1 on; docs/studio.md):
- S0: Studio on the DI API and the task layer; the library reads the `.slm` model kind and shows every model by type.
- S1: Model download overhaul for all model types:
  - Filters: model type / task (text generation, vision-language, speech-to-text, text-to-speech, image generation,
    video generation, vision tasks, audio, embeddings / retrieval, time series, 3D, bio, VLA; mapped from Hub pipeline
    tags and library names), format (GGUF, safetensors, diffusers, ONNX, whisper.cpp, `.slm`), quantization, size /
    fits this machine, license, gated, plus sort by trending / downloads / likes / recent.
  - Discover highlights per model type (Trending / New / Most downloaded per kind) with the Supported-only switch.
  - Multi-component repos (diffusers folders, encoder + decoder + vocoder sets, whisper.cpp files, ONNX) downloaded as
    one model, with per-kind compatibility badges and fit estimates; conversion to `.slm` for every kind.
- S1b: precompilation for every execution provider and device available on the machine through an explicit two-stage
  provider contract (portable code such as SPIR-V, then device-specific driver binaries; CPU: empty), written into the
  `.slm` in place (crash-safe) while the model is not resident; inspector shows the stages per provider/device.
- S2 (branch `studio/s2`: shared blocks `InferencePageViewModel` / `InferenceFrame` = picker, parameter panel with JSON presets,
  run queue, history, outputs gallery, export; task models resident on the `ModelHost` as `ModelKind.Task`; Transcribe and Speak
  pages done, see docs/studio.md *Inference pages*; later pages derive from `InferencePageViewModel` and add one `TaskKindClients`
  line): a dedicated inference page per model type, added as each type lands on main: Transcribe (file/microphone, timestamps,
  subtitles export), Speak (voices, streaming playback, WAV export), Image (txt2img, img2img, inpaint, seeds, gallery),
  Vision (drop an image: labels, boxes, masks, depth, OCR overlays), Audio (music generation, separation, codec playground),
  Video (text-to-video and image-to-video: prompt, frames / fps / resolution, seed, scheduler, live frame preview,
  playback, export to frames and MP4/WebM), Search (embeddings / retrieval over local documents), Time series
  (CSV → forecast chart), 3D viewer.
- S3: chat with voice in/out (STT + TTS) and image generation; the serving page and Sharplight.Server expose
  OpenAI-compatible endpoints for the new types (/v1/audio/transcriptions, /v1/audio/speech, /v1/images/generations,
  /v1/embeddings).
- S4: screenshot tests per page, docs, then a Linux Studio release when the user says.

**macOS track** (branch `macos/round-1`, docs/macos.md; user decisions 2026-11-15): no Mac locally, so testing runs on
GitHub's macOS / Linux-arm64 runners. Pushing only `macos/round-1` is allowed for CI runs (privacy scan first).
- M1: merge main into it (again after each big merge), keep macOS + Linux CI green, then merge it into main.
- M2: native Metal execution provider (plan in docs/macos.md), behind the same provider abstraction and device features.
- M3: NEON CPU kernel performance against llama.cpp on the Arm64 runners.
- Studio `.app` stays an unsigned CI artifact until the user asks for signing.

**Branches (user, 2026-11-15):** `main` is stable; `develop` is the unstable integration branch that every agent branch
merges into; develop merges into main once everything is green; releases come from `release/vX.Y` branches
(Nerdbank.GitVersioning).

**Release track (Studio for Linux, Windows, macOS):**
- R1: versioning and CI for the branch model: NBGV public releases from main and release/*, prerelease versions on
  develop; CI builds and tests develop, main and release/*; NuGet packages publish from release branches.
- R2: Studio app packages per OS: Linux (AppImage + tar.gz), Windows (Velopack Setup.exe, portable zip), macOS (.app in
  a Velopack .pkg / zip; signing and notarization when the user provides certificates). Self-contained, trimmed where safe.
- R3: Velopack install / update in Studio: startup hook; update checks; an update-available dialog with release notes,
  download progress and restart-to-apply; a settings section (version, channel, auto-check, check now). 10 channels:
  `<rid>-stable` (from release/*) and `<rid>-unstable` (from develop) for win-x64, win-arm64, linux-x64, linux-arm64,
  osx-arm64. CI uploads releases and delta packages to the user's Cloudflare R2 bucket (S3 API; GitHub secrets
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET); apps download from its public custom domain
  (configurable, repository variable STUDIO_UPDATE_URL).

**Per round:** privacy scan of the diff (no personal email, no home paths), merge to develop, build, all suites in parallel
across the GPUs, remove the worktree, the machine cleanup (prune worktrees, old temp files), start the next item. Keep
docs/model-support.md, docs/onnx-support.md and this file current. Push to GitHub only when the user says so; never push
a `v*` tag; `macos/round-1` stays separate.

Agent briefs: docs/bringup-brief.md (architecture rules; note it still says `master` — the branch is `main`, and the
importer projects are now `Sharplight.Import.*`), docs/cleanup-brief.md, docs/code-style.md. Tests are xunit v3
executables: `dotnet run --project tests/X -c Release --no-build -- -class "*Name"`.
