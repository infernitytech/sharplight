# Handoff: Sharplight development moves to several machines (2026-11-26)

Until 2026-11-26 one coordinator session drove all work from a single RunPod container (3 GPUs, from 2026-11-19 one GPU
and ~10 cores). The owner now runs development on several machines with their own agent platform, for all workstreams:
engine LLM performance, model bring-ups and the model feed, ONNX, Studio, Server and the website. This folder holds
what a new agent needs: goals, owner rules, the state of every branch, the full work queue, and the merge, feed and
test procedures. It replaces the 2026-11-15 handoff.

- `docs/handoff/memory/`: the Claude Code memory notes (owner rules and decisions, verbatim where they are rules;
  machine-specific parts rewritten as history). `MEMORY.md` is the index. `docs/handoff/restore.sh` installs them into
  the clone's Claude project memory directory.
- `tools/agent/`: the daily scripts (environment, verify and push, suites, model mirror, machine bootstrap), path-generic.

## 0. Start of a session

1. Clone the engine (`https://github.com/infernitytech/sharplight.git`), check out `develop`, fetch the branches of
   section 5. Commits use only the noreply identity: `git config user.name bglueck` and
   `git config user.email 826745+BernhardGlueck@users.noreply.github.com`. Agents never change it.
2. Run `docs/handoff/restore.sh` (memory notes), then read this file and `docs/handoff/memory/MEMORY.md`.
3. Set up the machine with `tools/agent/bootstrap-local.sh` (`--check` first to see what is missing) and source
   `tools/agent/env.sh` in every shell (section 11, tools/agent/README.md).
4. Continue from the work queue (section 8). Push only when the owner says so (section 3).

## 1. Repositories

| repo | visibility | previous local layout | branches |
|---|---|---|---|
| `infernitytech/sharplight` (engine, Studio, CLI, samples, docs, catalog, feed tooling) | public, MIT | working clone on local disk, `backup` remote on the persistent volume | `main` stable, `develop` integration, `release/vX.Y`, agent branches |
| `infernitytech/sharplight-server` (commercial Sharplight Server) | private, closed source | clone next to the engine repo | `main`, `develop`, feature/fix branches |
| `infernitytech/sharplight-website` (Jekyll, GitHub Pages) | public | clone next to the engine repo | `main` |

Server and website details: section 13.

## 2. Long-term goals (owner)

Notes: [core-task-utilities](memory/core-task-utilities.md), [project-goals-parity](memory/project-goals-parity.md),
[scope-audio-image-generation](memory/scope-audio-image-generation.md), [three-workstreams](memory/three-workstreams.md).

1. **All model types on one engine**: LLMs (every llama.cpp architecture), speech-to-text, text-to-speech, image and
   video generation, vision, audio (codecs, music, VAD, separation, CLAP), speech-to-speech, time series, retrieval, 3D,
   bio, robotics VLA, and all extra weight formats and containers. Never scope a model class out without asking.
2. **Layering**: Core gains only generic operations; task utilities (text generation, speech recognition, synthesis,
   image generation, ...) live in Core; importers (`Sharplight.Import.*`) and graph generation (architecture adapters)
   stay separate. Model- or vendor-specific handling stays in architectures and adapters.
3. **Everything storable in `.slm`** (multi-component models, configs, tokenizers, schedulers, task metadata).
4. **Studio** for every model type with dedicated pages; **Server** as the commercial product; **website** as the
   public face. Constant progress on engine, Studio and Server in parallel.
5. **Performance**: parity on ALL models and benchmarks with vLLM and llama.cpp, with onnxruntime for ONNX, and with
   the reference implementations for other model types (PyTorch for SeedVR2, diffusers, whisper.cpp, ...), plus some
   areas where Sharplight is clearly faster. vLLM parity is the minimum; aim for ~1.1x on the headline numbers. Long
   contexts (16K-128K+) are part of every comparison. Perf rounds get GPU priority; always keep one running.
6. **Official releases**: every bring-up is verified on the publisher's official files, added to the verified catalog
   and the model feed, and installable in Studio with one click.

## 3. Owner rules (summary; details in the memory notes)

| rule | note |
|---|---|
| Code style: one type per file, braces always, no single-line scopes, C# 14 extension blocks, explicit visibility, XML docs on public API, MS API conventions, DI instead of static registries, professional samples | [code-style-rules](memory/code-style-rules.md), docs/code-style.md |
| Format-agnostic Core: format names only in `Sharplight.Import.*`; Sharplight's own canonical names | [format-agnostic-core](memory/format-agnostic-core.md) |
| Generic optimizations: keyed on op, shape, dtype, quant format and device features, never on model names or thresholds tuned to one model; consumer APIs go through the provider abstraction | [generic-optimizations-provider-apis](memory/generic-optimizations-provider-apis.md) |
| Device features detected with fallbacks; never assume memory topology (small BAR, low VRAM, unified memory); every allocation falls back | [device-features-fallbacks](memory/device-features-fallbacks.md) |
| CUDA provider needs only the NVIDIA driver (runtime libs shipped by us); Vulkan must not lag; cuBLASLt optional and out of Studio | [cuda-provider](memory/cuda-provider.md) |
| No perf A/B environment switches; real choices become options (2026-11-25) | section 6 |
| Benchmarks like-for-like (same files, precision, procedure and statistic; `--feedback`/`--text` decode; a serving number counts only when the batched text check passed) | [benchmarking-shared-gpu](memory/benchmarking-shared-gpu.md), [marketing-perf-claims](memory/marketing-perf-claims.md) |
| Marketing claims vLLM parity and promotes only large measured leads | [marketing-perf-claims](memory/marketing-perf-claims.md) |
| Bring-ups: official release files, a catalog entry per official variant, a feed entry, one-click Studio install; community quants only as extra lower-precision options | [bringup-catalog-feed](memory/bringup-catalog-feed.md), [compatible-models-doc](memory/compatible-models-doc.md) |
| Branches: agents merge into `develop`; `develop` into `main` only when green; releases from `release/vX.Y` | [branching-and-studio-releases](memory/branching-and-studio-releases.md), [releases-and-macos](memory/releases-and-macos.md) |
| Public repo privacy: noreply identity only, no home paths, no personal emails or names, privacy scan before every push | [public-repo-privacy](memory/public-repo-privacy.md) |
| Push only with the owner's go-ahead; `develop` may be pushed when green after the scan; `main`, `release/*`, `v*` tags and the server repo need explicit approval | [branching-and-studio-releases](memory/branching-and-studio-releases.md), [speed-policy](memory/speed-policy.md) |
| Speed: merge each green branch right away; verify with build + style + affected suites; agents run their full affected suites before hand-back | [speed-policy](memory/speed-policy.md) |
| Keep the machine responsive: niced builds, build slots, agent cap sized to the machine, never `pkill -f`, no recursive searches over `/` | [keep-machine-responsive](memory/keep-machine-responsive.md) |
| Git worktrees for parallel work; logs and temp in a work directory, never /tmp; remove merged worktrees | [git-worktrees-parallel](memory/git-worktrees-parallel.md), [work-dirs-cleanup](memory/work-dirs-cleanup.md) |
| No inference or heavy work on the Studio UI thread | [ui-thread-async](memory/ui-thread-async.md) |
| Storage modes portable / system / custom for every file Studio writes | [storage-modes](memory/storage-modes.md) |
| Experts on disk (io_uring on Linux, pread fallback; DirectStorage later) | [experts-on-disk](memory/experts-on-disk.md) |
| Native audio library built by us (miniaudio + codecs, permissive licences only) | [native-audio-library](memory/native-audio-library.md) |
| macOS work stays on `macos/*` branches until the owner decides | [releases-and-macos](memory/releases-and-macos.md) |
| Tokens and keys (HF, GitHub runner, R2, signing) come from the owner; never create, fetch, print or commit them | [feed-runner](memory/feed-runner.md), [machine-runpod-3gpu](memory/machine-runpod-3gpu.md) |

## 4. State of `develop`, `main` and releases (2026-11-26)

- **Releases**: `v0.3.3` is published (NuGet packages and Studio). Tags: v0.1.22-alpha, v0.2.0, v0.3.0, v0.3.1, v0.3.3.
  Release branch `release/v0.3` on origin; `version.json` on develop is `0.4-alpha`. Process: docs/releasing.md
  (release branch -> NuGet + Studio via Velopack to R2, then merge into `main`; a `v*` tag push needs the owner's go-ahead).
- **`main`** = `43bd18d3` "Merge release/v0.3 into main (Sharplight 0.3.3)". Its two release fixes (Hub PyTorch
  checkpoints as a format, one progress bar per Studio download) are on develop as cherry-picks.
- **`develop`** contains the video-streaming merge `9d494bcf`
  (benchmarks §115: video enhancement streams frames through a bounded pipeline with memory independent of clip
  length, SeedVR2 VAE tiling for 1080p on 32 GB, Studio Enhance writes output incrementally and keeps partial results on
  cancel) and its section-number commit, pushed 2026-11-26 (verify: Core, Hosting, Studio green; Video's BundledFfmpegTests
  and one ControlNet test failed only for local environment reasons: no locally built FFmpeg, disk quota). It also has §114
  (perf/llm-r48: Llama-3.1-8B 16-stream ITL p99 84 -> 35 ms, vLLM 33.5) and the feed listing of the Nemotron 3.5 Bolt,
  SDXL ControlNet and SDXL TAESDXL packages.
- Suites were green for the affected suites at each merge. Known failures on develop: CosyVoiceTests.Streams_chunks_(gpu:
  False) rel RMS 0.432 (pre-existing), Studio PackageInspectorTests.EverySectionKindOfTheFormat_HasAView (WLAY section
  has no view). Flaky tests: section 8.8.

## 5. Engine branches

| branch | head | state | what is left |
|---|---|---|---|
| `develop` | e3e8aed1 | video streaming merged (§115) | push (owner) |
| `fix/feed-official` | a76bde2a (4 commits) | Official-only feed audit, round 2 in progress. Done: catalog uses official repos instead of mirrors for Llama 3.x, Gemma 3 1B, Orpheus, SD3.5 Medium, FLUX.2 Klein 9B and Krea 2 Turbo; a generic GitHub source type (Silero VAD from its official GitHub release); the SD 1.5 preservation copy recorded (owner accepted it); official Gemma 4 31B BF16 and QAT W4A16 variants. `tools/model-feed/PENDING.md` holds the proposed publish request (Gemma 3 1B BF16, Silero 1.0.1, Whisper tiny official F32 + q5_1 rebuild, Gemma 4 31B QAT W4A16 + BF16) and the unlist/removal commands | push as-is; finish round 2 (Whisper tiny official, Gemma 4 31B official variants, Silero from GitHub), merge, then publish with a separate `publish-request.json` commit (section 10); the owner runs the unlist/removal commands after promotion |
| `perf/llm-r49` | 5fa4c49a (4 commits) | Vulkan fused q / k / v as one tall multi-output FP8 GEMM launch; VK_NV_cooperative_matrix2 measured for prefill (FP8 / bf16 rejected); the dual GEMM's E4M3 store blocked; scripts in bench/r49, raw results in bench/results/r49 | push as-is; ready to merge as **§116** after the suites (Core.Tests, Vulkan.Tests, Serving.Tests); rename the `R49` headings and references |
| `perf/llm-r50` | 2ab74d15 (4 commits) | CUDA launch templates, shared variant conditions evaluated once per shape, prepared decode shapes captured ahead; Gemma 4 31B host time per step already at parity with vLLM 0.33 (0.07-0.22 ms, GPU 99.7 % busy); recording a new step shape 1.32 -> 0.81 ms | push as-is; serving reruns; follow-up: run the ahead capture in the background after load (it adds 0.9-2.7 s to start-up); merge as the section after §116 |
| `perf/llm-r46` | eec3ea0e (1 commit, 107 behind develop) | one leftover §109 docs commit (Qwen3-8B-FP8 perplexities with the FP8 KV cache, the queue / preempt rerun, pending items) | push as-is; cherry-pick the commit onto develop |
| `fix/video-streaming-io` | b05da545 | merged into develop through `fix/video-streaming` | delete after the develop push |
| `fix/video-streaming` | 182fac5c | merged (§115) | delete after the develop push |
| `macos/round-1`, `macos/round-2`, `macos/metal-r1..r4`, `macos/ffmpeg-1` | origin | macOS / Metal work | separate until the owner decides; merge develop into them now and then, keep their CI green |
| `release/v0.3` | origin | v0.3.x | patch releases only |

**r50 finding, now the top perf item**: the official Gemma 4 31B QAT w4a16 (compressed-tensors) against vLLM 0.33 on
CUDA is 33.6 vs 36.2 tok/s at 1 stream (0.93x) and **237 vs 304 tok/s at 16 streams (0.78x)**; the decode step takes
49.5 ms of GPU time at 16 rows. The int4 W4A16 GEMM at 16 rows is the lever (section 8.1).

## 6. Pending cleanup: perf A/B environment switches

Owner decision 2026-11-25: remove every perf A/B environment switch and the old code path behind it in one sweep; a
real choice becomes an option.

- Remove the `SHARPLIGHT_KV_ADMISSION` override from `src/Sharplight.Core/Llm/DiagnosticSwitches.cs` and its mentions in
  the `src/Sharplight.Core/Llm/Serving/EngineOptions.cs` docs, docs/hosting.md and docs/kv-cache.md. `KvAdmission` is a
  real `EngineOptions` option (Studio has a KV admission choice).
- Remove any `SHARPLIGHT_NATIVE_*_F8` switches (`src/Sharplight.Shaders/DiagnosticSwitches.cs`,
  `src/Sharplight.Shaders/Lowering/GraphLowerer.NativeAttention.cs`).
- Earlier list (cleanup-r1): PDL_SHARED, PDL_TRIGGER, CPU_POOL/AFFINITY, EXPERT_CACHE=lru, STREAM_*, SHARED_ARENA,
  VK_TRANSFER_QUEUE, CPU conv-plan; leftovers: CUDA_PDL_SHARED_FLOOR, STREAM_K/STREAM_K_COST, SPLIT_EXPERTS,
  SPARSE_EXPERT_ROWS, GPU conv shader switches. Check what still exists in the `DiagnosticSwitches.cs` files of Core,
  Shaders and Cuda.
- Diagnostics that do not change results (profiling, launch lists, validation) stay.

## 7. Waiting on the owner

- Push of `develop` and of the branches in section 5; then `develop` -> `main` and the next release.
- Register the self-hosted feed runner on one of the new machines (org registration token; section 10).
- Feed: models of 250-566 GB (MiniMax-M3.1-Flash, MiMo-V2.6-Pro-RL, DeepSeek-V4.1-Flash) are catalogued but not in the
  feed; they need a runner with a bigger disk, or a decision.
- Feed: promotion of unstable packages to stable; whether to re-precompile packages built before the block-scaled FP8
  default (kernel cache for FP8 on cc 12.x).
- Redistribution licence check for gated official sources (SD3.5, FLUX.1-dev, Llama, Gemma) before they enter the feed.
- gpt-oss ITL p99 at 16 streams (47 vs 13 ms) needs a StepFloor=Full call.
- FP4: offer W4A4 as a choice (PPL +4.3 % vs +2.5 % weights-only) through a load option, CLI and Studio toggle?
- Server: OIDC group membership refreshes only at sign-in (sessions up to 7 days), so removed group members keep space
  access up to a week: directory sync or a shorter refresh.
- Server: the 12 questions of the RAG research (section 8.7).
- Server pushes need explicit approval (a one-off approval exists for the engine-API update).
- macOS branches: when to merge.

## 8. Work queue

De-duplicated from the coordinator's queue (2026-11-18..26). Done items are kept short for context. Performance order
(owner end goal 2026-11-25): the behind-list first.

### 8.1 Engine performance (perf/llm-*, GPU priority, one round always running)

- **Top: int4 W4A16 GEMM at 16 rows** (official Gemma 4 31B QAT w4a16 at 0.78x vLLM at 16 streams). Combine with the
  CUDA weight-only follow-ups of fix/cuda-weight-only-quant: W8A16 GEMM (weight-only FP8 prefill is 0.53x of the W8A8
  path), W4A16 (0.77x), honour W8A8 static input scales (Super-VL FP8 activations still off in its CUDA test: 0.193 vs
  tolerance 0.1). In flight at the handoff: CUDA native NVFP4/FP8 paths quantized activations for weight-only
  checkpoints (fix).
- Merge perf/llm-r49 (§116) and perf/llm-r50; r50 follow-up: ahead graph capture in the background after load.
- Vulkan prefill vs vLLM (0.70-0.90x).
- ITL p99 tail at 16 streams: Qwen3-8B-FP8 24 vs 16 ms (§113); gpt-oss 47 vs 13 ms (owner call on StepFloor=Full).
- Coder-Next first-run time.
- TensorRT / onnxruntime gaps (§99); refresh the ONNX parity table against onnxruntime CUDA / TensorRT / CPU on current
  develop.
- VAE mid-block attention at head size 512 (SD-family VAEs, ~413 ms CUDA / 683 ms Vulkan at 1024², 1-2 TFLOPS): flash
  kernel for large head dims.
- Small kernel items: FP8 copy of the LM head (2.1-2.4 ms/step at 256 rows), empty flash blocks in mixed steps (~2.5 % of
  a 2,048-row step), Llama bf16 192-row partial token tile slower than 256.
- FP4 follow-ups: FP4 decode GEMM bandwidth 650-730 vs ~830 GB/s, fuse FP4 activation quantization into the norm, MXFP4
  dense native, static activation scales from checkpoints, sm_100 / Metal native.
- The CUDA translator skips `griddepcontrol.wait` for every kernel in a source that mentions it: fix generically (latent
  correctness trap).
- Conv / VAE: producers (group norm + SiLU) write channels-last f16 directly; ESRGAN-like many-in / few-out convs (f32
  input staging dominates); SeedVR2 DiT and VAE do not fit together at >= 720p (recompile 1-2 s).
- MoE offload: gpt-oss-120b CPU-offload prefill is 0.47x llama.cpp at the same statistic (§107): MXFP4 expert prefill on
  CPU offload. Experts on disk: prompt passes > 4 rows run experts on CPU with a cold cache (DeepSeek-V4-Flash first
  prompt 6 s vs 2 s for llama.cpp); next-layer expert prediction, VRAM cache tier, Vulkan cache-slot import, io_uring on a
  host that allows it. DeepSeek-V4 compressed layers: give compressed entries their own block table to restore the
  window-pool memory savings (after fix a885d924).
- Multi-node (§88 follow-ups): decode micro-batches in one serving engine, lower per-step overhead, optional f16
  boundaries, RDMA transport, metadata-only coordinator.
- Saturation follow-ups (r45): KV capacity and scheduling, speculative decoding (both engines).
- Sizing: the activation bound is too pessimistic for single-sequence sessions (Gemma 3.1 GiB estimated vs ~0.3 real);
  FeedTool should record the session footprint (window caches); a pre-load disk speed probe for the experts-on-disk estimate.
- [done] r12-r48: MoE MXFP4 GEMM, FP8 GEMM main loop, FA2-class prefill attention, serving rounds, K-quant GEMMs (feed
  rounds 1-5), stream-K, saturation + roofline (r45), FP8 KV cache with queue / preempt KV-full policies (r46),
  block-scaled FP8 on by default with a setting, prompt rows per slot 48, native FP4 (§106), Vulkan prefill analysis (r47),
  f16 / bf16 products from 65 rows (r48), f16-activation convs (§110), bf16 fallback for f16 overflow (§112).

### 8.2 Engine features, API and cleanup

- The switch cleanup of section 6.
- Recommended configurations in `.slm` (owner 2026-11-25, after v0.3): ship recommended load and execution configs with
  our own feed packages, used by Server and Studio as presets. A new `.slm` section; entries keyed by CSS-style selectors
  over hardware and environment (cluster / node count, GPU vendor, GPU model including edition such as RTX PRO 6000
  Blackwell Max-Q, GPU count, VRAM, driver version exact or range, provider CUDA / Vulkan / Metal / CPU); more specific
  selectors override less specific ones (cascade and specificity as in CSS). Covers all load and execution settings
  except per-session ones (sampling, seeds, per-request options). Design: selector grammar and specificity, matching
  against detected device features, merge semantics, Studio / Server UI showing which preset applied and why, tooling to
  author and benchmark presets for the feed. Format-agnostic, generic matching.
- Larger cleanup merges deferred from cleanup-r1: a shared provider segment / device base in Core, the two ggml
  dequantizers, the two LLM samplers, HF import helper copies, the L2-normalize variants.
- RAG prerequisite: partition the KV prefix cache by user / ACL hash for prompts with retrieved text (timing side channel
  inside a space).
- CUDA device reserve is too small for ~185-segment placements (Kimi K3 needed `SHARPLIGHT_DEVICE_RESERVE_MIB=3072`).
- [done] async text embed / rerank task clients, 64-bit seeds, audio API aligned with image / video, CopyStateRows on
  CUDA / Metal, the memory-topology audit, Sharplight.Cuda depends on Sharplight.Cuda.Runtime, experts on disk,
  multi-node round 1, streaming video enhancement.

### 8.3 Bring-ups (official files, catalog, feed, Studio)

- Inkling 2 Small (`inkling2` mapping: no loop, 384 experts, per-expert output scales; about 1 day) and the larger
  Inkling 2 release (transformers 5.24 `inkling2`, new ModelOpt layout, about 3-4 days). Inkling 2 Mini is done.
- New gaps from the section 23 sweep in docs/model-support.md.
- Super-VL: a 0.05 logprob gap, identical on all devices (suspected expert-selection near-tie; unchecked).
- Huge MoE: Kimi K3 narrow IQ1 kernels and a real (not truncated) run; DeepSeek-V3 GGUF `expert_group_count`; GLM-5.3 and
  Nemotron-3-Ultra runs; catalog entries for the huge models; the DeepSeek-V4-Flash CUDA multi-row bug with experts on
  disk was not reproduced on synthetic models.
- ChatTTS (DVAE decoder + 4 code heads, CC-BY-NC) not started; Stable Audio 3 post-trained repos and Cohere Transcribe
  Arabic need access (403 at the time).
- SeedVR2: no GGUF; the ComfyUI fp8 is a plain cast (29 dB).
- [done] OmniVoice, VoxCPM2, Stable Audio Open 1.0 and 3, Cohere Transcribe, SeedVR2 3B / 7B official (§108.5), Inkling 2
  Mini, bring-up round 2.

### 8.4 Model feed and catalog

- Official-only audit (`fix/feed-official`, section 5): every published package needs an official-source variant;
  mirror-built packages are replaced by official repos (adamo1139 SD3.5 Medium ungated, unsloth Llama-3.1-8B-Instruct,
  Llama-3.2-1B/3B-Instruct, gemma-3-1b-it, Krea-2-Turbo, FLUX.2-klein-9B, orpheus-3b-0.1-ft; the Silero mirror) and then
  unlisted. FLUX.1 from city96 GGUF stays only as community extras next to an official variant. The FLUX.1-schnell +
  TAEF1 entry was dropped on 2026-11-26 (runner download refused).
- SD3.5 Medium + TAESD3 (built from the mirror by run 45768889634) is uploaded to R2 but never listed: delete those 1.0.0
  objects only after the official rebuild is published (PENDING.md explains the versions).
- SeedVR2 3B / 7B / 7B-sharp publish started on 2026-11-26 (develop 8542acd1): check the run and the Studio feed listing.
- Catalog licences: record licences for ~30 entries without one (Phi, MiniCPM, CLIP, ...); review the licenseNotice texts
  for Falcon, DeepSeek, NVIDIA, DINOv3, OpenMDW, Krea 2 and Qwen research against the full licence texts; make sure catalog
  resolve does not revert the smollm2 licence.
- Republish the ms-marco reranker with its kind; republish MoE packages with expert figures where still missing.
- Guard: a merge from a branch based on an older develop once reverted `publish-request.json`; add a CI check that the
  request changes only in dedicated commits.
- [done] feed variants and ladders (36+ packages for 11+ models in unstable), the Studio feed install source, the large
  GGUF and safetensors variants on the self-hosted runner.

### 8.5 Studio

- Model kinds missing from `ModelKindCatalog` (non-runnable in filters and Tasks cards): music generation, audio
  generation, audio embedding (audio separation is there now). One line each.
- Controls for MinStepTokens / StepFloor / BlockScaledFp8 / rows per slot where they are not exposed yet; the FP4 W4A4
  toggle once the owner decides.
- PackageInspectorTests.EverySectionKindOfTheFormat_HasAView fails (WLAY section has no view).
- Studio real-model test classes grow to ~36 GB RAM: make them opt-in.
- Owner order from 2026-11-24: cleanup -> one consistent public API across model types -> Studio with all new features
  -> automated end-to-end UI tests. The first three shipped with v0.3; keep the end-to-end UI tests growing.
- Welcome page memory watch; KV cache types per provider (Metal).
- [done] feed as install source, licence-accept dialog, first-run sizing rounds 1 and 2, KV cache and KV admission
  choices, EmbeddedServer test on port 0, Enhance page with streaming output.

### 8.6 Website

- "Where Sharplight is behind": Llama-3.1-8B 16-stream ITL p99 now 35 vs vLLM 33.5 (§114); Qwen3-8B-FP8 p99 24 vs 16
  (§113); gpt-oss p99 still 47 vs 13.
- Switch `blob/develop/docs/benchmarks.md` links back to `main` (v0.3 is on main).
- The MoE offload chart returns only after re-measuring with the identical statistic on both sides.
- [done] SeedVR2 7B rows replaced with the official 7B parity (§108.5); no 7B speed ratio claimed.

### 8.7 Server (private repo)

- Merge `engine/api-r2` (already on the server's origin `main`, see section 13) into `develop` as well.
- Expose BlockScaledFp8 (NodeHostingOptions -> HostedModelOptions), StepFloor (ConfigureChat) and rows per slot.
- Server.Tests ClusterProcesses harness still syncs the live feed (make it offline; the soak test counts syncs); add
  `-diagnostics` to the CI scale-smoke run for the seed step tables.
- Server tests need `SHARPLIGHT_SERVER_TEST_POSTGRES` (20 fail without it; they should skip).
- Keycloak / OpenLDAP / KDC fixtures so the OIDC / SAML redirect and identity tests run (skipped now).
- First CI run of postgres-bundle: ubuntu:22.04 build and apt path, container matrix (22.04 / 24.04 / debian12 /
  fedora41 / arch), Windows DLL check; Rocky / RHEL 9 unsupported (glibc 2.34).
- GitHub Packages restore failed for agents (check the token's read:packages scope); builds used
  `-p:SharplightSource=local`.
- Licensing (S-30): the editions table is approved (Community / Evaluation / Standard / Enterprise in config/editions.json,
  adjustable); audit viewer, per-account retention, export (no SIEM export in v1).
- Multi-node in the server: workers per shard node, ordered worker list, mTLS from the cluster CA, the package on every
  shard node, replica = failure domain.
- RAG (phase 2): research done on 2026-11-25 with 12 questions for the owner in its section 8; start after the answers.
- [done] M0-M7, server rounds r1-r14, model variants UI, experts on disk, scale-smoke fixes, rebalance after probe.

### 8.8 Flaky tests to fix deterministically

- PrefixCacheTests.ConcurrentPrompts_WaitForBlocksAnotherPromptComputes_InsteadOfComputingThemTwice (expected 7, got 5).
- WindowKvCacheTests.ConcurrentPrompts_AttachAPrefixAnotherPromptComputes.
- Serving EngineTests.EqualPriority_IsFifo (timing).
- MultimodalEngineTests.OneImageInSeveralPromptsIsEncodedOnceAndCached.
- Cuda.Tests ManySegmentPlacementTests.SegmentsThatDoNotFitTheirGraphsRunWithout measures device-wide free memory, so
  another process allocating mid-test fails it; measure only this process.

## 9. Merge flow

1. The agent hands back: branch, commits, verified numbers, suites run with pass/fail counts. Check the authors
   (`git log develop..<branch> --format='%an %ae %cn %ce'`: noreply only) and grep the Core / Cpu / Shaders / Cuda diff for
   model or vendor names (each new op needs a generic justification).
2. Privacy scan of the diff: no home-directory paths (root's or any user's), no personal emails, not the owner's private name; no
   Windows-invalid file names in `git ls-files` (`: * ? " < > |` and backslash). `tools/agent/verify-push.sh` runs both
   (put extra private words into `SL_PRIVACY_EXTRA` in the local environment, never in a repo).
3. `git merge --no-ff <branch>` into `develop` in the working clone, with a message that says what landed, the headline
   numbers and the benchmarks section.
4. **Benchmarks section numbers**: branches write their docs/benchmarks.md section under a placeholder (`## §R50.`,
   `R49.1`, `RVID.3`, references `§R49`). At merge the section gets the next free number (last on develop: **115**, so
   r49 becomes 116); rename the heading, the subsections and every reference in docs and the merge message, and commit
   it as "Benchmarks 116: section number" / "subsection numbers".
5. `tools/agent/verify-push.sh <tag> <suites>` with the affected suites. Suite names are test projects without the
   `Sharplight.` prefix: `Core.Tests`, `Cpu.Tests`, `Cuda.Tests`, `Vulkan.Tests`, `Serving.Tests`, `Families.Tests`,
   `Import.Onnx.Tests`, `Import.Onnx.Conformance`, `Studio.Tests`, `ImageGeneration.Tests`, `Video.Tests`, `Speech.Tests`,
   `TextToSpeech.Tests`, `AudioModels.Tests` (check `ls tests/`). It pushes only when YAML, style, build and suites pass
   and the scans are clean (push permission: section 3). Afterwards check `gh run list --branch develop`.
6. Remove the worktree and the merged branch; refill the agent slot with the stream that has nothing running.
7. Model-feed requests (`tools/model-feed/publish-request.json`) always go in a **separate commit** after the code
   merge, never inside a merged branch.

## 10. Feed publishing

Details: docs/model-feed.md, `.github/workflows/model-feed.yml` and `model-feed-publish.yml`.

- Trigger: a push to `develop` that changes `tools/model-feed/publish-request.json` builds the listed entries (`entry` or
  `entry:variant`) into the unstable channel and promotes the listed versions; a manual dispatch builds given entries; a
  weekly schedule only re-signs the indexes.
- Packages are immutable (`models/packages/<id>/<version>/` in the R2 bucket, served from
  https://download.sharplight.dev/models); the `unstable` and `stable` channels have their own signed indexes.
- Signing: secret `MODEL_FEED_SIGNING_KEY` in the `studio-release` environment (ECDSA P-384; the secret is the only copy);
  hosts trust `tools/model-feed/keys/sharplight-feed.pub.pem`.
- Sources: official publisher repos only, gated ones included: the **`HF_TOKEN` repository secret now exists** for the
  workflow. GitHub-only publishers use the GitHub source type. Community quants only as extra lower-precision variants.
- Runners: entries above `MODEL_FEED_LARGE_BYTES` (6 GB default) or listed in `MODEL_FEED_LARGE_ENTRIES` run on the
  self-hosted runner labelled `sharplight-feed` when `MODEL_FEED_SELF_HOSTED` is `'true'`; otherwise they are skipped with
  a warning. **The previous runner lived on the RunPod pod and is gone: register a new org-level runner on one of the new
  machines** with a big disk (about 130 GB free for the Gemma 4 31B BF16 build, 600+ GB for the largest catalogued
  models) and set `MODEL_FEED_WORK_DIR`. The owner provides the registration token ([feed-runner](memory/feed-runner.md)).
- One run at a time (the workflow's concurrency keeps one pending run). After a run, check that the versions are listed
  and that Studio shows them.

## 11. Test environment setup

- Toolchain: .NET SDK 11.0.100; Vulkan loader + NVIDIA driver (in containers also `apt-get install libegl1 libgl1`);
  Python 3.12 with uv venvs (`ref`: CPU torch + transformers; `vllm` with CUDA for comparisons; optional per-model venvs);
  llama.cpp reference binaries under `$LLAMA_ROOT` (bench build b11177, Vulkan release b12483); ffmpeg; Playwright 1.64.1
  Chromium for the Studio and Server UI tests; PostgreSQL 17 + pgvector + PgBouncer for the server tests (server repo
  `tools/dev-postgres.sh`, port 55432, not as root). `tools/agent/bootstrap-local.sh` installs all of it (pins at the top).
- Models: tests read `~/models` (`hf/tiny-*`, `gguf/tiny-*`, `audio/`, `whisper.cpp/`, `sd/`, `onnx/`, ...). The tiny test
  models regenerate with `tools/model-assets` and `tools/mk_tiny_*.py`; real models download from the Hub at the
  catalog's pinned revisions (`bootstrap-local.sh --models`, `--models-big`). Run tests with `HOME=$SL_TEST_HOME`
  (`tools/agent/sl-test`) so they read the local mirror.
- Tests are xunit v3 executables: `dotnet run --project tests/Sharplight.X.Tests -c Release --no-build -- -class "*Name"`.
  Big GPU suites run one class per process (`tools/agent/run-suites.sh`).
- GPU pinning: `SHARPLIGHT_VK_DEVICES=<i>`, `GGML_VK_VISIBLE_DEVICES=<i>`, `CUDA_VISIBLE_DEVICES=<i>`. Processes that open a
  GPU context run inside the verify or bench lock. Real memory pressure: `grep -E "^anon " /sys/fs/cgroup/memory.stat`.
- GitHub CI runs the CPU suites on develop, main and release/*; locally the GPU suites matter most.

## 12. Agent rules (put into every brief)

- Work in a worktree of the working clone: `git -C $SL_REPO worktree add $SL_WORKTREES/<name> -b <branch> develop`;
  commit often in logical chunks; `git merge develop` before hand-back; never push, never tag; never set user.email or
  GIT_AUTHOR_EMAIL; end commit messages with the Co-Authored-By trailer of the session.
- Source `tools/agent/env.sh`. Builds and tests niced (`nice -n 15 ionice -c3`, `-m:4`); build only touched projects and
  run only affected test classes while iterating; run the affected suites fully once before hand-back (that run is the
  real gate; the merge check is lighter).
- Never search recursively from `/`, the home directory or a data volume; never kill processes by pattern (only PIDs you
  started; wait with `wait $pid`); logs and temp in `$SL_WORK`, never /tmp; keep local disk under 90 %.
- GPU: only the index in the brief; GPU contexts only inside the verify / bench lock; builds inside a lock with
  `-nodeReuse:false -p:UseSharedCompilation=false` and `9>&-`. Perf rounds have GPU priority; other rounds take the lock
  only for short checks and never hold it while waiting.
- No personal paths or emails in committed files (docs use `~/models`, `$LLAMA_ROOT`). Code per docs/code-style.md;
  format names only in `Sharplight.Import.*`; only generic ops in Core; everything storable in `.slm`; no inference on the
  Studio UI thread. Architecture rules: docs/bringup-brief.md (where it says `master`, the branch is `develop`).
- Bring-ups: official files, parity with the reference implementation, tests, a catalog entry
  (`catalog/verified-models.json`, pinned revision, docs/verified-catalog.md), a feed entry, docs/model-support.md and the
  generated docs/compatible-models.md.
- CI changes: reproduce the CI step locally (clean checkout, driver-less where CI has no GPU) before hand-back.
- Perf work: key changes on op, shape, dtype, format and device features; say which model classes benefit and measure
  one extra model; serving numbers count only with the batched text check passing.
- Server repo agents commit on their branch and report it; the coordinator merges and pushes (with approval).
- Hand-back report: branch, commits, what is verified (numbers), what is not, suites run with pass/fail counts.

## 13. Server and website repositories

### Sharplight Server (`infernitytech/sharplight-server`, private)

Commercial on-prem inference platform: Blazor Server modular monolith, PostgreSQL for everything, clustering with
capacity-based placement, the model feed, enterprise auth, OpenAI / Anthropic endpoints isolated per account and space.
It consumes the engine only through NuGet packages (develop CI can build against the engine's develop source). Same code
style as the engine. Never copy its code into the public repo. Owner decisions:
[sharplight-server-product](memory/sharplight-server-product.md).

- `develop` = `50fab31` (CI green: scale smoke offline, feed sync timeout and retry, placement room move).
- `engine/api-r2` = `ed1f9c8`, 2 commits on top of develop: `66745e0` engine task API round 2 (async retrieval anchors,
  64-bit seeds, `EmbedOptions` -> `TextEmbeddingOptions`) and `ed1f9c8` engine 0.3.1 from nuget.org (pins
  `SharplightPackageVersion` to 0.3.1, maps the Sharplight packages to nuget.org as well as the prerelease feed, central
  version for Sharplight.Hosting). On 2026-11-26 the server's origin `main` also points at `ed1f9c8`, so these 2 commits
  are on `main` but not yet merged into `develop`: merge them into `develop` (and bump to 0.3.3 if wanted).
- `feature/m3-model-feed` (origin `bca6ab1`, feed channels stable / unstable, per-channel rollback state, channel
  switching in the console) is contained in develop; delete it after checking.
- CI (`.github/workflows/`): `ci.yml` on pushes to develop, main and release/** and on pull requests (style checkers,
  build, tests, scale smoke, packaging; server packages published to R2 under `server/` per channel through the
  `server-release` environment; release/* always builds against the pinned engine NuGet package); `scale-nightly.yml`,
  `soak.yml`, `upgrade-published.yml`. Tests run against the R2-published artifacts, the builds users get.
- Server tests need `SHARPLIGHT_SERVER_TEST_POSTGRES` and the dev PostgreSQL (section 11).
- Open items: section 8.7.

### Website (`infernitytech/sharplight-website`, public)

Jekyll site on GitHub Pages; `main` = `f121a63` (side branches `benchmarks-overhaul`, `content/concise`). Promotes the
open-source engine (NuGet docs for developers), Studio downloads with platform detection and, calmly, Sharplight Server,
hardware bundles and consulting. The owner connects the sharplight.dev domain at go-live; until then URLs use the
github.io address (one config change). Decisions: [sharplight-website](memory/sharplight-website.md).

- `benchmarks.md` and **`benchmarks-technical.md` must stay apples to apples**: the same model files and precision, the
  same procedure and statistic for every engine, no notes about shared machines. Numbers that turn out not to be
  like-for-like (community conversions, different statistics) are withdrawn, not footnoted.
- Open items: section 8.6.
