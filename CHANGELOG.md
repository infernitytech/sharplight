# Changelog

Release notes per version are in [docs/release-notes](docs/release-notes). Prereleases (`-alpha`) are built from `develop` and
have no curated notes; their changes are the commit log.

## Unreleased

Task API made consistent across model kinds (breaking for callers of the renamed members):

- Every task client has `services.Add<Client>(...)` registrations; new: `AddAudioCodec`, `AddVoiceActivityDetector`, `AddAudioEmbedder`,
  `AddAudioSeparator`, `AddAudioGenerator`, `AddMusicGenerator`, and `AddSemanticSegmenter((o, services) => ...)`.
- `IImageGenerator.Defaults`, `IVideoGenerator.Defaults`, `IVideoEnhancer.Defaults` replace `GetService(typeof(...Defaults))`.
- Per-request vision and audio options are immutable records like the other kinds (`with` instead of setting properties).
- Renamed: `UpscaleOptions` → `ImageUpscalingOptions`, `EmbedOptions` → `TextEmbeddingOptions`,
  `IVoiceActivityDetector.ProbabilitiesAsync` → `ComputeProbabilitiesAsync`.
- `IImageUpscaler.UpscaleAsync` takes progress as an `IProgress<double>` parameter (was `UpscaleOptions.Progress`).
- `TaskClientProvider.Create(Type, TaskModel)`, `TaskKindClients.ClientTypes`, `DeviceOptions.ResolveSubmissionBudgetMs` /
  `ResolveConcurrentWorkgroups` added.
- Removed performance A/B environment switches and the non-default code paths behind them (the defaults stay):
  `SHARPLIGHT_CUDA_PDL_SHARED`, `SHARPLIGHT_CUDA_PDL_TRIGGER`, `SHARPLIGHT_CPU_POOL`, `SHARPLIGHT_CPU_AFFINITY`,
  `SHARPLIGHT_CPU_CONV_PLAN`, `SHARPLIGHT_CPU_NO_SHIFTED_CONV`, `SHARPLIGHT_CPU_CL_CHUNKS`, `SHARPLIGHT_EXPERT_CACHE`,
  `SHARPLIGHT_STREAM_SELECTED_EXPERTS`, `SHARPLIGHT_STREAM_EXPERTS_MIN_ROWS`, `SHARPLIGHT_STREAM_BUFFERS`, `SHARPLIGHT_STREAM_RESIDENT`,
  `SHARPLIGHT_STREAM_REGISTER`, `SHARPLIGHT_SHARED_ARENA` (and `SharedArena<T>.Enabled`), `SHARPLIGHT_VK_TRANSFER_QUEUE`.
  `PlacementOptions.StreamExpertsMinRows` / `StreamBuffers` remain as options.

Task API round 2 (breaking):

- Text embeddings and reranking are task clients (kinds `text-embedding`, `text-reranking`): `ITextEmbedder.EmbedAsync` /
  `EmbedTokensAsync` (plus `Tokenize`) and `ITextReranker.RerankAsync` replace the synchronous `Embed` / `EmbedTokens` / `Score`;
  both derive from `ITaskClient` (`Metadata.ModelName` replaces `Name`). Register them with `services.AddTextEmbedder(...)` /
  `AddTextReranker(...)` or create them with `TaskClientProvider.Create<ITextEmbedder>(taskModel)`. Removed: `EmbeddingModel`,
  `Reranker`, `EmbeddingLoadOptions`, `RerankerLoadOptions` (pooling and normalization overrides are `TaskLoadOptions.Adapter.EmbeddingPooling`
  / `EmbeddingNormalize`; the load-time reranker kind and instruction are gone, `RerankOptions.Instruction` stays), and in
  Sharplight.Hosting `EmbeddingModelLoaderFactory`, `RerankerModelLoaderFactory`, `RetrievalModelLoader`, `EmbeddingModelInstance`,
  `RerankerModelInstance`, `IEmbeddingModel`, `IRerankerModel`: hosted `Embedding` / `Reranker` models load through `TaskModelLoader`
  and a lease is an `ITaskClientModel` (`FindClient<ITextEmbedder>()`); `TaskClientInstance` wraps clients created elsewhere.
  New: `TextEmbeddingLoadDefaults`, `TaskLoadOptions.HalfPrecisionMatrices`, `TaskKindClients.HostedKind`.
- `SharplightEmbeddingGenerator.Load` is removed; `AddSharplightEmbeddingGenerator(o => ...)` takes `TaskLoadOptions` and registers
  the embedder with `AddTextEmbedder`.
- Seeds are `long?` everywhere: `SamplingOptions.Seed`, `SamplingParams.Seed` (LLM, speech, TTS), `AudioGenerationOptions.Seed`,
  `MusicGenerationOptions.Seed`. Seeds within the `int` range reproduce the output of earlier releases bit for bit.
- Audio and music generation follow the image / video shape: `GenerateAsync(request, options)` returns an `AudioGenerationResult`
  (audio and seed), `GenerateStreamingAsync(request, options)` yields `AudioGenerationUpdate(Step, TotalSteps, Result, Audio)`, and
  `IAudioGenerator.Defaults` (`AudioGenerationDefaults`) / `IMusicGenerator.Defaults` (`MusicGenerationDefaults`) replace
  `DefaultSteps`, `DefaultGuidanceScale` and `Supports*`. Steps, guidance, seed, shift, initial / step / decoder noise moved from
  `AudioGenerationRequest` to `AudioGenerationOptions`; guidance, greedy, temperature, top-k / top-p and seed moved from
  `MusicGenerationRequest` to `MusicGenerationOptions`. The `IProgress<(int Step, int Steps)>` parameters are gone.
- `VideoEnhancementUpdate` positional order is now `(Phase, Batch, TotalBatches, Result, Frames)`, like the image and video updates
  (counters, result, then previews / frames).

## 0.2 (first stable release)

Every model kind on one engine (text, speech recognition and synthesis, audio, vision, image and video generation), a verified-model
catalog, native audio and LGPL FFmpeg video I/O, Sharplight Studio pages for every kind, a native Metal provider (preview),
`Sharplight.Hosting` and multi-tenant serving seams, and the DI-based public API with the entry package `Sharplight`. Details, performance numbers, breaking changes since 0.1, known issues and license notes:
[docs/release-notes/v0.2.md](docs/release-notes/v0.2.md).

## 0.1.x (alpha)

Language models (GGUF, Hugging Face) and ONNX models on the CPU and Vulkan providers, the `.slm` package format, the
OpenAI/Anthropic-compatible server and Sharplight Studio (chat). Last alpha: `v0.1.22-alpha`.
