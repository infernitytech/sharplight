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

Reduced precision falls back to bf16 activations for graphs whose activations exceed f16's range (SDXL's VAE decoder: 11.2 → 67.8 dB
at the same speed): new `CompileOptions.ActivationPrecision`, `ActivationRangeProbe` / `ActivationRange` and
`TaskModel.CalibrateActivationPrecision` / `ActivationPrecisionOf`; the image generator calibrates its VAE graphs once.

## 0.2 (first stable release)

Every model kind on one engine (text, speech recognition and synthesis, audio, vision, image and video generation), a verified-model
catalog, native audio and LGPL FFmpeg video I/O, Sharplight Studio pages for every kind, a native Metal provider (preview),
`Sharplight.Hosting` and multi-tenant serving seams, and the DI-based public API with the entry package `Sharplight`. Details, performance numbers, breaking changes since 0.1, known issues and license notes:
[docs/release-notes/v0.2.md](docs/release-notes/v0.2.md).

## 0.1.x (alpha)

Language models (GGUF, Hugging Face) and ONNX models on the CPU and Vulkan providers, the `.slm` package format, the
OpenAI/Anthropic-compatible server and Sharplight Studio (chat). Last alpha: `v0.1.22-alpha`.
