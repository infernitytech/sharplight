# Changelog

Release notes per version are in [docs/release-notes](docs/release-notes). Prereleases (`-alpha`) are built from `develop` and
have no curated notes; their changes are the commit log.

## 0.2 (first stable release)

Every model kind on one engine (text, speech recognition and synthesis, audio, vision, image and video generation), a verified-model
catalog, native audio and LGPL FFmpeg video I/O, Sharplight Studio pages for every kind, a native Metal provider (preview),
`Sharplight.Hosting` and multi-tenant serving seams, and the DI-based public API with the entry package `Sharplight`. Details, performance numbers, breaking changes since 0.1, known issues and license notes:
[docs/release-notes/v0.2.md](docs/release-notes/v0.2.md).

## 0.1.x (alpha)

Language models (GGUF, Hugging Face) and ONNX models on the CPU and Vulkan providers, the `.slm` package format, the
OpenAI/Anthropic-compatible server and Sharplight Studio (chat). Last alpha: `v0.1.22-alpha`.
