# Sharplight

An inference engine for large language models and ONNX models, written in C# for .NET 11, with a Vulkan GPU backend
(Slang compute kernels, cooperative-matrix tensor cores where available) and a CPU backend.

- **Models:** GGUF and Hugging Face checkpoints for the llama.cpp architecture families ([docs/model-support.md](docs/model-support.md)),
  ONNX models including data-dependent shapes and control flow ([docs/onnx-support.md](docs/onnx-support.md)).
- **Serving:** OpenAI- and Anthropic-compatible HTTP APIs for ASP.NET Core (`Sharplight.AspNetCore`), continuous batching,
  paged KV cache with prefix sharing, quantized KV cache ([docs/kv-cache.md](docs/kv-cache.md)),
  `Microsoft.Extensions.AI` adapters ([docs/extensions-ai.md](docs/extensions-ai.md)).
- **Model packages:** the `.slm` format stores optimized, device-independent graphs and weights, plus optional precompiled
  artifacts per execution provider (SPIR-V, driver binaries) ([docs/model-format.md](docs/model-format.md)).

Start with the [public API overview](docs/api-overview.md). Benchmarks against llama.cpp and vLLM are in
[docs/benchmarks.md](docs/benchmarks.md).

## Building

```sh
dotnet build Sharplight.slnx -c Release
```

Tests are Microsoft.Testing.Platform executables: run `dotnet tests/<Project>/bin/Release/net11.0/<Project>.dll`.

## Samples

| Sample | Executable | What it does |
|---|---|---|
| `samples/Sharplight.Cli` | `sharplight` | Chat / generate / benchmark from the command line |
| `samples/Sharplight.Server` | `sharplight-server` | Serve one or several models over HTTP |
| `samples/Sharplight.Convert` | `sharplight-convert` | Convert models to `.slm` packages, precompile, inspect |
| `samples/Sharplight.PackageViewer` | `sharplight-viewer` | Avalonia viewer for `.slm` packages (graphs, IR, weights, artifacts) |
| `samples/Sharplight.OnnxRun` | `sharplight-onnxrun` | Run an ONNX model on the CPU or Vulkan provider |

## License

[MIT](LICENSE)
