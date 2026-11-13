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

## Model compatibility

| Source | CPU | Vulkan |
|---|:-:|:-:|
| GGUF | ✓ | ✓ |
| Hugging Face (safetensors) | ✓ | ✓ |
| ONNX | ✓ | ✓ |
| `.slm` package | ✓ | ✓ |

Of the 172 architectures llama.cpp knows, 79 are supported end to end, 39 partially and
54 not yet ([docs/model-support.md](docs/model-support.md) has the full matrix). Supported (llama.cpp architecture names):

| Family | Architectures |
|---|---|
| Llama-like | `llama`, `llama4`, `mistral3` |
| Qwen | `qwen`, `qwen2`, `qwen3`, `qwen3moe` |
| Qwen hybrid | `qwen3next`, `qwen35`, `qwen35moe` |
| Gemma | `gemma`, `gemma2`, `gemma3`, `gemma4` |
| Phi | `phi2`, `phi3` |
| GLM | `chatglm`, `glm4`, `glm4moe` |
| Cohere | `command-r`, `cohere2` |
| Granite | `granite`, `granitemoe` |
| NVIDIA | `nemotron`, `nemotron_h`, `nemotron_h_moe` |
| OLMo | `olmo`, `olmo2`, `olmoe` |
| MLA/DeepSeek-like | `minicpm3`, `deepseek2` |
| MoE | `exaone-moe`, `bailingmoe2`, `dots1`, `afmoe`, `laguna`, `ernie4_5-moe`, `hunyuan-moe`, `hy_v3`, `gpt-oss`, `smallthinker`, `minimax-m2`, `mimo2`, `flint` |
| Other dense | `internlm2`, `minicpm`, `openelm`, `bitnet`, `exaone`, `exaone4`, `arcee`, `ernie4_5`, `hunyuan-dense`, `smollm3`, `seed_oss`, `apertus` |
| GPT-style | `starcoder2`, `gpt2`, `gptj`, `gptneox`, `falcon`, `mpt`, `bloom`, `stablelm`, `xverse` |
| SSM | `mamba`, `mamba2`, `jamba`, `falcon-h1`, `granitehybrid`, `plamo2` |
| RNN | `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7` |
| Hybrid conv | `lfm2`, `lfm2moe` |
| Meta | `muse-glimmer` |
| Multimodal | `clip` |

Weight types (CPU and Vulkan): F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S,
IQ4_NL/XS, TQ1_0, TQ2_0, MXFP4, NVFP4. ONNX operator coverage is in [docs/onnx-support.md](docs/onnx-support.md).

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
