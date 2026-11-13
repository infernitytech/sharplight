# Sharplight.Cli (`sharplight`)

Loads a model (GGUF file, `.slm` package or Hugging Face checkpoint directory) and generates text, describes images,
dumps logits or benchmarks it on a Vulkan GPU or the CPU. Run `--help` for every option.

```bash
M=~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf

# List providers, devices and device features
dotnet run -c Release --project samples/Sharplight.Cli -- --devices

# Generate text (greedy by default; --temp enables sampling)
dotnet run -c Release --project samples/Sharplight.Cli -- $M -p "The capital of France is" -n 32

# Use the model's chat template
dotnet run -c Release --project samples/Sharplight.Cli -- $M --chat -p "Say hi" -n 64 --temp 0.7

# Benchmark: warm prefill of 512 synthetic tokens, then 64 decode steps
# (set SHARPLIGHT_VK_PROFILE=1 for per-kernel profiles, SHARPLIGHT_DUMP_SCHEDULE=1 for the dispatch schedule)
dotnet run -c Release --project samples/Sharplight.Cli -- $M --bench 512

# Describe an image with a vision model
dotnet run -c Release --project samples/Sharplight.Cli -- model.gguf --mmproj mmproj.gguf --image photo.png -p "What is this?"

# Save the built model with this device's compiled artifacts as one package (see also samples/Sharplight.Convert)
dotnet run -c Release --project samples/Sharplight.Cli -- $M --save qwen3-0.6b.slm
```

| Option | Meaning |
| --- | --- |
| `--cpu`, `--device N` | Provider and device index (default: first GPU). |
| `--ctx`, `--batch` | Context length (4096) and rows per forward pass (512). |
| `--split a,b`, `--gpu-mem GiB` | Open the device once per weight and cap each slot's memory (multi-device pipeline testing). |
| `--cpu-offload` | Run layers that do not fit, and unsupported ops, on the CPU. |
| `--disable-features a,b` | Never use these device features (e.g. `matrix_f16`); for fallback testing. |
| `-p`, `-n`, `--temp`, `--chat`, `--think` | Prompt, token limit, temperature (0 = greedy), chat template, thinking mode. |
| `--image`, `--mmproj`, `--image-min-tokens`, `--image-max-tokens` | Image input. |
| `--bench N`, `--logits FILE`, `--save FILE` | Benchmark, logits dump (raw float32), package export. |

Exit codes: `0` success, `1` invalid command line, `2` runtime error (message on stderr), `130` cancelled with Ctrl+C.
