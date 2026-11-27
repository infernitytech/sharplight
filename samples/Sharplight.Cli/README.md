# Sharplight.Cli (`sharplight`)

Loads a model (GGUF file, `.slm` package, Hugging Face checkpoint directory, a stored model or a Hugging Face repository
that is pulled first) and generates text, describes images, dumps logits or benchmarks it on a Vulkan GPU or the CPU. Run
`--help` for every option.

It also serves models over HTTP (`serve`, see [Serving](#serving)), chats in the terminal (`run`) and manages the model store
(`pull`, `search`, `list`, `show`, `rm`, `store`, `storage`; see [docs/model-hub.md](../../docs/model-hub.md)):

```bash
sharplight pull Qwen/Qwen3-0.6B-GGUF:Q8_0
sharplight qwen3-0.6b:q8_0 --chat -p "Say hi" -n 32       # stored name
sharplight Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M -p "Hi"  # pulled when not stored; --no-pull fails instead
sharplight list
sharplight run qwen3-0.6b:q8_0                  # chat in the terminal (Ctrl+C stops an answer, /clear, /bye); -p "..." answers once
sharplight serve qwen3-0.6b:q8_0 --listen http://127.0.0.1:8080
```

```bash
M=~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf

# List providers, devices and device features
dotnet run -c Release --project samples/Sharplight.Cli -- --devices

# Generate text (greedy by default; --temp enables sampling)
dotnet run -c Release --project samples/Sharplight.Cli -- $M -p "The capital of France is" -n 32

# Use the model's chat template
dotnet run -c Release --project samples/Sharplight.Cli -- $M --chat -p "Say hi" -n 64 --temp 0.7

# Benchmark: warm prefill of 512 synthetic tokens, then 64 decode steps
# (set SHARPLIGHT_VK_PROFILE=1 for per-kernel profiles, SHARPLIGHT_PROFILE_TOP=n for more than 30 entries, SHARPLIGHT_DUMP_SCHEDULE=1 for the dispatch schedule;
#  SHARPLIGHT_BENCH_PREFILL_REPEATS=N times 1 + N prefills (default 4), SHARPLIGHT_BENCH_DECODE_RUNS=R times R decode runs of
#  64 steps after the prompt (default 1); the summary's "per run" part is llama-bench's mean ± sd of the per-run tok/s;
#  SHARPLIGHT_BENCH_PROMPT=vocab draws the prompt from the whole vocabulary, as llama-bench does, instead of ids 1000-49999)
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
| `--experts-on-disk` | With `--cpu-offload`: routed experts that fit neither the GPU nor host memory stay in the model file and are read on demand (MoE models larger than RAM + VRAM; see docs/hardware.md). With `--cpu`, `--gpu-mem` caps the CPU's own share and the rest of the experts go to disk. |
| `--expert-cache <size>` | With `--experts-on-disk`: host memory for the cache of experts read from disk (default: from the free host memory). |
| `--disable-features a,b` | Never use these device features (e.g. `matrix_f16`); for fallback testing. |
| `--fp8-block-scaled on\|off` | Block-scaled FP8 (default `on`): models with block-scaled E4M3 weights run their prompt-sized products on the block-scaled MMA with power-of-two scales where the device has it (CUDA, compute capability 12.x; about 2 % more serving throughput, perplexity +0.3 to +0.8 %, vLLM's default form). `off` keeps the checkpoints' f32 scales ([docs/cuda.md](../../docs/cuda.md)). |
| `-p`, `-n`, `--temp`, `--chat`, `--think` | Prompt, token limit, temperature (0 = greedy), chat template, thinking mode. |
| `--image`, `--mmproj`, `--image-min-tokens`, `--image-max-tokens` | Image input. |
| `--bench N`, `--logits FILE`, `--save FILE` | Benchmark, logits dump (raw float32), package export. |
| `--models-dir DIR`, `--no-pull`, `--revision`, `--keep-source`, `--no-precompile` | Model store and how missing models are pulled. |
| `--workers host:port,...`, `--remote-only` | Run the later pipeline stages on segment workers on other machines (see below); `--remote-only` places no layers here. |

### Several machines

A model too large for one machine runs pipeline parallel over segment workers ([docs/multinode.md](../../docs/multinode.md)). Every
worker serves its own copy of the same model and loads only the layers placed on it:

```bash
# On each worker node (GPU 0 by default; --cpu, --provider cuda, --cpu-offload, --experts-on-disk as usual)
sharplight segment-worker ~/models/qwen3-8b-q4_k_m.gguf --listen 0.0.0.0:7070

# On the coordinator: the first stage here, then the workers in order
sharplight ~/models/qwen3-8b-q4_k_m.gguf --workers 10.0.0.2:7070,10.0.0.3:7070 -p "The capital of France is"
sharplight serve ~/models/qwen3-8b-q4_k_m.gguf --workers 10.0.0.2:7070,10.0.0.3:7070
```

`segment-worker` options: `--listen host:port` (port 0 picks a free one and prints it), `--advertise host:port` (the address neighbouring
workers use), `--gpus N`, `--memory SIZE` (memory offered per device; coordinators weight their split by it).

Exit codes: `0` success, `1` invalid command line, `2` runtime error (message on stderr), `130` cancelled with Ctrl+C.

## Serving

```bash
# One model (positional); any "model" name sent by clients is accepted
dotnet run -c Release --project samples/Sharplight.Cli -- serve ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf --listen http://127.0.0.1:8080

# Several models, loaded on first request, keeping 4 GiB of VRAM free for other applications
dotnet run -c Release --project samples/Sharplight.Cli -- serve \
    --model qwen=~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf,slots=4 \
    --model smol=~/models/gguf/SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf,ctx=2048 \
    --vram-reserve 4G --lazy

# Models and limits from a file
dotnet run -c Release --project samples/Sharplight.Cli -- serve --config hosting.json
```

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"smol","stream":true,"messages":[{"role":"user","content":"Count to 5"}]}'
curl localhost:8080/v1/messages -H 'content-type: application/json' -H 'x-api-key: KEY' -H 'anthropic-version: 2023-06-01' \
     -d '{"model":"qwen","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}'
curl localhost:8080/admin/models
curl -X POST localhost:8080/admin/models/smol/unload
```

Python: `openai.OpenAI(base_url="http://localhost:8080/v1", api_key="x")` or `anthropic.Anthropic(base_url="http://localhost:8080", api_key="x")`.

### `serve` options

| Option | Meaning |
| --- | --- |
| `<model>...`, `--model-name` | Models to serve: file paths, stored names or Hugging Face names (pulled if missing); `--model-name` sets the id the API reports for a single model (default: file name, or the name as given). |
| `--model NAME=PATH[,key=value...]` | Another model; keys: `ctx`, `slots`, `batch`, `device` (`auto`, `cpu`, an index, a device id such as `vulkan:1`, several joined with `+` to split), `mmproj`, `template`, `lazy`, `pin`, `idle` (seconds), `offload`, `kind` (`chat`, `embedding`, `reranker`), `task` (a task kind, e.g. `time-series`: served on `/v1/timeseries/forecast`). Repeatable. |
| `--config FILE` | Hosting JSON ([docs/hosting.md](../../docs/hosting.md)). Command-line options win over file values. |
| `--listen URL` (`--urls`), `--api openai\|anthropic\|both`, `--api-key K`, `--cors ORIGIN` | HTTP surface; keys also protect `/admin`. |
| `--cpu`, `--device N`, `--ctx`, `--batch`, `--cpu-offload`, `--disable-features`, `--fp8-block-scaled` | Defaults for models without their own setting. |
| `--step-tokens N`, `--min-step-tokens N`, `--step-floor waves\|always\|full`, `--step-rows-per-sequence N` | Token budget of an engine step (default: the model's batch size); the floor of the load-adaptive budget while few sequences run (768; 0: always the whole budget); when the floor applies: `waves` (default) only while no sequence decodes, so the first prompts of a wave decode through shorter steps (lower TTFT and ITL tail at light load), `always` every step (the earlier default), `full` the whole budget while some sequence decodes; prompt rows per sequence holding a slot while some sequence decodes (48; larger: fewer, longer prompt steps, lower ITL p95, higher TTFT). |
| `--slots N`, `--max-queue N`, `--prefill-chunk N` | Concurrent sequences per model (4), waiting requests before HTTP 429 (64), prompt rows per step (512). |
| `--kv-tokens N`, `--kv-type T` | Paged KV cache: one pool of N tokens (K/M suffixes) shared by all slots, each growing to `--ctx` as needed, so many short chats fit in the memory of a few long ones; requests are admitted by free cache blocks, the newest running sequences are preempted and recomputed when the pool runs out, and common prompt prefixes are shared between requests. Default: each slot reserves `--ctx` tokens. KV element type f16 (default), f32, q8_0 or q4_0. See [docs/kv-cache.md](../../docs/kv-cache.md). |
| `--vram-reserve SIZE`, `--vram-budget SIZE`, `--max-loaded N` | Memory kept free (1G), total cap, resident model limit with LRU eviction. |
| `--lazy`, `--idle-timeout SECONDS` | Load on first request; unload idle models (they reload on demand). |
| `--on-demand` | Also serve every model of the model store by name (ollama-style): listed in `/v1/models`, loaded on its first request; combine with `--idle-timeout`, `--max-loaded` and `--vram-reserve` for TTL and LRU eviction. |
| `--chat-template FILE`, `--temp T`, `--mmproj FILE` | Template override, default temperature (0.7), vision projector. |

Models that do not fit in free device memory are refused with a clear error (or, with `--cpu-offload`, partly run on the CPU);
idle models are evicted least-recently-used first to make room for the one requested.

### Configuration file

`--config` takes the JSON form of `Sharplight.Hosting`'s options (comments and trailing commas allowed), the same file a server reads
with `services.AddSharplightHosting("hosting.json")`; see [docs/hosting.md](../../docs/hosting.md):

```json
{
  "reserve": "2G",
  "maxLoadedModels": 2,
  "idleTimeoutSeconds": 600,
  "lazy": true,
  "defaults": { "device": "auto", "contextLength": 4096, "slots": 4 },
  "models": [
    { "name": "qwen", "path": "/models/Qwen3-0.6B-Q8_0.gguf", "device": 0, "lazy": false, "pinned": true, "cpuOffload": true },
    { "name": "bge", "path": "/models/bge-small-en-v1.5.gguf", "kind": "embedding" }
  ]
}
```
