# Sharplight.Server (`sharplight-server`)

Serves one or several models over the OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and
Anthropic-compatible (`/v1/messages`) HTTP APIs, plus `/health`, `/metrics` and `/admin/models`. Models share the opened
devices and run concurrently, each with its own scheduler. Run `--help` for every option; stop with Ctrl+C.

It also pulls models from Hugging Face, ollama style: `search`, `pull`, `list`, `show`, `rm`, `run` (interactive chat) and
`serve`. Pulled models are converted to `.slm` packages, precompiled for the local GPU and kept in a local store; `run` and
`serve` pull missing models automatically. See [docs/model-hub.md](../../docs/model-hub.md) for the details.

```bash
sharplight-server search qwen3 --format gguf
sharplight-server pull Qwen/Qwen3-0.6B-GGUF:Q8_0      # tag = GGUF quantization; default Q4_K_M, else Q8_0
sharplight-server run qwen3-0.6b:q8_0                  # chat in the terminal (/clear, /bye); -p "..." answers once
sharplight-server serve qwen3-0.6b:q8_0 Qwen/Qwen3-0.6B --listen http://127.0.0.1:8080
sharplight-server list
sharplight-server rm qwen3-0.6b:q8_0
```

| Option | Meaning |
| --- | --- |
| `--models-dir DIR` | Model store (default: `SHARPLIGHT_MODELS`, else `models` next to the executable). All commands. |
| `--revision REV`, `--force` | Branch, tag or commit to pull (`main`); rebuild a model that is up to date. |
| `--keep-source`, `--no-precompile` | Keep the downloaded files; skip kernel precompilation. |
| `--no-pull` | `run`/`serve`: fail instead of pulling a model that is not stored. |
| `HF_TOKEN`, `HF_ENDPOINT` | Token for gated/private repositories (never printed); Hub mirror. |

## Serving files

```bash
# One model (positional); any "model" name sent by clients is accepted
dotnet run -c Release --project samples/Sharplight.Server -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf --listen http://127.0.0.1:8080

# Several models, loaded on first request, keeping 4 GiB of VRAM free for other applications
dotnet run -c Release --project samples/Sharplight.Server -- \
    --model qwen=~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf,slots=4 \
    --model smol=~/models/gguf/SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf,ctx=2048 \
    --vram-reserve 4G --lazy

# Models and limits from a file
dotnet run -c Release --project samples/Sharplight.Server -- --config models.json
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

## Options

| Option | Meaning |
| --- | --- |
| `<model>...`, `--model-name` | Models to serve: file paths, stored names or Hugging Face names (pulled if missing); `--model-name` sets the id the API reports for a single model (default: file name, or the name as given). |
| `--model NAME=PATH[,key=value...]` | Another model; keys: `ctx`, `slots`, `batch`, `device` (index or `cpu`), `mmproj`, `template`, `lazy`, `pin`, `idle` (seconds), `offload`. Repeatable. |
| `--config FILE` | JSON, see below. Command-line options win over file values. |
| `--listen URL` (`--urls`), `--api openai\|anthropic\|both`, `--api-key K`, `--cors ORIGIN` | HTTP surface; keys also protect `/admin`. |
| `--cpu`, `--device N`, `--ctx`, `--batch`, `--cpu-offload`, `--disable-features` | Defaults for models without their own setting. |
| `--slots N`, `--max-queue N`, `--prefill-chunk N` | Concurrent sequences per model (4), waiting requests before HTTP 429 (64), prompt rows per step (512). |
| `--kv-tokens N`, `--kv-type T` | Paged KV cache: one pool of N tokens (K/M suffixes) shared by all slots, each growing to `--ctx` as needed, so many short chats fit in the memory of a few long ones; requests are admitted by free cache blocks, the newest running sequences are preempted and recomputed when the pool runs out, and common prompt prefixes are shared between requests. Default: each slot reserves `--ctx` tokens. KV element type f16 (default), f32, q8_0 or q4_0. See [docs/kv-cache.md](../../docs/kv-cache.md). |
| `--vram-reserve SIZE`, `--vram-budget SIZE`, `--max-loaded N` | Memory kept free (1G), total cap, resident model limit with LRU eviction. |
| `--lazy`, `--idle-timeout SECONDS` | Load on first request; unload idle models (they reload on demand). |
| `--chat-template FILE`, `--temp T`, `--mmproj FILE` | Template override, default temperature (0.7), vision projector. |

Models that do not fit in free device memory are refused with a clear error (or, with `--cpu-offload`, partly run on the CPU);
idle models are evicted least-recently-used first to make room for the one requested.

## Configuration file

Comments and trailing commas are allowed.

```json
{
  "vramReserve": "2G",
  "maxLoaded": 2,
  "idleTimeout": 600,
  "lazy": true,
  "models": [
    { "name": "qwen", "path": "/models/Qwen3-0.6B-Q8_0.gguf", "ctx": 4096, "slots": 4, "device": 0, "lazy": false, "pin": true, "offload": true }
  ]
}
```
