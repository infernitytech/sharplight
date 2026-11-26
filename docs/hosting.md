# Hosting models (`Sharplight.Hosting`)

`Sharplight.Hosting` turns a list of models and a memory policy into a running `ModelHost` that servers and apps share: chat,
embedding, reranker and task models (speech, image generation, vision, ...) on the same devices, loaded at startup or on the
first request, unloaded when idle, and evicted least recently used first when another model needs the room. It adds device
placement, fit planning and a status snapshot that health checks, dashboards and external cluster coordinators can read. The
`sharplight serve` command (`samples/Sharplight.Cli`) and the HTTP endpoints of `Sharplight.AspNetCore` are built on it.

## Registration

```csharp
builder.Services.AddSharplight();                       // formats, architectures, execution providers, task layer
builder.Services.AddSharplightHosting(options =>
{
    options.Reserve = 2L << 30;                          // keep 2 GiB free on every device
    options.MaxLoadedModels = 3;                         // LRU eviction beyond three resident models
    options.IdleTimeoutSeconds = 600;
    options.Models.Add(new HostedModelOptions { Name = "qwen3", Path = "/models/qwen3-4b.slm", Pinned = true });
    options.Models.Add(new HostedModelOptions { Name = "bge", Path = "/models/bge-small.slm", Kind = ModelKind.Embedding, Lazy = true });
    options.Models.Add(new HostedModelOptions { Name = "rerank", Path = "/models/bge-reranker.slm", Kind = ModelKind.Reranker, Lazy = true });
});
builder.Services.AddSharplightServer();                 // optional: the OpenAI / Anthropic endpoints over the same host
```

Or from a file: `services.AddSharplightHosting("hosting.json", options => options.Log = ...)`.

| Service | What it is |
|---|---|
| `IModelHostManager` (`ModelHostManager`) | owns the `ModelHost`; `Register`, `LoadAsync`, `UnloadAsync`, `RemoveAsync`, `GetStatus`, `PlanFit` |
| `ModelHost` | the host itself: `AcquireAsync(name, kind)` returns a lease that keeps the model loaded while a request uses it |
| `DeviceInventory` | the devices of the available execution providers (GPUs first), each opened once with the host's budget and disabled features |
| `IModelPlacementPolicy` (`DefaultModelPlacementPolicy`) | picks the devices of a model each time it loads; replace it for other placement rules |
| `IHostedModelLoaderFactory` (chat, embedding, reranker, task) | creates the loader of a model by kind; register another one to add a kind or override a built-in one (the last registered is asked first) |
| `IHostedModelResolver` (optional) | on-demand models (`OnDemand = true`): names that are not registered are looked up here, e.g. every model of a local store |
| hosted service | loads the non-lazy models when the application starts (`LoadStartupModels`) |

The manager is a singleton: the container disposes it (unloading every model) before the device inventory (closing the devices).

## Using a model

```csharp
var host = services.GetRequiredService<ModelHost>();
using (var lease = await host.AcquireAsync("qwen3", ModelKind.Chat, cancellationToken))     // loads it if needed
{
    var chat = ((IChatModel)lease.As<LoadedModel>()).Service;                              // ModelService
}

using (var lease = await host.AcquireAsync("bge", ModelKind.Embedding, cancellationToken))
{
    var embedder = ((ITaskClientModel)lease.As<LoadedModel>()).FindClient<ITextEmbedder>();  // ITextEmbedder
}
```

Embedding and reranking models load through the task loader like every other task model (`TaskModelLoaderFactory`, as kinds
`text-embedding` and `text-reranking` with `TextEmbeddingLoadDefaults`); a lease on any of them is an `ITaskClientModel`
(`FindClient<ITextReranker>()`, `FindClient<ISpeechRecognizer>()` and the like). `TaskClientInstance` registers clients an application
created itself.

## Options and the JSON form

`ModelHostingOptions` binds from code or from JSON (`ModelHostingJson.Load/Parse/Serialize`: camelCase names, comments and
trailing commas allowed, sizes as bytes or `"2G"`):

```json
{
  "reserve": "2G",
  "budget": null,
  "maxLoadedModels": 3,
  "idleTimeoutSeconds": 600,
  "lazy": false,
  "onDemand": false,
  "disabledFeatures": [],
  "defaults": {
    "device": "auto",
    "contextLength": 4096,
    "batchTokens": 0,
    "slots": 4,
    "kvCacheTokens": null,
    "kvCacheType": "f16",
    "cpuOffload": false,
    "blockScaledFp8": true,
    "stepFloor": "waves",
    "stepRowsPerSequence": 48,
    "kvAdmission": "queue",
    "maxQueue": 64
  },
  "models": [
    { "name": "qwen3", "path": "/models/qwen3-4b.slm", "device": 0, "pinned": true },
    { "name": "bge", "path": "/models/bge-small.slm", "kind": "embedding", "lazy": true },
    { "name": "rerank", "path": "/models/bge-reranker.slm", "kind": "reranker", "device": "cpu", "lazy": true },
    { "name": "whisper", "path": "/models/whisper-small.slm", "kind": "task", "task": "speech-recognition", "lazy": true }
  ]
}
```

| Host setting | Meaning |
|---|---|
| `reserve` | memory kept free on every device for other applications (default 1 GiB); a model loads only when it fits with this to spare |
| `budget` | cap on what the host uses of each GPU (default: what the driver reports as available) |
| `maxLoadedModels` | resident model limit (0 = none); loading another evicts the least recently used idle, unpinned model |
| `idleTimeoutSeconds` | unload models idle this long (they reload on demand) |
| `lazy` | models without their own `lazy` load on the first request instead of at startup |
| `onDemand` | also serve the models of the registered `IHostedModelResolver` |
| `disabledFeatures` | device features no model may use (fallback testing, broken drivers) |
| `defaults` | per-model settings for models that do not set their own: device, context, batch rows, slots, paged KV cache (`kvCacheTokens`), KV type, CPU offload, reduced precision, block-scaled FP8 (`blockScaledFp8`, below), queue limit, prefill chunk, step tokens (`stepTokens`, `minStepTokens`, `stepFloor`, `stepRowsPerSequence`, below), KV admission when the paged cache runs short (`kvAdmission`: `queue` or `preempt`, below), n-gram and MTP speculation, projector, chat template |

Per model: `name`, `path`, `kind` (`chat` default, `embedding`, `reranker`, `task` with `task` = the task kind), `device`,
`contextLength`, `batchTokens`, `slots`, `kvCacheTokens`, `kvCacheType`, `projector`, `chatTemplate`, `auxiliary` (files by role),
`lazy`, `pinned`, `cpuOffload`, `blockScaledFp8`, `idleTimeoutSeconds`, `workers` (segment workers for the later pipeline stages), `description`. `HostedModelSettings.Resolve` merges a model with the defaults.

Engine defaults that changed on 2026-11-23 (all can be set back):

| Setting | Default | Meaning |
|---|---|---|
| `blockScaledFp8` (defaults or per model; `TaskLoadOptions.BlockScaledFp8`) | `true` | models with block-scaled E4M3 weights run their prompt-sized products on the block-scaled MMA with power-of-two scales where the device has it (CUDA, compute capability 12.x): about 2 % more serving throughput at 64 streams, perplexity +0.3 to +0.8 % against the checkpoints' f32 scales, the form vLLM uses by default. `false` keeps the f32 scales. No effect on other models and devices (docs/cuda.md, "Block-scaled FP8 by default"). `SHARPLIGHT_FP8_BLOCK_SCALED=1\|0` overrides it for the process |
| `stepFloor` (defaults; `EngineOptions.StepFloor`) | `waves` | when the `minStepTokens` floor of the load-adaptive step budget applies: `waves` only while no sequence decodes (at 16 closed-loop streams TTFT p50 -22 to -28 %, ITL p99 -20 to -28 %, throughput unchanged for a dense FP8 model and -1.7 % for gpt-oss-20b; neutral at 64 streams; benchmarks.md §93.4), `always` every step (the earlier default), `full` the whole budget while some sequence decodes (lowest ITL tail, longest TTFT). `SHARPLIGHT_STEP_FLOOR_WAVES=1\|0\|full` overrides it for the process |
| `stepRowsPerSequence` (defaults; `EngineOptions.BatchTokensPerSequence`) | `48` (was 32) | prompt rows per sequence holding a slot that a step may carry while some sequence decodes (rounded up to a 128-row token tile: 768 rows at 16 slots). At 16 closed-loop streams against 32: ITL p95 about halved (gpt-oss-20b 20-22 → 12 ms, Qwen3-8B-FP8 25 → 15 ms), throughput +1 % on gpt-oss-20b and unchanged on Qwen3-8B-FP8, TTFT p50 +20 ms and closed-loop ITL p99 +25 % (benchmarks.md §100.1, §101). At 64 streams the budget is the whole step either way. `SHARPLIGHT_STEP_ROWS_PER_SEQUENCE=n` overrides it for the process |
| `EngineOptions.PricedPromptChunks` (engine option, no hosting key yet; `SHARPLIGHT_STEP_PRICED_CHUNKS=0\|1`) | on (2026-11-26) | while some sequence decodes, the engine prices the prompt rows of its own steps (their time over the decode-only step of the same decode rows, fitted as a fixed price + a price per row + a growth term) and takes `EngineOptions.SmallChunkRowsPerSequence` (16, 256 rows at 16 slots) instead of `stepRowsPerSequence` rows per slot when those rows cost at least 10 % less per row in the smaller chunks. Dense models, whose prompt rows ride the decode step's weight reads (Qwen3-8B-FP8: 33 µs per prompt row at 16 + 240 rows against 52 µs at 16 + 752), get shorter prompt-bearing steps: at 16 closed-loop streams ITL p99 ~48 → ~23 ms at equal throughput and lower TTFT (benchmarks.md §R47). Mixture-of-experts models, whose larger chunks are as cheap or cheaper per row, keep `stepRowsPerSequence` |

`kvAdmission` (defaults; `EngineOptions.KvAdmission`, `serve --kv-admission`, `SHARPLIGHT_KV_ADMISSION=queue|preempt`): what a paged KV cache
does when it runs short. `queue` (default) admits a request only with room for its prompt and up to 64 tokens of growth, so new requests wait
while the pool is full; `preempt` admits on the prompt alone and preempts the newest running sequences (recomputed later) when the running ones
need blocks, as vLLM does: at KV saturation a lower time to first token for some throughput (Llama-3.1-8B bf16, 256 streams over a pool of
100,480 tokens: TTFT p50 1.9 against 3.9 s, -4 % tok/s; docs/kv-cache.md 2.5, benchmarks.md §109). Both admit the same requests while the pool
has room.

Delegates that are not part of the JSON form: `Log` (load, unload and eviction messages; default the manager's `ILogger`),
`ConfigureLoad` (adjust each model's load options: progress, logging, compilation artifacts) and `ConfigureChat` (adjust each chat
model's options: request callbacks, speculators, service defaults).

## Devices and placement

| `device` | Placement |
|---|---|
| `auto` (default) | the GPU with the most free memory beyond the reserve, preferring one where a file-size estimate fits; the CPU when there is no GPU |
| `cpu` | the CPU |
| `0`, `1`, ... | the n-th device of the first available GPU provider |
| `vulkan:1` | a device id (`provider:index`, as `DeviceInventory.Devices` lists them) |
| `0,1` | the model is split over these devices in pipeline order, each weighted by its free memory beyond the reserve |

A model too large for this machine also takes **segment workers** on other machines (`workers`, `host:port` in pipeline order;
[multinode.md](multinode.md)): they follow the model's devices in pipeline order, each weighted by its free memory beyond the reserve, and
run the later layers on their own copy of the model (`"workers": ["10.0.0.2:7070", "10.0.0.3:7070"]`; `sharplight serve --workers`). The
workers' connections come from the `RemoteDevicePool` (`AddSharplightHosting()` registers it; `services.Configure<RemoteSegmentOptions>`
sets TLS and timeouts); a worker failure fails the model's requests with `SegmentWorkerException` until the model is loaded again.

The placement is decided each time a model loads, so `auto` follows the memory that is free at that moment. With `cpuOffload` a
GPU placement gets the CPU as fallback for layers that do not fit. Chat models are sized exactly (`LlmModel.EstimateFootprint`);
embedding, reranker and task models are sized from their files (weights plus a quarter plus 256 MiB).

## Status for health checks and coordinators

`IModelHostManager.GetStatus()` returns a `HostStatus` (serializable with `ToJson()` / `HostStatus.FromJson()`):

- `devices`: per device (and per memory pool a model was charged against) the total, used, free and available (free minus the
  reserve) bytes, the bytes held by this host's models, whether the figures are the driver's (`isExact`) and which models are on it;
  devices not opened yet report their total memory as free;
- `models`: per registered model the kind, state, devices, resident and estimated bytes, leases, and for chat models the queue
  depth, active sequences, slots, recent tokens per second, requests and token counts;
- totals: loaded models, queue depth, active sequences, tokens per second; the reserve and the resident model limit.

`PlanFit(footprint, device, cpuOffload)` answers whether a model of a given footprint would load on the named devices now, as the
host admits a load: fits, fits after unloading idle models (which ones), needs CPU offload (which share) or does not fit. It is the
same `FitPlanner` Studio uses for its fit badges.

## Extending

- A new model kind: implement `IHostedModelLoaderFactory` (`CanLoad` by `HostedModelSettings.Kind` or path, `CreateLoader` returning an
  `IHostedModelLoader`; wrap it in `PlacedModelLoader` to get per-load placement) and register it with
  `services.TryAddEnumerable(ServiceDescriptor.Singleton<IHostedModelLoaderFactory, MyFactory>())`.
- Other placement rules (labels, anti-affinity, a coordinator's decision): implement `IModelPlacementPolicy` and register it before
  `AddSharplightHosting` (it uses `TryAdd`).
- Models from a registry or store: implement `IHostedModelResolver`, set `OnDemand = true`.
