# Public API overview

This is a short tour of the types an application uses. Every public type and member carries XML documentation (shipped as
`*.xml` next to the assemblies), which is the reference; this page shows how the pieces fit together.

| Assembly | Role | Main entry points |
|---|---|---|
| `Sharplight.Llm` | Load models, generate, batch, host several models, embed and rerank | `LlmModel`, `InferenceSession`, `Generator`, `InferenceEngine`, `ModelService`, `ModelHost`, `EmbeddingModel`, `Reranker`, `VisionEncoder` |
| `Sharplight.AspNetCore` | OpenAI- and Anthropic-compatible HTTP endpoints for ASP.NET Core | `AddSharplight`, `UseSharplight`, `MapSharplight`, `AddSharplightEmbeddings`, `AddSharplightReranker` |
| `Sharplight.Packaging` | Convert models to `.slm` packages and precompile kernels into them | `ModelConverter`, `PackagePrecompiler` |
| `Sharplight.Hub` | Search and pull Hugging Face models into a local store ([model-hub.md](model-hub.md)) | `HubClient`, `ModelPuller`, `ModelStore` |
| `Sharplight.Extensions.AI` | `Microsoft.Extensions.AI` adapters | `SharplightChatClient`, `SharplightEmbeddingGenerator`, `AddSharplightChatClient`, `AddSharplightEmbeddingGenerator` |

Format support is not part of these assemblies. The host registers the importers it wants once at startup
(`GgufImporter.Register()` from `Sharplight.Gguf`, `HuggingFaceImporter.Register()` from `Sharplight.HuggingFace`, ...), and
opens the compute devices (`Sharplight.Cpu`, `Sharplight.Vulkan`) that it hands to the load options as `DeviceSlot`s.

```csharp
GgufImporter.Register();
var provider = new VulkanProvider();
var device = provider.OpenDevice(provider.GetDevices()[0]);
var load = new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 8192 };
```

## 1. Load a model

`LlmModel.Load` reads the file and builds the compute graphs. Nothing is compiled or uploaded to a device until you create a
session, a batch decoder or a vision encoder from the model. Dispose the model after everything created from it.

```csharp
using var model = LlmModel.Load("qwen3-4b-q4_k_m.gguf", load);
Console.WriteLine(model.Definition.Name);        // architecture, weights, metadata
string? template = model.ChatTemplate;           // the model's Jinja chat template, if it has one
```

`LlmLoadOptions` controls the devices (several devices split the layers; a CPU `Fallback` takes what does not fit), the context
length, the maximum rows per forward pass (`MaxBatchTokens`), disabled device features and auxiliary files such as a vision
projector (`Auxiliary = [("mmproj", path)]`).

## 2. Generate

The simplest path is one sequence: an `InferenceSession` holds the context (KV cache), a `Generator` runs the sampling loop.
Generation is a lazy iterator; successive calls continue the same context.

```csharp
using var session = model.CreateSession();
var generator = new Generator(session, model.Tokenizer);
var options = new GenerationOptions
{
    MaxNewTokens = 128,
    Sampling = new SamplingOptions { Temperature = 0.7f, TopP = 0.95f, Seed = 42 },
    StopStrings = ["\n\n"],
};
foreach (var token in generator.Generate("The capital of France is", options))
{
    Console.Write(token.Text);
}

Console.WriteLine($"{generator.LastStats!.DecodeTokensPerSecond:F1} tok/s");
```

Images: create the session on `WellKnown.DecodeMultimodalGraph`, pass a `VisionEncoder` to the generator and give the prompt
one image placeholder per image.

```csharp
using var vision = model.CreateVisionEncoder();
using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
var generator = new Generator(session, model.Tokenizer, vision);
var image = new ImageInput(width, height, rgbBytes);          // Sharplight.Vision
foreach (var token in generator.Generate(promptWithImagePlaceholder, [image], options)) { /* ... */ }
```

## 3. Serve many requests: the engine

`InferenceEngine` is a continuous-batching scheduler. It owns a batch decoder (several sequence slots advanced in one forward
pass), admits requests from any thread, shares cached prompt prefixes between them and streams results. Use it instead of
sessions whenever more than one caller is involved.

```csharp
using var engine = new InferenceEngine(model.CreateBatchDecoder(slots: 4), model.Tokenizer);

var request = new GenerationRequest
{
    PromptTokens = model.Tokenizer.Encode("Once upon a time", addSpecial: true, parseSpecial: false),
    MaxTokens = 64,
    StopStrings = ["."],
    Priority = 1,                            // higher runs first when requests queue
    Timeout = TimeSpan.FromSeconds(30),
};
await foreach (var chunk in engine.GenerateAsync(request))
{
    Console.Write(chunk.Text);               // the last chunk carries Finish and Usage
}

var snapshot = engine.GetMetrics();          // queue depth, slot use, TTFT / inter-token histograms, prefix-cache hits
```

Admission failures are exceptions with a machine-readable `Code` (`QueueFullException`, `PromptTooLongException`,
`EngineStoppedException`, all deriving from `EngineException`). `DrainAsync` stops admission and waits for running requests.

On top of the engine, `ModelService` is the chat layer: it renders messages and tools through the chat template, parses
reasoning blocks and tool calls out of the output (`ToolCallFormats`) and returns typed events.

```csharp
var service = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions { ModelName = "qwen3" });
var result = await service.StartChat(new ChatRequest
{
    Messages = [new ChatTurn("system", "Answer briefly."), new ChatTurn("user", "Name three primary colours.")],
    MaxTokens = 128,
}).CompleteAsync();
Console.WriteLine(result.Choices[0].Content);          // also .Reasoning, .ToolCalls, .Usage
```

### Scheduling

Every engine step has a token budget (`EngineOptions.MaxBatchTokens`, default the decoder's batch size): decoding sequences
get their rows first, prompts fill the rest in chunks of at most `PrefillChunk` rows, so a long prompt never stalls running
streams for more than one step (chunked prefill). Queued requests are admitted by `Priority`, then by fair share: among
equal priorities the `Tenant` with the least recent usage (tokens, decaying with `FairShareHalfLife`) goes first. A
cancelled request (token or client disconnect) frees its slot at the next step boundary. `MinTokens` suppresses end-of-
generation and stop tokens until reached; `StopAtEog = false` generates through them (benchmarks).

### Structured outputs

A request can carry an `IOutputConstraint`; the engine asks it for the allowed tokens before every sample, masks the logits
(a packed `TokenMask` bitset, so a GPU sampler can take the same mask as an input) and reports the chosen token back.
`GrammarConstraint` implements it for a byte-level context-free `Grammar` built from GBNF (the llama.cpp format), a regular
expression or a JSON schema. Masks are computed by walking a precompiled vocabulary trie of the tokenizer in step with the
grammar; per-stack-top results are cached and shared between requests, so a JSON string body costs a few hundred
microseconds after its first use (151k-token Qwen3 vocabulary).

```csharp
var schema = """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""";
var constraint = new GrammarConstraint(Grammar.FromJsonSchema(schema), model.Tokenizer);   // reuse across requests
var request = new GenerationRequest { PromptTokens = prompt, MaxTokens = 200, Constraint = constraint };
```

The chat layer does this for you: `ChatRequest.ResponseFormat = OutputFormat.JsonSchema(schema)` (also `JsonObject`,
`Regex`, `Gbnf`, `Choice`) constrains the answer while a reasoning block before it stays free, and tool calls are constrained
to the model's tool-call syntax and each tool's parameter schema when the call is forced (`ToolChoiceMode.Required` /
`Function`) or the tool is `Strict` (then the grammar starts at the call marker, plain answers stay free). Supported
JSON-schema subset: see `JsonSchemaCompiler` (types, properties/required in order, enums/consts, anyOf/oneOf/allOf, local
`$ref` incl. recursion, array bounds, string lengths, patterns, common formats).

### Speculative decoding

`EngineOptions.Speculator` proposes draft tokens that the target model verifies in one multi-row forward pass; a draft token
is kept while it equals what the target samples at that position. That is speculative rejection sampling for a point draft
(accept with probability p(draft), else sample from p without the draft), so greedy output is token-identical and sampled
output is distributed exactly as without speculation. Two speculators exist:

- `NgramSpeculator` drafts by prompt lookup (continuations of the last n-gram earlier in the context), which pays off for
  edits, code and quoting.
- `MtpSpeculator` (`LlmModel.CreateMtpSpeculator(slots, draftTokens)`, when `LlmModel.HasDraftHead`) runs the model's own
  multi-token-prediction head (the `mtp_draft` graph that adapters build from MTP / NextN layers, see `MtpDraftHead`): its
  own multi-sequence decoder with one attention layer and KV cache per MTP layer, fed the target's hidden states
  (`ISpeculator.UsesHiddenStates`, `Observe`; the decoder captures them through `IBatchDecoder.CaptureHiddenStates` /
  `HiddenStates`). Drafts are the head's greedy choices, chained k times per step.

The decoder must be created with room for the verification rows and the drafts
(`CreateBatchDecoder(slots, maxLogitsRows: ..., draftTokens: k)`; `ChatModelOptions.Speculator` / `MtpDraftTokens` do that
for hosted models). Attention-only models roll back any number of rows by truncation; models with recurrent state (gated
delta nets, Mamba, RWKV, short convs) get their recurrent ops compiled in rollback form for `k + 1` rows
(`Graph.RecurrentRollback`, `IBatchDecoder.MaxSpeculativeRows`): a speculative run records its per-row op inputs and leaves
the state at the run start, `Truncate` picks how many rows to keep, and the slot's next run replays them inside the same
kernel before its own rows. The engine speculates while at most `SpeculationMaxBatch` sequences decode and verifies at most
`SpeculationMaxDraftRows` drafts per step (fewer per sequence as the batch grows); requests with an output constraint are not
speculated. CLI and server: `--speculate-mtp k`, `--speculate-ngram k`; `sharplight-convert precompile --drafts k` stores
the kernels of the verifying decoders and the draft head.

### Several models

`ModelHost` registers models by name, loads them on first use within the device memory budget, evicts the least recently used
idle model when memory is short, unloads idle ones after a timeout and hands out leases so a model in use is never unloaded.

```csharp
using var host = new ModelHost(new ModelHostOptions { ReserveBytes = 2L << 30, IdleTimeout = TimeSpan.FromMinutes(10) });
host.Register(new ModelRegistration
{
    Name = "qwen3-4b",
    Loader = new LlmChatModelLoader("qwen3-4b", "qwen3-4b-q4_k_m.gguf", load, new ChatModelOptions { Slots = 4 }),
});

using var lease = await host.AcquireAsync("qwen3-4b");            // loads if needed
var chat = lease.As<ChatModelInstance>().Service;
```

## 4. Serve over HTTP

`Sharplight.AspNetCore` maps OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and Anthropic-compatible
(`/v1/messages`, `/v1/messages/count_tokens`) endpoints, `/v1/embeddings` and `/v1/rerank` for registered retrieval models,
`/health` (summary), `/health/live` and `/health/ready` (probes; ready answers 503 while draining or loading), `/metrics`
(Prometheus: queue depth, batch size, step time, TTFT / inter-token histograms, KV-cache use, speculation acceptance) and
`/admin/models`. Beyond the standard fields the APIs accept `ignore_eos`, `min_tokens`, `priority`, `user` (the fair-share
tenant; else the API key), `response_format` (`json_object`, `json_schema`), strict tools (`tools[].function.strict`), the
vLLM guided-decoding fields (`guided_json`, `guided_regex`, `guided_grammar`, `guided_choice`, `structured_outputs`) and,
on `/v1/messages`, `output_format` and `metadata.user_id`. A full queue answers 429 with `Retry-After`.

```csharp
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSharplight(host, o =>               // a ModelHost; or pass a ModelService for a single model
{
    o.ApiKeys.Add("secret");
    o.DefaultSampling = new SamplingParams { Temperature = 0.7f, TopP = 0.95f };
});

var app = builder.Build();
app.UseSharplight();      // CORS, API keys, error mapping; call before the Map* methods
app.MapSharplight();      // everything the options enable
app.Run();
```

`SharplightServerOptions` switches the APIs on and off, sets keys, CORS origins, request limits and timeouts, and decides
whether an unknown `model` falls back to the default one. Existing OpenAI / Anthropic clients only change their base URL.

## 5. Embeddings and reranking

`EmbeddingModel` runs encoder embedders (BERT family, bge, e5, MiniLM, nomic) and decoder embedders (Qwen3-Embedding).
Inputs are tokenized, truncated, packed and run in batches; pooling and normalization follow the model's metadata.

```csharp
using var embedder = EmbeddingModel.Load("bge-small-en-v1.5", new EmbeddingLoadOptions { Devices = [slot] });

EmbeddingResult result = embedder.Embed(["first text", "second text"]);
float[] vector = result.Vectors[0];                       // embedder.Dimensions long, L2-normalized when the model says so
float[] query = embedder.Embed("what is a vector?", new EmbedOptions { Prompt = "query" });   // a prompt the model defines
float[] small = embedder.Embed("text", new EmbedOptions { Dimensions = 256 });                // Matryoshka cut
```

`Reranker` scores documents against a query with a cross-encoder or a yes/no decoder reranker:

```csharp
using var reranker = Reranker.Load("bge-reranker-v2-m3", new RerankerLoadOptions { Devices = [slot] });
RerankResult scores = reranker.Score("what is a vector?", ["a quantity with direction", "a kind of animal"]);
// scores.Scores in [0, 1] per document, in input order
```

Both implement small interfaces (`ITextEmbedder`, `ITextReranker`) that the server and the Microsoft.Extensions.AI adapter
depend on, so a test double or a remote implementation can stand in.

## 6. Microsoft.Extensions.AI

`SharplightChatClient` implements `IChatClient`: messages and tools are rendered with the model's chat template, reasoning
becomes `TextReasoningContent`, tool calls become `FunctionCallContent` (wrap with `UseFunctionInvocation()` to run them) and
images are `DataContent`. `SharplightEmbeddingGenerator` implements `IEmbeddingGenerator<string, Embedding<float>>`.

```csharp
GgufImporter.Register();
var services = new ServiceCollection();
services.AddSharplightChatClient("qwen3-4b-q4_k_m.gguf", load, o => o.EnableThinking = false).UseFunctionInvocation();
services.AddSharplightEmbeddingGenerator("bge-small-en-v1.5", new EmbeddingLoadOptions { Devices = [slot] });

var provider = services.BuildServiceProvider();
var chat = provider.GetRequiredService<IChatClient>();
var answer = await chat.GetResponseAsync("Why is the sky blue?");
Console.WriteLine(answer.Text);
```

The default backend runs one generation at a time. To let many concurrent callers share a model, implement
`ITextGenerationBackend` over an `InferenceEngine` and pass it to the `SharplightChatClient` constructor. See also
[extensions-ai.md](extensions-ai.md).

## Conventions

- Types that own device memory or threads (`LlmModel`, `InferenceSession`, `InferenceEngine`, `ModelHost`, `EmbeddingModel`,
  `Reranker`, `VisionEncoder`) are `IDisposable`; dispose them in reverse order of creation.
- `InferenceEngine`, `ModelHost`, `ModelService`, `EmbeddingModel` and `Reranker` are thread-safe. Sessions, generators and
  samplers are not: one thread at a time.
- Options are immutable records or `init`-only classes with defaults that work; only `Devices` is required.
- Failures that a caller can act on are typed exceptions: `EngineException` and its subclasses for admission, `ModelHostException`
  and its subclasses for hosting.
