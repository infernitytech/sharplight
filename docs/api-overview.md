# Public API overview

This is a short tour of the types an application uses. Every public type and member carries XML documentation (shipped as
`*.xml` next to the assemblies), which is the reference; this page shows how the pieces fit together.

## Assemblies

The core engine and everything an application calls to run a model live in `Sharplight.Core`. Model formats (importers) and the
graph generation for model architectures (adapters) are separate assemblies, so the core never knows a file format: format names
(GGUF, safetensors, ONNX, diffusers, whisper.cpp, ...) appear only in `Sharplight.Import.*`.

| Assembly | Role | Main entry points |
|---|---|---|
| `Sharplight.Core` | Graph IR, model definition and description, runtime and provider abstractions; the task layer on top of compiled models: text generation (the former `Sharplight.Llm`: sessions, sampling, continuous batching, chat, structured output, speculative decoding, embeddings, reranking), speech recognition and synthesis, image generation; tokenizers and Jinja chat templates; audio (WAV, resampling) and image (PNG, planar tensors) I/O | `LlmModel`, `ITextGenerator`, `TaskModel`, `TaskClientProvider`, `ISpeechRecognizer`, `ISpeechSynthesizer`, `IImageGenerator`, `ModelDescription`, `InferenceEngine`, `ModelService`, `EmbeddingModel`, `Reranker`, `WavFile`, `PngCodec` |
| `Sharplight.Architectures` | Execution-graph generation: architecture adapters that turn canonical model sources into Sharplight graphs | `BuiltinAdapters` |
| `Sharplight.Import.Gguf`, `.HuggingFace`, `.Onnx`, `.WhisperCpp` | Format importers into the canonical model source | `GgufImporter`, `HuggingFaceImporter`, `OnnxImporter`, `WhisperCppImporter` |
| `Sharplight.Package` | Read and write `.slm` packages (every model kind: description, components, attachments), convert models to them and precompile kernels into them | `ModelPackage`, `ModelPackageReader`, `ModelConverter`, `PackagePrecompiler` |
| `Sharplight.Cpu`, `Sharplight.Vulkan` (+ `Sharplight.Shaders`) | Execution providers | `CpuExecutionProvider`, `VulkanProvider` |
| `Sharplight.Defaults` | One registration entry point that wires up everything above | `services.AddSharplight()`, `SharplightDefaults.CreateServiceProvider()` |
| `Sharplight.AspNetCore` | OpenAI- and Anthropic-compatible HTTP endpoints for ASP.NET Core | `AddSharplight`, `UseSharplight`, `MapSharplight`, `AddSharplightEmbeddings`, `AddSharplightReranker`, `AddSharplightSpeechRecognizer` |
| `Sharplight.Hub` | Search and pull Hugging Face models into a local store ([model-hub.md](model-hub.md)) | `HubClient`, `ModelPuller`, `ModelStore` |
| `Sharplight.Extensions.AI` | `Microsoft.Extensions.AI` adapters | `SharplightChatClient`, `SharplightEmbeddingGenerator`, `SharplightSpeechToTextClient`, `SharplightTextToSpeechClient`, `SharplightImageGenerator` |

## Registration

Composition goes through Microsoft.Extensions.DependencyInjection. `services.AddSharplight()` (Sharplight.Defaults) registers the
model formats (GGUF, Hugging Face, ONNX, whisper.cpp, `.slm`), the built-in architecture adapters, the execution providers (Vulkan, CPU) and the
task layer; hosts without a container use `SharplightDefaults.CreateServiceProvider()`. Narrower setups call
`services.AddSharplightModelFormats()`, `.AddSharplightProviders()` and `.AddSharplightTasks()` (Core) individually.

```csharp
var services = new ServiceCollection().AddSharplight();
var provider = new VulkanProvider();
var device = provider.OpenDevice(provider.GetDevices()[0]);
var load = new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 8192 };
```

The importers, adapters and providers still keep process-wide registries (`ModelImporterRegistry`, `ModelAdapterRegistry`,
`ProviderRegistry`, filled by `GgufImporter.Register()` and friends, which `AddSharplight()` calls once); they become services in a
later pass. The task layer has no static state.

## Tasks: one model concept, one client pattern

Every model kind is the same thing to the engine: a `ModelDefinition` with graphs and weights plus a `ModelDescription` (kind and
components, see [model-format.md](model-format.md)), loaded through one load path (importer, then adapter, or the prebuilt definition
of an `.slm` package) into a `TaskModel`. What an application calls is a **task client**, and all kinds follow the same pattern:

| kind (`ModelKinds`) | client | whole result | streaming | request options | result |
|---|---|---|---|---|---|
| `text-generation` | `ITextGenerator` | `GenerateAsync(prompt)` | `GenerateStreamingAsync` → `GeneratedToken` | `GenerationOptions` | `TextGenerationResult` |
| `speech-recognition` | `ISpeechRecognizer` | `TranscribeAsync(AudioBuffer)` | `TranscribeStreamingAsync` → `TranscriptionSegment` | `SpeechRecognitionOptions` | `Transcription` |
| `speech-synthesis` | `ISpeechSynthesizer` | `SynthesizeAsync(text)` | `SynthesizeStreamingAsync` → `AudioBuffer` | `SpeechSynthesisOptions` | `SpeechSynthesisResult` |
| `image-generation` | `IImageGenerator` | `GenerateAsync(ImageGenerationRequest)` | `GenerateStreamingAsync` → `ImageGenerationUpdate` | `ImageGenerationOptions` | `ImageGenerationResult` |
| `image-classification` | `IImageClassifier` | `ClassifyAsync(ImageInput)` | - (one image, one result) | `ImageClassificationOptions` | `ImageClassification` |
| `image-embedding` | `IImageEmbedder` | `EmbedImageAsync(ImageInput)`, `EmbedTextAsync`, `ClassifyZeroShotAsync` | - | `ImageEmbeddingOptions` | `float[]`, `ImageClassification` |
| `object-detection` | `IObjectDetector` | `DetectAsync(ImageInput)` | - | `ObjectDetectionOptions` (score threshold, max detections, optional NMS) | `ObjectDetection` (`DetectedObject`: class, label, score, `BoundingBox` in pixels) |

Every client is an `ITaskClient`: `Metadata` (kind, model name, architecture), `GetService(Type)` for the objects behind it,
`IDisposable` (it owns its compiled graphs, not the model), async calls with a `CancellationToken`. The shape mirrors
Microsoft.Extensions.AI (`IChatClient`, `ISpeechToTextClient`, `ITextToSpeechClient`, `IImageGenerator`), and
`Sharplight.Extensions.AI` adapts the clients to those interfaces; Core itself does not depend on Microsoft.Extensions.AI, whose speech
and image abstractions are still experimental.

Implementations are `ITaskClientFactory` services (kind, client type, `CanCreate(model)`, `Create(model)`), resolved by the
`TaskClientProvider` from the model's kind; the last registration that accepts a model wins. Architecture packages register theirs
next to their adapters (`services.AddTaskClientFactory<ISpeechRecognizer>(ModelKinds.SpeechRecognition, model => ...)`); a new kind
(depth estimation, audio codecs, time series, ...) is a new kind string, client interface and factory, without changes to Core types.
Text generation is built in: `LlmModel` is the text-generation model and `LlmModel.CreateTextGenerator()` its client.

A client for a configured model is one registration; the model is loaded once on first use with `TaskLoadOptions` from
Microsoft.Extensions.Options (named per client, so keyed registrations can load different models) and disposed by the container
after the client:

```csharp
var services = new ServiceCollection()
    .AddSharplight()
    .AddTextGenerator(o => { o.ModelPath = "qwen3-4b-q4_k_m.gguf"; o.Devices = [slot]; o.ContextLength = 8192; })
    .AddSpeechRecognizer(o => { o.ModelPath = "whisper-small.slm"; o.Devices = [slot]; }, serviceKey: "whisper");
await using var provider = services.BuildServiceProvider();

var generator = provider.GetRequiredService<ITextGenerator>();
TextGenerationResult completion = await generator.GenerateAsync("The capital of France is", new GenerationOptions { MaxNewTokens = 32 });

var recognizer = provider.GetRequiredKeyedService<ISpeechRecognizer>("whisper");
Transcription text = await recognizer.TranscribeAsync(WavFile.Read("talk.wav"));
```

Without a container: `using var model = TaskModel.Load(new TaskLoadOptions { ModelPath = path, Devices = [slot] })`, then
`taskClientProvider.Create<ISpeechRecognizer>(model)`. Task implementations use `TaskModel.CompileComponent(role)` (compiles a
component graph with the component's default bounds, consulting the package's precompiled artifacts), `CreateTokenizer(role)`,
`Metadata(role)` and `ReadAttachment(role, name)`.

Audio and image I/O for the task layer: `AudioBuffer` (interleaved float samples; `ToMono`, `Resample`, `ToModelInput`),
`AudioResampler` (band-limited windowed sinc), `WavFile` (read: integer PCM 8/16/24/32-bit and float 32/64-bit, plain or
`WAVE_FORMAT_EXTENSIBLE`; write: PCM16, PCM24, PCM32, float32), `PngCodec` (encode RGB; decode 8-bit gray, gray+alpha, RGB, RGBA and
palette PNGs) and `PlanarImage` (8-bit interleaved RGB to and from `[3, height, width]` float planes in any value range).

### Speech recognition

The built-in `ISpeechRecognizer` serves encoder-decoder models with a prompted decoder (Whisper and its derivatives, from Hugging Face
checkpoints, whisper.cpp `.bin` files or `.slm` packages); `AddSharplightTasks()` registers its factory. Its model description has an
`encoder` and a `decoder` component (graphs), a `tokenizer` component (the tokenizer as a JSON attachment) and a `feature-extractor`
component (log-mel configuration and the mel filter bank as an attachment); the decoder's canonical metadata carries the control
tokens (`SpeechKeys`). A request computes the log-mel features of the whole recording on the host, then per 30 s window runs the
encoder, detects the language when `SpeechRecognitionOptions.Language` is null, and decodes greedily (or with `Sampling`, or with a
beam search of `BeamSize` beams scored like transformers' `num_beams`) with Whisper's timestamp rules. Long recordings advance to the last complete timestamp segment of each window; `ConditionOnPreviousText`
prompts each window with the text so far, `Prompt` conditions the first. `Timestamps = TimestampGranularity.None` decodes without
timestamp tokens (one segment per window, as transformers does by default). Each `TranscriptionSegment` carries its token ids, mean log
probability and no-speech probability. Not yet: temperature fallback, word timestamps.

```csharp
var services = new ServiceCollection().AddSharplight()
    .AddSpeechRecognizer(o => { o.ModelPath = "ggml-small.bin"; o.Devices = [slot]; });   // or a HF directory / .slm
await using var provider = services.BuildServiceProvider();
var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
var result = await recognizer.TranscribeAsync(WavFile.Read("talk.wav"), new SpeechRecognitionOptions { Language = "en" });
foreach (var segment in result.Segments)
{
    Console.WriteLine($"[{segment.Start:mm\\:ss\\.ff} - {segment.End:mm\\:ss\\.ff}] {segment.Text}");
}
```

The sample `samples/Sharplight.Transcribe` (`sharplight-transcribe model audio.wav [--language en] [--translate] [--beam-size 5] [--format srt]`) is this
in a command-line tool. Models of other speech families (CTC, transducers, audio LLMs) plug in as their own `ITaskClientFactory` for the
same kind; the feature extraction sits behind a small internal interface so a graph-based front end can replace the host one.

Vision models share host-side preprocessing in Core: `ImageProcessorConfig` (resize mode and filter, center crop, rescale, mean and std;
the neutral form of a model's processor configuration, stored in the `feature-extractor` component's configuration under
`imageProcessor`) and `ImageProcessor` (Pillow-exact antialiased resize, crop, normalize to planar floats, `Patchify`). Importers translate
the model's own processor file into the config; the image clients never see a format name. Models with a `vision-encoder` component
(graph `image_encode`: patches in, pooled embedding and optionally logits out) run through the built-in classifier / embedder
implementation (`ImageEncoderAdapter`: ViT, DINOv2, ...); `services.AddImageClassifier(...)` and `services.AddImageEmbedder(...)` register
configured models like every other kind. CLIP and SigLIP (adapter `ImageTextAdapter`, `vision-encoder` + `text-encoder` + `tokenizer` components, graphs `image_encode` and `text_encode`) add the text tower
to the same embedder: `EmbedTextAsync` embeds a prompt into the image space and `ClassifyZeroShotAsync` scores an image against candidate prompts (softmax for CLIP, sigmoid for SigLIP, with the model's learned scale and bias).
Object detectors (DETR: `DetrAdapter`; RT-DETR: `RtDetrAdapter`; graph `detect`: normalized `pixels` in, per-query `logits` and `boxes` out) run through `services.AddObjectDetector(...)`; the box decoding, thresholding and
non-maximum suppression are generic host-side utilities in Core (`BoundingBox`, `BoxUtilities.NonMaxSuppression`, `ObjectDetectionOptions`). `samples/Sharplight.Vision` (`sharplight-vision classify|detect model image.png [--label prompt]... [-o out.png]`)
is this in a command-line tool, drawing the detections into a PNG.

The importers and architecture adapters for speech and image models come in their own workstreams; the task layer defines the
interfaces, options, results, the description and the host-side plumbing they plug into.

## 1. Load a model (text generation)

The rest of this page covers text generation in depth: `LlmModel` and the serving stack around it. `LlmModel.Load` reads the file and builds the compute graphs. Nothing is compiled or uploaded to a device until you create a
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
  `HiddenStates`). Drafts are the head's greedy choices, chained k times per step. Heads: Qwen3.5 / Qwen3-Next MTP layers,
  and the NextN layer of DeepSeek-V3 (`latent_decoder`) and GLM-4.5 (`decoder`, any family whose checkpoint carries a NextN
  layer after uniform layers; `MtpDraftHead.FirstNextnModule`). The head compiles against `LlmModel.SharedWeights`, so its
  embedding and LM head are the decoder's buffers (only its own layer adds device memory).

Verification runs on the device when the decoder samples there (`LlmLoadOptions.DeviceSampling`, `IDeviceSamplingDecoder`):
every verification row is sampled (greedy or from the sequence's seeded stream at that row's position) by the sampling head,
and the step reads back one token and log-probability per row instead of logits rows. The decoder downloads hidden states
only for the rows the draft head asks for (`HiddenStates(input, rows)`), and not at all when nothing captures them.

The decoder must be created with room for the verification rows and the drafts
(`CreateBatchDecoder(slots, maxLogitsRows: ..., draftTokens: k)`; `ChatModelOptions.Speculator` / `MtpDraftTokens` do that
for hosted models). Attention-only models roll back any number of rows by truncation; models with recurrent state (gated
delta nets, Mamba, RWKV, short convs) get their recurrent ops compiled in rollback form for `k + 1` rows
(`Graph.RecurrentRollback`, `IBatchDecoder.MaxSpeculativeRows`): a speculative run records its per-row op inputs and leaves
the state at the run start, `Truncate` picks how many rows to keep, and the slot's next run replays them inside the same
kernel before its own rows. The engine speculates while at most `SpeculationMaxBatch` sequences decode and verifies at most
`SpeculationMaxDraftRows` drafts per step (fewer per sequence as the batch grows); requests with an output constraint are not
speculated. CLI and server: `--speculate-mtp k`, `--speculate-ngram k`; `sharplight-convert precompile --drafts k` stores
the kernels of the verifying decoders and the draft head. `LlmModel.EstimateFootprint(slots, ctx, kvCacheTokens, draftTokens,
draftHead, maxLogitsRows)` counts the rollback record states, the verification logits rows and (with `draftHead`) the head's
own weights, KV cache and activations; the hosted loader uses it for admission when `MtpDraftTokens` is set.

### Weight sharing between compiled models

`CompileOptions.SharedWeights` (a `SharedWeights` registry) lets compiled models of the same `ModelSource` use one device
buffer per weight: buffers are keyed by device, weight source, type, size and the provider's layout (repack), reference
counted, filled by the first holder and freed with the last. Providers opt in through `IComputeDevice.SharesWeights` (CPU and
Vulkan do); placement does not count weights already resident. `LlmModel` passes its registry (`LlmModel.SharedWeights`) to
every decoder and draft head it compiles.

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
(`/v1/messages`, `/v1/messages/count_tokens`) endpoints, `/v1/embeddings` and `/v1/rerank` for registered retrieval models, `/v1/audio/transcriptions` and `/v1/audio/translations` for
registered speech recognizers (`services.AddSharplightSpeechRecognizer("whisper-1")` serves the container's `ISpeechRecognizer`; multipart
WAV upload, `json`, `text`, `srt`, `vtt` and `verbose_json` responses),
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
var services = new ServiceCollection().AddSharplight();
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

The other task kinds map the same way: `SharplightSpeechToTextClient` (`ISpeechToTextClient` over an `ISpeechRecognizer`, WAVE input),
`SharplightTextToSpeechClient` (`ITextToSpeechClient` over an `ISpeechSynthesizer`, WAVE output) and `SharplightImageGenerator`
(`IImageGenerator` over a Sharplight image generator, PNG output). `services.AddSharplightSpeechToTextClient(key)`,
`.AddSharplightTextToSpeechClient(key)` and `.AddSharplightImageGenerator(key)` wrap the container's task clients.

## Conventions

- Types that own device memory or threads (`LlmModel`, `TaskModel`, every task client, `InferenceSession`, `InferenceEngine`,
  `ModelHost`, `EmbeddingModel`, `Reranker`, `VisionEncoder`) are `IDisposable`; dispose them in reverse order of creation (a
  service container does that for the clients and models it created).
- `InferenceEngine`, `ModelHost`, `ModelService`, `EmbeddingModel` and `Reranker` are thread-safe. Sessions, generators and
  samplers are not: one thread at a time.
- Request options are immutable records with defaults that work. Load options are classes: `TaskLoadOptions` has settable
  properties for Microsoft.Extensions.Options; the older `LlmLoadOptions` / `EmbeddingLoadOptions` are `init`-only. Only `Devices` is
  required.
- Failures that a caller can act on are typed exceptions: `EngineException` and its subclasses for admission, `ModelHostException`
  and its subclasses for hosting.
- The CPU provider's kernels do not depend on the application's JIT settings. By default .NET first compiles a method with loops as
  unoptimized tier-0 code and switches to optimized code only after on-stack replacement and tiering catch up, so the first runs of a
  model (and short benchmarks) would measure unoptimized kernels. Every method with a loop in `Sharplight.Cpu` is marked
  `MethodImplOptions.AggressiveOptimization` and compiled fully optimized on first use, whatever `TieredCompilationQuickJitForLoops`
  the application sets. In an application with the .NET defaults, runs 1-5 of resnet50 on the CPU averaged about 48 ms without the
  attribute and 10 ms with it.
