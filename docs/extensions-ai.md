# Microsoft.Extensions.AI and Agent Framework adapters

Research snapshot: 2026-11-13. Everything below was restored and built against .NET 11 (SDK 11.0.100).

## What exists

### Microsoft.Extensions.AI (MEAI)

| Package | Version | Notes |
|---|---|---|
| `Microsoft.Extensions.AI.Abstractions` | 11.0.0 (stable) | `IChatClient`, `IEmbeddingGenerator<,>`, content types, `ChatOptions` |
| `Microsoft.Extensions.AI` | 11.0.0 (stable) | `ChatClientBuilder`, `UseFunctionInvocation()`, `AddChatClient(...)` DI, logging/OTel/caching middleware |

Shape of the API we implement:

- `IChatClient`: `GetResponseAsync(IEnumerable<ChatMessage>, ChatOptions?, ct)`, `GetStreamingResponseAsync(...)` returning `IAsyncEnumerable<ChatResponseUpdate>`, `GetService(Type, key)`, `IDisposable`.
- `ChatOptions`: `Temperature`, `TopP`, `TopK`, `MaxOutputTokens`, `StopSequences`, `Seed`, `FrequencyPenalty`, `PresencePenalty`, `ModelId`, `Instructions`, `ResponseFormat` (`ChatResponseFormat.Text/Json/ForJsonSchema`), `Tools` (`AITool`; `AIFunction`/`AIFunctionDeclaration` carry name, description and a JSON schema), `ToolMode` (`Auto/None/Required`), `AdditionalProperties`.
- Content model: `TextContent`, `TextReasoningContent`, `DataContent` (inline bytes + media type), `UriContent`, `FunctionCallContent` (CallId, Name, Arguments), `FunctionResultContent`, `UsageContent` (`UsageDetails` input/output/total tokens). Streaming updates carry these incrementally; `updates.ToChatResponse()` folds them into a `ChatResponse`.
- Tool calling is split in two on purpose: the client only reports `FunctionCallContent` (finish reason `ToolCalls`); the `FunctionInvokingChatClient` middleware executes the `AIFunction`s and re-calls the inner client with `FunctionResultContent` until the model answers. A provider therefore only needs to (a) render tool definitions and (b) parse calls.
- `IEmbeddingGenerator<string, Embedding<float>>`: `GenerateAsync(IEnumerable<string>, EmbeddingGenerationOptions?, ct)`.

### Microsoft Agent Framework

| Package | Version | Notes |
|---|---|---|
| `Microsoft.Agents.AI.Abstractions` / `Microsoft.Agents.AI` | 1.28.0 (stable) | `AIAgent`, `ChatClientAgent` (an agent = `IChatClient` + instructions + tools), sessions, workflows |
| `Microsoft.Agents.AI.Hosting` | 1.28.0-preview.261106.1 | DI/host integration: `builder.AddAIAgent("name", instructions)` returns an agent builder that resolves the registered `IChatClient` |
| `Microsoft.Agents.AI.Hosting.OpenAI` | 1.28.0-alpha.261106.1 | OpenAI-compatible HTTP endpoints for an agent (below) |
| `Microsoft.Agents.AI.Hosting.A2A(.AspNetCore)`, `.AGUI.AspNetCore` | 1.28.0-preview.261106.1 | Agent-to-agent and AG-UI protocols (not used here) |

`Microsoft.Agents.AI.Hosting.OpenAI` is real and restorable. It targets net8/9/10 and restores and runs fine on net11. It depends on the `OpenAI` SDK (>= 2.14) for the wire types, `Microsoft.Extensions.AI` >= 10.11 and `Microsoft.ML.Tokenizers`. API:

```csharp
builder.Services.AddChatClient(...);          // any IChatClient
builder.AddOpenAIChatCompletions();           // services for /chat/completions
builder.AddOpenAIResponses();                 // services for /responses
builder.AddOpenAIConversations();             // conversation store for the Responses API
var agent = builder.AddAIAgent("assistant", instructions: "...");
app.MapOpenAIChatCompletions(agent);          // POST /assistant/v1/chat/completions (stream: SSE chunks)
app.MapOpenAIResponses(agent);                // POST /assistant/v1/responses (+ response.* streaming events)
app.MapOpenAIConversations();                 // /v1/conversations
```

Paths can be overridden (`path:` / `responsesPath:`); several agents can be mapped side by side (the agent name is the route prefix and the accepted `model` value).

## What we use

- New library `src/Sharplight.Extensions.AI` (`Microsoft.Extensions.AI` 11.0.0, `StbImageSharp` for image decoding): `SharplightChatClient : IChatClient`, `ITextGenerationBackend` (engine abstraction), `SessionGenerationBackend`, `AddSharplightChatClient(...)` DI helpers (returns `ChatClientBuilder`, so `.UseFunctionInvocation()` etc. chain on).
- Sample `samples/Sharplight.AgentHost`: `SharplightChatClient` + `Microsoft.Agents.AI.Hosting(.OpenAI)` = a local model on `/assistant/v1/chat/completions` and `/assistant/v1/responses` in about 40 lines. Smoke-tested with Qwen3.5-0.8B (non-streaming and SSE streaming both work, usage is returned).
- Central package versions are in `Directory.Packages.props`.

## How the adapter works

1. Messages (`ChatMessage` + `ChatOptions.Instructions`) are mapped to Sharplight.Templating messages: system text is hoisted into one leading system message, `TextReasoningContent` becomes `reasoning_content`, `FunctionCallContent` becomes `tool_calls`, `FunctionResultContent` becomes `role: tool` messages with `tool_call_id`, images become `image` parts.
2. The model's own Jinja chat template (`tokenizer.chat_template`) renders prompt text; `tools` are passed to the template, `enable_thinking` too. Extra template variables: `SharplightChatClientOptions.TemplateVariables` or `ChatOptions.AdditionalProperties["template.<name>"]`.
3. The backend tokenizes and runs the generator; text streams back through an `OutputParser` which trims stop sequences, splits `<think>` reasoning into `TextReasoningContent`, and extracts tool calls into `FunctionCallContent` (finish reason `ToolCalls`).
4. The final update carries `UsageContent` (prompt tokens, generated tokens) and the finish reason (`Stop`, `Length`, `ToolCalls`).

Option mapping: `Temperature`, `TopP`, `TopK`, `Seed`, `MaxOutputTokens` (default 1024, clamped to remaining context), `StopSequences` map directly. `ModelId` only affects the reported model name. Frequency/presence penalties are ignored (Sharplight has a repetition penalty with different semantics).

Tool calling: formats are pluggable through `IToolCallParser` plus `ToolCallStart/End` markers. Built in: Hermes/Qwen2.5/Qwen3 (`<tool_call>{"name":..,"arguments":{..}}</tool_call>`) and Qwen3.5/Qwen3-Coder XML (`<function=..><parameter=..>`). XML parameter values are coerced using the declared JSON-schema type.

## Limits

- Generation is serialized: `SessionGenerationBackend` owns one `InferenceSession` and runs one request at a time (others wait on a semaphore). Each request starts from an empty context; there is no prefix/KV reuse across requests, so multi-turn chats re-prefill the whole history.
- To use a continuous-batching engine later, implement `ITextGenerationBackend` (stream text for a rendered prompt, fill `TextGenerationResult`) on top of the serving layer and hand it to `new SharplightChatClient(backend)`. The client holds no per-request model state and no lock of its own.
- Tools are best-effort: the template must support `tools` and the model must have been trained on its format; small models often do not call tools reliably. `ToolMode.Required` / a specific function cannot be enforced (no constrained decoding); `ToolMode.None` omits the tools. Hosted tools (web search etc.) are ignored. Parallel calls work if the model emits several `<tool_call>` blocks. Formats not listed above (Llama 3 `<|python_tag|>`, Mistral `[TOOL_CALLS]`, Gemma) need a custom parser/markers.
- `ResponseFormat.Json` / `ForJsonSchema` only adds an instruction to the system prompt; output is not grammar-constrained and may be invalid JSON.
- Images: `DataContent` with an `image/*` media type, user messages only, when `LlmModel.SupportsImages` (vision tower or mmproj loaded). `UriContent` is not downloaded. With a vision model the session runs on the multimodal decoder graph. Prompts with images skip the prompt-length clamp.
- Thinking: off by default (`SharplightChatClientOptions.EnableThinking`, per request `AdditionalProperties["enable_thinking"]`). Reasoning is returned as `TextReasoningContent`.
- Embeddings: `SharplightEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>` over `Sharplight.Llm.Embeddings.EmbeddingModel` (or any `ITextEmbedder`); DI via `services.AddSharplightEmbeddingGenerator(path, loadOptions)` (returns `EmbeddingGeneratorBuilder`). `EmbeddingGenerationOptions.Dimensions` cuts Matryoshka vectors (re-normalized); `AdditionalProperties["prompt_name"]` / `["input_type"]` select one of the model's prompts (e.g. Qwen3-Embedding's "query"), `["truncate"] = false` rejects over-long inputs. Usage reports input tokens. Text only: no `DataContent` image embeddings (no multimodal embedder is importable yet). There is no MEAI reranking abstraction; use `Sharplight.Llm.Embeddings.Reranker` directly or the server's `/v1/rerank`.
- Token usage `OutputTokenCount` includes the terminating end-of-generation token.

## Trade-offs: MEAI/Agent Framework hosting vs our own OpenAI/Anthropic server

| | MEAI + `Hosting.OpenAI` | Own hand-written server |
|---|---|---|
| Effort | Very low; protocol, SSE, Responses API, conversations come from the package | We own every wire detail |
| Ecosystem | Any MEAI middleware (function invocation, caching, OTel, rate limits), agents, workflows, A2A/AG-UI | None |
| Wire fidelity | Follows OpenAI spec as implemented by the package; observed quirks: `finish_reason: "stop"` on every streamed delta, request fields such as `max_tokens`/sampling are not necessarily forwarded to the `IChatClient` (agent-level options win) | Full control: logprobs, `n`, tool-choice, `stop`, usage chunks, model listing |
| Anthropic API | Not provided | Possible |
| Batching/serving features | Everything goes through the `IChatClient` text interface; slot management, prefix cache, per-request scheduling and metrics are invisible to it | Direct access to the engine |
| Dependencies / stability | Pulls `OpenAI`, `System.ClientModel`, OpenTelemetry, ML.Tokenizers; `Hosting` is preview and `Hosting.OpenAI` is alpha (weekly releases, API may break) | Only ASP.NET |
| AOT/trim | Reflection-heavy (JSON, DI) | Can be source-generated |

Recommendation: keep the MEAI adapter as the integration layer (it is the stable, high-value part: `IChatClient` is what Semantic Kernel, Agent Framework and most .NET apps consume) and use the Agent Framework hosting package for quick OpenAI-compatible exposure and as the demo. For a production serving endpoint (Anthropic API, full OpenAI parameter fidelity, direct engine control) a dedicated server on the serving layer remains the better fit; it can still share the template/tool-call/output-parsing code in this library.
