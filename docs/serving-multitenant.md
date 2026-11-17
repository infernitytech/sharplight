# Multi-tenant serving (`Sharplight.AspNetCore`)

The OpenAI- and Anthropic-compatible endpoints of `Sharplight.AspNetCore` serve one process-wide setup by default: the models of one
`ModelHost`, a static key list and global defaults. Hosts that serve several tenants, spaces or endpoints from one process plug in at
these seams; without them everything behaves as before.

| Seam | Interface / type | Default |
|---|---|---|
| Mapping into route groups | `routes.MapSharplightServer(o => ...)` on any `IEndpointRouteBuilder` (e.g. `app.MapGroup("/e/{endpoint}")`) | the root, once |
| Authentication | `IApiKeyValidator` → `ApiKeyValidationResult(IsValid, Principal, Scope)` | `StaticApiKeyValidator` over `SharplightServerOptions.ApiKeys` (empty = anonymous) |
| Per-request resolution | `IModelEndpointResolver` → `SharplightRequestContext` | `DefaultModelEndpointResolver`: every served model, global defaults, any tools, the key's scope |
| Quotas, rate limits, concurrency | `IRequestAdmission.AdmitAsync(context, model, http)`; throw `ApiException.TooManyRequests(...)` to refuse, return a handle to hold a slot | none |
| Usage | `ISharplightUsageObserver.OnRequestCompleted(SharplightUsage)` | none |
| Metrics | `SharplightServerMetrics`, meter `Sharplight.AspNetCore` | always registered |
| Prefix-cache isolation | `SharplightRequestContext.Scope` → `GenerationRequest.CachePartition` | one partition |

Hooks are taken from the `SharplightEndpointOptions` of the mapping first, then from the container, then the defaults.

## Per-request context

`SharplightRequestContext` is what the resolver returns for a request:

| Property | Effect |
|---|---|
| `Scope` | opaque isolation key (tenant, space, endpoint). Partitions the engine's prefix cache: a request only reuses or shares KV cache entries of requests with the same scope, so identical prompts of different scopes never share cache entries and a cache hit cannot tell one scope what another sent (timing side channel). Tags usage records and metrics. Null = the default scope. |
| `Principal` | who sends the request (from authentication); reported with the usage |
| `Tenant` | fair-share id for the engine's queue; default: scope plus the end-user id (`user`, `metadata.user_id`) or the key |
| `Models` | alias → registered model name. Only these names are accepted (others answer 404 `model_not_found`, as OpenAI does) and `/v1/models` lists only them; responses report the alias. Null = every served model. |
| `DefaultModel` | alias used when a request names no model |
| `Defaults` | `SharplightRequestDefaults`: sampling defaults, default and maximum generated tokens (requests above the limit are cut to it), request timeout, a default system prompt (put first when the messages have none) |
| `ModelDefaults` | `SharplightRequestDefaults` per alias, applied over `Defaults` field by field once the request's model is known, so the resolver does not have to read the body |
| `AllowedTools` | tool names a request may declare (OpenAI `tools[].function.name`, Anthropic `tools[].name`); others answer 400 `tool_not_allowed`. Null = any. |
| `Properties` | free data for usage observers and admission hooks (e.g. an endpoint or key id) |

Handlers and middleware read it with `HttpContext.GetSharplightRequestContext()`.

## Example: two endpoints with their own keys, models and defaults

```csharp
builder.Services.AddSharplight();
builder.Services.AddSharplightHosting("hosting.json");
builder.Services.AddSharplightServer(o => o.Admin = false);
builder.Services.AddSingleton<ISharplightUsageObserver, UsageWriter>();     // e.g. writes usage records

var app = builder.Build();
app.UseSharplightServer();                                                  // CORS and error mapping for the other routes
app.MapSharplightServer(o =>                                                // health and metrics only at the root
{
    o.OpenAi = false;
    o.Anthropic = false;
    o.Retrieval = false;
    o.Media = false;
});
app.MapGroup("/e/{endpoint}").MapSharplightServer(o =>
{
    o.Operations = false;
    o.Admin = false;
    o.KeyValidator = new EndpointKeyValidator(keyStore);                    // ApiKeyValidationResult(true, userId, scope: spaceId)
    o.Resolver = new EndpointResolver(endpointStore);                       // models, aliases, defaults, tools of the endpoint
    o.Admission = new QuotaAdmission(quotaStore);                           // ApiException.TooManyRequests(...) when over quota
});
```

A resolver for such an endpoint might read the route value and return:

```csharp
return new SharplightRequestContext
{
    Scope = authentication.Scope,
    Principal = authentication.Principal,
    Models = new Dictionary<string, string> { ["chat"] = "qwen3-4b", ["embed"] = "bge-small" },
    Defaults = new SharplightRequestDefaults { Sampling = new SamplingParams { Temperature = 0.3f }, MaxTokensLimit = 4096 },
    AllowedTools = new HashSet<string> { "search" },
    Properties = new Dictionary<string, object?> { ["endpoint"] = endpointId },
};
```

## Usage and metrics

After every API request (also failed ones) each observer gets a `SharplightUsage`: the operation as a short stable name
(`Api`: `openai.chat.completions`, `openai.completions`, `anthropic.messages`, `anthropic.messages.count_tokens`, `openai.embeddings`,
`rerank`, `openai.audio.transcriptions`, ..., see `SharplightApiNames`), the route pattern with its group prefix (`Route`), model alias, scope, principal, the
context, input / cached input / output tokens (streaming and non-streaming; embeddings and rerank count their input tokens), whether
it streamed, the HTTP status, the duration and the time to the first streamed token.

The same data feeds `System.Diagnostics.Metrics` instruments on the meter `Sharplight.AspNetCore` (collect them with OpenTelemetry's
`AddMeter("Sharplight.AspNetCore")` or `dotnet-counters`):

| Instrument | Unit | Tags |
|---|---|---|
| `sharplight.server.requests` (counter) | request | `sharplight.api`, `sharplight.model`, `sharplight.scope`, `http.response.status_code` |
| `sharplight.server.tokens` (counter) | token | `sharplight.api`, `sharplight.model`, `sharplight.scope`, `sharplight.token.type` (`input`, `cached_input`, `output`) |
| `sharplight.server.request.duration` (histogram) | s | as requests |
| `sharplight.server.time_to_first_token` (histogram) | s | `sharplight.api`, `sharplight.model`, `sharplight.scope` |

The Prometheus text on `/metrics` (engine metrics per model) is unchanged.

## In-process callers: `ServingChatClient`

Code in the same process (a chat UI, an agent) can use a model of the host through Microsoft.Extensions.AI with the same isolation:
`new ServingChatClient(modelService, new ServingChatClientOptions { Tenant = ..., CachePartition = scope, DefaultSystemPrompt = ... })`
(`Sharplight.Extensions.AI`) is an `IChatClient` over the batched serving engine; its requests are batched with the HTTP traffic,
queued fairly per tenant and isolated in the prefix cache per partition. Per call, `ChatOptions.AdditionalProperties` can override
the tenant, partition and priority (`ServingChatClient.TenantProperty`, `CachePartitionProperty`, `PriorityProperty`).

## Stored state

The serving layer keeps no per-request state between requests (no stored responses, files or batches): conversations are sent by
the client. Hosts that add stateful APIs key that state by `Scope` and `Principal` of the request context.
