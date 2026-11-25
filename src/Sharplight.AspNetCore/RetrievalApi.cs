using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Embeddings;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.AspNetCore;

/// <summary>
/// Retrieval endpoints: OpenAI-compatible <c>POST /v1/embeddings</c> and Cohere / Jina-compatible <c>POST /v1/rerank</c> (also
/// <c>/v2/rerank</c>, <c>/rerank</c>). Models are registered with <c>services.AddSharplightEmbeddings(...)</c> /
/// <c>services.AddSharplightReranker(...)</c>; <c>MapSharplightServer</c> maps these endpoints when any is registered, or call
/// <see cref="MapSharplightRetrieval"/> directly (e.g. for a host that serves only embeddings).
/// </summary>
/// <example>
/// <code>
/// var load = new TaskLoadOptions { ModelPath = "bge-small-en-v1.5.slm", Devices = [slot] };
/// TextEmbeddingLoadDefaults.Apply(load);   // then override what differs
/// using var model = TaskModel.Load(loader, load);
/// using var embedder = provider.Create&lt;ITextEmbedder&gt;(model);   // the TaskClientProvider
/// builder.Services.AddSharplightEmbeddings(embedder, name: "bge-small");
/// var app = builder.Build();
/// app.UseSharplightServer();
/// app.MapSharplightRetrieval();   // POST /v1/embeddings with {"model": "bge-small", "input": ["text"]}
/// </code>
/// </example>
public static class RetrievalApi
{
    extension(IEndpointRouteBuilder routes)
    {
        /// <summary>
        /// Maps <c>/v1/embeddings</c> and the rerank routes when embedding / rerank models are registered, either directly
        /// (<c>AddSharplightEmbeddings</c>, <c>AddSharplightReranker</c>) or as models of kind <c>Embedding</c> / <c>Reranker</c> of the server's
        /// <see cref="Llm.Serving.Hosting.ModelHost"/> (e.g. through Sharplight.Hosting); nothing when there are none.
        /// </summary>
        public IEndpointRouteBuilder MapSharplightRetrieval()
        {
            var models = routes.ServiceProvider.GetService<RetrievalModels>() ?? new RetrievalModels();
            var context = routes.ServiceProvider.GetService<SharplightServerContext>();
            var host = context?.Host;
            bool dynamic = context?.Options.ServeHostModels ?? false;
            bool HostHas(ModelKind kind) => host?.List().Any(model => model.Kind == kind) ?? false;
            if (dynamic || models.Embedders.Count > 0 || HostHas(ModelKind.Embedding))
            {
                routes.MapPost("/v1/embeddings", http => Handle(http, () => Embeddings(http, models)));
            }
            if (dynamic || models.Rerankers.Count > 0 || HostHas(ModelKind.Reranker))
            {
                foreach (var path in new[] { "/v1/rerank", "/v2/rerank", "/rerank" })
                {
                    routes.MapPost(path, http => Handle(http, () => Rerank(http, models)));
                }
            }
            return routes;
        }
    }

    /// <summary>Errors become API error bodies even when the host did not add <c>UseSharplight</c>.</summary>
    private static async Task Handle(HttpContext http, Func<Task> body)
    {
        try
        {
            await body();
        }
        catch (ApiException ex)
        {
            await ApiErrors.WriteAsync(http, ex);
        }
        catch (ArgumentException ex)
        {
            string? code = ex.Message.Contains("at most") ? "context_length_exceeded" : null;
            await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message, code);
        }
        catch (NotSupportedException ex)
        {
            await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message);
        }
    }

    /// <summary>
    /// The model a retrieval request names: a directly registered one, else a model of the server's host (leased until the response is
    /// written), within the request's aliases when its context has them. A single directly registered model answers to any name.
    /// </summary>
    private static async Task<(T Model, IDisposable? Lease, string Name)> PickAsync<T>(HttpContext http, Dictionary<string, T> all, string? requested,
        string? fallback, string kind, ModelKind hostKind, Func<LoadedModel, T?> extract)
        where T : class
    {
        var ctx = http.RequestServices.GetService<SharplightServerContext>();
        bool anthropic = ApiErrors.IsAnthropic(http);
        if (SharplightServerContext.Aliases(http, out var defaultModel) is { } aliases)
        {
            string? alias = requested ?? defaultModel ?? aliases.FirstOrDefault(entry => all.ContainsKey(entry.Value) || ctx?.Host.Get(entry.Value)?.Kind == hostKind).Key;
            if (alias is null || !aliases.TryGetValue(alias, out var target))
            {
                throw ApiException.ModelNotFound($"The {kind} model '{alias}' does not exist.", anthropic);
            }

            if (all.TryGetValue(target, out var registered))
            {
                await SharplightServerContext.AdmitAsync(http, alias);
                return (registered, null, alias);
            }

            return await FromHostAsync(http, ctx, alias, hostKind, kind, extract);
        }

        if (requested is not null && all.TryGetValue(requested, out var named))
        {
            await SharplightServerContext.AdmitAsync(http, requested);
            return (named, null, requested);
        }

        if (ctx is not null && (requested is null ? all.Count == 0 && ctx.DefaultName(hostKind) is not null : ctx.Host.Contains(requested) && ctx.IsServed(requested)))
        {
            return await FromHostAsync(http, ctx, requested, hostKind, kind, extract);
        }

        // A single model answers to any name (clients often hard-code e.g. "text-embedding-3-small").
        if (all.Count > 0 && (requested is null || all.Count == 1))
        {
            var name = fallback ?? all.Keys.First();
            await SharplightServerContext.AdmitAsync(http, requested ?? name);
            return (all[name], null, requested ?? name);
        }

        throw ApiException.ModelNotFound($"The {kind} model '{requested}' does not exist.", anthropic);
    }

    private static async Task<(T Model, IDisposable? Lease, string Name)> FromHostAsync<T>(HttpContext http, SharplightServerContext? ctx, string? name,
        ModelKind hostKind, string kind, Func<LoadedModel, T?> extract)
        where T : class
    {
        if (ctx is null)
        {
            throw ApiException.ModelNotFound($"The {kind} model '{name}' does not exist.", ApiErrors.IsAnthropic(http));
        }

        var lease = await ctx.AcquireAsync(http, name, hostKind);
        var model = extract(lease.As<LoadedModel>());
        if (model is null)
        {
            lease.Dispose();
            throw ApiException.ModelNotFound($"The model '{lease.Name}' is not a {kind} model.", ApiErrors.IsAnthropic(http));
        }

        return (model, lease, SharplightServerContext.ResponseName(http, lease));
    }

    // ---- /v1/embeddings -----------------------------------------------------------------------------------------------

    private static async Task Embeddings(HttpContext http, RetrievalModels models)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, models.MaxRequestBodyBytes);
        var b = doc.RootElement;
        string? requested = b.Str("model");
        var (model, lease, name) = await PickAsync(http, models.Embedders, requested, models.DefaultEmbedder, "embedding", ModelKind.Embedding,
            loaded => (loaded as ITaskClientModel)?.FindClient<ITextEmbedder>());
        using var held = lease;

        string format = b.Str("encoding_format") ?? "float";
        if (format is not ("float" or "base64"))
        {
            throw ApiException.BadRequest("'encoding_format' must be \"float\" or \"base64\".", "encoding_format");
        }

        int? dims = b.Int("dimensions");
        if (dims is int d && (d < 1 || d > model.Dimensions))
        {
            throw ApiException.BadRequest($"'dimensions' must be between 1 and {model.Dimensions}.", "dimensions");
        }

        string? prompt = ResolvePrompt(b, model);
        var options = new TextEmbeddingOptions { Prompt = prompt, Dimensions = dims, Truncate = b.Bool("truncate") ?? models.TruncateByDefault };

        var (texts, tokens) = ParseInput(b, models.MaxInputs);
        var result = texts is not null
            ? await model.EmbedAsync(texts, options, http.RequestAborted)
            : await model.EmbedTokensAsync(tokens!, options, http.RequestAborted);
        SharplightRequestFeature.Of(http)?.AddInput(result.TotalTokens);

        var data = EmbeddingData(result.Vectors, format == "base64");
        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["object"] = "list",
            ["data"] = data,
            ["model"] = name,
            ["usage"] = new JsonObject { ["prompt_tokens"] = result.TotalTokens, ["total_tokens"] = result.TotalTokens },
        });
    }

    /// <summary>
    /// The model prompt to prepend: <c>prompt_name</c> when given, else the prompt named like <c>input_type</c> (Cohere / Jina /
    /// Voyage style) when the model has one, else none (embed as is).
    /// </summary>
    private static string? ResolvePrompt(JsonElement body, ITextEmbedder model)
    {
        string? prompt = body.Str("prompt_name");
        if (prompt is not null)
        {
            return model.Prompts.ContainsKey(prompt)
                ? prompt
                : throw ApiException.BadRequest($"The model has no prompt named '{prompt}' (available: {string.Join(", ", model.Prompts.Keys)}).", "prompt_name");
        }

        if (body.Str("input_type") is not { } inputType)
        {
            return null;
        }

        string name = inputType.Replace("search_", "").Replace("retrieval.", "").Replace("passage", "document");
        return model.Prompts.ContainsKey(name) ? name : null;
    }

    private static JsonArray EmbeddingData(IReadOnlyList<float[]> vectors, bool base64)
    {
        var data = new JsonArray();
        for (int i = 0; i < vectors.Count; i++)
        {
            var vector = vectors[i];
            JsonNode embedding = base64
                ? JsonValue.Create(Convert.ToBase64String(MemoryMarshal.AsBytes(vector.AsSpan())))
                : new JsonArray([.. vector.Select(x => (JsonNode)JsonValue.Create(x))]);
            data.Add(new JsonObject { ["object"] = "embedding", ["index"] = i, ["embedding"] = embedding });
        }

        return data;
    }

    /// <summary>OpenAI's <c>input</c>: a string, an array of strings, an array of token ids or an array of token-id arrays.</summary>
    private static (List<string>? Texts, List<IReadOnlyList<int>>? Tokens) ParseInput(JsonElement b, int maxInputs)
    {
        const string Shape = "'input' must be a string, an array of strings, an array of token ids or an array of token-id arrays.";
        if (b.Get("input") is not { } input)
        {
            throw ApiException.BadRequest("'input' is required.", "input");
        }

        if (input.ValueKind == JsonValueKind.String)
        {
            return ([input.GetString()!], null);
        }

        if (input.ValueKind != JsonValueKind.Array)
        {
            throw ApiException.BadRequest(Shape, "input");
        }

        int n = input.GetArrayLength();
        if (n == 0)
        {
            throw ApiException.BadRequest("'input' must not be empty.", "input");
        }

        var first = input[0];
        if (first.ValueKind == JsonValueKind.Number)
        {
            return (null, [Ids(input)]);
        }

        if (n > maxInputs)
        {
            throw ApiException.BadRequest($"'input' has {n} items; at most {maxInputs} are accepted.", "input");
        }

        if (first.ValueKind == JsonValueKind.String)
        {
            return ([.. input.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String ? e.GetString()! : throw ApiException.BadRequest(Shape, "input"))], null);
        }

        if (first.ValueKind == JsonValueKind.Array)
        {
            return (null, [.. input.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Array ? Ids(e) : throw ApiException.BadRequest(Shape, "input"))]);
        }

        throw ApiException.BadRequest(Shape, "input");

        static IReadOnlyList<int> Ids(JsonElement a)
        {
            if (a.GetArrayLength() == 0)
            {
                throw ApiException.BadRequest("A token-id input must not be empty.", "input");
            }

            return [.. a.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int id) ? id : throw ApiException.BadRequest(Shape, "input"))];
        }
    }

    /// <summary>Cohere v1 / Jina accept <c>{ "text": ... }</c> objects as documents next to plain strings.</summary>
    private static List<string> ParseDocuments(JsonElement documents)
    {
        return [.. documents.EnumerateArray().Select(e => e.ValueKind switch
        {
            JsonValueKind.String => e.GetString()!,
            JsonValueKind.Object when e.Str("text") is { } text => text,
            _ => throw ApiException.BadRequest("Each document must be a string or an object with a \"text\" string.", "documents"),
        })];
    }

    /// <summary>The <paramref name="topN"/> best documents, highest score first (ties keep input order).</summary>
    private static JsonArray RankedResults(IReadOnlyList<float> scores, List<string> docs, int topN, bool returnDocuments)
    {
        var order = Enumerable.Range(0, docs.Count).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(topN);
        var results = new JsonArray();
        foreach (int i in order)
        {
            var result = new JsonObject { ["index"] = i, ["relevance_score"] = scores[i] };
            if (returnDocuments)
            {
                result["document"] = new JsonObject { ["text"] = docs[i] };
            }

            results.Add(result);
        }

        return results;
    }

    // ---- /v1/rerank ---------------------------------------------------------------------------------------------------

    private static async Task Rerank(HttpContext http, RetrievalModels models)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, models.MaxRequestBodyBytes);
        var b = doc.RootElement;
        string? requested = b.Str("model");
        var (model, lease, name) = await PickAsync(http, models.Rerankers, requested, models.DefaultReranker, "rerank", ModelKind.Reranker,
            loaded => (loaded as ITaskClientModel)?.FindClient<ITextReranker>());
        using var held = lease;
        string query = b.Str("query") ?? throw ApiException.BadRequest("'query' is required and must be a string.", "query");
        if (b.Get("documents") is not { ValueKind: JsonValueKind.Array } docsEl)
        {
            throw ApiException.BadRequest("'documents' is required and must be an array.", "documents");
        }
        var docs = ParseDocuments(docsEl);
        if (docs.Count > models.MaxInputs)
        {
            throw ApiException.BadRequest($"'documents' has {docs.Count} items; at most {models.MaxInputs} are accepted.", "documents");
        }

        int topN = b.Int("top_n") ?? docs.Count;
        if (topN < 1)
        {
            throw ApiException.BadRequest("'top_n' must be at least 1.", "top_n");
        }

        bool returnDocuments = b.Bool("return_documents") ?? false;
        var options = new RerankOptions { Instruction = b.Str("instruction"), Truncate = b.Bool("truncate") ?? true };

        var result = docs.Count == 0 ? new RerankResult([], [], []) : await model.RerankAsync(query, docs, options, http.RequestAborted);
        SharplightRequestFeature.Of(http)?.AddInput(result.TotalTokens);
        var results = RankedResults(result.Scores, docs, topN, returnDocuments);
        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["id"] = Guid.NewGuid().ToString(),
            ["model"] = name,
            ["results"] = results,
            ["usage"] = new JsonObject { ["prompt_tokens"] = result.TotalTokens, ["total_tokens"] = result.TotalTokens },
            ["meta"] = new JsonObject
            {
                ["billed_units"] = new JsonObject { ["search_units"] = 1 },
                ["tokens"] = new JsonObject { ["input_tokens"] = result.TotalTokens, ["output_tokens"] = 0 },
            },
        });
    }
}
