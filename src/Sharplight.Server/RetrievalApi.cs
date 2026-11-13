using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm.Embeddings;

namespace Sharplight.Server;

/// <summary>The embedding and reranking models a host serves, by the name clients pass as <c>model</c>.</summary>
public sealed class RetrievalModels
{
    public Dictionary<string, ITextEmbedder> Embedders { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, ITextReranker> Rerankers { get; } = new(StringComparer.Ordinal);
    /// <summary>Embedder used when a request names none (default: the first registered).</summary>
    public string? DefaultEmbedder { get; set; }
    /// <summary>Reranker used when a request names none (default: the first registered).</summary>
    public string? DefaultReranker { get; set; }
    /// <summary>
    /// Cut over-long inputs to the model's limit when the request does not say (<c>truncate</c>). Default false: like OpenAI's API,
    /// an over-long input is a 400 error. Rerankers always truncate (the document is shortened), as Cohere's API does.
    /// </summary>
    public bool TruncateByDefault { get; set; }
    /// <summary>Most inputs (texts or documents) in one request.</summary>
    public int MaxInputs { get; set; } = 2048;
    public long MaxRequestBodyBytes { get; set; } = 32 * 1024 * 1024;
}

/// <summary>
/// Retrieval endpoints: OpenAI-compatible <c>POST /v1/embeddings</c> and Cohere / Jina-compatible <c>POST /v1/rerank</c> (also
/// <c>/v2/rerank</c>, <c>/rerank</c>). Models are registered with <see cref="AddSharplightEmbeddings"/> /
/// <see cref="AddSharplightReranker"/>; <c>MapSharplight</c> maps these endpoints when any is registered, or call
/// <see cref="MapSharplightRetrieval"/> directly (e.g. for a host that serves only embeddings).
/// </summary>
public static class RetrievalApi
{
    /// <summary>Serves <paramref name="embedder"/> on <c>/v1/embeddings</c> under <paramref name="name"/> (default: its own name).</summary>
    public static IServiceCollection AddSharplightEmbeddings(this IServiceCollection services, ITextEmbedder embedder, string? name = null, Action<RetrievalModels>? configure = null)
    {
        var m = Models(services);
        m.Embedders[name ?? embedder.Name] = embedder;
        m.DefaultEmbedder ??= name ?? embedder.Name;
        configure?.Invoke(m);
        return services;
    }

    /// <summary>Serves <paramref name="reranker"/> on <c>/v1/rerank</c> under <paramref name="name"/> (default: its own name).</summary>
    public static IServiceCollection AddSharplightReranker(this IServiceCollection services, ITextReranker reranker, string? name = null, Action<RetrievalModels>? configure = null)
    {
        var m = Models(services);
        m.Rerankers[name ?? reranker.Name] = reranker;
        m.DefaultReranker ??= name ?? reranker.Name;
        configure?.Invoke(m);
        return services;
    }

    static RetrievalModels Models(IServiceCollection services)
    {
        if (services.FirstOrDefault(d => d.ServiceType == typeof(RetrievalModels))?.ImplementationInstance is RetrievalModels m) return m;
        m = new RetrievalModels();
        services.AddSingleton(m);
        return m;
    }

    /// <summary>Maps <c>/v1/embeddings</c> and the rerank routes for the registered models (nothing when none are registered).</summary>
    public static IEndpointRouteBuilder MapSharplightRetrieval(this IEndpointRouteBuilder routes)
    {
        if (routes.ServiceProvider.GetService<RetrievalModels>() is not { } models) return routes;
        if (models.Embedders.Count > 0) routes.MapPost("/v1/embeddings", http => Handle(http, () => Embeddings(http, models)));
        if (models.Rerankers.Count > 0)
            foreach (var path in new[] { "/v1/rerank", "/v2/rerank", "/rerank" })
                routes.MapPost(path, http => Handle(http, () => Rerank(http, models)));
        return routes;
    }

    /// <summary>Errors become API error bodies even when the host did not add <c>UseSharplight</c>.</summary>
    static async Task Handle(HttpContext http, Func<Task> body)
    {
        try { await body(); }
        catch (ApiException ex) { await ApiErrors.WriteAsync(http, ex.Status, ex.Type, ex.Message, ex.Code, ex.Param); }
        catch (ArgumentException ex) { await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message, ex.Message.Contains("at most") ? "context_length_exceeded" : null); }
        catch (NotSupportedException ex) { await ApiErrors.WriteAsync(http, 400, "invalid_request_error", ex.Message); }
    }

    static T Pick<T>(Dictionary<string, T> all, string? requested, string? fallback, string kind)
    {
        if (requested is not null && all.TryGetValue(requested, out var m)) return m;
        // A single model answers to any name (clients often hard-code e.g. "text-embedding-3-small").
        if (requested is null || all.Count == 1) return all[fallback ?? all.Keys.First()];
        throw new ApiException(404, "invalid_request_error", $"The {kind} model '{requested}' does not exist (available: {string.Join(", ", all.Keys)}).", "model_not_found", "model");
    }

    // ---- /v1/embeddings -----------------------------------------------------------------------------------------------

    static async Task Embeddings(HttpContext http, RetrievalModels models)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, models.MaxRequestBodyBytes);
        var b = doc.RootElement;
        string? requested = b.Str("model");
        var model = Pick(models.Embedders, requested, models.DefaultEmbedder, "embedding");

        string format = b.Str("encoding_format") ?? "float";
        if (format is not ("float" or "base64")) throw ApiException.BadRequest("'encoding_format' must be \"float\" or \"base64\".", "encoding_format");
        int? dims = b.Int("dimensions");
        if (dims is int d && (d < 1 || d > model.Dimensions)) throw ApiException.BadRequest($"'dimensions' must be between 1 and {model.Dimensions}.", "dimensions");
        string? prompt = b.Str("prompt_name");
        if (prompt is not null && !model.Prompts.ContainsKey(prompt))
            throw ApiException.BadRequest($"The model has no prompt named '{prompt}' (available: {string.Join(", ", model.Prompts.Keys)}).", "prompt_name");
        // input_type (Cohere / Jina / Voyage style): use the model's prompt of that name when it has one, else embed as is.
        if (prompt is null && b.Str("input_type") is { } type)
        {
            string t = type.Replace("search_", "").Replace("retrieval.", "").Replace("passage", "document");
            if (model.Prompts.ContainsKey(t)) prompt = t;
        }
        var options = new EmbedOptions { Prompt = prompt, Dimensions = dims, Truncate = b.Bool("truncate") ?? models.TruncateByDefault };

        var (texts, tokens) = ParseInput(b, models.MaxInputs);
        var result = await Task.Run(() => texts is not null ? model.Embed(texts, options) : model.EmbedTokens(tokens!, options), http.RequestAborted);

        var data = new JsonArray();
        for (int i = 0; i < result.Vectors.Count; i++)
        {
            var v = result.Vectors[i];
            JsonNode embedding = format == "base64"
                ? JsonValue.Create(Convert.ToBase64String(MemoryMarshal.AsBytes(v.AsSpan())))
                : new JsonArray([.. v.Select(x => (JsonNode)JsonValue.Create(x))]);
            data.Add(new JsonObject { ["object"] = "embedding", ["index"] = i, ["embedding"] = embedding });
        }
        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["object"] = "list",
            ["data"] = data,
            ["model"] = requested ?? model.Name,
            ["usage"] = new JsonObject { ["prompt_tokens"] = result.TotalTokens, ["total_tokens"] = result.TotalTokens },
        });
    }

    /// <summary>OpenAI's <c>input</c>: a string, an array of strings, an array of token ids or an array of token-id arrays.</summary>
    static (List<string>? Texts, List<IReadOnlyList<int>>? Tokens) ParseInput(JsonElement b, int maxInputs)
    {
        const string Shape = "'input' must be a string, an array of strings, an array of token ids or an array of token-id arrays.";
        if (b.Get("input") is not { } input) throw ApiException.BadRequest("'input' is required.", "input");
        if (input.ValueKind == JsonValueKind.String) return ([input.GetString()!], null);
        if (input.ValueKind != JsonValueKind.Array) throw ApiException.BadRequest(Shape, "input");
        int n = input.GetArrayLength();
        if (n == 0) throw ApiException.BadRequest("'input' must not be empty.", "input");
        var first = input[0];
        if (first.ValueKind == JsonValueKind.Number) return (null, [Ids(input)]);
        if (n > maxInputs) throw ApiException.BadRequest($"'input' has {n} items; at most {maxInputs} are accepted.", "input");
        if (first.ValueKind == JsonValueKind.String)
            return ([.. input.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String ? e.GetString()! : throw ApiException.BadRequest(Shape, "input"))], null);
        if (first.ValueKind == JsonValueKind.Array)
            return (null, [.. input.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Array ? Ids(e) : throw ApiException.BadRequest(Shape, "input"))]);
        throw ApiException.BadRequest(Shape, "input");

        static IReadOnlyList<int> Ids(JsonElement a)
        {
            if (a.GetArrayLength() == 0) throw ApiException.BadRequest("A token-id input must not be empty.", "input");
            return [.. a.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int id) ? id : throw ApiException.BadRequest(Shape, "input"))];
        }
    }

    // ---- /v1/rerank ---------------------------------------------------------------------------------------------------

    static async Task Rerank(HttpContext http, RetrievalModels models)
    {
        using var doc = await HttpJson.ReadBodyAsync(http, models.MaxRequestBodyBytes);
        var b = doc.RootElement;
        string? requested = b.Str("model");
        var model = Pick(models.Rerankers, requested, models.DefaultReranker, "rerank");
        string query = b.Str("query") ?? throw ApiException.BadRequest("'query' is required and must be a string.", "query");
        if (b.Get("documents") is not { ValueKind: JsonValueKind.Array } docsEl) throw ApiException.BadRequest("'documents' is required and must be an array.", "documents");
        // Cohere v1 / Jina accept { "text": ... } objects as documents.
        var docs = docsEl.EnumerateArray().Select(e => e.ValueKind switch
        {
            JsonValueKind.String => e.GetString()!,
            JsonValueKind.Object when e.Str("text") is { } t => t,
            _ => throw ApiException.BadRequest("Each document must be a string or an object with a \"text\" string.", "documents"),
        }).ToList();
        if (docs.Count > models.MaxInputs) throw ApiException.BadRequest($"'documents' has {docs.Count} items; at most {models.MaxInputs} are accepted.", "documents");
        int topN = b.Int("top_n") ?? docs.Count;
        if (topN < 1) throw ApiException.BadRequest("'top_n' must be at least 1.", "top_n");
        bool returnDocuments = b.Bool("return_documents") ?? false;
        var options = new RerankOptions { Instruction = b.Str("instruction"), Truncate = b.Bool("truncate") ?? true };

        var result = docs.Count == 0 ? new RerankResult([], [], []) : await Task.Run(() => model.Score(query, docs, options), http.RequestAborted);
        var order = Enumerable.Range(0, docs.Count).OrderByDescending(i => result.Scores[i]).ThenBy(i => i).Take(topN);
        var results = new JsonArray();
        foreach (int i in order)
        {
            var r = new JsonObject { ["index"] = i, ["relevance_score"] = result.Scores[i] };
            if (returnDocuments) r["document"] = new JsonObject { ["text"] = docs[i] };
            results.Add(r);
        }
        await HttpJson.WriteAsync(http, new JsonObject
        {
            ["id"] = Guid.NewGuid().ToString(),
            ["model"] = requested ?? model.Name,
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
