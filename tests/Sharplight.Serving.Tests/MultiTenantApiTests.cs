using System.Diagnostics.Metrics;
using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Hosting.Loaders;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

/// <summary>
/// The multi-tenant seams: two route groups over one model host, each with its own keys (scope), visible models (aliases), defaults,
/// allowed tools and admission; usage and metrics tagged with the scope; prefix-cache isolation between scopes.
/// </summary>
public sealed class MultiTenantApiTests : IAsyncLifetime
{
    private readonly RecordingUsageObserver _usage = new();
    private readonly ScopeAdmission _admission = new("blocked");
    private FakeLoader _a = null!;
    private FakeLoader _b = null!;
    private ModelHost _host = null!;
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public async ValueTask InitializeAsync()
    {
        var pool = new FakePool("gpu", 10L << 30);
        _a = new FakeLoader(pool, 1L << 30) { Reply = "I am A" };
        _b = new FakeLoader(pool, 1L << 30) { Reply = "I am B" };
        _host = new ModelHost(new ModelHostOptions { ReserveBytes = 0 });
        _host.Register(new ModelRegistration { Name = "model-a", Loader = _a });
        _host.Register(new ModelRegistration { Name = "model-b", Loader = _b });
        _host.Register(new ModelRegistration
        {
            Name = "embed-x",
            Loader = new FakeInstanceLoader(ModelKind.Embedding, () => new EmbeddingModelInstance(new FakeEmbedder(), 0)),
        });

        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSharplightServer(_host, o => o.ApiKeys.Add("root-key"));
        builder.Services.AddSingleton<ISharplightUsageObserver>(_usage);
        _app = builder.Build();
        _app.UseSharplightServer();
        _app.MapSharplightServer();
        _app.MapGroup("/t1").MapSharplightServer(o =>
        {
            o.Operations = false;
            o.Admin = false;
            o.KeyValidator = new ScopeKeyValidator("scope-1", "key-1");
            o.Resolver = new ScopeResolver(new SharplightRequestContext
            {
                Models = new Dictionary<string, string> { ["chat"] = "model-a", ["embed"] = "embed-x" },
                Defaults = new SharplightRequestDefaults { MaxTokensLimit = 3 },
            });
        });
        _app.MapGroup("/t2").MapSharplightServer(o =>
        {
            o.Operations = false;
            o.Admin = false;
            o.KeyValidator = new ScopeKeyValidator("scope-2", "key-2");
            o.Resolver = new ScopeResolver(new SharplightRequestContext
            {
                Models = new Dictionary<string, string> { ["chat"] = "model-b", ["same"] = "model-a" },
                AllowedTools = new HashSet<string>(["lookup"]),
            });
            o.Admission = _admission;
        });
        _app.MapGroup("/t3").MapSharplightServer(o =>
        {
            o.Operations = false;
            o.Admin = false;
            o.KeyValidator = new ScopeKeyValidator("blocked", "key-3");
            o.Admission = _admission;
        });
        _app.MapGroup("/t4").MapSharplightServer(o =>
        {
            o.Operations = false;
            o.Admin = false;
            o.KeyValidator = new ScopeKeyValidator("scope-4", "key-4");
            o.Resolver = new ScopeResolver(new SharplightRequestContext
            {
                Models = new Dictionary<string, string> { ["short"] = "model-a", ["long"] = "model-a" },
                Defaults = new SharplightRequestDefaults { MaxTokensLimit = 4 },
                ModelDefaults = new Dictionary<string, SharplightRequestDefaults>
                {
                    ["short"] = new() { MaxTokensLimit = 2, SystemPrompt = "Answer in one word." },
                },
            });
        });
        await _app.StartAsync(Ct);
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        _client.Dispose();
        await _app.StopAsync(Ct);
        await _app.DisposeAsync();
        _host.Dispose();
    }

    private async Task<(HttpStatusCode Status, JsonElement Body, string Text)> Send(HttpMethod method, string path, string? json, string? key, params (string, string)[] headers)
    {
        var request = new HttpRequestMessage(method, path);
        if (json is not null)
        {
            request.Content = new StringContent(json, Encoding.UTF8, "application/json");
        }

        if (key is not null)
        {
            request.Headers.TryAddWithoutValidation("Authorization", "Bearer " + key);
        }

        foreach (var (name, value) in headers)
        {
            request.Headers.TryAddWithoutValidation(name, value);
        }

        using var response = await _client.SendAsync(request, Ct);
        var text = await response.Content.ReadAsStringAsync(Ct);
        var body = text.StartsWith('{') ? JsonDocument.Parse(text).RootElement.Clone() : default;
        return (response.StatusCode, body, text);
    }

    private static string Chat(string model, string content = "hi", bool stream = false, string extra = "")
        => $$"""{"model":"{{model}}","messages":[{"role":"user","content":"{{content}}"}],"temperature":0,"stream":{{(stream ? "true" : "false")}}{{extra}}}""";

    private static string Content(JsonElement body) => body.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;

    [Fact]
    public async Task AKeyOfOneScopeIsRejectedInAnother()
    {
        Assert.Equal(HttpStatusCode.OK, (await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat"), "key-1")).Status);
        Assert.Equal(HttpStatusCode.Unauthorized, (await Send(HttpMethod.Post, "/t2/v1/chat/completions", Chat("chat"), "key-1")).Status);
        Assert.Equal(HttpStatusCode.Unauthorized, (await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat"), "root-key")).Status);
        Assert.Equal(HttpStatusCode.Unauthorized, (await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat"), null)).Status);
        // The root mapping keeps the static key list.
        Assert.Equal(HttpStatusCode.OK, (await Send(HttpMethod.Post, "/v1/chat/completions", Chat("model-b"), "root-key")).Status);
        Assert.Equal(HttpStatusCode.Unauthorized, (await Send(HttpMethod.Post, "/v1/chat/completions", Chat("model-b"), "key-1")).Status);
    }

    [Fact]
    public async Task EachGroupSeesItsOwnModelsUnderItsAliases()
    {
        var (_, one, _) = await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat"), "key-1");
        var (_, two, _) = await Send(HttpMethod.Post, "/t2/v1/chat/completions", Chat("chat"), "key-2");
        Assert.StartsWith("I a", Content(one));   // cut to the group's 3-token limit
        Assert.Equal("chat", one.GetProperty("model").GetString());
        Assert.Equal("I am B", Content(two));

        var (status, error, _) = await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("model-b"), "key-1");
        Assert.Equal(HttpStatusCode.NotFound, status);
        Assert.Equal("model_not_found", error.GetProperty("error").GetProperty("code").GetString());
        Assert.Equal(HttpStatusCode.NotFound, (await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("same"), "key-1")).Status);

        var (_, list, _) = await Send(HttpMethod.Get, "/t1/v1/models", null, "key-1");
        Assert.Equal(["chat", "embed"], list.GetProperty("data").EnumerateArray().Select(model => model.GetProperty("id").GetString()));
        Assert.Equal(HttpStatusCode.NotFound, (await Send(HttpMethod.Get, "/t1/v1/models/model-a", null, "key-1")).Status);
        Assert.Equal(HttpStatusCode.OK, (await Send(HttpMethod.Get, "/t1/v1/models/chat", null, "key-1")).Status);

        var (anthropicStatus, message, _) = await Send(HttpMethod.Post, "/t2/v1/messages",
            """{"model":"chat","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}""", "key-2", ("anthropic-version", "2023-06-01"));
        Assert.Equal(HttpStatusCode.OK, anthropicStatus);
        Assert.Equal("I am B", message.GetProperty("content")[0].GetProperty("text").GetString());
        Assert.Equal("chat", message.GetProperty("model").GetString());
        var (missing, anthropicError, _) = await Send(HttpMethod.Post, "/t2/v1/messages",
            """{"model":"model-a","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}""", "key-2", ("anthropic-version", "2023-06-01"));
        Assert.Equal(HttpStatusCode.NotFound, missing);
        Assert.Equal("not_found_error", anthropicError.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task DefaultsLimitsAndToolsComeFromTheResolver()
    {
        var (_, limited, _) = await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat", extra: ",\"max_tokens\":50"), "key-1");
        Assert.Equal(3, limited.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        Assert.Equal("length", limited.GetProperty("choices")[0].GetProperty("finish_reason").GetString());

        const string Tool = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"NAME\",\"parameters\":{\"type\":\"object\"}}}]";
        Assert.Equal(HttpStatusCode.OK, (await Send(HttpMethod.Post, "/t2/v1/chat/completions", Chat("chat", extra: Tool.Replace("NAME", "lookup")), "key-2")).Status);
        var (status, error, _) = await Send(HttpMethod.Post, "/t2/v1/chat/completions", Chat("chat", extra: Tool.Replace("NAME", "shell")), "key-2");
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal("tool_not_allowed", error.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task UsageIsReportedWithTheScopeForStreamingAndNonStreaming()
    {
        await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat"), "key-1");
        var (_, _, stream) = await Send(HttpMethod.Post, "/t2/v1/chat/completions", Chat("chat", stream: true), "key-2");
        Assert.Contains("[DONE]", stream, StringComparison.Ordinal);

        var usages = _usage.Usages;
        var one = Assert.Single(usages, usage => usage.Scope == "scope-1");
        Assert.Equal("chat", one.Model);
        Assert.Equal("user:key-1", one.Principal);
        Assert.Equal(SharplightApiNames.OpenAiChatCompletions, one.Api);
        Assert.Equal("/t1/v1/chat/completions", one.Route);
        Assert.False(one.Streaming);
        Assert.True(one.InputTokens > 0);
        Assert.Equal(3, one.OutputTokens);
        Assert.Equal(200, one.StatusCode);

        var two = Assert.Single(usages, usage => usage.Scope == "scope-2");
        Assert.True(two.Streaming);
        Assert.Equal("I am B".Length, two.OutputTokens);
        Assert.NotNull(two.TimeToFirstToken);
        Assert.Equal(0, _admission.Held);   // the concurrency slot was released after the response
        Assert.Equal(1, _admission.Admitted);
    }

    [Fact]
    public async Task AdmissionCanRefuseAScope()
    {
        var (status, error, _) = await Send(HttpMethod.Post, "/t3/v1/chat/completions", Chat("model-a"), "key-3");
        Assert.Equal(HttpStatusCode.TooManyRequests, status);
        Assert.Equal("rate_limit_exceeded", error.GetProperty("error").GetProperty("code").GetString());
        Assert.Single(_usage.Usages, usage => usage.Scope == "blocked" && usage.StatusCode == 429);
    }

    [Fact]
    public async Task IdenticalPromptsOfTwoScopesDoNotShareThePrefixCache()
    {
        string prompt = "a long shared system prompt and question";
        var (_, first, _) = await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat", prompt), "key-1");
        var (_, otherScope, _) = await Send(HttpMethod.Post, "/t2/v1/chat/completions", Chat("same", prompt), "key-2");
        var (_, sameScope, _) = await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat", prompt), "key-1");
        static int Cached(JsonElement body) => body.GetProperty("usage").GetProperty("prompt_tokens_details").GetProperty("cached_tokens").GetInt32();
        Assert.Equal("I am A", Content(otherScope));   // the same model ...
        Assert.Equal(0, Cached(first));
        Assert.Equal(0, Cached(otherScope));            // ... but no cache hit from the other scope
        Assert.True(Cached(sameScope) > 0);
    }

    [Fact]
    public async Task AClientCacheSaltIsRefusedUnlessTheServerAllowsIt()
    {
        // A client could otherwise believe its prefixes are isolated by a salt the server ignores.
        var (status, body, _) = await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat", "hi", extra: ",\"cache_salt\":\"x\""), "key-1");
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal("cache_salt", body.GetProperty("error").GetProperty("param").GetString());
    }

    [Fact]
    public async Task EmbeddingsComeFromTheModelHostUnderTheAlias()
    {
        var (status, body, text) = await Send(HttpMethod.Post, "/t1/v1/embeddings", """{"model":"embed","input":["hello"]}""", "key-1");
        Assert.True(status == HttpStatusCode.OK, text);
        Assert.Equal("embed", body.GetProperty("model").GetString());
        Assert.Equal(8, body.GetProperty("data")[0].GetProperty("embedding").GetArrayLength());
        Assert.Equal(HttpStatusCode.NotFound, (await Send(HttpMethod.Post, "/t1/v1/embeddings", """{"model":"embed-x","input":["hello"]}""", "key-1")).Status);
        Assert.Contains(_usage.Usages, usage => usage.Api == "openai.embeddings" && usage.Route == "/t1/v1/embeddings" && usage.InputTokens == 5 && usage.Model == "embed");
    }

    [Fact]
    public async Task MetricsCarryTheScope()
    {
        var tokens = new List<(long Value, string? Scope, string? Type)>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (instrument, l) =>
        {
            if (instrument.Meter.Name == SharplightServerMetrics.MeterName && instrument.Name == "sharplight.server.tokens")
            {
                l.EnableMeasurementEvents(instrument);
            }
        };
        listener.SetMeasurementEventCallback<long>((_, value, tags, _) =>
        {
            string? scope = null;
            string? type = null;
            foreach (var tag in tags)
            {
                scope = tag.Key == "sharplight.scope" ? tag.Value as string : scope;
                type = tag.Key == "sharplight.token.type" ? tag.Value as string : type;
            }

            lock (tokens)
            {
                tokens.Add((value, scope, type));
            }
        });
        listener.Start();

        await Send(HttpMethod.Post, "/t1/v1/chat/completions", Chat("chat"), "key-1");
        lock (tokens)
        {
            Assert.Contains(tokens, entry => entry.Scope == "scope-1" && entry.Type == "output" && entry.Value == 3);
            Assert.Contains(tokens, entry => entry.Scope == "scope-1" && entry.Type == "input" && entry.Value > 0);
        }
    }

    [Fact]
    public async Task PerModelDefaultsApplyByAlias()
    {
        var (_, longer, _) = await Send(HttpMethod.Post, "/t4/v1/chat/completions", Chat("long"), "key-4");
        var (_, shorter, _) = await Send(HttpMethod.Post, "/t4/v1/chat/completions", Chat("short"), "key-4");
        static int Usage(JsonElement body, string name) => body.GetProperty("usage").GetProperty(name).GetInt32();
        Assert.Equal(4, Usage(longer, "completion_tokens"));
        Assert.Equal(2, Usage(shorter, "completion_tokens"));
        Assert.True(Usage(shorter, "prompt_tokens") > Usage(longer, "prompt_tokens"));   // the default system prompt

        var (_, message, _) = await Send(HttpMethod.Post, "/t4/v1/messages",
            """{"model":"short","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}""", "key-4", ("anthropic-version", "2023-06-01"));
        Assert.Equal(2, message.GetProperty("usage").GetProperty("output_tokens").GetInt32());
        var (_, own, _) = await Send(HttpMethod.Post, "/t4/v1/chat/completions",
            """{"model":"short","messages":[{"role":"system","content":"x"},{"role":"user","content":"hi"}],"temperature":0}""", "key-4");
        Assert.True(Usage(own, "prompt_tokens") < Usage(shorter, "prompt_tokens"));   // a system message of its own wins
    }
}
