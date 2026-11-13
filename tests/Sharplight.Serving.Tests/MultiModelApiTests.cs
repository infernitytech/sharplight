using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Server;

namespace Sharplight.Serving.Tests;

/// <summary>Routing, listing and admin endpoints over a host with two (lazily loaded) models.</summary>
public sealed class MultiModelApiTests : IAsyncLifetime
{
    private FakePool _pool = null!;
    private FakeLoader _a = null!, _b = null!;
    private ModelHost _host = null!;
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async ValueTask InitializeAsync() => await Start(o => { });

    private async Task Start(Action<SharplightServerOptions> configure)
    {
        _pool = new FakePool("gpu", 10L << 30);
        _a = new FakeLoader(_pool, 1L << 30) { Reply = "I am A" };
        _b = new FakeLoader(_pool, 1L << 30) { Reply = "I am B" };
        _host = new ModelHost(new ModelHostOptions { ReserveBytes = 0 });
        _host.Register(new ModelRegistration { Name = "model-a", Loader = _a });
        _host.Register(new ModelRegistration { Name = "model-b", Loader = _b, Description = "second" });
        var b = WebApplication.CreateBuilder();
        b.WebHost.UseTestServer();
        b.Logging.ClearProviders();
        b.Services.AddSharplight(_host, configure);
        _app = b.Build();
        _app.UseSharplight();
        _app.MapSharplight();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        _client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        _host.Dispose();
    }

    private async Task<(HttpStatusCode, JsonElement)> Post(string path, string json, params (string, string)[] headers)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        foreach (var (k, v) in headers)
        {
            req.Headers.TryAddWithoutValidation(k, v);
        }

        using var r = await _client.SendAsync(req, TestContext.Current.CancellationToken);
        var text = await r.Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        return (r.StatusCode, text.Length == 0 ? default : JsonDocument.Parse(text).RootElement.Clone());
    }

    private string Chat(string model) => $$"""{"model":"{{model}}","messages":[{"role":"user","content":"hi"}],"temperature":0}""";

    [Fact]
    public async Task ModelField_SelectsTheModel_AndLoadsItLazily()
    {
        var (s1, b1) = await Post("/v1/chat/completions", Chat("model-b"));
        Assert.Equal(HttpStatusCode.OK, s1);
        Assert.Equal("I am B", b1.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("model-b", b1.GetProperty("model").GetString());
        Assert.Equal(0, _a.Loads);   // A was never touched
        var (_, b2) = await Post("/v1/chat/completions", Chat("model-a"));
        Assert.Equal("I am A", b2.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());

        var (s3, b3) = await Post("/v1/messages", """{"model":"model-a","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}""", ("anthropic-version", "2023-06-01"));
        Assert.Equal(HttpStatusCode.OK, s3);
        Assert.Equal("I am A", b3.GetProperty("content")[0].GetProperty("text").GetString());
        var (s4, b4) = await Post("/v1/completions", """{"model":"model-b","prompt":[120,1],"max_tokens":10}""");
        Assert.Equal(HttpStatusCode.OK, s4);
        Assert.Equal("I am B", b4.GetProperty("choices")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task UnknownModel_Is404_InEachApiShape()
    {
        var (s, b) = await Post("/v1/chat/completions", Chat("gpt-4"));
        Assert.Equal(HttpStatusCode.NotFound, s);
        Assert.Equal("model_not_found", b.GetProperty("error").GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", b.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("gpt-4", b.GetProperty("error").GetProperty("message").GetString());

        var (s2, b2) = await Post("/v1/messages", """{"model":"claude-x","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}""", ("anthropic-version", "2023-06-01"));
        Assert.Equal(HttpStatusCode.NotFound, s2);
        Assert.Equal("error", b2.GetProperty("type").GetString());
        Assert.Equal("not_found_error", b2.GetProperty("error").GetProperty("type").GetString());

        var (s3, _) = await Post("/v1/completions", """{"model":"nope","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, s3);
        using var g = await _client.GetAsync("/v1/models/nope", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.NotFound, g.StatusCode);
        Assert.Equal(0, _a.Prepares + _b.Prepares);
    }

    [Fact]
    public async Task NoModelField_UsesTheFirstRegisteredModel()
    {
        var (s, b) = await Post("/v1/chat/completions", """{"messages":[{"role":"user","content":"hi"}]}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal("model-a", b.GetProperty("model").GetString());
    }

    [Fact]
    public async Task ModelList_ShowsAllModels_LoadedOrNot()
    {
        await Post("/v1/chat/completions", Chat("model-b"));
        var list = JsonDocument.Parse(await _client.GetStringAsync("/v1/models", TestContext.Current.CancellationToken)).RootElement.GetProperty("data");
        Assert.Equal(["model-a", "model-b"], list.EnumerateArray().Select(m => m.GetProperty("id").GetString()).ToArray());
        Assert.Equal("unloaded", list[0].GetProperty("sharplight").GetProperty("state").GetString());
        Assert.Equal("loaded", list[1].GetProperty("sharplight").GetProperty("state").GetString());
        Assert.Equal("chat", list[1].GetProperty("sharplight").GetProperty("kind").GetString());

        var req = new HttpRequestMessage(HttpMethod.Get, "/v1/models");
        req.Headers.Add("anthropic-version", "2023-06-01");
        using var r = await _client.SendAsync(req, TestContext.Current.CancellationToken);
        var an = JsonDocument.Parse(await r.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement;
        Assert.Equal(2, an.GetProperty("data").GetArrayLength());
        Assert.Equal("model-a", an.GetProperty("first_id").GetString());
        Assert.Equal("model-b", an.GetProperty("last_id").GetString());
    }

    [Fact]
    public async Task Admin_LoadUnload_AndMemoryReport()
    {
        var (s, b) = await Post("/admin/models/model-a/load", "");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal("loaded", b.GetProperty("state").GetString());
        Assert.Equal(1L << 30, _pool.Ours);
        var report = JsonDocument.Parse(await _client.GetStringAsync("/admin/models", TestContext.Current.CancellationToken)).RootElement;
        Assert.Equal(2, report.GetProperty("models").GetArrayLength());
        Assert.Equal(1L << 30, report.GetProperty("devices")[0].GetProperty("used_by_sharplight_bytes").GetInt64());

        var (s2, b2) = await Post("/admin/models/model-a/unload", "");
        Assert.Equal(HttpStatusCode.OK, s2);
        Assert.Equal("unloaded", b2.GetProperty("state").GetString());
        Assert.Equal(0, _pool.Ours);
        var (s3, _) = await Post("/admin/models/missing/load", "");
        Assert.Equal(HttpStatusCode.NotFound, s3);
    }

    [Fact]
    public async Task Admin_NeedsTheApiKey()
    {
        await DisposeAsync();
        await Start(o => o.ApiKeys.Add("secret"));
        var (s, _) = await Post("/admin/models/model-a/load", "");
        Assert.Equal(HttpStatusCode.Unauthorized, s);
        var (s2, _) = await Post("/admin/models/model-a/load", "", ("Authorization", "Bearer secret"));
        Assert.Equal(HttpStatusCode.OK, s2);
        using var health = await _client.GetAsync("/health", TestContext.Current.CancellationToken);   // stays open
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);
    }

    [Fact]
    public async Task OverBudget_Is503_WithTheReason()
    {
        _pool.External = 9_500L << 20;   // leaves < 1 GiB for a 1 GiB model
        var (s, b) = await Post("/v1/chat/completions", Chat("model-a"));
        Assert.Equal(HttpStatusCode.ServiceUnavailable, s);
        Assert.Equal("insufficient_memory", b.GetProperty("error").GetProperty("code").GetString());
        Assert.Contains("MiB", b.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task HealthAndMetrics_CoverEveryModel()
    {
        await Post("/v1/chat/completions", Chat("model-a"));
        await Post("/v1/chat/completions", Chat("model-b"));
        var h = JsonDocument.Parse(await _client.GetStringAsync("/health", TestContext.Current.CancellationToken)).RootElement;
        Assert.Equal("ok", h.GetProperty("status").GetString());
        Assert.Equal(2, h.GetProperty("models").GetArrayLength());
        var m = await _client.GetStringAsync("/metrics", TestContext.Current.CancellationToken);
        Assert.Contains("sharplight_requests_submitted_total{model=\"model-a\"} 1", m);
        Assert.Contains("sharplight_requests_submitted_total{model=\"model-b\"} 1", m);
        Assert.Equal(1, System.Text.RegularExpressions.Regex.Matches(m, "# TYPE sharplight_queue_depth ").Count);
        Assert.Contains("sharplight_model_loaded{model=\"model-a\",kind=\"chat\"} 1", m);
        Assert.Contains("sharplight_device_memory_bytes", m);
    }
}
