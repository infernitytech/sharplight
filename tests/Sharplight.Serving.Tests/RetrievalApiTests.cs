using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Logging;
using Sharplight.Server;

namespace Sharplight.Serving.Tests;

public sealed class RetrievalApiTests : IAsyncLifetime
{
    private WebApplication _app = null!, _standalone = null!;
    private HttpClient _client = null!, _standaloneClient = null!;
    private readonly FakeEmbedder _embedder = new();
    private Sharplight.Llm.Serving.InferenceEngine? _engine;

    public async ValueTask InitializeAsync()
    {
        // Full server (chat + retrieval through MapSharplight) and a retrieval-only host (MapSharplightRetrieval).
        (_app, _client) = await Build(full: true);
        (_standalone, _standaloneClient) = await Build(full: false);
    }

    private async Task<(WebApplication, HttpClient)> Build(bool full)
    {
        var b = WebApplication.CreateBuilder();
        b.WebHost.UseTestServer();
        b.Logging.ClearProviders();
        if (full)
        {
            _engine = new Sharplight.Llm.Serving.InferenceEngine(new FakeBatchDecoder(1), new FakeTokenizer());
            b.Services.AddSharplight(new Sharplight.Llm.Serving.Chat.ModelService(_engine, ApiHost.Template, new() { ModelName = "fake-model" }));
        }
        b.Services.AddSharplightEmbeddings(_embedder).AddSharplightReranker(new FakeReranker());
        var app = b.Build();
        if (full)
        {
            app.UseSharplight();
            app.MapSharplight();
        }
        else
        {
            app.MapSharplightRetrieval();
        }

        await app.StartAsync();
        return (app, app.GetTestClient());
    }

    public async ValueTask DisposeAsync()
    {
        foreach (var a in new[] { _app, _standalone })
        {
            await a.StopAsync();
            await a.DisposeAsync();
        }
        _engine?.Dispose();
    }

    private async Task<(HttpStatusCode, JsonElement)> Post(string path, string json, bool standalone = false)
    {
        using var r = await (standalone ? _standaloneClient : _client).PostAsync(path, new StringContent(json, Encoding.UTF8, "application/json"));
        return (r.StatusCode, JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement.Clone());
    }

    [Fact]
    public async Task EmbeddingsOfStringAndArray()
    {
        var (s, body) = await Post("/v1/embeddings", """{"model":"fake-embed","input":["hello","world!"]}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal("list", body.GetProperty("object").GetString());
        Assert.Equal("fake-embed", body.GetProperty("model").GetString());
        var data = body.GetProperty("data");
        Assert.Equal(2, data.GetArrayLength());
        Assert.Equal(1, data[1].GetProperty("index").GetInt32());
        Assert.Equal("embedding", data[0].GetProperty("object").GetString());
        Assert.Equal(8, data[0].GetProperty("embedding").GetArrayLength());
        Assert.Equal(11, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(11, body.GetProperty("usage").GetProperty("total_tokens").GetInt32());

        (s, body) = await Post("/v1/embeddings", """{"input":"hello"}""");
        Assert.Equal(HttpStatusCode.OK, s);
        var expected = _embedder.Embed(["hello"]).Vectors[0];
        Assert.Equal(expected, body.GetProperty("data")[0].GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()));
    }

    [Fact]
    public async Task Base64EncodingIsLittleEndianFloat32()
    {
        var (s, body) = await Post("/v1/embeddings", """{"input":"hello","encoding_format":"base64"}""");
        Assert.Equal(HttpStatusCode.OK, s);
        var bytes = Convert.FromBase64String(body.GetProperty("data")[0].GetProperty("embedding").GetString()!);
        var floats = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        Assert.Equal(_embedder.Embed(["hello"]).Vectors[0], floats);
    }

    [Fact]
    public async Task TokenInputsDimensionsAndPrompts()
    {
        var (s, body) = await Post("/v1/embeddings", """{"input":[[104,105],[106]],"dimensions":4}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal(4, body.GetProperty("data")[0].GetProperty("embedding").GetArrayLength());
        Assert.Equal(3, body.GetProperty("usage").GetProperty("total_tokens").GetInt32());
        Assert.Equal(4, _embedder.Calls[^1].Dimensions);

        (s, body) = await Post("/v1/embeddings", """{"input":[104,105]}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal(1, body.GetProperty("data").GetArrayLength());

        (s, _) = await Post("/v1/embeddings", """{"input":"x","input_type":"search_query"}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal("query", _embedder.Calls[^1].Prompt);
        (s, _) = await Post("/v1/embeddings", """{"input":"x","input_type":"document"}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Null(_embedder.Calls[^1].Prompt);   // the model has no "document" prompt: embedded as is
    }

    [Theory]
    [InlineData("""{"input":[]}""", "input")]
    [InlineData("""{"input":[1,"a"]}""", "input")]
    [InlineData("""{"input":{"a":1}}""", "input")]
    [InlineData("""{}""", "input")]
    [InlineData("""{"input":"a","encoding_format":"int8"}""", "encoding_format")]
    [InlineData("""{"input":"a","dimensions":9}""", "dimensions")]
    [InlineData("""{"input":"a","prompt_name":"nope"}""", "prompt_name")]
    public async Task EmbeddingsRejectsBadRequests(string json, string param)
    {
        var (s, body) = await Post("/v1/embeddings", json);
        Assert.Equal(HttpStatusCode.BadRequest, s);
        Assert.Equal(param, body.GetProperty("error").GetProperty("param").GetString());
    }

    [Fact]
    public async Task OverLongInputIsAnErrorUnlessTruncated()
    {
        var (s, body) = await Post("/v1/embeddings", """{"input":"this text is longer than sixteen characters"}""");
        Assert.Equal(HttpStatusCode.BadRequest, s);
        Assert.Equal("context_length_exceeded", body.GetProperty("error").GetProperty("code").GetString());
        (s, body) = await Post("/v1/embeddings", """{"input":"this text is longer than sixteen characters","truncate":true}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal(16, body.GetProperty("usage").GetProperty("total_tokens").GetInt32());
    }

    [Fact]
    public async Task RerankSortsTakesTopNAndReturnsDocuments()
    {
        var (s, body) = await Post("/v1/rerank", """
            {"model":"fake-rerank","query":"capital of france","top_n":2,"return_documents":true,
             "documents":["Berlin is a city","Paris is the capital of France",{"text":"the capital"}]}
            """);
        Assert.Equal(HttpStatusCode.OK, s);
        var results = body.GetProperty("results");
        Assert.Equal(2, results.GetArrayLength());
        Assert.Equal(1, results[0].GetProperty("index").GetInt32());
        Assert.Equal(1f, results[0].GetProperty("relevance_score").GetSingle());
        Assert.Equal("Paris is the capital of France", results[0].GetProperty("document").GetProperty("text").GetString());
        Assert.Equal(2, results[1].GetProperty("index").GetInt32());
        Assert.True(body.GetProperty("usage").GetProperty("total_tokens").GetInt32() > 0);
        Assert.Equal(1, body.GetProperty("meta").GetProperty("billed_units").GetProperty("search_units").GetInt32());

        (s, body) = await Post("/v2/rerank", """{"query":"capital","documents":["a","capital"]}""");
        Assert.Equal(HttpStatusCode.OK, s);
        Assert.Equal(2, body.GetProperty("results").GetArrayLength());
        Assert.False(body.GetProperty("results")[0].TryGetProperty("document", out _));
    }

    [Theory]
    [InlineData("""{"documents":["a"]}""", "query")]
    [InlineData("""{"query":"q"}""", "documents")]
    [InlineData("""{"query":"q","documents":[1]}""", "documents")]
    [InlineData("""{"query":"q","documents":["a"],"top_n":0}""", "top_n")]
    public async Task RerankRejectsBadRequests(string json, string param)
    {
        var (s, body) = await Post("/v1/rerank", json);
        Assert.Equal(HttpStatusCode.BadRequest, s);
        Assert.Equal(param, body.GetProperty("error").GetProperty("param").GetString());
    }

    [Fact]
    public async Task RetrievalOnlyHostServesBothEndpoints()
    {
        var (s, _) = await Post("/v1/embeddings", """{"input":"hello"}""", standalone: true);
        Assert.Equal(HttpStatusCode.OK, s);
        (s, _) = await Post("/rerank", """{"query":"a","documents":["a"]}""", standalone: true);
        Assert.Equal(HttpStatusCode.OK, s);
        (s, var body) = await Post("/v1/embeddings", """{"input":[]}""", standalone: true);
        Assert.Equal(HttpStatusCode.BadRequest, s);
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
    }
}
