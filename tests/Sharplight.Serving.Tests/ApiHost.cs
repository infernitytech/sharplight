using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Logging;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>An in-memory server (TestServer) with the full endpoint set over a FakeBatchDecoder.</summary>
public sealed class ApiHost : IAsyncDisposable
{
    /// <summary>ChatML-style template; the trailing U+0001 is the marker the fake decoder scripts replies against.</summary>
    public const string Template =
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{% if m.content %}{{ m.content }}{% endif %}" +
        "{% if m.tool_calls %}{% for c in m.tool_calls %}[call {{ c.function.name }} {{ c.function.arguments | tojson }}]{% endfor %}{% endif %}<|im_end|>\n{% endfor %}" +
        "{% if tools %}TOOLS: {{ tools | tojson }}\n{% endif %}" +
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking == false %}<think>\n\n</think>\n\n{% endif %}\u0001{% endif %}";

    public FakeBatchDecoder Decoder { get; }
    public InferenceEngine Engine { get; }
    public ModelService Model { get; }
    public WebApplication App { get; private set; } = null!;
    public HttpClient Client { get; private set; } = null!;

    private ApiHost(FakeBatchDecoder d, string? template, ModelServiceOptions? mo, EngineOptions? eo)
    {
        Decoder = d;
        Engine = new InferenceEngine(d, new FakeTokenizer(), eo);
        Model = new ModelService(Engine, template, mo ?? new ModelServiceOptions { ModelName = "fake-model", DefaultMaxTokens = 256 });
    }

    public static async Task<ApiHost> Create(Action<SharplightServerOptions>? configure = null, int slots = 2, string? template = Template, ModelServiceOptions? mo = null, EngineOptions? eo = null)
    {
        var h = new ApiHost(new FakeBatchDecoder(slots), template, mo, eo);
        var b = WebApplication.CreateBuilder();
        b.WebHost.UseTestServer();
        b.Logging.ClearProviders();
        b.Services.AddSharplight(h.Model, configure);
        h.App = b.Build();
        h.App.UseSharplight();
        h.App.MapSharplight();
        await h.App.StartAsync();
        h.Client = h.App.GetTestClient();
        return h;
    }

    public async Task<HttpResponseMessage> PostAsync(string path, string json, params (string, string)[] headers)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        foreach (var (k, v) in headers)
        {
            req.Headers.TryAddWithoutValidation(k, v);
        }

        return await Client.SendAsync(req);
    }

    public async Task<(HttpStatusCode Status, JsonElement Body)> PostJson(string path, string json, params (string, string)[] headers)
    {
        using var r = await PostAsync(path, json, headers);
        var text = await r.Content.ReadAsStringAsync();
        return (r.StatusCode, JsonDocument.Parse(text).RootElement.Clone());
    }

    /// <summary>Posts a streaming request and returns the parsed SSE events (event name, data text).</summary>
    public async Task<List<(string? Event, string Data)>> PostSse(string path, string json, params (string, string)[] headers)
    {
        using var r = await PostAsync(path, json, headers);
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.StartsWith("text/event-stream", r.Content.Headers.ContentType!.ToString());
        return ParseSse(await r.Content.ReadAsStringAsync());
    }

    public static List<(string? Event, string Data)> ParseSse(string body)
    {
        var res = new List<(string?, string)>();
        foreach (var block in body.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string? ev = null;
            var data = new StringBuilder();
            foreach (var line in block.Split('\n'))
            {
                if (line.StartsWith("event: "))
                {
                    ev = line[7..];
                }
                else if (line.StartsWith("data: "))
                {
                    data.Append(line[6..]);
                }
            }
            res.Add((ev, data.ToString()));
        }
        return res;
    }

    public async ValueTask DisposeAsync()
    {
        Client.Dispose();
        await App.StopAsync();
        await App.DisposeAsync();
        Engine.Dispose();
    }
}
