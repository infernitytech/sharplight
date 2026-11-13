using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;

namespace Sharplight.Server;

/// <summary>Server-sent events writer.</summary>
internal sealed class SseWriter
{
    private readonly HttpContext _http;
    public SseWriter(HttpContext http)
    {
        _http = http;
        http.Response.StatusCode = 200;
        http.Response.ContentType = "text/event-stream; charset=utf-8";
        http.Response.Headers.CacheControl = "no-cache";
        http.Response.Headers["X-Accel-Buffering"] = "no";
    }

    public Task DataAsync(JsonNode node) => RawAsync($"data: {HttpJson.ToJson(node)}\n\n");
    public Task EventAsync(string name, JsonNode node) => RawAsync($"event: {name}\ndata: {HttpJson.ToJson(node)}\n\n");

    public async Task RawAsync(string text)
    {
        await _http.Response.WriteAsync(text, Encoding.UTF8, _http.RequestAborted);
        await _http.Response.Body.FlushAsync(_http.RequestAborted);
    }
}
