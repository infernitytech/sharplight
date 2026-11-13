using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Sharplight.Llm.Serving;

namespace Sharplight.Server;

/// <summary>An error the API layer reports to the client (the shape depends on the API).</summary>
public sealed class ApiException(int status, string type, string message, string? code = null, string? param = null) : Exception(message)
{
    public int Status { get; } = status;
    public string Type { get; } = type;
    public string? Code { get; } = code;
    public string? Param { get; } = param;
    public static ApiException BadRequest(string message, string? param = null) => new(400, "invalid_request_error", message, null, param);
}

static class ApiErrors
{
    public static bool IsAnthropic(HttpContext http)
        => (http.Request.Path.Value ?? "").StartsWith("/v1/messages", StringComparison.Ordinal) || http.Request.Headers.ContainsKey("anthropic-version");

    public static Task WriteAsync(HttpContext http, EngineException ex)
    {
        var anthropic = IsAnthropic(http);
        return ex.Code switch
        {
            "queue_full" => WriteAsync(http, 429, anthropic ? "rate_limit_error" : "rate_limit_exceeded", ex.Message, "queue_full", null, retryAfter: 1),
            "shutting_down" => WriteAsync(http, 503, anthropic ? "api_error" : "server_error", ex.Message, "shutting_down"),
            "context_length_exceeded" => WriteAsync(http, 400, "invalid_request_error", ex.Message, "context_length_exceeded"),
            "model_not_found" => WriteAsync(http, 404, anthropic ? "not_found_error" : "invalid_request_error", ex.Message, "model_not_found", "model"),
            "insufficient_memory" => WriteAsync(http, 503, anthropic ? "api_error" : "server_error", ex.Message, "insufficient_memory", retryAfter: 5),
            "model_busy" => WriteAsync(http, 503, anthropic ? "overloaded_error" : "server_error", ex.Message, "model_busy", retryAfter: 2),
            _ => WriteAsync(http, 400, "invalid_request_error", ex.Message, ex.Code),
        };
    }

    public static JsonObject Body(bool anthropic, string type, string message, string? code = null, string? param = null)
        => anthropic
            ? new JsonObject { ["type"] = "error", ["error"] = new JsonObject { ["type"] = type, ["message"] = message } }
            : new JsonObject { ["error"] = new JsonObject { ["message"] = message, ["type"] = type, ["param"] = param, ["code"] = code } };

    public static async Task WriteAsync(HttpContext http, int status, string type, string message, string? code = null, string? param = null, int? retryAfter = null)
    {
        if (http.Response.HasStarted) return;
        if (!IsAnthropic(http) && status == 401) http.Response.Headers.WWWAuthenticate = "Bearer";
        if (retryAfter is int ra) http.Response.Headers.RetryAfter = ra.ToString();
        http.Response.StatusCode = status;
        await HttpJson.WriteAsync(http, Body(IsAnthropic(http), type, message, code, param), status);
    }
}

static class HttpJson
{
    public static readonly JsonSerializerOptions Options = new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    public static string ToJson(JsonNode n) => n.ToJsonString(Options);

    public static async Task WriteAsync(HttpContext http, JsonNode body, int? status = null)
    {
        if (status is int s) http.Response.StatusCode = s;
        http.Response.ContentType = "application/json";
        await http.Response.WriteAsync(ToJson(body), Encoding.UTF8, http.RequestAborted);
    }

    /// <summary>Reads and parses the JSON request body, enforcing the size limit.</summary>
    public static async Task<JsonDocument> ReadBodyAsync(HttpContext http, long maxBytes)
    {
        if (http.Request.ContentLength > maxBytes) throw new ApiException(413, "request_too_large", $"The request body exceeds the limit of {maxBytes} bytes.");
        if (http.Features.Get<IHttpMaxRequestBodySizeFeature>() is { IsReadOnly: false } f) f.MaxRequestBodySize = maxBytes;
        using var ms = new MemoryStream();
        var buf = new byte[81920];
        int n;
        while ((n = await http.Request.Body.ReadAsync(buf, http.RequestAborted)) > 0)
        {
            ms.Write(buf, 0, n);
            if (ms.Length > maxBytes) throw new ApiException(413, "request_too_large", $"The request body exceeds the limit of {maxBytes} bytes.");
        }
        try
        {
            var doc = JsonDocument.Parse(ms.ToArray());
            if (doc.RootElement.ValueKind != JsonValueKind.Object) throw ApiException.BadRequest("The request body must be a JSON object.");
            return doc;
        }
        catch (JsonException ex) { throw ApiException.BadRequest("Invalid JSON: " + ex.Message); }
    }
}

/// <summary>Server-sent events writer.</summary>
sealed class SseWriter
{
    readonly HttpContext _http;
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

/// <summary>Typed accessors over a request JSON object.</summary>
static class Jx
{
    public static JsonElement? Get(this JsonElement o, string name)
        => o.ValueKind == JsonValueKind.Object && o.TryGetProperty(name, out var v) && v.ValueKind != JsonValueKind.Null ? v : null;
    public static string? Str(this JsonElement o, string name) => o.Get(name) is { ValueKind: JsonValueKind.String } v ? v.GetString() : null;
    public static int? Int(this JsonElement o, string name, string? param = null)
    {
        if (o.Get(name) is not { } v) return null;
        if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var i)) return i;
        if (v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out var d)) return (int)d;
        throw ApiException.BadRequest($"'{name}' must be an integer.", param ?? name);
    }
    public static float? Float(this JsonElement o, string name)
        => o.Get(name) is { } v ? v.ValueKind == JsonValueKind.Number ? v.GetSingle() : throw ApiException.BadRequest($"'{name}' must be a number.", name) : null;
    public static bool? Bool(this JsonElement o, string name)
        => o.Get(name) is { } v ? v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : throw ApiException.BadRequest($"'{name}' must be a boolean.", name) : null;
    public static IEnumerable<JsonElement> Arr(this JsonElement o, string name)
        => o.Get(name) is { } v ? v.ValueKind == JsonValueKind.Array ? v.EnumerateArray() : throw ApiException.BadRequest($"'{name}' must be an array.", name) : [];
    public static JsonElement? Obj(this JsonElement o, string name)
        => o.Get(name) is { ValueKind: JsonValueKind.Object } v ? v : null;
}
