using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Sharplight.Llm.Serving;

namespace Sharplight.Server;

internal static class ApiErrors
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
        if (http.Response.HasStarted)
        {
            return;
        }

        if (!IsAnthropic(http) && status == 401)
        {
            http.Response.Headers.WWWAuthenticate = "Bearer";
        }

        if (retryAfter is int ra)
        {
            http.Response.Headers.RetryAfter = ra.ToString();
        }

        http.Response.StatusCode = status;
        await HttpJson.WriteAsync(http, Body(IsAnthropic(http), type, message, code, param), status);
    }
}
