namespace Sharplight.AspNetCore;

/// <summary>
/// An error the API layer reports to the client, in the error shape of the API the request used (OpenAI or Anthropic). Throw it from
/// hooks (<see cref="IModelEndpointResolver"/>, <see cref="IRequestAdmission"/>, <see cref="IApiKeyValidator"/>) to refuse a request.
/// </summary>
/// <param name="status">The HTTP status.</param>
/// <param name="type">The error type (<c>invalid_request_error</c>, <c>authentication_error</c>, <c>permission_error</c>, ...).</param>
/// <param name="message">The message shown to the client.</param>
/// <param name="code">The OpenAI error code, or null.</param>
/// <param name="param">The request field the error is about, or null.</param>
public sealed class ApiException(int status, string type, string message, string? code = null, string? param = null) : Exception(message)
{
    /// <summary>The HTTP status.</summary>
    public int Status { get; } = status;

    /// <summary>The error type.</summary>
    public string Type { get; } = type;

    /// <summary>The OpenAI error code, or null.</summary>
    public string? Code { get; } = code;

    /// <summary>The request field the error is about, or null.</summary>
    public string? Param { get; } = param;

    /// <summary>Seconds after which the client may retry (<c>Retry-After</c>), or null.</summary>
    public int? RetryAfterSeconds { get; init; }

    /// <summary>A 400 <c>invalid_request_error</c>.</summary>
    /// <param name="message">The message.</param>
    /// <param name="param">The request field, or null.</param>
    /// <returns>The exception.</returns>
    public static ApiException BadRequest(string message, string? param = null) => new(400, "invalid_request_error", message, null, param);

    /// <summary>A 429 rate-limit error (quota, rate or concurrency limit) with <c>Retry-After</c>.</summary>
    /// <param name="message">The message.</param>
    /// <param name="retryAfterSeconds">Seconds after which the client may retry.</param>
    /// <returns>The exception.</returns>
    public static ApiException TooManyRequests(string message, int retryAfterSeconds = 1)
        => new(429, "rate_limit_error", message, "rate_limit_exceeded") { RetryAfterSeconds = retryAfterSeconds };

    /// <summary>A 404 for a model the request may not see or that does not exist.</summary>
    /// <param name="message">The message.</param>
    /// <param name="anthropic">Use the Anthropic error type.</param>
    /// <returns>The exception.</returns>
    internal static ApiException ModelNotFound(string message, bool anthropic)
        => new(404, anthropic ? "not_found_error" : "invalid_request_error", message, "model_not_found", "model");
}
