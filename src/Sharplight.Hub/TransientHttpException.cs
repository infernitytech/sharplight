using System.Net;

namespace Sharplight.Hub;

/// <summary>An HTTP status worth retrying (408, 429, 5xx), with the server's <c>Retry-After</c> if it sent one.</summary>
internal sealed class TransientHttpException(HttpStatusCode statusCode, TimeSpan? retryAfter, string message) : Exception(message)
{
    public HttpStatusCode StatusCode { get; } = statusCode;

    public TimeSpan? RetryAfter { get; } = retryAfter;
}
