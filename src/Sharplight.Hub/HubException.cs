using System.Net;

namespace Sharplight.Hub;

/// <summary>
/// A failure talking to the Hub or managing the model store, with a message meant for the user (authentication, gated or
/// missing repositories, failed downloads, checksum mismatches, insufficient disk space, ambiguous names).
/// </summary>
public sealed class HubException : Exception
{
    /// <summary>Creates the exception with the message shown to the user.</summary>
    /// <param name="message">What went wrong and, where possible, what to do about it.</param>
    /// <param name="statusCode">The HTTP status of the failed request, if any.</param>
    /// <param name="inner">The underlying error, if any.</param>
    public HubException(string message, HttpStatusCode? statusCode = null, Exception? inner = null)
        : base(message, inner)
    {
        StatusCode = statusCode;
    }

    /// <summary>The HTTP status of the failed request, or <see langword="null"/> when the failure was not an HTTP error.</summary>
    public HttpStatusCode? StatusCode { get; }
}
