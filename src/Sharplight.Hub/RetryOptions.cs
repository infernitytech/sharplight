namespace Sharplight.Hub;

/// <summary>
/// How transient failures (connection errors, timeouts, stalled transfers, HTTP 408/429/5xx) are retried: exponential backoff
/// from <see cref="InitialDelay"/> up to <see cref="MaxDelay"/>, honouring <c>Retry-After</c>. A download attempt that made
/// progress resets the count.
/// </summary>
public sealed class RetryOptions
{
    /// <summary>Attempts per request (or per download without progress) before giving up; at least 1.</summary>
    public int MaxAttempts { get; init; } = 8;

    /// <summary>Delay before the first retry; doubled for every further one.</summary>
    public TimeSpan InitialDelay { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>Upper bound of a single delay (also caps <c>Retry-After</c>).</summary>
    public TimeSpan MaxDelay { get; init; } = TimeSpan.FromSeconds(60);
}
