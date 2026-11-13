using System.Net;

namespace Sharplight.Hub;

/// <summary>Classifies failures as transient and computes backoff delays for <see cref="RetryOptions"/>.</summary>
internal static class RetryPolicy
{
    private const double JitterFraction = 0.2;

    /// <summary>True for failures a later attempt may not see: network errors, timeouts (not user cancellation) and transient HTTP statuses.</summary>
    public static bool IsTransient(Exception error, CancellationToken userToken)
    {
        return error switch
        {
            TransientHttpException => true,
            OperationCanceledException => !userToken.IsCancellationRequested,
            HttpRequestException http => http.StatusCode is null || IsTransient(http.StatusCode.Value),
            IOException => true,
            _ => false,
        };
    }

    public static bool IsTransient(HttpStatusCode status)
    {
        return status is HttpStatusCode.RequestTimeout or HttpStatusCode.TooManyRequests || (int)status >= 500;
    }

    /// <summary>Delay before retry number <paramref name="failures"/> (1 = the first retry).</summary>
    public static TimeSpan Delay(RetryOptions options, int failures, Exception error)
    {
        if (error is TransientHttpException { RetryAfter: { } retryAfter })
        {
            return retryAfter < options.MaxDelay ? retryAfter : options.MaxDelay;
        }

        double exponential = options.InitialDelay.TotalMilliseconds * Math.Pow(2, Math.Max(0, failures - 1));
        double capped = Math.Min(exponential, options.MaxDelay.TotalMilliseconds);
        double jitter = 1 + ((Random.Shared.NextDouble() * 2) - 1) * JitterFraction;
        return TimeSpan.FromMilliseconds(capped * jitter);
    }

    /// <summary>A short description of a transient failure for log lines.</summary>
    public static string Describe(Exception error)
    {
        return error is OperationCanceledException ? "timed out" : error.Message;
    }
}
