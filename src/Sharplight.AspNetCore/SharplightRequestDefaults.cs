using Sharplight.Llm.Serving;

namespace Sharplight.AspNetCore;

/// <summary>
/// Defaults and limits of one request, from the <see cref="IModelEndpointResolver"/> (e.g. per endpoint or per model alias); a value the
/// request gives wins within the limits. Null values fall back to <see cref="SharplightServerOptions"/>.
/// </summary>
public sealed record SharplightRequestDefaults
{
    /// <summary>Sampling parameters a request leaves out.</summary>
    public SamplingParams? Sampling { get; init; }

    /// <summary>Generated tokens when the request names no limit.</summary>
    public int? MaxTokens { get; init; }

    /// <summary>Upper bound on generated tokens; larger (or unlimited) requests are cut to it.</summary>
    public int? MaxTokensLimit { get; init; }

    /// <summary>Total time budget per request (queue + generation).</summary>
    public TimeSpan? RequestTimeout { get; init; }

    /// <summary>System prompt put first when the request's messages have no system message (chat APIs), or null.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>These defaults with every unset value taken from <paramref name="fallback"/>.</summary>
    /// <param name="fallback">The defaults to fall back to, or null.</param>
    /// <returns>The merged defaults.</returns>
    public SharplightRequestDefaults Over(SharplightRequestDefaults? fallback) => fallback is null ? this : new SharplightRequestDefaults
    {
        Sampling = Sampling ?? fallback.Sampling,
        MaxTokens = MaxTokens ?? fallback.MaxTokens,
        MaxTokensLimit = MaxTokensLimit ?? fallback.MaxTokensLimit,
        RequestTimeout = RequestTimeout ?? fallback.RequestTimeout,
        SystemPrompt = SystemPrompt ?? fallback.SystemPrompt,
    };
}
