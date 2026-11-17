namespace Sharplight.AspNetCore;

/// <summary>What one API request used, reported to the <see cref="ISharplightUsageObserver"/>s after its response is complete.</summary>
public sealed record SharplightUsage
{
    /// <summary>The route of the request (e.g. <c>/v1/chat/completions</c>, <c>/v1/messages</c>, <c>/v1/embeddings</c>).</summary>
    public required string Api { get; init; }

    /// <summary>The model name the client used (the alias), or null when the request failed before a model was chosen.</summary>
    public string? Model { get; init; }

    /// <summary>The request's isolation scope.</summary>
    public string? Scope { get; init; }

    /// <summary>The request's principal.</summary>
    public string? Principal { get; init; }

    /// <summary>The request's context (null when authentication failed).</summary>
    public SharplightRequestContext? Context { get; init; }

    /// <summary>Input tokens (prompt, or the embedded / scored texts).</summary>
    public long InputTokens { get; init; }

    /// <summary>Input tokens served from the prefix cache.</summary>
    public long CachedInputTokens { get; init; }

    /// <summary>Generated tokens (all choices).</summary>
    public long OutputTokens { get; init; }

    /// <summary>The response was streamed.</summary>
    public bool Streaming { get; init; }

    /// <summary>The HTTP status of the response (a stream that failed after it started reports 200).</summary>
    public int StatusCode { get; init; }

    /// <summary>Time from the request reaching the endpoint to the end of the response.</summary>
    public TimeSpan Duration { get; init; }

    /// <summary>Time to the first generated token of a streamed response, or null.</summary>
    public TimeSpan? TimeToFirstToken { get; init; }
}
