namespace Sharplight.Studio.Models;

/// <summary>Token counts and timings of one generated answer.</summary>
internal sealed record MessageStats
{
    /// <summary>Prompt length in tokens.</summary>
    public int PromptTokens { get; init; }

    /// <summary>Prompt tokens served from the prefix cache.</summary>
    public int CachedTokens { get; init; }

    /// <summary>Generated tokens.</summary>
    public int CompletionTokens { get; init; }

    /// <summary>Waiting for a free sequence slot, in milliseconds.</summary>
    public double QueueMs { get; init; }

    /// <summary>Prompt processing until the first token, in milliseconds.</summary>
    public double PrefillMs { get; init; }

    /// <summary>From the first to the last token, in milliseconds.</summary>
    public double DecodeMs { get; init; }

    /// <summary>End to end, in milliseconds.</summary>
    public double TotalMs { get; init; }

    /// <summary>Speculative draft tokens verified.</summary>
    public int DraftTokens { get; init; }

    /// <summary>Speculative draft tokens accepted.</summary>
    public int AcceptedDraftTokens { get; init; }

    /// <summary>Why generation ended (<c>stop</c>, <c>length</c>, <c>cancelled</c>, ...).</summary>
    public string Finish { get; init; } = "stop";

    /// <summary>Time to first token in milliseconds.</summary>
    public double TimeToFirstTokenMs => QueueMs + PrefillMs;

    /// <summary>Prompt tokens computed (not cached) per second of prefill.</summary>
    public double PrefillTokensPerSecond => PrefillMs <= 0 ? 0 : (PromptTokens - CachedTokens) / (PrefillMs / 1000);

    /// <summary>Generated tokens per second after the first one.</summary>
    public double DecodeTokensPerSecond => DecodeMs <= 0 ? 0 : Math.Max(0, CompletionTokens - 1) / (DecodeMs / 1000);

    /// <summary>Accepted divided by proposed draft tokens, or null without speculation.</summary>
    public double? AcceptanceRate => DraftTokens == 0 ? null : (double)AcceptedDraftTokens / DraftTokens;
}
