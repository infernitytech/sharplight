namespace Sharplight.Llm.Serving;

/// <summary>The complete outcome of a request, as returned by <see cref="GenerationHandle.CompleteAsync"/>.</summary>
public sealed class GenerationResult
{
    /// <summary>All generated text (a matched stop string is cut off).</summary>
    public string Text { get; init; } = "";
    /// <summary>Why the generation ended.</summary>
    public FinishReason Finish { get; init; }
    /// <summary>Token counts.</summary>
    public GenerationUsage Usage { get; init; }
    /// <summary>Failure message when <see cref="Finish"/> is <see cref="FinishReason.Error"/>.</summary>
    public string? Error { get; init; }
    /// <summary>The stop string that ended the generation, if any.</summary>
    public string? StopSequence { get; init; }
    /// <summary>Per-token log-probabilities when requested (<see cref="GenerationRequest.Logprobs"/>), else empty.</summary>
    public IReadOnlyList<TokenLogprob> Logprobs { get; init; } = [];
}
