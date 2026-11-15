namespace Sharplight.Llm.Serving.Chat;

/// <summary>One finished completion of a <see cref="ChatResult"/>.</summary>
public sealed class ChatChoice
{
    /// <summary>Position among the request's <see cref="ChatRequest.N"/> choices.</summary>
    public int Index { get; init; }
    /// <summary>The answer text without reasoning and tool-call blocks.</summary>
    public string Content { get; init; } = "";
    /// <summary>The model's reasoning text, or null when it produced none.</summary>
    public string? Reasoning { get; init; }
    /// <summary>Tool calls parsed from the output, in order.</summary>
    public IReadOnlyList<ToolCall> ToolCalls { get; init; } = [];
    /// <summary>Why generation ended.</summary>
    public ChatFinishReason Finish { get; init; }
    /// <summary>The stop string that ended generation, if any.</summary>
    public string? StopSequence { get; init; }
    /// <summary>Failure message for <see cref="ChatFinishReason.Error"/>.</summary>
    public string? Error { get; init; }
    /// <summary>Per-token log-probabilities when requested, else empty.</summary>
    public IReadOnlyList<TokenLogprob> Logprobs { get; init; } = [];
    /// <summary>Token counts of this choice.</summary>
    public GenerationUsage Usage { get; init; }
}
