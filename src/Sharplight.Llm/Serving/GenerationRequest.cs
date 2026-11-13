namespace Sharplight.Llm.Serving;

/// <summary>A token-level generation request.</summary>
public sealed class GenerationRequest
{
    public required int[] PromptTokens { get; init; }
    public SamplingParams Sampling { get; init; } = new();
    public int MaxTokens { get; init; } = 256;
    /// <summary>Extra token ids that end generation (in addition to the end-of-generation tokens when <see cref="StopAtEog"/>).</summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    public bool StopAtEog { get; init; } = true;
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    /// <summary>Higher runs first; equal priorities are served first-come first-served.</summary>
    public int Priority { get; init; }
    /// <summary>Total time budget from submission (queue + generation). Exceeding it finishes with <see cref="FinishReason.Timeout"/>.</summary>
    public TimeSpan? Timeout { get; init; }
    /// <summary>Maximum time to wait for a slot; exceeding it finishes with <see cref="FinishReason.Timeout"/> without generating.</summary>
    public TimeSpan? QueueTimeout { get; init; }
    /// <summary>Report the log-probability of every sampled token (and <see cref="TopLogprobs"/> alternatives).</summary>
    public bool Logprobs { get; init; }
    public int TopLogprobs { get; init; }
    /// <summary>Hide control tokens from the generated text.</summary>
    public bool SkipSpecialTokens { get; init; }
}
