namespace Sharplight.Llm.Serving;

/// <summary>A token-level generation request.</summary>
public sealed class GenerationRequest
{
    /// <summary>The prompt, already tokenized (not empty, ids inside the vocabulary).</summary>
    public required int[] PromptTokens { get; init; }
    /// <summary>How the next token is chosen.</summary>
    public SamplingParams Sampling { get; init; } = new();
    /// <summary>Maximum generated tokens; 0 or less means as many as the context window allows. Always capped by the room left in the context.</summary>
    public int MaxTokens { get; init; } = 256;
    /// <summary>Extra token ids that end generation (in addition to the end-of-generation tokens when <see cref="StopAtEog"/>).</summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    /// <summary>End generation at the model's end-of-generation tokens.</summary>
    public bool StopAtEog { get; init; } = true;
    /// <summary>Generation ends when one of these appears in the output; the matched string is not emitted.</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    /// <summary>Higher runs first; equal priorities are served first-come first-served.</summary>
    public int Priority { get; init; }
    /// <summary>Total time budget from submission (queue + generation). Exceeding it finishes with <see cref="FinishReason.Timeout"/>.</summary>
    public TimeSpan? Timeout { get; init; }
    /// <summary>Maximum time to wait for a slot; exceeding it finishes with <see cref="FinishReason.Timeout"/> without generating.</summary>
    public TimeSpan? QueueTimeout { get; init; }
    /// <summary>Report the log-probability of every sampled token (and <see cref="TopLogprobs"/> alternatives).</summary>
    public bool Logprobs { get; init; }
    /// <summary>Number of most likely alternatives reported per token (with <see cref="Logprobs"/>).</summary>
    public int TopLogprobs { get; init; }
    /// <summary>Hide control tokens from the generated text.</summary>
    public bool SkipSpecialTokens { get; init; }
}
