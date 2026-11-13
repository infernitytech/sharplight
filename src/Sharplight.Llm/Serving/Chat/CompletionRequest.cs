namespace Sharplight.Llm.Serving.Chat;

/// <summary>A plain text completion request (no chat template).</summary>
public sealed class CompletionRequest
{
    /// <summary>Prompt text (tokenized with special tokens added); ignored when <see cref="PromptTokens"/> is set.</summary>
    public string? Prompt { get; init; }
    /// <summary>Already tokenized prompt.</summary>
    public int[]? PromptTokens { get; init; }
    /// <summary>How the next token is chosen.</summary>
    public SamplingParams Sampling { get; init; } = new();
    /// <summary>Maximum generated tokens; 0 uses <see cref="ModelServiceOptions.DefaultMaxTokens"/>.</summary>
    public int MaxTokens { get; init; }
    /// <summary>Generation ends when one of these appears (it is cut off).</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    /// <summary>Extra token ids that end generation.</summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    /// <summary>Number of independent completions.</summary>
    public int N { get; init; } = 1;
    /// <summary>Higher runs first when requests queue for a slot.</summary>
    public int Priority { get; init; }
    /// <summary>Total time budget (queue plus generation); null uses the engine default.</summary>
    public TimeSpan? Timeout { get; init; }
    /// <summary>Report the log-probability of every sampled token.</summary>
    public bool Logprobs { get; init; }
    /// <summary>Alternatives reported per token (with <see cref="Logprobs"/>).</summary>
    public int TopLogprobs { get; init; }
}
