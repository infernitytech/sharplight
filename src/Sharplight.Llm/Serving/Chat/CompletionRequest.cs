namespace Sharplight.Llm.Serving.Chat;

/// <summary>A plain text completion request (no chat template).</summary>
public sealed class CompletionRequest
{
    public string? Prompt { get; init; }
    public int[]? PromptTokens { get; init; }
    public SamplingParams Sampling { get; init; } = new();
    public int MaxTokens { get; init; }
    public IReadOnlyList<string> StopStrings { get; init; } = [];
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];
    public int N { get; init; } = 1;
    public int Priority { get; init; }
    public TimeSpan? Timeout { get; init; }
    public bool Logprobs { get; init; }
    public int TopLogprobs { get; init; }
}
