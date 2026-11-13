namespace Sharplight.Llm.Serving;

public sealed class GenerationResult
{
    public string Text { get; init; } = "";
    public FinishReason Finish { get; init; }
    public GenerationUsage Usage { get; init; }
    public string? Error { get; init; }
    public string? StopSequence { get; init; }
    public IReadOnlyList<TokenLogprob> Logprobs { get; init; } = [];
}
