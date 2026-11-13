namespace Sharplight.Llm.Serving.Chat;

public sealed class ChatChoice
{
    public int Index { get; init; }
    public string Content { get; init; } = "";
    public string? Reasoning { get; init; }
    public IReadOnlyList<ToolCall> ToolCalls { get; init; } = [];
    public ChatFinishReason Finish { get; init; }
    public string? StopSequence { get; init; }
    public string? Error { get; init; }
    public IReadOnlyList<TokenLogprob> Logprobs { get; init; } = [];
    public GenerationUsage Usage { get; init; }
}
