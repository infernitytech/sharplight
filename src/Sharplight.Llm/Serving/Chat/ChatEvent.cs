namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// An event of a chat stream. Exactly one of <see cref="Content"/>, <see cref="Reasoning"/>, <see cref="ToolCall"/>, <see cref="Logprobs"/>
/// or (final) <see cref="Finish"/> is set per event.
/// </summary>
public sealed record ChatEvent(
    int Index,
    string? Content = null,
    string? Reasoning = null,
    ToolCall? ToolCall = null,
    IReadOnlyList<TokenLogprob>? Logprobs = null,
    ChatFinishReason? Finish = null,
    GenerationUsage? Usage = null,
    string? StopSequence = null,
    string? Error = null);
