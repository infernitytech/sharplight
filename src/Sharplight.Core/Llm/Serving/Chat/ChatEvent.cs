namespace Sharplight.Llm.Serving.Chat;

/// <summary>
/// An event of a chat stream. Exactly one of <see cref="Content"/>, <see cref="Reasoning"/>, <see cref="ToolCall"/>, <see cref="Logprobs"/>
/// or (final) <see cref="Finish"/> is set per event.
/// </summary>
/// <param name="Index">Which of the request's choices the event belongs to.</param>
/// <param name="Content">A piece of answer text.</param>
/// <param name="Reasoning">A piece of reasoning text.</param>
/// <param name="ToolCall">A complete tool call.</param>
/// <param name="Logprobs">Log-probabilities of newly sampled tokens.</param>
/// <param name="Finish">The final event of a choice carries the finish reason.</param>
/// <param name="Usage">Token counts (final event).</param>
/// <param name="StopSequence">The stop string that ended generation (final event).</param>
/// <param name="Error">Failure message (final event).</param>
/// <param name="Timings">Where the request's time went (final event).</param>
public sealed record ChatEvent(
    int Index,
    string? Content = null,
    string? Reasoning = null,
    ToolCall? ToolCall = null,
    IReadOnlyList<TokenLogprob>? Logprobs = null,
    ChatFinishReason? Finish = null,
    GenerationUsage? Usage = null,
    string? StopSequence = null,
    string? Error = null,
    GenerationTimings? Timings = null);
