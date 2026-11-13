namespace Sharplight.Llm.Serving;

/// <summary>
/// One streamed piece of a generation. Content chunks carry <see cref="Text"/>; the last chunk has <see cref="Finish"/> and
/// <see cref="Usage"/> set (and <see cref="Error"/> when it ended in an error).
/// </summary>
public sealed record GenerationChunk(
    string Text,
    IReadOnlyList<TokenLogprob>? Logprobs = null,
    FinishReason? Finish = null,
    GenerationUsage? Usage = null,
    string? Error = null,
    string? StopSequence = null);
