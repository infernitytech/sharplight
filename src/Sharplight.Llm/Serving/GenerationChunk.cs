namespace Sharplight.Llm.Serving;

/// <summary>
/// One streamed piece of a generation. Content chunks carry <see cref="Text"/>; the last chunk has <see cref="Finish"/> and
/// <see cref="Usage"/> set (and <see cref="Error"/> when it ended in an error).
/// </summary>
/// <param name="Text">New text; empty for chunks that carry only log-probabilities or the finish information.</param>
/// <param name="Logprobs">Log-probabilities of the tokens behind this chunk, when requested.</param>
/// <param name="Finish">Set on the last chunk only.</param>
/// <param name="Usage">Set on the last chunk only.</param>
/// <param name="Error">Failure message when <paramref name="Finish"/> is <see cref="FinishReason.Error"/>.</param>
/// <param name="StopSequence">The stop string that ended the generation (it is not part of the text), if that was the reason.</param>
/// <param name="Timings">Set on the last chunk only: where the request's time went.</param>
public sealed record GenerationChunk(
    string Text,
    IReadOnlyList<TokenLogprob>? Logprobs = null,
    FinishReason? Finish = null,
    GenerationUsage? Usage = null,
    string? Error = null,
    string? StopSequence = null,
    GenerationTimings? Timings = null);
