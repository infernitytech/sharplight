namespace Sharplight.Llm.Serving;

/// <summary>
/// Where the time of one finished request went, measured by the engine's scheduler, and how speculative decoding fared for it.
/// Reported on the final <see cref="GenerationChunk"/> and to <see cref="EngineOptions.RequestFinished"/>.
/// </summary>
/// <param name="QueueWait">From submission until the request got a slot.</param>
/// <param name="Prefill">From admission until the first generated token (prompt processing, shared with other sequences' steps).</param>
/// <param name="Decode">From the first generated token until the request finished; zero when nothing was generated.</param>
/// <param name="Total">From submission until the request finished.</param>
/// <param name="DraftTokens">Draft tokens proposed for this request and verified by the model.</param>
/// <param name="AcceptedDraftTokens">Draft tokens the model accepted.</param>
public readonly record struct GenerationTimings(
    TimeSpan QueueWait,
    TimeSpan Prefill,
    TimeSpan Decode,
    TimeSpan Total,
    int DraftTokens,
    int AcceptedDraftTokens)
{
    /// <summary>From submission until the first generated token (queue wait plus prefill).</summary>
    public TimeSpan TimeToFirstToken => QueueWait + Prefill;
}
