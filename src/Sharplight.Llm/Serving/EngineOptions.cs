using Sharplight.Llm.Serving.Speculative;

namespace Sharplight.Llm.Serving;

/// <summary>Scheduler settings of an <see cref="InferenceEngine"/>.</summary>
public sealed record EngineOptions
{
    /// <summary>Maximum number of requests waiting for a slot; further submissions throw <see cref="QueueFullException"/>.</summary>
    public int MaxQueue { get; init; } = 256;
    /// <summary>
    /// Token budget of one step: rows (decode tokens, draft tokens and prompt chunks summed over all sequences) per forward pass
    /// (0 = the decoder's <see cref="IBatchDecoder.MaxBatchTokens"/>). Decode rows are scheduled first, prompt chunks fill the rest,
    /// so this bounds how much a long prompt can slow down the running streams.
    /// </summary>
    public int MaxBatchTokens { get; init; }
    /// <summary>Maximum prompt rows of one sequence per step. Smaller chunks interleave long prefills with other sequences' decoding.</summary>
    public int PrefillChunk { get; init; } = 512;
    /// <summary>Reuse the cached context of idle slots for prompts sharing a prefix.</summary>
    public bool PrefixCache { get; init; } = true;
    /// <summary>Applied to requests without their own <see cref="GenerationRequest.Timeout"/>.</summary>
    public TimeSpan? DefaultTimeout { get; init; }
    /// <summary>Applied to requests without their own <see cref="GenerationRequest.QueueTimeout"/>.</summary>
    public TimeSpan? DefaultQueueTimeout { get; init; }
    /// <summary>Half-life of the per-tenant usage that fair queuing compares (see <see cref="GenerationRequest.Tenant"/>).</summary>
    public TimeSpan FairShareHalfLife { get; init; } = TimeSpan.FromSeconds(60);
    /// <summary>
    /// Speculative decoding: proposes draft tokens that the target model verifies in one pass (the engine takes ownership).
    /// Used only when the decoder can roll back rows (<see cref="IBatchDecoder.CanTruncate"/>, i.e. attention-only models) and
    /// returns more than one logits row per step (<see cref="LlmModel.CreateBatchDecoder"/> with <c>maxLogitsRows</c>); it is
    /// disposed and ignored otherwise.
    /// </summary>
    public ISpeculator? Speculator { get; init; }
    /// <summary>
    /// Speculate only while at most this many sequences decode: large batches are compute bound, where verifying drafts costs
    /// more than it saves.
    /// </summary>
    public int SpeculationMaxBatch { get; init; } = 8;
}
