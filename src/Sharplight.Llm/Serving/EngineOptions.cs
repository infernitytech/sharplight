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
    /// Used only when the decoder can roll back the rows of rejected drafts (<see cref="IBatchDecoder.MaxSpeculativeRows"/> above 1:
    /// attention-only models, or models with recurrent state from <see cref="LlmModel.CreateBatchDecoder"/> with
    /// <c>draftTokens</c>), returns more than one logits row per step (<c>maxLogitsRows</c>) and, for a speculator that reads
    /// hidden states, exposes them; it is disposed and ignored otherwise.
    /// </summary>
    public ISpeculator? Speculator { get; init; }
    /// <summary>
    /// Speculate only while at most this many sequences decode: large batches are compute bound, where verifying drafts costs
    /// more than it saves.
    /// </summary>
    public int SpeculationMaxBatch { get; init; } = 8;
    /// <summary>
    /// Most draft rows verified per step over all sequences: with many sequences decoding, each gets
    /// <c>SpeculationMaxDraftRows / sequences</c> drafts (at least 1, at most <see cref="ISpeculator.MaxDraftTokens"/>), so the
    /// step grows less as the batch, and with it the compute share of the step, grows.
    /// </summary>
    public int SpeculationMaxDraftRows { get; init; } = 16;
}
