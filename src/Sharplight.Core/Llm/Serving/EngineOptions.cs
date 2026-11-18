using Sharplight.Llm.Serving.Speculative;

namespace Sharplight.Llm.Serving;

/// <summary>Scheduler settings of an <see cref="InferenceEngine"/>.</summary>
public sealed record EngineOptions
{
    /// <summary>Maximum number of requests waiting for a slot; further submissions throw <see cref="QueueFullException"/>.</summary>
    public int MaxQueue { get; set; } = 256;
    /// <summary>
    /// Token budget of one step: rows (decode tokens, draft tokens and prompt chunks summed over all sequences) per forward pass
    /// (0 = the decoder's <see cref="IBatchDecoder.MaxBatchTokens"/>). Decode rows are scheduled first, prompt chunks fill the rest,
    /// so this bounds how much a long prompt can slow down the running streams.
    /// </summary>
    public int MaxBatchTokens { get; set; }
    /// <summary>
    /// Maximum prompt rows of one sequence per step (0, the default: the step's token budget, so a prompt takes whole passes of the
    /// decoder's rows). Smaller chunks interleave long prefills with other sequences' decoding at the cost of prefill speed.
    /// </summary>
    public int PrefillChunk { get; set; }
    /// <summary>
    /// Prefix caching: reuse the KV blocks (and, for models with recurrent state, state snapshots) of earlier prompts that share a
    /// prefix with a new one, within the same cache namespace (<see cref="GenerationRequest.CachePartition"/>, <see cref="GenerationRequest.CacheSalt"/>).
    /// </summary>
    public bool PrefixCache { get; set; } = true;
    /// <summary>
    /// Models with recurrent state: prompt tokens between recurrent-state snapshots during prefill (prefill chunks end at multiples of it
    /// and the state there is kept for later prompts, least recently used first out), in addition to the snapshots at prefixes shared with
    /// other contexts and at prompt ends. Rounded down to whole KV blocks; 0 turns the periodic snapshots off.
    /// </summary>
    public int StateSnapshotInterval { get; set; } = 512;
    /// <summary>Applied to requests without their own <see cref="GenerationRequest.Timeout"/>.</summary>
    public TimeSpan? DefaultTimeout { get; set; }
    /// <summary>Applied to requests without their own <see cref="GenerationRequest.QueueTimeout"/>.</summary>
    public TimeSpan? DefaultQueueTimeout { get; set; }
    /// <summary>Half-life of the per-tenant usage that fair queuing compares (see <see cref="GenerationRequest.Tenant"/>).</summary>
    public TimeSpan FairShareHalfLife { get; set; } = TimeSpan.FromSeconds(60);
    /// <summary>
    /// Speculative decoding: proposes draft tokens that the target model verifies in one pass (the engine takes ownership).
    /// Used only when the decoder can roll back the rows of rejected drafts (<see cref="IBatchDecoder.MaxSpeculativeRows"/> above 1:
    /// attention-only models, or models with recurrent state from <see cref="LlmModel.CreateBatchDecoder"/> with
    /// <c>draftTokens</c>), returns more than one logits row per step (<c>maxLogitsRows</c>) and, for a speculator that reads
    /// hidden states, exposes them; it is disposed and ignored otherwise.
    /// </summary>
    public ISpeculator? Speculator { get; set; }
    /// <summary>
    /// Speculate only while at most this many sequences decode: large batches are compute bound, where verifying drafts costs
    /// more than it saves.
    /// </summary>
    public int SpeculationMaxBatch { get; set; } = 16;
    /// <summary>
    /// Most draft rows verified per step over all sequences: with many sequences decoding, each gets
    /// <c>SpeculationMaxDraftRows / sequences</c> drafts (at least 1, at most <see cref="ISpeculator.MaxDraftTokens"/>), so the
    /// step grows less as the batch, and with it the compute share of the step, grows.
    /// </summary>
    public int SpeculationMaxDraftRows { get; set; } = 16;
    /// <summary>
    /// Called once per finished request (any reason) with its usage and timings, for statistics and request logs. Runs on the
    /// scheduler thread after the final chunk was written: keep it short and non-blocking; exceptions are ignored.
    /// </summary>
    public Action<FinishedRequest>? RequestFinished { get; set; }
}
