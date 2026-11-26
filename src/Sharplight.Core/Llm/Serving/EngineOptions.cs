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
    /// Load-adaptive step budget: while few sequences hold a slot, a step carries at most <see cref="BatchTokensPerSequence"/> rows
    /// per such sequence, but never fewer than this (0 turns the adaptive budget off: every step may use the whole
    /// <see cref="MaxBatchTokens"/>). A wave of prompts arriving at a lightly loaded engine is then prefilled in several smaller
    /// steps, so the first prompts finish (and stream their first token) earlier, while under load the steps grow to the full
    /// budget, whose larger GEMMs prefill more rows per second.
    /// </summary>
    public int MinBatchTokens { get; set; } = 768;
    /// <summary>
    /// When the floor of the adaptive step budget (<see cref="MinBatchTokens"/>) applies: <see cref="StepFloorPolicy.Waves"/> (default) only
    /// while no sequence decodes, <see cref="StepFloorPolicy.Always"/> to every step (the earlier default), <see cref="StepFloorPolicy.Full"/>
    /// the whole budget while some sequence decodes. The environment variable <c>SHARPLIGHT_STEP_FLOOR_WAVES</c> (<c>1</c>, <c>0</c>,
    /// <c>full</c>) overrides it for the process.
    /// </summary>
    public StepFloorPolicy StepFloor { get; set; } = StepFloorPolicy.Waves;
    /// <summary>
    /// Rows per sequence holding a slot that the adaptive step budget allows (see <see cref="MinBatchTokens"/>), rounded up to
    /// <see cref="BatchTokensGranule"/>: while some sequence decodes, a step of 16 slots carries at most 768 rows by default. Larger values
    /// prefill a wave of prompts in fewer, longer steps (fewer decode gaps carry a prompt: lower ITL p95, slightly higher throughput on
    /// MoE models) at the cost of TTFT and the ITL p99 of the prompts admitted first (benchmarks.md §100.1). Default 48. The environment
    /// variable <c>SHARPLIGHT_STEP_ROWS_PER_SEQUENCE</c> overrides it for the process.
    /// </summary>
    public int BatchTokensPerSequence { get; set; } = 48;
    /// <summary>
    /// The adaptive step budget (see <see cref="MinBatchTokens"/>) is rounded up to a multiple of this many rows (1: no rounding). A
    /// prompt-sized GEMM's time follows the count of its token tiles (128 rows on the CUDA provider's FP8 and f16 GEMMs), so a budget
    /// of 1,824 rows costs what 1,920 would (benchmarks.md §90).
    /// </summary>
    public int BatchTokensGranule { get; set; } = 128;
    /// <summary>
    /// Prompt rows scheduled around token tiles (<see cref="BatchTokensGranule"/>) on measured prices (default on): a step whose rows would
    /// end a few rows past a tile holds those prompt rows back for a later step's decode rows, and a prompt that the budget cuts a few
    /// rows short of its end takes them in this step, each only when the step costs measured on this engine say it is cheaper (the cost of
    /// another token tile against what the rows add to a decode step; benchmarks.md §101). The environment variable
    /// <c>SHARPLIGHT_STEP_TILE_AWARE=0|1</c> overrides it for the process.
    /// </summary>
    public bool TileAwarePrompts { get; set; } = true;
    /// <summary>
    /// Prompt chunk sizes chosen on measured prices while some sequence decodes (default on): when the prompt rows of the steps measured
    /// on this engine cost less per row in smaller chunks (their price over the decode-only step grows with the chunk, as where prompt
    /// rows ride the decode rows' weight reads), the budget is <see cref="SmallChunkRowsPerSequence"/> rows per slot instead of
    /// <see cref="BatchTokensPerSequence"/>; where larger chunks are as cheap or cheaper per row (expert GEMMs that read more experts as
    /// rows grow), it stays <see cref="BatchTokensPerSequence"/>. Needs <see cref="TileAwarePrompts"/> (its step prices).
    /// </summary>
    public bool PricedPromptChunks { get; set; } = true;
    /// <summary>
    /// Rows per sequence holding a slot of a decoding step's budget when <see cref="PricedPromptChunks"/> chooses the smaller chunks
    /// (default 16: 256 rows at 16 slots; at most <see cref="BatchTokensPerSequence"/>).
    /// </summary>
    public int SmallChunkRowsPerSequence { get; set; } = 16;
    /// <summary>
    /// Steps that carry prompt rows run in shape buckets (default on): their rows are padded up to a multiple of 64 on the last prompt
    /// chunk (positions the sequence has not reached yet), their logits rows up to a multiple of 8 and their run table to the slot count,
    /// so steps with nearby row counts replay one recorded step (a CUDA graph) instead of each paying the first run of a new shape (its
    /// recording, then its capture). Applies to decoders whose caches allow it (paged, causal attention, no recurrent state or
    /// window-sized caches); others run every step at its exact shape (benchmarks.md §R48).
    /// </summary>
    public bool StepShapeBuckets { get; set; } = true;
    /// <summary>
    /// Paged KV cache: what happens when the pool runs short of blocks. <see cref="KvAdmissionPolicy.Queue"/> (default) admits a request only
    /// with room for its prompt and some growth, so new requests wait while the pool is full; <see cref="KvAdmissionPolicy.Preempt"/> admits on
    /// the prompt alone and preempts the newest running sequences (recomputing them later) when the running ones need blocks. The
    /// environment variable <c>SHARPLIGHT_KV_ADMISSION</c> (<c>queue</c>, <c>preempt</c>) overrides it for the process.
    /// </summary>
    public KvAdmissionPolicy KvAdmission { get; set; } = KvAdmissionPolicy.Queue;
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
    /// Models with recurrent state: least prompt tokens between two state snapshots kept where a prefill chunk ends anyway (the step
    /// budget cut it; such snapshots cost no pass). Within that bound the engine keeps the state at every such chunk end where recomputing
    /// the tokens since the previous snapshot would cost more than copying the state, by its estimate of step costs. Snapshots at prefixes
    /// shared with other prompts and at prompt ends are priced the same way (one more pass for the prompt against the recomputation they
    /// save). 0 (the default): the cost estimate alone decides.
    /// </summary>
    public int StateSnapshotInterval { get; set; }
    /// <summary>
    /// Prepare the decode steps of every batch size when the engine starts (<see cref="IBatchDecoder.PrepareDecodeSteps"/>; e.g. Vulkan
    /// records their command buffers), so the first steps of a load do not wait for it. Costs start-up time (Qwen3.5-0.8B at 64 slots:
    /// see docs/kv-cache.md section 2.6). Default on.
    /// </summary>
    public bool PrepareDecodeSteps { get; set; } = true;
    /// <summary>
    /// Image and audio prompts (<see cref="GenerationRequest.PromptMedia"/>): bytes of encoder outputs kept by content hash, so a repeated
    /// image whose KV blocks are no longer cached skips the encoder (default 256 MiB; 0 keeps none).
    /// </summary>
    public long MediaCacheBytes { get; set; } = 256L << 20;
    /// <summary>
    /// Image and audio prompts: most embedding rows encoded before one step (at least one content, default 0: the step's token budget). The
    /// encoder runs on the scheduler thread, so every running stream waits for it; a smaller budget spreads a burst of new images over steps.
    /// </summary>
    public int MediaEncodeRowsPerStep { get; set; }
    /// <summary>The step cost estimate the engine starts from and keeps learning (null: defaults); tests pass a frozen one.</summary>
    internal StepCostModel? StepCost { get; set; }
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
