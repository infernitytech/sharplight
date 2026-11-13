namespace Sharplight.Llm.Serving;

public sealed record EngineOptions
{
    /// <summary>Maximum number of requests waiting for a slot; further submissions throw <see cref="QueueFullException"/>.</summary>
    public int MaxQueue { get; init; } = 256;
    /// <summary>Upper bound of rows per step (0 = the decoder's <see cref="IBatchDecoder.MaxBatchTokens"/>).</summary>
    public int MaxBatchTokens { get; init; }
    /// <summary>Maximum prompt rows of one sequence per step. Smaller chunks interleave long prefills with other sequences' decoding.</summary>
    public int PrefillChunk { get; init; } = 512;
    /// <summary>Reuse the cached context of idle slots for prompts sharing a prefix.</summary>
    public bool PrefixCache { get; init; } = true;
    /// <summary>Applied to requests without their own <see cref="GenerationRequest.Timeout"/>.</summary>
    public TimeSpan? DefaultTimeout { get; init; }
    public TimeSpan? DefaultQueueTimeout { get; init; }
}
