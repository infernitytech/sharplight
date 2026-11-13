namespace Sharplight.Llm.Serving.Hosting;

public sealed record ModelHostOptions
{
    /// <summary>Bytes kept free in every memory pool for other applications (a model is only loaded when it fits with this much to spare).</summary>
    public long ReserveBytes { get; init; } = 1L << 30;
    /// <summary>At most this many models loaded at once (0 = no limit); loading another evicts the least recently used idle one.</summary>
    public int MaxLoadedModels { get; init; }
    /// <summary>Unload models that served nothing for this long (null = never).</summary>
    public TimeSpan? IdleTimeout { get; init; }
    public TimeSpan IdleCheckInterval { get; init; } = TimeSpan.FromSeconds(10);
    /// <summary>How long an admin unload waits for running requests before it gives up (or cuts them off when forced).</summary>
    public TimeSpan UnloadDrainTimeout { get; init; } = TimeSpan.FromSeconds(30);
    public TimeProvider Time { get; init; } = TimeProvider.System;
    public Action<string>? Log { get; init; }
}
