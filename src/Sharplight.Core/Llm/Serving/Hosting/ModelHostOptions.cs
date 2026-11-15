namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Memory and lifetime policy of a <see cref="ModelHost"/>.</summary>
public sealed record ModelHostOptions
{
    /// <summary>Bytes kept free in every memory pool for other applications (a model is only loaded when it fits with this much to spare).</summary>
    public long ReserveBytes { get; init; } = 1L << 30;
    /// <summary>At most this many models loaded at once (0 = no limit); loading another evicts the least recently used idle one.</summary>
    public int MaxLoadedModels { get; init; }
    /// <summary>Unload models that served nothing for this long (null = never).</summary>
    public TimeSpan? IdleTimeout { get; init; }
    /// <summary>How often the idle sweep runs.</summary>
    public TimeSpan IdleCheckInterval { get; init; } = TimeSpan.FromSeconds(10);
    /// <summary>How long an admin unload waits for running requests before it gives up (or cuts them off when forced).</summary>
    public TimeSpan UnloadDrainTimeout { get; init; } = TimeSpan.FromSeconds(30);
    /// <summary>Clock and timers (replace for deterministic tests).</summary>
    public TimeProvider Time { get; init; } = TimeProvider.System;
    /// <summary>Receives load / unload / eviction messages.</summary>
    public Action<string>? Log { get; init; }
    /// <summary>
    /// On-demand registration: called (outside the host's locks) for a name that is not registered when a model is acquired; a
    /// returned registration is registered under that name and loaded like any other (e.g. a model of a local store, ollama-style).
    /// Null, or a null result, keeps the name unknown (<see cref="ModelNotFoundException"/>). The registration's name must equal
    /// the requested one.
    /// </summary>
    public Func<string, ModelRegistration?>? Resolve { get; init; }
}
