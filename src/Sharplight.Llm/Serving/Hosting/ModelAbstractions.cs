using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>What a registered model does. Chat models serve the completion APIs; the others are reserved for model types that other components register.</summary>
public enum ModelKind { Chat, Embedding, Reranker }

public enum ModelState { Unloaded, Loading, Loaded, Unloading, Failed }

/// <summary>A pool of memory models are charged against (normally one GPU). Same <see cref="Id"/> = same pool.</summary>
public interface IMemoryDomain
{
    string Id { get; }
    string Name { get; }
    DeviceMemoryInfo GetMemoryInfo();
}

/// <summary><see cref="IMemoryDomain"/> over an opened compute device (live budget from the driver when it has one).</summary>
public sealed class DeviceMemoryDomain(IComputeDevice device) : IMemoryDomain
{
    public IComputeDevice Device { get; } = device;
    public string Id { get; } = $"{device.Info.Provider}:{device.Info.Index}";
    public string Name => Device.Info.Name;
    public DeviceMemoryInfo GetMemoryInfo() => Device.GetMemoryInfo();
}

/// <param name="Domain">Where the memory is needed.</param>
/// <param name="Bytes">Estimated bytes (weights + state + activations).</param>
public sealed record MemoryRequirement(IMemoryDomain Domain, long Bytes);

/// <summary>A model that finished loading: owns its device memory until disposed.</summary>
public abstract class LoadedModel : IDisposable
{
    public abstract ModelKind Kind { get; }
    /// <summary>Device bytes this model holds, as far as known (0 = unknown).</summary>
    public virtual long ResidentBytes => 0;
    /// <summary>True when nothing is running on it (so it can be unloaded without cutting requests off).</summary>
    public virtual bool IsIdle => true;
    /// <summary>Stops accepting work and waits up to <paramref name="timeout"/> for running work; used before an unload.</summary>
    public virtual Task<bool> DrainAsync(TimeSpan timeout) => Task.FromResult(true);
    public abstract void Dispose();
}

/// <summary>Knows how to load one model. <see cref="Prepare"/> is cheap (reads metadata, maps files); <see cref="IPreparedModel.Load"/> allocates device memory.</summary>
public interface IModelLoader
{
    ModelKind Kind { get; }
    IPreparedModel Prepare(CancellationToken ct);
}

/// <summary>A model whose size is known but that holds no device memory yet. Dispose without <see cref="Load"/> releases the host-side resources.</summary>
public interface IPreparedModel : IDisposable
{
    /// <summary>Memory needed per pool. The first entry is the primary device.</summary>
    IReadOnlyList<MemoryRequirement> Requirements { get; }
    /// <summary>True when the model can be loaded into less than <see cref="Requirements"/> by moving layers to a fallback device (CPU).</summary>
    bool CanOffload { get; }
    /// <summary>
    /// Compiles and uploads the model. <paramref name="primaryLimit"/> caps what the primary device may hold (the rest must be
    /// offloaded or the load fails); null = no cap beyond the device's own budget.
    /// </summary>
    LoadedModel Load(long? primaryLimit);
}

/// <summary>Registration of a model under an API-visible name.</summary>
public sealed class ModelRegistration
{
    public required string Name { get; init; }
    public required IModelLoader Loader { get; init; }
    /// <summary>Load when the host starts (<see cref="ModelHost.LoadStartupModelsAsync"/>) instead of on the first request.</summary>
    public bool LoadAtStartup { get; init; }
    /// <summary>Never evicted for memory and never unloaded when idle.</summary>
    public bool Pinned { get; init; }
    /// <summary>Overrides <see cref="ModelHostOptions.IdleTimeout"/> for this model.</summary>
    public TimeSpan? IdleTimeout { get; init; }
    public string? Description { get; init; }
    public ModelKind Kind => Loader.Kind;
}

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

/// <summary>Read-only view of a registered model for listings and metrics.</summary>
public sealed record ModelInfo(
    string Name, ModelKind Kind, ModelState State, bool Pinned, bool LoadAtStartup, long ResidentBytes, long EstimatedBytes, int ActiveLeases,
    DateTimeOffset? LoadedAt, DateTimeOffset? LastUsed, string? Error, string? Description);

public class ModelHostException(string message, string code) : EngineException(message, code);
public sealed class ModelNotFoundException(string name) : ModelHostException($"The model '{name}' does not exist.", "model_not_found") { public string ModelName { get; } = name; }
public sealed class ModelKindMismatchException(string name, ModelKind actual, ModelKind wanted)
    : ModelHostException($"The model '{name}' is a {actual} model and cannot be used for {wanted} requests.", "model_not_found");
public sealed class ModelBusyException(string name, string detail) : ModelHostException($"The model '{name}' is busy: {detail}", "model_busy");
public sealed class ModelDoesNotFitException(string name, long needed, long available, long reserve, string detail)
    : ModelHostException(
        $"The model '{name}' needs about {needed / 1048576.0:F0} MiB of device memory but only {Math.Max(0, available) / 1048576.0:F0} MiB are free after keeping {reserve / 1048576.0:F0} MiB in reserve. {detail}",
        "insufficient_memory")
{
    public long Needed { get; } = needed;
    public long Available { get; } = available;
}
