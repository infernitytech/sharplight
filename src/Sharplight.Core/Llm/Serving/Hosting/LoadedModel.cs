namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A model that finished loading: owns its device memory until disposed.</summary>
public abstract class LoadedModel : IDisposable
{
    /// <summary>What the model does.</summary>
    public abstract ModelKind Kind { get; }
    /// <summary>Device bytes this model holds, as far as known (0 = unknown).</summary>
    public virtual long ResidentBytes => 0;
    /// <summary>True when nothing is running on it (so it can be unloaded without cutting requests off).</summary>
    public virtual bool IsIdle => true;
    /// <summary>Stops accepting work and waits up to <paramref name="timeout"/> for running work; used before an unload.</summary>
    /// <param name="timeout">How long to wait for running work.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    /// <returns>True when the work finished in time.</returns>
    public virtual Task<bool> DrainAsync(TimeSpan timeout, CancellationToken cancellationToken = default) => Task.FromResult(true);
    /// <summary>
    /// The routed experts the model reads from disk (<see cref="Tasks.TaskLoadOptions.ExpertsOnDisk"/>), or null when it has none on disk (the option
    /// off, or the experts fit memory). Its <see cref="Runtime.Experts.ExpertStore.Statistics"/> has the cache's hit rate and the bytes read.
    /// </summary>
    public virtual Runtime.Experts.ExpertStore? DiskExperts => null;
    /// <summary>Releases the device memory.</summary>
    public abstract void Dispose();
}
