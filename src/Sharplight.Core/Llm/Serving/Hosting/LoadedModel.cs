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
    /// <summary>Releases the device memory.</summary>
    public abstract void Dispose();
}
