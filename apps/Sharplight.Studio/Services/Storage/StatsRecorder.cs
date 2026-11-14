using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Storage;

/// <summary>Stores every request the runtime reports, off the engine thread.</summary>
internal sealed class StatsRecorder : IDisposable
{
    private readonly IModelRuntime _runtime;
    private readonly IStatsStore _store;

    public StatsRecorder(IModelRuntime runtime, IStatsStore store)
    {
        _runtime = runtime;
        _store = store;
        runtime.RequestCompleted += OnCompleted;
    }

    /// <inheritdoc/>
    public void Dispose() => _runtime.RequestCompleted -= OnCompleted;

    private void OnCompleted(object? sender, RequestRecord record)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await _store.AddAsync(record);
            }
            catch (Exception)
            {
                // Statistics are best effort; a locked or full database must not break generation.
            }
        });
    }
}
