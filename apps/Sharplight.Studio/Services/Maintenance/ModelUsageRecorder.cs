using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Maintenance;

/// <summary>Records the load time of every model that becomes resident (the runtime reports changes of its list of loaded models).</summary>
internal sealed class ModelUsageRecorder : IDisposable
{
    private readonly IModelRuntime _runtime;
    private readonly IModelUsageStore _store;
    private readonly TimeProvider _time;
    private readonly HashSet<string> _seen = new(StringComparer.Ordinal);

    public ModelUsageRecorder(IModelRuntime runtime, IModelUsageStore store, TimeProvider? time = null)
    {
        _runtime = runtime;
        _store = store;
        _time = time ?? TimeProvider.System;
        runtime.ModelsChanged += OnChanged;
        OnChanged(this, EventArgs.Empty);
    }

    /// <inheritdoc/>
    public void Dispose() => _runtime.ModelsChanged -= OnChanged;

    private void OnChanged(object? sender, EventArgs e)
    {
        var paths = _runtime.LoadedModels.Select(model => model.Path).ToHashSet(StringComparer.Ordinal);
        lock (_seen)
        {
            foreach (var path in paths.Where(_seen.Add))
            {
                _store.Record(path, _time.GetUtcNow());
            }

            _seen.IntersectWith(paths);
        }
    }
}
