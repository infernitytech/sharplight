using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Keeps several models resident on the devices the registered execution providers offer.</summary>
internal interface IModelRuntime : IDisposable
{
    /// <summary>Every device models can be loaded on (GPUs first, then the CPU).</summary>
    public IReadOnlyList<ComputeDeviceOption> Devices { get; }

    /// <summary>The registered models (loaded, loading or sleeping), in registration order.</summary>
    public IReadOnlyList<ILoadedModel> LoadedModels { get; }

    /// <summary>Raised (on any thread) when a model was added or removed or changed state.</summary>
    public event EventHandler? ModelsChanged;

    /// <summary>Raised (on any thread) when a generation request of any model finished (chat or API).</summary>
    public event EventHandler<RequestRecord>? RequestCompleted;

    /// <summary>Adds and loads a model, reporting stages and kernel compilation to <paramref name="progress"/>. Cancelling abandons the load and removes the model.</summary>
    /// <exception cref="InvalidOperationException">A model with the same id exists.</exception>
    public Task<ILoadedModel> LoadAsync(ModelLoadRequest request, IProgress<LoadStatus>? progress, CancellationToken cancellationToken);

    /// <summary>Waits for running requests (at most a few seconds), frees the model's memory and removes it.</summary>
    public Task UnloadAsync(string id);

    /// <summary>Memory use of the devices opened so far.</summary>
    public IReadOnlyList<DeviceUsage> GetDeviceUsage();
}
