using Sharplight.Runtime;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Keeps several models resident on the devices the registered execution providers offer.</summary>
internal interface IModelRuntime : IDisposable
{
    /// <summary>
    /// Every device models can be loaded on (GPUs first, then the CPU); with several GPUs also the choice that splits a model over
    /// all of them (<see cref="ComputeDeviceOption.AllGpusId"/>).
    /// </summary>
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

    /// <summary>
    /// The companion files a model file needs that the engine's importers find beside it, by auxiliary role (a transformer-only FLUX
    /// file: the <c>text-encoder</c>, <c>text-encoder-2</c> and <c>vae</c> files); empty when it needs none or none were found.
    /// </summary>
    /// <param name="path">The model file.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string path) => new Dictionary<string, string>();

    /// <summary>Waits for running requests (at most a few seconds), frees the model's memory and removes it.</summary>
    public Task UnloadAsync(string id);

    /// <summary>Memory use of the devices opened so far.</summary>
    public IReadOnlyList<DeviceUsage> GetDeviceUsage();

    /// <summary>
    /// The engine's estimate of the memory loading <paramref name="request"/> would take (weights, KV cache and state, activations),
    /// computed from the model's graph for its settings; nothing is allocated on a device.
    /// </summary>
    /// <param name="request">The model and settings; the context length, KV cache type, slots, batch and drafts count.</param>
    /// <param name="cancellationToken">Cancels reading the model.</param>
    public Task<ModelFootprint> EstimateFootprintAsync(ModelLoadRequest request, CancellationToken cancellationToken);

    /// <summary>Free and total memory of every physical device (GPUs first, then the CPU's system memory), opening devices as needed.</summary>
    /// <param name="disabledFeatures">Device features not to use, for devices opened by this call.</param>
    /// <param name="cancellationToken">Cancels the query.</param>
    public Task<IReadOnlyList<DeviceMemory>> GetDeviceMemoryAsync(IReadOnlyList<string> disabledFeatures, CancellationToken cancellationToken);
}
