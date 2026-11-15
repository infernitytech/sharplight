namespace Sharplight.Studio.Services.Runtime;

/// <summary>A model <see cref="EngineModelRuntime"/> registered on its host: the part of its life-cycle the runtime drives.</summary>
internal interface IEngineModel : ILoadedModel, IDisposable
{
    /// <summary>Receives load progress while <see cref="EngineModelRuntime.LoadAsync"/> waits for this model.</summary>
    public IProgress<LoadStatus>? LoadListener { get; set; }

    /// <summary>Cancelled to abandon a load in progress (the model is being removed).</summary>
    public CancellationTokenSource Abandon { get; }

    /// <summary>The host's models changed: frees what belongs to an instance that is gone.</summary>
    public void OnHostChanged();
}
