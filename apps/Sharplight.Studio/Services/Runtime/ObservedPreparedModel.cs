using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>The prepared form of <see cref="ObservedLoader"/>.</summary>
internal sealed class ObservedPreparedModel(IPreparedModel inner, Action<LoadedModel> loaded) : IPreparedModel
{
    /// <inheritdoc/>
    public IReadOnlyList<MemoryRequirement> Requirements => inner.Requirements;

    /// <inheritdoc/>
    public bool CanOffload => inner.CanOffload;

    /// <inheritdoc/>
    public LoadedModel Load(long? primaryLimit)
    {
        var model = inner.Load(primaryLimit);
        loaded(model);
        return model;
    }

    /// <inheritdoc/>
    public void Dispose() => inner.Dispose();
}
