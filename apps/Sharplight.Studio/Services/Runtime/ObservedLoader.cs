using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>Wraps a loader to learn about each instance it loads (the instance itself is passed through unchanged).</summary>
internal sealed class ObservedLoader(IModelLoader inner, Action<LoadedModel> loaded) : IModelLoader
{
    /// <inheritdoc/>
    public ModelKind Kind => inner.Kind;

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct) => new ObservedPreparedModel(inner.Prepare(ct), loaded);
}
