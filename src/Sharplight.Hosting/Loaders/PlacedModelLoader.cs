using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>
/// A loader that asks the placement policy for devices each time the model is prepared, then prepares it with a loader created for
/// that placement (so <c>auto</c> placement follows the memory that is free at load time).
/// </summary>
/// <param name="kind">The kind of model the inner loaders produce.</param>
/// <param name="context">The model and the placement policy.</param>
/// <param name="estimatedBytes">A rough footprint for choosing devices.</param>
/// <param name="create">Creates the loader for one placement.</param>
public sealed class PlacedModelLoader(ModelKind kind, HostedModelLoadContext context, Func<long> estimatedBytes, Func<DevicePlacement, IHostedModelLoader> create)
    : IHostedModelLoader
{
    /// <inheritdoc/>
    public ModelKind Kind => kind;

    /// <inheritdoc/>
    public IPreparedModel Prepare(CancellationToken ct)
    {
        var placement = context.Placement.Place(context.Settings, estimatedBytes());
        return create(placement).Prepare(ct);
    }
}
