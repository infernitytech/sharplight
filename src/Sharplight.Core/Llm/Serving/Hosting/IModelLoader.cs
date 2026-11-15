namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Knows how to load one model. <see cref="Prepare"/> is cheap (reads metadata, maps files); <see cref="IPreparedModel.Load"/> allocates device memory.</summary>
public interface IModelLoader
{
    /// <summary>The kind of model this loader produces.</summary>
    public ModelKind Kind { get; }
    /// <summary>Reads the model's metadata and sizes it without allocating device memory.</summary>
    public IPreparedModel Prepare(CancellationToken ct);
}
