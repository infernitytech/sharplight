namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Knows how to load one model. <see cref="Prepare"/> is cheap (reads metadata, maps files); <see cref="IPreparedModel.Load"/> allocates device memory.</summary>
public interface IModelLoader
{
    public ModelKind Kind { get; }
    public IPreparedModel Prepare(CancellationToken ct);
}
