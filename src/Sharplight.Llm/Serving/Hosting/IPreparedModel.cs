namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A model whose size is known but that holds no device memory yet. Dispose without <see cref="Load"/> releases the host-side resources.</summary>
public interface IPreparedModel : IDisposable
{
    /// <summary>Memory needed per pool. The first entry is the primary device.</summary>
    public IReadOnlyList<MemoryRequirement> Requirements { get; }
    /// <summary>True when the model can be loaded into less than <see cref="Requirements"/> by moving layers to a fallback device (CPU).</summary>
    public bool CanOffload { get; }
    /// <summary>
    /// Compiles and uploads the model. <paramref name="primaryLimit"/> caps what the primary device may hold (the rest must be
    /// offloaded or the load fails); null = no cap beyond the device's own budget.
    /// </summary>
    public LoadedModel Load(long? primaryLimit);
}
