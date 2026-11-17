using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>A prepared model sized from its files (spread evenly over its devices) that is read only when it is loaded.</summary>
/// <param name="requirements">Memory per device.</param>
/// <param name="load">Reads the model and returns the loaded instance.</param>
internal sealed class FileSizedPreparedModel(IReadOnlyList<MemoryRequirement> requirements, Func<LoadedModel> load) : IPreparedModel
{
    /// <summary>Spreads <paramref name="bytes"/> evenly over <paramref name="devices"/>.</summary>
    public static IReadOnlyList<MemoryRequirement> Spread(long bytes, IReadOnlyList<IMemoryDomain> devices)
    {
        long share = bytes / Math.Max(1, devices.Count);
        return [.. devices.Select(device => new MemoryRequirement(device, share))];
    }

    public IReadOnlyList<MemoryRequirement> Requirements => requirements;

    public bool CanOffload => false;

    public LoadedModel Load(long? primaryLimit) => load();

    public void Dispose()
    {
        // Nothing is held before Load.
    }
}
