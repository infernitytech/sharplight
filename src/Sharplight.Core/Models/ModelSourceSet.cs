namespace Sharplight.Models;

/// <summary>A primary model source plus optional auxiliary sources (e.g. a vision projector "mmproj").</summary>
public sealed class ModelSourceSet
{
    public required ModelSource Primary { get; init; }
    public IReadOnlyDictionary<string, ModelSource> Auxiliary { get; init; } = new Dictionary<string, ModelSource>();
}
