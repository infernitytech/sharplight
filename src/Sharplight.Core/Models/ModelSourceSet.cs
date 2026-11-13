namespace Sharplight.Models;

/// <summary>A primary model source plus optional auxiliary sources (e.g. a vision projector "mmproj").</summary>
public sealed class ModelSourceSet
{
    /// <summary>The main model source.</summary>
    public required ModelSource Primary { get; init; }
    /// <summary>
    /// Additional sources by role (for example <see cref="Vision.AuxiliaryRoles.Mmproj"/>); empty when there are none.
    /// </summary>
    public IReadOnlyDictionary<string, ModelSource> Auxiliary { get; init; } = new Dictionary<string, ModelSource>();
}
