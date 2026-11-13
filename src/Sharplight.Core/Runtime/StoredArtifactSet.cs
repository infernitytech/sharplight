namespace Sharplight.Runtime;

/// <summary>A read-only artifact set stored with a model (e.g. one section of a model package).</summary>
/// <param name="Descriptor">What the entries are and which devices may use them (see <see cref="ArtifactDescriptorKeys"/>).</param>
/// <param name="Entries">The stored entries; <see cref="ICompilationCache.Put"/> is not supported.</param>
public sealed record StoredArtifactSet(IReadOnlyDictionary<string, string> Descriptor, ICompilationCache Entries)
{
    /// <summary>The tier (<see cref="ArtifactTiers"/>), empty when the descriptor has none.</summary>
    public string Tier => ArtifactDescriptor.Get(Descriptor, ArtifactDescriptorKeys.Tier);
}
