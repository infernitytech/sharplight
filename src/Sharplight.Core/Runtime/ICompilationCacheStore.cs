namespace Sharplight.Runtime;

/// <summary>
/// Precompiled artifact sets stored with a model. Any number of sets for any providers and tiers; a loader consults the sets
/// that match a device's <see cref="IComputeDevice.ArtifactTargets"/> (see <see cref="CompilationArtifacts"/>) and ignores the rest.
/// </summary>
public interface ICompilationCacheStore
{
    /// <summary>The stored sets, newest first (a set stored later is consulted before older ones).</summary>
    public IReadOnlyList<StoredArtifactSet> ArtifactSets { get; }
}
