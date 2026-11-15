using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>Artifact sets "stored with a model" in memory, the last one newest.</summary>
/// <param name="sets">The sets, oldest first.</param>
internal sealed class MemoryStore(IReadOnlyList<CompilationCache> sets) : ICompilationCacheStore
{
    public IReadOnlyList<StoredArtifactSet> ArtifactSets { get; } = [.. sets.Reverse().Select(set => new StoredArtifactSet(set.Descriptor, set))];
}
