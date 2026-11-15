namespace Sharplight.Runtime;

/// <summary>
/// Stage one of precompilation (<see cref="IExecutionProvider.CompilePortableAsync"/>): device code in the provider's portable format
/// (<see cref="IExecutionProvider.PortableCodeFormat"/>, e.g. SPIR-V for Vulkan, later AIR for Metal), generated for a
/// <see cref="DeviceProfile"/> without a device or driver. Stored as the <see cref="ArtifactTiers.Code"/> tier. Entries are content
/// addressed, so a set may be shared by every device with the same profile and is safe to consult for any of them.
/// </summary>
public sealed class PortableCodeSet : CodeSet
{
    /// <summary>Creates a portable code set.</summary>
    /// <param name="provider">The provider that generated the code.</param>
    /// <param name="descriptor">Code-tier descriptor (provider, tier <see cref="ArtifactTiers.Code"/>, kind, target, profile keys); may be
    /// empty when <paramref name="entries"/> is.</param>
    /// <param name="entries">The code by key.</param>
    /// <param name="reusedEntries">Entries served from code already stored with the model.</param>
    /// <exception cref="ArgumentException">The set has entries but the descriptor is not a code-tier descriptor of the provider.</exception>
    public PortableCodeSet(string provider, IReadOnlyDictionary<string, string> descriptor, IReadOnlyDictionary<string, ReadOnlyMemory<byte>> entries,
        int reusedEntries = 0)
        : base(provider, ArtifactTiers.Code, descriptor, entries, reusedEntries)
    {
    }

    /// <summary>The portable format of the code (the descriptor's <see cref="ArtifactDescriptorKeys.Target"/>, e.g. <c>spirv_1_6</c>); empty for an empty set.</summary>
    public string Format => ArtifactDescriptor.Get(Descriptor, ArtifactDescriptorKeys.Target);

    /// <summary>An empty set: the provider has no portable stage (CPU) or the model needs no device code.</summary>
    /// <param name="provider">The provider name.</param>
    public static PortableCodeSet Empty(string provider) => new(provider, new Dictionary<string, string>(), new Dictionary<string, ReadOnlyMemory<byte>>());

    /// <summary>A portable code set from a recorded or stored artifact set of the code tier.</summary>
    /// <param name="set">The set; its descriptor names the provider.</param>
    /// <exception cref="ArgumentException">The set is not of the code tier.</exception>
    public static PortableCodeSet FromArtifactSet(CompilationCache set)
    {
        ArgumentNullException.ThrowIfNull(set);
        int reused = set.FallbackHits;
        return new PortableCodeSet(ArtifactDescriptor.Get(set.Descriptor, ArtifactDescriptorKeys.Provider), set.Descriptor, set.Entries, reused);
    }
}
