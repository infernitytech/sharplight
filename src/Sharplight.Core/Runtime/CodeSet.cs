namespace Sharplight.Runtime;

/// <summary>
/// Precompiled device code of one execution provider for one compilation stage: an opaque, keyed artifact set whose descriptor
/// (<see cref="ArtifactDescriptorKeys"/>) states what it was generated for (code target and profile, or device identity and driver)
/// and which kernel ABI it follows. The entries are provider defined and only meaningful to the provider that produced them; hosts
/// store them with the model (<see cref="ToArtifactSet"/>, the <c>code</c> and <c>binary</c> tiers of a <c>.slm</c> package) and
/// never interpret them. See <see cref="PortableCodeSet"/> and <see cref="DeviceCodeSet"/>.
/// </summary>
public abstract class CodeSet
{
    private protected CodeSet(string provider, string tier, IReadOnlyDictionary<string, string> descriptor,
        IReadOnlyDictionary<string, ReadOnlyMemory<byte>> entries, int reusedEntries)
    {
        ArgumentException.ThrowIfNullOrEmpty(provider);
        ArgumentNullException.ThrowIfNull(descriptor);
        ArgumentNullException.ThrowIfNull(entries);
        if (entries.Count > 0)
        {
            if (ArtifactDescriptor.Get(descriptor, ArtifactDescriptorKeys.Provider) != provider
                || ArtifactDescriptor.Get(descriptor, ArtifactDescriptorKeys.Tier) != tier
                || ArtifactDescriptor.Get(descriptor, ArtifactDescriptorKeys.Kind).Length == 0)
            {
                throw new ArgumentException($"A {tier}-tier code set of '{provider}' needs a descriptor with that provider, tier '{tier}' and a kind.",
                    nameof(descriptor));
            }
        }

        Provider = provider;
        Descriptor = new Dictionary<string, string>(descriptor, StringComparer.Ordinal);
        Entries = new Dictionary<string, ReadOnlyMemory<byte>>(entries, StringComparer.Ordinal);
        TotalBytes = Entries.Values.Sum(entry => (long)entry.Length);
        ReusedEntries = Math.Clamp(reusedEntries, 0, Entries.Count);
    }

    /// <summary>The execution provider that produced the code (<see cref="IExecutionProvider.Name"/>).</summary>
    public string Provider { get; }

    /// <summary>What the code was generated for; empty for an empty set of a provider that has nothing to generate.</summary>
    public IReadOnlyDictionary<string, string> Descriptor { get; }

    /// <summary>The code, by provider-defined key (content hashes for portable code).</summary>
    public IReadOnlyDictionary<string, ReadOnlyMemory<byte>> Entries { get; }

    /// <summary>Whether the set holds no code (always for providers without precompilable code, such as the CPU).</summary>
    public bool IsEmpty => Entries.Count == 0;

    /// <summary>Sum of the entry sizes in bytes.</summary>
    public long TotalBytes { get; }

    /// <summary>
    /// Entries that were served from code already stored with the model instead of being generated; equal to the entry count when
    /// the stored code was complete.
    /// </summary>
    public int ReusedEntries { get; }

    /// <summary>The artifact kind (e.g. <c>spirv</c>, <c>pipeline-binary</c>); empty for an empty set without a descriptor.</summary>
    public string Kind => ArtifactDescriptor.Get(Descriptor, ArtifactDescriptorKeys.Kind);

    /// <summary>The stable id of <see cref="Descriptor"/> (<see cref="ArtifactDescriptor.Id"/>): sets with equal ids hold interchangeable code.</summary>
    public string Id => Descriptor.Count == 0 ? "" : ArtifactDescriptor.Id(Descriptor);

    /// <summary>The set as an artifact set to store with a model (e.g. <c>ModelPackage.AddArtifacts</c>).</summary>
    public CompilationCache ToArtifactSet()
    {
        var set = new CompilationCache(Descriptor);
        foreach (var (key, data) in Entries)
        {
            set.Put(key, data);
        }

        return set;
    }

    /// <inheritdoc/>
    public override string ToString()
        => IsEmpty ? $"{Provider}: no code" : $"{ArtifactDescriptor.Format(Descriptor)} ({Entries.Count} entries, {TotalBytes} bytes)";
}
