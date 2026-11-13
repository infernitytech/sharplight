namespace Sharplight.Runtime;

/// <summary>
/// One tier of precompiled artifacts a device can produce and consume (see <see cref="IComputeDevice.ArtifactTargets"/>): the
/// descriptor stored with sets it produces, the descriptor keys a stored set must agree on to be consulted, and which cache keys
/// belong to this tier.
/// </summary>
public sealed class ArtifactTarget
{
    /// <summary>Creates a target.</summary>
    /// <param name="descriptor">Descriptor of sets produced for this target; must contain <see cref="ArtifactDescriptorKeys.Provider"/>,
    /// <see cref="ArtifactDescriptorKeys.Tier"/> and <see cref="ArtifactDescriptorKeys.Kind"/>.</param>
    /// <param name="matchKeys">Descriptor keys a stored set must have with equal values to be consulted. Provider, tier and kind are always
    /// compared. Keep this minimal for content-addressed entries (a hit is always valid), complete for opaque binaries.</param>
    /// <param name="keyPrefixes">Cache keys starting with one of these belong to this tier; empty = every key no other target of the
    /// device claims.</param>
    /// <exception cref="ArgumentException">The descriptor lacks provider, tier or kind.</exception>
    public ArtifactTarget(IReadOnlyDictionary<string, string> descriptor, IEnumerable<string> matchKeys, IEnumerable<string> keyPrefixes)
    {
        foreach (var required in new[] { ArtifactDescriptorKeys.Provider, ArtifactDescriptorKeys.Tier, ArtifactDescriptorKeys.Kind })
        {
            if (string.IsNullOrEmpty(ArtifactDescriptor.Get(descriptor, required)))
            {
                throw new ArgumentException($"An artifact descriptor needs '{required}'.", nameof(descriptor));
            }
        }

        Descriptor = new Dictionary<string, string>(descriptor, StringComparer.Ordinal);
        MatchKeys = [.. new[] { ArtifactDescriptorKeys.Provider, ArtifactDescriptorKeys.Tier, ArtifactDescriptorKeys.Kind }.Concat(matchKeys).Distinct(StringComparer.Ordinal)];
        KeyPrefixes = [.. keyPrefixes];
        Id = ArtifactDescriptor.Id(Descriptor);
    }

    /// <summary>Descriptor stored with artifact sets produced for this target.</summary>
    public IReadOnlyDictionary<string, string> Descriptor { get; }

    /// <summary>Descriptor keys a stored set must agree on (always including provider, tier and kind).</summary>
    public IReadOnlyList<string> MatchKeys { get; }

    /// <summary>Prefixes of the cache keys of this tier; empty claims every key no other target claims.</summary>
    public IReadOnlyList<string> KeyPrefixes { get; }

    /// <summary>Stable id of <see cref="Descriptor"/> (<see cref="ArtifactDescriptor.Id"/>).</summary>
    public string Id { get; }

    /// <summary>The provider name.</summary>
    public string Provider => Descriptor[ArtifactDescriptorKeys.Provider];

    /// <summary>The tier (<see cref="ArtifactTiers"/>).</summary>
    public string Tier => Descriptor[ArtifactDescriptorKeys.Tier];

    /// <summary>The artifact kind.</summary>
    public string Kind => Descriptor[ArtifactDescriptorKeys.Kind];

    /// <summary>Whether a stored set with <paramref name="stored"/> as descriptor may be consulted for this target.</summary>
    public bool Matches(IReadOnlyDictionary<string, string> stored)
        => MatchKeys.All(key => Descriptor.TryGetValue(key, out var mine) && stored.TryGetValue(key, out var theirs) && mine == theirs);

    /// <summary>Whether <paramref name="key"/> starts with one of <see cref="KeyPrefixes"/>.</summary>
    public bool Claims(string key) => KeyPrefixes.Any(prefix => key.StartsWith(prefix, StringComparison.Ordinal));

    /// <inheritdoc/>
    public override string ToString() => ArtifactDescriptor.Format(Descriptor);
}
