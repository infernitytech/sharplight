using Sharplight.ModelFeed.Index;

namespace Sharplight.ModelFeed.Client;

/// <summary>A manifest whose bytes matched the feed's digest chain and, when required, its signature. Immutable.</summary>
public sealed class VerifiedManifest
{
    internal VerifiedManifest(PackageManifest manifest, byte[] bytes, byte[]? signature, string? keyId, FeedPackageVersion entry)
    {
        Manifest = manifest;
        Bytes = bytes;
        Signature = signature;
        KeyId = keyId;
        Entry = entry;
        Digest = FeedDigest.Of(bytes);
    }

    /// <summary>The parsed manifest.</summary>
    public PackageManifest Manifest { get; }

    /// <summary>The exact manifest bytes (store these, not a re-serialization).</summary>
    public ReadOnlyMemory<byte> Bytes { get; }

    /// <summary>The detached signature bytes, or <see langword="null"/> for an unsigned feed.</summary>
    public ReadOnlyMemory<byte>? Signature { get; }

    /// <summary>The key that signed the manifest, or <see langword="null"/>.</summary>
    public string? KeyId { get; }

    /// <summary>Whether a trusted signature verified the manifest.</summary>
    public bool Signed => KeyId is not null;

    /// <summary>The version's entry in the package index (current state, publication time).</summary>
    public FeedPackageVersion Entry { get; }

    /// <summary>Digest of <see cref="Bytes"/>.</summary>
    public string Digest { get; }
}
