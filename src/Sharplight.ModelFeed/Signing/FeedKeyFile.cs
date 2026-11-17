namespace Sharplight.ModelFeed.Signing;

/// <summary>A key file (JSON): the public key, and the private key for signing keys. Private key files never belong in a repository.</summary>
public sealed record FeedKeyFile
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Key id (<see cref="FeedPublicKey.KeyId"/>).</summary>
    public required string KeyId { get; init; }

    /// <summary>Signature algorithm (<c>ES384</c>).</summary>
    public string Algorithm { get; init; } = FeedPublicKey.Algorithm;

    /// <summary>SubjectPublicKeyInfo, base64.</summary>
    public required string PublicKey { get; init; }

    /// <summary>PKCS#8 private key, base64; absent in public key files.</summary>
    public string? PrivateKey { get; init; }

    /// <summary>Free text (purpose, custodian).</summary>
    public string? Comment { get; init; }
}
