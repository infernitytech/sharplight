namespace Sharplight.ModelFeed.Signing;

/// <summary>A feed key with its validity.</summary>
public sealed record FeedKeySetEntry
{
    /// <summary>Key id.</summary>
    public required string KeyId { get; init; }

    /// <summary>SubjectPublicKeyInfo, base64.</summary>
    public required string PublicKey { get; init; }

    /// <summary>Valid from.</summary>
    public DateTimeOffset? NotBefore { get; init; }

    /// <summary>Valid until.</summary>
    public DateTimeOffset? NotAfter { get; init; }

    /// <summary>Revoked since (signatures are rejected).</summary>
    public DateTimeOffset? Revoked { get; init; }

    /// <summary>Free text.</summary>
    public string? Comment { get; init; }
}
