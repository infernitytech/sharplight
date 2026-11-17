namespace Sharplight.ModelFeed.Signing;

/// <summary>Protected header of a detached JWS signature.</summary>
public sealed record JwsHeader
{
    /// <summary>Algorithm (<c>ES384</c>).</summary>
    public required string Alg { get; init; }

    /// <summary>Signing key id.</summary>
    public required string Kid { get; init; }

    /// <summary>Content type of the signed document.</summary>
    public string? Typ { get; init; }
}
