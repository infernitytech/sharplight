namespace Sharplight.ModelFeed.Signing;

/// <summary>
/// Verifies a feed's key set against trusted root keys and resolves the feed keys it vouches for (TUF-style, minimal: roots sign
/// the key set, feed keys sign indexes and manifests). Immutable and thread safe.
/// </summary>
public sealed class FeedTrust
{
    private readonly Dictionary<string, FeedPublicKey> _roots;

    /// <summary>Creates the trust from root keys.</summary>
    /// <param name="roots">Trusted root keys (compiled into the host or configured by an administrator).</param>
    public FeedTrust(IEnumerable<FeedPublicKey> roots)
    {
        ArgumentNullException.ThrowIfNull(roots);
        _roots = roots.DistinctBy(key => key.KeyId).ToDictionary(key => key.KeyId, StringComparer.Ordinal);
    }

    /// <summary>The trusted root keys.</summary>
    public IReadOnlyCollection<FeedPublicKey> Roots => _roots.Values;

    /// <summary>Verifies a key set document and its root signature.</summary>
    /// <param name="keySet">Key set bytes.</param>
    /// <param name="signature">Root signature bytes.</param>
    /// <param name="now">Current time, for the key set's expiry.</param>
    /// <returns>The verified key set.</returns>
    /// <exception cref="FeedVerificationException">The signature is invalid, the root is not trusted, or the key set has expired.</exception>
    public FeedKeySet VerifyKeySet(ReadOnlySpan<byte> keySet, ReadOnlySpan<byte> signature, DateTimeOffset now)
    {
        DetachedSignature.Verify(keySet, signature, kid => _roots.GetValueOrDefault(kid), "The feed key set");
        var parsed = FeedJson.Read(keySet, FeedJsonContext.Default.FeedKeySet, "The feed key set");
        if (parsed.Expires is { } expires && expires < now)
        {
            throw new FeedVerificationException($"The feed key set expired on {expires:u}.");
        }

        return parsed;
    }

    /// <summary>The feed key with <paramref name="keyId"/> when the key set lists it as valid at <paramref name="at"/>, else <see langword="null"/>.</summary>
    /// <param name="keySet">A verified key set.</param>
    /// <param name="keyId">Key id from a signature.</param>
    /// <param name="at">Time the key must be valid at.</param>
    /// <returns>The key, or <see langword="null"/>.</returns>
    public static FeedPublicKey? ResolveFeedKey(FeedKeySet keySet, string keyId, DateTimeOffset at)
    {
        ArgumentNullException.ThrowIfNull(keySet);
        var entry = keySet.Keys.FirstOrDefault(key => key.KeyId == keyId);
        if (entry is null || entry.Revoked is not null || (entry.NotBefore is { } from && at < from) || (entry.NotAfter is { } until && at > until))
        {
            return null;
        }

        var key = FeedPublicKey.FromBase64(entry.PublicKey);
        return key.KeyId == keyId ? key : null;
    }
}
