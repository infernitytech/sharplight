namespace Sharplight.ModelFeed.Signing;

/// <summary>The feed keys a root key vouches for (<c>keys/keyset.json</c>, signed by a root key in <c>keys/keyset.sig</c>).</summary>
public sealed record FeedKeySet
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Increasing key set version.</summary>
    public long Version { get; init; }

    /// <summary>When issued.</summary>
    public DateTimeOffset Issued { get; init; }

    /// <summary>When the key set stops being valid, or <see langword="null"/>.</summary>
    public DateTimeOffset? Expires { get; init; }

    /// <summary>Feed keys.</summary>
    public IReadOnlyList<FeedKeySetEntry> Keys { get; init; } = [];
}
