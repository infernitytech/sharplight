using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.ModelFeed.Client;

/// <summary>A verified view of a feed at one index sequence: the index, its key set and every package summary. Immutable.</summary>
public sealed class FeedSnapshot
{
    private readonly Dictionary<string, FeedPackageSummary> _byId;
    private readonly Lazy<FeedModelListing> _models;

    internal FeedSnapshot(string location, string? channel, FeedServiceIndex index, string indexDigest, FeedKeySet? keySet, bool verified, IReadOnlyList<FeedPackageSummary> packages, IReadOnlyList<string> warnings)
    {
        Location = location;
        Channel = channel;
        Index = index;
        IndexDigest = indexDigest;
        KeySet = keySet;
        Verified = verified;
        Packages = packages;
        Warnings = warnings;
        _byId = packages.ToDictionary(package => package.Id, StringComparer.Ordinal);
        _models = new Lazy<FeedModelListing>(() => FeedModelListing.From(packages));
    }

    /// <summary>The feed location.</summary>
    public string Location { get; }

    /// <summary>The channel, or <see langword="null"/> for a feed without channels.</summary>
    public string? Channel { get; }

    /// <summary>The service index.</summary>
    public FeedServiceIndex Index { get; }

    /// <summary>Digest of the service index bytes.</summary>
    public string IndexDigest { get; }

    /// <summary>The verified key set, or <see langword="null"/> when signatures were not checked.</summary>
    public FeedKeySet? KeySet { get; }

    /// <summary>Whether the index was verified by a trusted signature.</summary>
    public bool Verified { get; }

    /// <summary>All package summaries, ordered by id.</summary>
    public IReadOnlyList<FeedPackageSummary> Packages { get; }

    /// <summary>The packages grouped into models with their variants (<see cref="FeedModelListing.From"/> over <see cref="Packages"/>).</summary>
    public FeedModelListing Models => _models.Value;

    /// <summary>Accepted problems (expired index in offline mode, unsigned feed).</summary>
    public IReadOnlyList<string> Warnings { get; }

    /// <summary>The summary of a package, or <see langword="null"/>.</summary>
    /// <param name="id">Package id.</param>
    /// <returns>The summary.</returns>
    public FeedPackageSummary? Find(string id) => _byId.GetValueOrDefault(id);
}
