using Sharplight.ModelFeed.Client;

namespace Sharplight.ModelFeed.Resolution;

/// <summary>Candidate source for <see cref="DependencyResolver"/> from a verified feed snapshot (one package index read per id).</summary>
public static class FeedCandidates
{
    /// <summary>A candidate function over <paramref name="client"/> and <paramref name="snapshot"/>.</summary>
    /// <param name="client">The feed client.</param>
    /// <param name="snapshot">A verified snapshot of the feed.</param>
    /// <returns>The candidate function (unknown ids give an empty list).</returns>
    public static Func<string, CancellationToken, Task<IReadOnlyList<PackageCandidate>>> From(FeedClient client, FeedSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(client);
        ArgumentNullException.ThrowIfNull(snapshot);
        return async (id, cancellationToken) =>
        {
            if (snapshot.Find(id) is null)
            {
                return [];
            }

            var index = await client.GetPackageIndexAsync(snapshot, id, cancellationToken).ConfigureAwait(false);
            return [.. index.Versions.Select(version => new PackageCandidate
            {
                Id = id,
                Version = version.Version,
                Requirements = version.Requirements,
                Dependencies = version.Dependencies,
                State = version.State,
            })];
        };
    }
}
