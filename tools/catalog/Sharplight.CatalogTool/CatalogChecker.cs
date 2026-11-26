using Sharplight.Hub;
using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>Checks over the network that every pinned file still exists at its commit with the recorded size and checksum.</summary>
/// <param name="hub">Reads repositories.</param>
/// <param name="log">Receives progress lines.</param>
/// <param name="github">Reads the sets that come from GitHub, or <see langword="null"/> to report them as unchecked.</param>
internal sealed class CatalogChecker(HubClient hub, Action<string> log, GitHubReader? github = null)
{
    /// <summary>Every problem found (missing repositories, commits or files, changed sizes or checksums).</summary>
    public async Task<IReadOnlyList<string>> CheckAsync(VerifiedCatalog catalog, CancellationToken cancellationToken)
    {
        var problems = new List<string>();
        var sets = catalog.Entries.SelectMany(entry => entry.Variants.Cast<CatalogFileSet>().Concat(entry.Companions).Select(set => (entry.Id, Set: set)));
        foreach (var group in sets.GroupBy(item => (item.Set.Repository, item.Set.Revision, item.Set.Source, item.Set.Release)))
        {
            log($"{group.Key.Repository} @ {group.Key.Revision}");
            HubRepository repository;
            try
            {
                repository = group.First().Set.IsHubSource
                    ? await hub.GetRepositoryAsync(group.Key.Repository, group.Key.Revision, cancellationToken).ConfigureAwait(false)
                    : await GetGitHubFilesAsync(group.First().Set, cancellationToken).ConfigureAwait(false);
            }
            catch (HubException error)
            {
                problems.Add($"{string.Join(", ", group.Select(item => item.Id).Distinct())}: {group.Key.Repository}@{group.Key.Revision}: {error.Message}");
                continue;
            }

            var files = repository.Files.ToDictionary(file => file.Path, StringComparer.Ordinal);
            foreach (var (entryId, set) in group)
            {
                foreach (var file in set.Files)
                {
                    if (!files.TryGetValue(file.Path, out var remote))
                    {
                        problems.Add($"{entryId}: {set.Repository}/{file.Path} no longer exists at {set.Revision}");
                    }
                    else if (remote.Size != file.Size || !Same(remote.Sha256, file.Sha256) || !Same(remote.GitOid, file.GitOid))
                    {
                        problems.Add($"{entryId}: {set.Repository}/{file.Path} differs from the catalog (size or checksum)");
                    }
                }
            }
        }

        return problems;
    }

    /// <summary>A checksum matches unless the source lists one that differs: GitHub lists git blob ids for repository files and sha256 digests for release assets only.</summary>
    private static bool Same(string? remote, string? recorded) => remote is null || remote == recorded;

    /// <summary>The files of a GitHub set as the Hub's would be listed: the repository at the pinned commit (checked to be the tag's), or the release's assets.</summary>
    private async Task<HubRepository> GetGitHubFilesAsync(CatalogFileSet set, CancellationToken cancellationToken)
    {
        if (github is null)
        {
            throw new HubException("GitHub sets are not checked without a GitHub reader.");
        }

        if (set.Release is { } tag && await github.ResolveCommitAsync(set.Repository, tag, cancellationToken).ConfigureAwait(false) != set.Revision)
        {
            throw new HubException($"the release tag {tag} no longer points at {set.Revision}.");
        }

        var files = set.Source == CatalogSources.GitHubRelease
            ? await github.ListReleaseAssetsAsync(set.Repository, set.Release ?? "", cancellationToken).ConfigureAwait(false)
            : await github.ListFilesAsync(set.Repository, set.Revision, cancellationToken).ConfigureAwait(false);
        return new HubRepository(set.Repository, set.Revision, set.Revision, false, files);
    }
}
