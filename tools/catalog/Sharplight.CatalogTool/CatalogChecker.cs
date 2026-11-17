using Sharplight.Hub;
using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>Checks over the network that every pinned file still exists at its commit with the recorded size and checksum.</summary>
/// <param name="hub">Reads repositories.</param>
/// <param name="log">Receives progress lines.</param>
internal sealed class CatalogChecker(HubClient hub, Action<string> log)
{
    /// <summary>Every problem found (missing repositories, commits or files, changed sizes or checksums).</summary>
    public async Task<IReadOnlyList<string>> CheckAsync(VerifiedCatalog catalog, CancellationToken cancellationToken)
    {
        var problems = new List<string>();
        var sets = catalog.Entries.SelectMany(entry => entry.Variants.Cast<CatalogFileSet>().Concat(entry.Companions).Select(set => (entry.Id, Set: set)));
        foreach (var group in sets.GroupBy(item => (item.Set.Repository, item.Set.Revision)))
        {
            log($"{group.Key.Repository} @ {group.Key.Revision}");
            HubRepository repository;
            try
            {
                repository = await hub.GetRepositoryAsync(group.Key.Repository, group.Key.Revision, cancellationToken).ConfigureAwait(false);
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
                    else if (remote.Size != file.Size || remote.Sha256 != file.Sha256 || remote.GitOid != file.GitOid)
                    {
                        problems.Add($"{entryId}: {set.Repository}/{file.Path} differs from the catalog (size or checksum)");
                    }
                }
            }
        }

        return problems;
    }
}
