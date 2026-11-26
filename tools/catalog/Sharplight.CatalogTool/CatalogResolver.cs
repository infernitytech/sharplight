using Sharplight.Hub;
using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>
/// Fills in what the Hub knows about the catalog's file sets: the commit a revision resolves to, the exact files (chosen by the
/// set's <see cref="CatalogFileSet.Include"/> patterns or by <see cref="ModelFileSelector"/> with its tag) with sizes and checksums,
/// the primary path, the license and gated flag, and the memory estimate. Sets that are already resolved are kept unless asked.
/// </summary>
/// <param name="hub">Reads repositories.</param>
/// <param name="log">Receives progress lines.</param>
/// <param name="github">Reads the sets that come from GitHub, or <see langword="null"/> to report them as failures.</param>
internal sealed class CatalogResolver(HubClient hub, Action<string> log, GitHubReader? github = null)
{
    private const string GitHubDefaultRef = "HEAD";
    private const string DefaultRevision = "main";
    private const int MaxBaseModelDepth = 3;
    private readonly Dictionary<(string Repository, string Revision), HubRepository> _repositories = [];
    private readonly List<string> _failures = [];

    /// <summary>The file sets that could not be resolved (left as they were), one line each.</summary>
    public IReadOnlyList<string> Failures => _failures;

    /// <summary>Resolves the entries of <paramref name="catalog"/>.</summary>
    /// <param name="catalog">The catalog.</param>
    /// <param name="mode">What to resolve again.</param>
    /// <param name="entryId">Only this entry, or <see langword="null"/> for all.</param>
    /// <param name="cancellationToken">Cancels the requests.</param>
    public async Task<VerifiedCatalog> ResolveAsync(VerifiedCatalog catalog, ResolveMode mode, string? entryId, CancellationToken cancellationToken)
    {
        var entries = new List<CatalogEntry>();
        foreach (var entry in catalog.Entries)
        {
            bool selected = entryId is null || string.Equals(entry.Id, entryId, StringComparison.OrdinalIgnoreCase);
            entries.Add(selected ? await ResolveEntryAsync(entry, mode, cancellationToken).ConfigureAwait(false) : entry);
        }

        return catalog with { Entries = entries };
    }

    private async Task<CatalogEntry> ResolveEntryAsync(CatalogEntry entry, ResolveMode mode, CancellationToken cancellationToken)
    {
        // Variants first: the license of the entry is that of the model's own repositories, not of a companion's.
        var repositories = new List<HubRepository>();
        var variants = new List<CatalogVariant>();
        foreach (var variant in entry.Variants)
        {
            var (resolved, repository) = await ResolveSetAsync(entry.Id, variant, mode, cancellationToken).ConfigureAwait(false);
            variants.Add(resolved);
            repositories.AddIfNotNull(repository);
        }

        var companions = new List<CatalogCompanion>();
        foreach (var companion in entry.Companions)
        {
            var (resolved, repository) = await ResolveSetAsync(entry.Id, companion, mode, cancellationToken).ConfigureAwait(false);
            companions.Add(resolved);
            repositories.AddIfNotNull(repository);
        }

        if (entry.License is null && repositories.Count == 0 && variants.FirstOrDefault() is { Revision.Length: 40, IsHubSource: true } first)
        {
            repositories.Add(await GetRepositoryAsync(first.Repository, first.Revision, cancellationToken).ConfigureAwait(false));
        }

        var updated = entry with { Companions = companions, Variants = variants };
        updated = updated with
        {
            Variants = [.. variants.Select(variant => variant.MemoryBytes > 0 && mode == ResolveMode.Missing ? variant : variant with { MemoryBytes = MemoryEstimate.For(updated, variant) })],
            License = entry.License ?? await LicenseOfAsync(repositories, cancellationToken).ConfigureAwait(false),
            Gated = entry.Gated || repositories.Any(repository => repository.Gated),
        };
        return updated;
    }

    /// <summary>
    /// The licence of the first repository that declares one. A repackaging (a repository the Hub lists as quantized from another model) carries
    /// its repackager's tag, which often differs from the original's (or is missing), so the licence of the original model is preferred.
    /// A manually set <see cref="CatalogEntry.License"/> is never replaced: this is only called for entries without one.
    /// </summary>
    private async Task<string?> LicenseOfAsync(IReadOnlyList<HubRepository> repositories, CancellationToken cancellationToken)
    {
        foreach (var repository in repositories)
        {
            var current = repository;
            for (int depth = 0; depth < MaxBaseModelDepth && current.BaseModel is { Length: > 0 } baseModel; depth++)
            {
                try
                {
                    current = await GetRepositoryAsync(baseModel, DefaultRevision, cancellationToken).ConfigureAwait(false);
                }
                catch (HubException error)
                {
                    log($"{repository.Id}: base model {baseModel} not readable ({error.Message}); using the repackaging's own licence");
                    current = repository;
                    break;
                }
            }

            if ((current.License ?? repository.License) is { } license)
            {
                return license;
            }
        }

        return null;
    }

    private async Task<(T Set, HubRepository? Repository)> ResolveSetAsync<T>(string entryId, T set, ResolveMode mode, CancellationToken cancellationToken)
        where T : CatalogFileSet
    {
        bool resolved = set.Files.Count > 0 && set.Revision.Length == 40;
        if (resolved && mode == ResolveMode.Missing)
        {
            return (set, null);
        }

        if (!set.IsHubSource)
        {
            return (await ResolveGitHubSetAsync(entryId, set, mode, cancellationToken).ConfigureAwait(false), null);
        }

        string revision = mode == ResolveMode.Update || set.Revision.Length == 0 ? DefaultRevision : set.Revision;
        log($"{entryId}: {set.Repository} @ {revision}");
        HubRepository repository;
        IReadOnlyList<HubFile> files;
        string primary;
        try
        {
            repository = await GetRepositoryAsync(set.Repository, revision, cancellationToken).ConfigureAwait(false);
            (files, primary) = await ChooseFilesAsync(repository, set, cancellationToken).ConfigureAwait(false);
        }
        catch (HubException error)
        {
            _failures.Add($"{entryId}: {set.Repository}@{revision}: {error.Message}");
            return (set, null);
        }

        var pinned = set with
        {
            Repository = repository.Id,
            Revision = repository.Commit,
            Files = [.. files.Select(file => new CatalogFile { Path = file.Path, Size = file.Size, Sha256 = file.Sha256, GitOid = file.GitOid })],
            Primary = primary,
        };
        return (pinned, repository);
    }

    /// <summary>
    /// Pins a set from GitHub: the commit of its release tag (or of the default branch when it has none; <c>--update</c> resolves the tag
    /// again), the files its include patterns choose (repository files, or the release's assets) with sizes and sha256 checksums.
    /// Repository files are downloaded to hash them; a file whose path and git blob id did not change keeps its recorded sha256.
    /// </summary>
    private async Task<T> ResolveGitHubSetAsync<T>(string entryId, T set, ResolveMode mode, CancellationToken cancellationToken)
        where T : CatalogFileSet
    {
        string gitRef = mode == ResolveMode.Update || set.Revision.Length == 0 ? set.Release ?? GitHubDefaultRef : set.Revision;
        log($"{entryId}: GitHub {set.Repository} @ {gitRef}");
        try
        {
            if (github is null)
            {
                throw new HubException("no GitHub reader");
            }

            if (set.Include.Count == 0)
            {
                throw new HubException("a GitHub set names its files with include patterns");
            }

            string commit = await github.ResolveCommitAsync(set.Repository, gitRef, cancellationToken).ConfigureAwait(false);
            bool assets = set.Source == CatalogSources.GitHubRelease;
            var available = assets
                ? await github.ListReleaseAssetsAsync(set.Repository, set.Release ?? "", cancellationToken).ConfigureAwait(false)
                : await github.ListFilesAsync(set.Repository, commit, cancellationToken).ConfigureAwait(false);
            var chosen = Matching(available, set.Include, $"{set.Repository}@{commit[..12]}");
            var pinned = set with { Revision = commit, Files = [] };
            var files = new List<CatalogFile>();
            foreach (var file in chosen)
            {
                var known = set.Files.FirstOrDefault(old => old.Path == file.Path && old.GitOid is not null && old.GitOid == file.GitOid);
                string sha256 = file.Sha256 ?? known?.Sha256
                    ?? await github.Sha256Async(pinned.GetDirectFileUri(file.Path)!, cancellationToken).ConfigureAwait(false);
                files.Add(new CatalogFile { Path = file.Path, Size = file.Size, Sha256 = sha256, GitOid = file.GitOid });
            }

            return pinned with { Files = files, Primary = set.Primary.Length > 0 ? set.Primary : PrimaryOf(chosen) };
        }
        catch (HubException error)
        {
            _failures.Add($"{entryId}: GitHub {set.Repository}@{gitRef}: {error.Message}");
            return set;
        }
    }

    /// <summary>The files that match the include patterns, in pattern order; a pattern without a match is an error.</summary>
    private static List<HubFile> Matching(IReadOnlyList<HubFile> available, IReadOnlyList<string> include, string where)
    {
        var files = new List<HubFile>();
        foreach (var pattern in include)
        {
            var matches = available.Where(file => GlobPattern.IsMatch(pattern, file.Path)).ToList();
            if (matches.Count == 0)
            {
                throw new HubException($"{where}: no file matches '{pattern}'.");
            }

            files.AddRange(matches.Where(match => !files.Contains(match)));
        }

        return files;
    }

    private async Task<HubRepository> GetRepositoryAsync(string id, string revision, CancellationToken cancellationToken)
    {
        if (!_repositories.TryGetValue((id, revision), out var repository))
        {
            repository = await hub.GetRepositoryAsync(id, revision, cancellationToken).ConfigureAwait(false);
            _repositories[(id, revision)] = repository;
        }

        return repository;
    }

    private async Task<(IReadOnlyList<HubFile> Files, string Primary)> ChooseFilesAsync(HubRepository repository, CatalogFileSet set, CancellationToken cancellationToken)
    {
        if (set.Include.Count > 0)
        {
            var files = Matching(repository.Files, set.Include, $"{repository.Id}@{repository.Commit[..12]}");
            return (files, set.Primary.Length > 0 ? set.Primary : PrimaryOf(files));
        }

        var indexPath = ModelFileSelector.IndexPath(repository, set.Tag);
        var index = indexPath is null ? null : await hub.GetFileTextAsync(repository, indexPath, cancellationToken).ConfigureAwait(false);
        var selection = ModelFileSelector.Select(repository, set.Tag, index);
        var chosen = selection.Files.Where(file => file.Path != selection.ProjectorPath).ToList();
        return (chosen, set.Primary.Length > 0 ? set.Primary : selection.PrimaryPath);
    }

    /// <summary>A single file, else the first shard of a GGUF set, else the deepest directory all files share.</summary>
    private static string PrimaryOf(IReadOnlyList<HubFile> files)
    {
        if (files.Count == 1)
        {
            return files[0].Path;
        }

        var ggufs = files.Where(file => file.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).Select(file => file.Path).Order(StringComparer.Ordinal).ToList();
        if (ggufs.Count > 0 && ggufs.Count == files.Count)
        {
            return ggufs[0];
        }

        var directory = files[0].Directory;
        while (directory.Length > 0 && !files.All(file => file.Path.StartsWith(directory + "/", StringComparison.Ordinal)))
        {
            directory = directory.LastIndexOf('/') is var slash and >= 0 ? directory[..slash] : "";
        }

        return directory;
    }
}
