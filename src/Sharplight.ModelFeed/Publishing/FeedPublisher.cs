using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Signing;
using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Publishing;

/// <summary>
/// Writes a feed into a folder (upload it to object storage as is, package objects first and <c>index.json</c> last): adds built
/// package versions (immutable, stored once in <c>packages/&lt;id&gt;/&lt;version&gt;/</c> and shared by all channels), signs manifests,
/// rewrites the channel's package indexes, catalog pages and service index with the channel's next sequence, and signs the index.
/// <see cref="PromoteAsync"/> lists already published versions in another channel without copying them. Not thread safe: one
/// publisher per feed folder at a time.
/// </summary>
public sealed class FeedPublisher
{
    private readonly TimeProvider _time;

    /// <summary>Creates a publisher.</summary>
    /// <param name="time">Clock; <see langword="null"/> for the system clock.</param>
    public FeedPublisher(TimeProvider? time = null)
    {
        _time = time ?? TimeProvider.System;
    }

    /// <summary>Writes the key set signed by a root key (<c>keys/keyset.json</c> and <c>keys/keyset.sig</c>).</summary>
    /// <param name="feedFolder">The feed folder.</param>
    /// <param name="keySet">The key set.</param>
    /// <param name="root">The root key (offline; only needed when the key set changes).</param>
    public static void WriteKeySet(string feedFolder, FeedKeySet keySet, FeedSigningKey root)
    {
        ArgumentNullException.ThrowIfNull(keySet);
        ArgumentNullException.ThrowIfNull(root);
        var bytes = FeedJson.Write(keySet, FeedJsonContext.Default.FeedKeySet);
        WriteAtomic(feedFolder, FeedLayout.KeySet, bytes);
        WriteAtomic(feedFolder, FeedLayout.KeySetSignature, root.SignDocument(bytes, "keyset"));
    }

    /// <summary>Adds built package versions and republishes the index.</summary>
    /// <param name="feedFolder">The feed folder (created when missing; must hold a key set listing <paramref name="key"/>).</param>
    /// <param name="packageFolders">Folders with a <c>manifest.json</c> and the package file it names.</param>
    /// <param name="key">The feed signing key.</param>
    /// <param name="options">Feed name, validity, page size.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>What was written.</returns>
    /// <exception cref="FeedFormatException">A package folder is incomplete or does not match its manifest, or a version exists with other content.</exception>
    public async Task<FeedPublishResult> PublishAsync(string feedFolder, IReadOnlyList<string> packageFolders, FeedSigningKey key, FeedPublishOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(packageFolders);
        ArgumentNullException.ThrowIfNull(key);
        options ??= new FeedPublishOptions();
        RequireKeyInKeySet(feedFolder, key);
        var now = _time.GetUtcNow();
        var added = new List<string>();
        var unchanged = new List<string>();
        foreach (var folder in packageFolders)
        {
            var manifestBytes = await File.ReadAllBytesAsync(Path.Combine(folder, FeedLayout.ManifestFileName), cancellationToken).ConfigureAwait(false);
            var manifest = FeedJson.ReadManifest(manifestBytes);
            var label = $"{manifest.Id}@{manifest.Version}";
            var packageIndex = ReadPackageIndex(feedFolder, options.Channel, manifest.Id) ?? new FeedPackageIndex { Id = manifest.Id };
            var version = SemanticVersion.Parse(manifest.Version);
            var digest = FeedDigest.Of(manifestBytes);
            var storedPath = Path.Combine(feedFolder, FeedLayout.Manifest(manifest.Id, manifest.Version));
            if (File.Exists(storedPath))
            {
                // Versions are stored once for all channels and never change.
                if (FeedDigest.Of(await File.ReadAllBytesAsync(storedPath, cancellationToken).ConfigureAwait(false)) != digest)
                {
                    throw new FeedFormatException($"{label} is published already with another manifest; versions are immutable, build a new version.");
                }

                if (packageIndex.Versions.Any(entry => SemanticVersion.Parse(entry.Version) == version))
                {
                    unchanged.Add(label);
                    continue;
                }

                packageIndex = AddVersion(packageIndex, Entry(manifest, digest, now));
                WriteAtomic(feedFolder, FeedLayout.InChannel(options.Channel, FeedLayout.PackageIndex(manifest.Id)), FeedJson.Write(packageIndex, FeedJsonContext.Default.FeedPackageIndex));
                added.Add(label);
                continue;
            }

            var file = manifest.PrimaryFile ?? throw new FeedFormatException($"{label}: the manifest lists no package file.");
            var reusePath = Path.Combine(folder, FeedLayout.PayloadReuseFileName);
            if (File.Exists(reusePath))
            {
                await PublishReusingPayloadAsync(feedFolder, manifest, manifestBytes, file, FeedJson.Read(await File.ReadAllBytesAsync(reusePath, cancellationToken).ConfigureAwait(false), FeedJsonContext.Default.PayloadReuse, $"{label}: {FeedLayout.PayloadReuseFileName}"), key, cancellationToken).ConfigureAwait(false);
                packageIndex = AddVersion(packageIndex, Entry(manifest, digest, now));
                WriteAtomic(feedFolder, FeedLayout.InChannel(options.Channel, FeedLayout.PackageIndex(manifest.Id)), FeedJson.Write(packageIndex, FeedJsonContext.Default.FeedPackageIndex));
                added.Add(label);
                continue;
            }

            var source = Path.Combine(folder, file.Name);
            var actual = await PackageFileHasher.DescribeAsync(source, file.Name, file.ChunkSize ?? PackageFileHasher.DefaultChunkSize, cancellationToken).ConfigureAwait(false);
            if (actual.Size != file.Size || actual.Sha256 != file.Sha256.ToLowerInvariant() || (file.Chunks.Count > 0 && !actual.Chunks.SequenceEqual(file.Chunks, StringComparer.OrdinalIgnoreCase)))
            {
                throw new FeedFormatException($"{label}: {file.Name} does not match the manifest (size or hashes).");
            }

            var target = Path.Combine(feedFolder, FeedLayout.VersionFile(manifest.Id, manifest.Version, file.Name));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            if (options.MovePackageFiles)
            {
                File.Move(source, target, overwrite: true);
            }
            else
            {
                File.Copy(source, target + ".tmp", overwrite: true);
                File.Move(target + ".tmp", target, overwrite: true);
            }

            WriteAtomic(feedFolder, FeedLayout.Manifest(manifest.Id, manifest.Version), manifestBytes);
            WriteAtomic(feedFolder, FeedLayout.ManifestSignature(manifest.Id, manifest.Version), key.SignDocument(manifestBytes, "package-manifest"));
            packageIndex = AddVersion(packageIndex, Entry(manifest, digest, now));
            WriteAtomic(feedFolder, FeedLayout.InChannel(options.Channel, FeedLayout.PackageIndex(manifest.Id)), FeedJson.Write(packageIndex, FeedJsonContext.Default.FeedPackageIndex));
            added.Add(label);
        }

        var (sequence, count) = await ReindexAsync(feedFolder, key, options, cancellationToken).ConfigureAwait(false);
        return new FeedPublishResult(sequence, added, unchanged, count);
    }

    /// <summary>
    /// Lists already published versions in <paramref name="channel"/> (typically <see cref="FeedChannels.Stable"/>) without copying
    /// package files, and republishes that channel's index. The state is taken from <paramref name="fromChannel"/> when the version is
    /// listed there (a yanked version cannot be promoted), otherwise from the manifest.
    /// </summary>
    /// <param name="feedFolder">The feed folder.</param>
    /// <param name="channel">The target channel.</param>
    /// <param name="packages">Versions as <c>id@version</c>, or <c>id</c> for the latest version listed in <paramref name="fromChannel"/>.</param>
    /// <param name="fromChannel">The channel the versions come from (for <c>id</c> without version and for the state), or <see langword="null"/>.</param>
    /// <param name="key">The feed signing key.</param>
    /// <param name="options">Feed name, validity, page size (its channel is ignored).</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>What was written.</returns>
    /// <exception cref="KeyNotFoundException">A version is not published.</exception>
    /// <exception cref="FeedFormatException">A version is yanked in <paramref name="fromChannel"/>.</exception>
    public async Task<FeedPublishResult> PromoteAsync(string feedFolder, string channel, IReadOnlyList<string> packages, string? fromChannel, FeedSigningKey key, FeedPublishOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(packages);
        ArgumentNullException.ThrowIfNull(key);
        options = (options ?? new FeedPublishOptions()) with { Channel = channel };
        RequireKeyInKeySet(feedFolder, key);
        var now = _time.GetUtcNow();
        var added = new List<string>();
        var unchanged = new List<string>();
        foreach (var request in packages)
        {
            var at = request.IndexOf('@', StringComparison.Ordinal);
            var id = at < 0 ? request : request[..at];
            var source = fromChannel is null ? null : ReadPackageIndex(feedFolder, fromChannel, id);
            string versionText;
            if (at >= 0)
            {
                versionText = request[(at + 1)..];
            }
            else
            {
                versionText = source is null || source.Versions.Count == 0
                    ? throw new KeyNotFoundException($"'{id}' is not listed in channel '{fromChannel}'; give the version as {id}@<version>.")
                    : Latest(source.Versions).Version;
            }

            var version = SemanticVersion.Parse(versionText);
            var manifestPath = Path.Combine(feedFolder, FeedLayout.Manifest(id, versionText));
            if (!File.Exists(manifestPath))
            {
                throw new KeyNotFoundException($"{id}@{versionText} is not published in this feed.");
            }

            var manifestBytes = await File.ReadAllBytesAsync(manifestPath, cancellationToken).ConfigureAwait(false);
            var manifest = FeedJson.ReadManifest(manifestBytes);
            var label = $"{id}@{manifest.Version}";
            var fromEntry = source?.Versions.FirstOrDefault(entry => SemanticVersion.Parse(entry.Version) == version);
            if (fromEntry?.State.Yanked is not null)
            {
                throw new FeedFormatException($"{label} is yanked in '{fromChannel}' and cannot be promoted.");
            }

            var target = ReadPackageIndex(feedFolder, channel, id) ?? new FeedPackageIndex { Id = id };
            if (target.Versions.Any(entry => SemanticVersion.Parse(entry.Version) == version))
            {
                unchanged.Add(label);
                continue;
            }

            var entry = fromEntry ?? Entry(manifest, FeedDigest.Of(manifestBytes), now);
            target = AddVersion(target, entry with { Published = entry.Published == default ? now : entry.Published });
            WriteAtomic(feedFolder, FeedLayout.InChannel(channel, FeedLayout.PackageIndex(id)), FeedJson.Write(target, FeedJsonContext.Default.FeedPackageIndex));
            added.Add(label);
        }

        var (sequence, count) = await ReindexAsync(feedFolder, key, options, cancellationToken).ConfigureAwait(false);
        return new FeedPublishResult(sequence, added, unchanged, count);
    }

    /// <summary>Changes the state of a published version (yank, deprecate, unlist, relist) and republishes the index.</summary>
    /// <param name="feedFolder">The feed folder.</param>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <param name="change">Computes the new state from the current one (in the channel of <paramref name="options"/>).</param>
    /// <param name="key">The feed signing key.</param>
    /// <param name="options">Feed options.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The new index sequence.</returns>
    /// <exception cref="KeyNotFoundException">The version is not in the feed.</exception>
    public async Task<long> SetStateAsync(string feedFolder, string id, string version, Func<PackageState, PackageState> change, FeedSigningKey key, FeedPublishOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(change);
        ArgumentNullException.ThrowIfNull(key);
        RequireKeyInKeySet(feedFolder, key);
        options ??= new FeedPublishOptions();
        var packageIndex = ReadPackageIndex(feedFolder, options.Channel, id) ?? throw new KeyNotFoundException($"The feed has no package '{id}'.");
        var wanted = SemanticVersion.Parse(version);
        var entry = packageIndex.Versions.FirstOrDefault(item => SemanticVersion.Parse(item.Version) == wanted)
            ?? throw new KeyNotFoundException($"The feed has no version {version} of '{id}'.");
        var updated = entry with { State = change(entry.State) };
        packageIndex = packageIndex with { Versions = [.. packageIndex.Versions.Select(item => ReferenceEquals(item, entry) ? updated : item)] };
        WriteAtomic(feedFolder, FeedLayout.InChannel(options.Channel, FeedLayout.PackageIndex(id)), FeedJson.Write(packageIndex, FeedJsonContext.Default.FeedPackageIndex));
        var (sequence, _) = await ReindexAsync(feedFolder, key, options, cancellationToken).ConfigureAwait(false);
        return sequence;
    }

    /// <summary>Rewrites catalog pages, catalog index and the signed service index (next sequence) from the package indexes.</summary>
    /// <param name="feedFolder">The feed folder.</param>
    /// <param name="key">The feed signing key.</param>
    /// <param name="options">Feed options.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The new sequence and the number of packages.</returns>
    public async Task<(long Sequence, int Packages)> ReindexAsync(string feedFolder, FeedSigningKey key, FeedPublishOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(key);
        options ??= new FeedPublishOptions();
        RequireKeyInKeySet(feedFolder, key);
        var channel = options.Channel;
        var summaries = new List<FeedPackageSummary>();
        var packagesFolder = Path.Combine(feedFolder, FeedLayout.InChannel(channel, "packages"));
        var ids = Directory.Exists(packagesFolder)
            ? Directory.EnumerateDirectories(packagesFolder).Select(Path.GetFileName).Where(FeedLayout.IsValidId).Order(StringComparer.Ordinal).ToList()
            : [];
        foreach (var id in ids)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var indexPath = Path.Combine(feedFolder, FeedLayout.InChannel(channel, FeedLayout.PackageIndex(id!)));
            if (!File.Exists(indexPath))
            {
                continue;
            }

            var indexBytes = await File.ReadAllBytesAsync(indexPath, cancellationToken).ConfigureAwait(false);
            var packageIndex = FeedJson.Read(indexBytes, FeedJsonContext.Default.FeedPackageIndex, $"The package index of {id}");
            if (packageIndex.Versions.Count == 0)
            {
                continue;
            }

            var latest = Latest(packageIndex.Versions);
            var manifestBytes = await File.ReadAllBytesAsync(Path.Combine(feedFolder, FeedLayout.Manifest(id!, latest.Version)), cancellationToken).ConfigureAwait(false);
            var manifest = FeedJson.ReadManifest(manifestBytes);
            summaries.Add(new FeedPackageSummary
            {
                Id = id!,
                LatestVersion = latest.Version,
                Versions = [.. packageIndex.Versions.Select(entry => entry.Version)],
                Display = manifest.Display,
                Publisher = manifest.Publisher,
                License = manifest.License,
                Model = manifest.Model,
                Requirements = manifest.Requirements,
                Dependencies = manifest.Dependencies,
                Size = latest.Size,
                State = latest.State,
                ManifestDigest = latest.ManifestDigest,
                IndexDigest = FeedDigest.Of(indexBytes),
            });
        }

        var pages = new List<FeedPageReference>();
        foreach (var (chunk, number) in summaries.Chunk(Math.Max(1, options.PageSize)).Select((chunk, index) => (chunk, index + 1)))
        {
            var path = FeedLayout.InChannel(channel, FeedLayout.CatalogPage(number));
            var bytes = FeedJson.Write(new FeedCatalogPage { Packages = chunk }, FeedJsonContext.Default.FeedCatalogPage);
            WriteAtomic(feedFolder, path, bytes);
            pages.Add(new FeedPageReference { Path = path, Count = chunk.Length, FirstId = chunk[0].Id, LastId = chunk[^1].Id, Digest = FeedDigest.Of(bytes) });
        }

        RemoveStalePages(feedFolder, channel, pages.Count);
        var catalogBytes = FeedJson.Write(new FeedCatalogIndex { Pages = pages }, FeedJsonContext.Default.FeedCatalogIndex);
        WriteAtomic(feedFolder, FeedLayout.InChannel(channel, FeedLayout.CatalogIndex), catalogBytes);

        var previousPath = Path.Combine(feedFolder, FeedLayout.InChannel(channel, FeedLayout.ServiceIndex));
        var previous = File.Exists(previousPath)
            ? FeedJson.Read(await File.ReadAllBytesAsync(previousPath, cancellationToken).ConfigureAwait(false), FeedJsonContext.Default.FeedServiceIndex, "The previous feed index")
            : null;
        var now = _time.GetUtcNow();
        var index = new FeedServiceIndex
        {
            Channel = channel,
            Name = options.Name,
            Updated = now,
            Expires = now + options.Validity,
            Sequence = (previous?.Sequence ?? 0) + 1,
            Resources = new FeedResources
            {
                Catalog = FeedLayout.InChannel(channel, FeedLayout.CatalogIndex),
                Packages = FeedLayout.InChannel(channel, FeedLayout.PackageIndexTemplate),
            },
            Catalog = new FeedCatalogSummary { Pages = pages.Count, Packages = summaries.Count, Digest = FeedDigest.Of(catalogBytes) },
        };
        var indexBytesOut = FeedJson.Write(index, FeedJsonContext.Default.FeedServiceIndex);
        WriteAtomic(feedFolder, FeedLayout.InChannel(channel, FeedLayout.ServiceIndex), indexBytesOut);
        WriteAtomic(feedFolder, FeedLayout.InChannel(channel, FeedLayout.ServiceIndexSignature), key.SignDocument(indexBytesOut, "feed-index"));
        return (index.Sequence, summaries.Count);
    }

    /// <summary>
    /// Publishes a manifest-only version whose package file is the payload of an earlier version (same SHA-256, size and chunks): the
    /// manifest and its signature are stored, the file is copied only when the earlier version's file is present in this folder; on
    /// object storage the uploader copies it server-side (the staging folder's <c>reuse.json</c> names the source).
    /// </summary>
    private static async Task PublishReusingPayloadAsync(string feedFolder, PackageManifest manifest, byte[] manifestBytes, PackageFile file, PayloadReuse reuse, FeedSigningKey key, CancellationToken cancellationToken)
    {
        var label = $"{manifest.Id}@{manifest.Version}";
        var earlierPath = Path.Combine(feedFolder, FeedLayout.Manifest(manifest.Id, reuse.Version));
        if (!File.Exists(earlierPath))
        {
            throw new FeedFormatException($"{label} reuses the package file of {reuse.Version}, which the feed does not have.");
        }

        var earlier = FeedJson.ReadManifest(await File.ReadAllBytesAsync(earlierPath, cancellationToken).ConfigureAwait(false)).PrimaryFile;
        if (earlier is null || earlier.Name != reuse.File || earlier.Size != file.Size || !string.Equals(earlier.Sha256, file.Sha256, StringComparison.OrdinalIgnoreCase)
            || !earlier.Chunks.SequenceEqual(file.Chunks, StringComparer.OrdinalIgnoreCase))
        {
            throw new FeedFormatException($"{label}: the package file does not match {reuse.File} of {reuse.Version} (size or hashes).");
        }

        var earlierFile = Path.Combine(feedFolder, FeedLayout.VersionFile(manifest.Id, reuse.Version, reuse.File));
        if (File.Exists(earlierFile))
        {
            var target = Path.Combine(feedFolder, FeedLayout.VersionFile(manifest.Id, manifest.Version, file.Name));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(earlierFile, target + ".tmp", overwrite: true);
            File.Move(target + ".tmp", target, overwrite: true);
        }

        WriteAtomic(feedFolder, FeedLayout.Manifest(manifest.Id, manifest.Version), manifestBytes);
        WriteAtomic(feedFolder, FeedLayout.ManifestSignature(manifest.Id, manifest.Version), key.SignDocument(manifestBytes, "package-manifest"));
    }

    private static FeedPackageVersion Latest(IReadOnlyList<FeedPackageVersion> versions)
    {
        var ordered = versions.OrderByDescending(entry => SemanticVersion.Parse(entry.Version)).ToList();
        bool Offered(FeedPackageVersion entry) => entry.State.Listed && entry.State.Yanked is null;
        return ordered.FirstOrDefault(entry => Offered(entry) && !SemanticVersion.Parse(entry.Version).IsPrerelease)
            ?? ordered.FirstOrDefault(Offered)
            ?? ordered[0];
    }

    private static FeedPackageVersion Entry(PackageManifest manifest, string digest, DateTimeOffset published) => new()
    {
        Version = manifest.Version,
        ManifestDigest = digest,
        Size = manifest.PrimaryFile?.Size ?? 0,
        Published = published,
        State = manifest.State ?? new PackageState(),
        Requirements = manifest.Requirements,
        Dependencies = manifest.Dependencies,
    };

    private static FeedPackageIndex AddVersion(FeedPackageIndex index, FeedPackageVersion entry) =>
        index with { Versions = [.. index.Versions.Append(entry).OrderBy(item => SemanticVersion.Parse(item.Version))] };

    private static FeedPackageIndex? ReadPackageIndex(string feedFolder, string? channel, string id)
    {
        var path = Path.Combine(feedFolder, FeedLayout.InChannel(channel, FeedLayout.PackageIndex(id)));
        return File.Exists(path) ? FeedJson.Read(File.ReadAllBytes(path), FeedJsonContext.Default.FeedPackageIndex, $"The package index of {id}") : null;
    }

    private static void RequireKeyInKeySet(string feedFolder, FeedSigningKey key)
    {
        var path = Path.Combine(feedFolder, FeedLayout.KeySet);
        if (!File.Exists(path))
        {
            throw new FeedFormatException($"{feedFolder} has no key set ({FeedLayout.KeySet}); write one with the root key first.");
        }

        var keySet = FeedJson.Read(File.ReadAllBytes(path), FeedJsonContext.Default.FeedKeySet, "The feed key set");
        if (FeedTrust.ResolveFeedKey(keySet, key.KeyId, DateTimeOffset.UtcNow) is null)
        {
            throw new FeedFormatException($"The key set does not list key {key.KeyId} as valid; add it with the root key first.");
        }
    }

    private static void RemoveStalePages(string feedFolder, string? channel, int pages)
    {
        for (var number = pages + 1; File.Exists(Path.Combine(feedFolder, FeedLayout.InChannel(channel, FeedLayout.CatalogPage(number)))); number++)
        {
            File.Delete(Path.Combine(feedFolder, FeedLayout.InChannel(channel, FeedLayout.CatalogPage(number))));
        }
    }

    private static void WriteAtomic(string feedFolder, string relativePath, byte[] bytes)
    {
        var path = Path.Combine(feedFolder, relativePath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + ".tmp";
        File.WriteAllBytes(temporary, bytes);
        File.Move(temporary, path, overwrite: true);
    }
}
