using Sharplight.ModelFeed.Client;

namespace Sharplight.ModelFeed.Publishing;

/// <summary>
/// Copies a verified feed into a folder byte for byte (an offline bundle for air-gapped sites): the copy keeps every signature and
/// digest, so a host trusts it exactly like the original. Package files are downloaded with resume and chunk verification.
/// </summary>
public static class FeedMirror
{
    /// <summary>Mirrors the feed behind <paramref name="client"/> into <paramref name="folder"/>.</summary>
    /// <param name="client">The source feed (verified with its trust).</param>
    /// <param name="folder">Target folder.</param>
    /// <param name="filter">Selects package ids to include package files for (all when <see langword="null"/>); listings stay complete so signatures stay valid.</param>
    /// <param name="log">Receives progress lines, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The number of package files copied.</returns>
    public static async Task<int> CopyAsync(FeedClient client, string folder, Func<string, bool>? filter = null, Action<string>? log = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(client);
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        var snapshot = await client.GetSnapshotAsync(cancellationToken).ConfigureAwait(false);
        var transport = client.Transport;
        var documents = new List<string> { FeedLayout.KeySet, FeedLayout.KeySetSignature, snapshot.Index.Resources.Catalog };
        for (var page = 1; page <= snapshot.Index.Catalog.Pages; page++)
        {
            documents.Add(FeedLayout.CatalogPage(page));
        }

        var copied = 0;
        var downloader = new PackageDownloader();
        foreach (var summary in snapshot.Packages)
        {
            documents.Add(FeedLayout.PackageIndex(summary.Id));
            var index = await client.GetPackageIndexAsync(snapshot, summary.Id, cancellationToken).ConfigureAwait(false);
            foreach (var entry in index.Versions)
            {
                var manifest = await client.GetManifestAsync(snapshot, summary.Id, entry.Version, cancellationToken).ConfigureAwait(false);
                documents.Add(FeedLayout.Manifest(summary.Id, entry.Version));
                documents.Add(FeedLayout.ManifestSignature(summary.Id, entry.Version));
                if (filter is null || filter(summary.Id))
                {
                    var file = manifest.Manifest.PrimaryFile!;
                    var target = Path.Combine(folder, FeedLayout.VersionFile(summary.Id, entry.Version, file.Name));
                    log?.Invoke($"{summary.Id} {entry.Version}: {file.Size} bytes");
                    await downloader.DownloadAsync(manifest.Manifest, [transport], target, null, cancellationToken).ConfigureAwait(false);
                    copied++;
                }
            }
        }

        documents.Add(FeedLayout.ServiceIndexSignature);
        documents.Add(FeedLayout.ServiceIndex);
        foreach (var path in documents)
        {
            var bytes = await transport.ReadAsync(path, long.MaxValue, cancellationToken).ConfigureAwait(false);
            if (bytes is null)
            {
                continue;
            }

            var target = Path.Combine(folder, path.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            await File.WriteAllBytesAsync(target, bytes, cancellationToken).ConfigureAwait(false);
        }

        return copied;
    }
}
