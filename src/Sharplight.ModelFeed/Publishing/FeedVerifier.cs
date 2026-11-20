using Sharplight.ModelFeed.Client;

namespace Sharplight.ModelFeed.Publishing;

/// <summary>Checks a whole feed the way a host does: the snapshot, every package index and manifest and, optionally, every package file's hashes.</summary>
public static class FeedVerifier
{
    /// <summary>Verifies the feed behind <paramref name="client"/>.</summary>
    /// <param name="client">The feed client (its trust decides what is accepted).</param>
    /// <param name="packageFiles">Whether to read and hash every package file (all chunks and the whole file).</param>
    /// <param name="log">Receives progress lines, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The problems found; empty when the feed is valid.</returns>
    public static async Task<IReadOnlyList<string>> VerifyAsync(FeedClient client, bool packageFiles, Action<string>? log = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(client);
        var problems = new List<string>();
        FeedSnapshot snapshot;
        try
        {
            snapshot = await client.GetSnapshotAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception exception) when (exception is FeedVerificationException or FeedFormatException or IOException)
        {
            return [$"index: {exception.Message}"];
        }

        log?.Invoke($"Index sequence {snapshot.Index.Sequence}, {snapshot.Packages.Count} package(s), {snapshot.Models.Models.Count} model(s), expires {snapshot.Index.Expires:u}, {(snapshot.Verified ? "signed" : "unsigned")}");
        problems.AddRange(await VerifyModelsAsync(client, snapshot, cancellationToken).ConfigureAwait(false));
        foreach (var summary in snapshot.Packages)
        {
            try
            {
                var index = await client.GetPackageIndexAsync(snapshot, summary.Id, cancellationToken).ConfigureAwait(false);
                foreach (var entry in index.Versions)
                {
                    var manifest = await client.GetManifestAsync(snapshot, summary.Id, entry.Version, cancellationToken).ConfigureAwait(false);
                    if (packageFiles)
                    {
                        await VerifyFileAsync(client.Transport, manifest.Manifest, cancellationToken).ConfigureAwait(false);
                    }

                    log?.Invoke($"  {summary.Id} {entry.Version}: ok{(entry.State.Yanked is not null ? " (yanked)" : "")}");
                }
            }
            catch (Exception exception) when (exception is FeedVerificationException or FeedFormatException or IOException or KeyNotFoundException)
            {
                problems.Add($"{summary.Id}: {exception.Message}");
            }
        }

        return problems;
    }

    // The published model listing must be the one the pages give, and a model has at most one recommended variant.
    private static async Task<IReadOnlyList<string>> VerifyModelsAsync(FeedClient client, FeedSnapshot snapshot, CancellationToken cancellationToken)
    {
        var problems = new List<string>();
        try
        {
            var published = await client.GetModelListingAsync(snapshot, cancellationToken).ConfigureAwait(false);
            var expected = snapshot.Models;
            var publishedText = System.Text.Encoding.UTF8.GetString(FeedJson.Write(published, FeedJsonContext.Default.FeedModelListing));
            var expectedText = System.Text.Encoding.UTF8.GetString(FeedJson.Write(expected, FeedJsonContext.Default.FeedModelListing));
            if (publishedText != expectedText)
            {
                problems.Add("models: the model listing does not match the catalog pages.");
            }

            if (snapshot.Index.Catalog.ModelsDigest is not null && snapshot.Index.Catalog.Models != published.Models.Count)
            {
                problems.Add($"models: the index counts {snapshot.Index.Catalog.Models} model(s), the listing has {published.Models.Count}.");
            }
        }
        catch (Exception exception) when (exception is FeedVerificationException or FeedFormatException or IOException)
        {
            problems.Add($"models: {exception.Message}");
        }

        foreach (var model in snapshot.Models.Models.Where(model => model.Variants.Count(variant => variant.Recommended) > 1))
        {
            problems.Add($"{model.Id}: {model.Variants.Count(variant => variant.Recommended)} variants are marked recommended (at most one).");
        }

        return problems;
    }

    private static async Task VerifyFileAsync(IFeedTransport transport, PackageManifest manifest, CancellationToken cancellationToken)
    {
        var file = manifest.PrimaryFile ?? throw new FeedFormatException("The manifest lists no package file.");
        var path = FeedLayout.VersionFile(manifest.Id, manifest.Version, file.Name);
        await using var stream = await transport.OpenReadAsync(path, 0, cancellationToken).ConfigureAwait(false);
        var actual = await PackageFileHasher.DescribeAsync(stream, file.Name, file.ChunkSize ?? PackageFileHasher.DefaultChunkSize, cancellationToken).ConfigureAwait(false);
        var chunksMatch = file.Chunks.Count == 0 || actual.Chunks.SequenceEqual(file.Chunks, StringComparer.OrdinalIgnoreCase);
        if (actual.Size != file.Size || actual.Sha256 != file.Sha256.ToLowerInvariant() || !chunksMatch)
        {
            throw new FeedVerificationException($"{file.Name} does not match its manifest (size or hashes).");
        }
    }
}
