using Sharplight.ModelFeed.Client;

namespace Sharplight.ModelFeed.Tests;

public sealed class DownloadTests
{
    [Fact]
    public async Task Download_VerifiesAndStoresThePackage()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0", size: 50_000));
        var manifest = await ManifestAsync(feed);
        var destination = Path.Combine(feed.Root, "store", "alpha.slm");
        var transferred = await new PackageDownloader().DownloadAsync(manifest, [new FolderFeedTransport(feed.Folder)], destination, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(50_000, transferred);
        Assert.Equal(manifest.PrimaryFile!.Sha256, Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken))));

        // Already present: nothing is transferred.
        Assert.Equal(0, await new PackageDownloader().DownloadAsync(manifest, [new FolderFeedTransport(feed.Folder)], destination, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Download_ResumesAfterAnAbort_KeepingOnlyVerifiedChunks()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0", size: 50_000));
        var manifest = await ManifestAsync(feed);
        var destination = Path.Combine(feed.Root, "store", "alpha.slm");
        var failing = new FlakyTransport(new FolderFeedTransport(feed.Folder), failAfter: 20_000);
        await Assert.ThrowsAsync<IOException>(() => new PackageDownloader().DownloadAsync(manifest, [failing], destination, cancellationToken: TestContext.Current.CancellationToken));
        var partial = destination + PackageDownloader.PartialSuffix;
        Assert.True(File.Exists(partial));

        // Damage the tail of the partial file: only whole, verified chunks are kept on resume.
        await using (var stream = new FileStream(partial, FileMode.Append))
        {
            stream.Write(new byte[100]);
        }

        var transferred = await new PackageDownloader().DownloadAsync(manifest, [new FolderFeedTransport(feed.Folder)], destination, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(50_000 - 4 * TestFeed.ChunkSize, transferred);
        Assert.False(File.Exists(partial));
        Assert.Equal(50_000, new FileInfo(destination).Length);
    }

    [Fact]
    public async Task Download_SwitchesSourceWhenOneSendsWrongBytes()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0", size: 30_000));
        var manifest = await ManifestAsync(feed);
        var corrupt = Path.Combine(feed.Root, "corrupt");
        CopyFolder(feed.Folder, corrupt);
        var file = Path.Combine(corrupt, FeedLayout.VersionFile("alpha.q8", "1.0.0", manifest.PrimaryFile!.Name));
        File.SetAttributes(file, FileAttributes.Normal);
        var bytes = await File.ReadAllBytesAsync(file, TestContext.Current.CancellationToken);
        bytes[10_000] ^= 0xFF;
        await File.WriteAllBytesAsync(file, bytes, TestContext.Current.CancellationToken);

        var destination = Path.Combine(feed.Root, "store", "alpha.slm");
        await new PackageDownloader().DownloadAsync(manifest, [new FolderFeedTransport(corrupt), new FolderFeedTransport(feed.Folder)], destination, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(30_000, new FileInfo(destination).Length);

        File.Delete(destination);
        await Assert.ThrowsAsync<FeedVerificationException>(() => new PackageDownloader().DownloadAsync(manifest, [new FolderFeedTransport(corrupt)], destination, cancellationToken: TestContext.Current.CancellationToken));
        Assert.False(File.Exists(destination));
    }

    [Fact]
    public async Task Download_OverHttp_UsesRangeRequests()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0", size: 50_000));
        using var handler = new FolderHttpHandler(feed.Folder);
        using var http = new HttpClient(handler);
        var transport = new HttpFeedTransport(http, new Uri("http://feed.test/models/v1"));
        var client = feed.Client(transport: transport);
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        var manifest = await client.GetManifestAsync(snapshot, "alpha.q8", "1.0.0", TestContext.Current.CancellationToken);
        var destination = Path.Combine(feed.Root, "store", "alpha.slm");
        var partial = destination + PackageDownloader.PartialSuffix;
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        var source = await File.ReadAllBytesAsync(feed.PathOf(FeedLayout.VersionFile("alpha.q8", "1.0.0", manifest.Manifest.PrimaryFile!.Name)), TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(partial, source.AsSpan(0, 3 * (int)TestFeed.ChunkSize).ToArray(), TestContext.Current.CancellationToken);

        var transferred = await new PackageDownloader().DownloadAsync(manifest.Manifest, [transport], destination, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(50_000 - 3 * TestFeed.ChunkSize, transferred);
        Assert.Contains(handler.Ranges, range => range == 3 * TestFeed.ChunkSize);
    }

    private static async Task<PackageManifest> ManifestAsync(TestFeed feed)
    {
        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        return (await client.GetManifestAsync(snapshot, "alpha.q8", "1.0.0", TestContext.Current.CancellationToken)).Manifest;
    }

    private static void CopyFolder(string source, string target)
    {
        foreach (var file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
        {
            var destination = Path.Combine(target, Path.GetRelativePath(source, file));
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            File.Copy(file, destination);
        }
    }
}
