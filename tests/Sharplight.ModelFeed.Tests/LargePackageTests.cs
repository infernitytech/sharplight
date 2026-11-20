using Sharplight.ModelFeed.Publishing;

namespace Sharplight.ModelFeed.Tests;

/// <summary>The disk-saving publication of large packages (docs/model-feed.md, "Self-hosted feed runner").</summary>
public sealed class LargePackageTests
{
    private static readonly FeedPublishOptions MoveToUnstable = new() { Channel = FeedChannels.Unstable, MovePackageFiles = true };

    [Fact]
    public async Task Publish_MovesPackageFile_WhenAsked()
    {
        using var feed = new TestFeed();
        var folder = await feed.StageAsync("big.q4", "1.0.0", size: 20_000);
        var staged = Path.Combine(folder, FeedLayout.PackageFileName("big.q4", "1.0.0"));

        await feed.Publisher.PublishAsync(feed.Folder, [folder], feed.FeedKey, MoveToUnstable, TestContext.Current.CancellationToken);

        Assert.False(File.Exists(staged));
        Assert.True(File.Exists(Path.Combine(folder, FeedLayout.ManifestFileName)));
        Assert.Equal(20_000, new FileInfo(feed.PathOf(FeedLayout.VersionFile("big.q4", "1.0.0", "big.q4.1.0.0.slm"))).Length);
        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(channel: FeedChannels.Unstable), packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Publish_ListsStoredVersion_FromManifestOnlyFolder_AfterPackageFileWasUploadedAndRemoved()
    {
        using var feed = new TestFeed();
        var folder = await feed.StageAsync("big.q4", "1.0.0", size: 20_000);
        await feed.Publisher.PublishAsync(feed.Folder, [folder], feed.FeedKey, MoveToUnstable, TestContext.Current.CancellationToken);

        // The runner uploads the package objects and deletes the local file; listings and verification need only the manifests.
        File.Delete(feed.PathOf(FeedLayout.VersionFile("big.q4", "1.0.0", "big.q4.1.0.0.slm")));
        await feed.Publisher.ReindexAsync(feed.Folder, feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Unstable }, TestContext.Current.CancellationToken);
        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(channel: FeedChannels.Unstable), packageFiles: false, cancellationToken: TestContext.Current.CancellationToken));

        // A run interrupted after the package objects were stored but before the listings: the next run stages only the stored manifest,
        // which lists the version without its file.
        var relist = Path.Combine(feed.Root, "relist", "big.q4-1.0.0");
        Directory.CreateDirectory(relist);
        File.Copy(feed.PathOf(FeedLayout.Manifest("big.q4", "1.0.0")), Path.Combine(relist, FeedLayout.ManifestFileName));
        var result = await feed.Publisher.PublishAsync(feed.Folder, [relist], feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Stable, MovePackageFiles = true }, TestContext.Current.CancellationToken);

        Assert.Equal(["big.q4@1.0.0"], result.Added);
        var stable = await feed.Client(channel: FeedChannels.Stable).GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Equal("1.0.0", stable.Find("big.q4")!.LatestVersion);
        var again = await feed.Publisher.PublishAsync(feed.Folder, [relist], feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Stable }, TestContext.Current.CancellationToken);
        Assert.Equal(["big.q4@1.0.0"], again.Unchanged);
    }
}
