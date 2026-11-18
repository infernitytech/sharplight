using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Publishing;

namespace Sharplight.ModelFeed.Tests;

public sealed class ChannelTests
{
    private static readonly FeedPublishOptions Unstable = new() { Channel = FeedChannels.Unstable };

    [Fact]
    public async Task Channels_ShareStoredVersions_AndListIndependently()
    {
        using var feed = new TestFeed();
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.0.0"), await feed.StageAsync("beta.f16", "1.0.0")], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.1.0", seed: 2)], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        var promoted = await feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8@1.0.0", "beta.f16"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(["alpha.q8@1.0.0", "beta.f16@1.0.0"], promoted.Added);

        var unstable = await feed.Client(channel: FeedChannels.Unstable).GetSnapshotAsync(TestContext.Current.CancellationToken);
        var stableClient = feed.Client(channel: FeedChannels.Stable);
        var stable = await stableClient.GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Equal("1.1.0", unstable.Find("alpha.q8")!.LatestVersion);
        Assert.Equal("1.0.0", stable.Find("alpha.q8")!.LatestVersion);
        Assert.Equal(["1.0.0"], stable.Find("alpha.q8")!.Versions);
        Assert.Equal(FeedChannels.Stable, stable.Index.Channel);

        // One stored copy: the stable manifest and package file are the ones published to unstable.
        var manifest = await stableClient.GetManifestAsync(stable, "alpha.q8", "1.0.0", TestContext.Current.CancellationToken);
        Assert.True(manifest.Signed);
        Assert.Single(Directory.EnumerateFiles(feed.Folder, "alpha.q8.1.0.0.slm", SearchOption.AllDirectories));
        Assert.False(Directory.Exists(feed.PathOf("stable/packages/alpha.q8/1.0.0")));
        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(channel: FeedChannels.Stable), packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(channel: FeedChannels.Unstable), packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));

        // Promoting again changes nothing; a version must be published before it can be promoted.
        Assert.Equal(["alpha.q8@1.0.0"], (await feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8@1.0.0"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken)).Unchanged);
        await Assert.ThrowsAsync<KeyNotFoundException>(() => feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8@9.0.0"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task OneChannelsIndex_IsRejectedForAnother()
    {
        using var feed = new TestFeed();
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.0.0")], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken);

        // A validly signed unstable index served where the stable one belongs.
        File.Copy(feed.PathOf("unstable/index.json"), feed.PathOf("stable/index.json"), overwrite: true);
        File.Copy(feed.PathOf("unstable/index.sig"), feed.PathOf("stable/index.sig"), overwrite: true);
        var error = await Assert.ThrowsAsync<FeedVerificationException>(() => feed.Client(channel: FeedChannels.Stable).GetSnapshotAsync(TestContext.Current.CancellationToken));
        Assert.Contains("channel 'unstable'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RollbackState_IsKeptPerChannel()
    {
        using var feed = new TestFeed();
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.0.0")], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.1.0", seed: 2)], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8@1.0.0"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken);
        var oldStable = (await File.ReadAllBytesAsync(feed.PathOf("stable/index.json"), TestContext.Current.CancellationToken), await File.ReadAllBytesAsync(feed.PathOf("stable/index.sig"), TestContext.Current.CancellationToken));
        await feed.Publisher.ReindexAsync(feed.Folder, feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Stable }, TestContext.Current.CancellationToken);

        var state = new MemoryFeedStateStore();
        var unstable = await feed.Client(state, channel: FeedChannels.Unstable).GetSnapshotAsync(TestContext.Current.CancellationToken);
        var stable = await feed.Client(state, channel: FeedChannels.Stable).GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Equal(2, unstable.Index.Sequence);
        Assert.Equal(2, stable.Index.Sequence);
        Assert.Equal(2, (await state.GetAsync(feed.Client(channel: FeedChannels.Stable).StateKey, TestContext.Current.CancellationToken))?.Sequence);
        Assert.NotEqual(feed.Client(channel: FeedChannels.Stable).StateKey, feed.Client(channel: FeedChannels.Unstable).StateKey);

        // Rolling stable back is detected; unstable is unaffected.
        await File.WriteAllBytesAsync(feed.PathOf("stable/index.json"), oldStable.Item1, TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(feed.PathOf("stable/index.sig"), oldStable.Item2, TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<FeedVerificationException>(() => feed.Client(state, channel: FeedChannels.Stable).GetSnapshotAsync(TestContext.Current.CancellationToken));
        Assert.Equal(2, (await feed.Client(state, channel: FeedChannels.Unstable).GetSnapshotAsync(TestContext.Current.CancellationToken)).Index.Sequence);
    }

    [Fact]
    public async Task Promotion_RefusesYankedVersions_AndStateIsPerChannel()
    {
        using var feed = new TestFeed();
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.0.0"), await feed.StageAsync("alpha.q8", "1.1.0", seed: 2)], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8@1.0.0"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken);
        await feed.Publisher.SetStateAsync(feed.Folder, "alpha.q8", "1.1.0", state => state with { Yanked = new PackageYank { Reason = "bad" } }, feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<FeedFormatException>(() => feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8@1.1.0"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken));

        // Deprecating in stable leaves unstable's entry alone.
        await feed.Publisher.SetStateAsync(feed.Folder, "alpha.q8", "1.0.0", state => state with { Deprecated = new PackageDeprecation { Reason = "old" } }, feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Stable }, TestContext.Current.CancellationToken);
        var stableClient = feed.Client(channel: FeedChannels.Stable);
        var unstableClient = feed.Client(channel: FeedChannels.Unstable);
        var stableIndex = await stableClient.GetPackageIndexAsync(await stableClient.GetSnapshotAsync(TestContext.Current.CancellationToken), "alpha.q8", TestContext.Current.CancellationToken);
        var unstableIndex = await unstableClient.GetPackageIndexAsync(await unstableClient.GetSnapshotAsync(TestContext.Current.CancellationToken), "alpha.q8", TestContext.Current.CancellationToken);
        Assert.NotNull(stableIndex.Versions[0].State.Deprecated);
        Assert.Null(unstableIndex.Versions[0].State.Deprecated);
    }

    [Fact]
    public async Task SameVersionWithOtherContent_IsRejectedAcrossChannels()
    {
        using var feed = new TestFeed();
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.0.0")], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        var other = await feed.StageAsync("alpha.q8", "1.0.0", size: 20_000, seed: 7);
        await Assert.ThrowsAsync<FeedFormatException>(() => feed.Publisher.PublishAsync(feed.Folder, [other], feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Stable }, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Mirror_CopiesBothChannelsIntoOneFolder()
    {
        using var feed = new TestFeed();
        await feed.Publisher.PublishAsync(feed.Folder, [await feed.StageAsync("alpha.q8", "1.0.0")], feed.FeedKey, Unstable, TestContext.Current.CancellationToken);
        await feed.Publisher.PromoteAsync(feed.Folder, FeedChannels.Stable, ["alpha.q8"], FeedChannels.Unstable, feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken);
        var mirror = Path.Combine(feed.Root, "mirror");
        await FeedMirror.CopyAsync(feed.Client(channel: FeedChannels.Stable), mirror, cancellationToken: TestContext.Current.CancellationToken);
        await FeedMirror.CopyAsync(feed.Client(channel: FeedChannels.Unstable), mirror, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(transport: new FolderFeedTransport(mirror), channel: FeedChannels.Stable), packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(transport: new FolderFeedTransport(mirror), channel: FeedChannels.Unstable), packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));
    }
}
