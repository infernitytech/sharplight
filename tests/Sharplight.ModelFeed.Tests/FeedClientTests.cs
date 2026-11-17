using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.ModelFeed.Tests;

public sealed class FeedClientTests
{
    [Fact]
    public async Task Snapshot_ListsPublishedPackages_AndManifestsVerify()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"), await feed.StageAsync("beta.f16", "1.0.0"));
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.1.0", seed: 2));

        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.True(snapshot.Verified);
        Assert.Equal(2, snapshot.Index.Sequence);
        Assert.Equal(["alpha.q8", "beta.f16"], snapshot.Packages.Select(package => package.Id));
        var alpha = snapshot.Find("alpha.q8")!;
        Assert.Equal("1.1.0", alpha.LatestVersion);
        Assert.Equal(["1.0.0", "1.1.0"], alpha.Versions);
        Assert.Equal("CH", alpha.Publisher?.CountryOfOrigin);

        var manifest = await client.GetManifestAsync(snapshot, "alpha.q8", "1.0.0", TestContext.Current.CancellationToken);
        Assert.True(manifest.Signed);
        Assert.Equal(feed.FeedKey.KeyId, manifest.KeyId);
        Assert.Equal("1.0.0", manifest.Manifest.Version);

        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(), packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Publish_RejectsChangedContentForAnExistingVersion()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        var changed = await feed.StageAsync("alpha.q8", "1.0.0", size: 20_000, seed: 9);
        await Assert.ThrowsAsync<FeedFormatException>(() => feed.PublishAsync(changed));
    }

    [Theory]
    [InlineData("index.json")]
    [InlineData("index.sig")]
    [InlineData("keys/keyset.json")]
    [InlineData("keys/keyset.sig")]
    [InlineData("catalog/index.json")]
    [InlineData("catalog/page/1.json")]
    public async Task Snapshot_RejectsTamperedFeedDocuments(string path)
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        Tamper(feed.PathOf(path));
        await Assert.ThrowsAnyAsync<Exception>(() => feed.Client().GetSnapshotAsync(TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData("packages/alpha.q8/index.json")]
    [InlineData("packages/alpha.q8/1.0.0/manifest.json")]
    [InlineData("packages/alpha.q8/1.0.0/manifest.sig")]
    public async Task Manifest_RejectsTamperedPackageDocuments(string path)
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        Tamper(feed.PathOf(path));
        await Assert.ThrowsAnyAsync<Exception>(() => client.GetManifestAsync(snapshot, "alpha.q8", "1.0.0", TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Manifest_SignedByAnotherKey_IsRejected()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        using var rogue = FeedSigningKey.Generate();
        var manifest = await File.ReadAllBytesAsync(feed.PathOf("packages/alpha.q8/1.0.0/manifest.json"), TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(feed.PathOf("packages/alpha.q8/1.0.0/manifest.sig"), rogue.SignDocument(manifest, "package-manifest"), TestContext.Current.CancellationToken);
        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<FeedVerificationException>(() => client.GetManifestAsync(snapshot, "alpha.q8", "1.0.0", TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Snapshot_RejectsUntrustedRoot_AndRevokedFeedKey()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        using var otherRoot = FeedSigningKey.Generate();
        var untrusted = new FeedClient(new FolderFeedTransport(feed.Folder), new FeedClientOptions { TrustedRoots = [otherRoot.PublicKey] }, time: feed.Time);
        await Assert.ThrowsAsync<FeedVerificationException>(() => untrusted.GetSnapshotAsync(TestContext.Current.CancellationToken));

        var revoked = TestFeed.KeySet(feed.FeedKey) with { Version = 2 };
        revoked = revoked with { Keys = [revoked.Keys[0] with { Revoked = feed.Time.Now.AddDays(-1) }] };
        FeedPublisher.WriteKeySet(feed.Folder, revoked, feed.RootKey);
        await Assert.ThrowsAsync<FeedVerificationException>(() => feed.Client().GetSnapshotAsync(TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Snapshot_RejectsRollbackToAnOlderIndex()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        var oldIndex = await File.ReadAllBytesAsync(feed.PathOf("index.json"), TestContext.Current.CancellationToken);
        var oldSignature = await File.ReadAllBytesAsync(feed.PathOf("index.sig"), TestContext.Current.CancellationToken);
        var oldCatalog = await File.ReadAllBytesAsync(feed.PathOf("catalog/index.json"), TestContext.Current.CancellationToken);
        var oldPage = await File.ReadAllBytesAsync(feed.PathOf("catalog/page/1.json"), TestContext.Current.CancellationToken);
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.1", seed: 3));

        var state = new MemoryFeedStateStore();
        Assert.Equal(2, (await feed.Client(state).GetSnapshotAsync(TestContext.Current.CancellationToken)).Index.Sequence);

        await File.WriteAllBytesAsync(feed.PathOf("index.json"), oldIndex, TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(feed.PathOf("index.sig"), oldSignature, TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(feed.PathOf("catalog/index.json"), oldCatalog, TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(feed.PathOf("catalog/page/1.json"), oldPage, TestContext.Current.CancellationToken);
        var error = await Assert.ThrowsAsync<FeedVerificationException>(() => feed.Client(state).GetSnapshotAsync(TestContext.Current.CancellationToken));
        Assert.Contains("rollback", error.Message, StringComparison.Ordinal);

        // A fresh client without state accepts the old (still valid) index: the state store is what protects.
        Assert.Equal(1, (await feed.Client().GetSnapshotAsync(TestContext.Current.CancellationToken)).Index.Sequence);
    }

    [Fact]
    public async Task Snapshot_RejectsExpiredIndex_UnlessOfflineAllowsIt()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        feed.Time.Now += TimeSpan.FromDays(31);
        var error = await Assert.ThrowsAsync<FeedVerificationException>(() => feed.Client().GetSnapshotAsync(TestContext.Current.CancellationToken));
        Assert.Contains("expired", error.Message, StringComparison.Ordinal);

        var offline = await feed.Client(allowExpired: true).GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Single(offline.Warnings);
    }

    [Fact]
    public async Task FileStateStore_PersistsSequence()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        var path = Path.Combine(feed.Root, "state.json");
        await feed.Client(new FileFeedStateStore(path)).GetSnapshotAsync(TestContext.Current.CancellationToken);
        var state = await new FileFeedStateStore(path).GetAsync(new FolderFeedTransport(feed.Folder).Location, TestContext.Current.CancellationToken);
        Assert.Equal(1, state?.Sequence);
    }

    [Fact]
    public async Task YankedVersion_IsFlaggedInIndex_AndLatestMovesBack()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"), await feed.StageAsync("alpha.q8", "1.1.0", seed: 4));
        await feed.Publisher.SetStateAsync(feed.Folder, "alpha.q8", "1.1.0",
            state => state with { Yanked = new PackageYank { Date = feed.Time.Now, Reason = "bad output", Severity = YankSeverities.Quality } }, feed.FeedKey);
        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Equal("1.0.0", snapshot.Find("alpha.q8")!.LatestVersion);
        var index = await client.GetPackageIndexAsync(snapshot, "alpha.q8", TestContext.Current.CancellationToken);
        Assert.NotNull(index.Versions.Single(version => version.Version == "1.1.0").State.Yanked);
    }

    [Fact]
    public async Task Paging_SplitsCatalog()
    {
        using var feed = new TestFeed();
        var folders = new List<string>();
        for (var index = 0; index < 5; index++)
        {
            folders.Add(await feed.StageAsync($"model-{index}.f16", "1.0.0", size: 100));
        }

        await feed.Publisher.PublishAsync(feed.Folder, folders, feed.FeedKey, new FeedPublishOptions { PageSize = 2 }, TestContext.Current.CancellationToken);
        var snapshot = await feed.Client().GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Equal(3, snapshot.Index.Catalog.Pages);
        Assert.Equal(5, snapshot.Packages.Count);
    }

    [Fact]
    public async Task Mirror_IsVerifiableLikeTheOriginal()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("alpha.q8", "1.0.0"));
        var mirror = Path.Combine(feed.Root, "mirror");
        Assert.Equal(1, await FeedMirror.CopyAsync(feed.Client(), mirror, cancellationToken: TestContext.Current.CancellationToken));
        var client = feed.Client(transport: new FolderFeedTransport(mirror));
        Assert.Empty(await FeedVerifier.VerifyAsync(client, packageFiles: true, cancellationToken: TestContext.Current.CancellationToken));
    }

    internal static void Tamper(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var position = Array.FindIndex(bytes, value => value is (byte)'1' or (byte)'a' or (byte)'e');
        bytes[position] = bytes[position] == (byte)'1' ? (byte)'2' : (byte)'b';
        File.WriteAllBytes(path, bytes);
    }
}
