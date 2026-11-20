using Sharplight.Hub;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Tests;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>Discover's Sharplight feed over a local test feed: listing, verified installs with dependencies, updates, failed verification, offline use.</summary>
public sealed class FeedCatalogTests
{
    private static readonly FeedPublishOptions s_stable = new() { Channel = FeedChannels.Stable };
    private static readonly FeedPublishOptions s_unstable = new() { Channel = FeedChannels.Unstable };

    private static string PackageFile(string staged, string id, string version) => Path.Combine(staged, FeedLayout.PackageFileName(id, version));

    /// <summary>A local feed, the real feed service over it, and the view model.</summary>
    internal sealed class Rig : IDisposable
    {
        public Rig(AppSettings? settings = null)
        {
            Feed = new TestFeed();
            Services = new TestServices(settings);
            Service = new FeedService(Services.Settings, new FeedOptions
            {
                TrustedRoots = [Feed.RootKey.PublicKey],
                StateFile = Path.Combine(Services.Directory, "feed-state.json"),
                CacheDirectory = Path.Combine(Services.Directory, "feed-cache"),
                TransportFactory = _ => Offline ? new OfflineTransport() : new FolderFeedTransport(Feed.Folder),
                Time = Feed.Time,
            });
            All = Services.Services with { Feed = Service };
            Catalog = new FeedCatalogViewModel(All, () => Store.List());
        }

        public TestFeed Feed { get; }

        public TestServices Services { get; }

        public FeedService Service { get; }

        public StudioServices All { get; }

        public FeedCatalogViewModel Catalog { get; }

        public ModelStore Store => new(Services.Settings.StoreRoot);

        public bool Offline { get; set; }

        public string StoredPackage(string id) => Path.Combine(Store.GetModelDirectory(FeedService.RepositoryOf(id), ""), ModelStore.PackageFileName);

        public void Dispose()
        {
            Service.Dispose();
            Services.Dispose();
            Feed.Dispose();
        }
    }

    private sealed class OfflineTransport : IFeedTransport
    {
        public string Location => "offline";

        public Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default) => throw new IOException("The network is unreachable.");

        public Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default) => throw new IOException("The network is unreachable.");
    }

    internal static async Task<string> PublishAlphaAsync(Rig rig, string version = "1.0.0", int size = 20_000, byte seed = 1, FeedPublishOptions? options = null)
    {
        var staged = await rig.Feed.StageAsync("alpha.q8", version, size, [new PackageDependency { Id = "beta.f16", Range = "[1.0.0,2.0.0)" }], seed: seed);
        var existing = Directory.Exists(rig.Feed.PathOf("packages/beta.f16")) ? [] : new[] { await rig.Feed.StageAsync("beta.f16", "1.0.0", 9_000) };
        await rig.Feed.Publisher.PublishAsync(rig.Feed.Folder, [staged, .. existing], rig.Feed.FeedKey, options ?? s_stable, TestContext.Current.CancellationToken);
        return staged;
    }

    [Fact]
    public async Task Listing_ShowsThePackagesOfTheChannel_Verified()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);

        await rig.Catalog.LoadAsync();

        Assert.Null(rig.Catalog.Failure);
        Assert.False(rig.Catalog.IsOffline);
        Assert.Equal(["alpha.q8", "beta.f16"], rig.Catalog.Entries.Select(package => package.Id));
        Assert.All(rig.Catalog.Entries, package => Assert.True(package.IsVerified && package.IsInstallable));
        Assert.Contains("2 packages", rig.Catalog.ListingText, StringComparison.Ordinal);
        Assert.Equal("1.0.0", rig.Catalog.Entries[0].Version);
        Assert.Equal(["beta.f16"], rig.Catalog.Entries[0].Dependencies);
    }

    [Fact]
    public async Task Install_VerifiesAndStoresThePackageAndItsDependencies()
    {
        using var rig = new Rig();
        string staged = await PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        int stored = 0;
        rig.Catalog.Installed += (_, _) => stored++;

        var install = rig.Catalog.Install(rig.Catalog.Entries[0]);
        await install.Completion;

        Assert.True(install.IsDone, install.Error);
        Assert.Equal(1, install.Progress);
        Assert.Equal(1, stored);
        var models = rig.Store.List();
        Assert.Equal(["alpha.q8", "beta.f16"], models.Select(model => model.FeedPackage).Order());
        Assert.All(models, model => Assert.True(model.Format == ModelFormat.Slm && File.Exists(model.PackagePath) && model.FeedChannel == "stable"));
        Assert.Equal(File.ReadAllBytes(PackageFile(staged, "alpha.q8", "1.0.0")), File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        rig.Catalog.RefreshInstalled();
        Assert.All(rig.Catalog.AllPackages, package => Assert.Equal("1.0.0", package.InstalledVersion));
        Assert.Null(rig.Catalog.NewerVersionOf(models[0]));
        Assert.Empty(Directory.EnumerateFiles(rig.Services.Settings.StoreRoot, "*.part*", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task Install_ResumesAPartialDownload()
    {
        using var rig = new Rig();
        string staged = await PublishAlphaAsync(rig);
        var directory = rig.Store.GetModelDirectory(FeedService.RepositoryOf("alpha.q8"), "");
        Directory.CreateDirectory(directory);
        var bytes = File.ReadAllBytes(PackageFile(staged, "alpha.q8", "1.0.0"));
        await File.WriteAllBytesAsync(Path.Combine(directory, "model.slm.pending.part"), bytes[..(2 * (int)TestFeed.ChunkSize)], TestContext.Current.CancellationToken);
        var reports = new List<FeedInstallProgress>();

        await rig.Service.InstallAsync("alpha.q8", new SynchronousReport(reports), TestContext.Current.CancellationToken);

        var firstDownload = reports.First(report => report.Phase == FeedInstallPhase.Downloading && report.PackageId == "alpha.q8");
        Assert.True(firstDownload.Received >= 2 * TestFeed.ChunkSize, $"resumed at {firstDownload.Received}");
        Assert.Equal(bytes, File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        Assert.Equal(FeedInstallPhase.Done, reports[^1].Phase);
    }

    private sealed class SynchronousReport(List<FeedInstallProgress> reports) : IProgress<FeedInstallProgress>
    {
        public void Report(FeedInstallProgress value) => reports.Add(value);
    }

    [Fact]
    public async Task Update_IsOfferedAndReplacesAtomically_KeepingTheOldVersionUntilTheNewOneVerified()
    {
        using var rig = new Rig();
        string v1 = await PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        await rig.Catalog.Install(rig.Catalog.Entries[0]).Completion;
        var oldBytes = File.ReadAllBytes(rig.StoredPackage("alpha.q8"));

        string v2 = await PublishAlphaAsync(rig, "1.1.0", seed: 2);
        await rig.Catalog.RefreshAsync();
        var installed = rig.Store.List().Single(model => model.FeedPackage == "alpha.q8");
        Assert.Equal("1.1.0", rig.Catalog.NewerVersionOf(installed));
        Assert.True(rig.Catalog.AllPackages.Single(package => package.Id == "alpha.q8").HasUpdate);

        // A corrupted package file in the feed: nothing changes, the old version stays loadable.
        var feedFile = rig.Feed.PathOf("packages/alpha.q8/1.1.0/" + FeedLayout.PackageFileName("alpha.q8", "1.1.0"));
        var good = File.ReadAllBytes(feedFile);
        var corrupted = (byte[])good.Clone();
        corrupted[corrupted.Length / 2] ^= 0xFF;
        File.WriteAllBytes(feedFile, corrupted);
        var failed = rig.Catalog.Update("alpha.q8")!;
        await failed.Completion;
        Assert.True(failed.IsVerificationFailure);
        Assert.Contains("verification failed", failed.Error, StringComparison.Ordinal);
        Assert.Equal("1.0.0", rig.Store.List().Single(model => model.FeedPackage == "alpha.q8").FeedVersion);
        Assert.Equal(oldBytes, File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        // Only chunks that verified may remain (as a partial file the next try resumes from); never a finished pending package.
        Assert.Empty(Directory.EnumerateFiles(rig.Services.Settings.StoreRoot, "*.pending", SearchOption.AllDirectories));

        File.WriteAllBytes(feedFile, good);
        var update = rig.Catalog.Update("alpha.q8")!;
        await update.Completion;
        Assert.True(update.IsDone, update.Error);
        var now = rig.Store.List().Single(model => model.FeedPackage == "alpha.q8");
        Assert.Equal("1.1.0", now.FeedVersion);
        Assert.Equal(File.ReadAllBytes(PackageFile(v2, "alpha.q8", "1.1.0")), File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        Assert.NotEqual(File.ReadAllBytes(PackageFile(v1, "alpha.q8", "1.0.0")), File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        Assert.Null(rig.Catalog.NewerVersionOf(now));
    }

    [Fact]
    public async Task ATamperedIndex_IsRefused_AndNothingIsListedOrInstalled()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        File.AppendAllText(rig.Feed.PathOf("stable/index.json"), " ");

        await rig.Catalog.LoadAsync();

        Assert.True(rig.Catalog.IsVerificationFailure);
        Assert.Equal(FeedFailureKind.Verification, rig.Catalog.Failure!.Kind);
        Assert.Contains("failed verification", rig.Catalog.Failure.Message, StringComparison.Ordinal);
        Assert.Empty(rig.Catalog.Entries);
        await Assert.ThrowsAsync<FeedVerificationException>(() => rig.Service.InstallAsync("alpha.q8", null, TestContext.Current.CancellationToken));
        Assert.Empty(rig.Store.List());
    }

    [Fact]
    public async Task ATamperedManifest_FailsTheInstall_AndStoresNothing()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        File.AppendAllText(rig.Feed.PathOf("packages/alpha.q8/1.0.0/manifest.json"), " ");

        var install = rig.Catalog.Install(rig.Catalog.Entries[0]);
        await install.Completion;

        Assert.True(install.IsVerificationFailure);
        Assert.False(install.CanRetry);
        Assert.False(install.IsDone);
        Assert.Empty(rig.Store.List());
    }

    [Fact]
    public async Task AFeedSignedByAnUntrustedRoot_IsRefused()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        using var stranger = new TestFeed();
        var untrusted = new FeedService(rig.Services.Settings, new FeedOptions
        {
            TrustedRoots = [stranger.RootKey.PublicKey],
            StateFile = Path.Combine(rig.Services.Directory, "other-state.json"),
            CacheDirectory = Path.Combine(rig.Services.Directory, "other-cache"),
            TransportFactory = _ => new FolderFeedTransport(rig.Feed.Folder),
            Time = rig.Feed.Time,
        });

        var listing = await untrusted.GetListingAsync(TestContext.Current.CancellationToken);

        Assert.Equal(FeedFailureKind.Verification, listing.Failure!.Kind);
        Assert.Empty(listing.Packages);
    }

    [Fact]
    public async Task Offline_UsesTheLastVerifiedListing_AndSaysSo()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        Assert.False(rig.Catalog.IsOffline);

        rig.Offline = true;
        await rig.Catalog.RefreshAsync();

        Assert.True(rig.Catalog.IsOffline);
        Assert.Null(rig.Catalog.Failure);
        Assert.Equal(2, rig.Catalog.Entries.Count);
        Assert.Contains("last verified listing", rig.Catalog.OfflineText, StringComparison.Ordinal);
        var install = rig.Catalog.Install(rig.Catalog.Entries[0]);
        await install.Completion;
        Assert.False(install.IsDone);
        Assert.NotNull(install.Error);
        Assert.Empty(rig.Store.List());
    }

    [Fact]
    public async Task Offline_AcceptsACachedIndexThatExpiredMeanwhile_ButAnOnlineExpiredFeedIsRefused()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        rig.Feed.Time.Now += TimeSpan.FromDays(120);

        await rig.Catalog.RefreshAsync();
        Assert.True(rig.Catalog.IsVerificationFailure);
        Assert.Contains("expired", rig.Catalog.Failure!.Message, StringComparison.Ordinal);

        rig.Offline = true;
        await rig.Catalog.RefreshAsync();
        Assert.True(rig.Catalog.IsOffline);
        Assert.Equal(2, rig.Catalog.Entries.Count);
    }

    [Fact]
    public async Task Offline_WithoutAnEarlierVisit_ExplainsThatNothingIsCached()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        rig.Offline = true;

        await rig.Catalog.LoadAsync();

        Assert.Equal(FeedFailureKind.Offline, rig.Catalog.Failure!.Kind);
        Assert.Empty(rig.Catalog.Entries);
        Assert.Contains("no verified copy", rig.Catalog.Failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheChannelSetting_SelectsTheListing_AndChangingItReloads()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig, options: s_unstable);
        await rig.Feed.Publisher.PromoteAsync(rig.Feed.Folder, FeedChannels.Stable, ["beta.f16"], FeedChannels.Unstable, rig.Feed.FeedKey, cancellationToken: TestContext.Current.CancellationToken);
        await rig.Catalog.LoadAsync();
        Assert.Equal(["beta.f16"], rig.Catalog.Entries.Select(package => package.Id));

        rig.Services.Settings.Update(current => current with { FeedChannel = "unstable" });
        await rig.Catalog.LoadAsync();

        Assert.Equal("unstable", rig.Service.Channel);
        Assert.Equal(["alpha.q8", "beta.f16"], rig.Catalog.Entries.Select(package => package.Id));
    }

    [Fact]
    public async Task VariantsOfAModel_AreListedTogether_SmallestFirst_WithRecommendedAndPartialBadges()
    {
        using var rig = new Rig();
        Func<PackageManifest, PackageManifest> Variant(string variant, double bits, bool recommended, string level) => manifest => manifest with
        {
            Group = new PackageGroup { Id = "model", Name = "Model", Variant = variant, BitsPerWeight = bits, Recommended = recommended, Quality = PackageQualityTiers.Good, QualityNote = "4 bit.", Reference = "BF16" },
            Verification = new PackageVerification { Level = level, Summary = level == "partial" ? "Not run on this file." : "Identical." },
        };
        var folders = new[]
        {
            await rig.Feed.StageAsync("model.q8", "1.0.0", customize: Variant("q8", 8.5, false, "verified")),
            await rig.Feed.StageAsync("model.iq2", "1.0.0", customize: Variant("iq2", 2.4, false, "partial")),
            await rig.Feed.StageAsync("model.q4", "1.0.0", customize: Variant("q4", 4.8, true, "verified")),
            await rig.Feed.StageAsync("aaa.f16", "1.0.0"),
        };
        await rig.Feed.Publisher.PublishAsync(rig.Feed.Folder, folders, rig.Feed.FeedKey, s_stable, TestContext.Current.CancellationToken);

        await rig.Catalog.LoadAsync();

        Assert.Equal(["aaa.f16", "model.iq2", "model.q4", "model.q8"], rig.Catalog.Entries.Select(package => package.Id));
        var iq2 = rig.Catalog.Entries[1];
        Assert.True(iq2.IsPartial);
        Assert.Equal("Not run on this file.", iq2.PartialText);
        Assert.False(iq2.IsRecommended);
        Assert.True(rig.Catalog.Entries[2].IsRecommended);
        Assert.False(rig.Catalog.Entries[2].IsPartial);
        Assert.Contains("2.4 bits per weight", iq2.Details, StringComparison.Ordinal);
        Assert.Contains("good vs BF16: 4 bit.", iq2.Details, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Filters_NarrowTheListing()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();

        rig.Catalog.Query = "alpha";
        Assert.Equal(["alpha.q8"], rig.Catalog.Entries.Select(package => package.Id));
        rig.Catalog.Query = "nothing-like-this";
        Assert.Empty(rig.Catalog.Entries);
        Assert.Equal("No package matches.", rig.Catalog.Status);
    }

    [Fact]
    public async Task Discover_HasAFeedMode()
    {
        using var rig = new Rig();
        await PublishAlphaAsync(rig);
        var search = new HubSearchViewModel(rig.All, new DownloadsViewModel(rig.All), () => new HashSet<string>(), () => rig.Store.List());

        search.ShowFeedCommand.Execute(null);
        await search.Feed.LoadAsync();

        Assert.True(search.IsFeedMode);
        Assert.False(search.IsHubMode);
        Assert.False(search.IsVerifiedMode);
        Assert.Equal(2, search.Feed.Entries.Count);
        search.ShowHubSearchCommand.Execute(null);
        Assert.True(search.IsHubMode);
        Assert.False(search.IsFeedMode);
    }
}
