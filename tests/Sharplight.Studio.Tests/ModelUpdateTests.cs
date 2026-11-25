using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Publishing;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>Updating every feed package with a newer version, and the optional periodic check with its badge.</summary>
public sealed class ModelUpdateTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    private static async Task PublishNewVersionsAsync(FeedCatalogTests.Rig rig, byte seed)
    {
        var alpha = await rig.Feed.StageAsync("alpha.q8", "1.1.0", 20_000, [new PackageDependency { Id = "beta.f16", Range = "[1.0.0,2.0.0)" }], seed: seed);
        var beta = await rig.Feed.StageAsync("beta.f16", "1.1.0", 9_000, seed: seed);
        await rig.Feed.Publisher.PublishAsync(rig.Feed.Folder, [alpha, beta], rig.Feed.FeedKey, new FeedPublishOptions { Channel = FeedChannels.Stable }, Token);
    }

    /// <summary>The rig's services with the real library over its model store, so feed installs are listed.</summary>
    private static Studio.Services.StudioServices Real(FeedCatalogTests.Rig rig)
        => rig.All with { Library = new Studio.Services.Library.ModelLibrary(rig.Services.Settings, null!) };

    [Fact]
    public async Task UpdateAll_UpdatesEveryModelWithANewerVersion_WithSizesAndProgress()
    {
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        await rig.Catalog.Install(rig.Catalog.Entries.Single(package => package.Id == "alpha.q8")).Completion;
        await PublishNewVersionsAsync(rig, 2);
        await rig.Catalog.RefreshAsync();
        var library = new LibraryViewModel(Real(rig), new FakeNavigator()) { Feed = rig.Catalog };
        await library.RefreshAsync();
        var raised = 0;
        library.UpdatesChanged += (_, _) => raised++;
        library.ApplyFeedUpdates();

        Assert.True(library.UpdateCount == 2, $"entries: {string.Join(", ", library.Entries.Select(e => e.Entry.Id + " " + e.Entry.Manifest?.FeedVersion + " -> " + e.FeedUpdateVersion))}; listing: {string.Join(", ", rig.Catalog.AllPackages.Select(p => p.Id + " " + p.Version + " " + p.IsInstallable))}");
        Assert.True(library.HasUpdates);
        Assert.Equal(29_000, library.UpdateBytes);
        Assert.Contains("Update all (2 · ", library.UpdateAllText, StringComparison.Ordinal);
        Assert.True(raised > 0);
        var oldBytes = File.ReadAllBytes(rig.StoredPackage("alpha.q8"));

        var progress = new List<double>();
        library.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(LibraryViewModel.UpdateAllProgress))
            {
                progress.Add(library.UpdateAllProgress);
            }
        };
        await library.UpdateAllAsync();

        Assert.False(library.IsUpdatingAll);
        Assert.Contains("Updated 2 of 2 models", library.Status, StringComparison.Ordinal);
        Assert.All(rig.Store.List(), model => Assert.Equal("1.1.0", model.FeedVersion));
        Assert.NotEqual(oldBytes, File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        Assert.Empty(Directory.EnumerateFiles(rig.Services.Settings.StoreRoot, "*.pending", SearchOption.AllDirectories));
        Assert.NotEmpty(progress);
        Assert.All(progress, value => Assert.InRange(value, 0, 1));
    }

    [Fact]
    public async Task UpdateAll_KeepsTheInstalledVersionOfAModelWhoseUpdateFailed()
    {
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        await rig.Catalog.Install(rig.Catalog.Entries.Single(package => package.Id == "alpha.q8")).Completion;
        var before = File.ReadAllBytes(rig.StoredPackage("alpha.q8"));
        await PublishNewVersionsAsync(rig, 2);
        var file = rig.Feed.PathOf("packages/alpha.q8/1.1.0/" + FeedLayout.PackageFileName("alpha.q8", "1.1.0"));
        var bytes = File.ReadAllBytes(file);
        bytes[bytes.Length / 2] ^= 0xFF;
        File.WriteAllBytes(file, bytes);
        await rig.Catalog.RefreshAsync();
        var library = new LibraryViewModel(Real(rig), new FakeNavigator()) { Feed = rig.Catalog };
        await library.RefreshAsync();

        await library.UpdateAllAsync();

        Assert.Contains("Updated 1 of 2 models", library.Status, StringComparison.Ordinal);
        Assert.Contains("alpha", library.Status, StringComparison.OrdinalIgnoreCase);
        Assert.Equal("1.0.0", rig.Store.List().Single(model => model.FeedPackage == "alpha.q8").FeedVersion);
        Assert.Equal(before, File.ReadAllBytes(rig.StoredPackage("alpha.q8")));
        Assert.Equal("1.1.0", rig.Store.List().Single(model => model.FeedPackage == "beta.f16").FeedVersion);
    }

    [Fact]
    public async Task ThePeriodicCheck_RunsOnlyWhenOnAndDue_AndTheBadgeShowsTheCount()
    {
        var time = new FakeTime(new DateTimeOffset(2026, 11, 19, 8, 0, 0, TimeSpan.Zero));
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        await rig.Catalog.Install(rig.Catalog.Entries.Single(package => package.Id == "alpha.q8")).Completion;
        await PublishNewVersionsAsync(rig, 3);

        var page = new ModelsPageViewModel(Real(rig), new FakeNavigator(), time, backgroundChecks: false);
        var watcher = page.UpdateWatcher;
        await page.Library.RefreshAsync();
        int checks = 0;
        page.Discover.Feed.ListingChanged += (_, _) => checks++;
        int? reported = null;
        watcher.Checked += (_, count) => reported = count;

        Assert.False(await watcher.CheckIfDueAsync());
        Assert.Equal(0, checks);

        rig.Services.Settings.Update(current => current with { CheckModelUpdates = true });
        Assert.True(await watcher.CheckIfDueAsync());
        Assert.Equal(2, reported);
        Assert.Equal(2, page.Badge);
        Assert.Equal(time.Now, rig.Services.Settings.Current.LastModelUpdateCheck);

        time.Now = time.Now.AddHours(5);
        Assert.False(await watcher.CheckIfDueAsync());
        time.Now = time.Now.AddHours(20);
        Assert.True(await watcher.CheckIfDueAsync());
        Assert.Equal(2, checks);
    }
}
