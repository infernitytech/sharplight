using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The Storage tab: the overview per location and the cleanup actions with their confirmations.</summary>
public sealed class StorageViewModelTests
{
    private static async Task<(StorageViewModel Storage, LibraryViewModel Library, DownloadsViewModel Downloads, FeedCatalogViewModel Feed)> OpenAsync(MaintenanceFixture fixture)
    {
        var library = new LibraryViewModel(fixture.Services, new FakeNavigator(), new FakeTime(new DateTimeOffset(2026, 11, 19, 0, 0, 0, TimeSpan.Zero)));
        await library.RefreshAsync();
        var downloads = new DownloadsViewModel(fixture.Services);
        var feed = new FeedCatalogViewModel(fixture.Services, () => library.StoredManifests);
        var storage = new StorageViewModel(fixture.Services, downloads, library, feed);
        await storage.RefreshAsync();
        return (storage, library, downloads, feed);
    }

    [Fact]
    public async Task TheOverview_ShowsEveryLocationWithItsSlices_AndTheBiggestModels()
    {
        using var fixture = new MaintenanceFixture();
        fixture.AddModel("compiled");
        fixture.AddModel("plain", precompiled: false);
        var layout = fixture.Test.Storage.Current;
        Directory.CreateDirectory(layout.ArtifactsDirectory);
        File.WriteAllBytes(Path.Combine(layout.ArtifactsDirectory, "k.bin"), new byte[2048]);
        Directory.CreateDirectory(layout.LogsDirectory);
        File.WriteAllBytes(Path.Combine(layout.LogsDirectory, "studio.log"), new byte[100]);

        var (storage, _, _, _) = await OpenAsync(fixture);

        Assert.True(storage.IsMeasured);
        Assert.Equal(["Model store", "Studio data", "Caches", "Logs"], storage.Locations.Select(location => location.Title));
        Assert.Equal(2048, storage.KernelCacheBytes);
        var logs = storage.Locations.Single(location => location.Title == "Logs");
        Assert.Equal(100, logs.UsedBytes);
        var store = storage.Locations.Single(location => location.Title == "Model store");
        Assert.Contains(store.Slices, slice => slice.Title == "Precompiled in packages" && slice.Bytes > 0);
        Assert.Equal(2, storage.LargestModels.Count);
        Assert.True(storage.LargestModels[0].Bytes >= storage.LargestModels[1].Bytes);
        Assert.Equal(1.0, storage.LargestModels[0].Fraction);
        Assert.Equal("No failed or partial downloads.", storage.PartialText);
    }

    [Fact]
    public async Task ClearingACache_AsksWithTheFreedSpace_AndFreesItAfterConfirming()
    {
        using var fixture = new MaintenanceFixture();
        var layout = fixture.Test.Storage.Current;
        Directory.CreateDirectory(layout.ArtifactsDirectory);
        var file = Path.Combine(layout.ArtifactsDirectory, "k.bin");
        File.WriteAllBytes(file, new byte[5 * 1024]);
        var (storage, _, _, _) = await OpenAsync(fixture);

        await storage.ClearKernelCacheCommand.ExecuteAsync(null);

        Assert.Contains("frees 5.0 KB", storage.Confirmation!.Message, StringComparison.Ordinal);
        Assert.True(File.Exists(file));
        storage.Confirmation.CancelCommand.Execute(null);
        Assert.True(File.Exists(file));

        await storage.ClearKernelCacheCommand.ExecuteAsync(null);
        await storage.Confirmation!.ConfirmCommand.ExecuteAsync(null);

        Assert.False(File.Exists(file));
        Assert.Equal(0, storage.KernelCacheBytes);
        Assert.Contains("freed 5.0 KB", storage.Status, StringComparison.Ordinal);

        await storage.ClearKernelCacheCommand.ExecuteAsync(null);
        Assert.Null(storage.Confirmation);
        Assert.Contains("Nothing to clear", storage.Status, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PartialDownloads_AreRemovedAfterConfirming_ButNotWhileADownloadRuns()
    {
        using var fixture = new MaintenanceFixture();
        var model = fixture.AddModel("whole");
        var partial = model.PackagePath + ".pending";
        File.WriteAllBytes(partial, new byte[3 * 1024]);
        var (storage, _, _, _) = await OpenAsync(fixture);
        Assert.True(storage.HasPartialDownloads);
        Assert.Contains("1 failed or partial download", storage.PartialText, StringComparison.Ordinal);

        await storage.RemovePartialDownloadsCommand.ExecuteAsync(null);
        Assert.Contains("frees 3.0 KB", storage.Confirmation!.Message, StringComparison.Ordinal);
        await storage.Confirmation.ConfirmCommand.ExecuteAsync(null);

        Assert.False(File.Exists(partial));
        Assert.True(File.Exists(model.PackagePath));
        Assert.False(storage.HasPartialDownloads);
        Assert.Contains("freed 3.0 KB", storage.Status, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PartialDownloadsAreKept_WhileADownloadIsRunning()
    {
        using var fixture = new MaintenanceFixture();
        var model = fixture.AddModel("whole");
        var partial = model.PackagePath + ".pending";
        File.WriteAllBytes(partial, new byte[1024]);
        var rig = await OpenAsync(fixture);
        // A running pull in the download queue blocks the removal.
        var downloads = new DownloadsViewModel(fixture.Services);
        var storage = new StorageViewModel(fixture.Services, downloads, rig.Library, rig.Feed);
        await storage.RefreshAsync();
        downloads.Enqueue(Sharplight.Hub.ModelReference.Parse("owner/name:q4"), 1000);
        await fixture.Test.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);
        Assert.True(downloads.ActiveCount > 0);

        await storage.RemovePartialDownloadsCommand.ExecuteAsync(null);

        Assert.Null(storage.Confirmation);
        Assert.Contains("download is running", storage.Status, StringComparison.Ordinal);
        Assert.True(File.Exists(partial));
        fixture.Test.Hub.Release();
    }

    [Fact]
    public async Task ReviewingUnusedModels_TicksThemInTheLibrary()
    {
        using var fixture = new MaintenanceFixture();
        fixture.AddModel("ancient", pulledAt: new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero));
        var (storage, library, _, _) = await OpenAsync(fixture);
        bool requested = false;
        storage.ReviewUnusedRequested += (_, _) => requested = true;
        Assert.True(storage.HasUnusedSuggestion);

        storage.ReviewUnusedCommand.Execute(null);

        Assert.True(requested);
        Assert.True(library.IsSelecting);
        Assert.Equal(1, library.SelectedCount);
    }
}
