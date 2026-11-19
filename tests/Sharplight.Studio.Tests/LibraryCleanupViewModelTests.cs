using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The library's disk management over real files: sorting, confirmations that name the freed space, resident protection, bulk actions, unused models, export and import.</summary>
public sealed class LibraryCleanupViewModelTests
{
    private static readonly DateTimeOffset s_now = new(2026, 11, 19, 12, 0, 0, TimeSpan.Zero);

    private static CancellationToken Token => TestContext.Current.CancellationToken;

    private static async Task<LibraryViewModel> OpenAsync(MaintenanceFixture fixture, FakeTime? time = null)
    {
        var library = new LibraryViewModel(fixture.Services, new FakeNavigator(), time ?? new FakeTime(s_now));
        await library.RefreshAsync();
        return library;
    }

    [Fact]
    public async Task SortBySizeAndLastUsed_OrdersTheVisibleModels()
    {
        using var fixture = new MaintenanceFixture();
        fixture.AddModel("small", precompiled: false, pulledAt: s_now.AddDays(-200));
        var big = fixture.AddModel("big", pulledAt: s_now.AddDays(-200));
        var middle = fixture.AddModel("middle", pulledAt: s_now.AddDays(-200));
        File.AppendAllText(middle.PackagePath, new string('x', 100));
        fixture.Test.Usage.Record(big.PackagePath, s_now.AddDays(-50));
        fixture.Test.Usage.Record(middle.PackagePath, s_now.AddDays(-1));
        var library = await OpenAsync(fixture);

        Assert.Equal(["big", "middle", "small"], library.VisibleEntries.Select(entry => entry.Entry.Id));

        library.SelectedSort = library.SortChoices.Single(choice => choice.Value == LibrarySort.Size);
        var sizes = library.VisibleEntries.Select(entry => entry.SizeBytes).ToList();
        Assert.Equal(sizes.OrderByDescending(size => size), sizes);

        library.SelectedSort = library.SortChoices.Single(choice => choice.Value == LibrarySort.LastUsed);
        Assert.Equal("middle", library.VisibleEntries[0].Entry.Id);
        Assert.Equal("big", library.VisibleEntries[1].Entry.Id);

        library.SelectedSort = library.SortChoices.Single(choice => choice.Value == LibrarySort.LeastRecentlyUsed);
        // A model never loaded counts from the day it was added, 200 days ago: the oldest.
        Assert.Equal("small", library.VisibleEntries[0].Entry.Id);
        Assert.Equal("big", library.VisibleEntries[1].Entry.Id);
        Assert.Contains("Used", library.Entries.Single(entry => entry.Entry.Id == "middle").LastUsedText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Delete_AsksWithTheFreedSpace_AndOnlyDeletesAfterConfirming()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("doomed");
        fixture.AddModel("stays");
        var library = await OpenAsync(fixture);
        var entry = library.Entries.Single(item => item.Entry.Id == "doomed");
        long expected = new FileInfo(manifest.PackagePath).Length;

        await library.DeleteCommand.ExecuteAsync(entry);

        Assert.NotNull(library.Confirmation);
        Assert.Contains("Delete", library.Confirmation.Message, StringComparison.Ordinal);
        Assert.Contains($"frees {ByteFormat.Format(expected)}", library.Confirmation.Message, StringComparison.Ordinal);
        Assert.True(File.Exists(manifest.PackagePath));

        library.Confirmation.CancelCommand.Execute(null);
        Assert.Null(library.Confirmation);
        Assert.True(File.Exists(manifest.PackagePath));
        Assert.Equal(2, library.Entries.Count);

        await library.DeleteCommand.ExecuteAsync(entry);
        await library.Confirmation!.ConfirmCommand.ExecuteAsync(null);

        Assert.Null(library.Confirmation);
        Assert.False(Directory.Exists(manifest.Directory));
        Assert.Equal(["stays"], library.Entries.Select(item => item.Entry.Id));
        Assert.Equal(["stays"], library.VisibleEntries.Select(item => item.Entry.Id));
        Assert.Contains("Deleted 1 model, freed", library.Status, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AResidentModelIsRefusedWithAMessage_AndNothingIsAskedOrDeleted()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("busy");
        var library = await OpenAsync(fixture);
        fixture.MakeResident(manifest);
        await Task.Yield();
        var entry = library.Entries.Single();
        entry.IsLoaded = true;

        await library.DeleteCommand.ExecuteAsync(entry);

        Assert.Null(library.Confirmation);
        Assert.Contains("Unload busy before deleting it", library.Status, StringComparison.Ordinal);
        Assert.True(File.Exists(manifest.PackagePath));

        // The runtime is asked, not only the flag of the card: a stale card cannot delete a model that became resident.
        entry.IsLoaded = false;
        await library.DeleteCommand.ExecuteAsync(entry);
        Assert.Null(library.Confirmation);
        Assert.Contains("resident", library.Status, StringComparison.Ordinal);
        Assert.True(File.Exists(manifest.PackagePath));
    }

    [Fact]
    public async Task ABulkDelete_SkipsTheResidentModelAndSaysSo()
    {
        using var fixture = new MaintenanceFixture();
        var a = fixture.AddModel("a");
        var b = fixture.AddModel("b", precompiled: false);
        var resident = fixture.AddModel("resident");
        var library = await OpenAsync(fixture);
        fixture.MakeResident(resident);
        library.IsSelecting = true;
        library.SelectAllCommand.Execute(null);
        Assert.Equal(3, library.SelectedCount);
        Assert.True(library.HasSelection);

        await library.DeleteSelectedCommand.ExecuteAsync(null);

        var message = library.Confirmation!.Message;
        long freed = new FileInfo(a.PackagePath).Length + new FileInfo(b.PackagePath).Length
            + new FileInfo(Path.Combine(a.Directory, "manifest.json")).Length + new FileInfo(Path.Combine(b.Directory, "manifest.json")).Length;
        Assert.Contains("Delete 2 models", message, StringComparison.Ordinal);
        Assert.Contains($"frees {ByteFormat.Format(freed)}", message, StringComparison.Ordinal);
        Assert.Contains("1 model is skipped", message, StringComparison.Ordinal);
        Assert.Contains("resident", message, StringComparison.Ordinal);

        await library.Confirmation.ConfirmCommand.ExecuteAsync(null);

        Assert.Equal(["resident"], library.Entries.Select(item => item.Entry.Id));
        Assert.True(File.Exists(resident.PackagePath));
        Assert.False(Directory.Exists(a.Directory));
        Assert.False(Directory.Exists(b.Directory));
    }

    [Fact]
    public async Task RemovingOnlyThePrecompiledArtifacts_KeepsTheModelAndNamesTheFreedSpace()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiled");
        fixture.AddModel("plain", precompiled: false);
        var library = await OpenAsync(fixture);
        var compiled = library.Entries.Single(item => item.Entry.Id == "compiled");
        var plain = library.Entries.Single(item => item.Entry.Id == "plain");
        Assert.True(compiled.IsPrecompiled);
        long artifacts = ModelPackageReader.Open(manifest.PackagePath).ArtifactSets.Sum(set => set.Length);

        await library.StripPrecompiledCommand.ExecuteAsync(plain);
        Assert.Null(library.Confirmation);
        Assert.Contains("no precompiled artifacts", library.Status, StringComparison.Ordinal);

        await library.StripPrecompiledCommand.ExecuteAsync(compiled);
        Assert.Contains($"frees {ByteFormat.Format(artifacts)}", library.Confirmation!.Message, StringComparison.Ordinal);
        await library.Confirmation.ConfirmCommand.ExecuteAsync(null);

        Assert.True(File.Exists(manifest.PackagePath));
        Assert.Empty(ModelPackageReader.Open(manifest.PackagePath).ArtifactSets);
        Assert.False(compiled.IsPrecompiled);
        Assert.Contains("freed", library.Status, StringComparison.Ordinal);
        Assert.Equal(2, library.Entries.Count);
    }

    [Fact]
    public async Task ModelsUnusedFor90Days_AreSuggested_AndReviewTicksThem()
    {
        using var fixture = new MaintenanceFixture();
        var old = fixture.AddModel("old", pulledAt: s_now.AddDays(-200));
        var usedLongAgo = fixture.AddModel("used-long-ago", pulledAt: s_now.AddDays(-300));
        var recent = fixture.AddModel("recent", pulledAt: s_now.AddDays(-300));
        var fresh = fixture.AddModel("fresh", pulledAt: s_now.AddDays(-10));
        fixture.Test.Usage.Record(usedLongAgo.PackagePath, s_now.AddDays(-120));
        fixture.Test.Usage.Record(recent.PackagePath, s_now.AddDays(-5));
        var library = await OpenAsync(fixture);

        Assert.True(library.HasUnusedSuggestion);
        Assert.Equal(["old", "used-long-ago"], library.UnusedEntries.Select(entry => entry.Entry.Id).Order(StringComparer.Ordinal));
        Assert.Contains("2 models were not loaded for 90 days", library.UnusedSuggestion, StringComparison.Ordinal);

        library.ReviewUnusedCommand.Execute(null);

        Assert.True(library.IsSelecting);
        Assert.Equal(["old", "used-long-ago"], library.Entries.Where(entry => entry.IsSelected).Select(entry => entry.Entry.Id).Order(StringComparer.Ordinal));

        // A resident model is never suggested, and loading one stamps it.
        fixture.MakeResident(old);
        await Task.Yield();
        Assert.DoesNotContain(library.UnusedEntries, entry => entry.Entry.Id == "old");
        Assert.Equal(s_now, library.Entries.Single(entry => entry.Entry.Id == "old").LastUsed);
        Assert.Equal(s_now, fixture.Test.Usage.LastUsed(old.PackagePath));
        Assert.NotNull(fresh);
    }

    [Fact]
    public async Task SizeDetails_ListWeightsAndArtifactsPerProviderAndDevice()
    {
        using var fixture = new MaintenanceFixture();
        fixture.AddModel("compiled");
        var library = await OpenAsync(fixture);
        var entry = library.Entries.Single();

        await library.ToggleBreakdownCommand.ExecuteAsync(entry);

        Assert.True(entry.IsBreakdownOpen);
        Assert.False(entry.IsLoadingBreakdown);
        Assert.Contains(entry.BreakdownLines, line => line.Title == "Weights" && line.Bytes > 0);
        var artifacts = entry.BreakdownLines.Where(line => line.Title.StartsWith("Precompiled", StringComparison.Ordinal)).ToList();
        Assert.Equal(2, artifacts.Count);
        Assert.All(artifacts, artifact => Assert.Contains("testgpu", artifact.Title, StringComparison.Ordinal));
        Assert.All(artifacts, artifact => Assert.InRange(artifact.Fraction, 0.0001, 1));

        await library.ToggleBreakdownCommand.ExecuteAsync(entry);
        Assert.False(entry.IsBreakdownOpen);
    }

    [Fact]
    public async Task Export_WritesAStandaloneFile_WithOrWithoutPrecompiledArtifacts()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiled");
        var library = await OpenAsync(fixture);
        var entry = library.Entries.Single();
        var full = Path.Combine(fixture.Root, "exports", "full.slm");
        var lean = Path.Combine(fixture.Root, "exports", "lean.slm");

        await library.ExportAsync(entry, includePrecompiled: true, full);
        Assert.Contains("Exported tests/compiled", library.Status, StringComparison.Ordinal);
        await library.ExportAsync(entry, includePrecompiled: false, lean);

        Assert.Contains("without precompiled artifacts", library.Status, StringComparison.Ordinal);
        Assert.Equal(2, ModelPackageReader.Open(full).ArtifactSets.Count);
        Assert.Empty(ModelPackageReader.Open(lean).ArtifactSets);
        Assert.Equal(2, ModelPackageReader.Open(manifest.PackagePath).ArtifactSets.Count);

        // Cancelling the save dialog exports nothing.
        fixture.Test.Platform.SaveTo = null;
        await library.ExportCommand.ExecuteAsync(entry);
        Assert.False(library.Operation.IsRunning);
    }

    [Fact]
    public async Task DroppedFilesAndFolders_AreValidatedFirst_ThenImportedAfterConfirming()
    {
        using var source = new MaintenanceFixture();
        source.AddModel("a");
        source.AddModel("b", precompiled: false);
        var incoming = Path.Combine(source.Root, "incoming");
        Directory.CreateDirectory(incoming);
        File.Copy(source.Store.List().Single(model => model.Alias == "a").PackagePath, Path.Combine(incoming, "a.slm"));
        File.Copy(source.Store.List().Single(model => model.Alias == "b").PackagePath, Path.Combine(incoming, "b.slm"));
        File.WriteAllText(Path.Combine(incoming, "broken.slm"), "nope");

        using var fixture = new MaintenanceFixture();
        var library = await OpenAsync(fixture);
        Assert.True(library.IsEmpty);

        await library.ImportPathsAsync([incoming]);

        var question = library.Confirmation!.Message;
        Assert.Contains("Import 2 packages", question, StringComparison.Ordinal);
        Assert.Contains("broken.slm", question, StringComparison.Ordinal);
        Assert.Empty(fixture.Store.List());

        await library.Confirmation.ConfirmCommand.ExecuteAsync(null);

        Assert.Equal(["a", "b"], library.Entries.Select(entry => entry.Entry.Id).Order(StringComparer.Ordinal));
        Assert.Contains("Imported 2 models", library.Status, StringComparison.Ordinal);
        Assert.Contains("broken.slm", library.Status, StringComparison.Ordinal);

        // Something that is not a package is reported and nothing is asked.
        var text = Path.Combine(fixture.Root, "readme.txt");
        File.WriteAllText(text, "hello");
        await library.ImportPathsAsync([text]);
        Assert.Null(library.Confirmation);
        Assert.Contains("Nothing to import", library.Status, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheFileOperationCanBeCancelled()
    {
        using var fixture = new MaintenanceFixture();
        var library = await OpenAsync(fixture);
        var started = new TaskCompletionSource();

        var run = library.Operation.RunAsync("Waiting", async (_, token) =>
        {
            started.SetResult();
            await Task.Delay(Timeout.Infinite, token);
        });
        await started.Task.WaitAsync(Token);
        Assert.True(library.Operation.IsRunning);

        library.Operation.CancelCommand.Execute(null);

        Assert.False(await run.WaitAsync(Token));
        Assert.False(library.Operation.IsRunning);
    }
}
