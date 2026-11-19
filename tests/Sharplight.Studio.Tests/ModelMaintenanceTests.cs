using Sharplight.Hub;
using Sharplight.Package;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.Tests;

/// <summary>Disk usage, deletion safety, resident protection, partial-download cleanup, cache clearing, and the .slm export and import round trip, on real files.</summary>
public sealed class ModelMaintenanceTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    [Fact]
    public async Task Overview_SplitsTheModelStoreIntoModelsArtifactsAndPartialDownloads()
    {
        using var fixture = new MaintenanceFixture();
        var plain = fixture.AddModel("plain", precompiled: false);
        var compiled = fixture.AddModel("compiled");
        File.WriteAllBytes(plain.PackagePath + ".pending", new byte[5000]);
        var orphan = Path.Combine(fixture.Store.Root, "tests", "orphan", "default");
        Directory.CreateDirectory(orphan);
        File.WriteAllBytes(Path.Combine(orphan, "model.gguf.partial"), new byte[7000]);

        var overview = await fixture.Maintenance.GetOverviewAsync(null, Token);

        var store = overview.Locations.Single(location => location.Title == "Model store");
        long Slice(string key) => store.Items.Single(item => item.Key == key).Bytes;
        var infoPlain = ModelPackageReader.Open(plain.PackagePath);
        var infoCompiled = ModelPackageReader.Open(compiled.PackagePath);
        long artifacts = infoCompiled.ArtifactSets.Sum(set => set.Length);
        Assert.True(artifacts > 0);
        Assert.Equal(artifacts, Slice(UsageKeys.PackageArtifacts));
        Assert.Equal(infoPlain.FileSize + infoCompiled.FileSize - artifacts, Slice(UsageKeys.Models));
        Assert.Equal(12_000, Slice(UsageKeys.Downloads));
        Assert.Equal(2, overview.PartialDownloads.Count);
        Assert.Contains(overview.PartialDownloads, item => !item.IsDirectory && item.Path.EndsWith(".pending", StringComparison.Ordinal) && item.Bytes == 5000);
        Assert.Contains(overview.PartialDownloads, item => item.IsDirectory && item.Path == orphan && item.Bytes == 7000);
        Assert.Equal(["Model store", "Studio data", "Caches", "Logs"], overview.Locations.Select(location => location.Title));
        Assert.All(overview.Locations, location => Assert.True(location.TotalBytes >= location.FreeBytes));
    }

    [Fact]
    public async Task Breakdown_ListsWeightsAttachmentsAndArtifactsPerProvider()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiled");
        var entry = await fixture.EntryAsync(manifest);

        var breakdown = await fixture.Maintenance.GetBreakdownAsync(entry, Token);

        Assert.True(breakdown.IsPackage);
        var info = ModelPackageReader.Open(manifest.PackagePath);
        Assert.Equal(info.FileSize, breakdown.TotalBytes);
        Assert.Equal(info.WeightBytes, breakdown.WeightBytes);
        // The portable (code) set and the device (binary) set have different descriptors, so they are two lines.
        Assert.Equal(2, breakdown.Artifacts.Count);
        Assert.All(breakdown.Artifacts, artifact => Assert.Equal("testgpu", artifact.Provider));
        Assert.Equal(info.ArtifactSets.Sum(set => set.Length), breakdown.ArtifactBytes);
        Assert.Equal(3, breakdown.Artifacts.Sum(artifact => artifact.Entries));
        Assert.True(breakdown.OtherBytes >= 0);
        Assert.Equal(breakdown.TotalBytes, breakdown.WeightBytes + breakdown.AttachmentBytes + breakdown.ArtifactBytes + breakdown.ReclaimableBytes + breakdown.OtherBytes);
    }

    [Fact]
    public async Task Delete_RemovesTheModelFolderAndReportsTheFreedBytes_AndForgetsItsLastUsedTime()
    {
        using var fixture = new MaintenanceFixture();
        var keep = fixture.AddModel("keep");
        var doomed = fixture.AddModel("doomed");
        var entry = await fixture.EntryAsync(doomed);
        fixture.Test.Usage.Record(doomed.PackagePath, DateTimeOffset.UtcNow);
        long expected = new FileInfo(doomed.PackagePath).Length + new FileInfo(Path.Combine(doomed.Directory, ModelStore.ManifestFileName)).Length;
        Assert.Equal(expected, fixture.Maintenance.FreedByDelete(entry));

        long freed = await fixture.Maintenance.DeleteAsync(entry, Token);

        Assert.Equal(expected, freed);
        Assert.False(Directory.Exists(doomed.Directory));
        Assert.True(File.Exists(keep.PackagePath));
        Assert.Null(fixture.Test.Usage.LastUsed(doomed.PackagePath));
        Assert.Equal(["keep"], fixture.Store.List().Select(model => model.Alias));
    }

    [Fact]
    public async Task AResidentModelIsNeverDeletedOrRewritten()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("busy");
        var entry = await fixture.EntryAsync(manifest);
        var before = File.ReadAllBytes(manifest.PackagePath);
        fixture.MakeResident(manifest);

        Assert.Contains("resident", fixture.Maintenance.WhyInUse(entry), StringComparison.Ordinal);
        await Assert.ThrowsAsync<ModelInUseException>(() => fixture.Maintenance.DeleteAsync(entry, Token));
        await Assert.ThrowsAsync<ModelInUseException>(() => fixture.Maintenance.StripPrecompiledAsync(entry, Token));

        Assert.True(File.Exists(manifest.PackagePath));
        Assert.Equal(before, File.ReadAllBytes(manifest.PackagePath));
        Assert.Single(fixture.Store.List());

        // Unloaded, the same calls work.
        await fixture.Test.Runtime.UnloadAsync("busy");
        Assert.Null(fixture.Maintenance.WhyInUse(entry));
        await fixture.Maintenance.DeleteAsync(entry, Token);
        Assert.Empty(fixture.Store.List());
    }

    [Fact]
    public async Task AModelBeingPrecompiledIsNotDeleted()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiling");
        var entry = await fixture.EntryAsync(manifest);
        fixture.Test.Precompiler.Gate = new TaskCompletionSource();
        var running = fixture.Test.Precompiler.PrecompileAsync(manifest.PackagePath, new Progress<Sharplight.Package.PrecompileProgress>(), Token);
        await Task.Delay(20, Token);

        await Assert.ThrowsAsync<ModelInUseException>(() => fixture.Maintenance.DeleteAsync(entry, Token));

        fixture.Test.Precompiler.Gate.SetResult();
        await running;
        Assert.True(File.Exists(manifest.PackagePath));
    }

    [Fact]
    public async Task AFileOutsideTheStoreIsOnlyRemovedFromTheList_NeverDeleted()
    {
        using var fixture = new MaintenanceFixture();
        var outside = Path.Combine(fixture.Root, "elsewhere", "mine.slm");
        Directory.CreateDirectory(Path.GetDirectoryName(outside)!);
        TestPackages.WriteSmall(outside);
        await fixture.Library.AddLocalAsync(outside);
        var entry = (await fixture.Library.ListAsync(Token)).Single();

        Assert.Equal(0, fixture.Maintenance.FreedByDelete(entry));
        long freed = await fixture.Maintenance.DeleteAsync(entry, Token);

        Assert.Equal(0, freed);
        Assert.True(File.Exists(outside));
        Assert.Empty(await fixture.Library.ListAsync(Token));
    }

    [Fact]
    public async Task StripPrecompiled_KeepsTheModelLoadableAndFreesTheArtifactBytes()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiled");
        var entry = await fixture.EntryAsync(manifest);
        long before = new FileInfo(manifest.PackagePath).Length;
        long estimate = await fixture.Maintenance.FreedByStripAsync(entry, Token);
        Assert.True(estimate > 0);

        long freed = await fixture.Maintenance.StripPrecompiledAsync(entry, Token);

        var info = ModelPackageReader.Open(manifest.PackagePath);
        Assert.Empty(info.ArtifactSets);
        Assert.Equal(before - freed, info.FileSize);
        // The estimate is the artifact bytes and the dead space; the section tables the rewrite drops add a few bytes.
        Assert.InRange(freed, estimate, estimate + 1024);
        Assert.Equal(2, info.Weights.Count);
        Assert.Equal("test_arch", info.Definition.Architecture);
    }

    [Fact]
    public async Task ClearCaches_EmptiesOnlyTheChosenCaches_AndKeepsTheFolders()
    {
        using var fixture = new MaintenanceFixture();
        var layout = fixture.Test.Storage.Current;
        Directory.CreateDirectory(Path.Combine(layout.ArtifactsDirectory, "cuda"));
        File.WriteAllBytes(Path.Combine(layout.ArtifactsDirectory, "cuda", "a.cubin"), new byte[3000]);
        File.WriteAllBytes(Path.Combine(layout.ArtifactsDirectory, "b.spv"), new byte[1000]);
        Directory.CreateDirectory(layout.ShaderSourcesDirectory);
        File.WriteAllBytes(Path.Combine(layout.ShaderSourcesDirectory, "s.comp"), new byte[500]);
        Directory.CreateDirectory(Path.Combine(layout.CacheDirectory, "media"));
        File.WriteAllBytes(Path.Combine(layout.CacheDirectory, "media", "thumb.png"), new byte[200]);

        Assert.Equal(4000, await fixture.Maintenance.MeasureCachesAsync(CacheKinds.Kernels, Token));
        Assert.Equal(200, await fixture.Maintenance.MeasureCachesAsync(CacheKinds.Media, Token));
        Assert.Equal(4700, await fixture.Maintenance.MeasureCachesAsync(CacheKinds.All, Token));

        long freed = await fixture.Maintenance.ClearCachesAsync(CacheKinds.Kernels, null, Token);

        Assert.Equal(4000, freed);
        Assert.True(Directory.Exists(layout.ArtifactsDirectory));
        Assert.Empty(Directory.EnumerateFiles(layout.ArtifactsDirectory, "*", SearchOption.AllDirectories));
        Assert.True(File.Exists(Path.Combine(layout.ShaderSourcesDirectory, "s.comp")));
        Assert.True(File.Exists(Path.Combine(layout.CacheDirectory, "media", "thumb.png")));

        Assert.Equal(700, await fixture.Maintenance.ClearCachesAsync(CacheKinds.Shaders | CacheKinds.Media, null, Token));
        // The shader and artifact folders survive a media clear that works on the rest of the cache folder.
        Assert.True(Directory.Exists(layout.ShaderSourcesDirectory));
    }

    [Fact]
    public async Task RemovePartialDownloads_DeletesLeftoversOnly_AndNeverTouchesModelsOrOutsideFiles()
    {
        using var fixture = new MaintenanceFixture();
        var model = fixture.AddModel("whole");
        var pending = model.PackagePath + ".pending";
        File.WriteAllBytes(pending, new byte[1234]);
        var orphan = Path.Combine(fixture.Store.Root, "tests", "half", "q4");
        Directory.CreateDirectory(Path.Combine(orphan, "source"));
        File.WriteAllBytes(Path.Combine(orphan, "source", "x.gguf.partial"), new byte[4321]);
        var outside = Path.Combine(fixture.Root, "precious.partial");
        File.WriteAllBytes(outside, new byte[10]);
        var overview = await fixture.Maintenance.GetOverviewAsync(null, Token);

        // A candidate list that also names a model folder and a file outside the store is only honoured inside the store and never for a model.
        var items = new List<PartialDownload>(overview.PartialDownloads)
        {
            new(outside, 10, false),
            new(model.Directory, 1, true),
        };
        long freed = await fixture.Maintenance.RemovePartialDownloadsAsync(items, Token);

        Assert.Equal(1234 + 4321, freed);
        Assert.False(File.Exists(pending));
        Assert.False(Directory.Exists(orphan));
        Assert.True(File.Exists(outside));
        Assert.True(File.Exists(model.PackagePath));
        Assert.Single(fixture.Store.List());
        Assert.Empty((await fixture.Maintenance.GetOverviewAsync(null, Token)).PartialDownloads);
    }

    [Fact]
    public async Task ExportAndImport_RoundTripAStandalonePackage_WithAndWithoutPrecompiledArtifacts()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiled");
        var entry = await fixture.EntryAsync(manifest);
        var full = Path.Combine(fixture.Root, "out", "full.slm");
        var lean = Path.Combine(fixture.Root, "out", "lean.slm");
        var reports = new List<MaintenanceProgress>();

        long fullSize = await fixture.Maintenance.ExportAsync(entry, full, includePrecompiled: true, new SyncProgress(reports), Token);
        long leanSize = await fixture.Maintenance.ExportAsync(entry, lean, includePrecompiled: false, null, Token);

        Assert.Equal(new FileInfo(manifest.PackagePath).Length, fullSize);
        Assert.Equal(File.ReadAllBytes(manifest.PackagePath), File.ReadAllBytes(full));
        Assert.True(leanSize < fullSize);
        Assert.Empty(ModelPackageReader.Open(lean).ArtifactSets);
        Assert.Equal(2, ModelPackageReader.Open(full).ArtifactSets.Count);
        Assert.NotEmpty(reports);
        Assert.Equal(fullSize, reports[^1].Done);
        Assert.Empty(Directory.EnumerateFiles(Path.Combine(fixture.Root, "out"), "*.export"));
        // The source package is untouched by the lean export.
        Assert.Equal(2, ModelPackageReader.Open(manifest.PackagePath).ArtifactSets.Count);

        // Import both into a second, empty store.
        using var other = new MaintenanceFixture();
        var candidates = await other.Maintenance.InspectForImportAsync(Path.Combine(fixture.Root, "out"), Token);
        Assert.Equal(2, candidates.Count);
        Assert.All(candidates, candidate => Assert.True(candidate.IsValid, candidate.Error));
        Assert.Equal("small test model".Length > 0 ? "small" : "", candidates[0].Name);

        var result = await other.Maintenance.ImportAsync(candidates, null, Token);

        Assert.Empty(result.Rejected);
        Assert.Equal(2, result.Imported.Count);
        var listed = await other.Library.ListAsync(Token);
        Assert.Equal(2, listed.Count);
        Assert.All(listed, item => Assert.Equal("test_arch", item.Architecture));
        Assert.Contains(listed, item => item.Precompiled.Count == 2);
        Assert.Contains(listed, item => item.Precompiled.Count == 0);
        Assert.Equal(["full", "lean"], other.Store.List().Select(model => model.Alias).Order(StringComparer.Ordinal));
        Assert.Equal(File.ReadAllBytes(full), File.ReadAllBytes(listed.Single(item => item.Precompiled.Count == 2).Path));
        Assert.Empty(Directory.EnumerateFiles(other.Store.Root, "*.pending", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task Import_ValidatesBeforeAdding_RejectsBrokenFilesAndNeverRegistersThem()
    {
        using var fixture = new MaintenanceFixture();
        var source = Path.Combine(fixture.Root, "incoming");
        Directory.CreateDirectory(Path.Combine(source, "nested"));
        TestPackages.WriteSmall(Path.Combine(source, "good.slm"));
        File.WriteAllText(Path.Combine(source, "garbage.slm"), "this is not a package");
        File.WriteAllBytes(Path.Combine(source, "truncated.slm"), File.ReadAllBytes(Path.Combine(source, "good.slm"))[..100]);
        TestPackages.WriteDescribed(Path.Combine(source, "nested", "image.slm"));
        File.WriteAllText(Path.Combine(source, "notes.txt"), "ignored");

        var candidates = await fixture.Maintenance.InspectForImportAsync(source, Token);

        Assert.Equal(4, candidates.Count);
        Assert.Equal(["good.slm", "image.slm"], candidates.Where(candidate => candidate.IsValid).Select(candidate => Path.GetFileName(candidate.Path)).Order(StringComparer.Ordinal));
        Assert.Contains("Not a valid package", candidates.Single(candidate => candidate.Path.EndsWith("garbage.slm", StringComparison.Ordinal)).Error, StringComparison.Ordinal);
        Assert.False(candidates.Single(candidate => candidate.Path.EndsWith("truncated.slm", StringComparison.Ordinal)).IsValid);

        var result = await fixture.Maintenance.ImportAsync(candidates, null, Token);

        Assert.Equal(2, result.Imported.Count);
        Assert.Equal(2, result.Rejected.Count);
        Assert.Equal(["good", "image"], fixture.Store.List().Select(model => model.Alias));

        // A single file that is not a package, and a missing folder.
        var wrong = await fixture.Maintenance.InspectForImportAsync(Path.Combine(source, "notes.txt"), Token);
        Assert.Contains("Only .slm", Assert.Single(wrong).Error, StringComparison.Ordinal);
        var empty = Path.Combine(fixture.Root, "empty");
        Directory.CreateDirectory(empty);
        Assert.Contains("no .slm", Assert.Single(await fixture.Maintenance.InspectForImportAsync(empty, Token)).Error, StringComparison.Ordinal);

        // Importing the same file again makes a second entry with its own name instead of overwriting the first.
        var again = await fixture.Maintenance.ImportAsync(await fixture.Maintenance.InspectForImportAsync(Path.Combine(source, "good.slm"), Token), null, Token);
        Assert.Single(again.Imported);
        Assert.Equal(["good", "good-2", "image"], fixture.Store.List().Select(model => model.Alias).Order(StringComparer.Ordinal));
    }

    [Fact]
    public async Task ACancelledExportAndImportLeaveNoPartialFiles()
    {
        using var fixture = new MaintenanceFixture();
        var manifest = fixture.AddModel("compiled");
        var entry = await fixture.EntryAsync(manifest);
        var destination = Path.Combine(fixture.Root, "out", "cancelled.slm");
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => fixture.Maintenance.ExportAsync(entry, destination, true, null, cancelled.Token));
        Assert.False(File.Exists(destination));
        Assert.Empty(Directory.EnumerateFiles(Path.Combine(fixture.Root, "out"), "*", SearchOption.AllDirectories));

        using var other = new MaintenanceFixture();
        var candidates = await other.Maintenance.InspectForImportAsync(manifest.PackagePath, Token);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => other.Maintenance.ImportAsync(candidates, null, cancelled.Token));
        Assert.Empty(other.Store.List());
        Assert.Empty((await other.Maintenance.GetOverviewAsync(null, Token)).PartialDownloads);
    }

    [Fact]
    public async Task ExportOfAFileThatIsNotAPackage_IsRefused()
    {
        using var fixture = new MaintenanceFixture();
        var gguf = Path.Combine(fixture.Root, "model.gguf");
        File.WriteAllBytes(gguf, new byte[16]);
        await fixture.Library.AddLocalAsync(gguf);
        var entry = (await fixture.Library.ListAsync(Token)).Single();

        await Assert.ThrowsAsync<InvalidOperationException>(() => fixture.Maintenance.ExportAsync(entry, Path.Combine(fixture.Root, "x.slm"), true, null, Token));
    }

    private sealed class SyncProgress(List<MaintenanceProgress> reports) : IProgress<MaintenanceProgress>
    {
        public void Report(MaintenanceProgress value) => reports.Add(value);
    }
}
