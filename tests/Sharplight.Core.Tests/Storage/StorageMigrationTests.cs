using Sharplight.Storage;
using Xunit;

namespace Sharplight.Core.Tests.Storage;

/// <summary>
/// Changing the storage location with the existing files: verified copies, the move finished at the next start (only for originals
/// unchanged since the copy), interruption safety, models left in place, and folders that would copy into themselves.
/// </summary>
public sealed class StorageMigrationTests : IDisposable
{
    private readonly StorageTestEnvironment _test = new();

    public void Dispose() => _test.Dispose();

    [Fact]
    public async Task Copy_TakesEveryCategoryAlong_AndKeepsTheOriginals()
    {
        var system = _test.Resolve();
        Seed(system);
        var portable = StorageLayout.Create(StorageMode.Portable, null, _test.Environment);

        var result = await new StorageMigration(system, portable, Enum.GetValues<StorageCategory>()).RunAsync(StorageTransferKind.Copy);

        Assert.Equal(5, result.FilesCopied);
        AssertSeeded(portable);
        AssertSeeded(system);
        Assert.Equal(StorageMode.Portable, _test.Resolve().Mode);   // committed for the next start
        Assert.False(File.Exists(Path.Combine(portable.ConfigDirectory, StorageMigration.JournalFileName)));
        Assert.Empty(Directory.EnumerateFiles(portable.Root!, "*.sl-partial", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task Move_DeletesTheOriginalsOnlyAtTheNextStart()
    {
        var system = _test.Resolve();
        Seed(system);
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);

        await new StorageMigration(system, custom, Enum.GetValues<StorageCategory>()).RunAsync(StorageTransferKind.Move);

        AssertSeeded(custom);
        Assert.True(File.Exists(Path.Combine(system.ConfigDirectory, "studio", "settings.json")));   // still open by the running app

        var next = _test.Resolve();
        Assert.True(next.HasSameFolders(custom));
        Assert.Equal(4, StorageMigration.CompletePending(next));   // the models were renamed in one step

        Assert.False(Directory.Exists(Path.Combine(system.DataDirectory, "studio")));
        Assert.False(Directory.Exists(system.ModelsDirectory));
        Assert.False(Directory.Exists(system.CacheDirectory));
        Assert.True(File.Exists(next.PointerFile));   // the pointer in the old configuration folder is not part of the move
        AssertSeeded(custom);
        Assert.Equal(0, StorageMigration.CompletePending(next));   // the journal is gone
    }

    [Fact]
    public async Task Move_KeepsOriginalsThatChangedAfterTheCopy()
    {
        var system = _test.Resolve();
        Seed(system);
        var history = Path.Combine(system.DataDirectory, "studio", "pages", "history.json");
        Write(history, "[]");
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);
        await new StorageMigration(system, custom, [StorageCategory.Data]).RunAsync(StorageTransferKind.Move);

        var database = Path.Combine(system.DataDirectory, "studio", "studio.db");
        File.AppendAllText(database, "written after the copy");
        File.SetLastWriteTimeUtc(database, DateTime.UtcNow.AddMinutes(1));

        Assert.Equal(1, StorageMigration.CompletePending(_test.Resolve()));
        Assert.True(File.Exists(database));
        Assert.False(File.Exists(history));
    }

    [Fact]
    public async Task KeepModels_LeavesTheModelStoreAlone()
    {
        var system = _test.Resolve();
        Seed(system);
        var portable = StorageLayout.Create(StorageMode.Portable, null, _test.Environment);
        var categories = Enum.GetValues<StorageCategory>().Where(category => category != StorageCategory.Models);

        await new StorageMigration(system, portable, categories).RunAsync(StorageTransferKind.Move);
        StorageMigration.CompletePending(_test.Resolve());

        Assert.False(Directory.Exists(portable.ModelsDirectory));
        Assert.Equal("weights", File.ReadAllText(Path.Combine(system.ModelsDirectory, "owner", "model", "default", "model.slm")));
        Assert.Equal("{}", File.ReadAllText(Path.Combine(portable.ConfigDirectory, "studio", "settings.json")));
    }

    [Fact]
    public async Task StartFresh_OnlyRecordsTheNewLocation()
    {
        var system = _test.Resolve();
        Seed(system);
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);

        var result = await new StorageMigration(system, custom, Enum.GetValues<StorageCategory>()).RunAsync(StorageTransferKind.StartFresh);

        Assert.Equal(0, result.FilesCopied);
        Assert.False(Directory.Exists(custom.Root));
        Assert.Equal(StorageMode.Custom, _test.Resolve().Mode);
        AssertSeeded(system);
    }

    [Fact]
    public async Task Cancelled_LeavesTheOldLocationInEffect_AndTheNextAttemptReusesTheCopies()
    {
        var system = _test.Resolve();
        Seed(system);
        var portable = StorageLayout.Create(StorageMode.Portable, null, _test.Environment);
        using var cancellation = new CancellationTokenSource();
        var progress = new SynchronousProgress<StorageMigrationProgress>(value =>
        {
            if (value.FilesDone == 2)
            {
                cancellation.Cancel();
            }
        });

        var migration = new StorageMigration(system, portable, Enum.GetValues<StorageCategory>());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => migration.RunAsync(StorageTransferKind.Move, progress, cancellation.Token));

        Assert.Equal(StorageMode.System, _test.Resolve().Mode);
        Assert.Equal(0, StorageMigration.CompletePending(_test.Resolve()));
        AssertSeeded(system);

        var result = await migration.RunAsync(StorageTransferKind.Copy);
        Assert.Equal(5, result.FilesCopied);
        AssertSeeded(portable);
    }

    [Fact]
    public async Task InterruptedBeforeTheCommit_TheJournalNeverDeletesAnything()
    {
        var system = _test.Resolve();
        Seed(system);
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);
        await new StorageMigration(system, custom, [StorageCategory.Config, StorageCategory.Data]).RunAsync(StorageTransferKind.Move);

        // Simulate a crash between the journal and the commit: the journal says "not committed".
        var journal = Path.Combine(custom.ConfigDirectory, StorageMigration.JournalFileName);
        File.WriteAllText(journal, File.ReadAllText(journal).Replace("\"committed\": true", "\"committed\": false", StringComparison.Ordinal));

        Assert.Equal(0, StorageMigration.CompletePending(_test.Resolve()));
        Assert.False(File.Exists(journal));
        AssertSeeded(system);
    }

    [Fact]
    public async Task AJournalOfAnotherLayout_IsIgnored()
    {
        var system = _test.Resolve();
        Seed(system);
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);
        await new StorageMigration(system, custom, [StorageCategory.Data]).RunAsync(StorageTransferKind.Move);
        StorageSelection.Apply(system, StorageMode.System, null);   // the user went back before restarting

        Assert.Equal(0, StorageMigration.CompletePending(_test.Resolve()));
        AssertSeeded(system);
    }

    [Fact]
    public async Task AGivenUpMove_IsForgottenByTheNextChangeIntoTheSameLocation()
    {
        var system = _test.Resolve();
        Seed(system);
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);
        await new StorageMigration(system, custom, [StorageCategory.Data]).RunAsync(StorageTransferKind.Move);
        StorageSelection.Apply(system, StorageMode.System, null);   // given up before the restart; the system folders stay in use

        await new StorageMigration(_test.Resolve(), custom, [StorageCategory.Data]).RunAsync(StorageTransferKind.StartFresh);

        Assert.Equal(0, StorageMigration.CompletePending(_test.Resolve()));
        AssertSeeded(system);
    }

    [Fact]
    public void NestedFolders_AreRejected()
    {
        var system = _test.Resolve();
        var inside = StorageLayout.Create(StorageMode.Custom, Path.Combine(system.DataDirectory, "mine"), _test.Environment);

        var problem = new StorageMigration(system, inside, Enum.GetValues<StorageCategory>()).Validate();

        Assert.NotNull(problem);
        Assert.Contains("inside", problem, StringComparison.Ordinal);
        Assert.Null(new StorageMigration(system, StorageLayout.Create(StorageMode.Portable, null, _test.Environment), Enum.GetValues<StorageCategory>()).Validate());
    }

    [Fact]
    public async Task NestedCategoryFolders_AreCopiedOnce()
    {
        // Windows nests the cache and logs in the data folder; each file goes to its own category's new folder.
        _test.Platform = StoragePlatform.Windows;
        _test.Variables["APPDATA"] = _test.Combine("Roaming");
        _test.Variables["LOCALAPPDATA"] = _test.Combine("Local");
        var system = _test.Resolve();
        Seed(system);
        var custom = StorageLayout.Create(StorageMode.Custom, _test.Combine("custom"), _test.Environment);

        var result = await new StorageMigration(system, custom, Enum.GetValues<StorageCategory>()).RunAsync(StorageTransferKind.Copy);

        Assert.Equal(5, result.FilesCopied);
        AssertSeeded(custom);
        Assert.False(Directory.Exists(Path.Combine(custom.DataDirectory, "cache", "cache")));
    }

    [Fact]
    public void Progress_ReportsTheFraction()
    {
        Assert.Equal(0.5, new StorageMigrationProgress(null, 50, 100, 1, 4).Fraction);
        Assert.Equal(0.25, new StorageMigrationProgress(null, 0, 0, 1, 4).Fraction);
        Assert.Equal(1, new StorageMigrationProgress(null, 0, 0, 0, 0).Fraction);
    }

    private static void Seed(StorageLayout layout)
    {
        Write(Path.Combine(layout.ConfigDirectory, "studio", "settings.json"), "{}");
        Write(Path.Combine(layout.DataDirectory, "studio", "studio.db"), "database");
        Write(Path.Combine(layout.ModelsDirectory, "owner", "model", "default", "model.slm"), "weights");
        Write(Path.Combine(layout.ArtifactsDirectory, "set", "entry.slc"), "kernel");
        Write(Path.Combine(layout.LogsDirectory, "studio.log"), "log");
    }

    private static void AssertSeeded(StorageLayout layout)
    {
        Assert.Equal("{}", File.ReadAllText(Path.Combine(layout.ConfigDirectory, "studio", "settings.json")));
        Assert.Equal("database", File.ReadAllText(Path.Combine(layout.DataDirectory, "studio", "studio.db")));
        Assert.Equal("weights", File.ReadAllText(Path.Combine(layout.ModelsDirectory, "owner", "model", "default", "model.slm")));
        Assert.Equal("kernel", File.ReadAllText(Path.Combine(layout.ArtifactsDirectory, "set", "entry.slc")));
        Assert.Equal("log", File.ReadAllText(Path.Combine(layout.LogsDirectory, "studio.log")));
    }

    private static void Write(string path, string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }
}
