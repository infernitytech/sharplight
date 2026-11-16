using Sharplight.Hub;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Settings;

namespace Sharplight.Studio.Tests;

public sealed class StoreSettingsViewModelTests
{
    [Fact]
    public void Settings_DefaultToTheStorageLayout_AndShowWhy()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        Assert.Equal(ModelStoreLocation.Application, page.StoreLocation.Value);
        Assert.False(page.IsCustomStore);
        Assert.Equal(services.Settings.ApplicationRoot, page.StoreRoot);
        Assert.Equal("Default: system storage", page.StoreRootSource);
        Assert.Equal(3, page.StoreLocations.Count);
    }

    [Fact]
    public void ChoosingALocation_SavesIt_AndUpdatesTheStoreInUse()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        page.StoreLocation = StoreLocationChoice.Get(ModelStoreLocation.User);
        Assert.Equal(ModelStoreLocation.User, services.Settings.Current.ModelsLocation);
        Assert.Equal(services.Settings.UserRoot, page.StoreRoot);
        Assert.Equal("Settings: system folder", page.StoreRootSource);

        page.StoreLocation = StoreLocationChoice.Get(ModelStoreLocation.Custom);
        Assert.True(page.IsCustomStore);
        page.ModelsDirectory = " /data/models ";
        Assert.Equal("/data/models", services.Settings.Current.ModelsDirectory);
        Assert.Equal("/data/models", page.StoreRoot);

        page.ResetModelsDirectoryCommand.Execute(null);
        Assert.Equal(ModelStoreLocation.Application, services.Settings.Current.ModelsLocation);
        Assert.Equal(services.Settings.ApplicationRoot, page.StoreRoot);
    }

    [Fact]
    public void UnrelatedChanges_DoNotTurnTheDefaultIntoAChoice()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        page.EnableThinking = !page.EnableThinking;
        Assert.True(services.Settings.Updates > 0);
        Assert.Null(services.Settings.Current.ModelsLocation);
    }

    [Fact]
    public async Task Browse_PicksACustomFolder()
    {
        using var services = new TestServices();
        services.Platform.Folder = "/picked";
        var page = new SettingsPageViewModel(services.Services);

        await page.BrowseModelsDirectoryCommand.ExecuteAsync(null);

        Assert.Equal(ModelStoreLocation.Custom, page.StoreLocation.Value);
        Assert.Equal(("/picked", ModelStoreLocation.Custom), (services.Settings.Current.ModelsDirectory, services.Settings.Current.ModelsLocation));
        Assert.Equal("/picked", page.StoreRoot);
    }

    [Fact]
    public void OpenFolder_CreatesAndOpensTheStore()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        page.OpenModelsDirectoryCommand.Execute(null);

        Assert.True(Directory.Exists(services.Settings.ApplicationRoot));
        Assert.Equal([services.Settings.ApplicationRoot], services.Platform.Opened);
    }

    [Fact]
    public async Task SettingsPage_ShowsAStoreChangedElsewhere()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        services.Settings.Update(current => current with { ModelsLocation = ModelStoreLocation.User });
        await page.ActivateAsync();

        Assert.Equal(ModelStoreLocation.User, page.StoreLocation.Value);
        Assert.Equal(services.Settings.UserRoot, page.StoreRoot);
    }

    [Fact]
    public void Migration_IsNotOffered_WithoutAPreviousStore_OrOnceAnswered()
    {
        using var services = new TestServices();
        var migration = new StoreMigrationViewModel(services.Services, () => { });
        migration.Check();
        Assert.False(migration.IsOpen);

        services.Settings.PreviousStore = services.Directory;
        services.Settings.Update(current => current with { StoreMigrationAnswered = true });
        migration.Check();
        Assert.False(migration.IsOpen);
    }

    [Fact]
    public void Migration_KeepUsingTheUserFolder()
    {
        using var services = new TestServices();
        var previous = new ModelStore(Path.Combine(services.Directory, "previous"));
        AddModel(previous, "owner/model", "Q8_0");
        services.Settings.PreviousStore = previous.Root;
        int changes = 0;
        var migration = new StoreMigrationViewModel(services.Services, () => changes++);

        migration.Check();
        Assert.True(migration.IsOpen);
        Assert.Equal(1, migration.ModelCount);
        Assert.Contains(previous.Root, migration.Message, StringComparison.Ordinal);

        migration.KeepPreviousCommand.Execute(null);

        Assert.False(migration.IsOpen);
        Assert.True(services.Settings.Current.StoreMigrationAnswered);
        Assert.Equal(previous.Root, services.Settings.StoreRoot);   // not the system folder here, so kept as a custom folder
        Assert.Equal(1, changes);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Migration_MovesOrCopiesTheModels(bool move)
    {
        using var services = new TestServices();
        var previous = new ModelStore(Path.Combine(services.Directory, "previous"));
        AddModel(previous, "owner/model", "Q8_0");
        services.Settings.PreviousStore = previous.Root;
        int changes = 0;
        var migration = new StoreMigrationViewModel(services.Services, () => changes++);
        migration.Check();

        await (move ? migration.MoveCommand : migration.CopyCommand).ExecuteAsync(null);

        Assert.True(migration.IsDone);
        Assert.Equal("Close", migration.DismissText);
        Assert.Equal("owner/model:Q8_0", Assert.Single(new ModelStore(services.Settings.StoreRoot).List()).Name);
        Assert.Equal(move ? 0 : 1, previous.List().Count);
        Assert.True(services.Settings.Current.StoreMigrationAnswered);
        Assert.Null(services.Settings.Current.ModelsLocation);   // the store stays where it is
        Assert.Equal(1, changes);

        migration.DismissCommand.Execute(null);
        Assert.False(migration.IsOpen);
    }

    [Fact]
    public void Migration_Dismissed_IsNotOfferedAgain()
    {
        using var services = new TestServices();
        services.Settings.PreviousStore = services.Directory;
        var migration = new StoreMigrationViewModel(services.Services, () => { });
        migration.Check();
        Assert.True(migration.IsOpen);

        migration.DismissCommand.Execute(null);

        Assert.False(migration.IsOpen);
        Assert.True(services.Settings.Current.StoreMigrationAnswered);
        migration.Check();
        Assert.False(migration.IsOpen);
    }

    private static void AddModel(ModelStore store, string repository, string tag)
    {
        var directory = store.GetModelDirectory(repository, tag);
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, ModelStore.PackageFileName), "weights");
        File.WriteAllText(Path.Combine(directory, ModelStore.ManifestFileName),
            $$"""{ "repository": "{{repository}}", "tag": "{{tag}}", "alias": "model", "commit": "c0ffee", "format": "Gguf" }""");
    }
}
