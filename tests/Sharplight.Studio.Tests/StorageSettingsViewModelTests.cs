using Sharplight.Hub;
using Sharplight.Storage;
using Sharplight.Studio.ViewModels.Settings;

namespace Sharplight.Studio.Tests;

/// <summary>The storage card of the settings page, over the real storage service in a temporary home and package folder.</summary>
public sealed class StorageSettingsViewModelTests
{
    [Fact]
    public void ShowsTheLocationInUse_AndTheFoldersPerKind()
    {
        using var services = new TestServices();
        var storage = new StorageSettingsViewModel(services.Services);

        Assert.Equal(StorageMode.System, storage.SelectedMode.Value);
        Assert.Equal("System folders", storage.CurrentDescription);
        Assert.False(storage.HasChange);
        Assert.False(storage.IsLocked);
        Assert.Null(storage.Notice);
        Assert.False(storage.ApplyCommand.CanExecute(null));
        Assert.Equal(3, storage.Modes.Count);
        Assert.Equal(
            ["Studio settings", "History, presets and outputs", "Models", "Compiled kernels", "Other caches", "Shared tool settings", "Logs"],
            storage.Paths.Select(row => row.Name));
        Assert.Equal(Path.Combine(services.Directory, "home", ".config", "sharplight", "studio"), storage.Paths[0].Current);
        Assert.All(storage.Paths, row => Assert.False(row.Changes));
    }

    [Fact]
    public async Task Portable_MovesTheFiles_RecordsTheMarker_AndAsksForARestart()
    {
        using var services = new TestServices();
        var current = services.Services.StorageLocation.Current;
        var settingsFile = Path.Combine(current.GetConfigDirectory("studio"), "settings.json");
        Directory.CreateDirectory(Path.GetDirectoryName(settingsFile)!);
        File.WriteAllText(settingsFile, "{}");
        var storage = new StorageSettingsViewModel(services.Services);

        storage.SelectedMode = StorageModeChoice.Get(StorageMode.Portable);

        var portableRoot = Path.Combine(services.Directory, "app", "data");
        Assert.True(storage.HasChange);
        Assert.Null(storage.Problem);
        Assert.Equal(Path.Combine(portableRoot, "config", "studio"), storage.Paths[0].Next);
        Assert.Equal(Path.Combine(portableRoot, "studio"), storage.Paths[1].Next);
        Assert.True(storage.CanKeepModels);
        Assert.True(storage.AskCommand.CanExecute(null));
        storage.AskCommand.Execute(null);
        Assert.True(storage.IsConfirming);
        Assert.Equal(0, services.Platform.Restarts);   // nothing happens before the confirmation

        await storage.ApplyCommand.ExecuteAsync(null);

        Assert.True(storage.NeedsRestart);
        Assert.False(storage.IsConfirming);
        Assert.Contains("Restarting Studio", storage.Message, StringComparison.Ordinal);
        Assert.Equal(1, services.Platform.Restarts);   // restarted by itself after the files were moved
        Assert.Equal("{}", File.ReadAllText(Path.Combine(portableRoot, "config", "studio", "settings.json")));
        Assert.True(File.Exists(Path.Combine(services.Directory, "app", StorageLayout.PortableMarkerFileName)));
        Assert.False(storage.HasChange);   // the selection is what the next start uses
        Assert.Equal(StorageMode.Portable, services.Services.StorageLocation.Pending?.Mode);

        storage.RestartCommand.Execute(null);   // still available when the automatic restart did not happen
        Assert.Equal(2, services.Platform.Restarts);
    }

    [Fact]
    public async Task Custom_NeedsAFolder_PickedWithBrowse()
    {
        using var services = new TestServices();
        var storage = new StorageSettingsViewModel(services.Services);

        storage.SelectedMode = StorageModeChoice.Get(StorageMode.Custom);
        Assert.True(storage.IsCustom);
        Assert.Equal("Choose a folder for the custom location.", storage.Problem);
        Assert.False(storage.ApplyCommand.CanExecute(null));

        services.Platform.Folder = Path.Combine(services.Directory, "big-disk");
        await storage.BrowseCommand.ExecuteAsync(null);

        Assert.Equal(services.Platform.Folder, storage.CustomRoot);
        Assert.Null(storage.Problem);
        Assert.Equal(Path.Combine(services.Platform.Folder, "models"), storage.Paths[2].Next);
    }

    [Fact]
    public void Custom_RefusesAFolderWithOtherFiles()
    {
        using var services = new TestServices();
        var other = Path.Combine(services.Directory, "documents");
        Directory.CreateDirectory(other);
        File.WriteAllText(Path.Combine(other, "letter.txt"), "");
        var storage = new StorageSettingsViewModel(services.Services);

        storage.SelectedMode = StorageModeChoice.Get(StorageMode.Custom);
        storage.CustomRoot = other;

        Assert.Contains("already holds other files", storage.Problem, StringComparison.Ordinal);

        Directory.CreateDirectory(Path.Combine(other, "models"));   // an existing Sharplight folder is fine
        storage.CustomRoot = other + Path.DirectorySeparatorChar;
        Assert.Null(storage.Problem);
    }

    [Fact]
    public async Task KeepModels_LeavesTheStore_AndKeepsUsingIt()
    {
        using var services = new TestServices();
        var oldStore = services.Settings.StoreRoot;
        var storage = new StorageSettingsViewModel(services.Services)
        {
            SelectedMode = StorageModeChoice.Get(StorageMode.Custom),
        };
        storage.CustomRoot = Path.Combine(services.Directory, "custom");
        storage.SelectedTransfer = StorageTransferChoice.Get(StorageTransferKind.Copy);
        storage.KeepModels = true;

        Assert.Equal("", storage.Paths[2].Next);   // the models row does not change
        await storage.ApplyCommand.ExecuteAsync(null);

        Assert.True(storage.NeedsRestart, storage.Message);
        Assert.Equal((ModelStoreLocation.Custom, oldStore), (services.Settings.Current.ModelsLocation, services.Settings.Current.ModelsDirectory));
    }

    [Fact]
    public void HomeVariable_LocksTheLocation()
    {
        using var services = new TestServices();
        var variables = new Dictionary<string, string> { [StorageLayout.HomeVariable] = Path.Combine(services.Directory, "pinned") };
        var pinned = services.Services with
        {
            StorageLocation = TestServices.CreateStorage(services.Settings, Path.Combine(services.Directory, "app"), Path.Combine(services.Directory, "home"), variables),
        };
        var storage = new StorageSettingsViewModel(pinned);

        Assert.True(storage.IsLocked);
        Assert.Equal(StorageMode.Custom, storage.SelectedMode.Value);
        storage.SelectedMode = StorageModeChoice.Get(StorageMode.System);
        Assert.Contains(StorageLayout.HomeVariable, storage.Problem, StringComparison.Ordinal);
        Assert.False(storage.ApplyCommand.CanExecute(null));
    }

    [Fact]
    public void Portable_IsRefused_InsideAnInstallation()
    {
        using var services = new TestServices();
        var application = Path.Combine(services.Directory, "image", "usr", "bin");
        var variables = new Dictionary<string, string> { ["APPDIR"] = Path.Combine(services.Directory, "image") };
        var installed = services.Services with
        {
            StorageLocation = TestServices.CreateStorage(services.Settings, application, Path.Combine(services.Directory, "home"), variables),
        };
        var storage = new StorageSettingsViewModel(installed);

        storage.SelectedMode = StorageModeChoice.Get(StorageMode.Portable);

        Assert.False(string.IsNullOrEmpty(storage.Problem));
    }

    [Fact]
    public void Portable_WithAppDirButNoAppImage_ExplainsWhy()
    {
        using var services = new TestServices();
        var application = Path.Combine(services.Directory, "image", "usr", "bin");
        var variables = new Dictionary<string, string> { ["APPDIR"] = Path.Combine(services.Directory, "image") };
        var installed = services.Services with
        {
            StorageLocation = TestServices.CreateStorage(services.Settings, application, Path.Combine(services.Directory, "home"), variables),
        };
        var storage = new StorageSettingsViewModel(installed);

        storage.SelectedMode = StorageModeChoice.Get(StorageMode.Portable);

        Assert.Contains("$APPIMAGE is not set", storage.Problem, StringComparison.Ordinal);
    }

    [Fact]
    public void SettingsPage_HasTheStorageCard()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        Assert.Equal(StorageMode.System, page.Storage.SelectedMode.Value);
    }
}
