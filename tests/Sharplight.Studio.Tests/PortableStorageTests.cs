using Sharplight.Storage;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.ViewModels.Settings;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Choosing the portable location in an AppImage: the choice is recorded next to the AppImage file (not inside the mounted image), the
/// existing files follow, settings changed before the restart are not written to the old location, and the next start finds
/// everything in the sibling <c>data</c> folder.
/// </summary>
public sealed class PortableStorageTests
{
    [Fact]
    public async Task AppImage_Portable_KeepsEverythingInTheSiblingDataFolder_AfterTheRestart()
    {
        using var dir = new TempDir();
        using var host = new TestServices();
        var downloads = dir.File("Applications");
        var mount = dir.File(Path.Combine("tmp", ".mount_SharplXYZ"));
        var home = dir.File("home");
        Directory.CreateDirectory(downloads);
        Directory.CreateDirectory(Path.Combine(mount, "usr", "bin"));
        var variables = new Dictionary<string, string>
        {
            ["APPIMAGE"] = Path.Combine(downloads, "Sharplight_Studio-1.0-x86_64.AppImage"),
            ["APPDIR"] = mount,
        };
        StorageEnvironment Start() => new()
        {
            ApplicationDirectory = Path.Combine(mount, "usr", "bin"),
            HomeDirectory = home,
            Platform = StoragePlatform.Linux,
            GetVariable = name => variables.GetValueOrDefault(name),
        };

        // First start: system folders, with files of every kind.
        var first = Boot(Start());
        var old = first.Layout;
        Assert.Equal(StorageMode.System, old.Mode);
        foreach (var file in new[] { Path.Combine(old.GetDataDirectory("studio"), "studio.db"), Path.Combine(old.LogsDirectory, "studio.log"),
            Path.Combine(old.CacheDirectory, "shaders", "a.spv"), Path.Combine(old.ModelsDirectory, "m", "model.slm") })
        {
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            File.WriteAllText(file, "x");
        }

        first.Settings.Update(settings => settings with { Theme = AppTheme.Light });
        var viewModel = new StorageSettingsViewModel(host.Services with { Settings = first.Settings, StorageLocation = first.Storage });
        viewModel.SelectedMode = StorageModeChoice.Get(StorageMode.Portable);
        Assert.Null(viewModel.Problem);
        viewModel.AskCommand.Execute(null);
        await viewModel.ApplyCommand.ExecuteAsync(null);
        Assert.Equal(1, host.Platform.Restarts);
        Assert.True(viewModel.NeedsRestart, viewModel.Message);

        // The choice sits next to the AppImage; nothing was put into the mounted image.
        Assert.True(File.Exists(Path.Combine(downloads, StorageLayout.PortableMarkerFileName)));
        Assert.False(Directory.Exists(Path.Combine(mount, "usr", "bin", "data")));

        // Settings changed before the restart go to the new location, not back into the old one.
        var oldSettings = File.ReadAllText(first.Paths.SettingsFile);
        first.Settings.Update(settings => settings with { Theme = AppTheme.System });
        Assert.Equal(oldSettings, File.ReadAllText(first.Paths.SettingsFile));

        // The next start (a new process with the same environment) is portable and has all the files.
        var second = Boot(Start());
        var data = Path.Combine(downloads, "data");
        Assert.Equal((StorageMode.Portable, data), (second.Layout.Mode, second.Layout.Root));
        Assert.Equal(AppTheme.System, second.Settings.Current.Theme);
        Assert.All(new[] { second.Paths.SettingsFile, Path.Combine(second.Paths.DataDirectory, "studio.db"), Path.Combine(second.Layout.LogsDirectory, "studio.log"),
            Path.Combine(second.Layout.ShaderSourcesDirectory, "a.spv"), Path.Combine(second.Layout.ModelsDirectory, "m", "model.slm") }, file =>
        {
            Assert.True(File.Exists(file), file);
            Assert.StartsWith(data + Path.DirectorySeparatorChar, file, StringComparison.Ordinal);
        });
        // The old settings file stays: it differs from the copy that was changed before the restart.
        Assert.False(File.Exists(Path.Combine(old.GetDataDirectory("studio"), "studio.db")));
    }

    private static Started Boot(StorageEnvironment environment)
    {
        var layout = StorageLayout.Resolve(environment);
        StorageMigration.CompletePending(layout);
        var paths = StudioPaths.From(layout, null);
        var settings = new JsonSettingsService(paths.SettingsFile, null, layout);
        return new Started(layout, paths, settings, new StorageLocationService(environment, layout, settings));
    }

    private sealed record Started(StorageLayout Layout, StudioPaths Paths, JsonSettingsService Settings, StorageLocationService Storage);
}
