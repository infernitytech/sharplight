using Sharplight.Storage;
using Xunit;

namespace Sharplight.Core.Tests.Storage;

/// <summary>
/// Resolution of the storage layout: the three modes, the platform conventions of the system folders, where the package of each
/// distribution form is (AppImage, Velopack, app bundle, plain folder), the bootstrap precedence and the fallback of an unwritable
/// portable or custom folder.
/// </summary>
public sealed class StorageLayoutTests : IDisposable
{
    private readonly StorageTestEnvironment _test = new();

    public void Dispose() => _test.Dispose();

    [Fact]
    public void NothingConfigured_UsesTheXdgFolders()
    {
        var layout = _test.Resolve();

        Assert.Equal((StorageMode.System, StorageModeSource.Default), (layout.Mode, layout.Source));
        Assert.Null(layout.Root);
        Assert.Equal(Path.Combine(_test.Home, ".config", "sharplight"), layout.ConfigDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".local", "share", "sharplight"), layout.DataDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".local", "share", "sharplight", "models"), layout.ModelsDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".cache", "sharplight"), layout.CacheDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".cache", "sharplight", "artifacts"), layout.ArtifactsDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".cache", "sharplight", "shaders"), layout.ShaderSourcesDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".local", "state", "sharplight", "logs"), layout.LogsDirectory);
        Assert.Equal(Path.Combine(layout.ConfigDirectory, StoragePointer.FileName), layout.PointerFile);
        Assert.Equal("System folders", layout.Description);
    }

    [Fact]
    public void XdgVariables_MoveTheSystemFolders_RelativeOnesAreIgnored()
    {
        _test.Variables["XDG_CONFIG_HOME"] = "/xdg/config";
        _test.Variables["XDG_DATA_HOME"] = "/xdg/data";
        _test.Variables["XDG_CACHE_HOME"] = "relative/cache";
        _test.Variables["XDG_STATE_HOME"] = "/xdg/state";

        var layout = _test.Resolve();

        Assert.Equal("/xdg/config/sharplight", layout.ConfigDirectory);
        Assert.Equal("/xdg/data/sharplight", layout.DataDirectory);
        Assert.Equal(Path.Combine(_test.Home, ".cache", "sharplight"), layout.CacheDirectory);
        Assert.Equal("/xdg/state/sharplight/logs", layout.LogsDirectory);
    }

    [Fact]
    public void Windows_UsesRoamingForConfig_AndLocalForTheRest()
    {
        _test.Platform = StoragePlatform.Windows;
        _test.Variables["APPDATA"] = "/users/u/AppData/Roaming";
        _test.Variables["LOCALAPPDATA"] = "/users/u/AppData/Local";

        var layout = _test.Resolve();

        Assert.Equal(Path.Combine("/users/u/AppData/Roaming", "sharplight"), layout.ConfigDirectory);
        Assert.Equal(Path.Combine("/users/u/AppData/Local", "sharplight"), layout.DataDirectory);
        Assert.Equal(Path.Combine("/users/u/AppData/Local", "sharplight", "cache"), layout.CacheDirectory);
        Assert.Equal(Path.Combine("/users/u/AppData/Local", "sharplight", "logs"), layout.LogsDirectory);
    }

    [Fact]
    public void MacOS_UsesTheLibraryFolders()
    {
        _test.Platform = StoragePlatform.MacOS;

        var layout = _test.Resolve();

        var support = Path.Combine(_test.Home, "Library", "Application Support", "sharplight");
        Assert.Equal(Path.Combine(support, "config"), layout.ConfigDirectory);
        Assert.Equal(support, layout.DataDirectory);
        Assert.Equal(Path.Combine(_test.Home, "Library", "Caches", "sharplight"), layout.CacheDirectory);
        Assert.Equal(Path.Combine(_test.Home, "Library", "Logs", "sharplight"), layout.LogsDirectory);
    }

    [Fact]
    public void PortableMarker_KeepsEverythingInTheDataFolderNextToThePlainFolder()
    {
        File.WriteAllText(Path.Combine(_test.Application, StorageLayout.PortableMarkerFileName), "");

        var layout = _test.Resolve();

        var root = Path.Combine(_test.Application, "data");
        Assert.Equal((StorageMode.Portable, StorageModeSource.PortableMarker, root), (layout.Mode, layout.Source, layout.Root));
        Assert.Equal(Path.Combine(root, "config"), layout.ConfigDirectory);
        Assert.Equal(root, layout.DataDirectory);
        Assert.Equal(Path.Combine(root, "models"), layout.ModelsDirectory);
        Assert.Equal(Path.Combine(root, "cache", "artifacts"), layout.ArtifactsDirectory);
        Assert.Equal(Path.Combine(root, "logs"), layout.LogsDirectory);
        Assert.Equal(Path.Combine(root, "studio"), layout.GetDataDirectory("studio"));
        Assert.Equal(Path.Combine(root, "config", "studio"), layout.GetConfigDirectory("studio"));
    }

    [Fact]
    public void AppImage_ThePackageIsTheImageFile_NeverTheMountedImage()
    {
        var downloads = _test.Combine("Downloads");
        var mount = _test.Combine("tmp", ".mount_Sharpl");
        Directory.CreateDirectory(downloads);
        _test.Application = Path.Combine(mount, "usr", "bin");
        Directory.CreateDirectory(_test.Application);
        _test.Variables["APPIMAGE"] = Path.Combine(downloads, "Sharplight-Studio.AppImage");
        _test.Variables["APPDIR"] = mount;
        File.WriteAllText(Path.Combine(downloads, StorageLayout.PortableMarkerFileName), "{}");

        var layout = _test.Resolve();

        Assert.Equal(downloads, layout.PackageDirectory);
        Assert.Equal((StorageMode.Portable, Path.Combine(downloads, "data")), (layout.Mode, layout.Root));
        Assert.False(Directory.Exists(Path.Combine(mount, "usr", "bin", "data")));
    }

    [Fact]
    public void AppImage_AMarkerInsideTheImage_DoesNotMakeItPortable()
    {
        var mount = _test.Combine("mount");
        _test.Application = Path.Combine(mount, "usr", "bin");
        Directory.CreateDirectory(_test.Application);
        _test.Variables["APPIMAGE"] = _test.Combine("Sharplight-Studio.AppImage");
        _test.Variables["APPDIR"] = mount;
        File.WriteAllText(Path.Combine(_test.Application, StorageLayout.PortableMarkerFileName), "{}");

        Assert.Equal(StorageMode.System, _test.Resolve().Mode);
    }

    [Fact]
    public void Velopack_ThePackageIsTheInstallRoot_NeverTheCurrentFolder()
    {
        var installRoot = _test.Combine("LocalAppData", "Sharplight");
        _test.Application = Path.Combine(installRoot, "current");
        Directory.CreateDirectory(_test.Application);
        File.WriteAllBytes(Path.Combine(installRoot, "Update.exe"), []);
        File.WriteAllText(Path.Combine(installRoot, StorageLayout.PortableMarkerFileName), "");

        var layout = _test.Resolve();

        Assert.Equal(installRoot, layout.PackageDirectory);
        Assert.Equal((StorageMode.Portable, Path.Combine(installRoot, "data")), (layout.Mode, layout.Root));
    }

    [Fact]
    public void Velopack_ACurrentFolderWithoutUpdater_IsAPlainFolder()
    {
        _test.Application = _test.Combine("build", "current");
        Directory.CreateDirectory(_test.Application);

        Assert.Equal(_test.Application, _test.Resolve().PackageDirectory);
    }

    [Fact]
    public void AppBundle_ThePortableFolderGoesNextToTheBundle_NotInsideIt()
    {
        var applications = _test.Combine("Applications");
        _test.Application = Path.Combine(applications, "Sharplight Studio.app", "Contents", "MacOS");
        Directory.CreateDirectory(_test.Application);
        File.WriteAllText(Path.Combine(applications, StorageLayout.PortableMarkerFileName), "");

        var layout = _test.Resolve();

        Assert.Equal(applications, layout.PackageDirectory);
        Assert.Equal(Path.Combine(applications, "data"), layout.Root);
        Assert.True(PackageLocation.IsInstallLocation(Path.Combine(_test.Application, "data"), _test.Environment));
        Assert.False(PackageLocation.IsInstallLocation(Path.Combine(applications, "data"), _test.Environment));
    }

    [Fact]
    public void PortableFolderNotWritable_FallsBackToTheSystemFolders_AndSaysSo()
    {
        File.WriteAllText(Path.Combine(_test.Application, StorageLayout.PortableMarkerFileName), "");
        _test.ReadOnly.Add(Path.Combine(_test.Application, "data"));

        var layout = _test.Resolve();

        Assert.Equal(StorageMode.System, layout.Mode);
        Assert.NotNull(layout.Notice);
        Assert.Contains("not writable", layout.Notice, StringComparison.Ordinal);
        Assert.Contains("not writable", layout.Description, StringComparison.Ordinal);
        Assert.False(StorageLayout.CanUsePortable(_test.Environment));
    }

    [Fact]
    public void SystemWideInstall_IsNeverPortable()
    {
        Assert.True(PackageLocation.IsInstallLocation("/usr/lib/sharplight/data", _test.Environment));
        Assert.False(PackageLocation.IsInstallLocation(_test.Combine("app", "data"), _test.Environment));

        _test.Platform = StoragePlatform.Windows;
        _test.Variables["ProgramFiles"] = _test.Combine("Program Files");
        Assert.True(PackageLocation.IsInstallLocation(_test.Combine("Program Files", "Sharplight", "data"), _test.Environment));
    }

    [Fact]
    public void PointerFile_SelectsTheCustomFolder()
    {
        var custom = _test.Combine("big-disk", "sharplight");
        var system = _test.Resolve();
        new StoragePointer { Mode = StorageMode.Custom, Root = custom }.Save(system.PointerFile);

        var layout = _test.Resolve();

        Assert.Equal((StorageMode.Custom, StorageModeSource.PointerFile, custom), (layout.Mode, layout.Source, layout.Root));
        Assert.Equal(Path.Combine(custom, "models"), layout.ModelsDirectory);
        Assert.Equal(system.PointerFile, layout.PointerFile);   // always in the system configuration folder
    }

    [Fact]
    public void CustomFolderNotAvailable_FallsBackToTheSystemFolders_AndSaysSo()
    {
        var custom = _test.Combine("unplugged");
        _test.ReadOnly.Add(custom);
        new StoragePointer { Mode = StorageMode.Custom, Root = custom }.Save(_test.Resolve().PointerFile);

        var layout = _test.Resolve();

        Assert.Equal(StorageMode.System, layout.Mode);
        Assert.Contains("not available", layout.Notice, StringComparison.Ordinal);
    }

    [Fact]
    public void Precedence_HomeVariable_ThenPortableMarker_ThenPointer_ThenSystem()
    {
        var custom = _test.Combine("custom");
        new StoragePointer { Mode = StorageMode.Custom, Root = custom }.Save(_test.Resolve().PointerFile);
        File.WriteAllText(Path.Combine(_test.Application, StorageLayout.PortableMarkerFileName), "");
        _test.Variables[StorageLayout.HomeVariable] = _test.Combine("home-variable");

        Assert.Equal((StorageMode.Custom, StorageModeSource.EnvironmentVariable), (_test.Resolve().Mode, _test.Resolve().Source));
        Assert.EndsWith($"(set by {StorageLayout.HomeVariable})", _test.Resolve().Description, StringComparison.Ordinal);

        _test.Variables.Remove(StorageLayout.HomeVariable);
        Assert.Equal(StorageMode.Portable, _test.Resolve().Mode);

        File.Delete(Path.Combine(_test.Application, StorageLayout.PortableMarkerFileName));
        Assert.Equal((StorageMode.Custom, custom), (_test.Resolve().Mode, _test.Resolve().Root));

        File.Delete(_test.Resolve().PointerFile);
        Assert.Equal(StorageMode.System, _test.Resolve().Mode);
    }

    [Fact]
    public void BrokenPointerFile_IsIgnored()
    {
        var pointer = _test.Resolve().PointerFile;
        Directory.CreateDirectory(Path.GetDirectoryName(pointer)!);
        File.WriteAllText(pointer, "{ not json");

        Assert.Equal(StorageMode.System, _test.Resolve().Mode);
        Assert.Null(StoragePointer.Load(pointer));
    }

    [Fact]
    public void Selection_WritesTheMarkerOrThePointer_ForTheNextResolve()
    {
        var current = _test.Resolve();

        StorageSelection.Apply(current, StorageMode.Portable, null);
        Assert.True(File.Exists(current.PortableMarkerFile));
        Assert.Equal(StorageMode.Portable, _test.Resolve().Mode);

        var custom = _test.Combine("custom");
        StorageSelection.Apply(current, StorageMode.Custom, custom);
        Assert.False(File.Exists(current.PortableMarkerFile));
        Assert.Equal(new StoragePointer { Mode = StorageMode.Custom, Root = custom }, StoragePointer.Load(current.PointerFile));
        Assert.Contains("\"mode\": \"Custom\"", File.ReadAllText(current.PointerFile), StringComparison.Ordinal);
        Assert.Equal((StorageMode.Custom, custom), (_test.Resolve().Mode, _test.Resolve().Root));

        StorageSelection.Apply(current, StorageMode.System, null);
        Assert.False(File.Exists(current.PointerFile));
        Assert.Equal(StorageMode.System, _test.Resolve().Mode);
        Assert.Throws<ArgumentException>(() => StorageSelection.Apply(current, StorageMode.Custom, " "));
    }

    [Fact]
    public void Create_PreviewsAModeWithoutTouchingTheDisk()
    {
        var portable = StorageLayout.Create(StorageMode.Portable, null, _test.Environment);

        Assert.Equal((StorageMode.Portable, StorageModeSource.Explicit), (portable.Mode, portable.Source));
        Assert.Equal(Path.Combine(_test.Application, "data"), portable.Root);
        Assert.False(Directory.Exists(portable.Root));
        Assert.True(portable.HasSameFolders(StorageLayout.Create(StorageMode.Portable, null, _test.Environment)));
        Assert.False(portable.HasSameFolders(_test.Resolve()));
        Assert.True(portable.Contains(Path.Combine(portable.Root!, "models", "a")));
        Assert.Throws<ArgumentException>(() => StorageLayout.Create(StorageMode.Custom, null, _test.Environment));
    }
}
