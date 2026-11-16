namespace Sharplight.Hub.Tests;

public sealed class StoreLocationTests : IDisposable
{
    private static readonly Func<string, bool> s_writable = _ => true;
    private static readonly Func<string, bool> s_readOnly = _ => false;

    private readonly TempDirectory _directory = new();

    public void Dispose() => _directory.Dispose();

    private string App => _directory.Combine("app");

    private string User => _directory.Combine("user");

    private string AppData => Path.Combine(App, ModelStore.ApplicationDirectoryName);

    [Fact]
    public void CommandLineThenEnvironmentThenSettingsThenApplicationFolder()
    {
        var settings = new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = "/settings" };

        Assert.Equal(new ModelStoreResolution("/cli", ModelStoreSource.CommandLine, ModelStoreLocation.Custom),
            ModelStore.Resolve("/cli", "/env", settings, App, User, s_writable));
        Assert.Equal(new ModelStoreResolution("/env", ModelStoreSource.EnvironmentVariable, ModelStoreLocation.Custom),
            ModelStore.Resolve(" ", "/env", settings, App, User, s_writable));
        Assert.Equal(new ModelStoreResolution("/settings", ModelStoreSource.Settings, ModelStoreLocation.Custom),
            ModelStore.Resolve(null, "", settings, App, User, s_writable));
        Assert.Equal(new ModelStoreResolution(AppData, ModelStoreSource.Default, ModelStoreLocation.Application),
            ModelStore.Resolve(null, null, null, App, User, s_writable));
    }

    [Fact]
    public void SettingsChooseTheApplicationUserOrCustomFolder()
    {
        Assert.Equal(new ModelStoreResolution(User, ModelStoreSource.Settings, ModelStoreLocation.User),
            ModelStore.Resolve(null, null, new ModelStoreSettings { Location = ModelStoreLocation.User }, App, User, s_writable));
        Assert.Equal(new ModelStoreResolution(AppData, ModelStoreSource.Settings, ModelStoreLocation.Application),
            ModelStore.Resolve(null, null, new ModelStoreSettings(), App, User, s_writable));

        // A custom location without a path is no choice: the default applies.
        Assert.Equal(new ModelStoreResolution(AppData, ModelStoreSource.Default, ModelStoreLocation.Application),
            ModelStore.Resolve(null, null, new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = " " }, App, User, s_writable));
    }

    [Fact]
    public void ApplicationFolderNotWritable_FallsBackToTheUserFolder_AndSaysSo()
    {
        var byDefault = ModelStore.Resolve(null, null, null, App, User, s_readOnly);
        Assert.Equal(new ModelStoreResolution(User, ModelStoreSource.Default, ModelStoreLocation.User, ApplicationFolderNotWritable: true), byDefault);
        Assert.Equal("Default: user folder, the application folder is not writable", byDefault.Description);

        var bySettings = ModelStore.Resolve(null, null, new ModelStoreSettings(), App, User, s_readOnly);
        Assert.Equal(User, bySettings.Root);
        Assert.True(bySettings.ApplicationFolderNotWritable);
        Assert.StartsWith("Settings:", bySettings.Description, StringComparison.Ordinal);

        // Only the application folder is probed: explicit directories are used as given.
        Assert.Equal("/custom", ModelStore.Resolve(null, null, new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = "/custom" }, App, User, s_readOnly).Root);
        Assert.Equal("/cli", ModelStore.Resolve("/cli", null, null, App, User, s_readOnly).Root);
    }

    [Fact]
    public void Descriptions_NameTheSource()
    {
        Assert.Equal("--models-dir", new ModelStoreResolution("/x", ModelStoreSource.CommandLine, ModelStoreLocation.Custom).Description);
        Assert.Equal(ModelStore.EnvironmentVariable, new ModelStoreResolution("/x", ModelStoreSource.EnvironmentVariable, ModelStoreLocation.Custom).Description);
        Assert.Equal("Default: application folder", new ModelStoreResolution("/x", ModelStoreSource.Default, ModelStoreLocation.Application).Description);
        Assert.Equal("Settings: user folder", new ModelStoreResolution("/x", ModelStoreSource.Settings, ModelStoreLocation.User).Description);
        Assert.Equal("Settings: custom folder", new ModelStoreResolution("/x", ModelStoreSource.Settings, ModelStoreLocation.Custom).Description);
    }

    [Fact]
    public void WriteProbe_CreatesTheFolder_AndRejectsReadOnlyFolders()
    {
        var data = _directory.Combine("writable", ModelStore.ApplicationDirectoryName);
        Assert.True(ModelStore.CanWrite(data));
        Assert.True(Directory.Exists(data));
        Assert.Empty(Directory.EnumerateFileSystemEntries(data));   // the probe file is gone

        if (OperatingSystem.IsWindows())
        {
            return;
        }

        var readOnly = _directory.Combine("read-only");
        Directory.CreateDirectory(readOnly);
        File.SetUnixFileMode(readOnly, UnixFileMode.UserRead | UnixFileMode.UserExecute);
        try
        {
            bool isPrivileged = Environment.UserName == "root";
            Assert.Equal(isPrivileged, ModelStore.CanWrite(Path.Combine(readOnly, ModelStore.ApplicationDirectoryName)));
            Assert.Equal(isPrivileged, ModelStore.CanWrite(readOnly));
        }
        finally
        {
            File.SetUnixFileMode(readOnly, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        }
    }

    [Fact]
    public void SystemInstallLocations_AreNotUsedForModels()
    {
        Assert.False(ModelStore.IsInstallLocation(_directory.Combine("app", "data")));
        if (OperatingSystem.IsLinux())
        {
            Assert.True(ModelStore.IsInstallLocation("/usr/lib/sharplight/data"));
            Assert.False(ModelStore.CanWrite("/usr/lib/sharplight/data"));
        }
    }

    [Fact]
    public void AppImageDirectory_IsNotUsedForModels()
    {
        var mounted = _directory.Combine("appimage-mount");
        Assert.True(ModelStore.IsInstallLocation(Path.Combine(mounted, "usr", "bin", "data"), mounted, App));
        Assert.False(ModelStore.IsInstallLocation(Path.Combine(User, "models"), mounted, App));
    }

    [Fact]
    public void VelopackApplicationDirectory_IsNotUsedForModels()
    {
        var root = _directory.Combine("velopack");
        var current = Path.Combine(root, "current");
        Directory.CreateDirectory(current);
        File.WriteAllBytes(Path.Combine(root, "Update.exe"), []);
        Assert.True(ModelStore.IsInstallLocation(Path.Combine(current, "data"), null, current));
        Assert.False(ModelStore.IsInstallLocation(Path.Combine(App, "data"), null, App));
    }

    [Fact]
    public void ApplicationRoot_IsTheDataFolderNextToTheExecutable()
    {
        Assert.Equal(Path.Combine(AppContext.BaseDirectory, "data"), ModelStore.ApplicationRoot);
    }

    [Fact]
    public void SharedSettingsFile_RoundTrips_AndIsOptional()
    {
        var path = _directory.Combine("config", "sharplight", ModelStoreSettings.FileName);
        Assert.Null(ModelStoreSettings.Load(path));

        new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = "/models" }.Save(path);
        Assert.Equal(new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = "/models" }, ModelStoreSettings.Load(path));
        Assert.Contains("\"location\": \"Custom\"", File.ReadAllText(path), StringComparison.Ordinal);

        File.WriteAllText(path, "{ not json");
        Assert.Null(ModelStoreSettings.Load(path));
        Assert.EndsWith(Path.Combine("sharplight", ModelStoreSettings.FileName), ModelStoreSettings.DefaultFile, StringComparison.Ordinal);
    }

    [Fact]
    public void PreviousStore_IsOfferedOnlyWhileTheStoreInEffectIsEmpty()
    {
        var current = _directory.Combine("current");
        var previous = _directory.Combine("previous");
        var empty = _directory.Combine("empty");

        Assert.Null(ModelStore.FindPreviousStore(current, [empty, previous]));
        Add(new ModelStore(previous), "owner/model", "Q8_0");
        Assert.Equal(new ModelStore(previous).Root, ModelStore.FindPreviousStore(current, [empty, previous])?.Root);
        Assert.Null(ModelStore.FindPreviousStore(previous, [previous]));   // the same store is not "previous"

        Add(new ModelStore(current), "owner/other", "Q4_K_M");
        Assert.Null(ModelStore.FindPreviousStore(current, [previous]));
    }

    [Fact]
    public void Transfer_CopiesOrMovesCompleteModels_AndSkipsWhatTheDestinationHas()
    {
        var source = new ModelStore(_directory.Combine("source"));
        var destination = new ModelStore(_directory.Combine("destination"));
        Add(source, "owner/a", "Q8_0");
        Add(source, "owner/b", "");
        Directory.CreateDirectory(source.GetModelDirectory("owner/partial", "Q4_0"));   // an interrupted pull: no manifest

        Assert.Equal(2, ModelStoreTransfer.Transfer(source, destination, move: false));
        Assert.Equal(["owner/a:Q8_0", "owner/b"], destination.List().Select(model => model.Name));
        Assert.Equal(2, source.List().Count);
        Assert.Equal("weights", File.ReadAllText(Path.Combine(destination.GetModelDirectory("owner/a", "Q8_0"), ModelStore.PackageFileName)));

        var lines = new List<string>();
        Assert.Equal(0, ModelStoreTransfer.Transfer(source, destination, move: true, lines.Add));
        Assert.All(lines, line => Assert.Contains("skipped", line, StringComparison.Ordinal));

        var moved = new ModelStore(_directory.Combine("moved"));
        Assert.Equal(2, ModelStoreTransfer.Transfer(source, moved, move: true));
        Assert.Empty(source.List());
        Assert.False(Directory.Exists(Path.Combine(source.Root, "owner", "a")));   // emptied parents are pruned
        Assert.Equal(2, moved.List().Count);
    }

    private static void Add(ModelStore store, string repository, string tag)
    {
        var directory = store.GetModelDirectory(repository, tag);
        Directory.CreateDirectory(Path.Combine(directory, "source"));
        File.WriteAllText(Path.Combine(directory, ModelStore.PackageFileName), "weights");
        File.WriteAllText(Path.Combine(directory, "source", "model.gguf"), "source");
        ModelStore.WriteManifest(new ModelManifest
        {
            Repository = repository,
            Tag = tag,
            Alias = ModelManifest.AliasFor(repository),
            Commit = "c0ffee",
            Format = ModelFormat.Gguf,
            PackageBytes = 7,
            PulledAt = new DateTimeOffset(2026, 11, 14, 12, 0, 0, TimeSpan.Zero),
            Directory = directory,
        });
    }
}
