using Sharplight.Storage;

namespace Sharplight.Hub.Tests;

public sealed class StoreLocationTests : IDisposable
{
    private readonly TempDirectory _directory = new();

    public void Dispose() => _directory.Dispose();

    private string App => _directory.Combine("app");

    private string User => _directory.Combine("user");

    private StorageEnvironment Environment => new() { ApplicationDirectory = App, HomeDirectory = _directory.Combine("home"), Platform = StoragePlatform.Linux };

    private StorageLayout Portable => StorageLayout.Create(StorageMode.Portable, null, Environment);

    [Fact]
    public void CommandLineThenEnvironmentThenSettingsThenTheStorageLayout()
    {
        var settings = new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = "/settings" };

        Assert.Equal(new ModelStoreResolution("/cli", ModelStoreSource.CommandLine, ModelStoreLocation.Custom),
            ModelStore.Resolve("/cli", "/env", settings, Portable, User));
        Assert.Equal(new ModelStoreResolution("/env", ModelStoreSource.EnvironmentVariable, ModelStoreLocation.Custom),
            ModelStore.Resolve(" ", "/env", settings, Portable, User));
        Assert.Equal(new ModelStoreResolution("/settings", ModelStoreSource.Settings, ModelStoreLocation.Custom),
            ModelStore.Resolve(null, "", settings, Portable, User));
        Assert.Equal(new ModelStoreResolution(Path.Combine(App, "data", "models"), ModelStoreSource.Default, ModelStoreLocation.Application, StorageMode.Portable),
            ModelStore.Resolve(null, null, null, Portable, User));
    }

    [Fact]
    public void SettingsChooseTheLayoutTheSystemOrACustomFolder()
    {
        Assert.Equal(new ModelStoreResolution(User, ModelStoreSource.Settings, ModelStoreLocation.User),
            ModelStore.Resolve(null, null, new ModelStoreSettings { Location = ModelStoreLocation.User }, Portable, User));
        Assert.Equal(Portable.ModelsDirectory, ModelStore.Resolve(null, null, new ModelStoreSettings(), Portable, User).Root);

        // A custom location without a path is no choice: the default applies.
        Assert.Equal(new ModelStoreResolution(Portable.ModelsDirectory, ModelStoreSource.Default, ModelStoreLocation.Application, StorageMode.Portable),
            ModelStore.Resolve(null, null, new ModelStoreSettings { Location = ModelStoreLocation.Custom, CustomPath = " " }, Portable, User));
    }

    [Fact]
    public void ThePortableFallback_IsReported()
    {
        File.WriteAllText(Path.Combine(_directory.Path, StorageLayout.PortableMarkerFileName), "");
        var environment = new StorageEnvironment
        {
            ApplicationDirectory = _directory.Path,
            HomeDirectory = _directory.Combine("home"),
            Platform = StoragePlatform.Linux,
            CanWrite = _ => false,
        };
        var layout = StorageLayout.Resolve(environment);

        var resolution = ModelStore.Resolve(null, null, null, layout, User);

        Assert.Equal(layout.ModelsDirectory, resolution.Root);
        Assert.StartsWith("Default: system storage; the portable folder", resolution.Description, StringComparison.Ordinal);
    }

    [Fact]
    public void Descriptions_NameTheSource()
    {
        Assert.Equal("--models-dir", new ModelStoreResolution("/x", ModelStoreSource.CommandLine, ModelStoreLocation.Custom).Description);
        Assert.Equal(ModelStore.EnvironmentVariable, new ModelStoreResolution("/x", ModelStoreSource.EnvironmentVariable, ModelStoreLocation.Custom).Description);
        Assert.Equal("Default: portable storage", new ModelStoreResolution("/x", ModelStoreSource.Default, ModelStoreLocation.Application, StorageMode.Portable).Description);
        Assert.Equal("Default: system storage", new ModelStoreResolution("/x", ModelStoreSource.Default, ModelStoreLocation.Application).Description);
        Assert.Equal("Settings: system folder", new ModelStoreResolution("/x", ModelStoreSource.Settings, ModelStoreLocation.User).Description);
        Assert.Equal("Settings: custom folder", new ModelStoreResolution("/x", ModelStoreSource.Settings, ModelStoreLocation.Custom).Description);
    }

    [Fact]
    public void PreviousStores_IncludeTheFoldersOfEarlierVersions()
    {
        var layout = StorageLayout.Create(StorageMode.System, null, Environment);
        var legacy = new ModelStore(Path.Combine(App, "data"));   // earlier versions kept models directly in data/ next to the executable
        Add(legacy, "owner/model", "Q8_0");

        var candidates = ModelStore.PreviousStoreCandidates(layout, User);
        var previous = ModelStore.FindPreviousStore(ModelStore.Resolve(null, null, null, layout, User).Root, candidates);

        Assert.Equal([User, Path.Combine(App, "data", "models"), Path.Combine(App, "data"), Path.Combine(App, "models")], candidates);
        Assert.Equal(legacy.Root, previous?.Root);
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
        Assert.Equal(Path.Combine(Portable.ConfigDirectory, ModelStoreSettings.FileName), ModelStoreSettings.FileFor(Portable));
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
