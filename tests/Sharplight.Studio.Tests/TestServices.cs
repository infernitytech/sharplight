using Sharplight.Audio;
using Sharplight.Storage;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Maintenance;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Storage;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>Studio services over fakes, with the real SQLite stores in a temporary directory.</summary>
internal sealed class TestServices : IDisposable
{
    private readonly TempDir _dir = new();

    public TestServices(AppSettings? settings = null, IUiDispatcher? dispatcher = null, IAudioDecoder? decoder = null)
    {
        Settings = new FakeSettings(settings, _dir.File("models"));
        var database = new StudioDatabase(_dir.File("studio.db"));
        Conversations = new SqliteConversationStore(database);
        Stats = new SqliteStatsStore(database);
        Storage = CreateStorage(Settings, _dir.File("app"), _dir.File("home"));
        Usage = new JsonModelUsageStore(_dir.File("model-usage.json"));
        Licenses = new LicenseGate(Settings);
        Memory = new Services.Fit.DeviceMemoryWatch(Runtime, Settings);
        Maintenance = new ModelMaintenanceService(Settings, Storage, Runtime, Library, Usage, Precompiler);
        Services = new StudioServices(Settings, Runtime, Hub, Library, Conversations, Stats, Server, new DocumentExtractor(), Platform,
            Platform, dispatcher ?? Platform, Platform, Updates, Precompiler, new FilePageStorage(_dir.File("pages")), Audio, Audio, decoder ?? new WavAudioDecoder(), AudioDevices,
            Storage, Feed, Usage, Maintenance, Licenses, Memory);
    }

    /// <summary>A custom storage layout in <paramref name="root"/>, for services that need one.</summary>
    public static StorageLayout CreateLayout(string root)
    {
        var environment = new StorageEnvironment { ApplicationDirectory = root, HomeDirectory = root, Platform = StoragePlatform.Linux };
        return StorageLayout.Create(StorageMode.Custom, System.IO.Path.Combine(root, "storage"), environment);
    }

    /// <summary>A storage location service for a package in <paramref name="application"/> and a user home in <paramref name="home"/>.</summary>
    public static StorageLocationService CreateStorage(FakeSettings settings, string application, string home,
        IReadOnlyDictionary<string, string>? variables = null)
    {
        System.IO.Directory.CreateDirectory(application);
        var environment = new StorageEnvironment
        {
            ApplicationDirectory = application,
            HomeDirectory = home,
            Platform = StoragePlatform.Linux,
            GetVariable = name => variables?.GetValueOrDefault(name),
        };
        return new StorageLocationService(environment, StorageLayout.Resolve(environment), settings);
    }

    public LicenseGate Licenses { get; }

    /// <summary>The shared memory figures over the fake runtime.</summary>
    public Services.Fit.DeviceMemoryWatch Memory { get; }

    public StorageLocationService Storage { get; }

    public JsonModelUsageStore Usage { get; }

    public ModelMaintenanceService Maintenance { get; }

    public FakeSettings Settings { get; }

    public FakeRuntime Runtime { get; } = new();

    public FakeHub Hub { get; } = new();

    public FakeLibrary Library { get; } = new();

    public FakeApiServer Server { get; } = new();

    public FakePlatform Platform { get; } = new();

    public FakeUpdates Updates { get; } = new();

    public FakeFeed Feed { get; } = new();

    public FakePrecompiler Precompiler { get; } = new();

    public FakeAudio Audio { get; } = new();

    public FakeAudioDevices AudioDevices { get; } = new();

    public SqliteConversationStore Conversations { get; }

    public SqliteStatsStore Stats { get; }

    public StudioServices Services { get; }

    public string Directory => _dir.Path;

    public void Dispose()
    {
        Memory.Dispose();
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        _dir.Dispose();
    }
}
