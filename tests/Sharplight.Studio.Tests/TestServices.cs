using Sharplight.Audio;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
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
        Services = new StudioServices(Settings, Runtime, Hub, Library, Conversations, Stats, Server, new DocumentExtractor(), Platform,
            Platform, dispatcher ?? Platform, Platform, Updates, Precompiler, new FilePageStorage(_dir.File("pages")), Audio, Audio, decoder ?? new WavAudioDecoder(), AudioDevices);
    }

    public FakeSettings Settings { get; }

    public FakeRuntime Runtime { get; } = new();

    public FakeHub Hub { get; } = new();

    public FakeLibrary Library { get; } = new();

    public FakeApiServer Server { get; } = new();

    public FakePlatform Platform { get; } = new();

    public FakeUpdates Updates { get; } = new();

    public FakePrecompiler Precompiler { get; } = new();

    public FakeAudio Audio { get; } = new();

    public FakeAudioDevices AudioDevices { get; } = new();

    public SqliteConversationStore Conversations { get; }

    public SqliteStatsStore Stats { get; }

    public StudioServices Services { get; }

    public string Directory => _dir.Path;

    public void Dispose()
    {
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        _dir.Dispose();
    }
}
