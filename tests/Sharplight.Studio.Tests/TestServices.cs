using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Storage;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>Studio services over fakes, with the real SQLite stores in a temporary directory.</summary>
internal sealed class TestServices : IDisposable
{
    private readonly TempDir _dir = new();

    public TestServices(AppSettings? settings = null, IUiDispatcher? dispatcher = null)
    {
        Settings = new FakeSettings(settings, _dir.File("models"));
        var database = new StudioDatabase(_dir.File("studio.db"));
        Conversations = new SqliteConversationStore(database);
        Stats = new SqliteStatsStore(database);
        Services = new StudioServices(Settings, Runtime, Hub, Library, Conversations, Stats, Server, new DocumentExtractor(), Platform,
            Platform, dispatcher ?? Platform, Platform);
    }

    public FakeSettings Settings { get; }

    public FakeRuntime Runtime { get; } = new();

    public FakeHub Hub { get; } = new();

    public FakeLibrary Library { get; } = new();

    public FakeApiServer Server { get; } = new();

    public FakePlatform Platform { get; } = new();

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
