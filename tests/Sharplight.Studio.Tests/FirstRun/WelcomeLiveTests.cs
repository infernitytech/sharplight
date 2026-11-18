using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Welcome;

namespace Sharplight.Studio.Tests.FirstRun;

/// <summary>
/// The welcome flow against the published feed (https://download.sharplight.dev/models): detects the machine, recommends from the real catalog,
/// installs the tiny Qwen3-0.6B with the embedded root key, loads it through the engine and opens a chat. Skipped without network access.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class WelcomeLiveTests
{
    private const string Preferences = """
        {
          // Keep the smoke test small whatever the catalog grows to: a chat model under 1 GiB, no speech and no embeddings.
          "roles": [ { "role": "Chat", "kind": "text-generation", "task": "chat", "preselected": true, "maxDownloadMiB": 1024, "prefer": ["qwen3"] } ]
        }
        """;

    private static async Task<bool> ReachableAsync()
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(8) };
            using var response = await http.GetAsync(FeedOptions.DefaultUrl + "/stable/index.json", TestContext.Current.CancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (Exception error) when (error is HttpRequestException or TaskCanceledException)
        {
            return false;
        }
    }

    [Fact]
    public async Task OfficialFeed_TinyQwen_InstallsLoadsAndOpensAChat()
    {
        Assert.SkipUnless(await ReachableAsync(), "download.sharplight.dev is not reachable");
        var ct = TestContext.Current.CancellationToken;
        using var fakes = new TestServices();
        using var feed = new FeedService(fakes.Settings, new FeedOptions
        {
            TrustedRoots = [FeedOptions.OfficialRootKey()],
            StateFile = Path.Combine(fakes.Directory, "feed-state.json"),
            CacheDirectory = Path.Combine(fakes.Directory, "feed-cache"),
        });
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Feed = feed, Runtime = runtime, Library = new ModelLibrary(fakes.Settings, TestHost.Converter) };
        string preferences = services.StorageLocation.Paths.StarterModelsFile;
        Directory.CreateDirectory(Path.GetDirectoryName(preferences)!);
        await File.WriteAllTextAsync(preferences, Preferences, ct);
        var navigator = new FakeNavigator();
        var models = new ModelsPageViewModel(services, navigator);
        var welcome = new WelcomePageViewModel(services, navigator, models);

        await welcome.ActivateAsync();
        await welcome.Loading.WaitAsync(TimeSpan.FromMinutes(2), ct);

        Assert.Null(models.Discover.Feed.Failure);
        var chat = Assert.Single(welcome.Items);
        Assert.StartsWith("qwen3-0.6b", chat.Package.Id, StringComparison.Ordinal);
        await welcome.GetStartedAsync().WaitAsync(TimeSpan.FromMinutes(10), ct);

        Assert.Null(welcome.Error);
        var model = Assert.Single(runtime.LoadedModels);
        Assert.Equal(ModelKinds.TextGeneration, model.Kind);
        Assert.Equal(model.Id, navigator.LastChatModel?.Id);
        Assert.False(string.IsNullOrWhiteSpace(Assert.Single(navigator.SuggestedPrompts)));
        Assert.True(fakes.Settings.Current.FirstRunDone);
        await runtime.UnloadAsync(model.Id);
    }
}
