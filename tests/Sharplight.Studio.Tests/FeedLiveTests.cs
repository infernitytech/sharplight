using Sharplight.Hub;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The published Sharplight feed (https://download.sharplight.dev/models): lists the packages, installs two of them with the embedded root key, and loads one
/// through the engine. Skipped without network access.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class FeedLiveTests
{
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
    public async Task OfficialFeed_ListsAndInstallsSignedPackages_ThatLoadWithoutConversion()
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
        var services = fakes.Services with { Feed = feed, Runtime = runtime };
        var catalog = new FeedCatalogViewModel(services, () => new ModelStore(fakes.Settings.StoreRoot).List());

        await catalog.LoadAsync();

        Assert.Null(catalog.Failure);
        Assert.False(catalog.IsOffline);
        Assert.True(catalog.IsListingVerified);
        var vad = catalog.AllPackages.Single(package => package.Id.StartsWith("silero-vad", StringComparison.Ordinal));
        var whisper = catalog.AllPackages.Single(package => package.Id.StartsWith("whisper-tiny", StringComparison.Ordinal));
        foreach (var package in new[] { vad, whisper })
        {
            var install = catalog.Install(package);
            await install.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);
            Assert.True(install.IsDone, install.Error);
        }

        var store = new ModelStore(fakes.Settings.StoreRoot);
        var stored = store.List();
        Assert.Contains(stored, model => model.FeedPackage == vad.Id && model.FeedVersion == vad.Version);
        var whisperModel = stored.Single(model => model.FeedPackage == whisper.Id);
        Assert.Equal(whisper.Summary.Size, new FileInfo(whisperModel.PackagePath).Length);
        Assert.Equal(ModelFormat.Slm, whisperModel.Format);

        // Ready to load: the library lists it as a package of the speech kind, and the engine loads it as is.
        var entry = (await new ModelLibrary(fakes.Settings, TestHost.Converter).ListAsync(ct)).Single(candidate => candidate.Manifest?.FeedPackage == whisper.Id);
        Assert.Equal(ModelKinds.SpeechRecognition, entry.Kind);
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var device = runtime.Devices[0];
        var load = new LoadSettings { DeviceId = device.Id, ContextLength = 1024, BatchTokens = 128, Slots = 1 };
        var model = await runtime.LoadAsync(new ModelLoadRequest("whisper-feed", entry.Path, load, [], Kind: ModelKinds.SpeechRecognition), null, ct);
        Assert.Equal("whisper-feed", model.Id);
        await runtime.UnloadAsync("whisper-feed");
        Assert.Empty(runtime.LoadedModels);

        // A second install of the same version changes nothing.
        var again = catalog.Install(whisper);
        await again.Completion;
        Assert.Equal("Already installed", again.Status);
    }
}
