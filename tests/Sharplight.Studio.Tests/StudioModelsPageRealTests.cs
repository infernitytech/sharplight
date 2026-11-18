using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Models page end to end on real packages: installing from the published feed, the library showing the right kind for every kind of
/// package, loading and unloading through the engine runtime, and the residency list (skipped without network access or a model).
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class StudioModelsPageRealTests
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
    public async Task FeedPackages_InstallIntoTheLibrary_ShowTheirKinds_LoadAndUnload_AndFillTheResidencyList()
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
        var page = new ModelsPageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        var catalog = page.Discover.Feed;
        await catalog.LoadAsync();
        Assert.Null(catalog.Failure);

        // Install one package of every kind the feed offers that is small: a chat model, an embedder, a reranker, speech and voice activity.
        (string Prefix, string Kind)[] wanted =
        [
            ("qwen3-0.6b", ModelKinds.TextGeneration),
            ("all-minilm-l6-v2", ModelKindCatalog.Embeddings),
            ("ms-marco-minilm-l6-v2", ModelKindCatalog.Reranker),
            ("whisper-tiny", ModelKinds.SpeechRecognition),
            ("silero-vad", ModelKinds.VoiceActivity),
        ];
        foreach (var (prefix, _) in wanted)
        {
            var package = catalog.AllPackages.FirstOrDefault(candidate => candidate.Id.StartsWith(prefix, StringComparison.Ordinal));
            Assert.SkipWhen(package is null, $"{prefix} is not in the feed");
            var install = catalog.Install(package!);
            await install.Completion.WaitAsync(TimeSpan.FromMinutes(10), ct);
            Assert.True(install.IsDone, install.Error);
        }

        // The library lists each with the kind it was published for: the chat package is a chat model, the cross-encoder a reranker.
        await page.Library.RefreshAsync();
        Assert.Equal(wanted.Length, page.Library.Entries.Count);
        foreach (var (prefix, kind) in wanted)
        {
            var entry = page.Library.Entries.Single(candidate => candidate.Entry.Manifest?.FeedPackage?.StartsWith(prefix, StringComparison.Ordinal) == true);
            Assert.Equal(kind, entry.Entry.Kind);
            Assert.Equal(ModelKindCatalog.For(kind).Title, entry.KindTitle);
            Assert.True(entry.IsFromFeed);
            Assert.True(entry.IsPackage);
        }

        Assert.Equal(wanted.Length + 1, page.Library.KindFilters.Count);
        Assert.NotEqual(ModelKindCatalog.For(ModelKindCatalog.Reranker).Title, page.Library.Entries.Single(entry => entry.Id.StartsWith("qwen3-0.6b", StringComparison.Ordinal)).KindTitle);
        page.Library.SelectedKindFilter = page.Library.KindFilters.Single(filter => filter.Kind == ModelKindCatalog.Reranker);
        Assert.Equal(["ms-marco-minilm-l6-v2.f32-safetensors"], page.Library.VisibleEntries.Select(entry => entry.Entry.Manifest!.FeedPackage));
        page.Library.SelectedKindFilter = page.Library.KindFilters[0];

        // Load from the library's panel; the residency tab lists the model, then unloading empties it.
        var chat = page.Library.Entries.Single(entry => entry.Entry.Kind == ModelKinds.TextGeneration);
        page.Library.LoadCommand.Execute(chat);
        var panel = page.Library.LoadPanel!;
        panel.ContextLength = 1024;
        await panel.LoadCommand.ExecuteAsync(null);
        Assert.True(string.IsNullOrEmpty(panel.Status) || !panel.Status.Contains("fail", StringComparison.OrdinalIgnoreCase), panel.Status);
        var whisper = page.Library.Entries.Single(entry => entry.Entry.Kind == ModelKinds.SpeechRecognition);
        page.Library.LoadCommand.Execute(whisper);
        await page.Library.LoadPanel!.LoadCommand.ExecuteAsync(null);

        page.SelectedTab = ModelsPageViewModel.ResidentTab;
        page.Resident.Refresh();
        Assert.Equal(2, page.Resident.Models.Count);
        Assert.Contains(page.Resident.Models, model => model.Model.Kind == ModelKinds.TextGeneration && model.IsReady);
        Assert.Contains(page.Resident.Models, model => model.Model.Kind == ModelKinds.SpeechRecognition && model.IsReady);
        Assert.All(page.Resident.Models, model => Assert.NotEmpty(model.MemoryText));
        Assert.NotEmpty(page.Resident.Devices);
        Assert.True(page.Resident.Devices.Any(device => device.ModelBytes > 0));
        Assert.All(runtime.LoadedModels, model => Assert.True(model.ResidentBytes > 0));
        Assert.Contains("MiB kept free", page.Resident.PolicyText, StringComparison.Ordinal);

        var chatModel = page.Resident.Models.Single(model => model.Model.Kind == ModelKinds.TextGeneration);
        await page.Resident.UnloadCommand.ExecuteAsync(chatModel);
        page.Resident.Refresh();
        Assert.Single(page.Resident.Models);
        Assert.Equal(ModelKinds.SpeechRecognition, page.Resident.Models[0].Model.Kind);
        await runtime.UnloadAsync(page.Resident.Models[0].Id);
        page.Resident.Refresh();
        Assert.True(page.Resident.IsEmpty);
    }

    [Theory]
    [InlineData("gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf", ModelKinds.TextGeneration)]
    [InlineData("hf/Qwen3-0.6B", ModelKinds.TextGeneration)]
    [InlineData("hf/Qwen2.5-0.5B-Instruct", ModelKinds.TextGeneration)]
    [InlineData("hf/gemma-3-1b-it", ModelKinds.TextGeneration)]
    [InlineData("hf/all-MiniLM-L6-v2", ModelKindCatalog.Embeddings)]
    [InlineData("hf/Qwen3-Embedding-0.6B", ModelKindCatalog.Embeddings)]
    [InlineData("hf/ms-marco-MiniLM-L6-v2", ModelKindCatalog.Reranker)]
    [InlineData("hf/bge-reranker-v2-m3", ModelKindCatalog.Reranker)]
    [InlineData("hf/Qwen3-Reranker-0.6B", ModelKindCatalog.Reranker)]
    [InlineData("hf/Qwen3-ASR-0.6B", ModelKinds.SpeechRecognition)]
    [InlineData("whisper.cpp/ggml-tiny.en.bin", ModelKinds.SpeechRecognition)]
    [InlineData("tts/kokoro", ModelKinds.SpeechSynthesis)]
    [InlineData("vision/vit-base-patch16-224", ModelKinds.ImageClassification)]
    [InlineData("vision/detr-resnet-50", ModelKinds.ObjectDetection)]
    [InlineData("vision/depth-anything-v2-small", ModelKinds.DepthEstimation)]
    [InlineData("vision/trocr-base-printed", ModelKinds.Ocr)]
    [InlineData("audio-models/silero-vad/silero_vad_16k_op15.onnx", ModelKinds.VoiceActivity)]
    [InlineData("audio-models/mimi", ModelKinds.AudioCodec)]
    [InlineData("audio-models/encodec_24khz", ModelKinds.AudioCodec)]
    [InlineData("audio-models/musicgen-small", ModelKinds.MusicGeneration)]
    [InlineData("audio-models/clap-htsat-unfused", ModelKinds.AudioEmbedding)]
    [InlineData("sd/tiny-random-sd/diffusers", ModelKinds.ImageGeneration)]
    [InlineData("video/tiny-random-wan/t2v/diffusers", ModelKinds.VideoGeneration)]
    public void Inspector_GivesEveryRealModelItsOwnKind(string relative, string kind)
    {
        string path = RealStudioSession.Require(relative);
        Assert.Equal(kind, ModelInspector.Inspect(path).Kind);
    }
}
