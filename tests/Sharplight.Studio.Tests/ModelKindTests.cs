using Sharplight.Hub;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Tasks;

namespace Sharplight.Studio.Tests;

/// <summary>The library knows model kinds: badges, the kind filter, the shared model picker and the Tasks page.</summary>
public sealed class ModelKindTests
{
    private static LibraryEntry Entry(string id, string kind, bool inferred = false)
        => new() { Id = id, DisplayName = id, Path = $"/models/{id}.slm", Format = "SLM", Kind = kind, KindInferred = inferred };

    private static LibraryViewModel CreateLibrary(TestServices services, FakeNavigator? navigator = null)
        => new(services.Services, navigator ?? new FakeNavigator());

    [Theory]
    [InlineData("all-MiniLM-L6-v2", """{"model_type":"bert","architectures":["BertModel"]}""", ModelKindCatalog.Embeddings)]
    [InlineData("bge-m3", """{"model_type":"xlm-roberta","architectures":["XLMRobertaModel"]}""", ModelKindCatalog.Embeddings)]
    [InlineData("harrier-embed-0.6b", """{"model_type":"qwen3","architectures":["Qwen3Model"]}""", ModelKindCatalog.Embeddings)]
    [InlineData("ms-marco-MiniLM-L6-v2", """{"model_type":"bert","architectures":["BertForSequenceClassification"],"id2label":{"0":"LABEL_0"}}""", ModelKindCatalog.Reranker)]
    [InlineData("bge-reranker-base", """{"model_type":"xlm-roberta","architectures":["XLMRobertaForSequenceClassification"]}""", ModelKindCatalog.Reranker)]
    [InlineData("Qwen3-Reranker-0.6B", """{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}""", ModelKindCatalog.Reranker)]
    [InlineData("Qwen3-0.6B", """{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}""", ModelKinds.TextGeneration)]
    [InlineData("Qwen3-TTS-12Hz-0.6B-CustomVoice", """{"model_type":"qwen3_tts","architectures":["Qwen3TTSForConditionalGeneration"]}""", ModelKinds.SpeechSynthesis)]
    [InlineData("sentiment", """{"model_type":"bert","architectures":["BertForSequenceClassification"],"id2label":{"0":"NEG","1":"NEU","2":"POS"}}""", ModelKinds.TextGeneration)]
    [InlineData("gpt2", """{"model_type":"gpt2","architectures":["GPT2LMHeadModel"]}""", ModelKinds.TextGeneration)]
    [InlineData("tiny-random-T5Model", """{"model_type":"t5","architectures":["T5Model"]}""", ModelKinds.TextGeneration)]
    public void Inspector_RecognizesEmbeddingAndRerankerFolders_ByTheirConfiguration(string folder, string config, string kind)
    {
        using var directory = new TempDir();
        string path = directory.File(folder);
        Directory.CreateDirectory(path);
        File.WriteAllText(Path.Combine(path, "config.json"), config);

        Assert.Equal(kind, ModelInspector.Inspect(path).Kind);
    }

    [Theory]
    [InlineData("Qwen3-0.6B-Q8_0.gguf", ModelKinds.TextGeneration)]
    [InlineData("Qwen3-Reranker-0.6B-Q8_0.gguf", ModelKindCatalog.Reranker)]
    [InlineData("qwen3-reranker-4b.q4_k_m.gguf", ModelKindCatalog.Reranker)]
    public void Inspector_RecognizesDecoderRerankerGgufFiles_ByTheirName_AndNeverChatModels(string file, string kind)
    {
        using var directory = new TempDir();
        string path = directory.File(file);
        File.WriteAllBytes(path, [1, 2, 3]);

        Assert.Equal(kind, ModelInspector.Inspect(path).Kind);
    }

    [Theory]
    // A Qwen3 chat package carries the embed and classify graphs every decoder gets: still a chat model.
    [InlineData(ModelKinds.TextGeneration, "decode embed classify", "qwen3-0.6b", null, ModelKinds.TextGeneration)]
    [InlineData(ModelKinds.TextGeneration, "decode embed classify", "Qwen3", "Qwen3 0.6B", ModelKinds.TextGeneration)]
    [InlineData(ModelKinds.TextGeneration, "decode embed classify", "qwen3", "Qwen3-Reranker-0.6B", ModelKindCatalog.Reranker)]
    [InlineData(ModelKinds.TextGeneration, "decode embed classify", "Qwen3-Reranker-0.6B", null, ModelKindCatalog.Reranker)]
    [InlineData(ModelKinds.TextGeneration, "decode embed", "Qwen3-Reranker-0.6B", null, ModelKinds.TextGeneration)]
    // Encoders have no decode graph.
    [InlineData(ModelKinds.TextGeneration, "embed classify", "ms-marco", null, ModelKindCatalog.Reranker)]
    [InlineData(ModelKinds.TextGeneration, "embed", "all-minilm", null, ModelKindCatalog.Embeddings)]
    [InlineData(ModelKinds.TextGeneration, "decode", "gpt2", null, ModelKinds.TextGeneration)]
    // A declared kind wins over the graphs.
    [InlineData(ModelKindCatalog.Reranker, "decode embed classify", "any", null, ModelKindCatalog.Reranker)]
    [InlineData(ModelKindCatalog.Embeddings, "embed", "all-minilm", null, ModelKindCatalog.Embeddings)]
    [InlineData(ModelKindCatalog.Embeddings, "embed classify decode", "any", null, ModelKindCatalog.Embeddings)]
    // A cross-encoder of an older feed package declares embeddings but carries the classify graph.
    [InlineData(ModelKindCatalog.Embeddings, "embed classify", "ms-marco", null, ModelKindCatalog.Reranker)]
    [InlineData(ModelKindCatalog.Embeddings, "classify", "bge-reranker", null, ModelKindCatalog.Reranker)]
    [InlineData(ModelKinds.SpeechRecognition, "decode classify", "whisper", null, ModelKinds.SpeechRecognition)]
    public void Inspector_KindOfAPackage_IsItsDeclaredKind_RefinedOnlyByWhatTheGraphsProve(string declared, string graphs, string name, string? generalName, string expected)
        => Assert.Equal(expected, ModelInspector.KindOf(declared, graphs.Split(' ').ToHashSet(), name, generalName));

    [Theory]
    [InlineData("qwen3_asr", ModelKinds.SpeechRecognition)]
    [InlineData("encodec", ModelKinds.AudioCodec)]
    [InlineData("mimi", ModelKinds.AudioCodec)]
    [InlineData("dac", ModelKinds.AudioCodec)]
    [InlineData("musicgen", ModelKinds.MusicGeneration)]
    [InlineData("clap", ModelKinds.AudioEmbedding)]
    public void Inspector_RecognizesAudioCheckpointFolders_ByTheirModelType(string modelType, string kind)
    {
        using var dir = new TempDir();
        string folder = dir.File(modelType);
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "config.json"), $$"""{"model_type": "{{modelType}}", "architectures": ["SomethingModel"]}""");

        Assert.Equal(kind, ModelInspector.Inspect(folder).Kind);
    }

    [Theory]
    [InlineData("WanPipeline", ModelKinds.VideoGeneration)]
    [InlineData("LTXPipeline", ModelKinds.VideoGeneration)]
    [InlineData("LTXConditionPipeline", ModelKinds.VideoGeneration)]
    [InlineData("HunyuanVideoPipeline", ModelKinds.VideoGeneration)]
    [InlineData("StableDiffusionPipeline", ModelKinds.ImageGeneration)]
    public void Inspector_RecognizesVideoPipelines_ByTheirModelIndex(string pipeline, string kind)
    {
        using var dir = new TempDir();
        Directory.CreateDirectory(dir.File("unet"));
        File.WriteAllText(dir.File("model_index.json"), $$"""{"_class_name": "{{pipeline}}", "unet": ["diffusers", "UNet2DConditionModel"]}""");

        Assert.Equal(kind, ModelInspector.Inspect(dir.Path).Kind);
    }

    [Fact]
    public void TheAudioKinds_HaveTaskClients_SoTheyDoNotLoadAsLanguageModels()
    {
        foreach (string kind in new[] { ModelKinds.AudioCodec, ModelKinds.VoiceActivity, ModelKinds.AudioEmbedding, ModelKinds.MusicGeneration, ModelKinds.AudioGeneration, ModelKinds.AudioSeparation })
        {
            Assert.True(Sharplight.Hosting.Loaders.TaskKindClients.Handles(kind), kind);
        }
    }

    [Fact]
    public void Inspector_RecognizesSileroVadFilesAndFolders()
    {
        using var dir = new TempDir();
        string folder = dir.File("silero-vad");
        Directory.CreateDirectory(folder);
        string file = Path.Combine(folder, "silero_vad_16k_op15.onnx");
        File.WriteAllBytes(file, [1, 2, 3]);

        Assert.Equal(ModelKinds.VoiceActivity, ModelInspector.Inspect(file).Kind);
        Assert.Equal(ModelKinds.VoiceActivity, ModelInspector.Inspect(folder).Kind);
    }

    [Fact]
    public void RerankersAreServedOnTheRerankRoutes_AndOpenTheSearchPage()
    {
        Assert.Equal(["/v1/rerank", "/v2/rerank", "/rerank"], Sharplight.Studio.Services.Server.ServedEndpoints.For(ModelKindCatalog.Reranker, openAi: true, anthropic: false));
        var info = ModelKindCatalog.For(ModelKindCatalog.Reranker);
        Assert.True(info.Runnable);
        Assert.Equal("Reranker", info.Title);
    }

    [Fact]
    public void Inspector_ReadsTheKindOfThePackageDescription()
    {
        using var directory = new TempDir();
        string described = directory.File("sd.slm");
        TestPackages.WriteDescribed(described);
        string plain = directory.File("plain.slm");
        TestPackages.WriteSmall(plain);

        var image = ModelInspector.Inspect(described);
        Assert.Equal(ModelKinds.ImageGeneration, image.Kind);
        Assert.False(image.KindInferred);

        // A package without a description resolves to text generation (the description is the default, not a guess from the file).
        var text = ModelInspector.Inspect(plain);
        Assert.Equal(ModelKinds.TextGeneration, text.Kind);
        Assert.False(text.KindInferred);
    }

    [Fact]
    public void Inspector_GuessesTheKindOfFilesWithoutADescription()
    {
        using var directory = new TempDir();
        string gguf = directory.File("model.gguf");
        File.WriteAllBytes(gguf, [1, 2, 3]);
        string onnx = directory.File("model.onnx");
        File.WriteAllBytes(onnx, [1, 2, 3]);

        var guessed = ModelInspector.Inspect(gguf);
        Assert.Equal(ModelKinds.TextGeneration, guessed.Kind);
        Assert.True(guessed.KindInferred);
        var other = ModelInspector.Inspect(onnx);
        Assert.Equal(ModelKindCatalog.Other, other.Kind);
        Assert.True(other.KindInferred);
    }

    [Fact]
    public async Task Library_ShowsKinds_AndFiltersByThem()
    {
        using var services = new TestServices();
        services.Library.Entries.AddRange([Entry("qwen", ModelKinds.TextGeneration), Entry("llama", ModelKinds.TextGeneration), Entry("sd", "time-series"),
            Entry("whisper", ModelKinds.SpeechRecognition, inferred: true)]);
        var library = CreateLibrary(services);
        await library.RefreshAsync();

        Assert.Equal(["All kinds (4)", "Text generation (2)", "Speech to text (1)", "Time series (1)"], library.KindFilters.Select(filter => filter.Title));
        Assert.True(library.HasKindFilter);
        Assert.Equal(4, library.VisibleEntries.Count);
        Assert.Equal("Text generation", library.Entries[0].KindTitle);
        Assert.True(library.Entries[0].IsRunnable);
        Assert.False(library.Entries[2].IsRunnable);
        Assert.Contains("cannot run models of this kind yet", library.Entries[2].KindToolTip, StringComparison.Ordinal);
        Assert.Contains("guessed", library.Entries[3].KindToolTip, StringComparison.Ordinal);

        library.SelectedKindFilter = library.KindFilters.Single(filter => filter.Kind == "time-series");
        Assert.Equal(["sd"], library.VisibleEntries.Select(entry => entry.Id));
        Assert.False(library.IsFilteredEmpty);

        library.SelectKind(ModelKinds.TextGeneration);
        Assert.Equal(["qwen", "llama"], library.VisibleEntries.Select(entry => entry.Id));
        library.SelectedKindFilter = library.KindFilters[0];
        Assert.Equal(4, library.VisibleEntries.Count);
    }

    [Fact]
    public async Task Library_FilterFollowsDeletes_AndFallsBackToAllKinds()
    {
        using var services = new TestServices();
        services.Library.Entries.AddRange([Entry("qwen", ModelKinds.TextGeneration), Entry("sd", "time-series")]);
        var library = CreateLibrary(services);
        await library.RefreshAsync();
        library.SelectKind("time-series");

        var sd = library.Entries.Single(entry => entry.Id == "sd");
        await library.DeleteCommand.ExecuteAsync(sd);
        await library.Confirmation!.ConfirmCommand.ExecuteAsync(null);

        Assert.Single(library.Entries);
        Assert.True(library.SelectedKindFilter!.IsAll);
        Assert.Equal(["qwen"], library.VisibleEntries.Select(entry => entry.Id));
        Assert.False(library.HasKindFilter);
    }

    [Fact]
    public async Task Library_DoesNotLoadAKindTheEngineCannotRun()
    {
        using var services = new TestServices();
        services.Library.Entries.Add(Entry("sd", "time-series"));
        var library = CreateLibrary(services);
        await library.RefreshAsync();

        library.LoadCommand.Execute(library.Entries[0]);

        Assert.Null(library.LoadPanel);
        Assert.Contains("time series model", library.Status, StringComparison.Ordinal);
        Assert.Contains("cannot run", library.Status, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Picker_OffersTheResidentModelsOfItsKind_AndKeepsTheChoice()
    {
        using var services = new TestServices();
        var qwen = new FakeModel("qwen");
        var whisper = new FakeModel("whisper") { Kind = ModelKinds.SpeechRecognition };
        services.Runtime.Add(qwen);
        services.Runtime.Add(whisper);
        var text = new ModelPickerViewModel(services.Services, ModelKinds.TextGeneration);
        var speech = new ModelPickerViewModel(services.Services, ModelKinds.SpeechRecognition);

        Assert.Equal(["qwen"], text.Models.Select(model => model.Id));
        Assert.Equal(["whisper"], speech.Models.Select(model => model.Id));
        Assert.True(text.HasModels);

        var second = new FakeModel("llama");
        services.Runtime.Add(second);
        text.Select(second);
        Assert.Equal("llama", text.SelectedModel?.Id);

        // The other kind's picker is untouched by the choice; unloading the chosen model moves the choice.
        Assert.Equal("whisper", speech.SelectedModel?.Id);
        await services.Runtime.UnloadAsync("llama");
        text.Refresh();
        Assert.Equal("qwen", text.SelectedModel?.Id);
    }

    [Fact]
    public async Task TasksPage_CountsModelsPerKind_AndRoutesTheCards()
    {
        using var services = new TestServices();
        services.Library.Entries.AddRange([Entry("qwen", ModelKinds.TextGeneration), Entry("sd", "time-series"), Entry("sd2", "time-series")]);
        services.Runtime.Add(new FakeModel("qwen"));
        var navigator = new FakeNavigator();
        var page = new TasksPageViewModel(services.Services, navigator);
        await page.ActivateAsync();

        var text = page.Kinds.Single(card => card.Info.Id == ModelKinds.TextGeneration);
        var image = page.Kinds.Single(card => card.Info.Id == "time-series");
        Assert.Equal(1, text.LibraryCount);
        Assert.Equal(1, text.ResidentCount);
        Assert.Equal("1 in library, 1 resident", text.Summary);
        Assert.Equal("2 in library. Not yet supported", image.Summary);
        Assert.True(image.IsPlanned);
        // Runnable kinds come first.
        Assert.True(page.Kinds[0].IsRunnable);
        Assert.DoesNotContain(page.Kinds, card => card.Info.Id == ModelKindCatalog.Other);

        text.OpenCommand.Execute(null);
        image.OpenCommand.Execute(null);
        image.BrowseCommand.Execute(null);
        image.ShowLibraryCommand.Execute(null);
        Assert.Equal(["chat", "library:time-series", "discover:TimeSeries", "library:time-series"], navigator.Visited);
    }

    [Fact]
    public async Task Downloads_ShowTheKindInferredFromThePipelineTag_AndKeepItAcrossRestarts()
    {
        using var services = new TestServices();
        var downloads = new DownloadsViewModel(services.Services);
        var item = downloads.Enqueue(new ModelReference("a/whisper", null), 1000, ModelKinds.SpeechRecognition);
        Assert.Equal("Speech to text", item.KindTitle);
        Assert.True(item.HasKind);
        await services.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);
        for (int i = 0; i < 100 && services.Settings.Current.Downloads.Count == 0; i++)
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        Assert.Equal(ModelKinds.SpeechRecognition, Assert.Single(services.Settings.Current.Downloads).Kind);

        var restored = new DownloadsViewModel(services.Services);
        restored.Restore();
        Assert.Equal(ModelKinds.SpeechRecognition, Assert.Single(restored.Items).Kind);
        services.Hub.Release();
    }
}
