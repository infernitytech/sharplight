using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Speech;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Transcribe and Speak pages on the real engine runtime and real small models (skipped when they are not on disk): the speech
/// models load through the runtime's residency like chat models, and the pages run them.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class SpeechRealModelTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_whisper = Path.Combine(s_home, "models/whisper.cpp/ggml-tiny.en.bin");
    private static readonly string s_jfk = Path.Combine(s_home, "models/audio/jfk.wav");
    private static readonly string s_outetts = Path.Combine(s_home, "models/tts/outetts-0.2/OuteTTS-0.2-500M-Q8_0.gguf");
    private static readonly string s_vocoder = Path.Combine(s_home, "models/tts/wavtokenizer/WavTokenizer-Large-75-F16.gguf");

    private static readonly string s_kokoro = Path.Combine(s_home, "models/tts/kokoro");

    private static LoadSettings Settings(EngineModelRuntime runtime) => new() { DeviceId = runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };

    [Fact]
    public async Task TranscribePage_RunsARealWhisperModel_LoadedThroughTheRuntime()
    {
        Assert.SkipUnless(File.Exists(s_whisper) && File.Exists(s_jfk), "ggml-tiny.en.bin and jfk.wav not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        var statuses = new List<LoadStatus>();
        var model = await runtime.LoadAsync(new ModelLoadRequest("whisper-tiny", s_whisper, Settings(runtime), [], Kind: ModelKinds.SpeechRecognition),
            new SynchronousProgress<LoadStatus>(statuses.Add), ct);
        Assert.Equal(ModelKinds.SpeechRecognition, model.Kind);
        Assert.Equal(ResidentState.Ready, model.State);
        Assert.True(model.ResidentBytes > 0);
        Assert.Contains(runtime.LoadedModels, loaded => loaded.Id == "whisper-tiny");

        var page = new TranscribePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal("whisper-tiny", page.SelectedModel?.Id);
        Assert.True(await page.SetAudioAsync(s_jfk));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(3), ct);

        Assert.Equal(RunStatus.Completed, item.Status);
        Assert.Contains("ask not what your country can do for you", page.TranscriptText, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(page.Segments);
        Assert.Single(page.Gallery.Items);
        Assert.Contains("speed:", page.ResultSummary, StringComparison.Ordinal);

        // A second run while the first is done reuses the resident model; unloading frees it like any other model.
        await runtime.UnloadAsync("whisper-tiny");
        Assert.Empty(runtime.LoadedModels);
        Assert.False(page.HasModels);
    }

    [Theory]
    [InlineData("moonshine-tiny", "ask not what your country can do for you")]
    [InlineData("parakeet-ctc-0.6b", "ask not what your country can do for you")]
    [InlineData("cohere-transcribe-03-2026", "ask not what your country can do for you")]
    public async Task TranscribePage_RunsTheOtherSpeechFamilies_ThroughTheSameTaskClient(string name, string expected)
    {
        string path = Path.Combine(s_home, "models/hf", name);
        Assert.SkipUnless(Directory.Exists(path) && File.Exists(s_jfk), $"{name} and jfk.wav not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        Assert.Equal(ModelKinds.SpeechRecognition, ModelInspector.Inspect(path).Kind);
        var model = await runtime.LoadAsync(new ModelLoadRequest(name, path, Settings(runtime), [], Kind: ModelKinds.SpeechRecognition), null, ct);
        Assert.Equal(ModelKinds.SpeechRecognition, model.Kind);

        var page = new TranscribePageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(name, page.SelectedModel?.Id);
        Assert.True(await page.SetAudioAsync(s_jfk));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);

        Assert.True(item.Status == RunStatus.Completed, item.StatusText);
        Assert.Contains(expected, page.TranscriptText, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(page.Segments);
        await runtime.UnloadAsync(name);
    }

    [Fact]
    public async Task SpeakPage_RunsARealOuteTtsModel_WithItsVocoder()
    {
        Assert.SkipUnless(File.Exists(s_outetts) && File.Exists(s_vocoder), "OuteTTS and WavTokenizer files not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        Assert.Equal(ModelKinds.SpeechSynthesis, ModelInspector.Inspect(s_outetts).Kind);
        var model = await runtime.LoadAsync(new ModelLoadRequest("outetts", s_outetts, Settings(runtime), [], Kind: ModelKinds.SpeechSynthesis, Auxiliary: [("vocoder", s_vocoder)]), null, ct);
        Assert.Equal(ModelKinds.SpeechSynthesis, model.Kind);

        var page = new SpeakPageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.Panel.Apply(new Dictionary<string, string> { ["temperature"] = "0", ["topk"] = "1", ["topp"] = "1", ["repetition"] = "1" });
        page.InputText = "Hello world.";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);

        Assert.True(item.Status == RunStatus.Completed, item.StatusText);
        var output = Assert.Single(page.Gallery.Items);
        var audio = WavFile.Read(output.Files[0].Path);
        Assert.True(audio.Duration.TotalSeconds > 0.5);
        Assert.Contains(audio.Samples, sample => MathF.Abs(sample) > 0.01f);
        await runtime.UnloadAsync("outetts");
    }

    [Fact]
    public async Task SpeakPage_RunsARealKokoroModel_WithItsVoicesAndLexicon()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(s_kokoro, "kokoro-v1_0.pth")) && File.Exists(Path.Combine(s_kokoro, "lexicon", "us_gold.json")), "Kokoro-82M not found");
        using var fakes = new TestServices();
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var services = fakes.Services with { Runtime = runtime };
        var ct = TestContext.Current.CancellationToken;

        Assert.Equal(ModelKinds.SpeechSynthesis, ModelInspector.Inspect(s_kokoro).Kind);
        var model = await runtime.LoadAsync(new ModelLoadRequest("kokoro", s_kokoro, Settings(runtime), [], Kind: ModelKinds.SpeechSynthesis), null, ct);
        Assert.Equal(ModelKinds.SpeechSynthesis, model.Kind);

        var page = new SpeakPageViewModel(services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        await page.RefreshVoicesAsync(model);
        page.Panel.Apply(new Dictionary<string, string> { ["voice"] = "af_heart" });
        page.InputText = "Hello world. This is Sharplight speaking.";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(5), ct);

        Assert.True(item.Status == RunStatus.Completed, item.StatusText);
        var output = Assert.Single(page.Gallery.Items);
        var audio = WavFile.Read(output.Files[0].Path);
        Assert.True(audio.Duration.TotalSeconds > 2, $"{audio.Duration.TotalSeconds:0.0} s of audio");
        Assert.Contains(audio.Samples, sample => MathF.Abs(sample) > 0.05f);
        await runtime.UnloadAsync("kokoro");
    }

    [Fact]
    public async Task LibraryInference_RecognizesSpeechModelFiles()
    {
        using var dir = new TempDir();
        string whisper = dir.File("ggml-tiny.bin");
        await File.WriteAllBytesAsync(whisper, [0x6c, 0x6d, 0x67, 0x67, 1, 2, 3, 4], TestContext.Current.CancellationToken);
        string other = dir.File("data.bin");
        await File.WriteAllBytesAsync(other, [1, 2, 3, 4, 5], TestContext.Current.CancellationToken);
        string hf = dir.File("whisper-hf");
        Directory.CreateDirectory(hf);
        await File.WriteAllTextAsync(Path.Combine(hf, "config.json"), """{"model_type": "whisper"}""", TestContext.Current.CancellationToken);

        Assert.Equal(ModelKinds.SpeechRecognition, ModelInspector.Inspect(whisper).Kind);
        Assert.True(ModelInspector.Inspect(whisper).KindInferred);
        Assert.Equal(ModelKindCatalog.Other, ModelInspector.Inspect(other).Kind);
        Assert.Equal(ModelKinds.SpeechRecognition, ModelInspector.Inspect(hf).Kind);
        foreach (string type in new[] { "moonshine", "parakeet_ctc", "parakeet_tdt", "canary", "cohere_asr" })
        {
            string folder = dir.File(type);
            Directory.CreateDirectory(folder);
            await File.WriteAllTextAsync(Path.Combine(folder, "config.json"), $$"""{"model_type": "{{type}}"}""", TestContext.Current.CancellationToken);
            Assert.Equal(ModelKinds.SpeechRecognition, ModelInspector.Inspect(folder).Kind);
        }

        Assert.Equal(ModelKinds.SpeechSynthesis, ModelInspector.Inspect(dir.File("OuteTTS-0.2-500M-Q8_0.gguf")).Kind);
    }
}
