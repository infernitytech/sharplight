using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Qwen3-ASR 1.7B (an audio language model: a chunked Whisper-style audio tower whose rows replace the audio placeholders of a Qwen3 decoder's
/// prompt) against transformers (fixtures from <c>tools/make_qwen3_asr_fixtures.py</c>, the <c>-hf</c> checkpoint in float32) on jfk.wav and a
/// 29 s LibriSpeech recording: the projected audio embeddings, and token-exact greedy decoding with the language detected by the model and with
/// English forced, through the speech-recognition task client, on the CPU and on Vulkan. The original release (<c>thinker.*</c>) is the
/// model under test; the <c>-hf</c> conversion and the transcribe.cpp GGUF must give the same tokens.
/// </summary>
public sealed class Qwen3AsrParityTests
{
    private const string Fixture = "qwen3-asr-1.7b.json";

    /// <summary>The checkpoints and the files they load from.</summary>
    public static string ModelPath(string layout) => layout switch
    {
        "hf" => SpeechEnvironment.HuggingFace("Qwen3-ASR-1.7B-hf"),
        "gguf-f16" => SpeechEnvironment.SpeechGguf("Qwen3-ASR-1.7B-F16.gguf"),
        _ => SpeechEnvironment.HuggingFace("Qwen3-ASR-1.7B"),
    };

    /// <summary>The recordings and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            cases.Add(clip, "cpu");
            cases.Add(clip, "vulkan");
        }

        return cases;
    }

    /// <summary>The recordings, devices and request modes ("auto": the model names the language; "en": English forced).</summary>
    public static TheoryData<string, string, string> DecodeCases()
    {
        var cases = new TheoryData<string, string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            foreach (string device in new[] { "cpu", "vulkan" })
            {
                cases.Add(clip, device, "auto");
                cases.Add(clip, device, "en");
            }
        }

        return cases;
    }

    /// <summary>Other layouts of the same model on jfk.wav (Vulkan).</summary>
    public static TheoryData<string> Layouts() => ["hf", "gguf-f16"];

    /// <summary>
    /// The log-mel features of the valid frames match transformers' feature extractor (jfk.wav: within 6e-4 of values up to 1.4, the frames
    /// near the recording's peak; the greedy tokens are identical all the same).
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void FeaturesMatchTransformers(string clip, string device)
    {
        string path = ModelPath("original");
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        int frames = expected.GetProperty("frames").GetInt32();
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        var window = new float[features.Channels * frames];
        features.CopyWindow(0, frames, window);
        double error = SpeechEnvironment.MaxError(window, expected.GetProperty("features"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip}: features max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-3 * Math.Max(scale, 1), $"features max abs error {error:G3}");
    }

    /// <summary>The projected audio embeddings match transformers' (relative to their largest value).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EmbeddingsMatchTransformers(string clip, string device)
    {
        string path = ModelPath("original");
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        using var recognizer = new AudioLanguageSpeechRecognizer(model);
        int frames = expected.GetProperty("frames").GetInt32();
        var embeddings = recognizer.Encode(features, 0, frames);
        double error = SpeechEnvironment.MaxError(embeddings, expected.GetProperty("embeddings"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: embeddings max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-2 * scale, $"embeddings max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens, text and language.</summary>
    [Theory]
    [MemberData(nameof(DecodeCases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device, string mode)
    {
        await CheckTokensAsync(ModelPath("original"), clip, device, mode);
    }

    /// <summary>The transformers conversion and the transcribe.cpp F16 GGUF give the same tokens as the original release.</summary>
    [Theory]
    [MemberData(nameof(Layouts))]
    public async Task OtherLayoutsGiveTheSameTokens(string layout)
    {
        await CheckTokensAsync(ModelPath(layout), SpeechEnvironment.JfkName, "vulkan", "auto");
    }

    private static async Task CheckTokensAsync(string path, string clip, string device, string mode)
    {
        Skip(path, clip, device);
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip).GetProperty(mode);
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
        var options = new SpeechRecognitionOptions { Language = mode == "auto" ? null : mode };
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), options, TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device} {mode}: [{result.Language}] {result.Text}");
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        const int EndOfTurn = 151645;
        const int TranscriptMarker = 151704;
        int end = ids.Length > 0 && ids[^1] == EndOfTurn ? ids.Length - 1 : ids.Length;
        int begin = mode == "auto" ? Array.IndexOf(ids, TranscriptMarker) + 1 : 0;
        Assert.Equal(ids[begin..end], result.Segments.SelectMany(s => s.Tokens).ToArray());
        string text = expected.GetProperty("text").GetString()!;
        Assert.Equal(text[(text.IndexOf("<asr_text>", StringComparison.Ordinal) is var at and >= 0 ? at + "<asr_text>".Length : 0)..].Trim(), result.Text);
        Assert.Equal("en", result.Language);
    }

    private static void Skip(string path, string clip, string device)
    {
        if (!SpeechEnvironment.Available(path, device) || !File.Exists(SpeechEnvironment.Audio(clip)))
        {
            Assert.Skip($"{path}, {clip}.wav or the {device} device is missing.");
        }
    }
}
