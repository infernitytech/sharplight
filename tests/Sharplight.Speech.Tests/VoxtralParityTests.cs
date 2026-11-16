using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Voxtral Mini 3B (Whisper large-v3 encoder, rows concatenated four at a time and projected into a Mistral decoder's prompt; tekken tokenizer)
/// against transformers (fixtures from <c>tools/make_voxtral_fixtures.py</c>, float32, mistral-common's transcription template) on jfk.wav and a
/// 29 s LibriSpeech recording: the projected audio embeddings and token-exact greedy decoding without a language and with English given,
/// through the speech-recognition task client, on the CPU and on Vulkan.
/// </summary>
public sealed class VoxtralParityTests
{
    private const string Model = "Voxtral-Mini-3B-2507";
    private const string Fixture = "voxtral-mini-3b.json";
    private const int EndOfSequence = 2;

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

    /// <summary>The recordings, devices and request modes ("auto": no language; "en": English given).</summary>
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

    /// <summary>The projected audio embeddings match transformers' (relative to their largest value).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EmbeddingsMatchTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        using var recognizer = new AudioLanguageSpeechRecognizer(model);
        var embeddings = recognizer.Encode(features, 0, Math.Min(features.Frames, 3000));
        double error = SpeechEnvironment.MaxError(embeddings, expected.GetProperty("embeddings"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: embeddings max abs error {error:G3} of max abs {scale:G3}");
        // The checkpoint's bf16 weights: the device's matrix products round differently from the CPU's (7e-3 to 1.2e-2 of the largest value on
        // Vulkan, below 2e-3 on the CPU); the greedy tokens are identical on both.
        double tolerance = device == "cpu" ? 2e-3 : 2e-2;
        Assert.True(error < tolerance * scale, $"embeddings max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens and text.</summary>
    [Theory]
    [MemberData(nameof(DecodeCases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device, string mode)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
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
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device} {mode}: {result.Text}");
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        int end = ids.Length > 0 && ids[^1] == EndOfSequence ? ids.Length - 1 : ids.Length;
        Assert.Equal(ids[..end], result.Segments.SelectMany(s => s.Tokens).ToArray());
        Assert.Equal(expected.GetProperty("text").GetString()!.Trim(), result.Text);
    }

    private static void Skip(string path, string clip, string device)
    {
        if (!SpeechEnvironment.Available(path, device) || !File.Exists(SpeechEnvironment.Audio(clip)))
        {
            Assert.Skip($"{path}, {clip}.wav or the {device} device is missing.");
        }
    }
}
