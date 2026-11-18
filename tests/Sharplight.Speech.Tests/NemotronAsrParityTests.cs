using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Nemotron 3.5 ASR streaming (cache-aware FastConformer with causal subsampling and convolutions, chunk-limited attention with three rows of
/// lookahead, a language prompt projector and an RNN-T head) against transformers (fixtures from <c>tools/make_nemotron_fixtures.py</c>, float32,
/// offline mode, language "auto") on jfk.wav and a 29 s LibriSpeech recording: the front end, the encoder output and token-exact greedy decoding
/// through the speech-recognition task client, on the CPU and on Vulkan.
/// </summary>
public sealed class NemotronAsrParityTests
{
    private const string ModelName = "nemotron-3.5-asr-streaming-0.6b";
    private const string Fixture = "nemotron-3.5-asr-streaming-0.6b.json";
    private const int Blank = 13087;

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

    /// <summary>The log-mel features match transformers' <c>NemotronAsrStreamingFeatureExtractor</c>.</summary>
    [Fact]
    public void FeaturesMatchTransformers()
    {
        string clip = SpeechEnvironment.JfkName;
        using var model = Load(clip, "cpu");
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip).GetProperty("features");
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        var timeMajor = TimeMajor(features, expected.GetProperty("rows").GetInt32());
        double error = SpeechEnvironment.MaxError(timeMajor, expected, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip}: log-mel max abs error {error:G3} (max abs {scale:G3})");
        Assert.True(error < 1e-3 * Math.Max(scale, 1), $"log-mel max abs error {error:G3} (scale {scale:G3})");
    }

    /// <summary>The encoder rows match transformers' (relative to their largest value), before and after the language prompt projector.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EncoderMatchesTransformers(string clip, string device)
    {
        using var model = Load(clip, device);
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var entry = fixture.RootElement.GetProperty(clip);
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        int frames = entry.GetProperty("features").GetProperty("rows").GetInt32();
        using var encoder = new SpeechEncoder(model, WellKnown.HiddenOutput);
        encoder.SelectLanguage(null);
        encoder.Encode(features.Values.AsSpan(), features.Frames);
        Assert.Equal(frames, features.Frames);
        var rows = entry.GetProperty("encoder").GetProperty("rows").GetInt32();
        Assert.Equal(rows, encoder.Rows);
        double error = SpeechEnvironment.MaxError(encoder.Output.ToArray(), entry.GetProperty("encoder"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: encoder max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-3 * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy RNN-T decoding through the task client gives transformers' tokens and text.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(ModelName);
        Skip(path, clip, device);
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), null, TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: {result.Text}");
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).Where(id => id != Blank)];
        Assert.Equal(ids, result.Segments.SelectMany(s => s.Tokens).ToArray());
        Assert.Equal(expected.GetProperty("text").GetString()!.Trim(), result.Text.Trim());
    }

    private static float[] TimeMajor(AudioFeatures features, int frames)
    {
        var result = new float[frames * features.Channels];
        for (int c = 0; c < features.Channels; c++)
        {
            for (int t = 0; t < frames; t++)
            {
                result[(t * features.Channels) + c] = features.Values[(c * features.Frames) + t];
            }
        }

        return result;
    }

    private static TaskModel Load(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(ModelName);
        Skip(path, clip, device);
        return TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
    }

    private static void Skip(string path, string clip, string device)
    {
        if (!SpeechEnvironment.Available(path, device) || !File.Exists(SpeechEnvironment.Audio(clip)))
        {
            Assert.Skip($"{path}, {clip}.wav or the {device} device is missing.");
        }
    }
}
