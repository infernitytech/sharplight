using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Parakeet (FastConformer) checkpoints of Hugging Face transformers against transformers itself (fixtures from
/// <c>tools/make_parakeet_fixtures.py</c>) on jfk.wav and a 29 s LibriSpeech recording, each one variable-length window: NeMo's log-mel
/// features, the encoder output, and token-exact greedy decoding (CTC for parakeet-ctc-0.6b, TDT for parakeet-tdt-0.6b-v3) through the
/// speech-recognition task client of the service container, on the CPU and on Vulkan.
/// </summary>
public sealed class ParakeetParityTests
{
    /// <summary>The checkpoints, recordings and devices.</summary>
    public static TheoryData<string, string, string> Cases()
    {
        var cases = new TheoryData<string, string, string>();
        foreach (string model in new[] { "parakeet-ctc-0.6b", "parakeet-tdt-0.6b-v3" })
        {
            foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
            {
                cases.Add(model, clip, "cpu");
                cases.Add(model, clip, "vulkan");
            }
        }

        return cases;
    }

    /// <summary>The log-mel features match transformers' <c>ParakeetFeatureExtractor</c>.</summary>
    [Theory]
    [InlineData("parakeet-ctc-0.6b", SpeechEnvironment.JfkName)]
    [InlineData("parakeet-tdt-0.6b-v3", SpeechEnvironment.LibriSpeechName)]
    public void FeaturesMatchTransformers(string name, string clip)
    {
        using var model = Load(name, clip, "cpu");
        using var fixture = SpeechEnvironment.Fixture($"{name}.json");
        var expected = fixture.RootElement.GetProperty(clip).GetProperty("features");
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        var timeMajor = TimeMajor(features, expected.GetProperty("rows").GetInt32());
        double error = SpeechEnvironment.MaxError(timeMajor, expected, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name} {clip}: log-mel max abs error {error:G3} (max abs {scale:G3})");
        Assert.True(error < 1e-3, $"log-mel max abs error {error:G3} (scale {scale:G3})");
    }

    /// <summary>The encoder output matches transformers' encoder (relative to its largest value).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EncoderMatchesTransformers(string name, string clip, string device)
    {
        using var model = Load(name, clip, device);
        using var fixture = SpeechEnvironment.Fixture($"{name}.json");
        var entry = fixture.RootElement.GetProperty(clip);
        var expected = entry.GetProperty("encoder");
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        int frames = entry.GetProperty("features").GetProperty("rows").GetInt32();
        int rows = expected.GetProperty("rows").GetInt32();
        using var encoder = model.CompileComponent(ModelComponentRoles.Encoder, symbolBounds: null);
        encoder.SetInput<float>(encoder.Graph.Input(InputRole.AudioFeatures), TimeMajor(features, frames));
        encoder.SetInput<int>(encoder.Graph.Input(InputRole.Positions), Enumerable.Range(0, (2 * rows) - 1).Select(i => rows - 1 - i).ToArray());
        var run = new RunParameters();
        run.Symbols[WellKnown.AudioFrameSymbol] = frames;
        encoder.Run(run);
        var output = new float[expected.GetProperty("count").GetInt32()];
        encoder.GetOutput<float>(WellKnown.HiddenOutput, output);
        double error = SpeechEnvironment.MaxError(output, expected, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name} {clip} {device}: encoder max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-3 * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens and text.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTokensMatchTransformers(string name, string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(name);
        Skip(path, clip, device);
        using var fixture = SpeechEnvironment.Fixture($"{name}.json");
        var expected = fixture.RootElement.GetProperty(clip);
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), new SpeechRecognitionOptions { Timestamps = TimestampGranularity.Word },
            TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name} {clip} {device}: {result.Text}");
        Assert.Equal(ExpectedTokens(name, expected), result.Segments.SelectMany(s => s.Tokens).ToArray());
        Assert.Equal(expected.GetProperty("text").GetString()!.Trim(), result.Text);
        var segment = Assert.Single(result.Segments);
        Assert.NotEmpty(segment.Words);
        Assert.True(segment.Words.Zip(segment.Words.Skip(1)).All(p => p.First.Start <= p.Second.Start), "word times must not decrease");
    }

    // CTC: transformers' per-row argmax, repeats merged and blanks dropped; TDT: the generated ids without the start token and blanks.
    private static int[] ExpectedTokens(string name, System.Text.Json.JsonElement expected)
    {
        using var config = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(SpeechEnvironment.HuggingFace(name), "config.json")));
        var root = config.RootElement;
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        if (root.GetProperty("model_type").GetString() == "parakeet_ctc")
        {
            int blank = root.GetProperty("pad_token_id").GetInt32();
            return [.. ids.Where((id, i) => id != blank && (i == 0 || ids[i - 1] != id))];
        }

        int tdtBlank = root.GetProperty("blank_token_id").GetInt32();
        return [.. ids.Skip(1).Where(id => id != tdtBlank)];
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

    private static TaskModel Load(string name, string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(name);
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
