using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// wav2vec 2.0 CTC checkpoints (facebook/wav2vec2-base-960h) against transformers (fixtures from <c>tools/make_wav2vec2_fixtures.py</c>) on
/// jfk.wav and a 29 s LibriSpeech recording, each one window: the encoder's hidden states and the greedy CTC ids through the speech-recognition
/// task client of the service container, on the CPU and on Vulkan.
/// </summary>
public sealed class Wav2Vec2ParityTests
{
    private const string Model = "wav2vec2-base-960h";

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

    /// <summary>The encoder's last hidden states match transformers (relative to the largest value).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EncoderMatchesTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture($"{Model}.json");
        var expected = fixture.RootElement.GetProperty(clip).GetProperty("encoder");
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        using var encoder = model.CompileComponent(ModelComponentRoles.Encoder, symbolBounds: null);
        encoder.SetInput<float>(encoder.Graph.Input(InputRole.AudioFeatures), features.Values);
        var run = new RunParameters();
        run.Symbols[WellKnown.AudioFrameSymbol] = features.Frames;
        encoder.Run(run);
        var output = new float[expected.GetProperty("count").GetInt32()];
        encoder.GetOutput<float>(WellKnown.HiddenOutput, output);
        double error = SpeechEnvironment.MaxError(output, expected, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: encoder max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 2e-3 * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy CTC decoding through the task client gives transformers' text and tokens.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTextMatchesTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
        Skip(path, clip, device);
        using var fixture = SpeechEnvironment.Fixture($"{Model}.json");
        var expected = fixture.RootElement.GetProperty(clip);
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), new SpeechRecognitionOptions(), TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: {result.Text}");
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        int[] tokens = [.. ids.Where((id, i) => id != 0 && (i == 0 || ids[i - 1] != id))];
        Assert.Equal(tokens, result.Segments.SelectMany(s => s.Tokens).ToArray());
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
