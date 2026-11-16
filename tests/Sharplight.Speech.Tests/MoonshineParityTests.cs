using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Moonshine checkpoints of Hugging Face transformers against transformers itself (fixtures from <c>tools/make_moonshine_fixtures.py</c>)
/// on jfk.wav and a 29 s LibriSpeech recording, each one variable-length window: encoder output over the raw waveform and token-exact
/// greedy decoding, on the CPU and on Vulkan, through the speech-recognition task client of the service container.
/// </summary>
public sealed class MoonshineParityTests
{
    /// <summary>The checkpoints, recordings and devices.</summary>
    public static TheoryData<string, string, string> Cases()
    {
        var cases = new TheoryData<string, string, string>();
        foreach (string model in new[] { "moonshine-tiny", "moonshine-base" })
        {
            foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
            {
                cases.Add(model, clip, "cpu");
                cases.Add(model, clip, "vulkan");
            }
        }

        return cases;
    }

    /// <summary>The encoder output matches transformers' encoder (relative to its largest value).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EncoderMatchesTransformers(string name, string clip, string device)
    {
        using var model = Load(name, clip, device);
        using var fixture = SpeechEnvironment.Fixture($"{name}.json");
        var expected = fixture.RootElement.GetProperty(clip).GetProperty("encoder");
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        using var encoder = model.CompileComponent(ModelComponentRoles.Encoder);
        int rows = expected.GetProperty("rows").GetInt32();
        encoder.SetInput<float>(encoder.Graph.Input(InputRole.AudioFeatures), samples);
        encoder.SetInput<int>(encoder.Graph.Input(InputRole.Positions), Enumerable.Range(0, rows).ToArray());
        var run = new RunParameters();
        run.Symbols[WellKnown.AudioFrameSymbol] = samples.Length;
        encoder.Run(run);
        var output = new float[expected.GetProperty("count").GetInt32()];
        encoder.GetOutput<float>(WellKnown.EncoderOutput, output);
        double error = SpeechEnvironment.MaxError(output, expected, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name} {clip} {device}: encoder max abs error {error:G3} of max abs {scale:G3}");
        // The waveform convolutions (127 taps over raw samples) and six to eight layers accumulate in f32 in a different order than PyTorch.
        Assert.True(error < 1e-4 * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
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
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), cancellationToken: TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name} {clip} {device}: {result.Text}");
        // transformers' ids: the start token, the text, the end token.
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        Assert.Equal(ids[1..^1], result.Segments.SelectMany(s => s.Tokens).ToArray());
        Assert.Equal(expected.GetProperty("text").GetString()!.Trim(), result.Text);
        var segment = Assert.Single(result.Segments);
        Assert.Equal(TimeSpan.Zero, segment.Start);
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
