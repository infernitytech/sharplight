using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Canary (FastConformer encoder, transformer decoder with a prompt template) of Hugging Face transformers against transformers itself
/// (fixtures from <c>tools/make_canary_fixtures.py</c>) on jfk.wav and a 29 s LibriSpeech recording, each one variable-length window: the
/// encoder output and token-exact greedy decoding of an English transcription with punctuation, through the speech-recognition task client
/// of the service container, on the CPU and on Vulkan.
/// </summary>
public sealed class CanaryParityTests
{
    private const string Model = "canary-1b-v2";

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

    /// <summary>The encoder output matches transformers' encoder (relative to its largest value).</summary>
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
        using var encoder = new SpeechEncoder(model, WellKnown.EncoderOutput);
        var window = new float[features.Channels * features.Frames];
        features.CopyWindow(0, features.Frames, window);
        encoder.Encode(window, features.Frames);
        Assert.Equal(expected.GetProperty("rows").GetInt32(), encoder.Rows);
        double error = SpeechEnvironment.MaxError(encoder.Output.ToArray(), expected, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: encoder max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-3 * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens and text.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device)
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
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), new SpeechRecognitionOptions { Language = "en" },
            TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: {result.Text}");
        int prompt = expected.GetProperty("prompt").GetArrayLength();
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).Skip(prompt)];
        int end = ids.Length > 0 && ids[^1] == 3 ? ids.Length - 1 : ids.Length;
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
