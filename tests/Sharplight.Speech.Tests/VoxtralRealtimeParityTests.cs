using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Voxtral Mini 4B Realtime (causal streaming audio tower, delayed-token decoder that adds one audio row to every step's token embedding) against
/// transformers (fixtures from <c>tools/make_voxtral_realtime_fixtures.py</c>, float32, offline mode) on jfk.wav and a 29 s LibriSpeech recording:
/// the projected audio embeddings of the padded recording and token-exact greedy decoding through the speech-recognition task client, on the CPU and
/// on Vulkan.
/// </summary>
public sealed class VoxtralRealtimeParityTests
{
    private const string Model = "Voxtral-Mini-4B-Realtime-2602";
    private const string Fixture = "voxtral-realtime-4b.json";

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
        using var recognizer = new CausalAudioSpeechRecognizer(model);
        var embeddings = recognizer.Encode(samples);
        double error = SpeechEnvironment.MaxError(embeddings, expected.GetProperty("embeddings"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: embeddings max abs error {error:G3} of max abs {scale:G3}");
        double tolerance = device == "cpu" ? 2e-3 : 2e-2;
        Assert.True(error < tolerance * scale, $"embeddings max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens (pad and word markers included) and text.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
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
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        int end = ids.Length > 0 && ids[^1] == 2 ? ids.Length - 1 : ids.Length;
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
