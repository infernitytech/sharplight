using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Cohere Transcribe (FastConformer encoder projected to a transformer decoder, prompt template, recordings over 35 s split on the waveform)
/// against transformers' native <c>cohere_asr</c> (fixtures from <c>tools/make_cohere_transcribe_fixtures.py</c>, dither off): the log-mel
/// features and projected encoder output of the first chunk, the energy-based chunk boundaries, and token-exact greedy decoding with
/// punctuation through the speech-recognition task client of the service container, on the CPU and on Vulkan. Recordings: jfk.wav, the 29 s
/// LibriSpeech utterance, and from <c>tools/make_cohere_transcribe_clips.py</c> the first 45 s of a public-domain Obama address (two chunks)
/// and Multilingual LibriSpeech test utterances joined to 60 s of German, 49 s of French (two chunks each) and 16 s of Spanish, and 55 s of
/// JSUT basic5000 Japanese (two chunks joined without a space).
/// </summary>
public sealed class CohereTranscribeParityTests
{
    private const string Model = "cohere-transcribe-03-2026";
    private const string Fixture = Model + ".json";

    private static readonly string[] s_clips =
        [SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName, "obama-45s", "mls-german-4", "mls-french-3", "mls-spanish-1", "jsut-japanese-14"];

    /// <summary>The single-chunk recordings and devices (encoder parity).</summary>
    public static TheoryData<string, string> EncoderCases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            cases.Add(clip, "cpu");
            cases.Add(clip, "vulkan");
        }

        return cases;
    }

    /// <summary>All recordings and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string clip in s_clips)
        {
            cases.Add(clip, "cpu");
            cases.Add(clip, "vulkan");
        }

        return cases;
    }

    /// <summary>All recordings.</summary>
    public static TheoryData<string> Clips() => [.. s_clips];

    /// <summary>The features and the projected encoder output match transformers' (relative to their largest values).</summary>
    [Theory]
    [MemberData(nameof(EncoderCases))]
    public void EncoderMatchesTransformers(string clip, string device)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        var window = new float[features.Channels * features.Frames];
        features.CopyWindow(0, features.Frames, window);
        // The window is channel-major [mels, frames]; the reference is time-major.
        var timeMajor = new float[window.Length];
        for (int c = 0; c < features.Channels; c++)
        {
            for (int t = 0; t < features.Frames; t++)
            {
                timeMajor[(t * features.Channels) + c] = window[(c * features.Frames) + t];
            }
        }

        double featureError = SpeechEnvironment.MaxError(timeMajor, expected.GetProperty("features"), out double featureScale);
        Assert.True(featureError < 1e-3 * featureScale, $"feature max abs error {featureError:G3} of max |value| {featureScale:G3}");

        using var encoder = new SpeechEncoder(model, WellKnown.EncoderOutput);
        encoder.Encode(window, features.Frames);
        var reference = expected.GetProperty("encoder");
        Assert.Equal(reference.GetProperty("rows").GetInt32(), encoder.Rows);
        double error = SpeechEnvironment.MaxError(encoder.Output.ToArray(), reference, out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: features max abs error {featureError:G3} of {featureScale:G3}, encoder {error:G3} of {scale:G3}");
        Assert.True(error < 1e-3 * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>The waveform chunks are the feature extractor's (35 s at most, cut at the quietest 0.1 s of the last 5 s).</summary>
    [Theory]
    [MemberData(nameof(Clips))]
    public void ChunksMatchTransformers(string clip)
    {
        if (!File.Exists(SpeechEnvironment.Audio(clip)))
        {
            Assert.Skip($"{clip}.wav is missing.");
        }

        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip).GetProperty("chunks").EnumerateArray()
            .Select(c => (c.GetProperty("start").GetInt32(), c.GetProperty("end").GetInt32())).ToArray();
        var chunking = WaveformChunking.From(new Dictionary<string, object>
        {
            [SpeechKeys.ChunkSeconds] = 35.0,
            [SpeechKeys.ChunkSearchSeconds] = 5.0,
            [SpeechKeys.ChunkEnergyWindow] = 1600,
        }, 16000)!;
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        Assert.Equal(expected, chunking.Split(samples).ToArray());
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens (chunk by chunk) and text.</summary>
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
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), new SpeechRecognitionOptions { Language = expected.GetProperty("language").GetString() },
            TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: {result.Text}");
        var ids = new List<int>();
        foreach (var chunk in expected.GetProperty("chunks").EnumerateArray())
        {
            int prompt = chunk.GetProperty("prompt").GetArrayLength();
            int[] generated = [.. chunk.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).Skip(prompt)];
            ids.AddRange(generated.Length > 0 && generated[^1] == 3 ? generated[..^1] : generated);
        }

        Assert.Equal(ids, result.Segments.SelectMany(s => s.Tokens).ToArray());
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
