using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// A tiny random VoxCPM2 checkpoint (<c>tools/make-tiny-voxcpm2.py</c>, in <c>~/models/hf/tiny-random-voxcpm2</c>) against the official voxcpm package's run of it
/// (<c>Fixtures/tiny-random-voxcpm2.json</c>: every 37th value): free-running generation with the reference's noise, zero-shot and with a 2 s reference recording
/// (the first 2 s of <c>~/models/audio/jfk.wav</c>), on the CPU and the GPU under test, and the <c>.slm</c> round trip.
/// </summary>
public class VoxCpm2TinyModelTests
{
    // The reference's max_len (tools/make-tiny-voxcpm2.py).
    private const int MaxSteps = 6;

    private static string ModelDirectory => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-voxcpm2");

    private static string Recording => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "audio", "jfk.wav");

    private static JsonElement Case(string kind)
        => JsonDocument.Parse(File.ReadAllText(TtsModels.Fixture("tiny-random-voxcpm2.json"))).RootElement.GetProperty("cases").GetProperty(kind);

    private static float[] Values(JsonElement c, string name) => [.. c.GetProperty(name).EnumerateArray().Select(v => v.GetSingle())];

    private static void RequireModel() => TtsModels.Require(Path.Combine(ModelDirectory, "model.safetensors"), Path.Combine(ModelDirectory, "audiovae.pth"), Recording);

    private static SpeechSynthesisOptions Options(JsonElement c)
    {
        var options = new SpeechSynthesisOptions { NormalizeText = false };
        int seconds = c.GetProperty("reference_seconds").GetInt32();
        if (seconds > 0)
        {
            var recording = WavFile.Read(Recording);
            options = options with { ReferenceAudio = new AudioBuffer(recording.Samples[..(seconds * recording.SampleRate)].ToArray(), recording.SampleRate) };
        }

        return options;
    }

    private static VoxCpm2Forcing Forcing(JsonElement c)
    {
        var noise = Values(c, "noise");
        int size = noise.Length / c.GetProperty("steps").GetInt32();
        return new VoxCpm2Forcing { Noise = [.. Enumerable.Range(0, noise.Length / size).Select(i => noise.AsSpan(i * size, size).ToArray())], MaxSteps = MaxSteps };
    }

    // Every stride-th value of a sequence.
    private static float[] Every(IEnumerable<float> values, int stride) => [.. values.Where((_, i) => i % stride == 0)];

    private static double RelativeRms(float[] actual, float[] expected)
    {
        double error = 0;
        double energy = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            energy += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(error / energy);
    }

    /// <summary>The prompt tokens, the patch count, the text-semantic states, the patches and the audio are the reference's.</summary>
    [Theory]
    [InlineData("zero", false)]
    [InlineData("clone", false)]
    [InlineData("zero", true)]
    [InlineData("clone", true)]
    public void Generation_matches_the_reference(string kind, bool gpu)
    {
        RequireModel();
        var c = Case(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), ModelDirectory);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var options = Options(c);
        string text = c.GetProperty("text").GetString()!;
        float[]? reference = options.ReferenceAudio is { } recording ? synthesizer.EncodeAudio(recording.Samples.ToArray(), padStart: false) : null;
        var prompt = synthesizer.BuildPrompt(VoxCpm2Synthesizer.TargetText(text, options), reference, null);
        Assert.Equal(c.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()), prompt.Tokens);

        var states = new List<float>();
        var generated = synthesizer.GeneratePatches(text, options, CancellationToken.None, Forcing(c), step => states.AddRange(step.LmHidden)).ToList();
        Assert.Equal(c.GetProperty("steps").GetInt32(), generated.Count);
        int stride = JsonDocument.Parse(File.ReadAllText(TtsModels.Fixture("tiny-random-voxcpm2.json"))).RootElement.GetProperty("stride").GetInt32();
        var audio = generated.SelectMany(p => p.Audio).ToArray();
        Assert.Equal(c.GetProperty("samples").GetInt32(), audio.Length);
        double stateError = RelativeRms(Every(states, stride), Values(c, "lm_hidden"));
        double patchError = RelativeRms(Every(generated.SelectMany(p => p.Latents), stride), Values(c, "patches"));
        double audioError = RelativeRms(Every(audio, stride), Values(c, "audio"));
        ParityReport.Write($"VoxCPM2 tiny {kind} {(gpu ? "gpu" : "cpu")}: {generated.Count} patches, relative RMS: states {stateError:E2}, patches {patchError:E2}, audio {audioError:E2}");
        Assert.True(stateError < (gpu ? 2e-2 : 1e-4), $"text-semantic states relative RMS {stateError:E2}");
        Assert.True(patchError < (gpu ? 2e-2 : 1e-4), $"patches relative RMS {patchError:E2}");
        Assert.True(audioError < (gpu ? 2e-2 : 1e-4), $"audio relative RMS {audioError:E2}");
    }

    /// <summary>Streaming yields the audio of the whole synthesis in buffers of two patches.</summary>
    [Fact]
    public async Task Streaming_equals_the_whole_synthesis()
    {
        RequireModel();
        using var model = TtsModels.Load(TtsModels.Open(false), ModelDirectory);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var options = new SpeechSynthesisOptions { NormalizeText = false, Sampling = new Llm.SamplingOptions { Seed = 11 } };
        var whole = await synthesizer.SynthesizeAsync("Hello, tiny world.", options, TestContext.Current.CancellationToken);
        var chunks = new List<float>();
        int buffers = 0;
        await foreach (var buffer in synthesizer.SynthesizeStreamingAsync("Hello, tiny world.", options, TestContext.Current.CancellationToken))
        {
            Assert.Equal(48000, buffer.SampleRate);
            Assert.True(buffer.Samples.Length <= VoxCpm2Synthesizer.StreamPatches * synthesizer.SamplesPerPatch);
            chunks.AddRange(buffer.Samples.ToArray());
            buffers++;
        }

        Assert.True(buffers > 1);
        Assert.Equal(whole.Audio.Samples.ToArray(), chunks);
    }

    /// <summary>The model converted to a <c>.slm</c> package and loaded again generates the same audio.</summary>
    [Fact]
    public void Package_round_trip_generates_the_same_audio()
    {
        RequireModel();
        var c = Case("clone");
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"voxcpm2-{Guid.NewGuid():N}.slm");
        try
        {
            TtsModels.Services.GetRequiredService<ModelConverter>().Convert(ModelDirectory, path);
            Assert.Equal(ModelKinds.SpeechSynthesis, ModelPackageReader.Open(path).Definition.Description.Kind);
            using var original = TtsModels.Load(TtsModels.Open(false), ModelDirectory);
            using var packaged = TtsModels.Load(TtsModels.Open(false), path);
            using var a = (VoxCpm2Synthesizer)TtsModels.Synthesizer(original);
            using var b = (VoxCpm2Synthesizer)TtsModels.Synthesizer(packaged);
            string text = c.GetProperty("text").GetString()!;
            var expected = a.GeneratePatches(text, Options(c), CancellationToken.None, Forcing(c)).SelectMany(p => p.Audio).ToArray();
            var actual = b.GeneratePatches(text, Options(c), CancellationToken.None, Forcing(c)).SelectMany(p => p.Audio).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            Assert.True(RelativeRms(actual, expected) == 0, $"audio relative RMS {RelativeRms(actual, expected):E2}");
        }
        finally
        {
            File.Delete(path);
        }
    }
}
