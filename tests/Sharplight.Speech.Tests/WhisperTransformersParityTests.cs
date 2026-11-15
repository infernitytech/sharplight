using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Whisper checkpoints of Hugging Face transformers against transformers itself on jfk.wav (fixtures from
/// <c>tools/make_whisper_fixtures.py</c>): log-mel features, encoder output, and token-exact greedy decoding without timestamps, with
/// timestamps and (multilingual models) with language detection, on the CPU and on Vulkan.
/// </summary>
public sealed class WhisperTransformersParityTests
{
    /// <summary>The checkpoints and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string model in new[] { "whisper-tiny", "whisper-tiny.en", "whisper-base", "whisper-small", "distil-small.en", "distil-large-v3.5" })
        {
            cases.Add(model, "cpu");
            cases.Add(model, "vulkan");
        }

        return cases;
    }

    /// <summary>The log-mel features match transformers' feature extractor.</summary>
    [Theory]
    [InlineData("whisper-tiny")]
    [InlineData("whisper-small")]
    public void FeaturesMatchTransformers(string name)
    {
        using var model = Load(name, "cpu");
        using var fixture = SpeechEnvironment.Fixture($"jfk.{name}.json");
        var audio = WavFile.Read(SpeechEnvironment.Jfk).ToModelInput(16000);
        var features = AudioFeatureExtractors.Create(model).Extract(audio.Samples);
        var window = new float[features.Channels * 3000];
        features.CopyWindow(0, 3000, window);
        double error = MaxError(window, fixture.RootElement.GetProperty("features"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name}: log-mel max abs error {error:G3} (max abs {scale:G3})");
        Assert.True(error < 2e-5 * Math.Max(1, scale), $"log-mel max abs error {error:G3} (scale {scale:G3})");
    }

    /// <summary>The encoder output matches transformers' encoder (relative to its largest value; looser for f16 weights on the GPU).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EncoderMatchesTransformers(string name, string device)
    {
        using var model = Load(name, device);
        using var fixture = SpeechEnvironment.Fixture($"jfk.{name}.json");
        var audio = WavFile.Read(SpeechEnvironment.Jfk).ToModelInput(16000);
        var features = AudioFeatureExtractors.Create(model).Extract(audio.Samples);
        using var encoder = model.CompileComponent(ModelComponentRoles.Encoder);
        var input = encoder.Graph.Input(InputRole.AudioFeatures);
        var window = new float[input.Shape.Elements()];
        features.CopyWindow(0, (int)input.Shape[1].Fixed, window);
        encoder.SetInput<float>(input, window);
        encoder.Run(new RunParameters());
        var output = new float[encoder.Graph.Outputs[WellKnown.EncoderOutput].Shape.Elements()];
        encoder.GetOutput<float>(WellKnown.EncoderOutput, output);
        double error = MaxError(output, fixture.RootElement.GetProperty("encoder"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{name} {device}: encoder max abs error {error:G3} of max abs {scale:G3}");
        // Half-precision checkpoints (distil-small.en) run f16 matrix products on the GPU; transformers upcasts them to f32.
        bool halfWeights = model.Definition.Weights.Values.Any(w => w.Type == DataType.F16);
        double tolerance = halfWeights && device != "cpu" ? 2e-3 : 2e-5;
        Assert.True(error < tolerance * scale, $"encoder max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding with a given language gives transformers' tokens, with and without timestamps.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task GreedyTokensMatchTransformers(string name, string device)
    {
        using var model = Load(name, device);
        using var recognizer = new EncoderDecoderSpeechRecognizerFactory().Create(model) as ISpeechRecognizer;
        using var fixture = SpeechEnvironment.Fixture($"jfk.{name}.json");
        var audio = WavFile.Read(SpeechEnvironment.Jfk);
        string? language = name.EndsWith(".en", StringComparison.Ordinal) ? null : "en";
        foreach (var (key, timestamps) in new[] { ("plain", TimestampGranularity.None), ("timestamps", TimestampGranularity.Segment) })
        {
            var result = await recognizer!.TranscribeAsync(audio, new SpeechRecognitionOptions { Language = language, Timestamps = timestamps });
            var expected = fixture.RootElement.GetProperty(key);
            Assert.Equal(Generated(expected, model), result.Segments.SelectMany(s => s.Tokens).ToArray());
            Assert.Equal(expected.GetProperty("text").GetString()!.Trim(), result.Text);
        }
    }

    /// <summary>A 5-beam search gives transformers' <c>num_beams=5</c> tokens, with and without timestamps.</summary>
    [Theory]
    [InlineData("whisper-tiny", "cpu")]
    [InlineData("whisper-tiny.en", "vulkan")]
    [InlineData("whisper-base", "cpu")]
    [InlineData("whisper-small", "vulkan")]
    [InlineData("distil-small.en", "cpu")]
    public async Task BeamSearchMatchesTransformers(string name, string device)
    {
        using var model = Load(name, device);
        using var recognizer = (ISpeechRecognizer)new EncoderDecoderSpeechRecognizerFactory().Create(model);
        using var fixture = SpeechEnvironment.Fixture($"jfk.{name}.json");
        var audio = WavFile.Read(SpeechEnvironment.Jfk);
        string? language = name.EndsWith(".en", StringComparison.Ordinal) ? null : "en";
        foreach (var (key, timestamps) in new[] { ("beam_plain", TimestampGranularity.None), ("beam_timestamps", TimestampGranularity.Segment) })
        {
            var options = new SpeechRecognitionOptions { Language = language, Timestamps = timestamps, BeamSize = 5 };
            var result = await recognizer.TranscribeAsync(audio, options);
            var expected = fixture.RootElement.GetProperty(key);
            Assert.Equal(Generated(expected, model), result.Segments.SelectMany(s => s.Tokens).ToArray());
        }
    }

    /// <summary>Multilingual models detect English and then decode as transformers does without a language.</summary>
    [Theory]
    [InlineData("whisper-tiny", "cpu")]
    [InlineData("whisper-base", "vulkan")]
    public async Task LanguageDetectionMatchesTransformers(string name, string device)
    {
        using var model = Load(name, device);
        using var recognizer = (ISpeechRecognizer)new EncoderDecoderSpeechRecognizerFactory().Create(model);
        using var fixture = SpeechEnvironment.Fixture($"jfk.{name}.json");
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Jfk), new SpeechRecognitionOptions { Timestamps = TimestampGranularity.None });
        Assert.Equal("en", result.Language);
        Assert.Equal(Generated(fixture.RootElement.GetProperty("detected"), model), result.Segments.SelectMany(s => s.Tokens).ToArray());
    }

    private static TaskModel Load(string name, string device)
    {
        string path = SpeechEnvironment.HuggingFace(name);
        if (!SpeechEnvironment.Available(path, device))
        {
            Assert.Skip($"{path}, jfk.wav or the {device} device is missing.");
        }

        return TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
    }

    /// <summary>The generated tokens of a transformers result: after the prompt (start, language, task, no-timestamps), without the end token.</summary>
    private static int[] Generated(JsonElement expected, TaskModel model)
    {
        var metadata = model.Metadata(ModelComponentRoles.Decoder);
        int end = Convert.ToInt32(metadata[SpeechKeys.EndId]);
        var prompt = new HashSet<int> { Convert.ToInt32(metadata[SpeechKeys.StartId]), Convert.ToInt32(metadata[SpeechKeys.NoTimestampsId]) };
        foreach (string key in new[] { SpeechKeys.TranscribeId, SpeechKeys.TranslateId })
        {
            if (metadata.TryGetValue(key, out var id))
            {
                prompt.Add(Convert.ToInt32(id));
            }
        }

        if (metadata.TryGetValue(SpeechKeys.LanguageIds, out var languages))
        {
            prompt.UnionWith((int[])languages);
        }

        var ids = expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToList();
        int first = ids.FindIndex(id => !prompt.Contains(id));
        return [.. ids.Skip(first).TakeWhile(id => id != end)];
    }

    /// <summary>The largest difference to the sampled reference values, and the largest reference magnitude.</summary>
    private static double MaxError(float[] actual, JsonElement reference, out double scale)
    {
        Assert.Equal(reference.GetProperty("count").GetInt32(), actual.Length);
        int stride = reference.GetProperty("stride").GetInt32();
        double error = 0;
        scale = 0;
        int index = 0;
        foreach (var value in reference.GetProperty("values").EnumerateArray())
        {
            double expected = value.GetDouble();
            error = Math.Max(error, Math.Abs(actual[index] - expected));
            scale = Math.Max(scale, Math.Abs(expected));
            index += stride;
        }

        return error;
    }
}
