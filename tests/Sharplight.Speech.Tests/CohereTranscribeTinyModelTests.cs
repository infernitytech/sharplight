using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// The structure of Cohere Transcribe against transformers on a tiny random checkpoint in transformers' own layout (<c>tools/make-tiny-cohere-asr.py</c>:
/// native configuration, NeMo tensor names under <c>model.</c>, random batch-norm statistics): features, projected encoder output and the decoder
/// logits after the prompt and after each of a few forced tokens on jfk.wav, on the CPU and on the GPU under test.
/// </summary>
public sealed class CohereTranscribeTinyModelTests
{
    private const string Model = "tiny-random-cohere-asr";

    /// <summary>The devices.</summary>
    public static TheoryData<string> Devices() => ["cpu", "vulkan"];

    /// <summary>Features, encoder output and teacher-forced decoder logits match transformers' (relative to their largest values).</summary>
    [Theory]
    [MemberData(nameof(Devices))]
    public void MatchesTransformers(string device)
    {
        string path = SpeechEnvironment.HuggingFace(Model);
        if (!SpeechEnvironment.Available(path, device))
        {
            Assert.Skip($"{path}, jfk.wav or the {device} device is missing.");
        }

        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Model + ".json");
        var expected = fixture.RootElement;
        var samples = WavFile.Read(SpeechEnvironment.Jfk).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        var window = new float[features.Channels * features.Frames];
        features.CopyWindow(0, features.Frames, window);
        var timeMajor = new float[window.Length];
        for (int c = 0; c < features.Channels; c++)
        {
            for (int t = 0; t < features.Frames; t++)
            {
                timeMajor[(t * features.Channels) + c] = window[(c * features.Frames) + t];
            }
        }

        Within(SpeechEnvironment.MaxError(timeMajor, expected.GetProperty("features"), out double scale), scale, "features", device);

        using (var encoder = new SpeechEncoder(model, WellKnown.EncoderOutput))
        {
            encoder.Encode(window, features.Frames);
            Within(SpeechEnvironment.MaxError(encoder.Output.ToArray(), expected.GetProperty("encoder"), out double encoderScale), encoderScale, "encoder", device);
        }

        using var network = new SpeechEncoderDecoder(model);
        network.Encode(window, features.Frames);
        Assert.Equal(expected.GetProperty("encoder").GetProperty("rows").GetInt32(), network.SourceRows);

        int[] prompt = [.. expected.GetProperty("prompt").EnumerateArray().Select(e => e.GetInt32())];
        int[] forced = [.. expected.GetProperty("forced").EnumerateArray().Select(e => e.GetInt32())];
        var logits = expected.GetProperty("logits");
        Assert.Equal(forced.Length + 1, logits.GetArrayLength());
        var actual = network.Decode(prompt);
        for (int step = 0; step <= forced.Length; step++)
        {
            Within(SpeechEnvironment.MaxError(actual, logits[step], out double logitScale), logitScale, $"logits {step}", device);
            if (step < forced.Length)
            {
                actual = network.Decode([forced[step]]);
            }
        }
    }

    private static void Within(double error, double scale, string what, string device)
    {
        TestContext.Current.TestOutputHelper?.WriteLine($"{device} {what}: max abs error {error:G3} of max |value| {scale:G3}");
        Assert.True(error < 1e-3 * scale, $"{what}: max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }
}
