using System.Text.Json;
using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>Silero VAD v5 against onnxruntime: frame probabilities and the segments of the reference implementation on two clips.</summary>
public class SileroVadTests
{
    private const string Model = "silero-vad/silero_vad_16k_op15.onnx";

    [Theory]
    [InlineData(false, "jfk")]
    [InlineData(true, "jfk")]
    [InlineData(false, "librispeech-1272-128104-0004")]
    [InlineData(true, "librispeech-1272-128104-0004")]
    public async Task Probabilities_and_segments_match_the_reference(bool gpu, string clip)
    {
        string wav = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "audio", clip + ".wav");
        AudioModels.Require(AudioModels.Model(Model), wav, AudioModels.Reference($"silero_probs_{clip}.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var detector = AudioModels.Client<IVoiceActivityDetector>(model);
        var audio = WavFile.Read(wav);

        var probabilities = await detector.ComputeProbabilitiesAsync(audio);
        var expected = AudioModels.Floats($"silero_probs_{clip}.f32");
        Assert.Equal(expected.Length, probabilities.Length);
        float worst = probabilities.Zip(expected, (a, b) => MathF.Abs(a - b)).Max();
        Assert.True(worst < 1e-3f, $"largest probability difference {worst}");

        var segments = await detector.DetectAsync(audio);
        var reference = JsonSerializer.Deserialize<List<Dictionary<string, long>>>(File.ReadAllText(AudioModels.Reference($"silero_segments_{clip}.json")))!;
        Assert.Equal(reference.Select(r => (r["start"], r["end"])), segments.Select(s => (s.StartSample, s.EndSample)));
    }
}
