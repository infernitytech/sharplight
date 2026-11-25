using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// A tiny random Stable Audio pipeline (<c>tools/make-tiny-stable-audio.py</c>: the diffusers layout with a 2-layer transformer, a 2-layer T5 encoder and a two-stage
/// Oobleck decoder) against diffusers' <c>StableAudioPipeline</c>: 4 DPM-Solver++ SDE steps with guidance 3 from the reference's noise, without and with a negative prompt.
/// </summary>
public class TinyStableAudioTests
{
    private const string Model = "tiny-random-stable-audio";

    private static string Reference(string name) => Path.Combine(AudioModels.Model(Model), "reference", name);

    private static float[] Floats(string name) => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Reference(name))).ToArray();

    private static AudioGenerationRequest Request(string kind) => new()
    {
        Prompt = "a dog barking twice in a quiet room",
        NegativePrompt = kind == "negative" ? "music, noise" : null,
        Duration = TimeSpan.FromSeconds(0.4),
    };

    private static AudioGenerationOptions Options(string kind, int count)
    {
        var stepNoise = Floats($"{kind}_step_noise.f32");
        return new AudioGenerationOptions
        {
            Steps = 4,
            GuidanceScale = 3,
            InitialNoise = Floats($"{kind}_noise.f32"),
            StepNoise = [.. Enumerable.Range(0, stepNoise.Length / count).Select(i => new ReadOnlyMemory<float>(stepNoise, i * count, count))],
        };
    }

    [Theory]
    [InlineData(false, "plain")]
    [InlineData(false, "negative")]
    [InlineData(true, "plain")]
    [InlineData(true, "negative")]
    public async Task Latents_and_audio_match_diffusers(bool gpu, string kind)
    {
        AudioModels.Require(Path.Combine(AudioModels.Model(Model), "diffusers"), Reference($"{kind}_wav.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), Path.Combine(AudioModels.Model(Model), "diffusers"));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        Assert.Equal(512, generator.SampleRate);
        Assert.Equal(64, generator.LatentFrameRate);
        Assert.Equal(0.5, generator.MaxDuration.TotalSeconds, 6);

        var request = Request(kind);
        var options = Options(kind, 32 * 8);
        var latents = await generator.GenerateLatentsAsync(request, options);
        var expectedLatents = Floats($"{kind}_latent.f32");
        double latentError = AudioModels.RelativeRms(latents.Values, expectedLatents);
        ParityReport.Write($"tiny-stable-audio {kind} gpu={gpu} latent relative RMS {latentError:E2}");
        Assert.True(latentError < (gpu ? 1e-3 : 1e-4), $"latent relative RMS {latentError}");

        var audio = (await generator.GenerateAsync(request, options)).Audio;
        var expected = Floats($"{kind}_wav.f32");
        Assert.Equal(expected.Length, audio.Samples.Length);
        double audioError = AudioModels.RelativeRms(audio.Samples, expected);
        ParityReport.Write($"tiny-stable-audio {kind} gpu={gpu} audio relative RMS {audioError:E2}");
        Assert.True(audioError < (gpu ? 1e-3 : 1e-4), $"audio relative RMS {audioError}");
    }
}
