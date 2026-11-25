using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// Two tiny random Stable Audio 3 checkpoints (<c>tools/make-tiny-stable-audio-3.py</c>: <c>small</c> with the chunked SAME decoder of the small models, <c>medium</c> with
/// the medium models' differential transformer attention and sliding-window, sine-gated decoder) against the stable-audio-3 package: 6 Euler steps with adaptive
/// projected guidance 3 without and with a negative prompt, and 4 ping-pong steps, from the reference's noise (latents, and the audio decoded without dither).
/// </summary>
public class TinyStableAudio3Tests
{
    private const string Model = "tiny-random-stable-audio-3";
    private const int Frames = 208;
    private const int Channels = 8;

    private static string Reference(string name) => Path.Combine(AudioModels.Model(Model), "reference", name);

    private static float[] Floats(string name) => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Reference(name))).ToArray();

    [Theory]
    [InlineData(false, "small", "euler")]
    [InlineData(false, "small", "negative")]
    [InlineData(false, "small", "pingpong")]
    [InlineData(false, "medium", "euler")]
    [InlineData(false, "medium", "negative")]
    [InlineData(false, "medium", "pingpong")]
    [InlineData(true, "small", "euler")]
    [InlineData(true, "small", "negative")]
    [InlineData(true, "small", "pingpong")]
    [InlineData(true, "medium", "euler")]
    [InlineData(true, "medium", "negative")]
    [InlineData(true, "medium", "pingpong")]
    public async Task Latents_and_audio_match_the_package(bool gpu, string variant, string kind)
    {
        string tag = $"{variant}_{kind}";
        AudioModels.Require(Path.Combine(AudioModels.Model(Model), variant), Reference($"{tag}_wav.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), Path.Combine(AudioModels.Model(Model), variant));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        var client = Assert.IsType<FlowAudioDiffusionClient>(generator);
        Assert.Equal(512, generator.SampleRate);
        Assert.Equal(32, generator.LatentFrameRate);
        Assert.Equal(8, generator.MaxDuration.TotalSeconds, 6);
        Assert.Equal(Frames, client.Frames(0.5));

        bool pingPong = kind == "pingpong";
        var request = new AudioGenerationRequest
        {
            Prompt = "a dog barking twice in a quiet room",
            NegativePrompt = kind == "negative" ? "music, noise" : null,
            Duration = TimeSpan.FromSeconds(0.5),
        };
        var options = new AudioGenerationOptions
        {
            Steps = pingPong ? 4 : 6,
            GuidanceScale = pingPong ? 1 : 3,
            InitialNoise = Floats($"{tag}_noise.f32"),
            DecoderNoise = 0,
        };
        if (pingPong)
        {
            var stepNoise = Floats($"{tag}_step_noise.f32");
            int count = Frames * Channels;
            options = options with { StepNoise = [.. Enumerable.Range(0, stepNoise.Length / count).Select(i => new ReadOnlyMemory<float>(stepNoise, i * count, count))] };
        }

        var latents = await client.GenerateLatentsAsync(request, options, pingPong);
        double latentError = AudioModels.RelativeRms(latents.Values, Floats($"{tag}_latent.f32"));
        ParityReport.Write($"tiny-stable-audio-3 {tag} gpu={gpu} latent relative RMS {latentError:E2}");
        Assert.True(latentError < (gpu ? 1e-3 : 1e-4), $"latent relative RMS {latentError}");

        var audio = client.Decode(latents, request, options, CancellationToken.None);
        var expected = Floats($"{tag}_wav.f32");
        Assert.Equal(expected.Length, audio.Samples.Length);
        double audioError = AudioModels.RelativeRms(audio.Samples, expected);
        ParityReport.Write($"tiny-stable-audio-3 {tag} gpu={gpu} audio relative RMS {audioError:E2}");
        Assert.True(audioError < (gpu ? 1e-3 : 1e-4), $"audio relative RMS {audioError}");
    }
}
