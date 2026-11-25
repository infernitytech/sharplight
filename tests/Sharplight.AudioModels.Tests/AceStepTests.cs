using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// ACE-Step 1.5 turbo against the official modeling code (<c>tools/audio-reference/reference_acestep.py</c>): the same noise, 8 Euler steps, the latents and the decoded audio
/// of a 10 s instrumental without reference audio.
/// </summary>
public class AceStepTests
{
    private const string Model = "ace-step15";
    private const string Caption = "A calm acoustic guitar melody with soft piano, warm and relaxed";

    private static void RequireAll()
        => AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("acestep_noise.f32"), AudioModels.Reference("acestep_latent.f32"), AudioModels.Reference("acestep_wav.f32"));

    private static AudioGenerationRequest Request() => new()
    {
        Prompt = Caption,
        Lyrics = "[Instrumental]",
        Duration = TimeSpan.FromSeconds(10),
    };

    private static AudioGenerationOptions Options() => new() { InitialNoise = AudioModels.Floats("acestep_noise.f32") };

    [Theory]
    [InlineData(false, 2e-3)]
    [InlineData(true, 1e-2)]
    public async Task Latents_and_audio_match_the_reference(bool gpu, double tolerance)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        Assert.Equal(48000, generator.SampleRate);
        Assert.Equal(2, generator.Channels);
        Assert.Equal(25, generator.LatentFrameRate);
        Assert.Equal(8, generator.Defaults.Steps);

        var latents = await generator.GenerateLatentsAsync(Request(), Options());
        var expected = AudioModels.Floats("acestep_latent.f32");
        Assert.Equal(expected.Length, latents.Values.Length);
        double latentError = AudioModels.RelativeRms(latents.Values, expected);
        ParityReport.Write($"acestep gpu={gpu} latent relative RMS {latentError:E2}");
        Assert.True(latentError < tolerance, $"latent relative RMS {latentError}");

        var audio = (await generator.GenerateAsync(Request(), Options())).Audio;
        var reference = AudioModels.Floats("acestep_wav.f32");
        Assert.Equal(reference.Length, audio.Samples.Length);
        double audioError = AudioModels.RelativeRms(audio.Samples, reference);
        ParityReport.Write($"acestep gpu={gpu} audio relative RMS {audioError:E2}");
        Assert.True(audioError < tolerance * 5, $"audio relative RMS {audioError}");
    }

    [Fact]
    public async Task A_seed_gives_reproducible_audio_and_other_durations_work()
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        var request = new AudioGenerationRequest { Prompt = Caption, Duration = TimeSpan.FromSeconds(2.4) };
        var options = new AudioGenerationOptions { Steps = 2, Seed = 7 };
        var steps = new List<int>();
        AudioBuffer? first = null;
        await foreach (var update in generator.GenerateStreamingAsync(request, options))
        {
            steps.Add(update.Step);
            first = update.Result?.Audio ?? first;
        }

        var second = (await generator.GenerateAsync(request, options)).Audio;
        Assert.Equal([1, 2, 2], steps);
        Assert.NotNull(first);
        Assert.Equal(48000 * 2.4, first.FrameCount, 1);
        Assert.Equal(first.Samples, second.Samples);
        Assert.Equal(2, first.Channels);
    }

    [Fact]
    public async Task Package_generates_the_same_audio()
    {
        AudioModels.Require(AudioModels.Model(Model));
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "audio-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"acestep-{Guid.NewGuid():N}.slm");
        try
        {
            AudioModels.Services.GetRequiredService<ModelConverter>().Convert(AudioModels.Model(Model), path);
            Assert.Equal(ModelKinds.AudioGeneration, ModelPackageReader.Open(path).Definition.Description.Kind);
            var request = new AudioGenerationRequest { Prompt = Caption, Duration = TimeSpan.FromSeconds(1.2) };
            var options = new AudioGenerationOptions { Steps = 2, Seed = 3 };
            using var original = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
            using var packaged = AudioModels.Load(AudioModels.Open(gpu: false), path);
            using var a = AudioModels.Client<IAudioGenerator>(original);
            using var b = AudioModels.Client<IAudioGenerator>(packaged);
            Assert.Equal((await a.GenerateAsync(request, options)).Audio.Samples, (await b.GenerateAsync(request, options)).Audio.Samples);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
