using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;

namespace Sharplight.AudioModels.Tests;

/// <summary>The audio models of round 2 in <c>.slm</c> packages: converted once, loaded again, same results.</summary>
public class PackageTests
{
    private static string TempPackage(string name)
    {
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "audio-tests");
        Directory.CreateDirectory(folder);
        return Path.Combine(folder, $"{name}-{Guid.NewGuid():N}.slm");
    }

    [Fact]
    public async Task Clap_package_embeds_like_the_checkpoint()
    {
        AudioModels.Require(AudioModels.Model("clap-htsat-unfused"), AudioModels.Reference("clap_in.f32"));
        string path = TempPackage("clap");
        try
        {
            AudioModels.Services.GetRequiredService<ModelConverter>().Convert(AudioModels.Model("clap-htsat-unfused"), path);
            Assert.Equal(ModelKinds.AudioEmbedding, ModelPackageReader.Open(path).Definition.Description.Kind);
            var audio = new AudioBuffer(AudioModels.Floats("clap_in.f32"), 48000);
            using var original = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model("clap-htsat-unfused"));
            using var packaged = AudioModels.Load(AudioModels.Open(gpu: false), path);
            using var a = AudioModels.Client<IAudioEmbedder>(original);
            using var b = AudioModels.Client<IAudioEmbedder>(packaged);
            Assert.Equal(await a.EmbedAudioAsync(audio), await b.EmbedAudioAsync(audio));
            Assert.Equal(await a.EmbedTextAsync("a dog barking"), await b.EmbedTextAsync("a dog barking"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task MusicGen_package_generates_the_same_codes()
    {
        AudioModels.Require(AudioModels.Model("musicgen-small"));
        string path = TempPackage("musicgen");
        try
        {
            AudioModels.Services.GetRequiredService<ModelConverter>().Convert(AudioModels.Model("musicgen-small"), path);
            Assert.Equal(ModelKinds.MusicGeneration, ModelPackageReader.Open(path).Definition.Description.Kind);
            var request = new MusicGenerationRequest { Prompt = "80s pop track with bassy drums and synth", Duration = TimeSpan.FromSeconds(0.5), Greedy = true };
            using var original = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model("musicgen-small"));
            using var packaged = AudioModels.Load(AudioModels.Open(gpu: false), path);
            using var a = AudioModels.Client<IMusicGenerator>(original);
            using var b = AudioModels.Client<IMusicGenerator>(packaged);
            Assert.Equal((await a.GenerateCodesAsync(request)).Values, (await b.GenerateCodesAsync(request)).Values);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task Demucs_package_separates_like_the_checkpoint()
    {
        AudioModels.Require(AudioModels.Model("htdemucs"), AudioModels.Reference("demucs_in.f32"));
        string path = TempPackage("htdemucs");
        try
        {
            AudioModels.Services.GetRequiredService<ModelConverter>().Convert(AudioModels.Model("htdemucs"), path);
            Assert.Equal(ModelKinds.AudioSeparation, ModelPackageReader.Open(path).Definition.Description.Kind);
            var planar = AudioModels.Floats("demucs_in.f32");
            int length = 44100 * 2;
            var interleaved = new float[length * 2];
            for (int i = 0; i < length; i++)
            {
                interleaved[2 * i] = planar[i];
                interleaved[(2 * i) + 1] = planar[(planar.Length / 2) + i];
            }

            var mix = new AudioBuffer(interleaved, 44100, 2);
            using var original = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model("htdemucs"));
            using var packaged = AudioModels.Load(AudioModels.Open(gpu: false), path);
            using var a = AudioModels.Client<IAudioSeparator>(original);
            using var b = AudioModels.Client<IAudioSeparator>(packaged);
            var first = await a.SeparateAsync(mix);
            var second = await b.SeparateAsync(mix);
            for (int s = 0; s < first.Count; s++)
            {
                Assert.Equal(first[s].Audio.Samples, second[s].Audio.Samples);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
