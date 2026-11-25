using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// Seeds became 64-bit in 0.3: a seed within the 32-bit range must reproduce the audio of the releases with 32-bit seeds. The expected hashes (FNV-1a over the
/// sample bits) were recorded on the CPU with the 32-bit implementation, for the tiny Stable Audio pipeline (seeded initial and SDE step noise) and the tiny
/// Stable Audio 3 one (seeded initial noise and decoder dither).
/// </summary>
public class SeedCompatibilityTests
{
    [Theory]
    [InlineData("tiny-random-stable-audio", "diffusers", 12345L, 3426414395903876178UL)]
    [InlineData("tiny-random-stable-audio", "diffusers", -7L, 1451384594584141933UL)]
    [InlineData("tiny-random-stable-audio-3", "small", 12345L, 4790541417137114047UL)]
    [InlineData("tiny-random-stable-audio-3", "small", -7L, 7406207753412927989UL)]
    public async Task Int32_seeds_generate_the_audio_of_32_bit_seeds(string model, string variant, long seed, ulong expected)
    {
        string path = Path.Combine(AudioModels.Model(model), variant);
        AudioModels.Require(path);
        using var loaded = AudioModels.Load(AudioModels.Open(false), path);
        using var generator = AudioModels.Client<IAudioGenerator>(loaded);
        var request = new AudioGenerationRequest { Prompt = "a dog barking twice in a quiet room", NegativePrompt = "music, noise", Duration = TimeSpan.FromSeconds(0.4) };
        var result = await generator.GenerateAsync(request, new AudioGenerationOptions { Steps = 3, Seed = seed }, TestContext.Current.CancellationToken);
        Assert.Equal(seed, result.Seed);
        Assert.Equal(408, result.Audio.Samples.Length);
        Assert.Equal(expected, Hash(result.Audio.Samples));
    }

    private static ulong Hash(float[] samples)
    {
        ulong h = 14695981039346656037UL;
        foreach (float f in samples)
        {
            h = (h ^ BitConverter.SingleToUInt32Bits(f)) * 1099511628211UL;
        }

        return h;
    }
}
