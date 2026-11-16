using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>Mimi against transformers' MimiModel: token-exact codes, decoded audio to a relative RMS of 1e-3.</summary>
public class MimiTests
{
    private const string Model = "mimi";

    [Theory]
    [InlineData(false, 8)]
    [InlineData(true, 8)]
    [InlineData(true, 32)]
    public async Task Codes_and_audio_match_the_reference(bool gpu, int codebooks)
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("mimi_in.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var codec = AudioModels.Client<IAudioCodec>(model);
        Assert.Equal(24000, codec.SampleRate);
        Assert.Equal(1920, codec.SamplesPerFrame);
        var input = new AudioBuffer(AudioModels.Floats("mimi_in.f32"), 24000);

        var codes = await codec.EncodeAsync(input, new AudioCodecOptions { Codebooks = codebooks });
        var expected = AudioModels.Ints($"mimi_codes_{codebooks}.i32");
        Assert.Equal(expected.Length, codes.Values.Length);
        int wrong = expected.Where((v, i) => v != codes.Values[i]).Count();
        Assert.True(wrong == 0, $"{wrong} of {expected.Length} codes differ");

        var decoded = await codec.DecodeAsync(codes);
        var reference = AudioModels.Floats($"mimi_dec_{codebooks}.f32");
        Assert.Equal(reference.Length, decoded.Samples.Length);
        double error = AudioModels.RelativeRms(decoded.Samples, reference);
        Assert.True(error < 1e-3, $"relative RMS {error}");
    }
}
