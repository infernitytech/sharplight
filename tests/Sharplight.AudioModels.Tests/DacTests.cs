using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>DAC 44 kHz against transformers' DacModel: token-exact codes, decoded audio to a relative RMS of 1e-3.</summary>
public class DacTests
{
    private const string Model = "dac_44khz";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Codes_and_audio_match_the_reference(bool gpu)
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("dac_in.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var codec = AudioModels.Client<IAudioCodec>(model);
        Assert.Equal(44100, codec.SampleRate);
        Assert.Equal(512, codec.SamplesPerFrame);
        Assert.Equal(9, codec.CodebookCount);
        var input = new AudioBuffer(AudioModels.Floats("dac_in.f32"), 44100);

        var codes = await codec.EncodeAsync(input);
        var expected = AudioModels.Ints("dac_codes.i32");
        Assert.Equal(expected.Length, codes.Values.Length);
        int wrong = expected.Where((v, i) => v != codes.Values[i]).Count();
        Assert.True(wrong == 0, $"{wrong} of {expected.Length} codes differ");

        var decoded = await codec.DecodeAsync(codes);
        var reference = AudioModels.Floats("dac_dec.f32");
        Assert.Equal(reference.Length, decoded.Samples.Length);
        double error = AudioModels.RelativeRms(decoded.Samples, reference);
        Assert.True(error < 1e-3, $"relative RMS {error}");
    }
}
