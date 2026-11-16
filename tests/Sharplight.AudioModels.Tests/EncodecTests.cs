using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>EnCodec 24 kHz against transformers' EncodecModel: token-exact codes, decoded audio to a relative RMS of 1e-3.</summary>
public class EncodecTests
{
    private const string Model = "encodec_24khz";

    [Theory]
    [InlineData(false, 8, "6")]
    [InlineData(true, 8, "6")]
    [InlineData(false, 32, "24")]
    [InlineData(true, 32, "24")]
    public async Task Codes_and_audio_match_the_reference(bool gpu, int codebooks, string bandwidth)
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("encodec_in.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var codec = AudioModels.Client<IAudioCodec>(model);
        Assert.Equal(24000, codec.SampleRate);
        Assert.Equal(320, codec.SamplesPerFrame);
        var input = new AudioBuffer(AudioModels.Floats("encodec_in.f32"), 24000);

        var codes = await codec.EncodeAsync(input, new AudioCodecOptions { Codebooks = codebooks });
        var expected = AudioModels.Ints($"encodec_codes_{bandwidth}.i32");
        Assert.Equal(expected.Length, codes.Values.Length);
        int wrong = expected.Where((v, i) => v != codes.Values[i]).Count();

        // Deep residual stages choose between entries whose distances differ by less than the float rounding of the matrix product: allow 0.1 % of the codes there.
        int allowed = codebooks > 8 ? expected.Length / 1000 : 0;
        Assert.True(wrong <= allowed, $"{wrong} of {expected.Length} codes differ");

        // The decoder alone, from the reference codes.
        var reference = AudioModels.Floats($"encodec_dec_{bandwidth}.f32");
        var fromReference = await codec.DecodeAsync(new CodecCodes(codebooks, expected.Length / codebooks, expected));
        Assert.Equal(reference.Length, fromReference.Samples.Length);
        double error = AudioModels.RelativeRms(fromReference.Samples, reference);
        Assert.True(error < 1e-3, $"relative RMS {error}");
    }

    [Fact]
    public async Task Fewer_codebooks_than_encoded_decode_to_the_lower_bit_rate_reconstruction()
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("encodec_in.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var codec = AudioModels.Client<IAudioCodec>(model);
        var codes = await codec.EncodeAsync(new AudioBuffer(AudioModels.Floats("encodec_in.f32"), 24000), new AudioCodecOptions { Codebooks = 32 });
        var truncated = new CodecCodes(8, codes.Frames, codes.Values[..(8 * codes.Frames)]);
        var decoded = await codec.DecodeAsync(truncated);
        double error = AudioModels.RelativeRms(decoded.Samples, AudioModels.Floats("encodec_dec_6.f32"));
        Assert.True(error < 1e-3, $"relative RMS {error}");
    }
}
