using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// The Higgs Audio v2 tokenizer (the <c>audio_tokenizer</c> folder of OmniVoice) against transformers' <c>HiggsAudioV2TokenizerModel</c> (reference from
/// <c>tools/audio-reference/reference_higgs_audio_tokenizer.py</c>): token-exact codes of the 8 codebooks (the encoder: resampling, HuBERT, the semantic and acoustic encoders and
/// the quantizer) and decoded audio to a relative RMS of 1e-4, on 10 s of jfk.wav.
/// </summary>
public class HiggsAudioCodecTests
{
    private const string Model = "higgs-audio-v2-tokenizer";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Codes_and_audio_match_the_reference(bool gpu)
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("higgs_codes.i32"), AudioModels.Reference("mimi_in.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var codec = AudioModels.Client<IAudioCodec>(model);
        Assert.Equal(24000, codec.SampleRate);
        Assert.Equal(960, codec.SamplesPerFrame);
        Assert.Equal(8, codec.CodebookCount);
        var input = new AudioBuffer(AudioModels.Floats("mimi_in.f32"), 24000);

        var codes = await codec.EncodeAsync(input);
        var expected = AudioModels.Ints("higgs_codes.i32");
        Assert.Equal(expected.Length, codes.Values.Length);
        int wrong = expected.Where((v, i) => v != codes.Values[i]).Count();
        ParityReport.Write($"Higgs Audio tokenizer {(gpu ? "vulkan" : "cpu")}: {wrong} of {expected.Length} codes differ");
        Assert.True(wrong == 0, $"{wrong} of {expected.Length} codes differ");

        var decoded = await codec.DecodeAsync(codes);
        var reference = AudioModels.Floats("higgs_dec.f32");
        Assert.Equal(reference.Length, decoded.Samples.Length);
        double error = AudioModels.RelativeRms(decoded.Samples, reference);
        ParityReport.Write($"Higgs Audio tokenizer {(gpu ? "vulkan" : "cpu")}: decoded audio relative RMS {error:E2}");
        Assert.True(error < 1e-4, $"relative RMS {error}");
    }
}
