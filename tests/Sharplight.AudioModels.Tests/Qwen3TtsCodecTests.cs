using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// The Qwen3-TTS 12 Hz tokenizer against the official qwen_tts modeling code (reference from <c>tools/audio-reference/reference_qwen3_tts_tokenizer.py</c>): token-exact codes of the
/// 16 codebooks and decoded audio to a relative RMS of 1e-3, on 5 s of jfk.wav.
/// </summary>
public class Qwen3TtsCodecTests
{
    private const string Model = "qwen3-tts-tokenizer-12hz";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Codes_and_audio_match_the_reference(bool gpu)
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("qtok_codes.i32"), AudioModels.Reference("mimi_in.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var codec = AudioModels.Client<IAudioCodec>(model);
        Assert.Equal(24000, codec.SampleRate);
        Assert.Equal(1920, codec.SamplesPerFrame);
        var input = new AudioBuffer(AudioModels.Floats("mimi_in.f32"), 24000);

        var codes = await codec.EncodeAsync(input, new AudioCodecOptions { Codebooks = 16 });
        var expected = AudioModels.Ints("qtok_codes.i32");
        Assert.Equal(expected.Length, codes.Values.Length);
        int wrong = expected.Where((v, i) => v != codes.Values[i]).Count();
        Assert.True(wrong == 0, $"{wrong} of {expected.Length} codes differ");

        var decoded = await codec.DecodeAsync(codes);
        var reference = AudioModels.Floats("qtok_dec.f32");
        Assert.Equal(reference.Length, decoded.Samples.Length);
        double error = AudioModels.RelativeRms(decoded.Samples, reference);
        Assert.True(error < 1e-3, $"relative RMS {error}");
    }
}
