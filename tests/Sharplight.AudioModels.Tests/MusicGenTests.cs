using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>MusicGen small against transformers' MusicgenForConditionalGeneration: greedy codes with guidance 3 token by token, the decoded audio, the streaming path.</summary>
public class MusicGenTests
{
    private const string Model = "musicgen-small";
    private const string Prompt = "80s pop track with bassy drums and synth";

    private static void RequireAll()
        => AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("musicgen_codes.i32"), AudioModels.Reference("musicgen_audio.f32"));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Greedy_codes_and_audio_match_the_reference(bool gpu)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IMusicGenerator>(model);
        Assert.Equal(32000, generator.SampleRate);
        var expected = AudioModels.Ints("musicgen_codes.i32");
        int frames = expected.Length / 4;
        var request = new MusicGenerationRequest { Prompt = Prompt, Duration = TimeSpan.FromSeconds(frames / generator.FrameRate) };
        var options = new MusicGenerationOptions { Greedy = true };

        var codes = await generator.GenerateCodesAsync(request, options);
        Assert.Equal(frames, codes.Frames);
        int wrong = expected.Where((v, i) => v != codes.Values[i]).Count();
        Assert.True(wrong == 0, $"{wrong} of {expected.Length} codes differ");

        var audio = (await generator.GenerateAsync(request, options)).Audio;
        var reference = AudioModels.Floats("musicgen_audio.f32");
        Assert.Equal(reference.Length, audio.Samples.Length);
        double error = AudioModels.RelativeRms(audio.Samples, reference);
        Assert.True(error < 2e-3, $"relative RMS {error}");
    }

    [Fact]
    public async Task Streaming_pieces_add_up_to_the_music()
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IMusicGenerator>(model);
        var expected = AudioModels.Ints("musicgen_codes.i32");
        int frames = expected.Length / 4;
        var request = new MusicGenerationRequest { Prompt = Prompt, Duration = TimeSpan.FromSeconds(frames / generator.FrameRate) };
        var samples = new List<float>();
        int pieces = 0;
        AudioGenerationResult? result = null;
        await foreach (var update in generator.GenerateStreamingAsync(request, new MusicGenerationOptions { Greedy = true }))
        {
            if (update.Audio is { } piece)
            {
                samples.AddRange(piece.Samples);
                pieces++;
            }

            result = update.Result ?? result;
        }

        Assert.NotNull(result);
        Assert.Equal(samples.Count, result.Audio.Samples.Length);

        var reference = AudioModels.Floats("musicgen_audio.f32");
        Assert.Equal(reference.Length, samples.Count);
        Assert.True(pieces >= 1);
        double error = AudioModels.RelativeRms(samples.ToArray(), reference);
        Assert.True(error < 0.2, $"relative RMS {error}");
    }

    [Fact]
    public async Task Sampling_is_reproducible_with_a_seed()
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IMusicGenerator>(model);
        var request = new MusicGenerationRequest { Prompt = Prompt, Duration = TimeSpan.FromSeconds(0.5) };
        var options = new MusicGenerationOptions { Seed = 7 };
        var a = await generator.GenerateCodesAsync(request, options);
        var b = await generator.GenerateCodesAsync(request, options);
        Assert.Equal(a.Values, b.Values);
        var c = await generator.GenerateCodesAsync(request, options with { Seed = 8 });
        Assert.NotEqual(a.Values, c.Values);
    }
}
