using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// Stable Audio 3 small-music-base (the checkpoint folder of <c>stabilityai/stable-audio-3-small-music-base</c>) against the stable-audio-3 package
/// (<c>tools/audio-reference/reference_stable_audio_3.py</c>, fp32 on CPU): the prompt's T5Gemma rows, the seconds row, one transformer evaluation, and a 10 s clip with
/// 8 Euler steps and adaptive projected guidance 7 from the reference's initial noise (latents and audio decoded without the autoencoder's dither), plus 8 ping-pong steps
/// (the post-trained models' sampler) on the same weights from the reference's step noise; small-sfx-base and medium-base (differential attention, the sliding-window SAME-L
/// decoder) with the same guided run.
/// </summary>
public class StableAudio3Tests
{
    private const string Model = "stable-audio-3-small-music-base";
    private const string Prompt = "A warm analog synth arpeggio over a slow hip hop drum loop, 90 BPM";
    private const int Frames = 174;
    private const int Channels = 256;

    private static void RequireAll(string tag = "sa3")
        => AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference($"{tag}_noise.f32"), AudioModels.Reference($"{tag}_latent.f32"),
            AudioModels.Reference($"{tag}_wav.f32"));

    private static AudioGenerationRequest Request(string prompt = Prompt) => new() { Prompt = prompt, Duration = TimeSpan.FromSeconds(10) };

    private static AudioGenerationOptions Options(string tag = "sa3", float guidance = 7)
    {
        var options = new AudioGenerationOptions
        {
            Steps = 8,
            GuidanceScale = guidance,
            InitialNoise = AudioModels.Floats($"{tag}_noise.f32"),
            DecoderNoise = 0,
        };
        if (File.Exists(AudioModels.Reference($"{tag}_step_noise.f32")))
        {
            var stepNoise = AudioModels.Floats($"{tag}_step_noise.f32");
            int count = Frames * Channels;
            options = options with { StepNoise = [.. Enumerable.Range(0, stepNoise.Length / count).Select(i => new ReadOnlyMemory<float>(stepNoise, i * count, count))] };
        }

        return options;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Conditioning_and_one_transformer_evaluation_match_the_reference(bool gpu)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        var ids = ImageGeneration.PromptTokenizer.Tokenize(tokenizer, 256, Prompt, out int tokens, "{prompt}");
        Assert.Equal(AudioModels.Ints("sa3_ids.i32"), ids[..tokens]);

        using var text = model.CompileComponent(ModelComponentRoles.TextEncoder, TimedAudioDiffusionGraphs.TextGraph);
        var mask = new float[256];
        mask.AsSpan(0, tokens).Fill(1f);
        text.SetInput<int>(text.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.TokensInput), ids);
        text.SetInput<float>(text.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.MaskInput), mask);
        text.Run(new RunParameters());
        var hidden = new float[256 * 768];
        text.GetOutput<float>(TimedAudioDiffusionGraphs.HiddenOutput, hidden);
        Report(gpu, "T5Gemma rows", hidden, AudioModels.Floats("sa3_text.f32"), gpu ? 1e-2 : 1e-3);

        using var seconds = model.CompileComponent(ModelComponentRoles.Encoder, TimedAudioDiffusionGraphs.TimingGraph);
        seconds.SetInput<float>(seconds.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.SecondsInput), [10f]);
        seconds.Run(new RunParameters());
        var row = new float[768];
        seconds.GetOutput<float>(TimedAudioDiffusionGraphs.TimingOutput, row);
        Report(gpu, "seconds row", row, AudioModels.Floats("sa3_seconds.f32"), 1e-4);

        using var denoise = model.CompileComponent(ModelComponentRoles.Denoiser, TimedAudioDiffusionGraphs.DenoiseGraph);
        var reference = AudioModels.Floats("sa3_dit_valid.f32");
        int rows = Frames + 64;
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.LatentInput), AudioModels.Floats("sa3_dit_x.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.TimestepInput), AudioModels.Floats("sa3_dit_t.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.ConditionInput), AudioModels.Floats("sa3_dit_cond.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.GlobalInput), AudioModels.Floats("sa3_seconds.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.LocalInput), new float[Frames * 257]);
        denoise.SetInput<int>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.PositionsInput),
            [.. Enumerable.Range(0, rows).Select(r => r < Frames ? 64 + r : r - Frames)]);
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.ValidInput), [.. reference, .. Enumerable.Repeat(1f, 64)]);
        var run = new RunParameters();
        run.Symbols[WellKnown.TokenSymbol] = Frames;
        denoise.Run(run);
        var prediction = new float[rows * Channels];
        denoise.GetOutput<float>(TimedAudioDiffusionGraphs.PredictionOutput, prediction);
        Report(gpu, "transformer output", prediction.AsSpan(0, Frames * Channels), AudioModels.Floats("sa3_dit_out.f32"), gpu ? 1e-2 : 1e-3);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void The_decoder_matches_the_reference(bool gpu)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        var client = Assert.IsType<FlowAudioDiffusionClient>(generator);
        var latents = new AudioLatents(Frames, Channels, AudioModels.Floats("sa3_latent.f32"));
        var audio = client.Decode(latents, Request(), Options(), CancellationToken.None);
        Report(gpu, "decoded reference latents", audio.Samples, AudioModels.Floats("sa3_wav.f32"), gpu ? 1e-2 : 1e-3);

        // The autoencoder's own dither moves the audio about as much as the reference's (34.7 dB there).
        var dithered = client.Decode(latents, Request(), Options() with { DecoderNoise = null, Seed = 5 }, CancellationToken.None);
        double error = AudioModels.RelativeRms(dithered.Samples, audio.Samples);
        ParityReport.Write($"stable-audio-3 gpu={gpu} decoder dither: SNR {-20 * Math.Log10(error):F1} dB (reference dither {Snr("sa3_wav_dither.f32", "sa3_wav.f32"):F1} dB)");
        Assert.InRange(-20 * Math.Log10(error), 25, 45);
    }

    [Theory]
    [InlineData(false, 1e-2)]
    [InlineData(true, 5e-2)]
    public async Task Latents_and_audio_match_the_reference(bool gpu, double tolerance)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        Assert.Equal(44100, generator.SampleRate);
        Assert.Equal(2, generator.Channels);
        Assert.Equal(Channels, generator.LatentChannels);
        Assert.Equal(50, generator.Defaults.Steps);
        Assert.Equal(7f, generator.Defaults.GuidanceScale);
        Assert.True(generator.Defaults.SupportsNegativePrompt);
        Assert.Equal(1300 * 4096 / 44100.0, generator.MaxDuration.TotalSeconds, 3);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        var latents = await generator.GenerateLatentsAsync(Request(), Options());
        ParityReport.Write($"stable-audio-3 gpu={gpu} 8 Euler steps with APG in {watch.Elapsed.TotalSeconds:F1} s");
        Assert.Equal(Frames, latents.Frames);
        Report(gpu, "final latents", latents.Values, AudioModels.Floats("sa3_latent.f32"), tolerance);

        var audio = (await generator.GenerateAsync(Request(), Options())).Audio;
        Report(gpu, "audio", audio.Samples, AudioModels.Floats("sa3_wav.f32"), tolerance * 2);
    }

    [Theory]
    [InlineData(false, 1e-2)]
    [InlineData(true, 5e-2)]
    public async Task Ping_pong_steps_match_the_reference(bool gpu, double tolerance)
    {
        RequireAll("sa3pp");
        AudioModels.Require(AudioModels.Reference("sa3pp_step_noise.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        var client = Assert.IsType<FlowAudioDiffusionClient>(generator);
        var latents = await client.GenerateLatentsAsync(Request(), Options("sa3pp", 1f), pingPong: true);
        Report(gpu, "ping-pong latents", latents.Values, AudioModels.Floats("sa3pp_latent.f32"), tolerance);
    }

    [Theory]
    [InlineData("stable-audio-3-small-sfx-base", "sa3sfx", "Heavy rain on a tin roof with distant rolling thunder", 1300, false, 1e-2)]
    [InlineData("stable-audio-3-small-sfx-base", "sa3sfx", "Heavy rain on a tin roof with distant rolling thunder", 1300, true, 5e-2)]
    [InlineData("stable-audio-3-medium-base", "sa3m", Prompt, 4096, false, 1e-2)]
    [InlineData("stable-audio-3-medium-base", "sa3m", Prompt, 4096, true, 5e-2)]
    public async Task The_other_base_models_match_the_reference(string name, string tag, string prompt, int maxFrames, bool gpu, double tolerance)
    {
        AudioModels.Require(AudioModels.Model(name), AudioModels.Reference($"{tag}_noise.f32"), AudioModels.Reference($"{tag}_latent.f32"),
            AudioModels.Reference($"{tag}_wav.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(name));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        Assert.Equal(maxFrames * 4096 / 44100.0, generator.MaxDuration.TotalSeconds, 3);
        var request = Request(prompt);
        var options = Options(tag);
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var latents = await generator.GenerateLatentsAsync(request, options);
        ParityReport.Write($"{name} gpu={gpu} 8 Euler steps with APG in {watch.Elapsed.TotalSeconds:F1} s");
        Report(gpu, $"{name} final latents", latents.Values, AudioModels.Floats($"{tag}_latent.f32"), tolerance);
        var client = Assert.IsType<FlowAudioDiffusionClient>(generator);
        Report(gpu, $"{name} audio", client.Decode(latents, request, options, CancellationToken.None).Samples, AudioModels.Floats($"{tag}_wav.f32"), tolerance * 2);
        Report(gpu, $"{name} decoded reference latents", client.Decode(new AudioLatents(Frames, Channels, AudioModels.Floats($"{tag}_latent.f32")), request,
            options, CancellationToken.None).Samples, AudioModels.Floats($"{tag}_wav.f32"), gpu ? 1e-2 : 1e-3);
    }

    [Fact]
    public async Task A_seed_gives_reproducible_audio_with_a_negative_prompt()
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        var request = new AudioGenerationRequest { Prompt = Prompt, NegativePrompt = "low quality, noise", Duration = TimeSpan.FromSeconds(1.5) };
        var options = new AudioGenerationOptions { Steps = 2, Seed = 7 };
        var first = (await generator.GenerateAsync(request, options)).Audio;
        var second = (await generator.GenerateAsync(request, options)).Audio;
        Assert.Equal((int)(1.5 * 44100), first.FrameCount);
        Assert.Equal(first.Samples, second.Samples);
        var other = (await generator.GenerateAsync(request with { NegativePrompt = null }, options)).Audio;
        Assert.NotEqual(first.Samples, other.Samples);
    }

    [Fact]
    public async Task Package_generates_the_same_audio()
    {
        AudioModels.Require(AudioModels.Model(Model));
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "audio-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"stable-audio-3-{Guid.NewGuid():N}.slm");
        try
        {
            AudioModels.Services.GetRequiredService<ModelConverter>().Convert(AudioModels.Model(Model), path);
            Assert.Equal(ModelKinds.AudioGeneration, ModelPackageReader.Open(path).Definition.Description.Kind);
            var request = new AudioGenerationRequest { Prompt = Prompt, Duration = TimeSpan.FromSeconds(1) };
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

    private static double Snr(string actual, string expected) => -20 * Math.Log10(AudioModels.RelativeRms(AudioModels.Floats(actual), AudioModels.Floats(expected)));

    private static void Report(bool gpu, string what, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = AudioModels.RelativeRms(actual, expected);
        double snr = -20 * Math.Log10(Math.Max(error, 1e-12));
        ParityReport.Write($"stable-audio-3 gpu={gpu} {what}: relative RMS {error:E2}, SNR {snr:F1} dB");
        Assert.True(error < tolerance, $"{what}: relative RMS {error} (SNR {snr:F1} dB)");
    }
}
