using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.AudioModels.Tests;

/// <summary>
/// Stable Audio Open 1.0 (the diffusers folders of <c>stabilityai/stable-audio-open-1.0</c>) against diffusers' <c>StableAudioPipeline</c>
/// (<c>tools/audio-reference/reference_stable_audio.py</c>): the prompt's T5 rows, the timing rows, one transformer evaluation, and a 10 s clip with 8 DPM-Solver++ SDE
/// steps and guidance 7 from the reference's initial and step noise (latents and decoded audio).
/// </summary>
public class StableAudioTests
{
    private const string Model = "stable-audio-open-1.0";
    private const string Prompt = "The sound of a hammer hitting a wooden surface, then a short drum loop at 120 BPM";

    private static void RequireAll()
        => AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("sao_noise.f32"), AudioModels.Reference("sao_step_noise.f32"),
            AudioModels.Reference("sao_latent.f32"), AudioModels.Reference("sao_wav.f32"));

    private static AudioGenerationRequest Request()
    {
        var stepNoise = AudioModels.Floats("sao_step_noise.f32");
        int count = 1024 * 64;
        return new AudioGenerationRequest
        {
            Prompt = Prompt,
            Duration = TimeSpan.FromSeconds(10),
            Steps = 8,
            GuidanceScale = 7,
            InitialNoise = AudioModels.Floats("sao_noise.f32"),
            StepNoise = [.. Enumerable.Range(0, stepNoise.Length / count).Select(i => new ReadOnlyMemory<float>(stepNoise, i * count, count))],
        };
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Conditioning_and_one_transformer_evaluation_match_the_reference(bool gpu)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        var ids = ImageGeneration.PromptTokenizer.Tokenize(tokenizer, 128, Prompt, out int tokens);
        Assert.Equal(AudioModels.Ints("sao_ids.i32"), ids[..tokens]);

        using var text = model.CompileComponent(ModelComponentRoles.TextEncoder, TimedAudioDiffusionGraphs.TextGraph);
        var mask = new float[128];
        mask.AsSpan(0, tokens).Fill(1f);
        text.SetInput<int>(text.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.TokensInput), ids);
        text.SetInput<float>(text.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.MaskInput), mask);
        text.Run(new RunParameters());
        var hidden = new float[128 * 768];
        text.GetOutput<float>(TimedAudioDiffusionGraphs.HiddenOutput, hidden);
        hidden.AsSpan(tokens * 768).Clear();
        Report(gpu, "T5 rows", hidden, AudioModels.Floats("sao_text.f32"), 1e-3);

        using var timing = model.CompileComponent(ModelComponentRoles.Encoder, TimedAudioDiffusionGraphs.TimingGraph);
        timing.SetInput<float>(timing.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.SecondsInput), [0f, 10f]);
        timing.Run(new RunParameters());
        var rows = new float[2 * 768];
        timing.GetOutput<float>(TimedAudioDiffusionGraphs.TimingOutput, rows);
        Report(gpu, "timing rows", rows, AudioModels.Floats("sao_timing.f32"), 1e-5);

        using var denoise = model.CompileComponent(ModelComponentRoles.Denoiser, TimedAudioDiffusionGraphs.DenoiseGraph);
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.LatentInput), AudioModels.Floats("sao_dit_in.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.TimestepInput), AudioModels.Floats("sao_dit_t.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.ConditionInput), AudioModels.Floats("sao_dit_cond.f32"));
        denoise.SetInput<float>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.GlobalInput), AudioModels.Floats("sao_dit_global.f32"));
        denoise.SetInput<int>(denoise.Graph.Inputs.First(v => v.Input!.Name == TimedAudioDiffusionGraphs.PositionsInput), [.. Enumerable.Range(0, 1025)]);
        denoise.Run(new RunParameters());
        var prediction = new float[1024 * 64];
        denoise.GetOutput<float>(TimedAudioDiffusionGraphs.PredictionOutput, prediction);
        Report(gpu, "transformer output", prediction, AudioModels.Floats("sao_dit_out.f32"), gpu ? 1e-2 : 2e-3);
    }

    [Theory]
    [InlineData(false, 2e-3)]
    [InlineData(true, 2e-2)]
    public async Task Latents_and_audio_match_the_reference(bool gpu, double tolerance)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        Assert.Equal(44100, generator.SampleRate);
        Assert.Equal(2, generator.Channels);
        Assert.Equal(64, generator.LatentChannels);
        Assert.Equal(100, generator.DefaultSteps);
        Assert.Equal(7f, generator.DefaultGuidanceScale);
        Assert.True(generator.SupportsNegativePrompt);
        Assert.False(generator.SupportsLyrics);
        Assert.Equal(47.55, generator.MaxDuration.TotalSeconds, 2);

        var steps = new List<int>();
        var latents = await generator.GenerateLatentsAsync(Request(), new Progress<(int Step, int Steps)>(p => steps.Add(p.Step)));
        Report(gpu, "final latents", latents.Values, AudioModels.Floats("sao_latent.f32"), tolerance);

        var audio = await generator.GenerateAsync(Request());
        var reference = AudioModels.Floats("sao_wav.f32");
        Assert.Equal(reference.Length, audio.Samples.Length);
        // The client limits the peak to 1; the reference is raw.
        float peak = reference.Max(MathF.Abs);
        var expected = peak > 1 ? [.. reference.Select(v => v / peak)] : reference;
        Report(gpu, "audio", audio.Samples, expected, tolerance * 5);
    }

    [Fact]
    public async Task A_seed_gives_reproducible_audio_with_a_negative_prompt()
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var generator = AudioModels.Client<IAudioGenerator>(model);
        var request = new AudioGenerationRequest { Prompt = Prompt, NegativePrompt = "low quality, noise", Duration = TimeSpan.FromSeconds(1.5), Steps = 2, Seed = 7 };
        var first = await generator.GenerateAsync(request);
        var second = await generator.GenerateAsync(request);
        Assert.Equal((int)(1.5 * 44100), first.FrameCount);
        Assert.Equal(first.Samples, second.Samples);
        var other = await generator.GenerateAsync(request with { NegativePrompt = null });
        Assert.NotEqual(first.Samples, other.Samples);
    }

    [Fact]
    public async Task Package_generates_the_same_audio()
    {
        AudioModels.Require(AudioModels.Model(Model));
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "audio-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"stable-audio-{Guid.NewGuid():N}.slm");
        try
        {
            AudioModels.Services.GetRequiredService<ModelConverter>().Convert(AudioModels.Model(Model), path);
            Assert.Equal(ModelKinds.AudioGeneration, ModelPackageReader.Open(path).Definition.Description.Kind);
            var request = new AudioGenerationRequest { Prompt = Prompt, Duration = TimeSpan.FromSeconds(1), Steps = 2, Seed = 3 };
            using var original = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
            using var packaged = AudioModels.Load(AudioModels.Open(gpu: false), path);
            using var a = AudioModels.Client<IAudioGenerator>(original);
            using var b = AudioModels.Client<IAudioGenerator>(packaged);
            Assert.Equal((await a.GenerateAsync(request)).Samples, (await b.GenerateAsync(request)).Samples);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void Report(bool gpu, string what, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = AudioModels.RelativeRms(actual, expected);
        double snr = -20 * Math.Log10(Math.Max(error, 1e-12));
        ParityReport.Write($"stable-audio gpu={gpu} {what}: relative RMS {error:E2}, SNR {snr:F1} dB");
        Assert.True(error < tolerance, $"{what}: relative RMS {error} (SNR {snr:F1} dB)");
    }
}
