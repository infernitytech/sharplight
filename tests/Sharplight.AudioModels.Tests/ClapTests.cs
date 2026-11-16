using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>CLAP (laion/clap-htsat-unfused) against transformers' ClapModel: log-mel features, audio and text embeddings and zero-shot scores.</summary>
public class ClapTests
{
    private const string Model = "clap-htsat-unfused";
    private static readonly string[] s_labels = ["a person speaking", "a dog barking", "music playing", "a car engine", "silence"];

    private static void RequireAll()
        => AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("clap_in.f32"), AudioModels.Reference("clap_audio_embed.f32"), AudioModels.Reference("clap_text_embed.f32"));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Audio_and_text_embeddings_match_the_reference(bool gpu)
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var embedder = AudioModels.Client<IAudioEmbedder>(model);
        Assert.Equal(512, embedder.Dimensions);
        Assert.Equal(48000, embedder.SampleRate);
        var input = new AudioBuffer(AudioModels.Floats("clap_in.f32"), 48000);

        var audio = await embedder.EmbedAudioAsync(input, new AudioEmbeddingOptions());
        double audioError = AudioModels.RelativeRms(audio, AudioModels.Floats("clap_audio_embed.f32"));
        Assert.True(audioError < 2e-3, $"audio embedding relative RMS {audioError}");

        var expected = AudioModels.Floats("clap_text_embed.f32");
        for (int i = 0; i < s_labels.Length; i++)
        {
            var text = await embedder.EmbedTextAsync(s_labels[i], new AudioEmbeddingOptions());
            double error = AudioModels.RelativeRms(text, expected.AsSpan(i * 512, 512));
            Assert.True(error < 2e-3, $"text embedding '{s_labels[i]}' relative RMS {error}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Zero_shot_scores_match_the_reference(bool gpu)
    {
        RequireAll();
        AudioModels.Require(AudioModels.Reference("clap_logits.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var embedder = AudioModels.Client<IAudioEmbedder>(model);
        var result = await embedder.ClassifyZeroShotAsync(new AudioBuffer(AudioModels.Floats("clap_in.f32"), 48000), s_labels);
        var logits = AudioModels.Floats("clap_logits.f32");
        Assert.Equal(Array.IndexOf(logits, logits.Max()), result.Top!.Index);
        foreach (var score in result.Predictions)
        {
            Assert.True(Math.Abs(score.Logit - logits[score.Index]) < 0.05f, $"{score.Label}: logit {score.Logit} vs {logits[score.Index]}");
        }
    }

    [Fact]
    public async Task Short_audio_is_repeated_and_long_audio_can_be_averaged()
    {
        RequireAll();
        using var model = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model(Model));
        using var embedder = AudioModels.Client<IAudioEmbedder>(model);
        var samples = AudioModels.Floats("clap_in.f32");
        var first = await embedder.EmbedAudioAsync(new AudioBuffer(samples, 48000));
        var twice = new float[samples.Length * 2];
        samples.CopyTo(twice, 0);
        samples.CopyTo(twice, samples.Length);
        var averaged = await embedder.EmbedAudioAsync(new AudioBuffer(twice, 48000), new AudioEmbeddingOptions { LongAudio = LongAudioMode.Average });
        Assert.True(AudioModels.RelativeRms(averaged, first) < 1e-4);
        var half = await embedder.EmbedAudioAsync(new AudioBuffer(samples[..(samples.Length / 2)], 48000));
        Assert.Equal(512, half.Length);
        Assert.True(Math.Abs(half.Sum(v => v * v) - 1.0) < 1e-3);
    }
}
