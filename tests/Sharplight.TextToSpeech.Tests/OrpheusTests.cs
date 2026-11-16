using System.Runtime.InteropServices;
using Sharplight.Architectures;
using Sharplight.Llm;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Orpheus 3B with the SNAC 24 kHz decoder against transformers and the snac package: the prompt, the logits of the language model along the greedy token
/// chain of transformers (teacher forced), and the decoder's waveform for fixed codes and the noise of the reference run.
/// </summary>
public class OrpheusTests
{
    private static int[] Ints(string directory, string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(directory, name + ".bin"))).ToArray();

    private static float[] Floats(string directory, string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(directory, name + ".bin"))).ToArray();

    private static string[] RequiredFiles()
        => [Path.Combine(TtsModels.Orpheus, "config.json"), Path.Combine(TtsModels.Snac, "pytorch_model.bin"), Path.Combine(TtsModels.OrpheusReference, "sequence.bin"),
            Path.Combine(TtsModels.SnacReference, "audio.bin")];

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoder_matches_the_reference_waveform(bool gpu)
    {
        TtsModels.Require(RequiredFiles());
        using var model = TtsModels.LoadOrpheus(TtsModels.Open(gpu));
        using var synthesizer = (OrpheusSynthesizer)TtsModels.Synthesizer(model);
        var layers = Enumerable.Range(0, 3).Select(i => Ints(TtsModels.SnacReference, "codes" + i)).ToArray();
        var noises = Enumerable.Range(0, 4).Select(i => Floats(TtsModels.SnacReference, "noise" + i)).ToArray();
        int call = 0;
        synthesizer.NoiseSource = _ => noises[call++ % noises.Length];
        int frames = layers[0].Length;
        var codes = new List<int>();
        for (int f = 0; f < frames; f++)
        {
            codes.AddRange([layers[0][f], layers[1][2 * f], layers[2][4 * f], layers[2][(4 * f) + 1], layers[1][(2 * f) + 1], layers[2][(4 * f) + 2], layers[2][(4 * f) + 3]]);
        }

        var audio = synthesizer.Decode(codes);
        var reference = Floats(TtsModels.SnacReference, "audio");
        Assert.Equal(reference.Length, audio.Samples.Length);
        double error = 0;
        double energy = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = audio.Samples[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        ParityReport.Write($"SNAC decoder {(gpu ? "vulkan" : "cpu")}: waveform relative RMS error {Math.Sqrt(error / energy):E2}");
        Assert.True(Math.Sqrt(error / energy) < 1e-4, $"relative RMS error {Math.Sqrt(error / energy):E2}");
    }

    /// <summary>
    /// The logits of the language model at every position of transformers' greedy chain: the argmax agrees unless the reference's own margin is small, and
    /// the reference's top logit is reproduced closely.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Teacher_forced_logits_match_transformers(bool gpu)
    {
        TtsModels.Require(RequiredFiles());
        using var model = TtsModels.LoadOrpheus(TtsModels.Open(gpu));
        using var synthesizer = (OrpheusSynthesizer)TtsModels.Synthesizer(model);
        var info = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(TtsModels.OrpheusReference, "info.json"))).RootElement;
        var sequence = Ints(TtsModels.OrpheusReference, "sequence");
        var values = Floats(TtsModels.OrpheusReference, "top2_values");
        var indices = Ints(TtsModels.OrpheusReference, "top2_ids");
        int promptLength = info.GetProperty("prompt_len").GetInt32();
        Assert.Equal(sequence[..promptLength], synthesizer.PromptFor(info.GetProperty("text").GetString()!, info.GetProperty("voice").GetString()));

        var llm = (LlmModel)synthesizer.GetService(typeof(LlmModel))!;
        using var session = llm.CreateSession();
        var logits = session.Process(sequence[..promptLength]);
        int hard = 0;
        int ties = 0;
        double worst = 0;
        for (int position = promptLength - 1; position < sequence.Length - 1; position++)
        {
            int best = 0;
            for (int t = 1; t < logits.Length; t++)
            {
                if (logits[t] > logits[best])
                {
                    best = t;
                }
            }

            float margin = values[2 * position] - values[(2 * position) + 1];
            if (best != indices[2 * position])
            {
                if (margin > ToleranceMargin)
                {
                    hard++;
                }
                else
                {
                    ties++;
                }
            }

            worst = Math.Max(worst, Math.Abs(logits[indices[2 * position]] - values[2 * position]));
            logits = session.Process([sequence[position + 1]]);
        }

        ParityReport.Write($"Orpheus {(gpu ? "vulkan" : "cpu")}: {hard} argmax disagreements beyond margin {ToleranceMargin}, {ties} within; worst top-logit delta {worst:F3} over {sequence.Length - promptLength} positions");
        Assert.True(hard == 0 && worst < MaxLogitDelta, $"{hard} disagreements beyond margin {ToleranceMargin}, {ties} within; worst logit delta {worst:F4} over {sequence.Length - promptLength} positions");
    }

    [Fact]
    public async Task Text_is_spoken_end_to_end_on_the_gpu()
    {
        TtsModels.Require(RequiredFiles());
        using var model = TtsModels.LoadOrpheus(TtsModels.Open(gpu: true));
        using var synthesizer = TtsModels.Synthesizer(model);
        Assert.Contains("tara", synthesizer.Voices);
        var options = new Sharplight.Speech.SpeechSynthesisOptions { Voice = "tara", Sampling = new SamplingOptions { Temperature = 0.6f, TopP = 0.8f, TopK = 0, RepetitionPenalty = 1.3f, RepetitionWindow = 4096, Seed = 7 } };
        var result = await synthesizer.SynthesizeAsync("Hello there.", options, TestContext.Current.CancellationToken);
        Assert.Equal(24000, result.Audio.SampleRate);
        Assert.InRange(result.Audio.Samples.Length / 24000.0, 0.5, 8.0);
        Assert.Contains(result.Audio.Samples, sample => MathF.Abs(sample) > 0.02f);
        int buffers = 0;
        int samples = 0;
        await foreach (var buffer in synthesizer.SynthesizeStreamingAsync("Hello there.", options, TestContext.Current.CancellationToken))
        {
            buffers++;
            samples += buffer.Samples.Length;
        }

        Assert.True(buffers >= 1 && samples > 12000, $"{buffers} buffers, {samples} samples");
    }

    private const float ToleranceMargin = 0.1f;
    private const double MaxLogitDelta = 0.05;
}
