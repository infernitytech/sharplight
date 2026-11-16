using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Chatterbox (T3 + S3Gen + HiFT, built-in voice) against the chatterbox-tts package (reference tensors from <c>tools/make_chatterbox_reference.py</c>,
/// float32 on the CPU): the text tokens, T3's conditioning rows, T3's guided greedy chain teacher forced (the best token agrees wherever the
/// reference's margin is not tiny), the flow's mel for the reference tokens and starting noise, and HiFT's F0 and waveform for the reference mel
/// with the reference's random draws, on the CPU and on Vulkan.
/// </summary>
public class ChatterboxTests
{
    private static string Model => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "chatterbox");

    private static string Reference => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "chatterbox-ref");

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Reference, name + ".bin"))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Reference, name + ".bin"))).ToArray();

    private static void RequireAll() => TtsModels.Require(Path.Combine(Model, "t3_cfg.safetensors"), Path.Combine(Model, "conds.pt"), Path.Combine(Reference, "audio.bin"));

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Reference, "chatterbox.json"))).RootElement;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Text_tokens_and_conditioning_match_the_reference(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = (ChatterboxSynthesizer)TtsModels.Synthesizer(model);
        Assert.Equal(Ints("text_tokens"), synthesizer.TextTokens(Info().GetProperty("text").GetString()!));
        var reference = Floats("cond");
        var rows = synthesizer.ConditionRows(synthesizer.LoadVoice("default"), ChatterboxSynthesizer.DefaultExaggeration);
        Assert.Equal(reference.Length, rows.Length);
        double error = RelativeError(reference, rows);
        ParityReport.Write($"Chatterbox conditioning {(gpu ? "vulkan" : "cpu")}: relative RMS error {error:E2}");
        Assert.True(error < 1e-4, $"conditioning relative RMS error {error:E2}");
    }

    /// <summary>T3 teacher forced along the reference's guided greedy chain: the best token agrees at every step where the reference's margin exceeds 0.05.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void T3_greedy_chain_matches_the_reference(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = (ChatterboxSynthesizer)TtsModels.Synthesizer(model);
        var reference = Ints("speech_tokens");
        var margins = Floats("margins");
        var chosen = new List<int>();
        var condition = synthesizer.ConditionRows(synthesizer.LoadVoice("default"), ChatterboxSynthesizer.DefaultExaggeration);
        var greedy = new SamplingOptions { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 1.2f };
        synthesizer.SpeechTokens(Ints("text_tokens"), condition, greedy, TestContext.Current.CancellationToken, (token, _) => chosen.Add(token), reference);
        Assert.Equal(reference.Length, chosen.Count);
        int disagreements = 0;
        int close = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (chosen[i] != reference[i])
            {
                close += margins[i] <= 0.05f ? 1 : 0;
                disagreements += margins[i] > 0.05f ? 1 : 0;
            }
        }

        ParityReport.Write($"Chatterbox T3 {(gpu ? "vulkan" : "cpu")}: {disagreements} argmax disagreements beyond a 0.05 margin, {close} within it, over {reference.Length} steps");
        Assert.Equal(0, disagreements);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Flow_mel_matches_the_reference(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = (ChatterboxSynthesizer)TtsModels.Synthesizer(model);
        var noise = Transpose(Floats("flow_noise"), 80);
        var mel = synthesizer.Mel(Ints("speech_tokens"), synthesizer.LoadVoice("default"), noise);
        var reference = Transpose(Floats("mel"), 80);
        Assert.Equal(reference.Length, mel.Length);
        double error = RelativeError(reference, mel);
        ParityReport.Write($"Chatterbox flow {(gpu ? "vulkan" : "cpu")}: mel relative RMS error {error:E2}");
        Assert.True(error < 1e-3, $"mel relative RMS error {error:E2}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Vocoder_matches_the_reference(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = (ChatterboxSynthesizer)TtsModels.Synthesizer(model);
        var mel = Transpose(Floats("mel"), 80);
        var (samples, f0) = synthesizer.Vocode(mel, Floats("source_phase"), Transpose(Floats("source_noise"), 9));
        var referenceF0 = Floats("f0");
        var reference = Floats("audio");
        Assert.Equal(reference.Length, samples.Length);
        double f0Error = RelativeError(referenceF0, f0);
        double error = RelativeError(reference, samples);
        ParityReport.Write($"Chatterbox HiFT {(gpu ? "vulkan" : "cpu")}: F0 relative RMS error {f0Error:E2}, waveform {error:E2}");
        Assert.True(f0Error < 1e-4, $"F0 relative RMS error {f0Error:E2}");
        Assert.True(error < 3e-2, $"waveform relative RMS error {error:E2}");
    }

    /// <summary>The whole pipeline through the task client (greedy tokens): audio of plausible length, and the real-time factor in the report.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Synthesizes_speech_end_to_end(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = TtsModels.Synthesizer(model);
        var options = new Sharplight.Speech.SpeechSynthesisOptions { Sampling = new SamplingOptions { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 1.2f } };
        string text = Info().GetProperty("text").GetString()!;
        await synthesizer.SynthesizeAsync(text, options, TestContext.Current.CancellationToken);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var result = await synthesizer.SynthesizeAsync(text, options, TestContext.Current.CancellationToken);
        double seconds = result.Audio.Samples.Length / (double)result.Audio.SampleRate;
        ParityReport.Write($"Chatterbox end to end {(gpu ? "vulkan" : "cpu")}: {seconds:F2} s of audio in {stopwatch.Elapsed.TotalSeconds:F2} s (real-time factor {stopwatch.Elapsed.TotalSeconds / seconds:F2})");
        Assert.InRange(seconds, 1.5, 15);
        Assert.Contains(result.Audio.Samples, v => MathF.Abs(v) > 0.05f);
    }

    // [channels, frames] → [frames, channels].
    private static float[] Transpose(float[] values, int channels)
    {
        int frames = values.Length / channels;
        var result = new float[values.Length];
        for (int c = 0; c < channels; c++)
        {
            for (int t = 0; t < frames; t++)
            {
                result[(t * channels) + c] = values[(c * frames) + t];
            }
        }

        return result;
    }

    private static double RelativeError(float[] reference, float[] actual)
    {
        double error = 0;
        double energy = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        return Math.Sqrt(error / energy);
    }
}
