using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// CosyVoice 3 (Fun-CosyVoice3-0.5B) against FunAudioLLM's CosyVoice code and onnxruntime (reference tensors from
/// <c>tools/make_cosyvoice3_reference.py</c>, float32 on the CPU, jfk.wav as the recording): the ONNX S3 tokenizer v3 and CAM++ on the reference's
/// inputs, the prompt log-mel, the language model teacher forced along the reference's greedy chain, the flow with the reference's noise, the causal
/// HiFT with the reference's source noise. <c>SL_COSYVOICE3</c> and <c>SL_COSYVOICE3_REF</c> override the model and reference folders.
/// </summary>
public class CosyVoiceTests
{
    private static string Model => Environment.GetEnvironmentVariable("SL_COSYVOICE3") is { Length: > 0 } path
        ? path
        : Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "Fun-CosyVoice3-0.5B");

    private static string Reference => Environment.GetEnvironmentVariable("SL_COSYVOICE3_REF") is { Length: > 0 } path
        ? path
        : Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "cosyvoice3-ref");

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Reference, name))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Reference, name))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Reference, "cosyvoice3.json"))).RootElement;

    private static void RequireFiles() => TtsModels.Require(Path.Combine(Model, "llm.pt"), Path.Combine(Model, "campplus.onnx"), Path.Combine(Reference, "cosyvoice3.json"));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Onnx_encoders_match_onnxruntime(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var encoders = new S3SpeechEncoders(model);
        var info = Info();
        string device = gpu ? "vulkan" : "cpu";
        var tokens = encoders.SpeechTokens(Floats("tok_mel.f32"), info.GetProperty("tok_frames").GetInt32());
        var reference = Ints("tokens.i32");
        int agree = tokens.Zip(reference).Count(p => p.First == p.Second);
        double xvectorError = RelativeError(Floats("xvector.f32"), encoders.XVector(Floats("fbank.f32"), info.GetProperty("fbank_frames").GetInt32()));
        var mel = encoders.PromptMel(Floats("ref24.f32"), out int frames);
        var referenceMel = Floats("prompt_feat.f32");
        double melError = RelativeError(referenceMel, mel[..referenceMel.Length]);
        ParityReport.Write($"CosyVoice 3 encoders {device}: S3 tokenizer v3 {agree} of {reference.Length} tokens identical "
            + $"({info.GetProperty("all_tokens").GetInt32()} before the cut), CAM++ x-vector {xvectorError:E2}, prompt log-mel {melError:E2} ({frames} frames)");
        Assert.Equal(info.GetProperty("all_tokens").GetInt32(), tokens.Length);
        Assert.True(agree >= reference.Length - 1, $"{agree} of {reference.Length} speech tokens identical");
        Assert.True(xvectorError < 1e-4, $"x-vector relative RMS error {xvectorError:E2}");
        Assert.True(melError < 1e-4, $"prompt mel relative RMS error {melError:E2}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Language_model_matches_the_teacher_forced_greedy_chain(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = new CosyVoiceSynthesizer(model);
        var info = Info();
        Assert.Equal(Ints("text_tokens.i32"), synthesizer.TextTokens(info.GetProperty("text").GetString()!));
        Assert.Equal(Ints("prompt_text_tokens.i32"), synthesizer.TextTokens(info.GetProperty("prompt_text").GetString()!));
        var greedy = Ints("greedy.i32");
        var referenceLogits = Floats("greedy_logits.f32");
        int steps = info.GetProperty("greedy_steps").GetInt32();
        int vocabulary = referenceLogits.Length / steps;
        var observed = new List<float[]>();
        synthesizer.SpeechTokens(Ints("text_tokens.i32"), Ints("prompt_text_tokens.i32"), Ints("tokens.i32"), SamplingOptions.Greedy,
            TestContext.Current.CancellationToken, logits => observed.Add((float[])logits.Clone()), greedy);
        int compared = Math.Min(observed.Count, steps);
        int disagreements = 0;
        double worst = 0;
        double minMargin = double.PositiveInfinity;
        for (int s = 0; s < compared; s++)
        {
            var reference = referenceLogits.AsSpan(s * vocabulary, vocabulary);
            var actual = observed[s];
            worst = Math.Max(worst, reference.ToArray().Zip(actual, (r, a) => Math.Abs(r - a)).Max());
            int referenceTop = ArgMax(reference);
            if (referenceTop != ArgMax(actual))
            {
                disagreements++;
                minMargin = Math.Min(minMargin, Margin(reference));
            }
        }

        ParityReport.Write($"CosyVoice 3 language model {(gpu ? "vulkan" : "cpu")}: teacher forced over {compared} steps, {disagreements} argmax "
            + $"disagreements, worst logit delta {worst:E2}");
        Assert.True(compared >= greedy.Length, $"{compared} steps compared, {greedy.Length} expected");
        Assert.True(disagreements == 0 || minMargin < 0.05, $"{disagreements} argmax disagreements (reference margin {minMargin:E2})");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Flow_matches_the_reference_with_its_noise(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = new CosyVoiceSynthesizer(model);
        var tokens = Ints("tokens.i32");
        var voice = new CosyVoiceVoice(tokens, Floats("prompt_feat.f32"), Floats("xvector.f32"));
        var mel = synthesizer.Mel(Ints("speech.i32"), voice, Floats("flow_noise.f32"));
        double error = RelativeError(Floats("flow_mel.f32"), mel);
        ParityReport.Write($"CosyVoice 3 flow {(gpu ? "vulkan" : "cpu")}: mel of {Ints("speech.i32").Length} tokens after {tokens.Length} prompt tokens, "
            + $"relative RMS {error:E2}");
        Assert.True(error < (gpu ? 1e-3 : 1e-4), $"flow mel relative RMS error {error:E2}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Vocoder_matches_the_reference_with_its_source_noise(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = new CosyVoiceSynthesizer(model);
        var (samples, f0) = synthesizer.Vocode(Floats("flow_mel.f32"), Floats("hift_noise.f32"));
        double f0Error = RelativeError(Floats("hift_f0.f32"), f0);
        double error = RelativeError(Floats("wav.f32"), samples);
        ParityReport.Write($"CosyVoice 3 causal HiFT {(gpu ? "vulkan" : "cpu")}: F0 relative RMS {f0Error:E2}, waveform {error:E2} ({samples.Length} samples)");
        Assert.True(f0Error < 1e-4, $"F0 relative RMS error {f0Error:E2}");
        Assert.True(error < 1e-2, $"waveform relative RMS error {error:E2}");
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static double Margin(ReadOnlySpan<float> values)
    {
        var sorted = values.ToArray().OrderDescending().ToArray();
        return sorted[0] - sorted[1];
    }

    private static double RelativeError(float[] reference, float[] actual)
    {
        Assert.Equal(reference.Length, actual.Length);
        double error = 0;
        double energy = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        return Math.Sqrt(error / Math.Max(energy, 1e-30));
    }
}
