using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// F5-TTS v1 Base with Vocos against the f5-tts package (reference tensors from <c>tools/make_f5tts_reference.py</c>, float32 on the CPU, jfk.wav as
/// the reference recording): the character ids and duration, the reference log-mel (the resamplers differ: 16 kHz to 24 kHz), the sampled mel for the
/// reference's own log-mel condition and starting noise (32 guided Euler steps) and Vocos' waveform, on the CPU and on Vulkan.
/// </summary>
public class F5TtsTests
{
    private static string Root => Path.GetDirectoryName(TtsModels.Kokoro)!;

    private static string Model => Path.Combine(Root, "F5-TTS", "F5TTS_v1_Base");

    private static string Vocos => Path.Combine(Root, "vocos-mel-24khz");

    private static string Reference => Path.Combine(Root, "f5tts-ref");

    private static string Jfk => Path.Combine(Root, "..", "audio", "jfk.wav");

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Reference, name + ".bin"))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Reference, name + ".bin"))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Reference, "f5tts.json"))).RootElement;

    private static TaskModel Load(bool gpu)
    {
        TtsModels.Require(Path.Combine(Model, "vocab.txt"), Path.Combine(Vocos, "pytorch_model.bin"), Path.Combine(Reference, "audio.bin"), Jfk);
        var options = new TaskLoadOptions { ModelPath = Model, Devices = [TtsModels.Open(gpu)] };
        options.Auxiliary[AuxiliaryRoles.Vocoder] = Vocos;
        return TaskModel.Load(TtsModels.Services.GetRequiredService<ModelLoader>(), options);
    }

    private static SpeechSynthesisOptions Request() => new()
    {
        ReferenceAudio = WavFile.Read(Jfk),
        ReferenceText = Info().GetProperty("reference_text").GetString(),
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Text_and_duration_match_the_reference(bool gpu)
    {
        using var model = Load(gpu);
        using var synthesizer = (F5Synthesizer)TtsModels.Synthesizer(model);
        var (text, cond, duration, _, _) = synthesizer.Prepare(Info().GetProperty("text").GetString()!, Request());
        Assert.Equal(Ints("text"), text);
        Assert.Equal(Info().GetProperty("duration").GetInt32(), duration);
        Assert.Equal(Floats("cond").Length, cond.Length);
    }

    /// <summary>
    /// The reference recording's log-mel, on a 24 kHz copy (<c>reference24.wav</c>, resampled by the tool) so that neither side resamples: the log floor
    /// of 1e-5 turns the resamplers' differences in near-silent frames into large log differences.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Reference_mel_matches_the_reference(bool gpu)
    {
        TtsModels.Require(Path.Combine(Reference, "reference24.wav"), Path.Combine(Reference, "cond24.bin"));
        using var model = Load(gpu);
        using var synthesizer = (F5Synthesizer)TtsModels.Synthesizer(model);
        var request = Request() with { ReferenceAudio = WavFile.Read(Path.Combine(Reference, "reference24.wav")) };
        var (_, cond, _, _, _) = synthesizer.Prepare(Info().GetProperty("text").GetString()!, request);
        var reference = Floats("cond24");
        Assert.Equal(reference.Length, cond.Length);
        double error = RelativeError(reference, cond);
        ParityReport.Write($"F5-TTS reference mel {(gpu ? "vulkan" : "cpu")}: relative RMS error {error:E2}, max abs error {MaxAbs(reference, cond):E2}");
        // Bins near the 1e-5 floor turn float differences of the magnitude into log differences up to 1 (5e-3 relative RMS overall).
        Assert.True(error < 1e-2, $"reference mel relative RMS error {error:E2}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Sampled_mel_and_waveform_match_the_reference(bool gpu)
    {
        using var model = Load(gpu);
        using var synthesizer = (F5Synthesizer)TtsModels.Synthesizer(model);
        var (text, _, duration, frames, rms) = synthesizer.Prepare(Info().GetProperty("text").GetString()!, Request());
        var mel = synthesizer.Sample(text, Floats("cond"), duration, Floats("noise"), TestContext.Current.CancellationToken);
        var referenceMel = Floats("mel");
        Assert.Equal(referenceMel.Length, mel.Length);
        double melError = RelativeError(referenceMel[(frames * 100)..], mel[(frames * 100)..]);
        var samples = synthesizer.Vocode(referenceMel[(frames * 100)..]);
        float gain = rms < 0.1f ? rms / 0.1f : 1f;
        samples = [.. samples.Select(v => v * gain)];
        var audio = Floats("audio");
        double audioError = RelativeError(audio, samples[..Math.Min(samples.Length, audio.Length)]);
        ParityReport.Write($"F5-TTS {(gpu ? "vulkan" : "cpu")}: sampled mel relative RMS error {melError:E2}, Vocos waveform {audioError:E2}");
        Assert.True(melError < 1e-2, $"mel relative RMS error {melError:E2}");
        Assert.True(audioError < 1e-3, $"waveform relative RMS error {audioError:E2}");
    }

    /// <summary>
    /// The first step's field (conditional and without reference and text) for the reference's inputs. Debug only: the <c>dbg_v0*</c> dumps come from an
    /// ad-hoc script outside the reference tool and do not match its inputs; the sampled-mel test covers the field through all 32 steps.
    /// </summary>
    [Theory(Skip = "Debug only: needs dbg_v0 dumps made from the same inputs as the reference tool; Sampled_mel_and_waveform_match_the_reference covers every step.")]
    [InlineData(false)]
    [InlineData(true)]
    public void First_step_velocity_matches_the_reference(bool gpu)
    {
        TtsModels.Require(Path.Combine(Reference, "dbg_v0.bin"));
        using var model = Load(gpu);
        using var synthesizer = (F5Synthesizer)TtsModels.Synthesizer(model);
        float t = Floats("times")[0];
        var noise = Floats("noise");
        double conditioned = RelativeError(Floats("dbg_v0"), synthesizer.Velocity(noise, Ints("text"), Floats("cond"), t, unconditional: false));
        double unconditioned = RelativeError(Floats("dbg_v0_null"), synthesizer.Velocity(noise, Ints("text"), Floats("cond"), t, unconditional: true));
        ParityReport.Write($"F5-TTS first step {(gpu ? "vulkan" : "cpu")}: velocity relative RMS error {conditioned:E2} (conditional), {unconditioned:E2} (unconditional)");
        Assert.True(conditioned < 1e-3, $"conditional velocity relative RMS error {conditioned:E2}");
        Assert.True(unconditioned < 1e-3, $"unconditional velocity relative RMS error {unconditioned:E2}");
    }

    private static double MaxAbs(float[] reference, float[] actual) => reference.Zip(actual, (a, b) => (double)Math.Abs(a - b)).Max();

    private static double RelativeError(float[] reference, float[] actual)
    {
        double error = 0;
        double energy = 0;
        for (int i = 0; i < Math.Min(reference.Length, actual.Length); i++)
        {
            double d = actual[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        return Math.Sqrt(error / energy);
    }
}
