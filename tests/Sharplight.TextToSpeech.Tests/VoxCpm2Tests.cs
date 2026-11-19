using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// VoxCPM2 against the official voxcpm package (2.0.3, <c>VoxCPM2Model</c>, CPU float32; <c>tools/make_voxcpm2_reference.py</c>): case <c>zero</c> without a voice,
/// <c>clone</c> with an isolated reference recording and <c>ultimate</c> with the reference also as a continuation prompt with its transcript. Compared: the prompt rows
/// (tokens and the AudioVAE encoder's patches), the AudioVAE decoder, the generation teacher forced along the reference's patches with the reference's noise (both language
/// models' states, the stop logits, the diffusion's patches, the local encoder's rows and the audio), and free-running generation with the reference's noise.
/// </summary>
public class VoxCpm2Tests
{
    private const int Feature = 64;
    private const int Patch = 4;
    private const int Hidden = 2048;

    private static string Reference(string kind, string name) => Path.Combine(TtsModels.VoxCpm2Reference, kind, name);

    private static float[] Floats(string kind, string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Reference(kind, name + ".bin"))).ToArray();

    private static JsonElement Info(string kind) => JsonDocument.Parse(File.ReadAllText(Reference(kind, "info.json"))).RootElement;

    private static List<float[]> Rows(float[] values, int width) => [.. Enumerable.Range(0, values.Length / width).Select(i => values.AsSpan(i * width, width).ToArray())];

    private static void RequireFiles(string kind)
        => TtsModels.Require(Path.Combine(TtsModels.VoxCpm2, "model.safetensors"), Path.Combine(TtsModels.VoxCpm2, "audiovae.pth"), Reference(kind, "info.json"));

    private static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0;
        double energy = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            energy += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(error / energy);
    }

    // The signal-to-noise ratio in dB of actual against expected.
    private static double Snr(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected) => -20 * Math.Log10(Math.Max(RelativeRms(actual, expected), 1e-12));

    // The request of a case: the reference recording (16 kHz, so it is not resampled) and, for the continuation case, its transcript.
    private static SpeechSynthesisOptions Options(string kind)
    {
        var info = Info(kind);
        var options = new SpeechSynthesisOptions { NormalizeText = false };
        if (info.GetProperty("reference").ValueKind == JsonValueKind.String)
        {
            options = options with { ReferenceAudio = WavFile.Read(Path.Combine(TtsModels.VoxCpm2Reference, info.GetProperty("reference").GetString()!)) };
        }

        if (info.GetProperty("prompt_text").ValueKind == JsonValueKind.String)
        {
            options = options with { ReferenceText = info.GetProperty("prompt_text").GetString() };
        }

        return options;
    }

    // The reference's patches, as [patch, feature] rows.
    private static List<float[]> Patches(string kind) => Rows(Floats(kind, "patches"), Patch * Feature);

    /// <summary>The prompt rows (tokens, which rows are audio) and the AudioVAE encoder's patches of the recording are the reference's.</summary>
    [Theory]
    [InlineData("zero", false)]
    [InlineData("clone", false)]
    [InlineData("ultimate", false)]
    [InlineData("ultimate", true)]
    public void Prompt_matches_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.VoxCpm2);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        var options = Options(kind);
        float[]? reference = null;
        (string, float[])? continuation = null;
        if (options.ReferenceAudio is { } recording)
        {
            var samples = recording.Samples.ToArray();
            reference = synthesizer.EncodeAudio(samples, padStart: false);
            if (options.ReferenceText is { } transcript)
            {
                continuation = (transcript, synthesizer.EncodeAudio(samples, padStart: true));
            }
        }

        var prompt = synthesizer.BuildPrompt(VoxCpm2Synthesizer.TargetText(info.GetProperty("text").GetString()!, options), reference, continuation);
        Assert.Equal(info.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()), prompt.Tokens);
        Assert.Equal(info.GetProperty("audio_mask").EnumerateArray().Select(t => t.GetInt32() == 1), prompt.Audio);
        var expected = Rows(Floats(kind, "prompt_feats"), Patch * Feature);
        double worst = 0;
        for (int row = 0, audio = 0; row < prompt.Tokens.Length; row++)
        {
            if (prompt.Audio[row])
            {
                worst = Math.Max(worst, RelativeRms(prompt.Patches[audio++], expected[row]));
            }
        }

        ParityReport.Write($"VoxCPM2 {kind} {(gpu ? "gpu" : "cpu")}: prompt of {prompt.Tokens.Length} rows ({prompt.Patches.Count} audio), worst AudioVAE encoder patch relative RMS {worst:E2}");
        Assert.True(worst < (gpu ? 5e-3 : 1e-4), $"AudioVAE encoder patches relative RMS {worst:E2}");
    }

    /// <summary>The AudioVAE decoder, patch by patch with its causal caches, turns the reference's latents into the reference's audio.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoder_matches_the_reference(bool gpu)
    {
        RequireFiles("zero");
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.VoxCpm2);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var latents = Floats("zero", "latents");
        int frames = latents.Length / Feature;
        var timeMajor = new float[latents.Length];
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < Feature; c++)
            {
                timeMajor[(f * Feature) + c] = latents[(c * frames) + f];
            }
        }

        var audio = synthesizer.DecodeLatents(timeMajor);
        var expected = Floats("zero", "audio");
        Assert.Equal(expected.Length, audio.Length);
        double error = RelativeRms(audio, expected);
        ParityReport.Write($"VoxCPM2 AudioVAE decoder {(gpu ? "gpu" : "cpu")}: {frames} frames, audio relative RMS {error:E2} (SNR {Snr(audio, expected):F1} dB)");
        Assert.True(error < (gpu ? 5e-3 : 1e-4), $"audio relative RMS {error:E2}");
    }

    /// <summary>
    /// Teacher forced along the reference's patches with the reference's noise: the language models' states, the stop logits, the patches the diffusion transformer makes
    /// (10 Euler steps, guidance 2) and the local encoder's rows agree with the reference's, and the AudioVAE turns the patches into the reference's audio.
    /// </summary>
    [Theory]
    [InlineData("zero", false)]
    [InlineData("zero", true)]
    [InlineData("clone", false)]
    [InlineData("ultimate", false)]
    [InlineData("ultimate", true)]
    public void Teacher_forced_generation_matches_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.VoxCpm2);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        int steps = info.GetProperty("steps").GetInt32();
        var lm = Floats(kind, "lm_hidden");
        var residual = Floats(kind, "residual_hidden");
        var stop = Floats(kind, "stop");
        var patches = Floats(kind, "patches");
        var embeds = Floats(kind, "embeds");
        var forcing = new VoxCpm2Forcing { Noise = Rows(Floats(kind, "noise"), Patch * Feature), Patches = Patches(kind), Steps = steps };
        double lmError = 0;
        double residualError = 0;
        double patchError = 0;
        double embedError = 0;
        double stopDelta = 0;
        int stopDisagreements = 0;
        var perStep = new List<double>();
        var audio = new List<float>();
        foreach (var generated in synthesizer.GeneratePatches(info.GetProperty("text").GetString()!, Options(kind), CancellationToken.None, forcing, step =>
        {
            int i = step.Index;
            lmError = Math.Max(lmError, RelativeRms(step.LmHidden, lm.AsSpan(i * Hidden, Hidden)));
            residualError = Math.Max(residualError, RelativeRms(step.ResidualHidden, residual.AsSpan(i * Hidden, Hidden)));
            double patch = RelativeRms(step.Patch, patches.AsSpan(i * Patch * Feature, Patch * Feature));
            perStep.Add(patch);
            patchError = Math.Max(patchError, patch);
            embedError = Math.Max(embedError, RelativeRms(step.Embedding, embeds.AsSpan(i * Hidden, Hidden)));
            stopDelta = Math.Max(stopDelta, Math.Max(Math.Abs(step.Stop[0] - stop[2 * i]), Math.Abs(step.Stop[1] - stop[(2 * i) + 1])));
            if ((step.Stop[1] > step.Stop[0]) != (stop[(2 * i) + 1] > stop[2 * i]))
            {
                stopDisagreements++;
            }
        }))
        {
            audio.AddRange(generated.Audio);
        }

        var expected = Floats(kind, "audio");
        int context = info.GetProperty("context").GetInt32();
        Assert.Equal(steps * synthesizer.SamplesPerPatch, audio.Count);
        Assert.Equal(expected.Length, audio.Count);
        double waveform = RelativeRms([.. audio], expected);
        ParityReport.Write($"VoxCPM2 {kind} {(gpu ? "gpu" : "cpu")}: teacher forced over {steps} patches (decoder context {context}), relative RMS: text-semantic state {lmError:E2}, residual state {residualError:E2}, "
            + $"diffusion patches {patchError:E2}, encoder rows {embedError:E2}, stop logits max delta {stopDelta:E2} ({stopDisagreements} decisions differ), audio {waveform:E2} (SNR {Snr([.. audio], expected):F1} dB)");
        ParityReport.Write($"  patch relative RMS per step: {string.Join(" ", perStep.Select(e => e.ToString("E1")))}");
        Assert.Equal(0, stopDisagreements);

        // The scalar quantizer rounds tanh · 9 to integers: a value within rounding error of a half step can land in the other bin (one level of one of 512 latents), which
        // shows as ~1e-2 in the text-semantic state of that step even on the CPU; the residual state and the patches inherit a fraction of it.
        Assert.True(lmError < (gpu ? 3e-2 : 2e-2), $"text-semantic state relative RMS {lmError:E2}");
        Assert.True(residualError < (gpu ? 2e-2 : 5e-3), $"residual state relative RMS {residualError:E2}");
        Assert.True(patchError < (gpu ? 5e-2 : 1e-2), $"diffusion patches relative RMS {patchError:E2}");
        Assert.True(embedError < (gpu ? 2e-2 : 1e-3), $"encoder rows relative RMS {embedError:E2}");
        Assert.True(waveform < (gpu ? 5e-3 : 1e-4), $"audio relative RMS {waveform:E2}");
    }

    /// <summary>Free running with the reference's noise: the same number of patches (the stop head ends at the same step), patches and audio close to the reference's.</summary>
    [Theory]
    [InlineData("zero", false)]
    [InlineData("clone", false)]
    [InlineData("zero", true)]
    public void Free_running_generation_follows_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.VoxCpm2);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        int steps = info.GetProperty("steps").GetInt32();
        var forcing = new VoxCpm2Forcing { Noise = Rows(Floats(kind, "noise"), Patch * Feature), MaxSteps = info.GetProperty("max_len").GetInt32() };
        var generated = synthesizer.GeneratePatches(info.GetProperty("text").GetString()!, Options(kind), CancellationToken.None, forcing).ToList();
        var reference = Patches(kind);
        var expected = Floats(kind, "audio");
        int compared = Math.Min(generated.Count, reference.Count);
        var actualPatches = generated.Take(compared).SelectMany(p => p.Latents).ToArray();
        double patchError = RelativeRms(actualPatches, reference.Take(compared).SelectMany(p => p).ToArray());
        var audio = generated.SelectMany(p => p.Audio).Take(expected.Length).ToArray();
        double snr = Snr(audio, expected.AsSpan(0, audio.Length));
        ParityReport.Write($"VoxCPM2 {kind} {(gpu ? "gpu" : "cpu")} free running (reference noise): {generated.Count} patches (reference {steps}), patches relative RMS {patchError:E2}, audio SNR {snr:F1} dB");
        Assert.Equal(steps, generated.Count);
        Assert.True(patchError < (gpu ? 0.5 : 0.05), $"patches relative RMS {patchError:E2}");
    }

    /// <summary>A saved voice (reference and continuation patches with the transcript) speaks like the recording it was made from.</summary>
    [Fact]
    public async Task Voice_profile_matches_the_recording()
    {
        RequireFiles("ultimate");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.VoxCpm2);
        using var synthesizer = (VoxCpm2Synthesizer)TtsModels.Synthesizer(model);
        var options = Options("ultimate");
        var profile = await synthesizer.CreateVoiceAsync("jfk", options.ReferenceAudio!, options.ReferenceText, TestContext.Current.CancellationToken);
        Assert.Equal(VoxCpm2Adapter.Family, profile.Architecture);
        Assert.Equal(options.ReferenceText, profile.ReferenceText);
        var info = Info("ultimate");
        var forcing = new VoxCpm2Forcing { Noise = Rows(Floats("ultimate", "noise"), Patch * Feature), Patches = Patches("ultimate"), Steps = 2 };
        var fromRecording = synthesizer.GeneratePatches(info.GetProperty("text").GetString()!, options, CancellationToken.None, forcing).SelectMany(p => p.Audio).ToArray();
        var fromProfile = synthesizer.GeneratePatches(info.GetProperty("text").GetString()!, new SpeechSynthesisOptions { NormalizeText = false, VoiceProfile = profile },
            CancellationToken.None, forcing).SelectMany(p => p.Audio).ToArray();
        Assert.Equal(fromRecording, fromProfile);
    }
}
