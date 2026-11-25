using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// VibeVoice-1.5B against transformers' <c>VibeVoiceForConditionalGeneration</c> (CPU, f32; <c>tools/make_vibevoice_tokenizer_reference.py</c> and <c>tools/make_vibevoice_reference.py</c>,
/// weights of the original Microsoft checkpoint renamed to transformers' layout): the shared speech tokenizers first (acoustic latents, semantic latents and the decoded waveform),
/// then generation.
/// </summary>
public class VibeVoiceTests
{
    private static string Reference(string folder, string name) => Path.Combine(TtsModels.VibeVoiceReference, folder, name);

    private static float[] Floats(string folder, string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Reference(folder, name + ".bin"))).ToArray();

    private static int[] Ints(string folder, string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Reference(folder, name + ".bin"))).ToArray();

    private static JsonElement Info(string folder) => JsonDocument.Parse(File.ReadAllText(Reference(folder, "info.json"))).RootElement;

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

    private static void RequireModel(string folder)
        => TtsModels.Require(Path.Combine(TtsModels.VibeVoice, "model.safetensors.index.json"), Path.Combine(TtsModels.VibeVoice, "tokenizer.json"), Reference(folder, "info.json"));

    /// <summary>The acoustic tokenizer's encoder, the semantic tokenizer's encoder and the acoustic decoder (as generation runs them, with convolution caches) against the reference on 6 s of speech.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Tokenizers_match_the_reference(bool gpu)
    {
        RequireModel("tokenizers");
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.VibeVoice);
        using var synthesizer = (VibeVoiceSynthesizer)TtsModels.Synthesizer(model);
        var audio = Floats("tokenizers", "audio");
        var (acoustic, semantic) = synthesizer.EncodeBoth(audio);
        var decoded = synthesizer.DecodeLatents(Floats("tokenizers", "acoustic_latents"), scaled: false);
        double acousticError = RelativeRms(acoustic, Floats("tokenizers", "acoustic_latents"));
        double semanticError = RelativeRms(semantic, Floats("tokenizers", "semantic_latents"));
        double waveformError = RelativeRms(decoded, Floats("tokenizers", "decoded"));
        ParityReport.Write($"VibeVoice tokenizers {(gpu ? "vulkan" : "cpu")}: acoustic latents relative RMS {acousticError:E2}, semantic latents {semanticError:E2}, decoded waveform {waveformError:E2}");
        Assert.Equal(Info("tokenizers").GetProperty("samples").GetInt32(), decoded.Length);
        Assert.True(acousticError < (gpu ? 1e-3 : 1e-4), $"acoustic latents relative RMS {acousticError:E2}");
        Assert.True(semanticError < (gpu ? 5e-3 : 1e-4), $"semantic latents relative RMS {semanticError:E2}");
        Assert.True(waveformError < (gpu ? 2e-3 : 2e-4), $"decoded waveform relative RMS {waveformError:E2}");
    }

    private static List<float[]> Rows(string folder, string name, int width)
    {
        var values = Floats(folder, name);
        return [.. Enumerable.Range(0, values.Length / width).Select(i => values.AsSpan(i * width, width).ToArray())];
    }

    // The voice of the reference run: the scaled acoustic latents of the recording (the mean of the encoder, as the reference ran with sample=False).
    private static VoiceProfile ReferenceVoice(string folder)
    {
        var latents = Floats(folder, "voice_latents");
        return new VoiceProfile
        {
            Name = "reference",
            Architecture = VibeVoiceAdapter.Family,
            Tensors = new Dictionary<string, VoiceTensor> { [VibeVoiceSynthesizer.VoiceKey] = VoiceTensor.From(latents, [latents.Length / 64, 64]) },
        };
    }

    private static VibeVoiceForcing Forcing(string folder, bool tokens, bool latents)
        => new() { Tokens = tokens ? Ints(folder, "tokens") : null, Latents = latents ? Rows(folder, "latents", 64) : null, Noise = Rows(folder, "noise", 64) };

    /// <summary>The prompt (system text, script, speech start) is the reference's token for token, with and without a voice prompt.</summary>
    [Theory]
    [InlineData("tts_single")]
    [InlineData("tts_two")]
    [InlineData("tts_voice")]
    public void Prompt_tokens_match_the_reference(string folder)
    {
        RequireModel(folder);
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.VibeVoice);
        using var synthesizer = (VibeVoiceSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(folder);
        (string, int)[] voices = folder == "tts_voice" ? [("0", Floats(folder, "voice_latents").Length / 64)] : [];
        var (tokens, _) = synthesizer.PromptTokens(VibeVoiceSynthesizer.ParseScript(info.GetProperty("script").GetString()!), voices);
        Assert.Equal(info.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()), tokens);
    }

    /// <summary>
    /// Teacher forced along the reference's tokens and latents with the reference's noise: the language model's choices, its hidden state (the diffusion condition), the context-free run's hidden state
    /// and the latents the diffusion head produces (DPM-Solver, 10 steps, guidance 1.3) agree with the reference's, and the acoustic decoder turns the latents into the reference's waveform.
    /// </summary>
    [Theory]
    [InlineData("tts_single", false)]
    [InlineData("tts_single", true)]
    [InlineData("tts_two", false)]
    [InlineData("tts_voice", false)]
    [InlineData("tts_voice", true)]
    public void Teacher_forced_generation_matches_the_reference(string folder, bool gpu)
    {
        RequireModel(folder);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.VibeVoice);
        using var synthesizer = (VibeVoiceSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(folder);
        var options = new SpeechSynthesisOptions { NormalizeText = false, VoiceProfile = folder == "tts_voice" ? ReferenceVoice(folder) : null };
        var logits4 = Floats(folder, "logits4");
        var hidden = Floats(folder, "hidden");
        var negative = Floats(folder, "negative_hidden");
        var latents = Floats(folder, "latents");
        var tokens = Ints(folder, "tokens");
        int[] ids = [151652, 151653, 151654, 151643];
        int hard = 0;
        double worst = 0;
        double hiddenError = 0;
        double negativeError = 0;
        double latentError = 0;
        int frame = 0;
        var perStep = new List<double>();
        var audio = new List<float>();
        foreach (var generated in synthesizer.GenerateFrames(info.GetProperty("script").GetString()!, options, CancellationToken.None, Forcing(folder, true, true), step =>
        {
            int i = step.Index;
            var reference = logits4.AsSpan(4 * i, 4);
            float best = float.NegativeInfinity;
            int arg = 0;
            for (int k = 0; k < 4; k++)
            {
                worst = Math.Max(worst, Math.Abs(step.Logits[ids[k]] - reference[k]));
                if (step.Logits[ids[k]] > best)
                {
                    best = step.Logits[ids[k]];
                    arg = k;
                }
            }

            if (ids[arg] != tokens[i] && Math.Abs(reference[arg] - reference[Array.IndexOf(ids, tokens[i])]) > 0.05f)
            {
                hard++;
            }

            double stepHidden = RelativeRms(step.Hidden, hidden.AsSpan(i * step.Hidden.Length, step.Hidden.Length));
            perStep.Add(stepHidden);
            hiddenError = Math.Max(hiddenError, stepHidden);
            if (step.Latent is not null)
            {
                negativeError = Math.Max(negativeError, RelativeRms(step.NegativeHidden!, negative.AsSpan(frame * step.Hidden.Length, step.Hidden.Length)));
                latentError = Math.Max(latentError, RelativeRms(step.Latent, latents.AsSpan(frame * 64, 64)));
                frame++;
            }
        }))
        {
            audio.AddRange(generated.Audio);
        }

        double waveform = RelativeRms([.. audio], Floats(folder, "audio"));
        ParityReport.Write($"VibeVoice {folder} {(gpu ? "vulkan" : "cpu")}: teacher forced over {tokens.Length} steps ({frame} frames), argmax disagreements {hard}, worst logit delta {worst:E2}, hidden relative RMS {hiddenError:E2} (context-free {negativeError:E2}), diffusion latents {latentError:E2}, waveform {waveform:E2}");
        ParityReport.Write($"  hidden relative RMS per step: {string.Join(" ", perStep.Select(e => e.ToString("E1")))}");
        Assert.Equal(info.GetProperty("frames").GetInt32(), frame);
        Assert.Equal(0, hard);
        Assert.True(hiddenError < (gpu ? 1e-2 : 5e-3), $"hidden relative RMS {hiddenError:E2}");
        Assert.True(negativeError < (gpu ? 1e-2 : 5e-3), $"context-free hidden relative RMS {negativeError:E2}");
        Assert.True(latentError < (gpu ? 2e-2 : 1e-2), $"latents relative RMS {latentError:E2}");
        Assert.True(waveform < (gpu ? 5e-3 : 1e-4), $"waveform relative RMS {waveform:E2}");
    }

    /// <summary>Free running with the reference's noise (greedy tokens): the same tokens, and latents and audio close to the reference's.</summary>
    [Fact]
    public void Free_running_generation_follows_the_reference()
    {
        RequireModel("tts_single");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.VibeVoice);
        using var synthesizer = (VibeVoiceSynthesizer)TtsModels.Synthesizer(model);
        var info = Info("tts_single");
        var tokens = new List<int>();
        var frames = synthesizer.GenerateFrames(info.GetProperty("script").GetString()!, new SpeechSynthesisOptions { NormalizeText = false }, CancellationToken.None,
            Forcing("tts_single", false, false), step => tokens.Add(step.Token)).ToList();
        var reference = Ints("tts_single", "tokens");
        int same = 0;
        while (same < Math.Min(tokens.Count, reference.Length) && tokens[same] == reference[same])
        {
            same++;
        }

        double error = RelativeRms([.. frames.SelectMany(f => f.Audio)], Floats("tts_single", "audio").AsSpan(0, frames.Count * 3200));
        ParityReport.Write($"VibeVoice free running (reference noise): {frames.Count} frames (reference {info.GetProperty("frames").GetInt32()}), identical token prefix {same} of {reference.Length}, waveform relative RMS {error:E2}");
        Assert.Equal(reference.Length, same);
        Assert.True(error < 0.15, $"waveform relative RMS {error:E2}");
    }

    /// <summary>A voice prompt (the recording's acoustic latents in the place of the audio placeholders) matches the reference: the latents, then the prompt rows through the language model's first hidden states.</summary>
    [Fact]
    public void Voice_prompt_matches_the_reference()
    {
        RequireModel("tts_voice");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.VibeVoice);
        using var synthesizer = (VibeVoiceSynthesizer)TtsModels.Synthesizer(model);
        var recording = new Sharplight.Audio.AudioBuffer(Floats("tts_voice", "voice_input"), 24000);
        var latents = synthesizer.EncodeVoice(recording, new VibeVoiceNoise(0), CancellationToken.None, sample: false);
        var reference = Floats("tts_voice", "voice_latents");
        double error = RelativeRms(latents, reference);
        ParityReport.Write($"VibeVoice voice prompt: {latents.Length / 64} frames, scaled acoustic latents relative RMS {error:E2}");
        Assert.Equal(reference.Length, latents.Length);
        Assert.True(error < 1e-3, $"voice latents relative RMS {error:E2}");
    }

    /// <summary>The model converted to a <c>.slm</c> package and loaded again generates the same frames.</summary>
    [Fact]
    public void Package_round_trip_generates_the_same_frames()
    {
        RequireModel("tts_single");
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"vibevoice-{Guid.NewGuid():N}.slm");
        try
        {
            TtsModels.Services.GetRequiredService<ModelConverter>().Convert(TtsModels.VibeVoice, path);
            Assert.Equal(ModelKinds.SpeechSynthesis, ModelPackageReader.Open(path).Definition.Description.Kind);
            var options = new SpeechSynthesisOptions { NormalizeText = false, Sampling = new SamplingOptions { Seed = 7 } };
            using var original = TtsModels.Load(TtsModels.Open(false), TtsModels.VibeVoice);
            using var packaged = TtsModels.Load(TtsModels.Open(false), path);
            using var a = (VibeVoiceSynthesizer)TtsModels.Synthesizer(original);
            using var b = (VibeVoiceSynthesizer)TtsModels.Synthesizer(packaged);
            string text = "Speaker 0: Hello there.";
            var expected = a.GenerateFrames(text, options, CancellationToken.None).Take(4).ToList();
            var actual = b.GenerateFrames(text, options, CancellationToken.None).Take(4).ToList();
            Assert.Equal(4, actual.Count);
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(expected[i].Latent, actual[i].Latent);
                Assert.Equal(expected[i].Audio, actual[i].Audio);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>The synthesizer speaks a script through the public interface, whole and streamed in buffers.</summary>
    [Fact]
    public async Task Synthesizes_and_streams_speech()
    {
        RequireModel("tts_single");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.VibeVoice);
        using var synthesizer = TtsModels.Synthesizer(model);
        var options = new SpeechSynthesisOptions { Sampling = new SamplingOptions { Seed = 3 }, DiffusionSteps = 5 };
        var result = await synthesizer.SynthesizeAsync("Speaker 0: Hi there.", options, TestContext.Current.CancellationToken);
        Assert.Equal(24000, result.Audio.SampleRate);
        Assert.InRange(result.Audio.Samples.Length, 3200 * 4, 3200 * 60);
        int buffers = 0;
        int samples = 0;
        await foreach (var buffer in synthesizer.SynthesizeStreamingAsync("Speaker 0: Hi there.", options, TestContext.Current.CancellationToken))
        {
            buffers++;
            samples += buffer.Samples.Length;
        }

        Assert.True(buffers >= 2, $"{buffers} buffers");
        Assert.Equal(result.Audio.Samples.Length, samples);
    }
}
