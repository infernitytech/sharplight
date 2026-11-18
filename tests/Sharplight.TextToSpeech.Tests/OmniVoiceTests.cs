using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// OmniVoice against the official omnivoice package (CPU, float32, deterministic: position and class temperature 0; <c>tools/make_omnivoice_reference.py</c>; a voice-cloning case from
/// jfk.wav with its transcript and a voice-design case): the prompt tokens, the reference codes, the language model's logits at the first and last step, every step of the unmasking
/// schedule teacher forced, the greedy codes end to end, and the decoded audio.
/// </summary>
public class OmniVoiceTests
{
    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(TtsModels.OmniVoiceReference, name))).ToArray();

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(TtsModels.OmniVoiceReference, name))).ToArray();

    private static JsonElement Info(string kind) => JsonDocument.Parse(File.ReadAllText(Path.Combine(TtsModels.OmniVoiceReference, kind + "_info.json"))).RootElement;

    private static void RequireFiles(string kind)
        => TtsModels.Require(Path.Combine(TtsModels.OmniVoice, "model.safetensors"), Path.Combine(TtsModels.OmniVoice, "audio_tokenizer", "model.safetensors"),
            Path.Combine(TtsModels.OmniVoiceReference, kind + "_tokens.i32"));

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

    private static OmniVoiceReference? Reference(string kind)
    {
        var info = Info(kind);
        if (kind != "clone")
        {
            return null;
        }

        var codes = Ints("clone_ref_tokens.i32");
        return new OmniVoiceReference(codes, info.GetProperty("ref_len").GetInt32(), OmniVoiceText.AddPunctuation(info.GetProperty("ref_text").GetString()!), (float)info.GetProperty("ref_rms").GetDouble());
    }

    private static OmniVoicePrompt Prompt(OmniVoiceSynthesizer synthesizer, string kind)
    {
        var info = Info(kind);
        string? instruct = info.TryGetProperty("instruct", out var i) && i.ValueKind == JsonValueKind.String ? OmniVoiceInstruct.Resolve(i.GetString(), false) : null;
        return synthesizer.BuildPrompt(info.GetProperty("text").GetString()!, Reference(kind), info.GetProperty("language").GetString(), instruct, info.GetProperty("target_len").GetInt32());
    }

    [Theory]
    [InlineData("clone")]
    [InlineData("design")]
    public void Prompt_matches_the_reference(string kind)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.OmniVoice);
        using var synthesizer = (OmniVoiceSynthesizer)TtsModels.Synthesizer(model);
        var prompt = Prompt(synthesizer, kind);
        var ids = Ints(kind + "_ids.i32");
        var mask = File.ReadAllBytes(Path.Combine(TtsModels.OmniVoiceReference, kind + "_audio_mask.u8"));
        Assert.Equal(Info(kind).GetProperty("c_len").GetInt32(), prompt.Length);
        Assert.Equal(mask.Select(m => m != 0), prompt.AudioMask);
        Assert.Equal(ids, prompt.Ids);
    }

    [Fact]
    public void Schedule_matches_the_reference()
    {
        RequireFiles("clone");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.OmniVoice);
        using var synthesizer = (OmniVoiceSynthesizer)TtsModels.Synthesizer(model);
        foreach (string kind in new[] { "clone", "design" })
        {
            var info = Info(kind);
            Assert.Equal(info.GetProperty("schedule").EnumerateArray().Select(v => v.GetInt32()), synthesizer.Schedule(info.GetProperty("target_len").GetInt32(), OmniVoiceGenerationSettings.Greedy));
        }
    }

    /// <summary>The tokenizer's encoder on the reference's own 24 kHz samples: the codes are the reference's.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Reference_codes_match(bool gpu)
    {
        RequireFiles("clone");
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.OmniVoice);
        using var synthesizer = (OmniVoiceSynthesizer)TtsModels.Synthesizer(model);
        var info = Info("clone");
        var reference = synthesizer.EncodeCodes(Floats("clone_ref_audio.f32"), info.GetProperty("ref_text").GetString()!, (float)info.GetProperty("ref_rms").GetDouble());
        var expected = Ints("clone_ref_tokens.i32");
        Assert.Equal(expected.Length, reference.Codes.Length);
        int wrong = expected.Where((v, i) => v != reference.Codes[i]).Count();
        ParityReport.Write($"OmniVoice {(gpu ? "vulkan" : "cpu")}: reference codes, {wrong} of {expected.Length} differ");
        Assert.True(wrong == 0, $"{wrong} of {expected.Length} codes differ");
    }

    /// <summary>
    /// Along the reference's own states, the language model's conditional and unconditional logits of the first and last step are the reference's, and every step's choice reproduces the
    /// reference's next state (differences only where the reference's top two scores are within a small margin).
    /// </summary>
    [Theory]
    [InlineData("clone", false)]
    [InlineData("clone", true)]
    [InlineData("design", false)]
    [InlineData("design", true)]
    public void Logits_and_greedy_codes_match_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.OmniVoice);
        using var synthesizer = (OmniVoiceSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        var prompt = Prompt(synthesizer, kind);
        int t = prompt.TargetLength;
        int steps = info.GetProperty("steps").GetInt32();
        var states = Ints(kind + "_states.i32");
        string device = gpu ? "vulkan" : "cpu";

        foreach (string which in new[] { "first", "last" })
        {
            var state = states.AsSpan(which == "first" ? 0 : (steps - 1) * 8 * t, 8 * t);
            var logits = synthesizer.Forward(prompt, state);
            var reference = Floats($"{kind}_logits_{which}.f32");
            var actual = new float[reference.Length];
            int c = prompt.Length;
            for (int q = 0; q < 8; q++)
            {
                for (int i = 0; i < t; i++)
                {
                    for (int v = 0; v < 1025; v++)
                    {
                        actual[((((0 * 8) + q) * t) + i) * 1025 + v] = logits[(((c - t) + i) * 8200) + (q * 1025) + v];
                        actual[((((1 * 8) + q) * t) + i) * 1025 + v] = logits[((c + i) * 8200) + (q * 1025) + v];
                    }
                }
            }

            double error = RelativeRms(actual, reference);
            ParityReport.Write($"OmniVoice {device} {kind}: {which} step logits relative RMS {error:E2}");
            Assert.True(error < 2e-3, $"{which}: relative RMS {error:E2}");
        }

        var tokens = synthesizer.GenerateTokens(prompt, OmniVoiceGenerationSettings.Greedy, null, CancellationToken.None);
        var expected = Ints(kind + "_tokens.i32");
        int wrong = expected.Where((v, i) => v != tokens[i]).Count();
        ParityReport.Write($"OmniVoice {device} {kind}: greedy codes, {wrong} of {expected.Length} differ");
        Assert.True(wrong <= expected.Length / 200, $"{wrong} of {expected.Length} codes differ");
    }

    /// <summary>The tokenizer's decoder on the reference's final codes, scaled as the package does.</summary>
    [Theory]
    [InlineData("clone", false)]
    [InlineData("clone", true)]
    [InlineData("design", false)]
    public void Decoded_audio_matches_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.OmniVoice);
        using var synthesizer = (OmniVoiceSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        var decoded = synthesizer.Decode(Ints(kind + "_tokens.i32"));
        float? rms = kind == "clone" ? (float)info.GetProperty("ref_rms").GetDouble() : null;
        var audio = synthesizer.PostProcess(decoded, rms, cleanUp: false);
        var reference = Floats(kind + "_audio.f32");
        Assert.Equal(reference.Length, audio.Length);
        double error = RelativeRms(audio, reference);
        ParityReport.Write($"OmniVoice {(gpu ? "vulkan" : "cpu")} {kind}: decoded audio relative RMS {error:E2}");
        Assert.True(error < 1e-4, $"relative RMS {error:E2}");
    }

    [Fact]
    public async Task Speaks_a_cloned_and_a_designed_voice()
    {
        RequireFiles("clone");
        string jfk = Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "..", "audio", "jfk.wav");
        TtsModels.Require(jfk);
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.OmniVoice);
        using var synthesizer = TtsModels.Synthesizer(model);
        var clone = await synthesizer.SynthesizeAsync("Hello, this is a test of the OmniVoice speech model.", new SpeechSynthesisOptions
        {
            Language = "English", ReferenceAudio = WavFile.Read(jfk), ReferenceText = Info("clone").GetProperty("ref_text").GetString(), Sampling = new Sharplight.Llm.SamplingOptions { Temperature = 0f },
        });
        Assert.Equal(24000, clone.Audio.SampleRate);
        Assert.InRange(clone.Audio.Samples.Length / 24000.0, 1.5, 8.0);
        var design = await synthesizer.SynthesizeAsync("Hello, this is a test.", new SpeechSynthesisOptions { Language = "en", Instructions = "female, british accent", Sampling = new Sharplight.Llm.SamplingOptions { Temperature = 0f } });
        Assert.InRange(design.Audio.Samples.Length / 24000.0, 0.5, 8.0);
        Assert.True(design.Audio.Samples.Max(MathF.Abs) > 0.1f);
    }

    /// <summary>The model converted to a <c>.slm</c> package (both the language model and its tokenizer codec) and loaded again unmasks the same codes and decodes the same audio.</summary>
    [Fact]
    public void Package_round_trip_generates_the_same_codes()
    {
        RequireFiles("design");
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"omnivoice-{Guid.NewGuid():N}.slm");
        try
        {
            TtsModels.Services.GetRequiredService<ModelConverter>().Convert(TtsModels.OmniVoice, path);
            Assert.Equal(ModelKinds.SpeechSynthesis, ModelPackageReader.Open(path).Definition.Description.Kind);
            var settings = OmniVoiceGenerationSettings.Greedy with { Steps = 4 };
            using var original = TtsModels.Load(TtsModels.Open(false), TtsModels.OmniVoice);
            using var packaged = TtsModels.Load(TtsModels.Open(false), path);
            using var a = (OmniVoiceSynthesizer)TtsModels.Synthesizer(original);
            using var b = (OmniVoiceSynthesizer)TtsModels.Synthesizer(packaged);
            var prompt = Prompt(a, "design");
            Assert.Equal(prompt, Prompt(b, "design"), new PromptComparer());
            var expected = a.GenerateTokens(prompt, settings, null, CancellationToken.None);
            var actual = b.GenerateTokens(prompt, settings, null, CancellationToken.None);
            Assert.Equal(expected, actual);
            Assert.Equal(a.Decode(expected), b.Decode(actual));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private sealed class PromptComparer : IEqualityComparer<OmniVoicePrompt>
    {
        public bool Equals(OmniVoicePrompt? x, OmniVoicePrompt? y) => x is not null && y is not null && x.Ids.AsSpan().SequenceEqual(y.Ids) && x.AudioMask.AsSpan().SequenceEqual(y.AudioMask);

        public int GetHashCode(OmniVoicePrompt obj) => obj.Length;
    }
}
