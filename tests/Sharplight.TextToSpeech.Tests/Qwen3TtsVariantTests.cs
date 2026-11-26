using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Speech;
using Sharplight.Tests;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Qwen3-TTS Base (voice cloning: in-context prompt with the reference's transcript and frames, or the speaker embedding alone) and VoiceDesign (an instruction)
/// against the official qwen_tts package (CPU, f32; <c>tools/make_qwen3_tts_variant_reference.py</c>, greedy talker and code predictor with the wrapper's
/// default text modes): the speaker encoder's log-mel and embedding, the reference's codec frames, the text tokens, the talker's prompt and interleaved text rows,
/// the talker's and the code predictor's logits along the reference's frames (teacher forced), the greedy frames (up to the reference's first near tie) and the
/// decoded audio; plus saved voices and the <c>.slm</c> round trip.
/// </summary>
public class Qwen3TtsVariantTests
{
    private const float NearTie = 0.01f;

    private static readonly string[] s_cases = ["base-0.6b-clone", "base-0.6b-xvector", "design-1.7b", "base-1.7b-clone"];

    /// <summary>The reference cases (folders under <see cref="TtsModels.Qwen3TtsVariantReference"/>).</summary>
    public static TheoryData<string> Cases => [.. s_cases];

    /// <summary>The cases of the voice-cloning models.</summary>
    public static TheoryData<string> CloningCases => [.. s_cases.Where(c => c.StartsWith("base", StringComparison.Ordinal))];

    /// <summary>Every case on the CPU and on the GPU under test.</summary>
    public static TheoryData<string, bool> CasesOnDevices => [.. s_cases.SelectMany(c => new[] { (c, false), (c, true) })];

    /// <summary>The voice-cloning cases on the CPU and on the GPU under test.</summary>
    public static TheoryData<string, bool> CloningCasesOnDevices => [.. s_cases.Where(c => c.StartsWith("base", StringComparison.Ordinal)).SelectMany(c => new[] { (c, false), (c, true) })];

    private static string Provider(bool gpu) => gpu ? TestGpu.ProviderName : "cpu";

    private static string Folder(string name) => Path.Combine(TtsModels.Qwen3TtsVariantReference, name);

    private static string ModelOf(string name) => name switch
    {
        "design-1.7b" => TtsModels.Qwen3TtsVoiceDesign,
        "base-1.7b-clone" => TtsModels.Qwen3TtsBaseLarge,
        _ => TtsModels.Qwen3TtsBase,
    };

    private static int[] Ints(string name, string file) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Folder(name), file + ".bin"))).ToArray();

    private static float[] Floats(string name, string file) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Folder(name), file + ".bin"))).ToArray();

    private static JsonElement Info(string name) => JsonDocument.Parse(File.ReadAllText(Path.Combine(Folder(name), "info.json"))).RootElement;

    private static void RequireFiles(string name)
        => TtsModels.Require(Path.Combine(ModelOf(name), "model.safetensors"), Path.Combine(ModelOf(name), "speech_tokenizer", "model.safetensors"),
            Path.Combine(Folder(name), "codes.bin"));

    private static List<int[]> Frames(int[] codes) => [.. Enumerable.Range(0, codes.Length / 16).Select(f => codes.AsSpan(f * 16, 16).ToArray())];

    private static bool HasReference(string name) => File.Exists(Path.Combine(Folder(name), "speaker.bin"));

    // The request options of a case; the reference voice is the reference's own speaker embedding and frames, so the talker is compared on identical inputs.
    private static (SpeechSynthesisOptions Options, Qwen3TtsReference? Reference) Request(string name, SamplingOptions? sampling = null)
    {
        var info = Info(name);
        var options = new SpeechSynthesisOptions
        {
            Language = info.GetProperty("language").GetString(), NormalizeText = false,
            Instructions = info.TryGetProperty("instruction", out var i) ? i.GetString() : null,
            Sampling = sampling ?? new SamplingOptions { Temperature = 0f, RepetitionPenalty = 1.05f, RepetitionWindow = 8192 },
        };
        if (!HasReference(name))
        {
            return (options, null);
        }

        string? text = info.TryGetProperty("reference_text", out var t) ? t.GetString() : null;
        var reference = new Qwen3TtsReference(Floats(name, "speaker"), text is null ? null : Frames(Ints(name, "ref_codes")), text);
        return (options, reference);
    }

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

    [Theory]
    [MemberData(nameof(Cases))]
    public void Prompt_matches_the_reference(string name)
    {
        RequireFiles(name);
        using var model = TtsModels.Load(TtsModels.Open(false), ModelOf(name));
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(name);
        string text = info.GetProperty("text").GetString()!;
        Assert.Equal(info.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()), synthesizer.TokensFor(text));
        var (options, reference) = Request(name);
        var prompt = synthesizer.BuildPrompt(text, options, reference);
        var rows = Floats(name, "prompt");
        var trailing = Floats(name, "trailing");
        Assert.Equal(rows.Length, prompt.Rows.Length);
        double error = RelativeRms(prompt.Rows, rows);

        // The reference pads the interleaved text with one text pad row when it has none left.
        int width = rows.Length / info.GetProperty("prompt_rows").GetInt32();
        bool padOnly = prompt.Trailing.Length == 0 && trailing.Length == width;
        double trailingError = padOnly || trailing.Length == 0 ? 0 : RelativeRms(prompt.Trailing, trailing);
        Assert.True(padOnly || trailing.Length == prompt.Trailing.Length, $"trailing rows {prompt.Trailing.Length / width} (reference {trailing.Length / width})");
        ParityReport.Write($"Qwen3-TTS {name} prompt: {rows.Length / width} rows relative RMS {error:E2}, {trailing.Length / width} interleaved text rows relative RMS {trailingError:E2}");
        Assert.True(error < 1e-5, $"prompt relative RMS {error:E2}");
        Assert.True(trailingError < 1e-5, $"interleaved text relative RMS {trailingError:E2}");
    }

    /// <summary>The speaker encoder's log-mel front end on the host and its embedding on the CPU and the GPU under test, and the reference's codec frames.</summary>
    [Theory]
    [MemberData(nameof(CloningCasesOnDevices))]
    public void Speaker_embedding_and_reference_frames_match_the_reference(string name, bool gpu)
    {
        RequireFiles(name);
        using var model = TtsModels.Load(TtsModels.Open(gpu), ModelOf(name));
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var samples = Floats(name, "reference");
        if (!gpu)
        {
            var features = AudioFeatureExtractors.Create(model, Qwen3TtsAdapter.SpeakerFeaturesRole).Extract(samples);
            var mel = S3SpeechEncoders.TimeMajor(features);
            var expectedMel = Floats(name, "mel");
            Assert.Equal(expectedMel.Length, mel.Length);
            double melError = mel.Zip(expectedMel, (a, b) => (double)Math.Abs(a - b)).Max();
            ParityReport.Write($"Qwen3-TTS {name} speaker log-mel: {features.Frames} frames, worst absolute error {melError:E2}");
            Assert.True(melError < 1e-3, $"log-mel worst error {melError:E2}");
        }

        var speaker = synthesizer.SpeakerEmbedding(samples);
        double error = RelativeRms(speaker, Floats(name, "speaker"));
        var frames = synthesizer.ReferenceFrames(samples);
        var expectedFrames = Frames(Ints(name, "ref_codes"));
        Assert.Equal(expectedFrames.Count, frames.Count);
        int same = frames.Zip(expectedFrames).Sum(p => p.First.Zip(p.Second).Count(c => c.First == c.Second));
        ParityReport.Write($"Qwen3-TTS {name} {Provider(gpu)}: speaker embedding relative RMS {error:E2}; reference codes {same} of {16 * frames.Count} identical");
        Assert.True(error < (gpu ? 1e-2 : 1e-4), $"speaker embedding relative RMS {error:E2}");
        Assert.True(same >= 16 * frames.Count * (gpu ? 0.97 : 0.995), $"{same} of {16 * frames.Count} reference codes identical");
    }

    /// <summary>
    /// Along the reference's frames, the talker's first-codebook choice and the code predictor's choices agree with the reference's unless its own margin is small, and the
    /// greedy frames are the reference's up to its first near tie.
    /// </summary>
    [Theory]
    [MemberData(nameof(CasesOnDevices))]
    public void Logits_and_greedy_frames_match_the_reference(string name, bool gpu)
    {
        RequireFiles(name);
        using var model = TtsModels.Load(TtsModels.Open(gpu), ModelOf(name));
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        string text = Info(name).GetProperty("text").GetString()!;
        var (options, reference) = Request(name);
        var prompt = synthesizer.BuildPrompt(text, options, reference);
        var frames = Frames(Ints(name, "codes"));
        var top2 = Ints(name, "top2_ids");
        var top2Values = Floats(name, "top2_values");
        var predictorTop2 = Ints(name, "cp_top2_ids");
        var predictorValues = Floats(name, "cp_top2_values");
        int step = 0;
        int hard = 0;
        int predictorHard = 0;
        double worst = 0;
        foreach (var _ in synthesizer.GenerateFrames(prompt, options, CancellationToken.None, frames, (logits, predictor) =>
        {
            if (Sampler.ArgMax(logits) != top2[2 * step] && top2Values[2 * step] - top2Values[(2 * step) + 1] > 0.05f)
            {
                hard++;
            }

            worst = Math.Max(worst, Math.Abs(logits[top2[2 * step]] - top2Values[2 * step]));
            for (int g = 0; g < predictor.Count; g++)
            {
                int row = (step * 15) + g;
                if (Sampler.ArgMax(predictor[g]) != predictorTop2[2 * row] && predictorValues[2 * row] - predictorValues[(2 * row) + 1] > 0.05f)
                {
                    predictorHard++;
                }

                worst = Math.Max(worst, Math.Abs(predictor[g][predictorTop2[2 * row]] - predictorValues[2 * row]));
            }

            step++;
        }))
        {
        }

        string provider = Provider(gpu);
        Assert.Equal(frames.Count + 1, step);
        ParityReport.Write($"Qwen3-TTS {name} {provider}: teacher forced over {frames.Count} frames, talker disagreements {hard}, code predictor {predictorHard}, worst top logit error {worst:E2}");
        Assert.Equal(0, hard);
        Assert.Equal(0, predictorHard);
        Assert.True(worst < (gpu ? 0.15 : 0.05), $"worst top logit error {worst:E2}");

        var greedy = synthesizer.GenerateFrames(prompt, options, CancellationToken.None).ToList();
        int same = 0;
        while (same < Math.Min(greedy.Count, frames.Count) && greedy[same].SequenceEqual(frames[same]))
        {
            same++;
        }

        // The code predictor's top two logits are often within a hundredth of each other: greedy decoding follows the reference up to its first such near tie.
        int tie = 0;
        while (tie < frames.Count && top2Values[2 * tie] - top2Values[(2 * tie) + 1] >= NearTie
               && Enumerable.Range(0, 15).All(g => predictorValues[2 * ((tie * 15) + g)] - predictorValues[(2 * ((tie * 15) + g)) + 1] >= NearTie))
        {
            tie++;
        }

        ParityReport.Write($"Qwen3-TTS {name} {provider}: greedy frames {greedy.Count} (reference {frames.Count}), identical prefix {same}, first reference near tie at frame {tie}");
        if (!gpu)
        {
            Assert.True(same >= tie, $"greedy frames diverge at {same}, before the reference's first near tie at {tie}");
        }
    }

    /// <summary>The wrapper's output audio: the reference's and the generated frames decoded together and the reference's share cut off.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Decoded_audio_matches_the_reference(string name)
    {
        RequireFiles(name);
        using var model = TtsModels.Load(TtsModels.Open(false), ModelOf(name));
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var (_, reference) = Request(name);
        var referenceFrames = reference?.Frames ?? [];
        var frames = new List<int[]>(referenceFrames);
        frames.AddRange(Frames(Ints(name, "codes")));
        var audio = synthesizer.DecodeAfter(frames, referenceFrames.Count);
        var expected = Floats(name, "audio");
        Assert.Equal(expected.Length, audio.Samples.Length);
        double error = RelativeRms(audio.Samples, expected);
        ParityReport.Write($"Qwen3-TTS {name} decoded audio: relative RMS {error:E2}");
        Assert.True(error < 1e-4, $"relative RMS {error:E2}");
    }

    /// <summary>Cloning from the recording itself (the speaker encoder and the codec encoder run) builds the reference's prompt.</summary>
    [Theory]
    [MemberData(nameof(CloningCases))]
    public void Prompt_from_the_recording_matches_the_reference(string name)
    {
        RequireFiles(name);
        using var model = TtsModels.Load(TtsModels.Open(false), ModelOf(name));
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(name);
        var (options, _) = Request(name);
        options = options with
        {
            ReferenceAudio = new AudioBuffer(Floats(name, "reference"), 24000),
            ReferenceText = info.TryGetProperty("reference_text", out var t) ? t.GetString() : null,
        };
        var prompt = synthesizer.BuildPrompt(info.GetProperty("text").GetString()!, options, TestContext.Current.CancellationToken);
        double error = RelativeRms(prompt.Rows, Floats(name, "prompt"));
        ParityReport.Write($"Qwen3-TTS {name} prompt from the recording: relative RMS {error:E2}");
        Assert.True(error < 1e-3, $"relative RMS {error:E2}");
    }

    /// <summary>A saved voice (<see cref="IVoiceCloner"/>) speaks like the recording it was made from.</summary>
    [Fact]
    public async Task Saved_voice_matches_the_recording()
    {
        const string name = "base-0.6b-clone";
        RequireFiles(name);
        using var model = TtsModels.Load(TtsModels.Open(false), ModelOf(name));
        using var synthesizer = TtsModels.Synthesizer(model);
        var cloner = (IVoiceCloner)synthesizer.GetService(typeof(IVoiceCloner))!;
        var recording = new AudioBuffer(Floats(name, "reference"), 24000);
        string transcript = Info(name).GetProperty("reference_text").GetString()!;
        var profile = await cloner.CreateVoiceAsync("clone", recording, transcript, TestContext.Current.CancellationToken);
        var stored = VoiceLibraryRoundTrip(profile);
        // The model's own sampling with a seed (greedy in-context cloning tends not to stop).
        var seeded = new SamplingOptions { Seed = 7 };
        var fromProfile = await synthesizer.SynthesizeAsync("Good morning.", new SpeechSynthesisOptions { Language = "english", VoiceProfile = stored, Sampling = seeded },
            TestContext.Current.CancellationToken);
        var fromRecording = await synthesizer.SynthesizeAsync("Good morning.", new SpeechSynthesisOptions
        {
            Language = "english", ReferenceAudio = recording, ReferenceText = transcript, Sampling = seeded,
        }, TestContext.Current.CancellationToken);
        Assert.Equal(fromRecording.Audio.Samples, fromProfile.Audio.Samples);
        Assert.InRange(fromProfile.Audio.Samples.Length, 4800, 24000 * 10);
    }

    // The voice saved to a voice library and read back.
    private static VoiceProfile VoiceLibraryRoundTrip(VoiceProfile profile)
    {
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests", $"voices-{Guid.NewGuid():N}");
        try
        {
            var library = new VoiceLibrary(folder);
            library.Save(profile);
            return library.Load(profile.Name);
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    /// <summary>The voice-design model speaks a request with the voice its instruction describes (the model's own sampling).</summary>
    [Fact]
    public async Task Voice_design_speaks_with_an_instruction()
    {
        RequireFiles("design-1.7b");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Qwen3TtsVoiceDesign);
        using var synthesizer = TtsModels.Synthesizer(model);
        Assert.Empty(synthesizer.Voices);
        Assert.True(((Qwen3TtsSynthesizer)synthesizer).AcceptsInstructions);
        Assert.Null(synthesizer.GetService(typeof(IVoiceCloner)));
        var result = await synthesizer.SynthesizeAsync("Hi there.", new SpeechSynthesisOptions
        {
            Language = "english", Instructions = "An old man with a deep, raspy voice.", Sampling = new SamplingOptions { Seed = 1 },
        }, TestContext.Current.CancellationToken);
        Assert.Equal(24000, result.Audio.SampleRate);
        Assert.InRange(result.Audio.Samples.Length, 4800, 24000 * 10);
    }

    /// <summary>A voice-cloning model has no built-in voices, reads no instructions and asks for a reference recording.</summary>
    [Fact]
    public void Cloning_model_requires_a_reference()
    {
        RequireFiles("base-0.6b-clone");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Qwen3TtsBase);
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        Assert.Empty(synthesizer.Voices);
        Assert.False(synthesizer.AcceptsInstructions);
        Assert.Throws<ArgumentException>(() => synthesizer.BuildPrompt("Hello.", null, TestContext.Current.CancellationToken));
    }

    /// <summary>The voice-cloning model converted to a <c>.slm</c> package (speaker encoder, codec encoder and decoder) and loaded again builds the same prompt and frames.</summary>
    [Fact]
    public void Package_round_trip_generates_the_same_frames()
    {
        const string name = "base-0.6b-clone";
        RequireFiles(name);
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"qwen3-tts-base-{Guid.NewGuid():N}.slm");
        try
        {
            TtsModels.Services.GetRequiredService<ModelConverter>().Convert(TtsModels.Qwen3TtsBase, path);
            Assert.Equal(ModelKinds.SpeechSynthesis, ModelPackageReader.Open(path).Definition.Description.Kind);
            using var original = TtsModels.Load(TtsModels.Open(false), TtsModels.Qwen3TtsBase);
            using var packaged = TtsModels.Load(TtsModels.Open(false), path);
            using var a = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(original);
            using var b = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(packaged);
            var info = Info(name);
            var (options, _) = Request(name);
            options = options with { ReferenceAudio = new AudioBuffer(Floats(name, "reference"), 24000), ReferenceText = info.GetProperty("reference_text").GetString() };
            string text = info.GetProperty("text").GetString()!;
            var expected = a.BuildPrompt(text, options, TestContext.Current.CancellationToken);
            var actual = b.BuildPrompt(text, options, TestContext.Current.CancellationToken);
            Assert.Equal(expected.Rows, actual.Rows);
            Assert.Equal(expected.ReferenceFrames, actual.ReferenceFrames);
            Assert.Equal(a.GenerateFrames(expected, options, CancellationToken.None).Take(8), b.GenerateFrames(actual, options, CancellationToken.None).Take(8));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
