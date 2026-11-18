using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Qwen3-TTS 0.6B CustomVoice against the official qwen_tts package (CPU, f32; <c>tools/tts-reference/reference_qwen3_tts.py</c>, greedy talker and code predictor,
/// non-streaming text): the text tokens and the talker's prompt rows, the talker's and the code predictor's logits along the reference's frames (teacher forced),
/// the greedy frames (up to the reference's first near tie), and the decoded audio of the reference's frames.
/// </summary>
public class Qwen3TtsTests
{
    private const float NearTie = 0.01f;

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(TtsModels.Qwen3TtsReference, name + ".bin"))).ToArray();

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(TtsModels.Qwen3TtsReference, name + ".bin"))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(TtsModels.Qwen3TtsReference, "info.json"))).RootElement;

    private static void RequireFiles()
        => TtsModels.Require(Path.Combine(TtsModels.Qwen3Tts, "model.safetensors"), Path.Combine(TtsModels.Qwen3Tts, "speech_tokenizer", "model.safetensors"),
            Path.Combine(TtsModels.Qwen3TtsReference, "codes.bin"));

    private static List<int[]> ReferenceFrames()
    {
        var codes = Ints("codes");
        return [.. Enumerable.Range(0, codes.Length / 16).Select(f => codes.AsSpan(f * 16, 16).ToArray())];
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

    [Fact]
    public void Prompt_matches_the_reference()
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Qwen3Tts);
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var info = Info();
        string text = info.GetProperty("text").GetString()!;
        Assert.Equal(info.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()), synthesizer.TokensFor(text));
        var prompt = synthesizer.PromptFor(text, info.GetProperty("speaker").GetString(), info.GetProperty("language").GetString());
        var reference = Floats("prompt");
        Assert.Equal(reference.Length, prompt.Length);
        double error = RelativeRms(prompt, reference);
        ParityReport.Write($"Qwen3-TTS prompt rows: relative RMS {error:E2}");
        Assert.True(error < 1e-5, $"relative RMS {error:E2}");
    }

    /// <summary>
    /// Along the reference's frames, the talker's first-codebook choice and the code predictor's choices agree with the reference's unless its own margin is small, and the
    /// greedy frames are the reference's.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Logits_and_greedy_frames_match_the_reference(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Qwen3Tts);
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var info = Info();
        string text = info.GetProperty("text").GetString()!;
        var options = new SpeechSynthesisOptions
        {
            Voice = info.GetProperty("speaker").GetString(), Language = info.GetProperty("language").GetString(), NormalizeText = false,
            Sampling = new SamplingOptions { Temperature = 0f, RepetitionPenalty = 1.05f, RepetitionWindow = 8192 },
        };
        var frames = ReferenceFrames();
        var top2 = Ints("top2_ids");
        var top2Values = Floats("top2_values");
        var predictorTop2 = Ints("cp_top2_ids");
        var predictorValues = Floats("cp_top2_values");
        int step = 0;
        int hard = 0;
        int predictorHard = 0;
        double worst = 0;
        foreach (var _ in synthesizer.GenerateFrames(text, options, CancellationToken.None, frames, (logits, predictor) =>
        {
            var masked = logits.ToArray();
            int best = Sampler.ArgMax(masked);
            if (best != top2[2 * step] && top2Values[2 * step] - top2Values[(2 * step) + 1] > 0.05f)
            {
                hard++;
            }

            worst = Math.Max(worst, Math.Abs(masked[top2[2 * step]] - top2Values[2 * step]));
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

        Assert.Equal(frames.Count + 1, step);
        ParityReport.Write($"Qwen3-TTS {(gpu ? "vulkan" : "cpu")}: teacher forced over {frames.Count} frames, talker disagreements {hard}, code predictor {predictorHard}, worst top logit error {worst:E2}");
        Assert.Equal(0, hard);
        Assert.Equal(0, predictorHard);
        Assert.True(worst < 0.05, $"worst top logit error {worst:E2}");

        var greedy = synthesizer.GenerateFrames(text, options, CancellationToken.None).ToList();
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

        ParityReport.Write($"Qwen3-TTS {(gpu ? "vulkan" : "cpu")}: greedy frames {greedy.Count} (reference {frames.Count}), identical prefix {same}, first reference near tie at frame {tie}");
        Assert.True(same >= tie, $"greedy frames diverge at {same}, before the reference's first near tie at {tie}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoded_audio_matches_the_reference(bool gpu)
    {
        RequireFiles();
        TtsModels.Require(Path.Combine(TtsModels.Qwen3TtsReference, "audio.bin"));
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Qwen3Tts);
        using var synthesizer = (Qwen3TtsSynthesizer)TtsModels.Synthesizer(model);
        var audio = synthesizer.Decode(ReferenceFrames());
        var reference = Floats("audio");
        Assert.Equal(reference.Length, audio.Samples.Length);
        double error = RelativeRms(audio.Samples, reference);
        ParityReport.Write($"Qwen3-TTS codec {(gpu ? "vulkan" : "cpu")}: relative RMS {error:E2}");
        Assert.True(error < 1e-4, $"relative RMS {error:E2}");
    }

    [Fact]
    public async Task Synthesizes_speech_with_every_voice_listed()
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Qwen3Tts);
        using var synthesizer = TtsModels.Synthesizer(model);
        Assert.Contains("ryan", synthesizer.Voices);
        var result = await synthesizer.SynthesizeAsync("Hi there.", new SpeechSynthesisOptions { Voice = "vivian", Language = "english", Sampling = new SamplingOptions { Seed = 1 } },
            TestContext.Current.CancellationToken);
        Assert.Equal(24000, result.Audio.SampleRate);
        Assert.InRange(result.Audio.Samples.Length, 4800, 24000 * 10);
    }
}
