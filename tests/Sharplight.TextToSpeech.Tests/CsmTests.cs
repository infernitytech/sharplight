using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// CSM-1B against transformers' <c>CsmForConditionalGeneration</c> (CPU, f32; <c>tools/make_csm_reference.py</c>, greedy backbone and depth decoder, no audio context): the text
/// tokens, the backbone's and the depth decoder's logits along the reference's frames (teacher forced), the greedy frames, and the Mimi decode of the reference's frames.
/// </summary>
public class CsmTests
{
    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(TtsModels.CsmReference, name + ".bin"))).ToArray();

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(TtsModels.CsmReference, name + ".bin"))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(TtsModels.CsmReference, "info.json"))).RootElement;

    private static void RequireFiles()
        => TtsModels.Require(Path.Combine(TtsModels.Csm, "model.safetensors"), Path.Combine(TtsModels.CsmReference, "codes.bin"));

    private static List<int[]> ReferenceFrames()
    {
        var codes = Ints("codes");
        return [.. Enumerable.Range(0, codes.Length / 32).Select(f => codes.AsSpan(f * 32, 32).ToArray())];
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
    public void Text_tokens_match_the_reference()
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Csm);
        using var synthesizer = (CsmSynthesizer)TtsModels.Synthesizer(model);
        var info = Info();
        Assert.Equal(info.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()), synthesizer.TokensFor(info.GetProperty("text").GetString()!, info.GetProperty("speaker").GetString()));
    }

    /// <summary>Along the reference's frames, the backbone's and the depth decoder's choices agree with the reference's unless its own margin is small, and the greedy frames are its.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Logits_and_greedy_frames_match_the_reference(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Csm);
        using var synthesizer = (CsmSynthesizer)TtsModels.Synthesizer(model);
        var info = Info();
        string text = info.GetProperty("text").GetString()!;
        var options = new SpeechSynthesisOptions { Voice = info.GetProperty("speaker").GetString(), NormalizeText = false, Sampling = new SamplingOptions { Temperature = 0f } };
        var frames = ReferenceFrames();
        var top2 = Ints("top2_ids");
        var top2Values = Floats("top2_values");
        var depthTop2 = Ints("depth_top2_ids");
        var depthValues = Floats("depth_top2_values");
        var referenceHidden = Floats("hidden");
        int step = 0;
        int hard = 0;
        int depthHard = 0;
        double worst = 0;
        double hiddenError = 0;
        foreach (var _ in synthesizer.GenerateFrames(text, options, CancellationToken.None, frames, (logits, hidden, depth) =>
        {
            int best = Sampler.ArgMax(logits);
            if (best != top2[2 * step] && top2Values[2 * step] - top2Values[(2 * step) + 1] > 0.05f)
            {
                hard++;
            }

            worst = Math.Max(worst, Math.Abs(logits[top2[2 * step]] - top2Values[2 * step]));
            hiddenError = Math.Max(hiddenError, RelativeRms(hidden, referenceHidden.AsSpan(step * hidden.Length, hidden.Length)));
            for (int g = 0; g < depth.Count; g++)
            {
                int row = (step * 31) + g;
                if (Sampler.ArgMax(depth[g]) != depthTop2[2 * row] && depthValues[2 * row] - depthValues[(2 * row) + 1] > 0.05f)
                {
                    depthHard++;
                }

                worst = Math.Max(worst, Math.Abs(depth[g][depthTop2[2 * row]] - depthValues[2 * row]));
            }

            step++;
        }))
        {
        }

        Assert.Equal(frames.Count, step);
        ParityReport.Write($"CSM {(gpu ? "vulkan" : "cpu")}: teacher forced over {frames.Count} frames, backbone disagreements {hard}, depth decoder {depthHard}, worst top logit error {worst:E2}, hidden state relative RMS {hiddenError:E2}");
        Assert.Equal(0, hard);
        Assert.Equal(0, depthHard);
        Assert.True(worst < 0.1, $"worst top logit error {worst:E2}");

        var greedy = synthesizer.GenerateFrames(text, options, CancellationToken.None).ToList();
        int same = 0;
        while (same < Math.Min(greedy.Count, frames.Count) && greedy[same].SequenceEqual(frames[same]))
        {
            same++;
        }

        int tie = 0;
        while (tie < frames.Count && top2Values[2 * tie] - top2Values[(2 * tie) + 1] >= 0.01f
               && Enumerable.Range(0, 31).All(g => depthValues[2 * ((tie * 31) + g)] - depthValues[(2 * ((tie * 31) + g)) + 1] >= 0.01f))
        {
            tie++;
        }

        ParityReport.Write($"CSM {(gpu ? "vulkan" : "cpu")}: greedy frames {greedy.Count} (reference {frames.Count - 1}), identical prefix {same}, first reference near tie at frame {tie}");
        Assert.True(same >= tie, $"greedy frames diverge at {same}, before the reference's first near tie at {tie}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoded_audio_matches_the_reference(bool gpu)
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Csm);
        using var synthesizer = (CsmSynthesizer)TtsModels.Synthesizer(model);
        var audio = synthesizer.Decode(ReferenceFrames());
        var reference = Floats("audio");
        Assert.Equal(reference.Length, audio.Samples.Length);
        double error = RelativeRms(audio.Samples, reference);
        ParityReport.Write($"CSM codec {(gpu ? "vulkan" : "cpu")}: relative RMS {error:E2}");
        Assert.True(error < (gpu ? 1e-3 : 1e-4), $"relative RMS {error:E2}");
    }

    [Fact]
    public async Task Synthesizes_speech()
    {
        RequireFiles();
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Csm);
        using var synthesizer = TtsModels.Synthesizer(model);
        var result = await synthesizer.SynthesizeAsync("Hi there.", new SpeechSynthesisOptions { Voice = "0", Sampling = new SamplingOptions { Seed = 1 } }, TestContext.Current.CancellationToken);
        Assert.Equal(24000, result.Audio.SampleRate);
        Assert.InRange(result.Audio.Samples.Length, 4800, 24000 * 10);
    }

    /// <summary>The model converted to a <c>.slm</c> package and loaded again generates the same frames.</summary>
    [Fact]
    public void Package_round_trip_generates_the_same_frames()
    {
        RequireFiles();
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"csm-{Guid.NewGuid():N}.slm");
        try
        {
            TtsModels.Services.GetRequiredService<ModelConverter>().Convert(TtsModels.Csm, path);
            Assert.Equal(ModelKinds.SpeechSynthesis, ModelPackageReader.Open(path).Definition.Description.Kind);
            var options = new SpeechSynthesisOptions { Voice = "0", NormalizeText = false, Sampling = new SamplingOptions { Temperature = 0f } };
            using var original = TtsModels.Load(TtsModels.Open(false), TtsModels.Csm);
            using var packaged = TtsModels.Load(TtsModels.Open(false), path);
            using var a = (CsmSynthesizer)TtsModels.Synthesizer(original);
            using var b = (CsmSynthesizer)TtsModels.Synthesizer(packaged);
            string text = "Hello from Sesame.";
            var expected = a.GenerateFrames(text, options, CancellationToken.None).Take(6).ToList();
            var actual = b.GenerateFrames(text, options, CancellationToken.None).Take(6).ToList();
            Assert.Equal(6, actual.Count);
            for (int i = 0; i < 6; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }

            Assert.Equal(a.Decode(expected).Samples.ToArray(), b.Decode(actual).Samples.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }
}
