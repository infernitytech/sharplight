using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Dia-1.6B against transformers' <c>DiaForConditionalGeneration</c> (CPU, f32; <c>tools/make_dia_reference.py</c>; greedy decoding with classifier-free guidance, case <c>a</c> without and
/// case <c>b</c> with an audio prompt): the text tokens, the delay table, the conditional and unconditional logits along the reference's sequence (teacher forced), the greedy sequence
/// end to end, the audio prompt's codes, and the DAC decode of the reference's frames.
/// </summary>
public class DiaTests
{
    private const int Channels = 9;
    private const int Vocabulary = 1028;

    private static string Path_(string kind, string name) => Path.Combine(TtsModels.DiaReference, "dia-" + kind, name);

    private static int[] Ints(string kind, string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path_(kind, name))).ToArray();

    private static float[] Floats(string kind, string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path_(kind, name))).ToArray();

    private static JsonElement Info(string kind) => JsonDocument.Parse(File.ReadAllText(Path_(kind, "info.json"))).RootElement;

    private static List<int[]> Rows(int[] values) => [.. Enumerable.Range(0, values.Length / Channels).Select(r => values.AsSpan(r * Channels, Channels).ToArray())];

    private static void RequireFiles(string kind)
        => TtsModels.Require(Path.Combine(TtsModels.Dia, "model.safetensors"), Path.Combine(TtsModels.Dia, "dac", "model.safetensors"), Path_(kind, "info.json"));

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

    private static List<int[]>? Prompt(string kind) => kind == "b" ? Rows(Ints(kind, "prompt_codes.bin")) : null;

    private static SpeechSynthesisOptions Greedy => new() { Sampling = new SamplingOptions { Temperature = 0f, TopK = 50 }, GuidanceScale = 3f };

    [Theory]
    [InlineData("a")]
    [InlineData("b")]
    public void Text_tokens_and_delay_table_match_the_reference(string kind)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Dia);
        using var synthesizer = (DiaSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        Assert.Equal(Ints(kind, "input_ids.bin"), synthesizer.TokensFor(info.GetProperty("text").GetString()!));
        var prompt = Prompt(kind);
        var table = DiaDelay.Table(prompt, [0, 8, 9, 10, 11, 12, 13, 14, 15], 1026, 1025);
        Assert.Equal(Rows(Ints(kind, "decoder_input.bin")), table);
    }

    /// <summary>Along the reference's sequence the conditional and the unconditional logits agree with the reference's.</summary>
    [Theory]
    [InlineData("a", false)]
    [InlineData("a", true)]
    [InlineData("b", false)]
    public void Logits_match_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Dia);
        using var synthesizer = (DiaSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        var sequence = Rows(Ints(kind, "sequence.bin"));
        var reference = Floats(kind, "logits.bin");
        int perStep = 2 * Channels * Vocabulary;
        int step = 0;
        double worst = 0;
        int hard = 0;
        synthesizer.Generate(info.GetProperty("text").GetString()!, synthesizer.Settings(Greedy), Prompt(kind), sequence, (conditional, unconditional) =>
        {
            var expected = reference.AsSpan(step * perStep, perStep);
            worst = Math.Max(worst, Math.Max(RelativeRms(conditional, expected[..(Channels * Vocabulary)]), RelativeRms(unconditional, expected[(Channels * Vocabulary)..])));
            for (int c = 0; c < Channels; c++)
            {
                var row = conditional.Slice(c * Vocabulary, Vocabulary);
                var want = expected.Slice(c * Vocabulary, Vocabulary);
                int best = Sampler.ArgMax(row.ToArray());
                int wanted = Sampler.ArgMax(want.ToArray());
                if (best != wanted && Math.Abs(want[best] - want[wanted]) > 0.05f)
                {
                    hard++;
                }
            }

            step++;
        }, CancellationToken.None);
        Assert.Equal(reference.Length / perStep, step);
        ParityReport.Write($"Dia {kind} {(gpu ? "vulkan" : "cpu")}: teacher forced over {step} steps, {hard} argmax disagreements beyond a 0.05 margin, worst logits relative RMS {worst:E2}");
        Assert.Equal(0, hard);
        Assert.True(worst < (gpu ? 1e-2 : 1e-3), $"worst relative RMS {worst:E2}");
    }

    /// <summary>The greedy chain with guidance, the end tokens forced by the delay pattern and the frames after the delay are the reference's.</summary>
    [Theory]
    [InlineData("a", false)]
    [InlineData("a", true)]
    [InlineData("b", false)]
    public void Greedy_sequence_matches_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Dia);
        using var synthesizer = (DiaSynthesizer)TtsModels.Synthesizer(model);
        var info = Info(kind);
        var sequence = Rows(Ints(kind, "sequence.bin"));
        var result = synthesizer.Generate(info.GetProperty("text").GetString()!, synthesizer.Settings(Greedy), Prompt(kind), null, null, CancellationToken.None, sequence.Count);
        int same = 0;
        while (same < Math.Min(sequence.Count, result.Rows.Count) && sequence[same].SequenceEqual(result.Rows[same]))
        {
            same++;
        }

        ParityReport.Write($"Dia {kind} {(gpu ? "vulkan" : "cpu")}: greedy sequence {result.Rows.Count} rows (reference {sequence.Count}), identical prefix {same}");
        Assert.Equal(info.GetProperty("start").GetInt32(), result.Start);
        Assert.Equal(info.GetProperty("end").GetInt32(), result.End);
        Assert.Equal(sequence.Count, result.Rows.Count);
        Assert.True(same >= (gpu ? sequence.Count / 2 : sequence.Count), $"the greedy sequence diverges at row {same} of {sequence.Count}");
    }

    [Fact]
    public void Prompt_codes_match_the_reference()
    {
        RequireFiles("b");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Dia);
        using var synthesizer = (DiaSynthesizer)TtsModels.Synthesizer(model);
        var recording = new AudioBuffer(Floats("b", "prompt_audio.bin"), 44100);
        var codes = synthesizer.EncodePrompt(recording);
        var expected = Rows(Ints("b", "prompt_codes.bin"));
        Assert.Equal(expected.Count, codes.Count);
        int different = Enumerable.Range(0, codes.Count).Sum(f => codes[f].Zip(expected[f]).Count(p => p.First != p.Second));
        ParityReport.Write($"Dia prompt codes: {different} of {codes.Count * Channels} differ");
        Assert.True(different <= codes.Count * Channels / 100, $"{different} prompt codes differ");
    }

    [Theory]
    [InlineData("a", false)]
    [InlineData("a", true)]
    [InlineData("b", false)]
    public void Decoded_audio_matches_the_reference(string kind, bool gpu)
    {
        RequireFiles(kind);
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Dia);
        using var synthesizer = (DiaSynthesizer)TtsModels.Synthesizer(model);
        var sequence = Rows(Ints(kind, "sequence.bin"));
        var info = Info(kind);
        var frames = DiaDelay.Revert(sequence, [0, 8, 9, 10, 11, 12, 13, 14, 15], info.GetProperty("start").GetInt32(), info.GetProperty("end").GetInt32());
        Assert.Equal(Rows(Ints(kind, "codes.bin")), frames);
        var audio = synthesizer.Decode(frames);
        var reference = Floats(kind, "audio.bin");
        Assert.Equal(reference.Length, audio.Samples.Length);
        double error = RelativeRms(audio.Samples, reference);
        ParityReport.Write($"Dia codec {kind} {(gpu ? "vulkan" : "cpu")}: relative RMS {error:E2}");
        Assert.True(error < (gpu ? 1e-3 : 1e-4), $"relative RMS {error:E2}");
    }

    [Fact]
    public async Task Synthesizes_a_dialogue_and_clones_a_voice()
    {
        RequireFiles("b");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Dia);
        using var synthesizer = TtsModels.Synthesizer(model);
        ((DiaSynthesizer)synthesizer).MaxLength = 560;
        var cloner = (IVoiceCloner)synthesizer.GetService(typeof(IVoiceCloner))!;
        var recording = new AudioBuffer(Floats("b", "prompt_audio.bin"), 44100);
        var voice = await cloner.CreateVoiceAsync("jfk", recording, "[S1] And so my fellow Americans.", TestContext.Current.CancellationToken);
        Assert.Equal("dia", voice.Architecture);
        var options = new SpeechSynthesisOptions { VoiceProfile = voice, Sampling = new SamplingOptions { Seed = 3 } };
        var result = await synthesizer.SynthesizeAsync("[S1] Short test. [S2] Yes.", options, TestContext.Current.CancellationToken);
        Assert.Equal(44100, result.Audio.SampleRate);
        Assert.InRange(result.Audio.Samples.Length, 44100 / 2, 44100 * 40);
    }

    /// <summary>Streamed chunks are the same audio as the whole synthesis (up to the codec's edge effects at the chunk joins).</summary>
    [Fact]
    public async Task Streaming_yields_the_audio_in_chunks()
    {
        RequireFiles("a");
        using var model = TtsModels.Load(TtsModels.Open(false), TtsModels.Dia);
        using var synthesizer = TtsModels.Synthesizer(model);
        ((DiaSynthesizer)synthesizer).MaxLength = 200;
        var options = new SpeechSynthesisOptions { Sampling = new SamplingOptions { Temperature = 0f, TopK = 50 }, GuidanceScale = 3f };
        var whole = await synthesizer.SynthesizeAsync("[S1] Hi.", options, TestContext.Current.CancellationToken);
        var streamed = new List<float>();
        int chunks = 0;
        await foreach (var chunk in synthesizer.SynthesizeStreamingAsync("[S1] Hi.", options, TestContext.Current.CancellationToken))
        {
            streamed.AddRange(chunk.Samples.ToArray());
            chunks++;
        }

        Assert.Equal(whole.Audio.Samples.Length, streamed.Count);
        Assert.True(RelativeRms([.. streamed], whole.Audio.Samples) < 0.05);
        Assert.True(chunks >= 1);
    }

    /// <summary>The model converted to a <c>.slm</c> package (the language model and its codec) and loaded again generates the same rows and decodes the same audio.</summary>
    [Fact]
    public void Package_round_trip_generates_the_same_rows()
    {
        RequireFiles("a");
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "tts-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"dia-{Guid.NewGuid():N}.slm");
        try
        {
            TtsModels.Services.GetRequiredService<ModelConverter>().Convert(TtsModels.Dia, path);
            Assert.Equal(ModelKinds.SpeechSynthesis, ModelPackageReader.Open(path).Definition.Description.Kind);
            using var original = TtsModels.Load(TtsModels.Open(false), TtsModels.Dia);
            using var packaged = TtsModels.Load(TtsModels.Open(false), path);
            using var a = (DiaSynthesizer)TtsModels.Synthesizer(original);
            using var b = (DiaSynthesizer)TtsModels.Synthesizer(packaged);
            var settings = a.Settings(Greedy);
            var expected = a.Generate("[S1] Hello.", settings, null, null, null, CancellationToken.None, 40);
            var actual = b.Generate("[S1] Hello.", settings, null, null, null, CancellationToken.None, 40);
            Assert.Equal(expected.Rows, actual.Rows);
            var frames = DiaDelay.Revert(expected.Rows, [0, 8, 9, 10, 11, 12, 13, 14, 15], expected.Start, expected.End);
            Assert.Equal(a.Decode(frames).Samples.ToArray(), b.Decode(frames).Samples.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }
}
