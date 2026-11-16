using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Llm;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// XTTS v2 (GPT-2 audio-code model, HiFi-GAN decoder, a built-in speaker) against the coqui-tts package (reference tensors from
/// <c>tools/make_xtts_reference.py</c>, float32 on the CPU): the text tokens, the GPT's greedy codes teacher forced (the best code agrees wherever the
/// reference's margin is not tiny), the latents of the reference codes and the decoder's waveform for the reference latents, on the CPU and on Vulkan.
/// </summary>
public class XttsTests
{
    private static string Model => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "XTTS-v2");

    private static string Reference => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "xtts-ref");

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Reference, name + ".bin"))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Reference, name + ".bin"))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Reference, "xtts.json"))).RootElement;

    private static void RequireAll() => TtsModels.Require(Path.Combine(Model, "model.pth"), Path.Combine(Reference, "audio.bin"));

    /// <summary>Text cleaning and tokenization against coqui-tts's <c>VoiceBpeTokenizer.encode(text, "en")</c> (abbreviation, numbers, punctuation).</summary>
    [Fact]
    public void Text_tokens_match_the_reference()
    {
        TtsModels.Require(Path.Combine(Model, "model.pth"));
        using var model = TtsModels.Load(TtsModels.Open(gpu: false), Model);
        using var synthesizer = (XttsSynthesizer)TtsModels.Synthesizer(model);
        Assert.Equal(
            [261, 259, 42, 2, 194, 91, 24, 2, 243, 190, 2, 182, 37, 2, 1081, 26, 1093, 2, 5555, 44, 2, 42, 2, 494, 2457, 2, 6064, 7, 2, 53, 2, 17, 1024, 1176, 2, 3139, 22, 40, 2, 1899, 17, 2, 232, 50, 210, 8, 840, 76, 2, 134, 84, 59, 32, 2, 182, 31, 2, 40, 2948, 2, 155, 178, 61, 9, 0],
            synthesizer.TextTokens("The quick brown fox jumps over the lazy dog, and Dr. Smith paid 25 dollars for 3 apples."));
        Assert.Equal(
            [261, 259, 120, 59, 178, 343, 583, 2, 1523, 213, 32, 2, 103, 2, 42, 2, 484, 91, 18, 2, 58, 2, 37, 33, 192, 7, 2, 28, 111, 2, 174, 126, 2, 48, 2, 14, 2, 78, 80, 9, 0],
            synthesizer.TextTokens("Sharplight speaks with the voice of XTTS, one code at a time."));
        Assert.Equal(
            [261, 259, 41, 2, 232, 28, 2, 40, 206, 53, 2, 53, 2, 232, 50, 210, 8, 182, 121, 7, 2, 28, 111, 2, 2870, 481, 49, 2, 53, 2, 232, 50, 210, 8, 840, 76, 2, 124, 161, 64, 2, 1899, 17, 2, 28, 111, 2, 40, 206, 53, 2, 53, 2, 28, 111, 2, 134, 84, 59, 32, 2, 53, 2, 182, 31, 210, 2, 124, 31, 1217, 33, 2, 726, 18, 9, 0],
            synthesizer.TextTokens("In 2024, 125 people paid 1,001 dollars & 40% more."));
    }

    /// <summary>The whole pipeline through the task client with a built-in speaker: audio of plausible length, and the real-time factor in the report.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Synthesizes_speech_end_to_end(bool gpu)
    {
        TtsModels.Require(Path.Combine(Model, "model.pth"), Path.Combine(Model, "speakers_xtts.pth"));
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = TtsModels.Synthesizer(model);
        Assert.Contains("Claribel Dervla", synthesizer.Voices);
        var options = new Sharplight.Speech.SpeechSynthesisOptions
        {
            Voice = "Claribel Dervla",
            Sampling = new SamplingOptions { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 10f },
        };
        const string Text = "Sharplight speaks with the voice of XTTS, one code at a time.";
        await synthesizer.SynthesizeAsync(Text, options, TestContext.Current.CancellationToken);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var result = await synthesizer.SynthesizeAsync(Text, options, TestContext.Current.CancellationToken);
        double seconds = result.Audio.Samples.Length / (double)result.Audio.SampleRate;
        ParityReport.Write($"XTTS end to end {(gpu ? "vulkan" : "cpu")}: {seconds:F2} s of audio in {stopwatch.Elapsed.TotalSeconds:F2} s (real-time factor {stopwatch.Elapsed.TotalSeconds / seconds:F2})");
        Assert.InRange(seconds, 1.5, 15);
        Assert.Contains(result.Audio.Samples, v => MathF.Abs(v) > 0.05f);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Greedy_codes_and_latents_match_the_reference(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = (XttsSynthesizer)TtsModels.Synthesizer(model);
        var info = Info();
        var tokens = synthesizer.TextTokens(info.GetProperty("text").GetString()!);
        Assert.Equal(Ints("text_tokens"), tokens);
        var greedy = new SamplingOptions { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 10f };
        var (codes, latents) = synthesizer.Generate(tokens, synthesizer.LoadVoice(info.GetProperty("speaker").GetString()!), greedy, TestContext.Current.CancellationToken);
        var reference = Ints("codes");
        var margins = Floats("margins");
        int prefix = 0;
        while (prefix < Math.Min(codes.Length, reference.Length) && codes[prefix] == reference[prefix])
        {
            prefix++;
        }

        string difference = prefix < reference.Length ? $" (first difference where the reference's margin is {margins[prefix]:E2})" : "";
        ParityReport.Write($"XTTS {(gpu ? "vulkan" : "cpu")}: {prefix} of {reference.Length} greedy codes identical{difference}");
        Assert.True(prefix == reference.Length || margins[prefix] < 0.05f, $"greedy codes differ at {prefix} where the reference's margin is {margins[prefix]:E2}");
        if (prefix == reference.Length && codes.Length == reference.Length)
        {
            double error = RelativeError(Floats("latents"), latents);
            ParityReport.Write($"XTTS {(gpu ? "vulkan" : "cpu")}: latents relative RMS error {error:E2}");
            Assert.True(error < 1e-3, $"latents relative RMS error {error:E2}");
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoder_matches_the_reference(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = (XttsSynthesizer)TtsModels.Synthesizer(model);
        var samples = synthesizer.Decode(Floats("latents"), synthesizer.LoadVoice(Info().GetProperty("speaker").GetString()!));
        var reference = Floats("audio");
        Assert.Equal(reference.Length, samples.Length);
        double error = RelativeError(reference, samples);
        ParityReport.Write($"XTTS decoder {(gpu ? "vulkan" : "cpu")}: waveform relative RMS error {error:E2}");
        Assert.True(error < 1e-3, $"waveform relative RMS error {error:E2}");
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

        return Math.Sqrt(error / energy);
    }
}
