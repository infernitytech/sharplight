using System.Globalization;
using Sharplight.Architectures;
using Sharplight.Llm;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// OuteTTS 0.2 with the WavTokenizer decoder against llama.cpp: the codec tokens of a greedy generation (llama-completion on the same prompt)
/// and the waveform of the decoder (llama.cpp's WavTokenizer graph through libllama, then the inverse STFT of its tts example in numpy).
/// </summary>
public class OuteTtsTests
{
    private const string Text = "Hello, this is a test of the speech synthesis system.";

    [Fact]
    public void Text_is_normalized_to_words()
        => Assert.Equal(["hello", "world", "forty", "two", "point", "five", "x"], OuteTtsPrompt.Words("Hello, World! 42.5 -- x_"));

    [Fact]
    public void Prompt_has_the_outetts_layout()
    {
        string prompt = OuteTtsPrompt.Build(["hello", "world"], null);
        Assert.Equal("<|im_start|>\n<|text_start|>hello<|text_sep|>world<|text_end|>\n<|audio_start|>\n", prompt);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Greedy_codes_start_like_llama_cpp(bool gpu)
    {
        TtsModels.Require(TtsModels.OuteTts, TtsModels.WavTokenizer);
        using var model = TtsModels.LoadOuteTts(TtsModels.Open(gpu));
        using var synthesizer = (OuteTtsSynthesizer)TtsModels.Synthesizer(model);
        var options = new SpeechSynthesisOptions { Sampling = SamplingOptions.Greedy };
        var codes = await synthesizer.GenerateCodesAsync(Text, options);
        // llama.cpp's first 15 codes (the chain then passes a near-tie, see the teacher-forced test).
        Assert.Equal(ReferenceCodes()[..15], codes[..15]);
        Assert.InRange(codes.Length, 120, 400);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Decoder_matches_the_reference_waveform(bool gpu)
    {
        TtsModels.Require(TtsModels.OuteTts, TtsModels.WavTokenizer);
        using var model = TtsModels.LoadOuteTts(TtsModels.Open(gpu));
        using var synthesizer = (OuteTtsSynthesizer)TtsModels.Synthesizer(model);
        var audio = synthesizer.Decode(ReferenceCodes());
        var reference = ReferenceWaveform();
        Assert.Equal(reference.Length, audio.Samples.Length);
        double error = 0;
        double energy = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = audio.Samples[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        Assert.True(Math.Sqrt(error / energy) < 2e-3, $"relative RMS error {Math.Sqrt(error / energy):E2}");
    }

    /// <summary>
    /// Codec tokens are sampled among near-ties (the runner-up logit is often within 0.01 of the winner), so two correct implementations
    /// leave a greedy chain at different points. The comparison therefore feeds llama.cpp's greedy tokens to both and checks, at every
    /// position, that the argmax agrees unless llama.cpp's own margin is below the tolerance, and that the logit of llama.cpp's token is close.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Teacher_forced_logits_match_llama_cpp(bool gpu)
    {
        TtsModels.Require(TtsModels.OuteTts, TtsModels.WavTokenizer);
        using var model = TtsModels.LoadOuteTts(TtsModels.Open(gpu));
        var llm = LlmModel.FromTaskModel(model);
        using var session = llm.CreateSession();
        var prompt = llm.Tokenizer.Encode(OuteTtsPrompt.Build(OuteTtsPrompt.Words(Text), null), addSpecial: true, parseSpecial: true);
        var reference = File.ReadAllLines(TtsModels.Fixture("outetts02-hello-greedy-q8.txt"))
            .Select(line => line.Split(' '))
            .Select(p => (Id: int.Parse(p[0], CultureInfo.InvariantCulture), Logit: float.Parse(p[1], CultureInfo.InvariantCulture),
                Margin: float.Parse(p[1], CultureInfo.InvariantCulture) - float.Parse(p[3], CultureInfo.InvariantCulture)))
            .ToArray();
        var logits = session.Process(prompt);
        int hard = 0;
        int ties = 0;
        double worst = 0;
        foreach (var (id, logit, margin) in reference)
        {
            int best = 0;
            for (int t = 1; t < logits.Length; t++)
            {
                if (logits[t] > logits[best])
                {
                    best = t;
                }
            }

            if (best != id)
            {
                if (margin > ToleranceMargin)
                {
                    hard++;
                }
                else
                {
                    ties++;
                }
            }

            worst = Math.Max(worst, Math.Abs(logits[id] - logit));
            logits = session.Process([id]);
        }

        Assert.True(hard == 0 && worst < MaxLogitDelta, $"{hard} disagreements beyond margin {ToleranceMargin}, {ties} within; worst logit delta {worst:F4} of {reference.Length} positions");
    }

    private const float ToleranceMargin = 0.1f;
    private const double MaxLogitDelta = 0.75;

    private static int[] ReferenceCodes()
        => [.. File.ReadAllText(TtsModels.Fixture("outetts02-hello-codes.txt")).Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s, CultureInfo.InvariantCulture))];

    private static float[] ReferenceWaveform()
    {
        var bytes = File.ReadAllBytes(TtsModels.Fixture("outetts02-hello-reference.s16"));
        var samples = new float[bytes.Length / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = BitConverter.ToInt16(bytes, i * 2) / 32767f;
        }

        return samples;
    }
}
