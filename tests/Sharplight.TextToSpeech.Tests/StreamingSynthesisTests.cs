using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Streaming synthesis of XTTS (token-chunked within a sentence) and Chatterbox (clause and sentence chunks): the chunks add up to the audio of the
/// whole-utterance path and the first one arrives early (the time to first audio goes to the parity report).
/// </summary>
public class StreamingSynthesisTests
{
    private const string Text = "The weather today is sunny with a light breeze, perfect for a walk in the park. Tomorrow it may rain.";

    private static string Root => Path.GetDirectoryName(TtsModels.Kokoro)!;

    private static string Recording => Path.Combine(Path.GetDirectoryName(Root)!, "audio", "librispeech-1272-128104-0004.wav");

    [Theory]
    [InlineData(true)]
    public async Task Xtts_streams_chunks_close_to_the_whole_decode(bool gpu)
    {
        string model = Path.Combine(Root, "XTTS-v2");
        TtsModels.Require(Path.Combine(model, "mel_stats.pth"), Recording);
        using var loaded = TtsModels.Load(TtsModels.Open(gpu), model);
        using var synthesizer = new XttsSynthesizer(loaded);
        var recording = WavFile.Read(Recording);
        var options = new SpeechSynthesisOptions { ReferenceAudio = recording, Sampling = new SamplingOptions { Seed = 7 } };
        await synthesizer.SynthesizeAsync("Warm up.", options, TestContext.Current.CancellationToken);

        var (chunks, first, total) = await Collect(synthesizer.SynthesizeStreamingAsync(Text, options, TestContext.Current.CancellationToken));
        var streamed = chunks.SelectMany(c => c).ToArray();
        var whole = (await synthesizer.SynthesizeAsync(Text, options, TestContext.Current.CancellationToken)).Audio.Samples;
        double seconds = streamed.Length / (double)synthesizer.SampleRate;
        ParityReport.Write($"XTTS streaming {(gpu ? "vulkan" : "cpu")}: {chunks.Count} chunks, {seconds:F2} s of audio, time to first audio {first:F2} s, "
            + $"whole utterance {total:F2} s; length difference to the one-shot path {streamed.Length - whole.Length} samples, relative RMS {RelativeError(whole, streamed):E2}");
        Assert.True(chunks.Count >= 3, $"{chunks.Count} chunks");
        Assert.True(first < total * 0.5, $"first audio after {first:F2} s of {total:F2} s");
        Assert.True(Math.Abs(streamed.Length - whole.Length) <= 2 * 1200, "lengths");
        Assert.True(RelativeError(whole, streamed) < 0.05, $"relative RMS {RelativeError(whole, streamed):E2}");
    }

    [Theory]
    [InlineData(true)]
    public async Task Chatterbox_streams_the_first_clause_early(bool gpu)
    {
        string model = Path.Combine(Root, "chatterbox");
        TtsModels.Require(Path.Combine(model, "ve.safetensors"), Recording);
        using var loaded = TtsModels.Load(TtsModels.Open(gpu), model);
        using var synthesizer = new ChatterboxSynthesizer(loaded);
        var options = new SpeechSynthesisOptions { ReferenceAudio = WavFile.Read(Recording), Sampling = new SamplingOptions { Seed = 7, Temperature = 0.8f, MinP = 0.05f, TopP = 1f, RepetitionPenalty = 1.2f } };
        await synthesizer.SynthesizeAsync("Warm up.", options, TestContext.Current.CancellationToken);
        var (chunks, first, total) = await Collect(synthesizer.SynthesizeStreamingAsync(Text, options, TestContext.Current.CancellationToken));
        double seconds = chunks.Sum(c => c.Length) / (double)synthesizer.SampleRate;
        ParityReport.Write($"Chatterbox streaming {(gpu ? "vulkan" : "cpu")}: {chunks.Count} chunks (clauses and sentences), {seconds:F2} s of audio, time to first audio {first:F2} s, whole utterance {total:F2} s");
        Assert.True(chunks.Count >= 3, $"{chunks.Count} chunks");
        Assert.True(first < total * 0.5, $"first audio after {first:F2} s of {total:F2} s");
    }

    private static async Task<(List<float[]> Chunks, double First, double Total)> Collect(IAsyncEnumerable<AudioBuffer> stream)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var chunks = new List<float[]>();
        double first = 0;
        await foreach (var chunk in stream)
        {
            if (chunks.Count == 0)
            {
                first = watch.Elapsed.TotalSeconds;
            }

            chunks.Add(chunk.Samples);
        }

        return (chunks, first, watch.Elapsed.TotalSeconds);
    }

    private static double RelativeError(float[] reference, float[] actual)
    {
        int n = Math.Min(reference.Length, actual.Length);
        double error = 0;
        double energy = 0;
        for (int i = 0; i < n; i++)
        {
            double d = actual[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        return Math.Sqrt(error / Math.Max(energy, 1e-30));
    }
}
