using System.Runtime.InteropServices;
using Sharplight.Architectures;
using Sharplight.Import.Kokoro;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Kokoro-82M against the kokoro Python package (torch): the weight-normalized checkpoint import, the predicted durations and F0 curve, and the waveform with the
/// excitation noise of the reference run fed to both.
/// </summary>
public class KokoroTests
{
    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(TtsModels.KokoroReference, name + ".bin"))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(TtsModels.KokoroReference, name + ".bin"))).ToArray();

    [Fact]
    public void The_importer_reads_weights_voices_and_the_vocabulary()
    {
        TtsModels.Require(Path.Combine(TtsModels.Kokoro, "config.json"), Path.Combine(TtsModels.Kokoro, "kokoro-v1_0.pth"));
        Assert.True(new KokoroImporter().CanOpen(TtsModels.Kokoro));
        var source = KokoroImporter.Open(TtsModels.Kokoro);
        try
        {
            Verify(source);
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    private static void Verify(Sharplight.Models.ModelSource source)
    {
        Assert.Equal("kokoro", source.Architecture);
        Assert.Equal(512, source.Arch<int>("hidden_size"));
        Assert.Contains("decoder.generator.resblocks.5.convs2.2.weight", source.Weights.Keys);
        Assert.DoesNotContain("decoder.generator.resblocks.5.convs2.2.weight_g", source.Weights.Keys);
        Assert.Equal(new Sharplight.Tensors.Shape(1024, 1090, 3), source.Weights["decoder.decode.0.conv1.weight"].Shape);
        Assert.Contains(KokoroImporter.VocabularyAttachment, source.Attachments.Keys);
        Assert.Contains(source.Attachments.Keys, k => k == "voices/af_heart.f32");
        Assert.Equal(510L * 256 * sizeof(float), source.Attachments["voices/af_heart.f32"].ByteLength);
        Assert.Equal(source.Weights["predictor.lstm.weight_ih_l0"].Shape, new Sharplight.Tensors.Shape(1024, 640));
        Assert.Contains("predictor.lstm.bias_l0_reverse", source.Weights.Keys);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void The_waveform_matches_the_reference(bool gpu)
    {
        TtsModels.Require(Path.Combine(TtsModels.Kokoro, "kokoro-v1_0.pth"), Path.Combine(TtsModels.KokoroReference, "audio.bin"));
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Kokoro);
        using var synthesizer = (KokoroSynthesizer)TtsModels.Synthesizer(model);
        string phonemes = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(TtsModels.KokoroReference, "kokoro.json"))).RootElement.GetProperty("phonemes").GetString()!;
        var noise = Floats("noise");
        synthesizer.NoiseSource = _ => noise;
        var run = synthesizer.Run(phonemes, "af_heart");

        Assert.Equal(Ints("ids"), run.Ids);
        Assert.Equal(Ints("pred_dur"), run.Durations);
        var rawReference = Floats("duration");
        for (int i = 0; i < rawReference.Length; i++)
        {
            Assert.InRange(Math.Abs(run.RawDurations[i] - rawReference[i]), 0, 5e-3);
        }

        var f0 = Floats("F0");
        Assert.Equal(f0.Length, run.F0.Length);
        Assert.True(RelativeError(f0, run.F0) < 1e-4, $"F0 relative RMS error {RelativeError(f0, run.F0):E2}");
        var reference = Floats("audio");
        Assert.Equal(reference.Length, run.Samples.Length);
        double error = RelativeError(reference, run.Samples);
        ParityReport.Write($"Kokoro {(gpu ? "vulkan" : "cpu")}: waveform relative RMS error {error:E2}, F0 {RelativeError(f0, run.F0):E2}");
        Assert.True(error < 3e-2, $"waveform relative RMS error {error:E2}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Text_is_spoken_sentence_by_sentence_through_the_front_end(bool gpu)
    {
        TtsModels.Require(Path.Combine(TtsModels.Kokoro, "kokoro-v1_0.pth"), Path.Combine(TtsModels.Kokoro, "lexicon", "us_gold.json"));
        using var model = TtsModels.Load(TtsModels.Open(gpu), TtsModels.Kokoro);
        using var synthesizer = TtsModels.Synthesizer(model);
        Assert.Contains("af_heart", synthesizer.Voices);
        var options = new Sharplight.Speech.SpeechSynthesisOptions { Voice = "af_heart" };
        var buffers = new List<Sharplight.Audio.AudioBuffer>();
        await foreach (var buffer in synthesizer.SynthesizeStreamingAsync("Hello world. How are you today?", options, TestContext.Current.CancellationToken))
        {
            buffers.Add(buffer);
        }

        Assert.Equal(2, buffers.Count);
        Assert.All(buffers, buffer => Assert.Equal(24000, buffer.SampleRate));
        Assert.InRange(buffers.Sum(b => b.Samples.Length) / 24000.0, 1.5, 6.0);
        Assert.Contains(buffers[0].Samples, sample => MathF.Abs(sample) > 0.05f);
        var faster = await synthesizer.SynthesizeAsync("Hello world.", options with { Speed = 1.5f }, TestContext.Current.CancellationToken);
        var normal = await synthesizer.SynthesizeAsync("Hello world.", options, TestContext.Current.CancellationToken);
        Assert.True(faster.Audio.Samples.Length < normal.Audio.Samples.Length * 0.8, "speed 1.5 should shorten the audio");
    }

    private static double RelativeError(float[] reference, float[] actual)
    {
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
