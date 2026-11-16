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
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_KOKORO_DEBUG") == "1")
        {
            string directory = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_KOKORO_DEBUG_ROOT") ?? ".", gpu ? "mine_gpu" : "mine_cpu");
            Directory.CreateDirectory(directory);
            Environment.SetEnvironmentVariable("SHARPLIGHT_KOKORO_DEBUG_DIR", directory);
        }

        var run = synthesizer.Run(phonemes, "af_heart");

        Assert.Equal(Ints("ids"), run.Ids);
        Assert.Equal(Ints("pred_dur"), run.Durations);
        var rawReference = Floats("duration");
        for (int i = 0; i < rawReference.Length; i++)
        {
            Assert.InRange(Math.Abs(run.RawDurations[i] - rawReference[i]), 0, 2e-3);
        }

        var f0 = Floats("F0");
        Assert.Equal(f0.Length, run.F0.Length);
        Assert.True(RelativeError(f0, run.F0) < 2e-3, $"F0 relative RMS error {RelativeError(f0, run.F0):E2}");
        var reference = Floats("audio");
        Assert.Equal(reference.Length, run.Samples.Length);
        double error = RelativeError(reference, run.Samples);
        Assert.True(error < 3e-2, $"waveform relative RMS error {error:E2}");
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
