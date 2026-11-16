using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>Demucs (htdemucs) against the demucs package's HTDemucs with the same weights: the stems of a 10 s stereo mix, split into overlapping segments.</summary>
public class DemucsTests
{
    private const string Model = "htdemucs";

    private static double Sdr(ReadOnlySpan<float> estimate, ReadOnlySpan<float> reference)
    {
        double signal = 0;
        double noise = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            signal += (double)reference[i] * reference[i];
            double d = estimate[i] - reference[i];
            noise += d * d;
        }

        return 10 * Math.Log10(Math.Max(signal, 1e-20) / Math.Max(noise, 1e-20));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Stems_match_the_reference(bool gpu)
    {
        AudioModels.Require(AudioModels.Model(Model), AudioModels.Reference("demucs_in.f32"), AudioModels.Reference("demucs_out.f32"));
        using var model = AudioModels.Load(AudioModels.Open(gpu), AudioModels.Model(Model));
        using var separator = AudioModels.Client<IAudioSeparator>(model);
        Assert.Equal(44100, separator.SampleRate);
        Assert.Equal(["drums", "bass", "other", "vocals"], separator.Stems);

        // The reference tensors are channel-major [channels, samples] (mixture) and [stems, channels, samples].
        var planar = AudioModels.Floats("demucs_in.f32");
        int length = planar.Length / 2;
        var interleaved = new float[planar.Length];
        for (int i = 0; i < length; i++)
        {
            interleaved[2 * i] = planar[i];
            interleaved[(2 * i) + 1] = planar[length + i];
        }

        var stems = await separator.SeparateAsync(new AudioBuffer(interleaved, 44100, 2));
        var expected = AudioModels.Floats("demucs_out.f32");
        Assert.Equal(4, stems.Count);
        for (int s = 0; s < 4; s++)
        {
            Assert.Equal(length, stems[s].Audio.FrameCount);
            for (int c = 0; c < 2; c++)
            {
                var estimate = new float[length];
                for (int i = 0; i < length; i++)
                {
                    estimate[i] = stems[s].Audio.Samples[(2 * i) + c];
                }

                double sdr = Sdr(estimate, expected.AsSpan(((s * 2) + c) * length, length));
                Assert.True(sdr > 35, $"{stems[s].Name} channel {c}: SDR against the reference {sdr:0.0} dB");
            }
        }
    }

    [Fact]
    public void Spectral_transform_inverts_itself()
    {
        var signal = new float[20000];
        var random = new Random(1);
        for (int i = 0; i < signal.Length; i++)
        {
            signal[i] = (float)random.NextDouble() - 0.5f;
        }

        var transform = new SpectralTransform(4096, 1024);
        int frames = transform.Frames(signal.Length);
        var re = new float[transform.Bins * frames];
        var im = new float[re.Length];
        transform.Forward(signal, re, im);
        var back = transform.Inverse(re, im, frames, signal.Length);
        Assert.True(AudioModels.RelativeRms(back, signal) < 1e-5);
    }
}
