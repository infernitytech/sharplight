using Sharplight.Audio;
using Xunit;

namespace Sharplight.Core.Tests.Audio;

public class AudioResamplerTests
{
    [Theory]
    [InlineData(48000, 16000)]
    [InlineData(44100, 16000)]
    [InlineData(16000, 48000)]
    [InlineData(22050, 24000)]
    public void ToneInsideBothBandsKeepsFrequencyAndLevel(int from, int to)
    {
        const double frequency = 1000;
        var input = WavFileTests.Sine(from, 1, from / 2, frequency, amplitude: 0.5);
        var output = input.Resample(to);

        Assert.Equal(to, output.SampleRate);
        Assert.Equal((int)Math.Round((double)input.FrameCount * to / from), output.FrameCount);
        // Away from the edges the output is the same tone sampled at the new rate.
        int margin = to / 50;
        double maxError = 0;
        for (int i = margin; i < output.FrameCount - margin; i++)
        {
            double expected = 0.5 * Math.Sin(2 * Math.PI * frequency * i / to);
            maxError = Math.Max(maxError, Math.Abs(output.Samples[i] - expected));
        }

        Assert.True(maxError < 0.01, $"max error {maxError}");
    }

    [Fact]
    public void DownsamplingRemovesContentAboveTheNewNyquistFrequency()
    {
        var tone = WavFileTests.Sine(48000, 1, 24000, frequency: 11000, amplitude: 0.5);
        var output = tone.Resample(16000);
        int margin = 400;
        double rms = Math.Sqrt(output.Samples.Skip(margin).SkipLast(margin).Average(s => (double)s * s));
        Assert.True(rms < 0.01, $"aliased residue rms {rms}");
    }

    [Fact]
    public void ConstantSignalKeepsItsLevelEverywhere()
    {
        var input = new AudioBuffer(Enumerable.Repeat(0.25f, 4410).ToArray(), 44100);
        var output = input.Resample(16000);
        Assert.All(output.Samples, s => Assert.Equal(0.25f, s, 4));
    }

    [Fact]
    public void ChannelsAreResampledIndependently()
    {
        var stereo = WavFileTests.Sine(32000, 2, 3200);
        var output = stereo.Resample(16000);
        var left = new AudioBuffer([.. stereo.Samples.Where((_, i) => i % 2 == 0)], 32000).Resample(16000);
        Assert.Equal(2, output.Channels);
        for (int i = 0; i < left.FrameCount; i++)
        {
            Assert.Equal(left.Samples[i], output.Samples[i * 2], 5);
        }
    }

    [Fact]
    public void ModelInputIsMonoAtTheModelRate()
    {
        var stereo = new AudioBuffer([0.2f, 0.4f, 0.2f, 0.4f, 0.2f, 0.4f, 0.2f, 0.4f], 8000, 2);
        var mono = stereo.ToMono();
        Assert.Equal(1, mono.Channels);
        Assert.All(mono.Samples, s => Assert.Equal(0.3f, s, 5));
        Assert.Same(mono, mono.ToMono());
        Assert.Same(mono, mono.Resample(8000));
        var input = stereo.ToModelInput(16000);
        Assert.Equal((1, 16000, 8), (input.Channels, input.SampleRate, input.FrameCount));
        Assert.Equal(TimeSpan.FromMilliseconds(0.5), input.Duration);
    }

    [Fact]
    public void RejectsInvalidArguments()
    {
        Assert.Throws<ArgumentException>(() => new AudioBuffer([1f, 2f, 3f], 16000, 2));
        Assert.Throws<ArgumentException>(() => new AudioBuffer([1f], 0));
        Assert.Throws<ArgumentException>(() => AudioResampler.Resample([1f, 2f], 1, 16000, -1));
    }
}
