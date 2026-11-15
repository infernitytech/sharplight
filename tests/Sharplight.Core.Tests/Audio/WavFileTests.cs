using System.Buffers.Binary;
using Sharplight.Audio;
using Xunit;

namespace Sharplight.Core.Tests.Audio;

public class WavFileTests
{
    [Theory]
    [InlineData(WavSampleFormat.Pcm16, 1, 1.0 / 32767)]
    [InlineData(WavSampleFormat.Pcm16, 2, 1.0 / 32767)]
    [InlineData(WavSampleFormat.Pcm24, 1, 1.0 / 8388607)]
    [InlineData(WavSampleFormat.Pcm24, 2, 1.0 / 8388607)]
    [InlineData(WavSampleFormat.Pcm32, 1, 1e-7)]
    [InlineData(WavSampleFormat.Pcm32, 2, 1e-7)]
    [InlineData(WavSampleFormat.Float32, 1, 0)]
    [InlineData(WavSampleFormat.Float32, 2, 0)]
    public void RoundTripKeepsRateChannelsAndSamples(WavSampleFormat format, int channels, double tolerance)
    {
        var audio = Sine(sampleRate: 22050, channels, frames: 1001);
        using var stream = new MemoryStream();
        WavFile.Write(stream, audio, format);
        stream.Position = 0;
        var read = WavFile.Read(stream);

        Assert.Equal(audio.SampleRate, read.SampleRate);
        Assert.Equal(channels, read.Channels);
        Assert.Equal(audio.FrameCount, read.FrameCount);
        for (int i = 0; i < audio.Samples.Length; i++)
        {
            Assert.True(Math.Abs(audio.Samples[i] - read.Samples[i]) <= tolerance + 1e-9, $"sample {i}: {audio.Samples[i]} vs {read.Samples[i]}");
        }
    }

    [Fact]
    public void IntegerFormatsClipOutOfRangeSamples()
    {
        var audio = new AudioBuffer([1.5f, -2f, 0.5f], 8000);
        using var stream = new MemoryStream();
        WavFile.Write(stream, audio, WavSampleFormat.Pcm16);
        stream.Position = 0;
        var read = WavFile.Read(stream);
        Assert.Equal(1f, read.Samples[0], 3);
        Assert.Equal(-1f, read.Samples[1], 3);
        Assert.Equal(0.5f, read.Samples[2], 3);
    }

    [Fact]
    public void ReadsExtensibleFormatAndSkipsUnknownChunks()
    {
        short[] pcm = [0, 16384, -16384, 32767];
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.ASCII, leaveOpen: true))
        {
            writer.Write("RIFF"u8);
            writer.Write(0);
            writer.Write("WAVE"u8);
            writer.Write("fmt "u8);
            writer.Write(40);
            writer.Write((ushort)0xFFFE);
            writer.Write((ushort)1);
            writer.Write(16000);
            writer.Write(32000);
            writer.Write((ushort)2);
            writer.Write((ushort)16);
            writer.Write((ushort)22);
            writer.Write((ushort)16);
            writer.Write(4);
            writer.Write((ushort)1); // KSDATAFORMAT_SUBTYPE_PCM starts with the plain format tag
            writer.Write(new byte[14]);
            writer.Write("LIST"u8);
            writer.Write(3);
            writer.Write(new byte[4]); // odd chunk plus its pad byte
            writer.Write("data"u8);
            writer.Write(pcm.Length * 2);
            foreach (short sample in pcm)
            {
                writer.Write(sample);
            }
        }

        stream.Position = 0;
        var read = WavFile.Read(stream);
        Assert.Equal(16000, read.SampleRate);
        Assert.Equal([0f, 0.5f, -0.5f, 32767 / 32768f], read.Samples);
    }

    [Fact]
    public void RejectsFilesThatAreNotWave()
    {
        var bytes = new byte[44];
        "RIFX"u8.CopyTo(bytes);
        Assert.Throws<InvalidDataException>(() => WavFile.Read(new MemoryStream(bytes)));
        "RIFF"u8.CopyTo(bytes);
        "AVI "u8.CopyTo(bytes.AsSpan(8));
        Assert.Throws<InvalidDataException>(() => WavFile.Read(new MemoryStream(bytes)));
    }

    [Fact]
    public void WritesTheCanonicalHeader()
    {
        using var stream = new MemoryStream();
        WavFile.Write(stream, new AudioBuffer(new float[6], 48000, 2), WavSampleFormat.Pcm24);
        var bytes = stream.ToArray();
        Assert.Equal(44 + 18, bytes.Length);
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(20)));
        Assert.Equal(2, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(22)));
        Assert.Equal(48000, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(24)));
        Assert.Equal(48000 * 6, BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(28)));
        Assert.Equal(24, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(34)));
    }

    internal static AudioBuffer Sine(int sampleRate, int channels, int frames, double frequency = 440, double amplitude = 0.8)
    {
        var samples = new float[frames * channels];
        for (int frame = 0; frame < frames; frame++)
        {
            for (int channel = 0; channel < channels; channel++)
            {
                double phase = 2 * Math.PI * frequency * frame / sampleRate + channel;
                samples[frame * channels + channel] = (float)(amplitude * Math.Sin(phase));
            }
        }

        return new AudioBuffer(samples, sampleRate, channels);
    }
}
