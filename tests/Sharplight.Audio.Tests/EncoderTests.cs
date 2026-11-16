namespace Sharplight.Audio.Tests;

public sealed class EncoderTests
{
    private readonly NativeAudioEncoder _encoder = new();
    private readonly NativeAudioDecoder _decoder = new();

    [Fact]
    public void Float32WavRoundTripIsExact()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        var audio = new AudioBuffer(TestSignals.Tone(22050, 2, 5000), 22050, 2);
        byte[] bytes = _encoder.Encode(audio, new AudioEncodeOptions { Format = AudioEncodingFormat.WavFloat32 });
        AudioBuffer decoded = _decoder.Decode(bytes);
        Assert.Equal(audio.Samples, decoded.Samples);
        Assert.Equal(22050, decoded.SampleRate);

        // The managed WavFile reads the native encoder's output too.
        using var stream = new MemoryStream(bytes);
        Assert.Equal(audio.Samples, WavFile.Read(stream).Samples);
    }

    [Theory]
    [InlineData(AudioEncodingFormat.WavPcm16, 85)]
    [InlineData(AudioEncodingFormat.WavPcm24, 130)]
    public void IntegerWavRoundTripIsClose(AudioEncodingFormat format, double minimumSnr)
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        var audio = new AudioBuffer(TestSignals.Tone(16000, 1, 4000), 16000, 1);
        AudioBuffer decoded = _decoder.Decode(_encoder.Encode(audio, new AudioEncodeOptions { Format = format }));
        Assert.True(TestSignals.SnrDb(audio.Samples, decoded.Samples) >= minimumSnr);
    }

    [Theory]
    [InlineData(AudioEncodingFormat.WavPcm16, WavSampleFormat.Pcm16)]
    [InlineData(AudioEncodingFormat.WavPcm24, WavSampleFormat.Pcm24)]
    [InlineData(AudioEncodingFormat.WavFloat32, WavSampleFormat.Float32)]
    public void NativeWavIsByteIdenticalToTheManagedWriter(AudioEncodingFormat native, WavSampleFormat managedFormat)
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        float[] tone = TestSignals.Tone(16000, 2, 3000);
        float[] edges = [1f, -1f, 1.5f, -1.5f, 0.99999f, -0.99999f, 0.5f / 32768f, 1.5f / 32768f, 2.5f / 32768f, -0.5f / 32768f, -1.5f / 32768f, 0f];
        var audio = new AudioBuffer([.. tone, .. edges], 16000, 2);
        using var managed = new MemoryStream();
        WavFile.Write(managed, audio, managedFormat);
        byte[] actual = _encoder.Encode(audio, new AudioEncodeOptions { Format = native });
        Assert.Equal(managed.ToArray(), actual);
    }

    [Theory]
    [InlineData(48000, 2)]
    [InlineData(48000, 1)]
    [InlineData(16000, 1)]
    [InlineData(44100, 2)]
    public void OpusRoundTripKeepsTheSignal(int rate, int channels)
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        var audio = new AudioBuffer(TestSignals.Tone(rate, channels, rate), rate, channels);
        byte[] bytes = _encoder.Encode(audio, new AudioEncodeOptions { Format = AudioEncodingFormat.OggOpus, BitrateBitsPerSecond = 96000 });
        Assert.True(bytes.Length is > 1000 and < 60000);
        Assert.Equal("OggS", System.Text.Encoding.ASCII.GetString(bytes, 0, 4));

        AudioBuffer decoded = _decoder.Decode(bytes);
        Assert.Equal(channels, decoded.Channels);
        Assert.Equal(rate == 44100 ? 48000 : rate == 16000 ? 48000 : rate, decoded.SampleRate);
        AudioBuffer expected = audio.SampleRate == decoded.SampleRate ? audio : audio.Resample(decoded.SampleRate);
        Assert.Equal(expected.FrameCount, decoded.FrameCount);
        double snr = TestSignals.SnrDb(expected.Samples, decoded.Samples);
        Assert.True(snr > 10, $"SNR {snr:F1} dB");
    }

    [Fact]
    public void OpusFileSaveAndReadBack()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        string path = Path.Combine(Path.GetTempPath(), $"sl-test-{Guid.NewGuid():N}.opus");
        try
        {
            var audio = new AudioBuffer(TestSignals.Tone(48000, 1, 24000), 48000, 1);
            _encoder.Save(audio, path, new AudioEncodeOptions { Format = AudioEncodingFormat.OggOpus });
            using IAudioReader reader = _decoder.OpenReader(path);
            Assert.Equal("opus", reader.Info.Codec);
            Assert.Equal(24000, reader.Info.TotalFrames);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RejectsUnsupportedConfigurations()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        var surround = new AudioBuffer(new float[480 * 3], 48000, 3);
        Assert.Throws<NotSupportedException>(() => _encoder.Encode(surround, new AudioEncodeOptions { Format = AudioEncodingFormat.OggOpus }));
    }

    [Fact]
    public void ConvertToMixesChannelsAndRates()
    {
        var mono = new AudioBuffer([0.1f, 0.2f, 0.3f], 16000, 1);
        AudioBuffer stereo = mono.ConvertTo(16000, 2);
        Assert.Equal([0.1f, 0.1f, 0.2f, 0.2f, 0.3f, 0.3f], stereo.Samples);
        Assert.Equal([0.1f, 0.2f, 0.3f], stereo.ConvertTo(16000, 1).Samples, new LooseComparer());
        Assert.Same(mono, mono.ConvertTo(16000, 1));
    }

    private sealed class LooseComparer : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => Math.Abs(x - y) < 1e-6f;

        public int GetHashCode(float obj) => 0;
    }
}
