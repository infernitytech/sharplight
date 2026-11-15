namespace Sharplight.Audio.Tests;

public sealed class DecoderTests
{
    private readonly NativeAudioDecoder _decoder = new();

    public static TheoryData<string, string, int, double> Fixtures => new()
    {
        { "tone_s16.wav", "wav", 16000, 70 },
        { "tone_s24.wav", "wav", 16000, 100 },
        { "tone_f32.wav", "wav", 16000, 120 },
        { "tone.flac", "flac", 16000, 70 },
        { "tone.mp3", "mp3", 16000, 18 },
        { "tone.ogg", "vorbis", 16000, 18 },
        { "tone.opus", "opus", 48000, 14 },
    };

    public static TheoryData<string, string> StreamingFixtures => new()
    {
        { "tone_s16.wav", "wav" },
        { "tone.flac", "flac" },
        { "tone.mp3", "mp3" },
        { "tone.ogg", "vorbis" },
        { "tone.opus", "opus" },
    };

    private static string Path(string name) => System.IO.Path.Combine(TestSignals.FixtureDirectory, name);

    [Fact]
    public void LibraryIsPresentOnCi()
    {
        if (Environment.GetEnvironmentVariable("GITHUB_ACTIONS") == "true")
        {
            Assert.True(Native.NativeAudioLibrary.IsAvailable, Native.NativeAudioLibrary.UnavailableReason);
        }
    }

    [Theory]
    [MemberData(nameof(Fixtures))]
    public void DecodesFixtureCloseToTheSourceSignal(string file, string codec, int rate, double minimumSnr)
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        using IAudioReader reader = _decoder.OpenReader(Path(file));
        Assert.Equal(codec, reader.Info.Codec);
        Assert.Equal(rate, reader.Info.SampleRate);
        Assert.Equal(2, reader.Info.Channels);

        AudioBuffer decoded = _decoder.Decode(Path(file));
        Assert.Equal(rate, decoded.SampleRate);
        Assert.InRange(decoded.Duration.TotalSeconds, 0.49, 0.6);

        // Opus is decoded at 48 kHz; compare against the signal sampled at that rate, others at 16 kHz.
        float[] reference = TestSignals.Fixture(rate, decoded.FrameCount);
        double snr = TestSignals.BestSnrDb(reference, decoded.Samples, 2, codec is "wav" or "flac" ? 0 : 2500, out int lag);
        Assert.True(snr >= minimumSnr, $"{file}: SNR {snr:F1} dB at lag {lag} is below {minimumSnr} dB");
        if (codec is "wav" or "flac")
        {
            Assert.Equal(0, lag);
        }
    }

    [Theory]
    [MemberData(nameof(StreamingFixtures))]
    public void StreamingReadsEqualTheWholeDecodeAndSeekWorks(string file, string codec)
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        AudioBuffer whole = _decoder.Decode(Path(file));
        using IAudioReader reader = _decoder.OpenReader(Path(file));
        var chunk = new float[1000 * 2];
        var collected = new List<float>();
        int frames;
        while ((frames = reader.Read(chunk)) > 0)
        {
            collected.AddRange(chunk.AsSpan(0, frames * 2).ToArray());
        }

        Assert.Equal(whole.Samples.Length, collected.Count);
        Assert.Equal(whole.Samples, collected.ToArray());

        reader.Seek(500);
        var after = new float[20];
        Assert.Equal(10, reader.Read(after));
        if (codec is "wav" or "flac" or "opus")
        {
            Assert.Equal(whole.Samples.AsSpan(1000, 20).ToArray(), after, new FloatComparer(1e-4f));
        }
    }

    [Fact]
    public void DecodesFromMemory()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        byte[] bytes = File.ReadAllBytes(Path("tone.flac"));
        AudioBuffer fromMemory = _decoder.Decode(bytes);
        AudioBuffer fromFile = _decoder.Decode(Path("tone.flac"));
        Assert.Equal(fromFile.Samples, fromMemory.Samples);
    }

    [Fact]
    public void MatchesFfmpegWhenAvailable()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        foreach (string file in new[] { "tone_s16.wav", "tone.flac", "tone.mp3", "tone.ogg", "tone.opus" })
        {
            float[]? reference = FfmpegReference.Decode(Path(file));
            if (reference is null)
            {
                Assert.Skip("ffmpeg not installed");
            }

            AudioBuffer decoded = _decoder.Decode(Path(file));
            double snr = TestSignals.BestSnrDb(reference, decoded.Samples, 2, 2500, out int lag);
            bool lossless = file.EndsWith(".wav") || file.EndsWith(".flac");
            Assert.True(snr > (lossless ? 90 : 25), $"{file}: SNR against ffmpeg {snr:F1} dB at lag {lag}");
        }
    }

    [Fact]
    public void ReportsErrorsAsExceptions()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        Assert.Throws<FileNotFoundException>(() => _decoder.Decode("/nonexistent/none.wav"));
        Assert.Throws<InvalidDataException>(() => _decoder.Decode(new byte[100]));
        Assert.Throws<InvalidDataException>(() => _decoder.Decode(ReadOnlyMemory<byte>.Empty));
    }

    private sealed class FloatComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;

        public int GetHashCode(float obj) => 0;
    }
}
