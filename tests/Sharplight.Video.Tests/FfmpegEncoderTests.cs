using Microsoft.Extensions.Options;
using Sharplight.Audio;

namespace Sharplight.Video.Tests;

public sealed class FfmpegEncoderTests
{
    /// <summary>Every software codec of every format with the PSNR its round trip must reach (lossless codecs must be exact).</summary>
    public static TheoryData<string, string, string, double> SoftwareCodecs => new()
    {
        { VideoFormats.Mp4, VideoCodecs.H264, ".mp4", 30 },
        { VideoFormats.Mp4, VideoCodecs.Av1, ".mp4", 30 },
        { VideoFormats.Mp4, VideoCodecs.Vp9, ".mp4", 30 },
        { VideoFormats.WebM, VideoCodecs.Vp9, ".webm", 30 },
        { VideoFormats.WebM, VideoCodecs.Vp8, ".webm", 30 },
        { VideoFormats.WebM, VideoCodecs.Av1, ".webm", 30 },
        { VideoFormats.Mov, VideoCodecs.ProRes, ".mov", 30 },
        { VideoFormats.Mov, VideoCodecs.H264, ".mov", 30 },
        { VideoFormats.Matroska, VideoCodecs.Ffv1, ".mkv", double.PositiveInfinity },
        { VideoFormats.Matroska, VideoCodecs.H264, ".mkv", 30 },
        { VideoFormats.Apng, VideoCodecs.Apng, ".apng", double.PositiveInfinity },
        { VideoFormats.Gif, VideoCodecs.Gif, ".gif", 25 },
    };

    [Theory]
    [MemberData(nameof(SoftwareCodecs))]
    public async Task RoundTripKeepsSizeFrameCountAndQuality(string format, string codec, string extension, double minimumPsnr)
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(64, 48, 10);
        string path = TestClips.TempPath(extension);
        try
        {
            var encoder = new FfmpegVideoEncoder(format);
            await new VideoEncoderCatalog([encoder]).SaveAsync(clip, path, new VideoEncodeOptions { Codec = codec, Quality = 95 }, TestContext.Current.CancellationToken);

            (string _, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal((64, 48, 10), (width, height, frames));
            List<byte[]> decoded = FfmpegReference.Decode(path);
            for (int i = 0; i < 10; i++)
            {
                double psnr = TestClips.Psnr(decoded[i], clip.Frames[i].Rgb);
                Assert.True(psnr >= minimumPsnr, $"{format}/{codec} frame {i}: {psnr:F1} dB");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task OddSizesArePaddedForChromaSubsampledCodecs()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(33, 17, 6, 23.976);
        string path = TestClips.TempPath(".mp4");
        try
        {
            await using (FileStream file = File.Create(path))
            {
                await new FfmpegVideoEncoder(VideoFormats.Mp4).EncodeAsync(clip, file, new VideoEncodeOptions { BitrateKilobitsPerSecond = 400 }, TestContext.Current.CancellationToken);
            }

            VideoInfo info = await new FfmpegVideoDecoder().ProbeAsync(path, TestContext.Current.CancellationToken);
            Assert.Equal(("h264", 34, 18), (info.VideoCodec, info.Width, info.Height));
            Assert.Equal(23.976, info.FramesPerSecond, 3);
            Assert.Equal(6, FfmpegReference.Decode(path).Count);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task NonSeekableOutputWritesAFragmentedMp4()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(32, 32, 5);
        using var buffer = new MemoryStream();
        await new FfmpegVideoEncoder(VideoFormats.Mp4).EncodeAsync(clip, new ForwardOnlyStream(buffer), null, TestContext.Current.CancellationToken);

        string path = TestClips.TempPath(".mp4");
        try
        {
            await File.WriteAllBytesAsync(path, buffer.ToArray(), TestContext.Current.CancellationToken);
            Assert.Equal(5, FfmpegReference.Decode(path).Count);
            Assert.True(buffer.ToArray().AsSpan().IndexOf("moof"u8) >= 0, "no movie fragment");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(VideoFormats.Mp4, VideoCodecs.Aac, "aac")]
    [InlineData(VideoFormats.Mp4, VideoCodecs.Opus, "opus")]
    [InlineData(VideoFormats.WebM, VideoCodecs.Opus, "opus")]
    [InlineData(VideoFormats.Mov, VideoCodecs.Pcm, "pcm_s16le")]
    [InlineData(VideoFormats.Matroska, VideoCodecs.Flac, "flac")]
    public async Task AudioTrackIsWrittenWithTheClipsDuration(string format, string audioCodec, string expectedCodec)
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        // 2 s of video, 1.5 s of a 440 Hz tone: the track is padded with silence to the video's length.
        VideoClip clip = TestClips.MovingGradient(32, 32, 20) with { Audio = Tone(440, 1.5, 44_100) };
        string extension = new FfmpegVideoEncoder(format).FileExtensions[0];
        string path = TestClips.TempPath(extension);
        try
        {
            await new VideoEncoderCatalog([new FfmpegVideoEncoder(format)]).SaveAsync(clip, path, new VideoEncodeOptions { AudioCodec = audioCodec }, TestContext.Current.CancellationToken);

            var decoder = new FfmpegVideoDecoder();
            VideoInfo info = await decoder.ProbeAsync(path, TestContext.Current.CancellationToken);
            Assert.Equal(expectedCodec, info.AudioCodec);
            AudioBuffer audio = (await decoder.ReadAudioAsync(path, 48_000, TestContext.Current.CancellationToken))!;
            Assert.InRange(audio.Duration.TotalSeconds, 1.95, 2.1);
            double toneEnergy = Rms(audio, 0.2, 1.3);
            double silenceEnergy = Rms(audio, 1.6, 1.9);
            Assert.InRange(toneEnergy, 0.25, 0.45);
            Assert.True(silenceEnergy < 0.02, $"silence rms {silenceEnergy}");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task OptionsSelectPresetPixelFormatRateFactorAndKeyframes()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(64, 48, 12);
        string path = TestClips.TempPath(".webm");
        try
        {
            var options = new VideoEncodeOptions
            {
                Codec = VideoCodecs.Vp9,
                Preset = VideoPresets.Fastest,
                PixelFormat = "yuv444p",
                ConstantRateFactor = 20,
                KeyframeInterval = 4,
                FramesPerSecond = 24,
            };
            await new VideoEncoderCatalog([new FfmpegVideoEncoder(VideoFormats.WebM)]).SaveAsync(clip, path, options, TestContext.Current.CancellationToken);

            VideoInfo info = await new FfmpegVideoDecoder().ProbeAsync(path, TestContext.Current.CancellationToken);
            Assert.Equal(("vp9", 24.0), (info.VideoCodec, Math.Round(info.FramesPerSecond)));
            Assert.Equal(0.5, info.Duration.TotalSeconds, 2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task InvalidOptionsAreRejectedWithTheChoices()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        var encoder = new FfmpegVideoEncoder(VideoFormats.WebM);
        VideoClip clip = TestClips.MovingGradient(16, 16, 2);
        CancellationToken token = TestContext.Current.CancellationToken;

        var codec = await Assert.ThrowsAsync<ArgumentException>(() => encoder.EncodeAsync(clip, new MemoryStream(), new VideoEncodeOptions { Codec = VideoCodecs.ProRes }, token));
        var preset = await Assert.ThrowsAsync<ArgumentException>(() => encoder.EncodeAsync(clip, new MemoryStream(), new VideoEncodeOptions { Preset = "ludicrous" }, token));
        var factor = await Assert.ThrowsAsync<ArgumentException>(() => encoder.EncodeAsync(clip, new MemoryStream(), new VideoEncodeOptions { ConstantRateFactor = 99 }, token));
        var audio = await Assert.ThrowsAsync<ArgumentException>(() => new FfmpegVideoEncoder(VideoFormats.Gif).EncodeAsync(clip, new MemoryStream(), new VideoEncodeOptions { AudioCodec = VideoCodecs.Aac }, token));

        Assert.Contains("vp9", codec.Message, StringComparison.Ordinal);
        Assert.Contains("fastest", preset.Message, StringComparison.Ordinal);
        Assert.Contains("[0, 63]", factor.Message, StringComparison.Ordinal);
        Assert.Contains("no audio", audio.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void CodecsDescribeTheChoicesDefaultFirst()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        IReadOnlyList<VideoCodecInfo> mp4 = new FfmpegVideoEncoder(VideoFormats.Mp4).Codecs;
        IReadOnlyList<VideoCodecInfo> mov = new FfmpegVideoEncoder(VideoFormats.Mov).Codecs;
        IReadOnlyList<VideoCodecInfo> software = new FfmpegVideoEncoder(VideoFormats.Mp4, Options.Create(new SharplightVideoOptions { EnableHardwareEncoders = false })).Codecs;

        Assert.Equal([VideoCodecs.H264, VideoCodecs.Av1, VideoCodecs.Vp9], software.Select(c => c.Id));
        Assert.Equal(VideoCodecs.H264, mp4[0].Id);
        Assert.Equal([VideoCodecs.Aac, VideoCodecs.Opus], mp4[0].AudioCodecs);
        VideoCodecInfo av1 = mp4.Single(c => c.Id == VideoCodecs.Av1);
        Assert.Equal((1, 63), av1.ConstantRateFactorRange);
        Assert.Equal([VideoPresets.Fastest, VideoPresets.Fast, VideoPresets.Medium, VideoPresets.Slow, VideoPresets.Slowest], av1.Presets);
        Assert.Equal(VideoCodecs.ProRes, mov[0].Id);
        Assert.Equal("hq", mov[0].DefaultPreset);
        Assert.All(mp4.Where(c => c.IsHardware), c => Assert.Contains(c.Id, new[] { VideoCodecs.H264Nvenc, VideoCodecs.HevcNvenc, VideoCodecs.Av1Nvenc, VideoCodecs.H264VideoToolbox, VideoCodecs.HevcVideoToolbox, VideoCodecs.H264MediaFoundation, VideoCodecs.HevcMediaFoundation }));
    }

    [Fact]
    public async Task HardwareEncodersThatProbeAsUsableEncode()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        var encoder = new FfmpegVideoEncoder(VideoFormats.Mp4);
        foreach (VideoCodecInfo codec in encoder.Codecs.Where(c => c.IsHardware))
        {
            string path = TestClips.TempPath(".mp4");
            try
            {
                Task save = new VideoEncoderCatalog([encoder]).SaveAsync(TestClips.MovingGradient(256, 192, 8), path, new VideoEncodeOptions { Codec = codec.Id }, TestContext.Current.CancellationToken);
                Task finished = await Task.WhenAny(save, Task.Delay(TimeSpan.FromMinutes(2), TestContext.Current.CancellationToken));
                Assert.True(ReferenceEquals(finished, save), $"Encoding with {codec.Id} did not finish within two minutes.");
                await save;
                List<byte[]> frames = FfmpegReference.Decode(path);
                Assert.Equal(8, frames.Count);
            }
            finally
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public async Task CancellationStopsTheEncode()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        using var cancellation = new CancellationTokenSource();
        await cancellation.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new FfmpegVideoEncoder(VideoFormats.WebM).EncodeAsync(TestClips.MovingGradient(), new MemoryStream(), null, cancellation.Token));
    }

    private static AudioBuffer Tone(double frequency, double seconds, int rate)
    {
        int count = (int)(seconds * rate);
        float[] samples = new float[count * 2];
        for (int i = 0; i < count; i++)
        {
            float value = (float)(0.5 * Math.Sin(2 * Math.PI * frequency * i / rate));
            samples[2 * i] = value;
            samples[2 * i + 1] = value;
        }

        return new AudioBuffer(samples, rate, 2);
    }

    private static double Rms(AudioBuffer audio, double from, double to)
    {
        int first = (int)(from * audio.SampleRate) * audio.Channels;
        int last = Math.Min(audio.Samples.Length, (int)(to * audio.SampleRate) * audio.Channels);
        double sum = 0;
        for (int i = first; i < last; i++)
        {
            sum += audio.Samples[i] * (double)audio.Samples[i];
        }

        return Math.Sqrt(sum / Math.Max(1, last - first));
    }
}
