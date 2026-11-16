namespace Sharplight.Video.Tests;

public sealed class FfmpegDecoderTests
{
    [Fact]
    public async Task ProbeReportsTheStreams()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        string path = await EncodeAsync(TestClips.MovingGradient(64, 48, 20, 10), ".mkv", VideoCodecs.Ffv1);
        try
        {
            VideoInfo info = await new FfmpegVideoDecoder().ProbeAsync(path, TestContext.Current.CancellationToken);

            Assert.Equal(("matroska,webm", "ffv1", 64, 48), (info.Container, info.VideoCodec, info.Width, info.Height));
            Assert.Equal(10, info.FramesPerSecond, 3);
            Assert.Equal(2, info.Duration.TotalSeconds, 1);
            Assert.Null(info.AudioCodec);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task FramesCarryTimestampsAndExactPixelsOfALosslessFile()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(64, 48, 20, 10);
        string path = await EncodeAsync(clip, ".mkv", VideoCodecs.Ffv1);
        try
        {
            List<VideoFrame> frames = await ReadAsync(path, null);

            Assert.Equal(20, frames.Count);
            Assert.Equal(Enumerable.Range(0, 20), frames.Select(f => f.Index));
            Assert.Equal(1.5, frames[15].Timestamp.TotalSeconds, 3);
            Assert.True(frames[15].ToImage().Rgb.AsSpan().SequenceEqual(clip.Frames[15].Rgb));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task RangeLimitRateAndSizeOptionsApply()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(64, 48, 20, 10);
        string path = await EncodeAsync(clip, ".mkv", VideoCodecs.Ffv1);
        try
        {
            List<VideoFrame> range = await ReadAsync(path, new VideoDecodeOptions { Start = TimeSpan.FromSeconds(0.5), Duration = TimeSpan.FromSeconds(0.5) });
            List<VideoFrame> limited = await ReadAsync(path, new VideoDecodeOptions { MaxFrames = 3 });
            List<VideoFrame> doubled = await ReadAsync(path, new VideoDecodeOptions { FramesPerSecond = 20 });
            List<VideoFrame> halved = await ReadAsync(path, new VideoDecodeOptions { FramesPerSecond = 5 });
            List<VideoFrame> scaled = await ReadAsync(path, new VideoDecodeOptions { Width = 32, MaxFrames = 1 });

            Assert.Equal(5, range.Count);
            Assert.True(range[0].Data.AsSpan().SequenceEqual(clip.Frames[5].Rgb));
            Assert.Equal(0.5, range[0].Timestamp.TotalSeconds, 3);
            Assert.Equal(3, limited.Count);
            Assert.Equal(40, doubled.Count);
            Assert.True(doubled[7].Data.AsSpan().SequenceEqual(clip.Frames[3].Rgb));
            Assert.Equal(10, halved.Count);
            Assert.True(halved[3].Data.AsSpan().SequenceEqual(clip.Frames[6].Rgb));
            Assert.Equal((32, 24), (scaled[0].Width, scaled[0].Height));
            Assert.Equal(32 * 24 * 3, scaled[0].Data.Length);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task Yuv420FramesHaveThreePlanes()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        string path = await EncodeAsync(TestClips.MovingGradient(33, 17, 3), ".webm", VideoCodecs.Vp9);
        try
        {
            List<VideoFrame> frames = await ReadAsync(path, new VideoDecodeOptions { Format = VideoFrameFormat.Yuv420 });

            Assert.Equal(3, frames.Count);
            Assert.Equal(33 * 17 + 2 * 17 * 9, frames[0].Data.Length);
            Assert.Throws<InvalidOperationException>(() => frames[0].ToImage());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task CatalogReadsAClipWithItsAudio()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        var audio = new Audio.AudioBuffer(new float[48_000], 48_000, 1);
        VideoClip clip = TestClips.MovingGradient(32, 32, 10) with { Audio = audio };
        string path = TestClips.TempPath(".webm");
        try
        {
            await using (FileStream file = File.Create(path))
            {
                await new FfmpegVideoEncoder(VideoFormats.WebM).EncodeAsync(clip, file, null, TestContext.Current.CancellationToken);
            }

            var catalog = new VideoDecoderCatalog([new FfmpegVideoDecoder()]);

            VideoClip read = await catalog.ReadClipAsync(path, cancellationToken: TestContext.Current.CancellationToken);

            Assert.Equal(10, read.Frames.Count);
            Assert.Equal(10, read.FramesPerSecond, 3);
            Assert.NotNull(read.Audio);
            Assert.InRange(read.Audio.Duration.TotalSeconds, 0.95, 1.05);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task UnreadableFilesThrowInvalidData()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        string path = TestClips.TempPath(".mp4");
        try
        {
            await File.WriteAllBytesAsync(path, "not a video at all"u8.ToArray(), TestContext.Current.CancellationToken);

            await Assert.ThrowsAsync<InvalidDataException>(() => new FfmpegVideoDecoder().ProbeAsync(path, TestContext.Current.CancellationToken));
            await Assert.ThrowsAsync<InvalidDataException>(() => ReadAsync(path, null));
            await Assert.ThrowsAsync<FileNotFoundException>(() => new FfmpegVideoDecoder().ProbeAsync(path + ".missing", TestContext.Current.CancellationToken));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task LeavingTheEnumerationEarlyStopsTheDecoder()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        string path = await EncodeAsync(TestClips.MovingGradient(64, 48, 40), ".mkv", VideoCodecs.Ffv1);
        try
        {
            int count = 0;
            await foreach (VideoFrame frame in new FfmpegVideoDecoder().ReadFramesAsync(path, null, TestContext.Current.CancellationToken))
            {
                count++;
                if (frame.Index == 2)
                {
                    break;
                }
            }

            Assert.Equal(3, count);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static async Task<string> EncodeAsync(VideoClip clip, string extension, string codec)
    {
        string path = TestClips.TempPath(extension);
        var catalog = new VideoEncoderCatalog([.. FfmpegVideoEncoder.Formats.Select(f => new FfmpegVideoEncoder(f))]);
        await catalog.SaveAsync(clip, path, new VideoEncodeOptions { Codec = codec, Quality = 95 }, TestContext.Current.CancellationToken);
        return path;
    }

    private static async Task<List<VideoFrame>> ReadAsync(string path, VideoDecodeOptions? options)
    {
        var frames = new List<VideoFrame>();
        await foreach (VideoFrame frame in new FfmpegVideoDecoder().ReadFramesAsync(path, options, TestContext.Current.CancellationToken))
        {
            frames.Add(frame);
        }

        return frames;
    }
}
