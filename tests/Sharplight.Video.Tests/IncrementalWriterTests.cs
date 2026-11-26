using Sharplight.Audio;
using Sharplight.Vision;

namespace Sharplight.Video.Tests;

/// <summary>
/// Incremental encodes (<see cref="IVideoFrameWriter"/>): the same output as a clip encode, a valid file after any number of frames, and the
/// sound track of a source file copied or transcoded, starting at an offset and ending with the video.
/// </summary>
public sealed class IncrementalWriterTests
{
    [Theory]
    [InlineData(VideoFormats.Matroska, VideoCodecs.Ffv1, ".mkv")]
    [InlineData(VideoFormats.Mp4, VideoCodecs.H264, ".mp4")]
    [InlineData(VideoFormats.WebM, VideoCodecs.Vp9, ".webm")]
    public async Task WriterEncodesLikeAClipEncode(string format, string codec, string extension)
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        CancellationToken token = TestContext.Current.CancellationToken;
        VideoClip clip = TestClips.MovingGradient(64, 48, 12);
        var options = new VideoEncodeOptions { Codec = codec, Quality = 90, MaxThreads = 1 };
        var catalog = new VideoEncoderCatalog([new FfmpegVideoEncoder(format)]);
        string whole = TestClips.TempPath(extension);
        string streamed = TestClips.TempPath(extension);
        try
        {
            await catalog.SaveAsync(clip, whole, options, token);
            await using (IVideoFrameWriter writer = catalog.OpenWriter(streamed, Settings(clip), options))
            {
                foreach (ImageInput frame in clip.Frames)
                {
                    await writer.WriteAsync(frame, token);
                }

                Assert.Equal(12, writer.FramesWritten);
                await writer.CompleteAsync(token);
            }

            List<byte[]> expected = FfmpegReference.Decode(whole);
            List<byte[]> actual = FfmpegReference.Decode(streamed);
            Assert.Equal(expected.Count, actual.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }
        finally
        {
            File.Delete(whole);
            File.Delete(streamed);
        }
    }

    [Fact]
    public async Task CompletingEarlyLeavesAValidVideoOfTheFramesWritten()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        CancellationToken token = TestContext.Current.CancellationToken;
        string source = await SourceAsync(".mp4", VideoCodecs.Aac, TestClips.Tone(440, 3, 44_100), token);
        string path = TestClips.TempPath(".mp4");
        try
        {
            VideoClip clip = TestClips.MovingGradient(32, 32, 30);
            await using (IVideoFrameWriter writer = Catalog(VideoFormats.Mp4).OpenWriter(path, Settings(clip) with { AudioSourcePath = source }))
            {
                // A run cancelled after 12 of 30 frames finishes what it has.
                foreach (ImageInput frame in clip.Frames.Take(12))
                {
                    await writer.WriteAsync(frame, token);
                }

                await writer.CompleteAsync(token);
            }

            Assert.Equal(12, FfmpegReference.Decode(path).Count);
            AudioBuffer audio = (await new FfmpegVideoDecoder().ReadAudioAsync(path, 48_000, token))!;
            Assert.InRange(audio.Duration.TotalSeconds, 1.15, 1.25);
        }
        finally
        {
            File.Delete(source);
            File.Delete(path);
        }
    }

    /// <summary>
    /// The sound of a source file starting at 1 s (1 s of silence, then a tone): copied when the output format takes its codec (Opus from
    /// Matroska into MP4, AAC from MP4 into Matroska), transcoded to the format's codec otherwise (AAC into WebM: Opus).
    /// </summary>
    [Theory]
    [InlineData(".mkv", VideoCodecs.Opus, VideoFormats.Mp4, "opus")]
    [InlineData(".mp4", VideoCodecs.Aac, VideoFormats.Matroska, "aac")]
    [InlineData(".mp4", VideoCodecs.Aac, VideoFormats.WebM, "opus")]
    public async Task SourceAudioIsCopiedOrTranscodedFromItsStartToTheVideosEnd(string sourceExtension, string sourceCodec, string format, string expectedCodec)
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        CancellationToken token = TestContext.Current.CancellationToken;
        string source = await SourceAsync(sourceExtension, sourceCodec, SilenceThenTone(1, 3, 48_000), token);
        string path = TestClips.TempPath(new FfmpegVideoEncoder(format).FileExtensions[0]);
        try
        {
            VideoClip clip = TestClips.MovingGradient(32, 32, 20);
            await using (IVideoFrameWriter writer = Catalog(format).OpenWriter(path, Settings(clip) with { AudioSourcePath = source, AudioSourceStart = TimeSpan.FromSeconds(1) }))
            {
                foreach (ImageInput frame in clip.Frames)
                {
                    await writer.WriteAsync(frame, token);
                }

                await writer.CompleteAsync(token);
            }

            var decoder = new FfmpegVideoDecoder();
            VideoInfo info = await decoder.ProbeAsync(path, token);
            Assert.Equal(expectedCodec, info.AudioCodec);
            Assert.Equal(20, FfmpegReference.Decode(path).Count);
            AudioBuffer audio = (await decoder.ReadAudioAsync(path, 48_000, token))!;
            Assert.InRange(audio.Duration.TotalSeconds, 1.9, 2.1);
            // The track starts at the tone (the source's first second is skipped).
            Assert.InRange(TestClips.Rms(audio, 0.1, 1.8), 0.25, 0.45);
        }
        finally
        {
            File.Delete(source);
            File.Delete(path);
        }
    }

    [Fact]
    public async Task ASourceWithoutSoundOrIncludeAudioOffGivesASilentVideo()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        CancellationToken token = TestContext.Current.CancellationToken;
        VideoClip clip = TestClips.MovingGradient(32, 32, 10);
        string silent = TestClips.TempPath(".mp4");
        string withSound = await SourceAsync(".mp4", VideoCodecs.Aac, TestClips.Tone(440, 1, 44_100), token);
        string first = TestClips.TempPath(".mp4");
        string second = TestClips.TempPath(".mp4");
        try
        {
            await Catalog(VideoFormats.Mp4).SaveAsync(clip, silent, null, token);
            await WriteAllAsync(Catalog(VideoFormats.Mp4).OpenWriter(first, Settings(clip) with { AudioSourcePath = silent }), clip, token);
            await WriteAllAsync(Catalog(VideoFormats.Mp4).OpenWriter(second, Settings(clip) with { AudioSourcePath = withSound }, new VideoEncodeOptions { IncludeAudio = false }), clip, token);

            var decoder = new FfmpegVideoDecoder();
            Assert.Null((await decoder.ProbeAsync(first, token)).AudioCodec);
            Assert.Null((await decoder.ProbeAsync(second, token)).AudioCodec);
            Assert.Equal(10, FfmpegReference.Decode(first).Count);
        }
        finally
        {
            File.Delete(silent);
            File.Delete(withSound);
            File.Delete(first);
            File.Delete(second);
        }
    }

    [Fact]
    public async Task ACatalogWriterDisposedWithoutCompletingDeletesItsFile()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        CancellationToken token = TestContext.Current.CancellationToken;
        VideoClip clip = TestClips.MovingGradient(32, 32, 4);
        string path = TestClips.TempPath(".webm");
        await using (IVideoFrameWriter writer = Catalog(VideoFormats.WebM).OpenWriter(path, Settings(clip)))
        {
            await writer.WriteAsync(clip.Frames[0], token);
            Assert.True(File.Exists(path));
            await Assert.ThrowsAsync<ArgumentException>(async () => await writer.WriteAsync(new ImageInput(16, 16, new byte[16 * 16 * 3]), token));
        }

        Assert.False(File.Exists(path));
    }

    [Fact]
    public void GifIsWrittenAsAWholeClipOnly()
    {
        var gif = new FfmpegVideoEncoder(VideoFormats.Gif);
        Assert.False(gif.SupportsIncrementalWriting);
        Assert.True(new FfmpegVideoEncoder(VideoFormats.Mp4).SupportsIncrementalWriting);
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        string path = TestClips.TempPath(".gif");
        Assert.Throws<NotSupportedException>(() => Catalog(VideoFormats.Gif).OpenWriter(path, new VideoStreamSettings { Width = 8, Height = 8, FramesPerSecond = 10 }));
        Assert.False(File.Exists(path));
    }

    [Fact]
    public async Task MemoryDoesNotGrowWithTheFramesWritten()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            return;
        }

        CancellationToken token = TestContext.Current.CancellationToken;
        string path = TestClips.TempPath(".mkv");
        var settings = new VideoStreamSettings { Width = 160, Height = 96, FramesPerSecond = 30 };
        try
        {
            await using IVideoFrameWriter writer = Catalog(VideoFormats.Matroska).OpenWriter(path, settings, new VideoEncodeOptions { Codec = VideoCodecs.Ffv1 });
            long afterWarmUp = 0;
            for (int i = 0; i < 600; i++)
            {
                var rgb = new byte[settings.Width * settings.Height * 3];
                rgb.AsSpan().Fill((byte)i);
                await writer.WriteAsync(new ImageInput(settings.Width, settings.Height, rgb), token);
                if (i == 99)
                {
                    afterWarmUp = Managed();
                }
            }

            long growth = Managed() - afterWarmUp;
            await writer.CompleteAsync(token);
            // 500 more frames of 46 KB each would be 23 MB if they were kept.
            Assert.True(growth < 4 << 20, $"managed memory grew by {growth / 1024} KB over 500 frames");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static long Managed()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        return GC.GetTotalMemory(forceFullCollection: true);
    }

    private static VideoEncoderCatalog Catalog(string format) => new([new FfmpegVideoEncoder(format)]);

    private static VideoStreamSettings Settings(VideoClip clip) => new() { Width = clip.Width, Height = clip.Height, FramesPerSecond = clip.FramesPerSecond };

    private static async Task WriteAllAsync(IVideoFrameWriter writer, VideoClip clip, CancellationToken token)
    {
        await using (writer)
        {
            foreach (ImageInput frame in clip.Frames)
            {
                await writer.WriteAsync(frame, token);
            }

            await writer.CompleteAsync(token);
        }
    }

    /// <summary>A small video file whose sound track is <paramref name="audio"/> in <paramref name="audioCodec"/>.</summary>
    private static async Task<string> SourceAsync(string extension, string audioCodec, AudioBuffer audio, CancellationToken token)
    {
        string path = TestClips.TempPath(extension);
        var catalog = new VideoEncoderCatalog(FfmpegVideoEncoder.Formats.Select(format => new FfmpegVideoEncoder(format)));
        VideoClip clip = TestClips.MovingGradient(32, 32, (int)Math.Ceiling(audio.Duration.TotalSeconds * 10)) with { Audio = audio };
        await catalog.SaveAsync(clip, path, new VideoEncodeOptions { AudioCodec = audioCodec }, token);
        return path;
    }

    private static AudioBuffer SilenceThenTone(double silence, double seconds, int rate)
    {
        AudioBuffer tone = TestClips.Tone(440, seconds - silence, rate);
        int lead = (int)(silence * rate) * 2;
        float[] samples = new float[lead + tone.Samples.Length];
        tone.Samples.CopyTo(samples, lead);
        return new AudioBuffer(samples, rate, 2);
    }
}
