namespace Sharplight.Video.Tests;

public sealed class ApngEncoderTests
{
    [Fact]
    public async Task RoundTripsEveryFrameLosslessly()
    {
        VideoClip clip = TestClips.MovingGradient(33, 17, 6, 12);
        using var stream = new MemoryStream();

        await new ApngVideoEncoder().EncodeAsync(clip, stream, null, TestContext.Current.CancellationToken);

        var decoded = ApngReader.Read(stream.ToArray());
        Assert.Equal((33, 17, 0), (decoded.Width, decoded.Height, decoded.PlayCount));
        Assert.Equal(6, decoded.Frames.Count);
        for (int i = 0; i < 6; i++)
        {
            Assert.True(decoded.Frames[i].Rgb.AsSpan().SequenceEqual(clip.Frames[i].Rgb), $"frame {i}");
            Assert.Equal((1, 12), (decoded.Frames[i].DelayNumerator, decoded.Frames[i].DelayDenominator));
        }
    }

    [Fact]
    public async Task StoresOnlyTheChangedRegionAndHandlesIdenticalFrames()
    {
        VideoClip moving = TestClips.MovingSquare();
        var frames = moving.Frames.ToList();
        frames.Insert(3, frames[2]);
        var clip = new VideoClip(frames, 7.5);
        using var stream = new MemoryStream();

        await new ApngVideoEncoder().EncodeAsync(clip, stream, new VideoEncodeOptions { Loop = false }, TestContext.Current.CancellationToken);

        var decoded = ApngReader.Read(stream.ToArray());
        Assert.Equal(1, decoded.PlayCount);
        for (int i = 0; i < frames.Count; i++)
        {
            Assert.True(decoded.Frames[i].Rgb.AsSpan().SequenceEqual(frames[i].Rgb), $"frame {i}");
        }

        Assert.Equal((133, 1000), (decoded.Frames[0].DelayNumerator, decoded.Frames[0].DelayDenominator));
        using var full = new MemoryStream();
        await new ApngVideoEncoder().EncodeAsync(new VideoClip([frames[0], frames[0], frames[0]], 7.5), full, null, TestContext.Current.CancellationToken);
        Assert.True(stream.Length < full.Length * 9 / 3 + 1000);
    }

    [Fact]
    public async Task FirstFrameIsAPlainPng()
    {
        VideoClip clip = TestClips.MovingGradient(20, 10, 3);
        using var stream = new MemoryStream();
        await new ApngVideoEncoder().EncodeAsync(clip, stream, null, TestContext.Current.CancellationToken);

        var first = Sharplight.Imaging.PngCodec.Decode(stream.ToArray());

        Assert.True(first.Rgb.AsSpan().SequenceEqual(clip.Frames[0].Rgb));
    }

    [Fact]
    public async Task IsDecodedByFfmpeg()
    {
        if (!FfmpegReference.IsInstalled)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(32, 24, 5);
        string path = TestClips.TempPath(".apng");
        try
        {
            await new VideoEncoderCatalog([new ApngVideoEncoder()]).SaveAsync(clip, path, null, TestContext.Current.CancellationToken);

            (string codec, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal(("apng", 32, 24, 5), (codec, width, height, frames));
            List<byte[]> decoded = FfmpegReference.Decode(path, 32, 24);
            Assert.Equal(5, decoded.Count);
            Assert.True(decoded[4].AsSpan().SequenceEqual(clip.Frames[4].Rgb));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
