namespace Sharplight.Video.Tests;

public sealed class GifEncoderTests
{
    [Fact]
    public async Task FewColorFramesRoundTripExactly()
    {
        VideoClip clip = TestClips.MovingSquare();
        using var stream = new MemoryStream();

        await new GifVideoEncoder().EncodeAsync(clip, stream, null, TestContext.Current.CancellationToken);

        var decoded = GifReader.Read(stream.ToArray());
        Assert.Equal((40, 30, 0), (decoded.Width, decoded.Height, decoded.LoopCount));
        Assert.Equal(clip.Frames.Count, decoded.Frames.Count);
        for (int i = 0; i < clip.Frames.Count; i++)
        {
            Assert.True(decoded.Frames[i].Rgb.AsSpan().SequenceEqual(clip.Frames[i].Rgb), $"frame {i}");
        }
    }

    [Fact]
    public async Task GradientFramesAreQuantizedClosely()
    {
        VideoClip clip = TestClips.MovingGradient(64, 48, 4);
        using var plain = new MemoryStream();
        using var dithered = new MemoryStream();

        await new GifVideoEncoder().EncodeAsync(clip, plain, new VideoEncodeOptions { Quality = 100 }, TestContext.Current.CancellationToken);
        await new GifVideoEncoder().EncodeAsync(clip, dithered, null, TestContext.Current.CancellationToken);

        foreach (MemoryStream stream in new[] { plain, dithered })
        {
            var decoded = GifReader.Read(stream.ToArray());
            Assert.Equal(4, decoded.Frames.Count);
            for (int i = 0; i < 4; i++)
            {
                Assert.True(TestClips.Psnr(decoded.Frames[i].Rgb, clip.Frames[i].Rgb) > 28, $"frame {i}");
            }
        }
    }

    [Fact]
    public async Task DelaysKeepTheAverageFrameRateAndLoopCanBeDisabled()
    {
        VideoClip clip = TestClips.MovingSquare(frames: 9, fps: 24);
        using var stream = new MemoryStream();

        await new GifVideoEncoder().EncodeAsync(clip, stream, new VideoEncodeOptions { Loop = false }, TestContext.Current.CancellationToken);

        var decoded = GifReader.Read(stream.ToArray());
        Assert.Equal(-1, decoded.LoopCount);
        Assert.Equal(38, decoded.Frames.Sum(f => f.DelayCentiseconds));
        Assert.All(decoded.Frames, f => Assert.InRange(f.DelayCentiseconds, 4, 5));
    }

    [Fact]
    public async Task LargeNoisyFramesExerciseTheFullDictionary()
    {
        var random = new Random(7);
        var rgb = new byte[200 * 150 * 3];
        random.NextBytes(rgb);
        var clip = new VideoClip([new Sharplight.Vision.ImageInput(200, 150, rgb)], 10);
        using var stream = new MemoryStream();

        await new GifVideoEncoder().EncodeAsync(clip, stream, new VideoEncodeOptions { Quality = 100 }, TestContext.Current.CancellationToken);

        var decoded = GifReader.Read(stream.ToArray());
        Assert.Single(decoded.Frames);
    }

    [Fact]
    public async Task IsDecodedByFfmpeg()
    {
        if (!FfmpegReference.IsInstalled)
        {
            return;
        }

        VideoClip clip = TestClips.MovingSquare();
        string path = TestClips.TempPath(".gif");
        try
        {
            await new VideoEncoderCatalog([new GifVideoEncoder()]).SaveAsync(clip, path, null, TestContext.Current.CancellationToken);

            (string codec, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal(("gif", 40, 30, 8), (codec, width, height, frames));
            List<byte[]> decoded = FfmpegReference.Decode(path);
            Assert.Equal(8, decoded.Count);
            Assert.True(decoded[7].AsSpan().SequenceEqual(clip.Frames[7].Rgb));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
