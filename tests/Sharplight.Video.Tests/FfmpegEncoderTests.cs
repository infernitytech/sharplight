using Microsoft.Extensions.Options;

namespace Sharplight.Video.Tests;

public sealed class FfmpegEncoderTests
{
    [Fact]
    public async Task EncodesMp4WithH264()
    {
        if (!FfmpegReference.HasH264)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(64, 48, 10);
        string path = TestClips.TempPath(".mp4");
        try
        {
            var encoder = new FfmpegVideoEncoder();
            Assert.True(encoder.IsAvailable);
            await new VideoEncoderCatalog([encoder]).SaveAsync(clip, path, new VideoEncodeOptions { Quality = 95 }, TestContext.Current.CancellationToken);

            (string codec, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal(("h264", 64, 48, 10), (codec, width, height, frames));
            List<byte[]> decoded = FfmpegReference.Decode(path, 64, 48);
            Assert.True(TestClips.Psnr(decoded[5], clip.Frames[5].Rgb) > 30);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task OddSizesArePaddedToEven()
    {
        if (!FfmpegReference.HasH264)
        {
            return;
        }

        string path = TestClips.TempPath(".mp4");
        try
        {
            using var stream = new MemoryStream();
            await new FfmpegVideoEncoder().EncodeAsync(TestClips.MovingGradient(33, 17, 4), stream, null, TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(path, stream.ToArray(), TestContext.Current.CancellationToken);

            (string _, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal((34, 18, 4), (width, height, frames));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task MissingFfmpegMeansUnavailable()
    {
        var encoder = new FfmpegVideoEncoder(Options.Create(new SharplightVideoOptions { FfmpegPath = "/nonexistent/ffmpeg" }));

        Assert.False(encoder.IsAvailable);
        await Assert.ThrowsAsync<PlatformNotSupportedException>(() => encoder.EncodeAsync(TestClips.MovingGradient(), new MemoryStream(), null, TestContext.Current.CancellationToken));
    }
}
