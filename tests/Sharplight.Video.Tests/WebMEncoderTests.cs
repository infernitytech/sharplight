using Microsoft.Extensions.Options;

namespace Sharplight.Video.Tests;

public sealed class WebMEncoderTests
{
    public static TheoryData<WebMCodec, string> Codecs => new() { { WebMCodec.Vp9, "vp9" }, { WebMCodec.Vp8, "vp8" } };

    [Theory]
    [MemberData(nameof(Codecs))]
    public async Task EncodedClipDecodesWithTheSameSizeFrameCountAndHighPsnr(WebMCodec codec, string codecName)
    {
        if (!NativeAvailable() || !FfmpegReference.IsInstalled)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(64, 48, 10);
        string path = TestClips.TempPath(".webm");
        try
        {
            var encoder = new WebMVideoEncoder(Options.Create(new SharplightVideoOptions { WebMCodec = codec }));
            await new VideoEncoderCatalog([encoder]).SaveAsync(clip, path, new VideoEncodeOptions { Quality = 95 }, TestContext.Current.CancellationToken);

            (string name, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal((codecName, 64, 48, 10), (name, width, height, frames));
            List<byte[]> decoded = FfmpegReference.Decode(path, 64, 48);
            Assert.Equal(10, decoded.Count);
            for (int i = 0; i < 10; i++)
            {
                double psnr = TestClips.Psnr(decoded[i], clip.Frames[i].Rgb);
                Assert.True(psnr > 30, $"frame {i}: {psnr:F1} dB");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task OddSizesAndBitrateModeWork()
    {
        if (!NativeAvailable() || !FfmpegReference.IsInstalled)
        {
            return;
        }

        VideoClip clip = TestClips.MovingGradient(33, 17, 8, 23.976);
        using var stream = new MemoryStream();

        await new WebMVideoEncoder().EncodeAsync(clip, stream, new VideoEncodeOptions { BitrateKilobitsPerSecond = 400 }, TestContext.Current.CancellationToken);

        string path = TestClips.TempPath(".webm");
        try
        {
            await File.WriteAllBytesAsync(path, stream.ToArray(), TestContext.Current.CancellationToken);
            (string _, int width, int height, int frames) = FfmpegReference.Probe(path);
            Assert.Equal((33, 17, 8), (width, height, frames));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task CancellationStopsTheEncode()
    {
        if (!NativeAvailable())
        {
            return;
        }

        using var cancellation = new CancellationTokenSource();
        await cancellation.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new WebMVideoEncoder().EncodeAsync(TestClips.MovingGradient(), new MemoryStream(), null, cancellation.Token));
    }

    [Fact]
    public async Task UnavailableNativeLibraryIsReportedClearly()
    {
        if (NativeAvailable())
        {
            Assert.True(new WebMVideoEncoder().IsAvailable);
            return;
        }

        Assert.False(new WebMVideoEncoder().IsAvailable);
        await Assert.ThrowsAsync<PlatformNotSupportedException>(() => new WebMVideoEncoder().EncodeAsync(TestClips.MovingGradient(), new MemoryStream(), null, TestContext.Current.CancellationToken));
    }

    private static bool NativeAvailable() => Native.NativeVideoLibrary.IsAvailable;
}
