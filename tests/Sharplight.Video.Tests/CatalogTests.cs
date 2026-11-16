using Microsoft.Extensions.DependencyInjection;

namespace Sharplight.Video.Tests;

public sealed class CatalogTests
{
    [Fact]
    public void CoreRegistrationProvidesApngAndGif()
    {
        var services = new ServiceCollection();
        services.AddVideoEncoders();
        services.AddVideoEncoders();
        using ServiceProvider provider = services.BuildServiceProvider();

        var catalog = provider.GetRequiredService<VideoEncoderCatalog>();

        Assert.Equal(2, provider.GetServices<IVideoEncoder>().Count());
        Assert.Equal([VideoFormats.Apng, VideoFormats.Gif], catalog.Available.Select(e => e.Format));
        Assert.IsType<GifVideoEncoder>(catalog.FindForPath("clip.GIF"));
        Assert.IsType<ApngVideoEncoder>(catalog.FindForPath("/x/clip.png"));
        Assert.Null(catalog.FindForPath("clip.webm"));
        Assert.False(catalog.TryFind(VideoFormats.WebM, out _));
    }

    [Fact]
    public void SharplightVideoAddsTheBundledEncodersAheadOfTheManagedOnes()
    {
        var services = new ServiceCollection();
        services.AddVideoEncoders();
        services.AddSharplightVideo(o => o.EnableHardwareEncoders = false);
        services.AddSharplightVideo();
        using ServiceProvider provider = services.BuildServiceProvider();

        var catalog = provider.GetRequiredService<VideoEncoderCatalog>();
        var decoders = provider.GetRequiredService<VideoDecoderCatalog>();

        Assert.Equal(8, provider.GetServices<IVideoEncoder>().Count());
        Assert.Single(provider.GetServices<IVideoDecoder>());
        Assert.Equal(BundledFfmpeg.IsAvailable, catalog.TryFind(VideoFormats.Mp4, out _));
        Assert.Equal(BundledFfmpeg.IsAvailable, decoders.IsAvailable);
        Assert.Equal(BundledFfmpeg.IsAvailable, catalog.FindForPath("clip.gif") is FfmpegVideoEncoder);
        Assert.Equal(BundledFfmpeg.IsAvailable, catalog.FindForPath("clip.apng") is FfmpegVideoEncoder);
    }

    [Fact]
    public void TheHighestPriorityAvailableEncoderWinsAndUnavailableOnesFallBack()
    {
        var managed = new GifVideoEncoder();
        var preferred = new FakeEncoder(VideoFormats.Gif, priority: 10, available: true);
        var missing = new FakeEncoder(VideoFormats.Gif, priority: 20, available: false);

        Assert.Same(preferred, new VideoEncoderCatalog([managed, missing, preferred]).FindForPath("a.gif"));
        Assert.Same(managed, new VideoEncoderCatalog([managed, missing]).FindForPath("a.gif"));
        Assert.Equal([preferred], new VideoEncoderCatalog([managed, preferred]).Available);
    }

    [Fact]
    public async Task SaveAsyncChoosesByExtensionAndRejectsUnknownOnes()
    {
        var services = new ServiceCollection();
        services.AddVideoEncoders();
        using ServiceProvider provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<VideoEncoderCatalog>();
        string path = TestClips.TempPath(".gif");
        try
        {
            await catalog.SaveAsync(TestClips.MovingSquare(), path, null, TestContext.Current.CancellationToken);

            Assert.Equal("GIF89a", System.Text.Encoding.ASCII.GetString(File.ReadAllBytes(path), 0, 6));
            await Assert.ThrowsAsync<NotSupportedException>(() => catalog.SaveAsync(TestClips.MovingSquare(), TestClips.TempPath(".mkv"), null, TestContext.Current.CancellationToken));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task SaveAsyncDeletesThePartialFileWhenCancelled()
    {
        var catalog = new VideoEncoderCatalog([new GifVideoEncoder()]);
        string path = TestClips.TempPath(".gif");
        using var cancellation = new CancellationTokenSource();
        await cancellation.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => catalog.SaveAsync(TestClips.MovingSquare(), path, null, cancellation.Token));

        Assert.False(File.Exists(path));
    }
}
