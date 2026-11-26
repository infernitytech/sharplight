using Avalonia.Controls;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.Studio.Views;
using Sharplight.Video;
using Sharplight.VideoEnhancement;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Enhance page's live before/after comparison over a fake video decoder and enhancer: the newest finished frame after each batch,
/// paired with the input frame of the same index; the blend slider and split mode on the result layer; scrubbing; the memory cap.
/// </summary>
public sealed class EnhancementComparisonTests
{
    private const int InputWidth = 32;
    private const int InputHeight = 16;
    private const int ClipFrames = 7;

    /// <summary>The input frames: frame i has the blue value (byte)(i · 30 · 37) everywhere.</summary>
    private static readonly IReadOnlyList<ImageInput> s_clip = [.. Enumerable.Range(0, ClipFrames).Select(i => FakeImageGenerator.Gradient(InputWidth, InputHeight, i * 30))];

    /// <summary>The blue value of input frame <paramref name="index"/>.</summary>
    private static byte OriginalBlue(long index) => (byte)(index * 30 * 37);

    /// <summary>The blue value of the fake enhancer's frame <paramref name="index"/>.</summary>
    private static byte EnhancedBlue(long index) => (byte)(index * 10 * 37);

    private static byte Blue(ImageInput image) => image.Rgb[2];

    private static FakeVideoEnhancer AddEnhancer(TestServices services)
    {
        var client = new FakeVideoEnhancer();
        var model = new FakeModel("enhancer-tiny") { Kind = ModelKinds.VideoEnhancement };
        model.Clients[typeof(IVideoEnhancer)] = client;
        services.Runtime.Add(model);
        return client;
    }

    private static async Task<VideoEnhancementPageViewModel> OpenAsync(TestServices services)
    {
        var decoders = new VideoDecoderCatalog([new FakeVideoDecoder(s_clip, 8)]);
        var page = new VideoEnhancementPageViewModel(services.Services, new FakeNavigator(), decoders: decoders);
        await page.ActivateAsync();
        await page.ModelDefaults;
        await Task.Yield();
        return page;
    }

    /// <summary>A file for the fake decoder to "read" (the page checks that the input exists).</summary>
    private static string ClipFile(TestServices services)
    {
        string path = Path.Combine(services.Directory, "clip.mp4");
        File.WriteAllBytes(path, [0]);
        return path;
    }

    private static void SetNumber(VideoEnhancementPageViewModel page, string key, double value)
        => page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key).Value = value;

    /// <summary>Holds the fake enhancer after each batch until the test releases it.</summary>
    private sealed class BatchGate
    {
        private readonly TaskCompletionSource[] _reached;
        private readonly TaskCompletionSource[] _release;

        public BatchGate(int batches)
        {
            _reached = [.. Enumerable.Range(0, batches).Select(_ => new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously))];
            _release = [.. Enumerable.Range(0, batches).Select(_ => new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously))];
        }

        public Task After(int batch)
        {
            _reached[batch].TrySetResult();
            return _release[batch].Task;
        }

        public Task Reached(int batch) => _reached[batch].Task.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);

        public void Release(int batch) => _release[batch].TrySetResult();
    }

    [Fact]
    public async Task EachBatch_ShowsItsNewestFrame_PairedWithTheInputFrameOfTheSameIndex()
    {
        using var services = new TestServices();
        var client = AddEnhancer(services);
        var gate = new BatchGate(2);
        client.AfterBatch = gate.After;
        var page = await OpenAsync(services);
        Assert.True(await page.SetInputAsync(ClipFile(services)));
        SetNumber(page, "scale", 2);
        SetNumber(page, "batch", 4);
        page.RunCommand.Execute(null);

        int[] newest = [3, 6];
        for (int batch = 0; batch < newest.Length; batch++)
        {
            await gate.Reached(batch);
            await page.Compare.Idle;
            var shown = Assert.IsType<ComparisonFrame>(page.Compare.Current);
            Assert.Equal(newest[batch], shown.Index);
            Assert.Equal(newest[batch] + 1, page.Compare.Count);
            Assert.Equal(OriginalBlue(shown.Index), Blue(shown.Original));
            Assert.Equal(EnhancedBlue(shown.Index), Blue(shown.Enhanced));

            // The original is enlarged to the output size, so the layers overlay exactly.
            Assert.Equal((64, 32), (shown.Original.Width, shown.Original.Height));
            Assert.Equal((64, 32), (shown.Enhanced.Width, shown.Enhanced.Height));
            Assert.Contains($"{newest[batch] + 1} of {ClipFrames} finished", page.Compare.PositionText, StringComparison.Ordinal);
            gate.Release(batch);
        }

        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        Assert.Equal(ClipFrames, page.Compare.Count);
        Assert.False(page.ShowPlayer);
        Assert.True(page.Compare.Player.CanPlay);

        // Scrubbing selects the input and the result of the same frame.
        foreach (int index in new[] { 0, 3, 6, 1 })
        {
            page.Compare.Player.Seek(index);
            var shown = page.Compare.Current!;
            Assert.Equal(index, shown.Index);
            Assert.Equal(OriginalBlue(index), Blue(shown.Original));
            Assert.Equal(EnhancedBlue(index), Blue(shown.Enhanced));
        }

        // Playing loops over the frames; stepping past the end wraps.
        page.Compare.Player.Seek(6);
        page.Compare.Player.Advance();
        Assert.Equal(0, page.Compare.Current!.Index);

        // A new input forgets the comparison.
        Assert.True(await page.SetInputAsync(ClipFile(services)));
        Assert.False(page.Compare.HasFrames);
        Assert.True(page.ShowPlayer);
    }

    [Fact]
    public async Task AnImage_IsComparedAsASingleFrame()
    {
        using var services = new TestServices();
        AddEnhancer(services);
        var page = await OpenAsync(services);
        string still = Path.Combine(services.Directory, "still.png");
        File.WriteAllBytes(still, ImageFiles.ToPng(FakeImageGenerator.Gradient(20, 10, 4)));
        Assert.True(await page.SetInputAsync(still));
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await page.Compare.Idle;

        var shown = Assert.Single(page.Compare.Frames);
        Assert.Equal(0, shown.Index);
        Assert.Equal((60, 30), (shown.Original.Width, shown.Original.Height));
        Assert.Equal((60, 30), (shown.Enhanced.Width, shown.Enhanced.Height));
        Assert.Equal((byte)(4 * 37), Blue(shown.Original));
        Assert.False(page.Compare.Player.CanPlay);
    }

    [Fact]
    public void Blend_IsTheResultOpacity_SplitMakesItADivider()
    {
        var compare = new EnhancementComparisonViewModel(new FakePlatform());
        Assert.Equal(1, compare.EnhancedOpacity);
        compare.Blend = 0.25;
        Assert.Equal(0.25, compare.EnhancedOpacity);
        Assert.Equal("Enhanced 25%", compare.BlendText);
        compare.IsSplit = true;
        Assert.Equal(1, compare.EnhancedOpacity);
        Assert.Equal("Divider 25%", compare.BlendText);
    }

    [Fact]
    public void PreviewSize_FitsTheExpectedFramesIntoTheBudget_DownToAFloor()
    {
        const long budget = 1L << 30;
        Assert.Equal((1280, 720), EnhancementComparisonViewModel.PreviewSizeFor(1280, 720, 100, budget));

        var (width, height) = EnhancementComparisonViewModel.PreviewSizeFor(3840, 2160, 240, budget);
        Assert.True(width < 3840 && height < 2160);
        Assert.True(240L * width * height * 6 <= budget * 1.01);
        Assert.Equal(16.0 / 9, (double)width / height, 2);

        Assert.Equal(EnhancementComparisonViewModel.MinPreviewSide, EnhancementComparisonViewModel.PreviewSizeFor(3840, 2160, 1_000_000, budget).Height);
    }

    [Fact]
    public async Task MemoryCap_ReducesBothLayersAlike_AndLetsTheOldestFramesGo()
    {
        static ImageInput[] Frames(int first, int count, int width, int height)
            => [.. Enumerable.Range(first, count).Select(i => FakeImageGenerator.Gradient(width, height, i))];

        // Ten 720 × 360 pairs expected, room for ten 360 × 180 pairs: both layers are kept at half size.
        long pairBytes = 360 * 180 * 6;
        var compare = new EnhancementComparisonViewModel(new FakePlatform()) { MemoryBudgetBytes = 10 * pairBytes };
        ImageInput[] originals = Frames(0, 15, 72, 36);
        compare.Begin(expectedFrames: 10, framesPerSecond: 8);
        compare.Add(0, originals[0..4], Frames(0, 4, 720, 360));
        await compare.Idle;
        Assert.All(compare.Frames, frame => Assert.Equal((360, 180, 360, 180), (frame.Original.Width, frame.Original.Height, frame.Enhanced.Width, frame.Enhanced.Height)));
        Assert.Equal(4 * pairBytes, compare.StoredBytes);
        Assert.Contains("Compared at 360 × 180 (the result is 720 × 360)", compare.PreviewNote, StringComparison.Ordinal);

        // More frames than expected: the oldest are let go so the kept frames stay within the cap; the newest is shown.
        compare.Add(4, originals[4..10], Frames(4, 6, 720, 360));
        compare.Add(10, originals[10..15], Frames(10, 5, 720, 360));
        await compare.Idle;
        Assert.Equal(10, compare.Count);
        Assert.True(compare.StoredBytes <= compare.MemoryBudgetBytes);
        Assert.Equal(5, compare.Frames[0].Index);
        Assert.Equal(14, compare.Current!.Index);
        Assert.Equal(15, compare.FinishedFrames);
        Assert.Contains("first 5 frames were let go", compare.PreviewNote, StringComparison.Ordinal);

        // Scrubbing addresses the kept frames; indices stay those of the clip.
        compare.Player.Seek(0);
        Assert.Equal(5, compare.Current!.Index);
        Assert.Equal((byte)(5 * 37), Blue(compare.Current.Enhanced));

        // A new run starts over.
        compare.Begin(expectedFrames: 2, framesPerSecond: 8);
        Assert.False(compare.HasFrames);
        Assert.Equal(0, compare.StoredBytes);
    }

    [Fact]
    public Task View_ShowsTheNewestPairAfterEachBatch_AndTheSliderDrivesTheResultLayer() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var client = AddEnhancer(services);
        var gate = new BatchGate(2);
        client.AfterBatch = gate.After;
        var page = await OpenAsync(services);
        var window = new Window { Width = 1280, Height = 1000, Content = new VideoEnhancementPage { DataContext = page } };
        window.Show();
        try
        {
            // The model's defaults are applied on the UI thread; let them land before the test's own settings.
            await HeadlessApp.RenderAsync(window);
            Assert.True(await page.SetInputAsync(ClipFile(services)));
            SetNumber(page, "scale", 2);
            SetNumber(page, "batch", 4);
            page.RunCommand.Execute(null);
            await HeadlessApp.RenderAsync(window);
            var view = window.GetVisualDescendants().OfType<ComparisonView>().Single();

            int[] newest = [3, 6];
            for (int batch = 0; batch < newest.Length; batch++)
            {
                await gate.Reached(batch);
                int expected = newest[batch];
                await HeadlessApp.WaitForAsync(window, () => view.ShownIndex == expected);
                Assert.Equal(expected, view.ShownIndex);
                Assert.True(view.IsEffectivelyVisible);
                Assert.Equal(new Avalonia.PixelSize(64, 32), ((Avalonia.Media.Imaging.Bitmap)view.OriginalLayer.Source!).PixelSize);
                Assert.Equal(new Avalonia.PixelSize(64, 32), ((Avalonia.Media.Imaging.Bitmap)view.EnhancedLayer.Source!).PixelSize);
                gate.Release(batch);
            }

            await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
            await HeadlessApp.RenderAsync(window);

            // The blend slider is the result layer's opacity.
            var slider = window.GetVisualDescendants().OfType<Slider>().Single(control => control.Name == "BlendSlider");
            slider.Value = 0.4;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(0.4, page.Compare.Blend, 6);
            Assert.Equal(0.4, view.EnhancedLayer.Opacity, 6);
            Assert.Null(view.EnhancedLayer.Clip);

            // Split mode: the result is opaque and clipped right of the divider.
            page.Compare.IsSplit = true;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(1, view.EnhancedLayer.Opacity);
            var clip = Assert.IsType<Avalonia.Media.RectangleGeometry>(view.EnhancedLayer.Clip);
            Assert.True(clip.Rect.X > 0 && clip.Rect.X < view.Bounds.Width);

            // Scrubbing moves the shown pair.
            var scrubber = window.GetVisualDescendants().OfType<Slider>().Single(control => control.Name == "CompareScrubber");
            scrubber.Value = 1;
            await HeadlessApp.WaitForAsync(window, () => view.ShownIndex == 1);
            Assert.Equal(1, view.ShownIndex);
            Assert.Equal(1, page.Compare.Current!.Index);
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
        }
    });
}
