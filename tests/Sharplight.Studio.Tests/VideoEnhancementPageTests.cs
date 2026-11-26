using Avalonia.Controls;
using Avalonia.Logging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.Studio.Views;
using Sharplight.Video;
using Sharplight.VideoEnhancement;

namespace Sharplight.Studio.Tests;

/// <summary>The Enhance page over a fake video enhancer: input validation, output size, progress, cancel, the written result, and its routing by kind.</summary>
public sealed class VideoEnhancementPageTests
{
    private static FakeModel Enhancer(TestServices services, out FakeVideoEnhancer client)
    {
        client = new FakeVideoEnhancer();
        var model = new FakeModel("enhancer-tiny") { Kind = ModelKinds.VideoEnhancement };
        model.Clients[typeof(IVideoEnhancer)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static NumberParameterViewModel Number(VideoEnhancementPageViewModel page, string key)
        => page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key);

    private static async Task<VideoEnhancementPageViewModel> OpenAsync(TestServices services, VideoDecoderCatalog? decoders = null)
    {
        var page = new VideoEnhancementPageViewModel(services.Services, new FakeNavigator(), decoders: decoders);
        await page.ActivateAsync();
        await page.ModelDefaults;
        await Task.Yield();
        return page;
    }

    private static string WriteImage(TestServices services, string name, int width, int height)
    {
        string path = Path.Combine(services.Directory, name);
        File.WriteAllBytes(path, ImageFiles.ToPng(FakeImageGenerator.Gradient(width, height, 3)));
        return path;
    }

    [Fact]
    public async Task WithoutModelOrInput_TheRunExplainsWhatIsMissing()
    {
        using var services = new TestServices();
        var page = await OpenAsync(services);
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Equal("Load a video enhancement model first.", page.RunHint);

        Enhancer(services, out _);
        page = await OpenAsync(services);
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Equal("Choose a video or an image first.", page.RunHint);
        Assert.Equal("Choose a video or an image.", page.InputSummary);
    }

    [Fact]
    public async Task ModelDefaults_FillTheScale_TheBatch_AndTheColourCorrection()
    {
        using var services = new TestServices();
        Enhancer(services, out _);
        var page = await OpenAsync(services);

        Assert.Equal(3, Number(page, "scale").Value);
        Assert.Equal(5, Number(page, "batch").IntValue);
        Assert.Equal(nameof(VideoColorCorrection.Wavelet), page.Panel.GetChoice("color"));
        Assert.Equal(VideoColorCorrection.Wavelet, page.Panel.Options.ColorCorrection);
    }

    [Fact]
    public async Task Inputs_AreValidated()
    {
        using var services = new TestServices();
        Enhancer(services, out _);
        var page = await OpenAsync(services);

        Assert.False(await page.SetInputAsync(Path.Combine(services.Directory, "missing.png")));
        Assert.Contains("does not exist", page.Notice, StringComparison.Ordinal);

        string text = Path.Combine(services.Directory, "notes.txt");
        File.WriteAllText(text, "not a picture");
        Assert.False(await page.SetInputAsync(text));
        Assert.Equal("notes.txt is neither a video nor an image file.", page.Notice);

        string broken = Path.Combine(services.Directory, "broken.png");
        File.WriteAllText(broken, "not a png");
        Assert.False(await page.SetInputAsync(broken));
        Assert.Contains("broken.png cannot be read", page.Notice, StringComparison.Ordinal);
        Assert.False(page.HasInput);
        Assert.False(page.RunCommand.CanExecute(null));

        Assert.True(await page.SetInputAsync(WriteImage(services, "still.png", 40, 24)));
        Assert.Equal("", page.Notice);
        Assert.Equal("still.png · image · 40 × 24", page.InputSummary);
        Assert.True(page.Player.HasFrames);
        Assert.Equal("Output: 120 × 72", page.OutputSizeText);
        Assert.True(page.RunCommand.CanExecute(null));

        // A target height wins over the scale; the width follows the aspect ratio.
        Number(page, "height").Value = 48;
        Assert.Equal("Output: 80 × 48", page.OutputSizeText);
        Number(page, "height").Value = 8;
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Contains("choose a larger height or scale", page.RunHint, StringComparison.Ordinal);
        Number(page, "height").Value = 4320;
        Number(page, "height").Value = 0;
        Number(page, "scale").Value = 8;
        Assert.True(page.RunCommand.CanExecute(null));

        Assert.True(await page.SetInputAsync(WriteImage(services, "wide.png", 1920, 1080)));
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Contains("larger than 8K", page.RunHint, StringComparison.Ordinal);

        page.ClearInputCommand.Execute(null);
        Assert.False(page.HasInput);
        Assert.False(page.Player.HasFrames);
    }

    [Fact]
    public async Task Videos_WithoutTheBundledFfmpeg_AreRefusedWithAClearMessage()
    {
        using var services = new TestServices();
        Enhancer(services, out _);
        var page = await OpenAsync(services, new VideoDecoderCatalog([]));
        string video = Path.Combine(services.Directory, "clip.mp4");
        File.WriteAllBytes(video, [0, 0, 0, 24]);

        Assert.False(page.CanReadVideos);
        Assert.Contains("FFmpeg", page.VideoSupportNotice, StringComparison.Ordinal);
        Assert.False(await page.SetInputAsync(video));
        Assert.Equal(page.VideoSupportNotice, page.Notice);
        Assert.False(page.HasInput);

        // Images still work.
        Assert.True(await page.SetInputAsync(WriteImage(services, "still.png", 16, 16)));
    }

    [Fact]
    public async Task Enhance_AnImage_ReportsTheStages_WritesThePng_AndOpensIt()
    {
        using var services = new TestServices();
        Enhancer(services, out var client);
        var page = await OpenAsync(services);
        Assert.True(await page.SetInputAsync(WriteImage(services, "still.png", 40, 24)));
        Number(page, "height").Value = 48;
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "12";
        var steps = new List<string>();
        page.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(VideoEnhancementPageViewModel.StepText) && page.StepText.Length > 0)
            {
                steps.Add(page.StepText);
            }
        };

        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await Task.Yield();

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        var (request, options) = Assert.Single(client.Requests);
        Assert.Single(request.Frames);
        Assert.Equal((80, 48), (options.Width, options.Height));
        Assert.Equal(12, options.Seed);
        Assert.Equal(VideoColorCorrection.Wavelet, options.ColorCorrection);
        Assert.Contains("restoring", steps);
        Assert.True(page.HasResultFile);
        Assert.EndsWith(".png", page.ResultPath, StringComparison.Ordinal);
        var written = ImageFiles.Read(page.ResultPath);
        Assert.Equal((80, 48), (written.Width, written.Height));
        Assert.Contains("40 × 24 → 80 × 48", page.ResultSummary, StringComparison.Ordinal);
        Assert.Contains("still.png", Assert.Single(page.History.Items).Record.Input["input"], StringComparison.Ordinal);

        page.OpenResultCommand.Execute(null);
        Assert.Equal(page.ResultPath, Assert.Single(services.Platform.Opened));

        string copy = Path.Combine(services.Directory, "copy.png");
        services.Platform.SaveTo = copy;
        await page.SaveResultCommand.ExecuteAsync(null);
        Assert.True(File.Exists(copy));
    }

    [Fact]
    public async Task Enhance_AVideo_ReadsItsFrames_InBatches_AndWritesAVideo()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            Assert.Skip("the bundled FFmpeg libraries are not built");
        }

        using var services = new TestServices();
        Enhancer(services, out var client);
        var page = await OpenAsync(services);
        string source = Path.Combine(services.Directory, "clip.mp4");
        var frames = Enumerable.Range(0, 7).Select(i => FakeImageGenerator.Gradient(32, 16, i * 30)).ToList();
        await new VideoEncoderCatalog([new FfmpegVideoEncoder(VideoFormats.Mp4)]).SaveAsync(new VideoClip(frames, 8), source, null, TestContext.Current.CancellationToken);

        Assert.True(await page.SetInputAsync(source));
        Assert.StartsWith("clip.mp4 · video · 32 × 16 · 7 frames at 8 fps", page.InputSummary, StringComparison.Ordinal);
        Number(page, "scale").Value = 2;
        Number(page, "batch").Value = 4;
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await Task.Yield();

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        var (request, options) = Assert.Single(client.Requests);
        // The frames are streamed from the decoder, never collected into a list.
        Assert.Null(request.Frames);
        Assert.Equal(7, request.FrameCount);
        Assert.Equal(7, client.FramesRead);
        Assert.Equal(4, options.FramesPerBatch);
        await page.Compare.Idle;
        Assert.Equal(7, page.Compare.Count);
        Assert.EndsWith(".mp4", page.ResultPath, StringComparison.Ordinal);
        var info = await new FfmpegVideoDecoder().ProbeAsync(page.ResultPath, TestContext.Current.CancellationToken);
        Assert.Equal((64, 32), (info.Width, info.Height));
        Assert.Equal(7, await CountFramesAsync(page.ResultPath));
    }

    [Fact]
    public async Task Enhance_AVideo_ShowsFramesPlaybackTimeAndTimeLeft_AndKeepsTheSoundTrack()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            Assert.Skip("the bundled FFmpeg libraries are not built");
        }

        using var services = new TestServices();
        Enhancer(services, out _);
        var page = await OpenAsync(services);
        string source = Path.Combine(services.Directory, "voiced.mp4");
        var frames = Enumerable.Range(0, 24).Select(i => FakeImageGenerator.Gradient(32, 16, i * 9)).ToList();
        var tone = new Sharplight.Audio.AudioBuffer([.. Enumerable.Range(0, 48000 * 3).Select(i => 0.3f * MathF.Sin(i * 0.05f))], 48000, 1);
        await new VideoEncoderCatalog([new FfmpegVideoEncoder(VideoFormats.Mp4)]).SaveAsync(new VideoClip(frames, 8) { Audio = tone }, source, null,
            TestContext.Current.CancellationToken);
        Assert.True(await page.SetInputAsync(source));
        Number(page, "scale").Value = 2;
        Number(page, "batch").Value = 4;
        var steps = new List<string>();
        page.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(VideoEnhancementPageViewModel.StepText) && page.StepText.Length > 0)
            {
                steps.Add(page.StepText);
            }
        };

        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await Task.Yield();

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        Assert.Contains(steps, step => step.Contains("of 24 frames", StringComparison.Ordinal) && step.Contains("of 0:03", StringComparison.Ordinal));
        Assert.Contains(steps, step => step.Contains("left", StringComparison.Ordinal));
        var info = await new FfmpegVideoDecoder().ProbeAsync(page.ResultPath, TestContext.Current.CancellationToken);
        Assert.NotNull(info.AudioCodec);
        Assert.Equal(24, await CountFramesAsync(page.ResultPath));
    }

    [Fact]
    public async Task Cancel_DuringAVideo_KeepsTheFramesFinishedSoFar_AsAPlayableVideo()
    {
        if (!BundledFfmpeg.IsAvailable)
        {
            Assert.Skip("the bundled FFmpeg libraries are not built");
        }

        using var services = new TestServices();
        Enhancer(services, out var client);
        var secondBatch = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        client.AfterBatch = batch =>
        {
            if (batch == 1)
            {
                secondBatch.TrySetResult();
                return Task.Delay(Timeout.Infinite);
            }

            return Task.CompletedTask;
        };
        var page = await OpenAsync(services);
        string source = Path.Combine(services.Directory, "long.mp4");
        var frames = Enumerable.Range(0, 20).Select(i => FakeImageGenerator.Gradient(32, 16, i * 11)).ToList();
        await new VideoEncoderCatalog([new FfmpegVideoEncoder(VideoFormats.Mp4)]).SaveAsync(new VideoClip(frames, 8), source, null, TestContext.Current.CancellationToken);
        Assert.True(await page.SetInputAsync(source));
        Number(page, "scale").Value = 2;
        Number(page, "batch").Value = 4;
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await secondBatch.Task.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);
        page.CancelCommand.Execute(null);
        await item.Completion.WaitAsync(TestContext.Current.CancellationToken);
        await Task.Yield();

        Assert.Equal(RunStatus.Cancelled, item.Status);
        Assert.True(page.HasResultFile);
        Assert.Contains("8 frames finished so far were saved", page.Notice, StringComparison.Ordinal);
        Assert.Equal(8, await CountFramesAsync(page.ResultPath));
        Assert.Equal(RunOutcome.Cancelled, Assert.Single(page.History.Items).Record.Outcome);
    }

    [Fact]
    public async Task Enhance_ALongVideo_HoldsABoundedComparison_AndReadsEveryFrameOnce()
    {
        using var services = new TestServices();
        Enhancer(services, out var client);
        var clip = Enumerable.Range(0, 400).Select(i => FakeImageGenerator.Gradient(32, 16, i)).ToList();
        var page = await OpenAsync(services, new VideoDecoderCatalog([new FakeVideoDecoder(clip, 25)]));
        page.Compare.MemoryBudgetBytes = 64L * 32 * 6 * 50;
        string path = Path.Combine(services.Directory, "long.mp4");
        File.WriteAllBytes(path, [0]);
        Assert.True(await page.SetInputAsync(path));
        Number(page, "scale").Value = 2;
        Number(page, "batch").Value = 8;
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await page.Compare.Idle;

        Assert.Equal(400, client.FramesRead);
        Assert.Equal(400, page.Compare.FinishedFrames);
        Assert.True(page.Compare.StoredBytes <= page.Compare.MemoryBudgetBytes, $"{page.Compare.StoredBytes} bytes kept");
        Assert.True(page.Compare.Count <= 50, $"{page.Compare.Count} frames kept");
        Assert.Equal(399, page.Compare.Current!.Index);
    }

    private static async Task<int> CountFramesAsync(string path)
    {
        int count = 0;
        await foreach (var frame in new FfmpegVideoDecoder().ReadFramesAsync(path, null, TestContext.Current.CancellationToken))
        {
            count++;
        }

        return count;
    }

    [Fact]
    public async Task Cancel_StopsTheRun_AndKeepsNoResult()
    {
        using var services = new TestServices();
        Enhancer(services, out var client);
        client.Gate = new TaskCompletionSource();
        var page = await OpenAsync(services);
        Assert.True(await page.SetInputAsync(WriteImage(services, "still.png", 16, 16)));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await Task.Delay(100, TestContext.Current.CancellationToken);
        Assert.True(page.IsEnhancing);
        page.CancelCommand.Execute(null);
        await item.Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Cancelled, item.Status);
        Assert.False(page.HasResultFile);
        Assert.False(page.IsEnhancing);
    }

    [Fact]
    public void Kind_IsRunnable_AndRoutesToTheEnhancePage()
    {
        Assert.True(ModelKindCatalog.For(ModelKinds.VideoEnhancement).Runnable);
        Assert.True(Sharplight.Hosting.Loaders.TaskKindClients.Handles(ModelKinds.VideoEnhancement));

        using var services = new TestServices();
        var window = new MainWindowViewModel(services.Services);
        Assert.Contains(window.Enhance, window.Pages);
        Assert.True(window.OpenKind(ModelKinds.VideoEnhancement, null));
    }

    [Fact]
    public void Library_RecognizesVideoEnhancementCheckpointFolders()
    {
        using var dir = new TempDir();
        string folder = dir.File("seedvr2-3b");
        Directory.CreateDirectory(folder);
        foreach (string name in new[] { "seedvr2_ema_3b.pth", "ema_vae.pth", "pos_emb.pt", "neg_emb.pt" })
        {
            File.WriteAllBytes(Path.Combine(folder, name), [0]);
        }

        Assert.Equal(ModelKinds.VideoEnhancement, ModelInspector.Inspect(folder).Kind);
        Assert.Equal(ModelKinds.VideoEnhancement, ModelInspector.Inspect(Path.Combine(folder, "seedvr2_ema_3b.pth")).Kind);

        // Without its VAE and embeddings the folder is not such a model.
        File.Delete(Path.Combine(folder, "ema_vae.pth"));
        Assert.NotEqual(ModelKinds.VideoEnhancement, ModelInspector.Inspect(folder).Kind);
    }

    [Fact]
    public Task Page_RendersWithoutBindingErrors_AndShowsTheHint() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        Enhancer(services, out _);
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Enhance);
            await viewModel.Enhance.ModelDefaults;
            await HeadlessApp.RenderAsync(window);
            var page = window.GetVisualDescendants().OfType<VideoEnhancementPage>().Single(view => view.IsEffectivelyVisible);
            var hint = page.FindControl<TextBlock>("RunHintText")!;
            Assert.Equal("Choose a video or an image first.", hint.Text);
            var enhance = page.FindControl<Button>("EnhanceButton")!;
            Assert.False(enhance.IsEffectivelyEnabled);

            Assert.True(await viewModel.Enhance.SetInputAsync(WriteImage(services, "still.png", 40, 24)));
            await HeadlessApp.RenderAsync(window);
            Assert.Equal("", hint.Text);
            Assert.True(enhance.IsEffectivelyEnabled);
            Assert.Equal("Output: 120 × 72", page.FindControl<TextBlock>("OutputSizeLabel")!.Text);
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });
}
