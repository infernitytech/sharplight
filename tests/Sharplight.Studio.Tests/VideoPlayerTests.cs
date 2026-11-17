using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Avalonia.Media.Imaging;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.Studio.Views;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The video player is an ordinary Avalonia control (no native child window that would draw above the UI): the view model plays, pauses,
/// seeks and steps a short clip with its sound track, 1080p playback keeps the UI thread responsive, and the Video page shows a frame.
/// </summary>
public sealed class VideoPlayerTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    private static List<ImageInput> Clip(int count, int width = 16, int height = 12) => [.. Enumerable.Range(0, count).Select(i => FakeImageGenerator.Gradient(width, height, i))];

    private static async Task WaitAsync(Func<bool> condition)
    {
        for (int i = 0; i < 400 && !condition(); i++)
        {
            Dispatcher.UIThread.RunJobs();
            await Task.Delay(10);
        }

        Assert.True(condition());
    }

    [Fact]
    public Task Player_PlaysPausesSeeksAndSteps_WithItsSound() => HeadlessApp.RunAsync(async () =>
    {
        var audio = new FakeAudio();
        var player = new VideoPlayerViewModel(audio);
        var frames = Clip(24);
        player.Load(frames, 24, new AudioBuffer(new float[48000], 48000));
        Assert.True(player.HasAudio);
        Assert.False(player.IsPlaying);

        player.Play();
        Assert.True(player.IsPlaying);
        await WaitAsync(() => player.Position >= 5);
        player.Pause();
        Assert.False(player.IsPlaying);
        double paused = player.Position;
        await Task.Delay(100);
        Assert.Equal(paused, player.Position);
        await WaitAsync(() => audio.Streamed.Count == 1);

        // Seek (the scrubber) and frame steps.
        player.Seek(10);
        Assert.Equal(frames[10], player.CurrentFrame);
        player.StepForwardCommand.Execute(null);
        Assert.Equal(frames[11], player.CurrentFrame);
        player.StepBackwardCommand.Execute(null);
        player.StepBackwardCommand.Execute(null);
        Assert.Equal(frames[9], player.CurrentFrame);

        // Seeking while playing continues from there and restarts the sound at that time.
        player.Loop = false;
        player.Play();
        player.Seek(18);
        await WaitAsync(() => audio.Streamed.Count >= 2);
        Assert.Equal(12000, audio.Streamed[^1].FrameCount);
        await WaitAsync(() => !player.IsPlaying);
        Assert.Equal(frames[^1], player.CurrentFrame);

        // Looping wraps around.
        player.Loop = true;
        player.Play();
        await WaitAsync(() => player.Position < 5 && player.Position > 0 || player.Position == 0 && player.IsPlaying);
        player.Pause();
        Assert.Contains("s", player.PositionText, StringComparison.Ordinal);
    });

    [Fact]
    public Task Video_Page_HasNoNativeControl_AndShowsTheFrame() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var generator = new FakeVideoGenerator();
        var model = new FakeModel("wan-tiny") { Kind = ModelKinds.VideoGeneration };
        model.Clients[typeof(IVideoGenerator)] = generator;
        services.Runtime.Add(model);
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1280, Height = 800 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Video);
            viewModel.OpenKind(ModelKinds.VideoGeneration, model);
            viewModel.Video.Prompt = "a red fox trotting through snow";
            viewModel.Video.RunCommand.Execute(null);
            await viewModel.Video.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Video.Player.CanPlay);
            var view = window.GetVisualDescendants().OfType<VideoView>().Single();
            await view.Idle;
            await HeadlessApp.RenderAsync(window);

            // A normal control: it sits inside the page's layout and no native control host exists anywhere in the window.
            Assert.Empty(window.GetVisualDescendants().OfType<NativeControlHost>());
            Assert.NotNull(view.GetVisualAncestors().OfType<VideoPage>().SingleOrDefault());
            Assert.True(view.Bounds.Width > 100 && view.Bounds.Height > 50);
            Assert.NotNull(view.Viewer.Source);
            // It stays inside the card: no overlap with the header above or the controls below.
            var origin = view.TranslatePoint(default, window)!.Value;
            var card = view.GetVisualAncestors().OfType<Border>().First(border => border.Classes.Contains("card"));
            var cardOrigin = card.TranslatePoint(default, window)!.Value;
            Assert.True(origin.Y >= cardOrigin.Y && origin.Y + view.Bounds.Height <= cardOrigin.Y + card.Bounds.Height + 0.5, $"view {origin} {view.Bounds}, card {cardOrigin} {card.Bounds}");
            var controls = window.GetVisualDescendants().OfType<Slider>().First(slider => slider.Maximum == viewModel.Video.Player.LastPosition && slider.Name is null);
            Assert.True(controls.TranslatePoint(default, window)!.Value.Y >= origin.Y + view.Bounds.Height - 0.5, "the controls are below the video");
            Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
            using var captured = window.CaptureRenderedFrame();
            Assert.NotNull(captured);
            if (s_screenshots is not null)
            {
                Directory.CreateDirectory(s_screenshots);
                captured.Save(Path.Combine(s_screenshots, "studio-video.png"), new PngBitmapEncoderOptions());
            }
        }
        finally
        {
            window.Close();
        }
    });

    [Fact]
    public Task Player_At1080p_KeepsTheUiResponsive() => HeadlessApp.RunAsync(async () =>
    {
        var player = new VideoPlayerViewModel();
        var frames = Clip(12, 1920, 1080);
        player.Load(frames, 30);
        var view = new VideoView();
        var window = new Avalonia.Controls.Window { Width = 800, Height = 500, Content = view };
        window.Show();
        view.Bind(VideoView.FrameProperty, new Avalonia.Data.Binding(nameof(VideoPlayerViewModel.CurrentFrame)) { Source = player });
        var heartbeat = UiHeartbeat.Start();
        player.Loop = true;
        player.Play();
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (watch.Elapsed < TimeSpan.FromSeconds(2))
        {
            await Task.Delay(20);
            Dispatcher.UIThread.RunJobs();
        }

        player.Pause();
        heartbeat.StopAndAssertResponsive("1080p playback");
        await view.Idle;
        Assert.NotNull(view.Viewer.Source);
        window.Close();
    });
}
