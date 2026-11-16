using Sharplight.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.VideoGeneration;

namespace Sharplight.Studio.Tests;

/// <summary>The Video page over a fake video generator, and the library's recognition of Wan folders.</summary>
public sealed class VideoPageTests
{
    private static FakeModel Generator(TestServices services, out FakeVideoGenerator client)
    {
        client = new FakeVideoGenerator();
        var model = new FakeModel("wan-tiny") { Kind = ModelKinds.VideoGeneration };
        model.Clients[typeof(IVideoGenerator)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static NumberParameterViewModel Number(VideoPageViewModel page, string key) => page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key);

    private static async Task<VideoPageViewModel> OpenAsync(TestServices services)
    {
        var page = new VideoPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.ModelDefaults;
        await Task.Yield();
        return page;
    }

    [Fact]
    public async Task ModelDefaults_FillTheParameters_TheNegativePrompt_AndTheStartImage()
    {
        using var services = new TestServices();
        Generator(services, out _);
        var page = await OpenAsync(services);

        Assert.Equal(64, Number(page, "width").IntValue);
        Assert.Equal(9, Number(page, "frames").IntValue);
        Assert.Equal(12, Number(page, "fps").IntValue);
        Assert.Equal("blurry, static", page.NegativePrompt);
        Assert.True(page.SupportsImage);
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Contains("Type a prompt", page.RunHint, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Generate_SnapsSizeAndFrames_ShowsDecodedFrames_AndExports()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        var page = await OpenAsync(services);
        page.Prompt = "a red fox";
        Number(page, "width").Value = 50;
        Number(page, "height").Value = 30;
        Number(page, "frames").Value = 10;
        Number(page, "steps").Value = 3;
        var shownCounts = new List<int>();
        page.Player.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(VideoPlayerViewModel.CurrentFrame))
            {
                shownCounts.Add(page.Player.Frames.Count);
            }
        };

        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await Task.Yield();

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        var (request, options) = Assert.Single(client.Requests);
        Assert.Equal("a red fox", request.Prompt);
        Assert.Equal("blurry, static", request.NegativePrompt);
        Assert.Equal((48, 32), (options.Width, options.Height));
        Assert.Equal(9, options.Frames);
        Assert.True(options.PreviewInterval > 0);
        Assert.Equal(9, page.Player.Frames.Count);
        Assert.Contains(4, shownCounts);
        Assert.True(page.HasResult);
        Assert.True(page.Player.CanPlay);
        Assert.Contains("frames: 9 at 12 fps", page.ResultSummary, StringComparison.Ordinal);

        string folder = Path.Combine(services.Directory, "export");
        Directory.CreateDirectory(folder);
        foreach (string name in new[] { "clip.gif", "clip.apng" })
        {
            await page.ExportToAsync(Path.Combine(folder, name));
            Assert.True(new FileInfo(Path.Combine(folder, name)).Length > 0, name);
        }

        Assert.Equal("", page.Notice);
        Assert.Contains(page.ExportFormats, format => format is { Format: "gif", IsAvailable: true });
        var output = Assert.Single(page.Gallery.Items);
        Assert.Contains("first-frame.png", output.Files.Select(file => file.Name));
    }

    [Fact]
    public async Task Player_StepsLoopsAndStops()
    {
        var player = new VideoPlayerViewModel();
        var frames = Enumerable.Range(0, 3).Select(i => FakeImageGenerator.Gradient(4, 4, i)).ToList();
        player.Load(frames, 8);
        Assert.Equal(frames[0], player.CurrentFrame);
        player.Advance();
        player.Advance();
        Assert.Equal(frames[2], player.CurrentFrame);
        player.Advance();
        Assert.Equal(frames[0], player.CurrentFrame);
        player.Loop = false;
        player.Position = 2;
        player.Advance();
        Assert.Equal(frames[2], player.CurrentFrame);
        Assert.False(player.IsPlaying);
        await Task.CompletedTask;
    }

    [Fact]
    public async Task Cancel_StopsTheRun_AndKeepsNoResult()
    {
        using var services = new TestServices();
        Generator(services, out var client);
        client.Gate = new TaskCompletionSource();
        var page = await OpenAsync(services);
        page.Prompt = "a red fox";
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await Task.Delay(200, TestContext.Current.CancellationToken);
        page.CancelCommand.Execute(null);
        await item.Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Cancelled, item.Status);
        Assert.False(page.HasResult);
    }

    [Fact]
    public void Library_RecognizesWanDiffusersFolders_AsVideoModels()
    {
        using var dir = new TempDir();
        string wan = dir.File("wan");
        Directory.CreateDirectory(Path.Combine(wan, "transformer"));
        Directory.CreateDirectory(Path.Combine(wan, "vae"));
        File.WriteAllText(Path.Combine(wan, "transformer", "config.json"), """{ "_class_name": "WanTransformer3DModel" }""");
        File.WriteAllText(Path.Combine(wan, "vae", "config.json"), """{ "_class_name": "AutoencoderKLWan" }""");
        Assert.Equal(ModelKinds.VideoGeneration, ImageModelProbe.KindOf(wan));

        string sd = dir.File("sd");
        Directory.CreateDirectory(Path.Combine(sd, "transformer"));
        Directory.CreateDirectory(Path.Combine(sd, "vae"));
        File.WriteAllText(Path.Combine(sd, "transformer", "config.json"), """{ "_class_name": "FluxTransformer2DModel" }""");
        File.WriteAllText(Path.Combine(sd, "vae", "config.json"), """{ "_class_name": "AutoencoderKL" }""");
        Assert.Equal(ModelKinds.ImageGeneration, ImageModelProbe.KindOf(sd));
    }

    [Fact]
    public void VideoKind_IsRunnable()
    {
        Assert.True(Studio.Models.ModelKindCatalog.For(ModelKinds.VideoGeneration).Runnable);
        Assert.True(Studio.Services.Runtime.TaskKindClients.Handles(ModelKinds.VideoGeneration));
    }

    [Fact]
    public async Task Generate_StoresTheVideoInTheGallery_WithFirstFrameAndSidecar_AndSurvivesARestart()
    {
        using var services = new TestServices();
        Generator(services, out _);
        var page = await OpenAsync(services);
        await page.Generations.Loading;
        Assert.True(page.Generations.IsEmpty);
        page.Prompt = "a red fox";
        page.NegativePrompt = "blurry";
        Number(page, "frames").Value = 5;
        Number(page, "steps").Value = 3;
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await Task.Yield();

        var entry = Assert.Single(page.Generations.Visible).Entry;
        Assert.Equal(("wan-tiny", "a red fox", "blurry"), (entry.Model, entry.Prompt, entry.NegativePrompt));
        Assert.Equal(5, entry.Frames);
        var store = new Sharplight.Studio.Services.Gallery.ImageGalleryStore(services.Services.Pages.GenerationsDirectory("videos"));
        Assert.Contains(Path.Combine("generations", "videos"), store.PathOf(entry), StringComparison.Ordinal);
        Assert.True(File.Exists(store.PathOf(entry)));
        Assert.True(File.Exists(Path.ChangeExtension(store.PathOf(entry), ".json")));
        Assert.True(entry.Video.Length > 0, "no encoder produced a video file");
        Assert.True(File.Exists(store.PathOfVideo(entry)));

        // A new page lists it from disk, shows its first frame and reuses its settings.
        var reopened = await OpenAsync(services);
        await reopened.Generations.Loading;
        Assert.Equal(entry.Id, Assert.Single(reopened.Generations.Visible).Entry.Id);
        reopened.Prompt = "";
        reopened.Generations.ReuseSettingsCommand.Execute(reopened.Generations.Visible[0]);
        Assert.Equal("a red fox", reopened.Prompt);
        Assert.Equal(5, Number(reopened, "frames").IntValue);
        reopened.Generations.SelectedItem = reopened.Generations.Visible[0];
        for (int i = 0; i < 100 && !reopened.Player.HasFrames; i++)
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        Assert.True(reopened.Player.HasFrames);

        // Deleting removes the first frame, the sidecar and the video.
        reopened.Generations.DeleteCommand.Execute(null);
        reopened.Generations.DeleteCommand.Execute(null);
        await reopened.Generations.Deleting;
        Assert.False(File.Exists(store.PathOfVideo(entry)));
        Assert.Empty(store.Load());
    }
}
