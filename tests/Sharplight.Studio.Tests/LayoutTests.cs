using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Presenters;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Audio;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.Views;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The result and preview areas of the pages stay inside their cards at every window size: nothing is arranged outside its parent and
/// painted over the headers, buttons and controls around it (a viewer with a minimum height in a page that does not scroll did).
/// </summary>
public sealed class LayoutTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    private static readonly (double Width, double Height)[] s_sizes = [(1440, 900), (1100, 700), (900, 600)];

    [Fact]
    public Task VisionModes_KeepTheResultInsideItsCard() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var vision = new FakeVision();
        var models = new Dictionary<string, FakeModel>();
        void Add(string key, string kind, Type client)
        {
            var model = new FakeModel(key) { Kind = kind };
            model.Clients[client] = vision;
            services.Runtime.Add(model);
            models[key] = model;
        }

        Add("detr", ModelKinds.ObjectDetection, typeof(IObjectDetector));
        Add("depth", ModelKinds.DepthEstimation, typeof(IDepthEstimator));
        Add("vit", ModelKinds.ImageClassification, typeof(IImageClassifier));
        Add("segformer", ModelKinds.SemanticSegmentation, typeof(ISemanticSegmenter));
        Add("sam", ModelKinds.ImageSegmentation, typeof(IImageSegmenter));
        Add("trocr", ModelKinds.Ocr, typeof(ITextRecognizer));
        Add("esrgan", ModelKinds.ImageUpscaling, typeof(IImageUpscaler));
        string photo = services.Directory + "/street.png";
        PngCodec.Write(photo, FakeImageGenerator.Gradient(480, 360, 2));
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Vision);
            viewModel.OpenKind(ModelKinds.ObjectDetection, models["detr"]);
            await viewModel.Vision.Source.SetFileAsync(photo);
            await AssertLayoutAsync(window, "vision empty result", typeof(VisionPage));
            foreach (string key in new[] { "detr", "depth", "vit", "segformer", "sam", "trocr", "esrgan" })
            {
                viewModel.Vision.SelectModel(models[key]);
                viewModel.Vision.RunCommand.Execute(null);
                await viewModel.Vision.Queue.Items[^1].Completion;
                await viewModel.Vision.Refresh;
                await AssertLayoutAsync(window, "vision " + key, typeof(VisionPage));
            }
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
        }
    });

    [Fact]
    public Task ImagePage_KeepsResultPreviewAndCompareInsideTheCard() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var sd = new FakeModel("sd") { Kind = ModelKinds.ImageGeneration };
        sd.Clients[typeof(IImageGenerator)] = new FakeImageGenerator { NativeSize = 384 };
        services.Runtime.Add(sd);
        string photo = services.Directory + "/street.png";
        PngCodec.Write(photo, FakeImageGenerator.Gradient(480, 360, 2));
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Image);
            viewModel.OpenKind(ModelKinds.ImageGeneration, sd);
            await AssertLayoutAsync(window, "image empty", typeof(ImagePage));
            viewModel.Image.Prompt = "a lighthouse";
            viewModel.Image.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "count").Value = 3;
            viewModel.Image.RunCommand.Execute(null);
            await viewModel.Image.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Image.HasResult && viewModel.Image.Generations.Visible.Count == 3 && viewModel.Image.Generations.Visible.All(item => item.Thumbnail is not null));
            await AssertLayoutAsync(window, "image result", typeof(ImagePage));
            await viewModel.Image.Source.SetFileAsync(photo);
            viewModel.Image.CompareWithInput = true;
            await AssertLayoutAsync(window, "image compare", typeof(ImagePage));
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
        }
    });

    [Fact]
    public Task VideoSpeakAndTranscribePages_KeepTheirResultsInsideTheCards() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var wan = new FakeModel("wan-tiny") { Kind = ModelKinds.VideoGeneration };
        wan.Clients[typeof(IVideoGenerator)] = new FakeVideoGenerator();
        var whisper = new FakeModel("whisper-small") { Kind = ModelKinds.SpeechRecognition };
        whisper.Clients[typeof(ISpeechRecognizer)] = new FakeRecognizer();
        var tts = new FakeModel("outetts") { Kind = ModelKinds.SpeechSynthesis };
        tts.Clients[typeof(ISpeechSynthesizer)] = new FakeSynthesizer { Voices = ["female_1", "male_1"] };
        services.Runtime.Add(wan);
        services.Runtime.Add(whisper);
        services.Runtime.Add(tts);
        string wav = services.Directory + "/jfk.wav";
        WavFile.Write(wav, new AudioBuffer([.. Enumerable.Range(0, 16000 * 11).Select(i => 0.2f * MathF.Sin(2 * MathF.PI * 220 * i / 16000))], 16000));
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Video);
            viewModel.OpenKind(ModelKinds.VideoGeneration, wan);
            viewModel.Video.Prompt = "a red fox";
            viewModel.Video.RunCommand.Execute(null);
            await viewModel.Video.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Video.HasResult);
            await AssertLayoutAsync(window, "video result", typeof(VideoPage));

            await viewModel.NavigateAsync(viewModel.Transcribe);
            viewModel.OpenKind(ModelKinds.SpeechRecognition, whisper);
            await viewModel.Transcribe.SetAudioAsync(wav);
            viewModel.Transcribe.RunCommand.Execute(null);
            await viewModel.Transcribe.Queue.Items.Single().Completion;
            await AssertLayoutAsync(window, "transcribe result", typeof(TranscribePage));

            viewModel.OpenKind(ModelKinds.SpeechSynthesis, tts);
            viewModel.Speak.InputText = "Ask not what your country can do for you.";
            viewModel.Speak.RunCommand.Execute(null);
            await viewModel.Speak.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Speak.HasResult);
            await AssertLayoutAsync(window, "speak result", typeof(SpeakPage));
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
        }
    });

    [Fact]
    public Task ChatLibraryAndDiscover_KeepEverythingInsideItsParents() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        await DemoData.FillAsync(services);
        string photo = services.Directory + "/street.png";
        PngCodec.Write(photo, FakeImageGenerator.Gradient(480, 360, 2));
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Chat);
            await HeadlessApp.WaitForAsync(window, () => viewModel.Chat.Current is not null);
            await viewModel.Chat.Current!.AddFilesAsync([photo]);
            await AssertWindowLayoutAsync(window, "chat with an attachment");

            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
            await AssertWindowLayoutAsync(window, "library");
            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            await HeadlessApp.WaitForAsync(window, () => !viewModel.Models.Discover.IsLoadingHighlights);
            await AssertWindowLayoutAsync(window, "discover");
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
        }
    });

    [Fact]
    public Task SearchPageChatVoiceAndServerPage_KeepEverythingInsideItsParents() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var chat = new FakeModel("qwen3");
        var whisper = new FakeModel("whisper-small") { Kind = ModelKinds.SpeechRecognition };
        whisper.Clients[typeof(ISpeechRecognizer)] = new FakeRecognizer();
        var tts = new FakeModel("outetts") { Kind = ModelKinds.SpeechSynthesis };
        tts.Clients[typeof(ISpeechSynthesizer)] = new FakeSynthesizer();
        var sd = new FakeModel("sdxl-turbo") { Kind = ModelKinds.ImageGeneration };
        sd.Clients[typeof(IImageGenerator)] = new FakeImageGenerator();
        var embed = new FakeModel("bge-small") { Kind = ModelKindCatalog.Embeddings, Embedder = new FakeTextEmbedder() };
        foreach (var model in new[] { chat, whisper, tts, sd, embed })
        {
            services.Runtime.Add(model);
        }

        string docs = services.Directory + "/docs";
        Directory.CreateDirectory(docs);
        File.WriteAllText(docs + "/pets.md", "The cat sleeps on the sofa. " + new string('x', 300) + " The dog barks at the ship in the harbour near the engine room.");
        File.WriteAllText(docs + "/sea.txt", "The ship engine roared. " + string.Join(' ', Enumerable.Repeat("harbour", 80)));
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Chat);
            await HeadlessApp.WaitForAsync(window, () => viewModel.Chat.Current is not null);
            viewModel.Chat.Current!.InputText = "/image a lighthouse at dusk, with a long description that wraps over several lines of the composer " + new string('z', 40);
            await viewModel.Chat.Current.SendCommand.ExecuteAsync(null);
            await HeadlessApp.WaitForAsync(window, () => viewModel.Chat.Current.Messages.Count == 2 && !viewModel.Chat.Current.IsGenerating);
            Assert.Single(viewModel.Chat.Current.Messages[1].Attachments);
            await AssertWindowLayoutAsync(window, "chat with a generated picture and voice settings");

            await viewModel.NavigateAsync(viewModel.Search);
            viewModel.OpenKind(ModelKindCatalog.Embeddings, embed);
            viewModel.Search.SetFolder(docs);
            viewModel.Search.IndexCommand.Execute(null);
            await viewModel.Search.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Search.HasIndex);
            viewModel.Search.Query = "ship engine";
            viewModel.Search.SearchCommand.Execute(null);
            await viewModel.Search.Queue.Items.Last().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Search.HasResults);
            await AssertLayoutAsync(window, "search results", typeof(SearchPage));

            await viewModel.NavigateAsync(viewModel.Server);
            await AssertWindowLayoutAsync(window, "server with every kind of model");
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
        }
    });

    /// <summary>Checks the pictures and cards of the whole window (without popups) at every test window size; the shared chrome has its own checks.</summary>
    private static async Task AssertWindowLayoutAsync(MainWindow window, string scenario)
    {
        var failures = new List<string>();
        foreach (var (width, height) in s_sizes)
        {
            window.Width = width;
            window.Height = height;
            await HeadlessApp.RenderAsync(window);
            foreach (string problem in LayoutChecker.Find(window, LayoutChecker.IsContentVisual).Distinct().Take(6))
            {
                failures.Add($"{scenario} at {width}x{height}: {problem}");
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>Checks the page of type <paramref name="page"/> at every test window size, and that its image viewers lie inside their cards.</summary>
    private static async Task AssertLayoutAsync(MainWindow window, string scenario, Type page)
    {
        var failures = new List<string>();
        foreach (var (width, height) in s_sizes)
        {
            window.Width = width;
            window.Height = height;
            await HeadlessApp.RenderAsync(window);
            var root = window.GetVisualDescendants().OfType<Control>().FirstOrDefault(control => control.GetType() == page);
            Assert.NotNull(root);
            // The page's own workspace; the model picker, the run queue and the side panel are shared chrome.
            var workspace = root.GetVisualDescendants().OfType<ContentPresenter>().Single(presenter => presenter.Name == "WorkspacePresenter");
            foreach (string problem in LayoutChecker.Find(workspace).Distinct().Take(6))
            {
                failures.Add($"{scenario} at {width}x{height}: {problem}");
            }

            if (s_screenshots is not null && width == 1100)
            {
                Directory.CreateDirectory(s_screenshots);
                using var frame = window.CaptureRenderedFrame();
                frame!.Save(Path.Combine(s_screenshots, $"studio-layout-{scenario.Replace(' ', '-')}-light.png"), new PngBitmapEncoderOptions());
            }

            foreach (var viewer in root.GetVisualDescendants().OfType<ImageViewer>().Where(viewer => viewer.IsEffectivelyVisible))
            {
                var card = viewer.GetVisualAncestors().OfType<Border>().FirstOrDefault(border => border.Classes.Contains("card"));
                Assert.NotNull(card);
                var painted = LayoutChecker.Painted(viewer, window, window);
                var cardBounds = LayoutChecker.BoundsIn(card, window);
                if (painted.Width > 0 && (painted.Top < cardBounds.Top - 1.5 || painted.Bottom > cardBounds.Bottom + 1.5 || painted.Left < cardBounds.Left - 1.5 || painted.Right > cardBounds.Right + 1.5))
                {
                    failures.Add($"{scenario} at {width}x{height}: image viewer {painted} leaves its card {cardBounds}");
                }
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
