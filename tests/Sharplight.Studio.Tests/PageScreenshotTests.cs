using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Studio.Models;
using Sharplight.Studio.Tests.FirstRun;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// One screenshot test per page and per Settings tab and Discover source, in the light and the dark theme at 1440 and 900 pixels wide. The pages
/// are filled with demo data, and each test checks that the page binds without errors and that no control is clipped or painted over its
/// neighbours (<see cref="LayoutChecker"/>), then saves the frame as a PNG. There are no golden images: the PNGs are test artifacts that CI
/// uploads (<c>studio.yml</c>) for a human to look at, into <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> (default: <c>page-screenshots</c> in the test output).
/// </summary>
public sealed class PageScreenshotTests
{
    private static readonly (double Width, double Height)[] s_sizes = [(1440, 900), (900, 600)];

    private static readonly (string Name, Func<MainWindowViewModel, MainWindow, Task> Show)[] s_pages =
    [
        ("chat", (model, _) => model.NavigateAsync(model.Chat)),
        ("welcome", async (model, window) =>
        {
            await model.NavigateAsync(model.Welcome);
            await model.Welcome.Loading;
            await HeadlessApp.WaitForAsync(window, () => model.Welcome.HasItems);
        }),
        ("chat-get-model", async (model, _) =>
        {
            await model.NavigateAsync(model.Chat);
            model.Chat.NewChatCommand.Execute(null);
        }),
        ("transcribe", (model, _) => model.NavigateAsync(model.Transcribe)),
        ("speak", (model, _) => model.NavigateAsync(model.Speak)),
        ("search", (model, _) => model.NavigateAsync(model.Search)),
        ("audio", (model, _) => model.NavigateAsync(model.Audio)),
        ("image", (model, _) => model.NavigateAsync(model.Image)),
        ("video", (model, _) => model.NavigateAsync(model.Video)),
        ("vision", (model, _) => model.NavigateAsync(model.Vision)),
        ("tasks", (model, _) => model.NavigateAsync(model.Tasks)),
        ("server", (model, _) => model.NavigateAsync(model.Server)),
        ("stats", (model, _) => model.NavigateAsync(model.Stats)),
        ("models-discover-verified", async (model, window) => await ShowDiscoverAsync(model, window, discover => discover.ShowVerifiedCommand.Execute(null))),
        ("models-discover-hub", async (model, window) => await ShowDiscoverAsync(model, window, discover => discover.ShowHubSearchCommand.Execute(null))),
        ("models-discover-feed", async (model, window) =>
        {
            await ShowDiscoverAsync(model, window, discover => discover.ShowFeedCommand.Execute(null));
            await model.Models.Discover.Feed.LoadAsync();
            await HeadlessApp.WaitForAsync(window, () => model.Models.Discover.Feed.Entries.Count > 0);
        }),
        ("models-library", async (model, _) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
        }),
        ("models-resident", async (model, _) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.ResidentTab;
        }),
        ("models-downloads", async (model, _) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.DownloadsTab;
        }),
        ("settings-general", (model, _) => ShowSettingsAsync(model, SettingsTab.General)),
        ("settings-storage", (model, _) => ShowSettingsAsync(model, SettingsTab.Storage)),
        ("settings-models", (model, _) => ShowSettingsAsync(model, SettingsTab.Models)),
        ("settings-engine", (model, _) => ShowSettingsAsync(model, SettingsTab.Engine)),
        ("settings-updates", (model, _) => ShowSettingsAsync(model, SettingsTab.Updates)),
        ("settings-data", (model, _) => ShowSettingsAsync(model, SettingsTab.DataAndAbout)),
    ];

    public static TheoryData<string, string, int> Cases
    {
        get
        {
            var data = new TheoryData<string, string, int>();
            foreach (var (name, _) in s_pages)
            {
                foreach (string theme in new[] { "light", "dark" })
                {
                    foreach (var (width, _) in s_sizes)
                    {
                        data.Add(name, theme, (int)width);
                    }
                }
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public Task Page_RendersWithoutBindingErrorsOrClippedControls(string page, string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        double height = s_sizes.Single(size => (int)size.Width == width).Height;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await WelcomeFeed.PublishAsync(rig);
        await DemoData.FillAsync(rig.Services);
        if (page == "chat-get-model")
        {
            foreach (var resident in rig.Services.Runtime.LoadedModels.ToList())
            {
                await rig.Services.Runtime.UnloadAsync(resident.Id);
            }
        }

        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All);
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            await s_pages.Single(entry => entry.Name == page).Show(viewModel, window);
            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.RenderAsync(window);

            var problems = LayoutChecker.Find(window, IsChecked).Distinct().Take(8).ToList();
            Save(window, $"{page}-{theme}-{width}");
            Assert.True(problems.Count == 0, $"{page} ({theme}, {width} px):\n" + string.Join("\n", problems));
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    public static TheoryData<string, int> SearchCases
    {
        get
        {
            var data = new TheoryData<string, int>();
            foreach (string theme in new[] { "light", "dark" })
            {
                foreach (var (width, _) in s_sizes)
                {
                    data.Add(theme, (int)width);
                }
            }

            return data;
        }
    }

    /// <summary>The Search page after a hybrid Ask: highlighted passages with their places, the answer with its citations, the watch switch and the chat model picker.</summary>
    [Theory]
    [MemberData(nameof(SearchCases))]
    public Task SearchPage_WithResultsAndAnAnswer_RendersWithoutBindingErrorsOrClippedControls(string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        double height = s_sizes.Single(size => (int)size.Width == width).Height;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await DemoData.FillAsync(rig.Services);
        rig.Services.Runtime.Add(new Fakes.FakeModel("fake-embed") { Kind = ModelKindCatalog.Embeddings, Embedder = new Fakes.FakeTextEmbedder() });
        var chat = rig.Services.Runtime.LoadedModels.First(model => model.Id == "qwen3-0.6b");
        string folder = Path.Combine(rig.Services.Directory, "docs");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "harbour.md"), "# Harbour\nThe ship engine roared as the ship left the harbour. The cat watched the ship from the quay.\n");
        DocumentFixtures.Pdf(Path.Combine(folder, "manual.pdf"), "Page one is about taxes.", "Engine maintenance: check the ship engine every week.");
        File.WriteAllText(Path.Combine(folder, "pets.txt"), "The dog barks. The cat sleeps.\n");

        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All with { Dispatcher = new AvaloniaDispatcher() });
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Search);
            var page = viewModel.Search;
            await HeadlessApp.WaitForAsync(window, () => page.HasModel && page.CanAskModel);
            page.SetFolder(folder);
            page.WatchFolder = true;
            page.IndexCommand.Execute(null);
            await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromSeconds(30));
            await HeadlessApp.WaitForAsync(window, () => page.HasIndex);
            page.Panel.Apply(new Dictionary<string, string> { ["mode"] = "hybrid", ["topk"] = "3" });
            page.Query = "how do I look after the ship engine";
            Assert.True(page.AskCommand.CanExecute(null));
            ((Fakes.FakeModel)chat).Script.Clear();
            ((Fakes.FakeModel)chat).Script.AddRange([Services.Runtime.ChatStreamUpdate.Content("Check the ship engine every week [1], and watch it when the ship leaves the harbour [2]."),
                Services.Runtime.ChatStreamUpdate.Finished(new MessageStats())]);
            page.AskCommand.Execute(null);
            await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromSeconds(30));
            await HeadlessApp.WaitForAsync(window, () => page.HasResults && page.HasAnswer);
            Assert.NotEmpty(page.AnswerCitations);
            page.FocusResultCommand.Execute(page.Results[0]);
            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.RenderAsync(window);

            var problems = LayoutChecker.Find(window, IsChecked).Distinct().Take(8).ToList();
            Save(window, $"search-results-{theme}-{width}");
            Assert.True(problems.Count == 0, $"search results ({theme}, {width} px):\n" + string.Join("\n", problems));
            page.WatchFolder = false;
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    /// <summary>Everything except sliders (their template overlaps the row's caption by design) and bare text (its line boxes overlap by a pixel or two).</summary>
    private static bool IsChecked(Avalonia.Visual visual) => visual is not (Avalonia.Controls.Slider or Avalonia.Controls.TextBlock);

    private static async Task ShowDiscoverAsync(MainWindowViewModel model, MainWindow window, Action<HubSearchViewModel> choose)
    {
        await model.NavigateAsync(model.Models);
        model.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
        await HeadlessApp.WaitForAsync(window, () => !model.Models.Discover.IsLoadingHighlights);
        choose(model.Models.Discover);
    }

    private static async Task ShowSettingsAsync(MainWindowViewModel model, SettingsTab tab)
    {
        await model.NavigateAsync(model.Settings);
        model.Settings.ShowTab(tab);
    }

    private static void Save(MainWindow window, string name)
    {
        string directory = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS") is { Length: > 0 } configured
            ? configured
            : Path.Combine(AppContext.BaseDirectory, "page-screenshots");
        Directory.CreateDirectory(directory);
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame.Save(Path.Combine(directory, $"studio-page-{name}.png"), new PngBitmapEncoderOptions());
    }
}
