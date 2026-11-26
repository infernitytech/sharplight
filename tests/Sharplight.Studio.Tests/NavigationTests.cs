using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Headless;
using Avalonia.Input;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Navigation;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The grouped navigation: every runnable kind opens its page under the expected modality group (and expands it), every page is listed,
/// collapsed groups and the folded rail are remembered, the Models tabs are entries of their own, and the headless window shows the groups
/// when wide and the rail when narrow, with keyboard navigation. Screenshots go to <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> (default:
/// <c>page-screenshots</c> in the test output) as <c>studio-nav-*.png</c>.
/// </summary>
public sealed class NavigationTests
{
    /// <summary>Kind, group id and entry title: the grouping by the modality of the output.</summary>
    public static TheoryData<string, string, string> KindCases => new()
    {
        { ModelKinds.TextGeneration, "text", "Chat" },
        { ModelKindCatalog.Embeddings, "text", "Search" },
        { ModelKindCatalog.Reranker, "text", "Search" },
        { ModelKinds.TextEmbedding, "text", "Search" },
        { ModelKinds.TextReranking, "text", "Search" },
        { ModelKinds.ImageGeneration, "images", "Generate" },
        { ModelKinds.ImageClassification, "images", "Vision" },
        { ModelKinds.ImageEmbedding, "images", "Vision" },
        { ModelKinds.ObjectDetection, "images", "Vision" },
        { ModelKinds.ImageSegmentation, "images", "Vision" },
        { ModelKinds.SemanticSegmentation, "images", "Vision" },
        { ModelKinds.DepthEstimation, "images", "Vision" },
        { ModelKinds.Ocr, "images", "Vision" },
        { ModelKinds.ImageUpscaling, "images", "Vision" },
        { ModelKinds.VideoGeneration, "video", "Generate" },
        { ModelKinds.VideoEnhancement, "video", "Enhance" },
        { ModelKinds.SpeechRecognition, "audio", "Transcribe" },
        { ModelKinds.SpeechSynthesis, "audio", "Speak" },
        { ModelKinds.AudioCodec, "audio", "Music & tools" },
        { ModelKinds.VoiceActivity, "audio", "Music & tools" },
        { ModelKinds.AudioEmbedding, "audio", "Music & tools" },
        { ModelKinds.MusicGeneration, "audio", "Music & tools" },
        { ModelKinds.AudioGeneration, "audio", "Music & tools" },
        { ModelKinds.AudioSeparation, "audio", "Music & tools" },
        { ModelKinds.TimeSeriesForecasting, "data", "Time series" },
    };

    [Theory]
    [MemberData(nameof(KindCases))]
    public void Kind_OpensItsPage_UnderItsGroup_AndExpandsTheGroup(string kind, string groupId, string title)
    {
        using var services = new TestServices();
        var main = new MainWindowViewModel(services.Services);
        var group = main.Navigation.Group(groupId)!;
        group.IsExpanded = false;

        Assert.True(main.OpenKind(kind, null), kind);

        var current = main.Navigation.Current;
        Assert.NotNull(current);
        Assert.Same(main.CurrentPage, current.Page);
        Assert.Same(group, current.Group);
        Assert.Equal(title, current.Title);
        Assert.True(current.IsCurrent);
        Assert.True(group.IsExpanded);
        Assert.True(group.ContainsCurrent);
        Assert.Single(main.Navigation.Items, item => item.IsCurrent);
        Assert.All(main.Navigation.Groups.Where(other => other != group), other => Assert.False(other.ContainsCurrent));
    }

    [Fact]
    public void EveryRunnableKindOfTheCatalog_HasAnEntry()
    {
        using var services = new TestServices();
        var main = new MainWindowViewModel(services.Services);
        var cases = KindCases.Select(row => row.Data.Item1).ToHashSet();
        foreach (var kind in ModelKindCatalog.All.Where(kind => kind.Runnable))
        {
            Assert.Contains(kind.Id, cases);
            Assert.NotNull(main.Navigation.ForKind(kind.Id));
        }

        Assert.Null(main.Navigation.ForKind("3d-generation"));
        Assert.False(main.OpenKind("3d-generation", null));
    }

    [Fact]
    public void EveryPage_IsListed_InTheExpectedPlace()
    {
        using var services = new TestServices();
        var main = new MainWindowViewModel(services.Services);
        var navigation = main.Navigation;

        Assert.Equal(["text", "images", "video", "audio", "data", "models"], navigation.Groups.Select(group => group.Id));
        Assert.Equal(["Chat", "Search"], navigation.Group("text")!.Items.Select(item => item.Title));
        Assert.Equal(["Generate", "Vision"], navigation.Group("images")!.Items.Select(item => item.Title));
        Assert.Equal(["Generate", "Enhance"], navigation.Group("video")!.Items.Select(item => item.Title));
        Assert.Equal(["Transcribe", "Speak", "Music & tools"], navigation.Group("audio")!.Items.Select(item => item.Title));
        Assert.Equal(["Time series"], navigation.Group("data")!.Items.Select(item => item.Title));
        Assert.Equal(["Library", "Discover", "Downloads", "Loaded", "Storage", "Tasks"], navigation.Group("models")!.Items.Select(item => item.Title));
        Assert.Equal(["Server", "Statistics", "Settings"], navigation.Footer.Select(item => item.Title));
        Assert.All(navigation.Footer, item => Assert.Null(item.Group));

        PageViewModel[] pages = [main.Chat, main.Search, main.Image, main.Vision, main.Video, main.Enhance, main.Transcribe, main.Speak, main.Audio, main.TimeSeries,
            main.Models, main.Tasks, main.Server, main.Stats, main.Settings];
        Assert.Equal(pages.ToHashSet(), navigation.Items.Select(item => item.Page).ToHashSet());
        Assert.Equal(pages.Where(page => page != main.Settings), main.Pages);
        Assert.All(navigation.Items, item => Assert.False(string.IsNullOrEmpty(item.Description)));
    }

    [Fact]
    public async Task ModelsTabs_AreEntries_FollowingTheTab()
    {
        using var services = new TestServices();
        var main = new MainWindowViewModel(services.Services);
        var models = main.Navigation.Group("models")!;
        await main.NavigateAsync(main.Chat);

        models.Items.Single(item => item.Title == "Discover").OpenCommand.Execute(null);
        Assert.Same(main.Models, main.CurrentPage);
        Assert.Equal(ModelsPageViewModel.DiscoverTab, main.Models.SelectedTab);
        Assert.Equal("Discover", main.Navigation.Current!.Title);

        // The tab strip of the page moves the highlight too.
        main.Models.SelectedTab = ModelsPageViewModel.StorageTab;
        Assert.Equal("Storage", main.Navigation.Current!.Title);
        main.OpenDownloads();
        Assert.Equal("Downloads", main.Navigation.Current!.Title);
        main.OpenResident();
        Assert.Equal("Loaded", main.Navigation.Current!.Title);
        main.OpenLibrary(ModelKinds.SpeechSynthesis);
        Assert.Equal("Library", main.Navigation.Current!.Title);
        Assert.Single(main.Navigation.Items, item => item.IsCurrent);

        await main.NavigateAsync(main.Welcome);
        Assert.Null(main.Navigation.Current);
        Assert.DoesNotContain(main.Navigation.Items, item => item.IsCurrent);
    }

    [Fact]
    public void CollapsedGroups_AndTheFoldedRail_AreRemembered()
    {
        using var services = new TestServices();
        var main = new MainWindowViewModel(services.Services);
        Assert.All(main.Navigation.Groups, group => Assert.True(group.IsExpanded));
        Assert.False(main.Navigation.IsCompact);

        main.Navigation.ToggleGroupCommand.Execute(main.Navigation.Group("images"));
        main.Navigation.Group("models")!.IsExpanded = false;
        main.Navigation.ToggleCompactCommand.Execute(null);
        Assert.Equal(["images", "models"], services.Settings.Current.CollapsedNavigationGroups);
        Assert.True(services.Settings.Current.CompactNavigation);
        Assert.True(main.Navigation.IsRail);

        var again = new MainWindowViewModel(services.Services);
        Assert.False(again.Navigation.Group("images")!.IsExpanded);
        Assert.False(again.Navigation.Group("models")!.IsExpanded);
        Assert.True(again.Navigation.Group("text")!.IsExpanded);
        Assert.True(again.Navigation.IsCompact);

        again.Navigation.Group("images")!.IsExpanded = true;
        Assert.Equal(["models"], services.Settings.Current.CollapsedNavigationGroups);
    }

    [Fact]
    public void NarrowWindow_IsARail_ThatCannotBeUnfolded()
    {
        using var services = new TestServices();
        var navigation = new MainWindowViewModel(services.Services).Navigation;
        navigation.IsNarrow = true;
        Assert.True(navigation.IsRail);
        Assert.False(navigation.IsExpandedSidebar);
        Assert.False(navigation.CanToggleCompact);
        navigation.IsNarrow = false;
        Assert.True(navigation.IsExpandedSidebar);
        Assert.True(navigation.CanToggleCompact);
    }

    public static TheoryData<string, int> WindowCases => new() { { "dark", 1440 }, { "light", 1440 }, { "dark", 900 }, { "light", 900 } };

    /// <summary>
    /// The window: groups with labels when wide (a collapsed group hides its entries, the header toggles it, the arrow keys move between
    /// the entries), the rail when narrow (one icon per group, opening the group's entries); no binding errors or clipped controls.
    /// </summary>
    [Theory]
    [MemberData(nameof(WindowCases))]
    public Task Window_ShowsTheGroups_OrTheRail(string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        double height = width == 1440 ? 900 : 600;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        await DemoData.FillAsync(services);
        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            viewModel.OpenKind(ModelKinds.ImageGeneration, null);
            await HeadlessApp.RenderAsync(window);
            var sidebar = window.FindControl<Border>("Sidebar")!;
            var groups = window.FindControl<ItemsControl>("Nav")!;
            var rail = window.FindControl<ItemsControl>("NavRail")!;
            bool narrow = width < NavigationViewModel.RailWidth;
            Assert.Equal(narrow, viewModel.Navigation.IsRail);
            Assert.Equal(!narrow, groups.IsEffectivelyVisible);
            Assert.Equal(narrow, rail.IsEffectivelyVisible);
            Assert.Equal(narrow ? 64 : 232, sidebar.Bounds.Width, 0.5);

            var problems = LayoutChecker.Find(window, visual => visual is not (Slider or TextBlock)).Distinct().Take(8).ToList();
            Assert.True(problems.Count == 0, $"navigation ({theme}, {width} px):\n" + string.Join("\n", problems));

            if (narrow)
            {
                var railButtons = rail.GetVisualDescendants().OfType<Button>().Where(button => button.Classes.Contains("navrail")).ToList();
                Assert.Equal(viewModel.Navigation.Groups.Select(group => group.Title), railButtons.Select(button => Avalonia.Automation.AutomationProperties.GetName(button)));
                var images = railButtons[1];
                Assert.Contains("current", images.Classes);
                Save(window, $"nav-{theme}-{width}");

                // The group's flyout lists its entries; choosing one navigates and closes it.
                images.Flyout!.ShowAt(images);
                await HeadlessApp.RenderAsync(window);
                Assert.True(images.Flyout.IsOpen);
                Save(window, $"nav-rail-flyout-{theme}-{width}");
                viewModel.Navigation.Group("images")!.Items[1].OpenCommand.Execute(null);
                await HeadlessApp.RenderAsync(window);
                Assert.Same(viewModel.Vision, viewModel.CurrentPage);
                Assert.False(images.Flyout.IsOpen);
            }
            else
            {
                Save(window, $"nav-{theme}-{width}");
                var headers = groups.GetVisualDescendants().OfType<ToggleButton>().Where(button => button.Classes.Contains("navgroup")).ToList();
                Assert.Equal(viewModel.Navigation.Groups.Select(group => group.Title), headers.Select(button => Avalonia.Automation.AutomationProperties.GetName(button)));
                Assert.Contains("current", headers[1].Classes);

                // The header collapses its group; the entries leave the layout.
                var video = headers[2];
                video.IsChecked = false;
                await HeadlessApp.RenderAsync(window);
                Assert.False(viewModel.Navigation.Group("video")!.IsExpanded);
                Assert.Equal(["video"], services.Settings.Current.CollapsedNavigationGroups);
                Assert.DoesNotContain(VisibleEntries(groups), button => Avalonia.Automation.AutomationProperties.GetName(button) == "Enhance");
                Save(window, $"nav-collapsed-{theme}-{width}");

                // The current entry is highlighted; Down moves to the next entry, Left goes to the header and collapses it, Right expands it.
                var current = VisibleEntries(groups).Single(button => button.Classes.Contains("current"));
                Assert.Equal("Generate", Avalonia.Automation.AutomationProperties.GetName(current));
                current.Focus(NavigationMethod.Tab);
                window.KeyPressQwerty(PhysicalKey.ArrowDown, RawInputModifiers.None);
                Assert.Equal("Vision", Avalonia.Automation.AutomationProperties.GetName((Control)window.FocusManager!.GetFocusedElement()!));
                window.KeyPressQwerty(PhysicalKey.ArrowLeft, RawInputModifiers.None);
                Assert.Same(headers[1], window.FocusManager!.GetFocusedElement());
                window.KeyPressQwerty(PhysicalKey.ArrowLeft, RawInputModifiers.None);
                Assert.False(viewModel.Navigation.Group("images")!.IsExpanded);
                window.KeyPressQwerty(PhysicalKey.ArrowRight, RawInputModifiers.None);
                Assert.True(viewModel.Navigation.Group("images")!.IsExpanded);
                window.KeyPressQwerty(PhysicalKey.ArrowDown, RawInputModifiers.None);
                window.KeyPressQwerty(PhysicalKey.Enter, RawInputModifiers.None);
                await HeadlessApp.RenderAsync(window);
                Assert.Same(viewModel.Image, viewModel.CurrentPage);

                // Folding shows the rail; the choice sticks.
                window.FindControl<Button>("FoldButton")!.Command!.Execute(null);
                await HeadlessApp.RenderAsync(window);
                Assert.True(rail.IsEffectivelyVisible);
                Assert.Equal(64, sidebar.Bounds.Width, 0.5);
                Save(window, $"nav-folded-{theme}-{width}");
                Assert.True(services.Settings.Current.CompactNavigation);
            }
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    private static List<Button> VisibleEntries(Visual root)
        => [.. root.GetVisualDescendants().OfType<Button>().Where(button => button.Classes.Contains("navitem") && button.IsEffectivelyVisible)];

    private static void Save(MainWindow window, string name)
    {
        string directory = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS") is { Length: > 0 } configured
            ? configured
            : Path.Combine(AppContext.BaseDirectory, "page-screenshots");
        Directory.CreateDirectory(directory);
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame.Save(Path.Combine(directory, $"studio-{name}.png"), new PngBitmapEncoderOptions());
    }
}
