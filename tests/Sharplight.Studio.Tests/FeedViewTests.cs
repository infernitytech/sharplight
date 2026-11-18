using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The feed list in Discover, the update badge in the library and the Settings fields render without binding errors and keep their cards, lists and
/// banners inside their parents at every window size. Set <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> to a directory to also save PNGs.
/// </summary>
public sealed class FeedViewTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    private static readonly (double Width, double Height)[] s_sizes = [(1440, 900), (1100, 700), (900, 600)];

    private static async Task AssertLayoutAsync(MainWindow window, string scenario, string? screenshot = null)
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

            if (screenshot is not null && s_screenshots is not null && width == 1440)
            {
                Directory.CreateDirectory(s_screenshots);
                using var frame = window.CaptureRenderedFrame();
                frame?.Save(Path.Combine(s_screenshots, $"studio-{screenshot}.png"), new PngBitmapEncoderOptions());
            }
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public Task FeedList_RendersInEveryState_WithoutBindingErrorsOrOverlap() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(rig.All);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            var discover = viewModel.Models.Discover;
            discover.ShowFeedCommand.Execute(null);
            await discover.Feed.LoadAsync();
            await HeadlessApp.WaitForAsync(window, () => discover.Feed.Entries.Count > 0);
            await AssertLayoutAsync(window, "feed list", "discover-feed");

            // A finished install and a failed verification.
            await discover.Feed.Install(discover.Feed.Entries[1]).Completion;
            File.AppendAllText(rig.Feed.PathOf("packages/alpha.q8/1.0.0/manifest.json"), " ");
            discover.Feed.Selected = discover.Feed.Entries[0];
            await discover.Feed.Install(discover.Feed.Entries[0]).Completion;
            Assert.Equal(2, discover.Feed.Installs.Count);
            await AssertLayoutAsync(window, "feed installs", "discover-feed-installs");

            // A tampered feed, and an unreachable one with a verified copy.
            File.AppendAllText(rig.Feed.PathOf("stable/index.json"), " ");
            await discover.Feed.RefreshAsync();
            Assert.True(discover.Feed.IsVerificationFailure);
            await AssertLayoutAsync(window, "feed verification failure", "discover-feed-failed");
            Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages));
        }
        finally
        {
            window.Close();
        }
    });

    [Fact]
    public Task FeedList_OfflineBanner_RendersInsideThePage() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            var feed = viewModel.Models.Discover.Feed;
            viewModel.Models.Discover.ShowFeedCommand.Execute(null);
            await feed.LoadAsync();
            rig.Offline = true;
            await feed.RefreshAsync();
            Assert.True(feed.IsOffline);
            await HeadlessApp.WaitForAsync(window, () => feed.Entries.Count > 0);
            await AssertLayoutAsync(window, "feed offline", "discover-feed-offline");
            Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages));
        }
        finally
        {
            window.Close();
        }
    });

    [Fact]
    public Task Library_ShowsTheFeedBadgeAndTheUpdateButton() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(rig.All);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            var models = viewModel.Models;
            await viewModel.NavigateAsync(models);
            await models.Discover.Feed.LoadAsync();
            await models.Discover.Feed.Install(models.Discover.Feed.Entries[0]).Completion;

            // A newer version appears in the feed.
            await FeedCatalogTests.PublishAlphaAsync(rig, "1.1.0", seed: 2);
            await models.Discover.Feed.RefreshAsync();
            var stored = rig.Store.List().Single(model => model.FeedPackage == "alpha.q8");
            rig.Services.Library.Entries.Add(new LibraryEntry { Id = stored.ShortName, DisplayName = stored.Name, Path = stored.PackagePath, Format = "Slm", SizeBytes = 20_000, Manifest = stored });
            models.SelectedTab = ModelsPageViewModel.LibraryTab;
            await models.Library.RefreshAsync();

            var entry = Assert.Single(models.Library.Entries);
            Assert.True(entry.IsFromFeed);
            Assert.Equal("Feed 1.0.0", entry.FeedText);
            Assert.Equal("Update 1.1.0", entry.FeedUpdateText);
            await AssertLayoutAsync(window, "library update", "library-feed-update");

            models.Library.UpdateFromFeedCommand.Execute(entry);
            await models.Discover.Feed.Installs[0].Completion;
            await HeadlessApp.WaitForAsync(window, () => !entry.HasFeedUpdate);
            Assert.Equal("1.1.0", rig.Store.List().Single(model => model.FeedPackage == "alpha.q8").FeedVersion);
            Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages));
        }
        finally
        {
            window.Close();
        }
    });

    [Fact]
    public Task Settings_ModelsTab_HasTheFeedChannelAndAddress() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        new ThemeService(Application.Current!).Apply(AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Settings);
            viewModel.Settings.ShowTab(Sharplight.Studio.ViewModels.Settings.SettingsTab.Models);
            await HeadlessApp.RenderAsync(window);
            Assert.Equal("stable", viewModel.Settings.FeedChannel.Value);
            viewModel.Settings.FeedChannel = viewModel.Settings.FeedChannels[1];
            viewModel.Settings.FeedUrl = "  https://example.invalid/models ";
            Assert.Equal("unstable", services.Settings.Current.FeedChannel);
            Assert.Equal("https://example.invalid/models", services.Settings.Current.FeedUrl);
            viewModel.Settings.FeedChannel = viewModel.Settings.FeedChannels[0];
            viewModel.Settings.FeedUrl = "";
            Assert.Null(services.Settings.Current.FeedChannel);
            Assert.Null(services.Settings.Current.FeedUrl);
            await AssertLayoutAsync(window, "settings models", "settings-models-feed");
            Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages));
        }
        finally
        {
            window.Close();
        }
    });
}
