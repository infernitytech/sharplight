using Avalonia.Controls;
using Avalonia.VisualTree;
using Sharplight.Studio.Controls;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Updates;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.ViewModels.Updates;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

public sealed class UpdatesViewModelTests
{
    private static readonly UpdateOffer s_offer = new("0.2.0", "## What's new\n\n- faster", 12_000_000);

    [Fact]
    public void Startup_WithAutoCheck_FindsTheUpdate_AndOpensTheDialog()
    {
        using var services = new TestServices();
        services.Updates.Result = UpdateCheckResult.Found(s_offer);
        using var updates = new UpdatesViewModel(services.Services, interval: TimeSpan.FromDays(1));

        _ = updates.StartAsync();

        Assert.Equal(1, services.Updates.Checks);
        Assert.True(updates.IsDialogOpen);
        Assert.Equal(s_offer, updates.Offer);
        Assert.Contains("0.2.0", updates.OfferTitle);
        Assert.Contains("0.1.5", updates.OfferTitle);
        Assert.NotNull(services.Settings.Current.LastUpdateCheck);
    }

    [Fact]
    public void Startup_WithAutoCheckOff_DoesNotCheck()
    {
        using var services = new TestServices(new AppSettings { AutoCheckForUpdates = false });
        using var updates = new UpdatesViewModel(services.Services, interval: TimeSpan.FromDays(1));

        _ = updates.StartAsync();

        Assert.Equal(0, services.Updates.Checks);
        Assert.False(updates.AutoCheck);
    }

    [Fact]
    public void ToggleAutoCheck_IsSaved()
    {
        using var services = new TestServices();
        using var updates = new UpdatesViewModel(services.Services);

        updates.AutoCheck = false;

        Assert.False(services.Settings.Current.AutoCheckForUpdates);
    }

    [Fact]
    public async Task UnsupportedBuild_ShowsWhy_AndNeverChecks()
    {
        using var services = new TestServices();
        services.Updates.Support = UpdateSupport.Unsupported("Updates are unavailable for this build: not installed.");
        services.Updates.Channel = null;
        using var updates = new UpdatesViewModel(services.Services);

        _ = updates.StartAsync();
        await updates.CheckNowCommand.ExecuteAsync(null);

        Assert.Equal(0, services.Updates.Checks);
        Assert.False(updates.IsSupported);
        Assert.Contains("unavailable for this build", updates.UnavailableReason);
        Assert.Contains("unavailable for this build", updates.StatusText);
        Assert.False(updates.IsDialogOpen);
    }

    [Fact]
    public async Task CheckNow_WhenUpToDate_SaysSo_WithoutADialog()
    {
        using var services = new TestServices();
        using var updates = new UpdatesViewModel(services.Services);

        await updates.CheckNowCommand.ExecuteAsync(null);

        Assert.Equal(1, services.Updates.Checks);
        Assert.False(updates.IsDialogOpen);
        Assert.Null(updates.Offer);
        Assert.Contains("up to date", updates.StatusText);
        Assert.Equal("just now", updates.LastCheckedText);
    }

    [Fact]
    public async Task CheckNow_ReportsFailure_ButAutomaticChecksStaySilent()
    {
        using var services = new TestServices();
        services.Updates.Result = UpdateCheckResult.Failed("no network");
        using var updates = new UpdatesViewModel(services.Services);

        await updates.CheckAsync(manual: false);
        Assert.Equal("", updates.StatusText);
        await updates.CheckNowCommand.ExecuteAsync(null);
        Assert.Contains("no network", updates.StatusText);
    }

    [Fact]
    public async Task Later_ClosesTheDialog_KeepsTheOffer_AndDoesNotReopenForTheSameVersion()
    {
        using var services = new TestServices();
        services.Updates.Result = UpdateCheckResult.Found(s_offer);
        using var updates = new UpdatesViewModel(services.Services);
        await updates.CheckAsync(manual: false);
        Assert.True(updates.IsDialogOpen);

        updates.LaterCommand.Execute(null);
        await updates.CheckAsync(manual: false);

        Assert.False(updates.IsDialogOpen);
        Assert.True(updates.HasOffer);
        Assert.True(updates.CanDownload);
    }

    [Fact]
    public async Task Download_ReportsProgress_ThenOffersRestart_ThatAppliesTheUpdate()
    {
        using var services = new TestServices();
        services.Updates.Result = UpdateCheckResult.Found(s_offer);
        using var updates = new UpdatesViewModel(services.Services);
        await updates.CheckAsync(manual: false);
        Assert.False(updates.RestartCommand.CanExecute(null));

        await updates.DownloadCommand.ExecuteAsync(null);

        Assert.Equal(100, updates.DownloadPercent);
        Assert.True(updates.IsReady);
        Assert.False(updates.IsDownloading);
        Assert.False(updates.CanDownload);
        Assert.Contains(40, services.Updates.Reported);
        updates.RestartCommand.Execute(null);
        Assert.Equal(s_offer, services.Updates.Applied);
    }

    [Fact]
    public async Task Download_Failure_IsShown_AndCanBeRetried()
    {
        using var services = new TestServices();
        services.Updates.Result = UpdateCheckResult.Found(s_offer);
        services.Updates.DownloadError = new IOException("disk full");
        using var updates = new UpdatesViewModel(services.Services);
        await updates.CheckAsync(manual: false);

        await updates.DownloadCommand.ExecuteAsync(null);

        Assert.Contains("disk full", updates.Error);
        Assert.False(updates.IsReady);
        Assert.True(updates.CanDownload);
    }

    [Fact]
    public async Task Offer_WithoutNotes_ShowsAPlaceholder()
    {
        using var services = new TestServices();
        services.Updates.Result = UpdateCheckResult.Found(new UpdateOffer("0.2.0", null, 0));
        using var updates = new UpdatesViewModel(services.Services);

        await updates.CheckAsync(manual: false);

        Assert.Contains("No release notes", updates.ReleaseNotes);
    }

    [Fact]
    public void SettingsPage_ShowsVersionAndChannel()
    {
        using var services = new TestServices();
        var page = new SettingsPageViewModel(services.Services);

        Assert.Equal("0.1.5", page.Updates.CurrentVersion);
        Assert.Equal("linux-x64-stable", page.Updates.ChannelText);
    }

    [Fact]
    public Task Dialog_RendersNotesAndButtons_WithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Avalonia.Logging.Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        services.Updates.Result = UpdateCheckResult.Found(s_offer);
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1200, Height = 800 };
        window.Show();
        try
        {
            var dialog = window.FindControl<Panel>("UpdateDialog")!;
            Assert.False(dialog.IsVisible);

            await viewModel.Updates.CheckAsync(manual: true);
            await HeadlessApp.RenderAsync(window);

            Assert.True(dialog.IsVisible);
            Assert.True(window.FindControl<Button>("UpdateDownloadButton")!.IsVisible);
            Assert.False(window.FindControl<Button>("UpdateRestartButton")!.IsVisible);
            Assert.NotEmpty(dialog.GetVisualDescendants().OfType<MarkdownView>());

            await viewModel.Updates.DownloadCommand.ExecuteAsync(null);
            await HeadlessApp.RenderAsync(window);
            Assert.True(window.FindControl<Button>("UpdateRestartButton")!.IsVisible);

            viewModel.Updates.LaterCommand.Execute(null);
            await HeadlessApp.RenderAsync(window);
            Assert.False(dialog.IsVisible);
            Assert.Empty(sink.Messages);
        }
        finally
        {
            window.Close();
        }
    });
}
