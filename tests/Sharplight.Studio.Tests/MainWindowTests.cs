using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Renders the main window with every page on the headless platform and checks that the views bind cleanly. Set
/// <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> to a directory to also save each page as a PNG (used for docs/studio.md).
/// </summary>
public sealed class MainWindowTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    [Fact]
    public Task EveryPage_RendersWithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        await DemoData.FillAsync(services);
        var theme = new ThemeService(Application.Current!);
        theme.Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Chat);
            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.WaitForAsync(window, () => viewModel.Chat.Current is not null);
            Assert.NotNull(viewModel.Chat.Current);
            Assert.Equal(2, viewModel.Chat.Current.Messages.Count);
            Save(window, "chat");

            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(3, viewModel.Models.Library.Entries.Count);
            Save(window, "library");

            viewModel.Models.SelectedTab = ModelsPageViewModel.ResidentTab;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(3, viewModel.Models.Resident.Models.Count);
            Save(window, "resident");

            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            viewModel.Models.Discover.Query = "qwen3 gguf";
            await viewModel.Models.Discover.SearchCommand.ExecuteAsync(null);
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(4, viewModel.Models.Discover.Results.Count);
            Save(window, "discover");

            viewModel.Models.Downloads.Enqueue(Sharplight.Hub.ModelReference.Parse("Qwen/Qwen3-8B-GGUF:Q4_K_M"), 5_030_000_000);
            await services.Hub.Started.Task;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(ModelsPageViewModel.DownloadsTab, viewModel.Models.SelectedTab);
            Assert.Equal(1, viewModel.Downloads.ActiveCount);
            Save(window, "downloads");

            await viewModel.NavigateAsync(viewModel.Server);
            await HeadlessApp.RenderAsync(window);
            Save(window, "server");

            await viewModel.NavigateAsync(viewModel.Stats);
            viewModel.Stats.SampleNow();
            await HeadlessApp.RenderAsync(window);
            Assert.False(viewModel.Stats.IsEmpty);
            Save(window, "stats");

            await viewModel.NavigateAsync(viewModel.Settings);
            await HeadlessApp.RenderAsync(window);
            Save(window, "settings");

            theme.Apply(AppTheme.Light, "Ocean");
            await viewModel.NavigateAsync(viewModel.Chat);
            await HeadlessApp.RenderAsync(window);
            Save(window, "chat-light");
        }
        finally
        {
            foreach (var item in viewModel.Models.Downloads.Items)
            {
                item.CancelCommand.Execute(null);
            }

            services.Hub.Release();
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    private static void Save(MainWindow window, string name)
    {
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        if (s_screenshots is null)
        {
            return;
        }

        Directory.CreateDirectory(s_screenshots);
        frame.Save(Path.Combine(s_screenshots, $"studio-{name}.png"), new PngBitmapEncoderOptions());
    }
}
