using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Renders Discover's verified-model list with the committed catalog (text generation, and an image model installed with its companions)
/// and checks that the view binds cleanly. Set <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> to a directory to also save the pages as PNGs.
/// </summary>
public sealed class CatalogScreenshotTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    [Fact]
    public Task VerifiedCatalog_RendersWithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        await DemoData.FillAsync(services);
        services.Hub.Catalog = await new VerifiedCatalogSource(null, null).GetAsync(CancellationToken.None);
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            var discover = viewModel.Models.Discover;
            var verified = discover.Verified;
            await HeadlessApp.WaitForAsync(window, () => verified.AllEntries.Count > 0 && discover.IsVerifiedMode);
            verified.SelectedKind = verified.Kinds.Single(kind => kind.Value == "text-generation");
            verified.SelectedEntry = verified.Entries.Single(entry => entry.Entry.Id == "qwen3-8b");
            await HeadlessApp.RenderAsync(window);
            Save(window, "discover-verified");

            verified.SelectedKind = verified.Kinds.Single(kind => kind.Value == "image-generation");
            verified.SelectedEntry = verified.Entries.Single(entry => entry.Entry.Id == "flux.1-schnell");
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(3, verified.SelectedVariant!.Companions.Count);
            Save(window, "discover-verified-companions");

            discover.ShowHubSearchCommand.Execute(null);
            await HeadlessApp.RenderAsync(window);
            Assert.True(discover.IsHubMode);
            Assert.Empty(sink.Messages);
        }
        finally
        {
            window.Close();
        }
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
