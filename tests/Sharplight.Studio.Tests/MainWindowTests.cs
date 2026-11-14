using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Studio.Models;
using Sharplight.Runtime;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
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

            // The load panel with its memory estimate: fits once an idle model unloads; at 64k tokens it needs CPU offload.
            services.Runtime.Estimator = request =>
            {
                var settings = request.Settings;
                double kv = settings.ContextLength * (128.0 * 1024) * DownloadEstimate.KvBytesPerElement(settings.KvCacheType) * settings.Slots / 2;
                return new ModelFootprint(8_700_000_000, (long)kv, 600_000_000);
            };
            var library = viewModel.Models.Library;
            library.LoadCommand.Execute(library.Entries[2]);
            var panel = library.LoadPanel!;
            await SettleAsync(window, panel);
            Assert.Equal(FitVerdict.FitsAfterUnloading, panel.Fit!.Verdict);
            Save(window, "load-fit");
            panel.ContextLength = 65536;
            await SettleAsync(window, panel);
            Assert.Equal(FitVerdict.NeedsOffload, panel.Fit!.Verdict);
            Assert.True(panel.Fit.HasSuggestions);
            Save(window, "load-fit-offload");
            panel.CloseCommand.Execute(null);

            viewModel.Models.SelectedTab = ModelsPageViewModel.ResidentTab;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(3, viewModel.Models.Resident.Models.Count);
            Save(window, "resident");

            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            viewModel.Models.Discover.Query = "qwen3 gguf";
            await viewModel.Models.Discover.SearchCommand.ExecuteAsync(null);
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(4, viewModel.Models.Discover.Results.Count);
            await viewModel.Models.Discover.FitEstimate;
            await HeadlessApp.RenderAsync(window);
            Assert.All(viewModel.Models.Discover.Variants, variant => Assert.NotNull(variant.Fit));
            Save(window, "discover");

            // A quantization that would not load as is: the warning sits above the download button, which stays enabled.
            services.Runtime.Memory = () =>
            [
                new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 12L << 30, 6L << 30, true),
                new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 32L << 30, 24L << 30, true),
            ];
            var discover = viewModel.Models.Discover;
            discover.SelectedResult = null;
            discover.SelectedResult = discover.Results[0];
            await HeadlessApp.WaitForAsync(window, () => discover.Variants.Count == 3);
            await discover.FitEstimate;
            discover.SelectedVariant = discover.Variants[^1];
            await HeadlessApp.RenderAsync(window);
            Assert.True(discover.HasFitWarning);
            Assert.True(discover.DownloadCommand.CanExecute(null));
            Save(window, "discover-fit");
            services.Runtime.Memory = null;

            // A repository whose architecture has no importer mapping: badged on the result and every quantization, explained above the
            // download button (which stays enabled), and listed after the supported results.
            services.Hub.Results.Clear();
            services.Hub.Results.AddRange(DemoData.CompatibilityResults);
            services.Hub.Compatibility = (repository, _) => repository.Contains("HRM", StringComparison.Ordinal)
                ? new ModelCompatibility(CompatibilityState.UnsupportedArchitecture, "hrm_text",
                    "No importer mapping exists for the GGUF architecture 'hrm_text', so importing it fails.")
                : new ModelCompatibility(CompatibilityState.Supported, "qwen3",
                    "The GGUF architecture 'qwen3' is supported. Tensor types: F32 (113), Q4_K (108), Q6_K (37), all with kernels.");
            discover.Query = "text gguf";
            await discover.SearchCommand.ExecuteAsync(null);
            var unsupported = discover.Results.Single(result => result.Id.Contains("HRM", StringComparison.Ordinal));
            Assert.Same(unsupported, discover.Results[^2]);
            discover.SelectedResult = unsupported;
            await HeadlessApp.WaitForAsync(window, () => discover.Variants.Count == 3);
            await discover.FitEstimate;
            await HeadlessApp.RenderAsync(window);
            Assert.True(discover.SelectedCompatibility!.IsBad);
            Assert.Equal("Unsupported architecture hrm_text", discover.SelectedCompatibility.Title);
            Assert.True(discover.DownloadCommand.CanExecute(null));
            Save(window, "discover-unsupported");

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

    /// <summary>Waits for the load panel's latest estimate (one can schedule another) and renders it.</summary>
    private static async Task SettleAsync(MainWindow window, LoadOptionsViewModel panel)
    {
        for (int i = 0; i < 20; i++)
        {
            var estimate = panel.Estimate;
            await estimate;
            await HeadlessApp.RenderAsync(window);
            if (ReferenceEquals(estimate, panel.Estimate) && !panel.IsEstimating)
            {
                return;
            }
        }
    }

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
