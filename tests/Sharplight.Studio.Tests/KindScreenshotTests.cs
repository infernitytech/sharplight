using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Hub;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Renders the kind-aware pages (library kind filter, Tasks, Discover filters and per-type highlights, a multi-component repository) and
/// checks that the views bind cleanly. Set <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> to a directory to also save each page as a PNG.
/// </summary>
public sealed class KindScreenshotTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    [Fact]
    public Task KindPages_RenderWithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        await DemoData.FillAsync(services);
        AddOtherKinds(services);
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
            await HeadlessApp.RenderAsync(window);
            var library = viewModel.Models.Library;
            Assert.Equal(6, library.Entries.Count);
            Assert.True(library.HasKindFilter);
            Save(window, "library-kinds");

            library.SelectKind("video-generation");
            await HeadlessApp.RenderAsync(window);
            Assert.Single(library.VisibleEntries);
            Save(window, "library-kind-filter");
            library.SelectKind(ModelKinds.TextGeneration);

            await viewModel.NavigateAsync(viewModel.Tasks);
            await HeadlessApp.RenderAsync(window);
            Assert.Contains(viewModel.Tasks.Kinds, card => card.IsPlanned);
            Save(window, "tasks");

            // A model type with filters: the speech models, and the filter bar with its choices applied.
            var discover = viewModel.Models.Discover;
            await viewModel.NavigateAsync(viewModel.Models);
            viewModel.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
            discover.Filters.SelectedTask = discover.Filters.Tasks.Single(option => option.Value == ModelTask.SpeechToText);
            await HeadlessApp.WaitForAsync(window, () => discover.Highlights.Count == 2 && !discover.IsLoadingHighlights);
            await HeadlessApp.RenderAsync(window);
            Save(window, "discover-task-highlights");

            // The image models as a search: a multi-component repository is one model with its components.
            services.Hub.Results.Clear();
            services.Hub.Results.AddRange(
            [
                new HubModelSummary("example-lab/Tiny-Diffusion-XL", 640_000, 3_900, false, "diffusers", "text-to-video", null, [ModelFormat.Diffusers], [], License: "apache-2.0"),
                new HubModelSummary("example-lab/Pixel-Upscaler", 88_000, 410, false, "diffusers", "image-to-image", null, [ModelFormat.Diffusers], [], License: "mit"),
                new HubModelSummary("example-lab/Mini-Flow", 12_000, 120, true, "diffusers", "text-to-image", null, [ModelFormat.Diffusers], [], License: "other"),
            ]);
            services.Hub.Details = repository => new RepositoryDetails(new HubRepository(repository, "main", "abc", false, [], null), ModelFormat.Diffusers,
                [new Studio.Services.Hub.VariantInfo("fp16", 3_470_000_000, 14, false, true), new Studio.Services.Hub.VariantInfo("full", 6_940_000_000, 14, false, false)], ModelFormat.Diffusers,
                [new HubComponent("unet", 5_140_000_000, 1), new HubComponent("text_encoder_2", 1_390_000_000, 1), new HubComponent("text_encoder", 246_000_000, 1),
                    new HubComponent("vae", 167_000_000, 1)]);
            discover.Filters.SelectedTask = discover.Filters.Tasks.Single(option => option.Value == ModelTask.ImageGeneration);
            discover.Filters.SelectedSort = discover.Filters.Sorts.Single(option => option.Value == HubSortOrder.Trending);
            discover.Filters.SelectedGated = discover.Filters.GatedChoices.Single(option => option.Value == GatedFilter.Hide);
            await HeadlessApp.WaitForAsync(window, () => discover.IsShowingResults && !discover.IsSearching && discover.Results.Count == 2);
            Assert.Equal(2, discover.Results.Count);
            await HeadlessApp.WaitForAsync(window, () => discover.IsMultiComponent);
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(4, discover.Components.Count);
            Assert.True(discover.IsPulledAsOne);
            Assert.Equal(["fp16", "full"], discover.Variants.Select(variant => variant.Tag));
            Assert.Equal("", discover.NoDownloadReason);
            Save(window, "discover-multi-component");

            // Reset: back to the default highlights.
            discover.Filters.ResetCommand.Execute(null);
            await HeadlessApp.WaitForAsync(window, () => discover.IsShowingHighlights);
            await HeadlessApp.RenderAsync(window);
            Assert.False(discover.Filters.HasActiveFilters);
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    private static void AddOtherKinds(TestServices services)
    {
        var now = DateTimeOffset.UtcNow;
        LibraryEntry Other(string id, string kind, long size, string architecture, bool inferred = false) => new()
        {
            Id = id, DisplayName = id, Path = $"/models/{id}.slm", Architecture = architecture, Format = "SLM", SizeBytes = size, Kind = kind, KindInferred = inferred,
            AddedAt = now.AddDays(-2),
        };
        services.Library.Entries.Add(Other("example-lab/Tiny-Diffusion-XL", "video-generation", 6_900_000_000, "sdxl"));
        services.Library.Entries.Add(Other("openai/whisper-small", ModelKinds.SpeechRecognition, 488_000_000, "whisper"));
        services.Library.Entries.Add(Other("example-lab/bge-small", ModelKindCatalog.Embeddings, 133_000_000, "bert"));
        services.Hub.TaskHighlights[ModelTask.SpeechToText] =
        [
            new HighlightSection(HighlightKind.Trending,
            [
                new HubModelSummary("ggerganov/whisper.cpp", 480_000, 1_400, false, null, "automatic-speech-recognition", null, [ModelFormat.WhisperCpp], [], License: "mit"),
                new HubModelSummary("openai/whisper-large-v3-turbo", 3_200_000, 2_600, false, "transformers", "automatic-speech-recognition", 809_000_000, [ModelFormat.Safetensors], [],
                    ModelType: "whisper", License: "mit"),
            ]),
            new HighlightSection(HighlightKind.New, []),
            new HighlightSection(HighlightKind.Popular,
            [
                new HubModelSummary("nvidia/parakeet-tdt-0.6b-v3", 1_900_000, 900, false, "nemo", "automatic-speech-recognition", 600_000_000, [], [], License: "cc-by-4.0"),
            ]),
        ];
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
