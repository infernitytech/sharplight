using Sharplight.Hub;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>Discover's filters: model type, format, quantization, size, fit, license, gated and sort; saved choices; per-type highlights.</summary>
public sealed class DiscoverFiltersTests
{
    private static HubModelSummary Model(string id, string? pipeline = "text-generation", long? parameters = null, bool gated = false, string? license = null,
        ModelFormat[]? formats = null, string[]? variants = null, string? method = null, string? library = null)
        => new(id, 1000, 10, gated, library, pipeline, parameters, formats ?? [ModelFormat.Gguf], variants ?? ["Q4_K_M"], GgufArchitecture: "qwen3",
            QuantizationMethod: method, License: license);

    private static HubSearchViewModel CreateSearch(TestServices services)
        => new(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());

    [Fact]
    public void Filters_StartAtTheirDefaults_AndAreSavedInTheSettings()
    {
        using var services = new TestServices();
        var filters = new DiscoverFiltersViewModel(services.Settings);
        Assert.Null(filters.SelectedTask.Value);
        Assert.False(filters.HasActiveFilters);
        Assert.Equal(HubSortOrder.Downloads, filters.SelectedSort.Value);
        Assert.Equal(ModelTasks.All.Count + 1, filters.Tasks.Count);

        int changes = 0;
        filters.FiltersChanged += (_, _) => changes++;
        filters.SelectedTask = filters.Tasks.Single(option => option.Value == ModelTask.SpeechToText);
        filters.SelectedFormat = filters.Formats.Single(option => option.Value == ModelFormat.WhisperCpp);
        filters.SelectedQuantization = filters.Quantizations.Single(option => option.Value == "Q5");
        filters.SelectedSize = filters.Sizes.Single(option => option.Value == ModelSizeFilter.From1To4B);
        filters.FitsThisMachine = true;
        filters.SelectedLicense = filters.Licenses.Single(option => option.Value == "mit");
        filters.SelectedGated = filters.GatedChoices.Single(option => option.Value == GatedFilter.Hide);
        filters.SelectedSort = filters.Sorts.Single(option => option.Value == HubSortOrder.Trending);
        Assert.Equal(8, changes);
        Assert.True(filters.HasActiveFilters);

        var saved = services.Settings.Current.Discover;
        Assert.Equal(ModelTask.SpeechToText, saved.Task);
        Assert.Equal(ModelFormat.WhisperCpp, saved.Format);
        Assert.Equal("Q5", saved.Quantization);
        Assert.Equal(ModelSizeFilter.From1To4B, saved.Size);
        Assert.True(saved.FitsThisMachine);
        Assert.Equal("mit", saved.License);
        Assert.Equal(GatedFilter.Hide, saved.Gated);
        Assert.Equal(HubSortOrder.Trending, saved.Sort);

        // A new view model reads the choices back; reset restores the defaults.
        var reopened = new DiscoverFiltersViewModel(services.Settings);
        Assert.Equal(saved, reopened.Current);
        reopened.ResetCommand.Execute(null);
        Assert.Equal(new DiscoverFilterSettings(), services.Settings.Current.Discover);
    }

    [Fact]
    public void Filters_SurviveTheSettingsFile()
    {
        using var directory = new TempDir();
        var service = new Studio.Services.Settings.JsonSettingsService(directory.File("settings.json"), null);
        service.Update(settings => settings with
        {
            Discover = new DiscoverFilterSettings { Task = ModelTask.VideoGeneration, Format = ModelFormat.Diffusers, Gated = GatedFilter.Only, Sort = HubSortOrder.Modified },
        });

        var reread = new Studio.Services.Settings.JsonSettingsService(directory.File("settings.json"), null).Current.Discover;
        Assert.Equal(ModelTask.VideoGeneration, reread.Task);
        Assert.Equal(ModelFormat.Diffusers, reread.Format);
        Assert.Equal(GatedFilter.Only, reread.Gated);
        Assert.Equal(HubSortOrder.Modified, reread.Sort);
    }

    [Fact]
    public void Matches_AppliesGatedQuantizationSizeAndFit()
    {
        const long gib = 1L << 30;
        var open = Model("a/open", parameters: 8_000_000_000);
        var gated = Model("a/gated", gated: true);
        var none = new DiscoverFilterSettings();

        Assert.True(DiscoverFilter.Matches(gated, none, null));
        Assert.False(DiscoverFilter.Matches(gated, none with { Gated = GatedFilter.Hide }, null));
        Assert.True(DiscoverFilter.Matches(open, none with { Gated = GatedFilter.Hide }, null));
        Assert.False(DiscoverFilter.Matches(open, none with { Gated = GatedFilter.Only }, null));

        // Quantization families match name segments: Q4 matches Q4_K_M but not IQ4_XS, F16 does not match BF16.
        var variants = Model("a/quants", variants: ["Q4_K_M", "IQ3_XS", "Q8_0", "BF16"]);
        Assert.True(DiscoverFilter.Matches(variants, none with { Quantization = "Q4" }, null));
        Assert.True(DiscoverFilter.Matches(variants, none with { Quantization = "IQ" }, null));
        Assert.True(DiscoverFilter.Matches(variants, none with { Quantization = "BF16" }, null));
        Assert.False(DiscoverFilter.Matches(variants, none with { Quantization = "F16" }, null));
        Assert.False(DiscoverFilter.Matches(variants, none with { Quantization = "Q5" }, null));
        Assert.True(DiscoverFilter.Matches(Model("a/fp8", formats: [ModelFormat.Safetensors], variants: [], method: "fp8"), none with { Quantization = "FP8" }, null));

        // Size ranges; a repository without a parameter count always passes.
        Assert.True(DiscoverFilter.Matches(open, none with { Size = ModelSizeFilter.From4To14B }, null));
        Assert.False(DiscoverFilter.Matches(open, none with { Size = ModelSizeFilter.From1To4B }, null));
        Assert.False(DiscoverFilter.Matches(open, none with { Size = ModelSizeFilter.Under1B }, null));
        Assert.True(DiscoverFilter.Matches(Model("a/unknown"), none with { Size = ModelSizeFilter.Over40B }, null));

        // Fits: the smallest likely download (a 4-bit GGUF is about 0.55 bytes per parameter) against the machine's memory.
        var big = Model("a/big", parameters: 70_000_000_000);
        Assert.False(DiscoverFilter.Matches(big, none with { FitsThisMachine = true }, 24 * gib));
        Assert.True(DiscoverFilter.Matches(big, none with { FitsThisMachine = true }, 64 * gib));
        Assert.True(DiscoverFilter.Matches(big, none with { FitsThisMachine = true }, null));
        Assert.True(DiscoverFilter.Matches(Model("a/unknown"), none with { FitsThisMachine = true }, gib));
        // A bf16-only checkpoint is two bytes per parameter, an fp8 one about one.
        var checkpoint = Model("a/checkpoint", parameters: 8_000_000_000, formats: [ModelFormat.Safetensors], variants: []);
        Assert.False(DiscoverFilter.Matches(checkpoint, none with { FitsThisMachine = true }, 12 * gib));
        Assert.True(DiscoverFilter.Matches(checkpoint with { QuantizationMethod = "fp8" }, none with { FitsThisMachine = true }, 12 * gib));
        Assert.Equal(DiscoverFilter.Capacity(24 * gib, 64 * gib), 64 * gib);
    }

    [Fact]
    public void ToQuery_SendsWhatTheHubFilters()
    {
        var query = DiscoverFilter.ToQuery(new DiscoverFilterSettings
        {
            Task = ModelTask.Embeddings, Format = ModelFormat.Onnx, License = "mit", Sort = HubSortOrder.Likes, Gated = GatedFilter.Hide, Quantization = "Q4",
        }, "bge", 30);
        Assert.Equal("bge", query.Text);
        Assert.Equal(ModelTask.Embeddings, query.Task);
        Assert.Equal(ModelFormat.Onnx, query.Format);
        Assert.Equal("mit", query.License);
        Assert.Equal(HubSortOrder.Likes, query.Sort);
        Assert.Equal(30, query.Limit);
    }

    [Fact]
    public async Task Search_UsesTheFilters_AndOnlyFiltersLocallyWhenTheHubQueryIsUnchanged()
    {
        using var services = new TestServices();
        services.Hub.Results.AddRange(
        [
            Model("a/small-gguf", parameters: 600_000_000, license: "mit", variants: ["Q4_K_M"]),
            Model("a/medium-gguf", parameters: 7_000_000_000, license: "mit", variants: ["Q8_0"]),
            Model("a/gated-gguf", parameters: 7_000_000_000, gated: true, variants: ["Q4_K_M"]),
        ]);
        var search = CreateSearch(services);
        search.Filters.SelectedFormat = search.Filters.Formats.Single(option => option.Value == ModelFormat.Gguf);
        search.Filters.SelectedSort = search.Filters.Sorts.Single(option => option.Value == HubSortOrder.Modified);
        search.Query = "gguf";
        await search.SearchCommand.ExecuteAsync(null);

        var query = services.Hub.Queries[^1];
        Assert.Equal("gguf", query.Text);
        Assert.Equal(ModelFormat.Gguf, query.Format);
        Assert.Equal(HubSortOrder.Modified, query.Sort);
        Assert.Equal(3, search.Results.Count);

        int searches = services.Hub.Queries.Count;
        search.Filters.SelectedQuantization = search.Filters.Quantizations.Single(option => option.Value == "Q4");
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Assert.Equal(searches, services.Hub.Queries.Count);
        Assert.Equal(["a/small-gguf", "a/gated-gguf"], search.Results.Select(result => result.Id));
        Assert.Equal("1 model does not match the filters.", search.HiddenResultsText);

        search.Filters.SelectedGated = search.Filters.GatedChoices.Single(option => option.Value == GatedFilter.Hide);
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Assert.Equal(["a/small-gguf"], search.Results.Select(result => result.Id));
        Assert.Equal("2 models do not match the filters.", search.HiddenResultsText);

        // A filter the Hub applies asks it again.
        search.Filters.SelectedLicense = search.Filters.Licenses.Single(option => option.Value == "mit");
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Assert.Equal("mit", services.Hub.Queries[^1].License);
        Assert.True(services.Hub.Queries.Count > searches);
    }

    [Fact]
    public async Task EmptyQueryWithAFilter_ListsTheHubInTheChosenOrder()
    {
        using var services = new TestServices();
        services.Hub.Results.Add(Model("a/listed"));
        var search = CreateSearch(services);
        Assert.True(search.IsShowingHighlights);

        search.Filters.SelectedSort = search.Filters.Sorts.Single(option => option.Value == HubSortOrder.Likes);
        await Task.Delay(50, TestContext.Current.CancellationToken);

        Assert.True(search.IsShowingResults);
        Assert.Equal(HubSortOrder.Likes, services.Hub.Queries[^1].Sort);
        Assert.Equal("", services.Hub.Queries[^1].Text);
        Assert.Equal(["a/listed"], search.Results.Select(result => result.Id));
    }

    [Fact]
    public async Task ChoosingAModelType_ShowsThatTypesHighlights_AndNamesTheKindsTheEngineCannotRun()
    {
        using var services = new TestServices();
        var diffuser = Model("stabilityai/sd-turbo", pipeline: "text-to-video", formats: [ModelFormat.Safetensors], variants: []);
        services.Hub.TaskHighlights[ModelTask.VideoGeneration] =
        [
            new HighlightSection(HighlightKind.Trending, [diffuser]),
            new HighlightSection(HighlightKind.New, []),
            new HighlightSection(HighlightKind.Popular, [diffuser with { Id = "stabilityai/sdxl" }]),
        ];
        var search = CreateSearch(services);
        await search.LoadHighlightsAsync();
        Assert.Equal([(ModelTask?)null], services.Hub.HighlightTasks);

        search.Filters.SelectedTask = search.Filters.Tasks.Single(option => option.Value == ModelTask.VideoGeneration);
        await Task.Delay(100, TestContext.Current.CancellationToken);
        await search.LoadHighlightsAsync();

        Assert.Equal([(ModelTask?)null, ModelTask.VideoGeneration], services.Hub.HighlightTasks);
        Assert.True(search.IsShowingHighlights);
        Assert.Equal("Video generation models", search.HighlightsHeading);
        Assert.Equal(["Trending", "Most downloaded"], search.Highlights.Select(section => section.Title));
        var badge = search.Highlights[0].Items[0].Compatibility;
        Assert.Equal("Not yet supported: Video generation", badge.Text);
        Assert.True(badge.IsBad);
        Assert.Contains("Video generation", search.Highlights[0].Items[0].KindTitle, StringComparison.Ordinal);

        // "Supported only" hides them, and says why the page is empty; turning it off brings them back.
        search.ShowOnlySupported = true;
        Assert.Empty(search.Highlights);
        Assert.Contains("cannot run video generation models yet", search.HighlightsStatus, StringComparison.Ordinal);
        search.ShowOnlySupported = false;
        Assert.Equal(2, search.Highlights.Count);
        Assert.Equal("", search.HighlightsStatus);
    }

    [Fact]
    public async Task ModelTypeSearch_ListsUnsupportedKinds_UnlessSupportedOnlyIsOn()
    {
        using var services = new TestServices();
        services.Hub.Results.AddRange(
        [
            new HubModelSummary("black-forest-labs/FLUX.1-schnell", 5000, 900, false, "diffusers", "text-to-video", null, [ModelFormat.Diffusers], []),
            Model("a/qwen-chat"),
        ]);
        var search = CreateSearch(services);
        search.Filters.SelectedTask = search.Filters.Tasks.Single(option => option.Value == ModelTask.VideoGeneration);
        search.Query = "flux";
        await search.SearchCommand.ExecuteAsync(null);

        Assert.Equal(ModelTask.VideoGeneration, services.Hub.Queries[^1].Task);
        // The kind is named, nothing is hidden silently, and the supported result comes first.
        Assert.Equal(["a/qwen-chat", "black-forest-labs/FLUX.1-schnell"], search.Results.Select(result => result.Id));
        var flux = search.Results[1];
        Assert.Equal("Not yet supported: Video generation", flux.Compatibility.Text);
        Assert.Equal("Video generation", flux.KindTitle);
        Assert.Equal("DIFFUSERS", flux.Formats);

        search.ShowOnlySupported = true;
        Assert.Equal(["a/qwen-chat"], search.Results.Select(result => result.Id));
        Assert.Equal("1 model is hidden because it is not supported.", search.HiddenResultsText);
    }

    [Fact]
    public async Task DiffusersRepository_IsShownAsOneModelWithItsComponents_AndCannotBeDownloaded()
    {
        using var services = new TestServices();
        var summary = new HubModelSummary("o/tiny-sd", 10, 1, false, "diffusers", "text-to-video", null, [ModelFormat.Diffusers], []);
        services.Hub.Results.Add(summary);
        services.Hub.Details = repository => new RepositoryDetails(new HubRepository(repository, "main", "abc", false, [], null), null, [], ModelFormat.Diffusers,
            [new HubComponent("unet", 3_400_000_000, 1), new HubComponent("text_encoder", 490_000_000, 1), new HubComponent("vae", 330_000_000, 1)]);
        var search = CreateSearch(services);
        search.Filters.SelectedTask = search.Filters.Tasks.Single(option => option.Value == ModelTask.VideoGeneration);
        search.Query = "tiny";
        await search.SearchCommand.ExecuteAsync(null);
        await search.FitEstimate;

        Assert.True(search.IsMultiComponent);
        Assert.Equal(["unet", "text_encoder", "vae"], search.Components.Select(component => component.Name));
        Assert.Contains("Diffusers · 3 parts", search.DetailFormat, StringComparison.Ordinal);
        Assert.Contains("Not yet supported: video generation", search.NoDownloadReason, StringComparison.Ordinal);
        Assert.Empty(search.Variants);
        Assert.False(search.DownloadCommand.CanExecute(null));
        // The kind verdict stays; it is not replaced by "no model files".
        Assert.Equal("Not yet supported: Video generation", search.SelectedResult!.Compatibility.Text);
    }

    [Fact]
    public async Task GatedRepository_SaysWhatToDo_DependingOnTheToken()
    {
        using var services = new TestServices();
        services.Hub.Results.Add(Model("meta/gated", gated: true));
        var search = CreateSearch(services);
        search.Query = "gated";
        await search.SearchCommand.ExecuteAsync(null);
        await search.FitEstimate;
        if (Environment.GetEnvironmentVariable("HF_TOKEN") is null)
        {
            Assert.Contains("add your access token in Settings", search.GatedNotice, StringComparison.Ordinal);
        }

        services.Settings.Update(settings => settings with { HuggingFaceToken = "hf_test" });
        search.SelectedResult = null;
        search.SelectedResult = search.Results[0];
        await search.FitEstimate;
        Assert.Contains("access token is used", search.GatedNotice, StringComparison.Ordinal);
    }

    [Fact]
    public void KindVerdict_NamesTheKind_AndLeavesLanguageModelsToTheArchitectureCheck()
    {
        var image = CompatibilityChecker.FromSummary(new HubModelSummary("a/sd", 1, 1, false, "diffusers", "text-to-video", null, [ModelFormat.Diffusers], []));
        Assert.Equal(CompatibilityState.UnsupportedKind, image.State);
        Assert.Equal("Not yet supported: Video generation", image.Text);
        Assert.True(image.IsUnsupported);

        // No pipeline tag: judged by the files, as before. Language and embedding tasks run, so they are checked by architecture.
        Assert.Equal(CompatibilityState.Supported, CompatibilityChecker.FromSummary(Model("a/untagged", pipeline: null)).State);
        Assert.Equal(CompatibilityState.Supported, CompatibilityChecker.FromSummary(Model("a/vlm", pipeline: "image-text-to-text")).State);
        Assert.Equal(CompatibilityState.Supported, CompatibilityChecker.FromSummary(Model("a/embedder", pipeline: "feature-extraction")).State);
        Assert.Equal(CompatibilityState.UnsupportedKind, CompatibilityChecker.FromSummary(Model("a/ocr", pipeline: "image-to-text")).State);
    }

    [Fact]
    public void KindCatalog_CoversEveryTask()
    {
        foreach (var task in ModelTasks.All)
        {
            var kind = ModelKindCatalog.For(task);
            Assert.Contains(task, kind.Tasks);
        }

        Assert.True(ModelKindCatalog.For(ModelKinds.TextGeneration).Runnable);
        Assert.False(ModelKindCatalog.For("video-generation").Runnable);
        // A kind nobody listed gets a readable generic entry.
        var unknown = ModelKindCatalog.For("point-cloud-segmentation");
        Assert.Equal("point cloud segmentation", unknown.Title);
        Assert.False(unknown.Runnable);
        Assert.Equal("Other", ModelKindCatalog.For((string?)null).Title);
    }
}
