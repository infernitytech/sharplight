using Sharplight.Hub;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The highlighted models on Discover and the "Supported only" filter.</summary>
public sealed class HighlightsTests
{
    private static readonly DateTimeOffset s_now = new(2026, 11, 1, 12, 0, 0, TimeSpan.Zero);

    private static HubModelSummary Gguf(string id, long downloads, string architecture = "qwen3", int ageDays = 400, string? pipeline = "text-generation")
        => new(id, downloads, 1, false, "gguf", pipeline, null, [ModelFormat.Gguf], ["Q4_K_M"], GgufArchitecture: architecture,
            CreatedAt: s_now.AddDays(-ageDays));

    private static HubModelSummary Checkpoint(string id, long downloads, string modelType = "llama", string? quantization = null)
        => new(id, downloads, 1, false, "transformers", "text-generation", null, [ModelFormat.Safetensors], [], ModelType: modelType,
            QuantizationMethod: quantization, CreatedAt: s_now.AddDays(-400));

    [Fact]
    public void Build_KeepsSupportedLanguageModels_InOneSectionEach()
    {
        IReadOnlyList<HubModelSummary> trendingGguf =
        [
            Gguf("a/Trend-GGUF", 10),
            Gguf("a/Unsupported-GGUF", 9, architecture: "toy_unmapped"),
            Gguf("a/Fresh-GGUF", 8, ageDays: 3),
            Gguf("a/Embedding-GGUF", 7, pipeline: "feature-extraction"),
        ];
        IReadOnlyList<HubModelSummary> trendingCheckpoints = [Checkpoint("b/Trend", 10), Checkpoint("b/Aqlm", 9, quantization: "aqlm")];
        IReadOnlyList<HubModelSummary> downloaded = [Checkpoint("b/Popular", 5000), Gguf("a/Trend-GGUF", 10), Gguf("a/Older-New-GGUF", 4000, ageDays: 60)];

        var sections = HighlightsCatalog.Build([trendingGguf, trendingCheckpoints], [downloaded], s_now);

        Assert.Equal([HighlightKind.Trending, HighlightKind.New, HighlightKind.Popular], sections.Select(section => section.Kind));
        // Interleaved by format, unsupported architectures, quantizations and pipelines left out.
        Assert.Equal(["a/Trend-GGUF", "b/Trend", "a/Fresh-GGUF"], sections[0].Models.Select(model => model.Id));
        // Recent models that are not already trending, newest first.
        Assert.Equal(["a/Older-New-GGUF"], sections[1].Models.Select(model => model.Id));
        Assert.Equal(["b/Popular"], sections[2].Models.Select(model => model.Id));
        Assert.All(sections.SelectMany(section => section.Models), model => Assert.True(HighlightsCatalog.IsEligible(model)));
    }

    [Fact]
    public async Task Catalog_CachesInAFile_AndShowsTheOldAnswerWhileTheHubFails()
    {
        using var dir = new TempDir();
        string file = dir.File("highlights.json");
        var time = new FakeTime(s_now);
        int requests = 0;
        bool fail = false;
        Task<IReadOnlyList<HubModelSummary>> Search(HubSearchQuery query, CancellationToken cancellationToken)
        {
            requests++;
            Assert.Equal("", query.Text);
            return fail
                ? Task.FromException<IReadOnlyList<HubModelSummary>>(new HubException("offline"))
                : Task.FromResult<IReadOnlyList<HubModelSummary>>([Gguf($"a/{query.Sort}-{query.Format}", 10)]);
        }

        var catalog = new HighlightsCatalog(Search, file, time);
        var first = await catalog.GetAsync(CancellationToken.None);
        Assert.Equal(4, requests);
        Assert.True(File.Exists(file));
        Assert.Contains(first.SelectMany(section => section.Models), model => model.Id == "a/Trending-Gguf");

        // Within the lifetime nothing is asked, also by a new catalog over the same file (the next session).
        time.Now = s_now + HighlightsCatalog.CacheLifetime - TimeSpan.FromMinutes(1);
        await catalog.GetAsync(CancellationToken.None);
        var reopened = await new HighlightsCatalog(Search, file, time).GetAsync(CancellationToken.None);
        Assert.Equal(4, requests);
        Assert.Equal(first.SelectMany(section => section.Models).Select(model => model.Id), reopened.SelectMany(section => section.Models).Select(model => model.Id));

        // Expired and the Hub cannot be reached: the old answer is shown.
        time.Now = s_now + HighlightsCatalog.CacheLifetime + TimeSpan.FromMinutes(1);
        fail = true;
        var stale = await catalog.GetAsync(CancellationToken.None);
        Assert.Equal(8, requests);
        Assert.Equal(first.Count, stale.Count);

        // Nothing cached and the Hub cannot be reached: the error reaches the caller.
        await Assert.ThrowsAsync<HubException>(() => new HighlightsCatalog(Search, dir.File("other.json"), time).GetAsync(CancellationToken.None));
    }

    [Fact]
    public async Task Discover_ShowsHighlightsUntilSomethingIsSearched()
    {
        using var services = new TestServices();
        services.Hub.Highlights.AddRange(DemoData.Highlights);
        services.Hub.Results.AddRange(DemoData.CompatibilityResults);
        var search = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());

        Assert.True(search.IsShowingHighlights);
        await search.LoadHighlightsAsync();
        await search.LoadHighlightsAsync();
        Assert.Equal(1, services.Hub.HighlightRequests);
        Assert.Equal(["Trending", "New", "Most downloaded"], search.Highlights.Select(section => section.Title));

        // A card opens the repository's details and marks itself.
        var card = search.Highlights[1].Items[0];
        search.SelectResultCommand.Execute(card);
        Assert.Same(card, search.SelectedResult);
        Assert.True(card.IsSelected);
        await search.FitEstimate;
        Assert.Equal(3, search.Variants.Count);

        search.Query = "qwen";
        await search.SearchCommand.ExecuteAsync(null);
        Assert.False(search.IsShowingHighlights);
        Assert.True(search.IsShowingResults);
        Assert.False(card.IsSelected);
        Assert.Equal(4, search.Results.Count);

        // Clearing the query goes back to the highlights.
        search.Query = "  ";
        Assert.True(search.IsShowingHighlights);
        Assert.Null(search.SelectedResult);
    }

    [Fact]
    public async Task Discover_HighlightsThatFail_SayWhy_AndAreAskedAgain()
    {
        using var services = new TestServices();
        services.Hub.HighlightsError = new HubException("offline");
        var search = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());

        await search.LoadHighlightsAsync();
        Assert.Empty(search.Highlights);
        Assert.False(search.IsLoadingHighlights);
        Assert.Contains("could not be reached", search.HighlightsStatus, StringComparison.Ordinal);

        services.Hub.HighlightsError = null;
        services.Hub.Highlights.AddRange(DemoData.Highlights);
        await search.LoadHighlightsAsync();
        Assert.Equal(2, services.Hub.HighlightRequests);
        Assert.Equal(3, search.Highlights.Count);
        Assert.Equal("", search.HighlightsStatus);
    }

    [Fact]
    public async Task SupportedOnly_HidesResultsAndQuantizations_AndIsSaved()
    {
        using var services = new TestServices();
        services.Hub.Results.AddRange(DemoData.CompatibilityResults);
        services.Hub.Compatibility = (_, tag) => tag == "Q8_0"
            ? new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, "Q8_K", "1 tensor is stored as Q8_K, which has no compute kernels.")
            : new ModelCompatibility(CompatibilityState.Supported, "qwen3", "The GGUF architecture 'qwen3' is supported.");
        var search = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());
        search.Query = "qwen";
        await search.SearchCommand.ExecuteAsync(null);
        await search.FitEstimate;
        Assert.Equal(4, search.Results.Count);
        Assert.Equal("", search.HiddenResultsText);

        // Q8_0's badge is known once its tensor table is read.
        search.SelectedVariant = search.Variants.Single(variant => variant.Tag == "Q8_0");
        await search.CompatibilityCheck;
        Assert.True(search.SelectedVariant!.Compatibility!.IsBad);

        search.ShowOnlySupported = true;
        Assert.True(services.Settings.Current.ShowOnlySupported);
        Assert.Equal(["Qwen/Qwen3-4B-GGUF", "Qwen/Qwen3-4B"], search.Results.Select(result => result.Id));
        Assert.Equal("2 models are hidden because they are not supported.", search.HiddenResultsText);
        // Quantizations whose badge is not known yet stay; the hidden selection moves to the recommended one.
        Assert.Equal(["Q4_K_M", "Q5_K_M"], search.Variants.Select(variant => variant.Tag));
        Assert.Equal("1 quantization is hidden because it is not supported.", search.HiddenVariantsText);
        Assert.Equal("Q4_K_M", search.SelectedVariant!.Tag);

        // A quantization whose badge turns bad later leaves the list too.
        services.Hub.Compatibility = (_, _) => new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, "IQ1_M", "No kernels.");
        search.SelectedVariant = search.Variants.Single(variant => variant.Tag == "Q5_K_M");
        await search.CompatibilityCheck;
        Assert.Equal(["Q4_K_M"], search.Variants.Select(variant => variant.Tag));

        search.ShowOnlySupported = false;
        Assert.False(services.Settings.Current.ShowOnlySupported);
        Assert.Equal(4, search.Results.Count);
        Assert.Equal(3, search.Variants.Count);
        Assert.Equal("", search.HiddenVariantsText);

        // The choice is read back from the settings.
        services.Settings.Update(settings => settings with { ShowOnlySupported = true });
        var reopened = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());
        Assert.True(reopened.ShowOnlySupported);
    }
}
