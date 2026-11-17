using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>Discover's verified-model catalog: the mode switch, the per-task list, variants, fit, installs and the catalog source.</summary>
public sealed class VerifiedCatalogTests
{
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";
    private const string Sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    private static CatalogVariant Variant(string id, string repository, long bytes, long memory, bool recommended = false, params string[] companions) => new()
    {
        Id = id, Label = id.ToUpperInvariant(), Format = "gguf", Quantization = id.ToUpperInvariant(), Repository = repository, Revision = Commit,
        Primary = "model.gguf", Files = [new CatalogFile { Path = "model.gguf", Size = bytes, Sha256 = Sha }], MemoryBytes = memory, Recommended = recommended,
        Companions = companions,
    };

    private static CatalogEntry Entry(string id, string kind, params CatalogVariant[] variants) => new()
    {
        Id = id, Name = id, Kind = kind, Variants = variants, Verification = new CatalogVerification { Summary = "identical to llama.cpp", Providers = ["CPU", "Vulkan"], Date = "2026-11-16" },
    };

    private static VerifiedCatalog Sample(string updated = "2026-11-16")
    {
        var vae = new CatalogCompanion
        {
            Id = "vae", Role = "vae", Label = "FLUX VAE", Repository = "owner/Vae", Revision = Commit, Primary = "ae.safetensors",
            Files = [new CatalogFile { Path = "ae.safetensors", Size = 300, Sha256 = Sha }],
        };
        return new VerifiedCatalog
        {
            Updated = updated,
            Entries =
            [
                Entry("qwen-small", "text-generation", Variant("q8-0", "owner/Qwen-GGUF", 600, 1L << 30, recommended: true), Variant("bf16", "owner/Qwen", 1200, 2L << 30)),
                // Recommended variant too large for the 32 GB GPU and 64 GB system memory: the variant that fits is picked.
                Entry("big-model", "text-generation", Variant("q8-0", "owner/Big-GGUF", 900, 200L << 30, recommended: true), Variant("q2-k", "owner/Big-GGUF", 400, 20L << 30)),
                Entry("flux-like", "image-generation", Variant("q8-0", "owner/Flux-GGUF", 1000, 4L << 30, true, "vae")) with { Companions = [vae] },
                Entry("not-installable", "text-generation") with { Level = CatalogLevel.Loads },
            ],
        };
    }

    private static HubSearchViewModel Discover(TestServices services, Func<IReadOnlyList<ModelManifest>>? stored = null)
        => new(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>(), stored);

    [Fact]
    public async Task Discover_StartsWithTheVerifiedList_WhenTheCatalogHasModelsOfTheTask()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Sample();
        var search = Discover(services);

        await search.LoadHighlightsAsync();
        await search.Verified.LoadAsync();

        Assert.True(search.IsVerifiedMode);
        Assert.Equal(["qwen-small", "big-model", "flux-like"], search.Verified.Entries.Select(entry => entry.Name));
        Assert.Equal(["All tasks", "Text generation (2)", "Image generation (1)"], search.Verified.Kinds.Select(kind => kind.Name));
        Assert.Contains("3 models", search.Verified.CatalogText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Discover_FollowsTheModelType_UntilAModeIsPicked()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Sample();
        var search = Discover(services);
        search.Filters.SelectedTask = search.Filters.Tasks.First(option => option.Value == ModelTask.Robotics);
        await search.LoadHighlightsAsync();
        await search.Verified.LoadAsync();
        Assert.False(search.IsVerifiedMode);

        search.Filters.SelectedTask = search.Filters.Tasks.First(option => option.Value == ModelTask.ImageGeneration);
        Assert.True(search.IsVerifiedMode);
        Assert.Equal("image-generation", search.Verified.SelectedKind.Value);
        Assert.Equal(["flux-like"], search.Verified.Entries.Select(entry => entry.Name));

        search.ShowHubSearchCommand.Execute(null);
        search.Filters.SelectedTask = search.Filters.Tasks.First(option => option.Value == ModelTask.TextGeneration);
        Assert.True(search.IsHubMode);
        search.ShowVerifiedCommand.Execute(null);
        Assert.True(search.IsVerifiedMode);
    }

    [Fact]
    public async Task VerifiedList_FiltersBySearchTextAndFit_AndPicksAVariantThatFits()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Sample();
        services.Runtime.Memory = () => [new("gpu", "GPU", true, 32L << 30, 30L << 30, true), new("cpu", "CPU", false, 64L << 30, 40L << 30, true)];
        var verified = Discover(services).Verified;
        await verified.LoadAsync();

        verified.SelectedEntry = verified.Entries.Single(entry => entry.Name == "big-model");
        Assert.Equal("q2-k", verified.SelectedVariant!.Variant.Id);
        Assert.True(verified.SelectedVariant.IsGood);
        Assert.Equal("Too large", verified.SelectedEntry.Variants[0].FitText);

        verified.Query = "flux";
        Assert.Equal(["flux-like"], verified.Entries.Select(entry => entry.Name));
        verified.Query = "";
        verified.OnlyFitting = true;
        Assert.Equal(3, verified.Entries.Count);
    }

    [Fact]
    public async Task Install_QueuesTheVariantWithItsCompanions_AndMarksItInstalled()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Sample();
        var stored = new List<ModelManifest>();
        var downloads = new DownloadsViewModel(services.Services);
        var search = new HubSearchViewModel(services.Services, downloads, () => new HashSet<string>(), () => stored);
        var verified = search.Verified;
        await verified.LoadAsync();
        verified.SelectedEntry = verified.Entries.Single(entry => entry.Name == "flux-like");
        Assert.Equal(["FLUX VAE · 300 B"], verified.SelectedVariant!.Companions);

        verified.InstallCommand.Execute(null);
        await services.Hub.Started.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        var item = Assert.Single(downloads.Items);
        Assert.Equal("flux-like/q8-0", item.CatalogKey);
        Assert.Equal("flux-like · Q8-0", item.Name);
        var plan = Assert.Single(services.Hub.Installs);
        Assert.Equal(1300, plan.TotalBytes);
        Assert.Equal("vae", Assert.Single(plan.Companions).Role);
        Assert.Equal("owner/Flux-GGUF:Q8-0", plan.Reference.ToString());

        stored.Add(new ModelManifest { Repository = "owner/Flux-GGUF", Tag = "Q8-0", Alias = "flux", Commit = Commit, Catalog = "flux-like/q8-0" });
        verified.RefreshInstalled();
        Assert.True(verified.SelectedEntry.IsInstalled);
        Assert.True(verified.SelectedVariant.IsInstalled);
        Assert.StartsWith("Installed", verified.InstallText, StringComparison.Ordinal);
        services.Hub.Release();
    }

    [Fact]
    public async Task Source_UsesANewerValidDownload_AndKeepsItForOfflineStarts()
    {
        using var dir = new TempDir();
        var cache = dir.File("catalog.json");
        var time = new FakeTime(new DateTimeOffset(2026, 11, 17, 8, 0, 0, TimeSpan.Zero));
        string Embedded() => VerifiedCatalogSerializer.Serialize(Sample("2026-11-16"));
        int downloads = 0;
        string remote = VerifiedCatalogSerializer.Serialize(Sample("2026-11-20"));

        Task<string> Download(CancellationToken cancellationToken)
        {
            downloads++;
            return Task.FromResult(remote);
        }

        var source = new VerifiedCatalogSource(Download, cache, Embedded, time);
        Assert.Equal("2026-11-20", (await source.GetAsync(TestContext.Current.CancellationToken)).Updated);
        Assert.Equal("2026-11-20", (await source.GetAsync(TestContext.Current.CancellationToken)).Updated);
        Assert.Equal(1, downloads);

        var offline = new VerifiedCatalogSource(_ => Task.FromException<string>(new HttpRequestException("offline")), cache, Embedded, time);
        Assert.Equal("2026-11-20", (await offline.GetAsync(TestContext.Current.CancellationToken)).Updated);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("""{"schemaVersion": 9, "updated": "2027-01-01", "entries": []}""")]
    [InlineData("""{"schemaVersion": 1, "updated": "2027-01-01", "entries": [{"id": "BAD ID", "name": "x", "kind": "text-generation"}]}""")]
    public async Task Source_IgnoresInvalidDownloads(string remote)
    {
        var source = new VerifiedCatalogSource(_ => Task.FromResult(remote), null, () => VerifiedCatalogSerializer.Serialize(Sample("2026-11-16")));

        var catalog = await source.GetAsync(TestContext.Current.CancellationToken);

        Assert.Equal("2026-11-16", catalog.Updated);
        Assert.Equal(4, catalog.Entries.Count);
    }

    [Fact]
    public async Task Source_KeepsTheEmbeddedCopy_WhenTheDownloadIsOlder()
    {
        var source = new VerifiedCatalogSource(_ => Task.FromResult(VerifiedCatalogSerializer.Serialize(Sample("2026-01-01"))), null,
            () => VerifiedCatalogSerializer.Serialize(Sample("2026-11-16")));

        Assert.Equal("2026-11-16", (await source.GetAsync(TestContext.Current.CancellationToken)).Updated);
    }

    [Fact]
    public async Task EmbeddedCatalog_IsTheCommittedOne()
    {
        var source = new VerifiedCatalogSource(null, null);

        var catalog = await source.GetAsync(TestContext.Current.CancellationToken);

        Assert.Empty(CatalogValidator.Validate(catalog));
        Assert.True(catalog.Entries.Count(entry => entry.IsInstallable) > 100);
    }
}
