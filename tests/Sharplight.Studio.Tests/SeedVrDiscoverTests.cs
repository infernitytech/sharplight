using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>
/// SeedVR2 in Discover: the verified entries (PyTorch folders of kind video-enhancement, the 7B with the text embeddings of the 3B
/// repository in its folder) are listed, found and installed, the Enhance kind opens their list, a Hub result of the official
/// repository shows its PyTorch format and downloads, and a published catalog of the same date replaces the embedded copy.
/// </summary>
public sealed class SeedVrDiscoverTests
{
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";
    private const string Sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    private static CatalogFile File(string path, long size) => new() { Path = path, Size = size, Sha256 = Sha };

    private static VerifiedCatalog Catalog(string updated = "2026-11-25")
    {
        var verification = new CatalogVerification { Summary = "official PyTorch inference", Providers = ["CPU", "Vulkan", "CUDA"], Date = "2026-11-25" };
        var embeddings = new CatalogCompanion
        {
            Id = "text-embeddings", Role = "text-encoder", Label = "SeedVR2 text embeddings", Repository = "ByteDance-Seed/SeedVR2-3B", Revision = Commit,
            Files = [File("pos_emb.pt", 595_100), File("neg_emb.pt", 656_540)], IntoVariantFolder = true,
        };
        return new VerifiedCatalog
        {
            Updated = updated,
            Entries =
            [
                new CatalogEntry
                {
                    Id = "seedvr2-3b", Name = "SeedVR2 3B", Family = "SeedVR2", Kind = "video-enhancement", Architecture = "seedvr", License = "apache-2.0",
                    Verification = verification,
                    Variants =
                    [
                        new CatalogVariant
                        {
                            Id = "official", Label = "Official checkpoint (F32)", Format = "pytorch", Recommended = true, Repository = "ByteDance-Seed/SeedVR2-3B",
                            Revision = Commit, MemoryBytes = 17L << 30,
                            Files = [File("seedvr2_ema_3b.pth", 13_566_090_228), File("ema_vae.pth", 1_002_691_902), File("pos_emb.pt", 595_100), File("neg_emb.pt", 656_540)],
                        },
                    ],
                },
                new CatalogEntry
                {
                    Id = "seedvr2-7b", Name = "SeedVR2 7B", Family = "SeedVR2", Kind = "video-enhancement", Architecture = "seedvr", License = "apache-2.0",
                    Verification = verification, Companions = [embeddings],
                    Variants =
                    [
                        new CatalogVariant
                        {
                            Id = "official", Label = "Official checkpoint (F32)", Format = "pytorch", Recommended = true, Repository = "ByteDance-Seed/SeedVR2-7B",
                            Revision = Commit, MemoryBytes = 30L << 30, Companions = ["text-embeddings"],
                            Files = [File("seedvr2_ema_7b.pth", 32_958_774_606), File("ema_vae.pth", 1_002_691_902)],
                        },
                    ],
                },
                new CatalogEntry
                {
                    Id = "qwen-small", Name = "Qwen small", Kind = "text-generation", Verification = verification,
                    Variants = [new CatalogVariant { Id = "q8-0", Label = "Q8_0", Format = "gguf", Repository = "owner/Qwen-GGUF", Revision = Commit, Primary = "m.gguf", Files = [File("m.gguf", 600)] }],
                },
            ],
        };
    }

    [Fact]
    public void Catalog_IsValid()
    {
        Assert.Empty(CatalogValidator.Validate(Catalog()));
    }

    [Fact]
    public async Task VerifiedList_ShowsSeedVr_AndInstallsThe7bWithItsEmbeddings()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Catalog();
        var downloads = new DownloadsViewModel(services.Services);
        var search = new HubSearchViewModel(services.Services, downloads, () => new HashSet<string>());
        var verified = search.Verified;
        await verified.LoadAsync();

        Assert.Contains("Video enhancement (2)", verified.Kinds.Select(kind => kind.Name));
        verified.Query = "seedvr";
        Assert.Equal(["SeedVR2 3B", "SeedVR2 7B"], verified.Entries.Select(entry => entry.Entry.Name));
        verified.SelectedEntry = verified.Entries.Single(entry => entry.Entry.Id == "seedvr2-7b");
        Assert.Equal(["SeedVR2 text embeddings · 1.2 MB"], verified.SelectedVariant!.Companions);

        await verified.InstallCommand.ExecuteAsync(null);
        await services.Hub.Started.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

        var item = Assert.Single(downloads.Items);
        Assert.Equal("video-enhancement", item.Kind);
        var plan = Assert.Single(services.Hub.Installs);
        Assert.True(Assert.Single(plan.Companions).IntoVariantFolder);
        Assert.Equal(32_958_774_606 + 1_002_691_902 + 595_100 + 656_540, plan.TotalBytes);
        services.Hub.Release();
    }

    [Fact]
    public async Task EnhanceKind_OpensTheVerifiedListOfItsKind()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Catalog();
        var search = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());

        search.ShowVerifiedKind("video-enhancement");
        await search.Verified.LoadAsync();

        Assert.True(search.IsVerifiedMode);
        Assert.Equal("video-enhancement", search.Verified.SelectedKind.Value);
        Assert.Equal(["SeedVR2 3B", "SeedVR2 7B"], search.Verified.Entries.Select(entry => entry.Entry.Name));
    }

    [Fact]
    public async Task HubResult_OfTheOfficialRepository_ShowsPyTorch_AndDownloads()
    {
        using var services = new TestServices();
        services.Hub.Catalog = Catalog();
        var repository = new HubRepository("ByteDance-Seed/SeedVR2-3B", "main", Commit, false,
        [
            new HubFile("README.md", 5_250, null, null), new HubFile("apex-0.1-cp39-cp39-linux_x86_64.whl", 4_839_299, Sha, null),
            new HubFile("ema_vae.pth", 1_002_691_902, Sha, null), new HubFile("neg_emb.pt", 656_540, Sha, null), new HubFile("pos_emb.pt", 595_100, Sha, null),
            new HubFile("seedvr2_ema_3b.pth", 13_566_090_228, Sha, null),
        ]);
        var selection = ModelFileSelector.Select(repository, null);
        services.Hub.Details = _ => new RepositoryDetails(repository, selection.Format,
            [new VariantInfo(selection.Tag, selection.TotalBytes, selection.Files.Count, false, true, selection.PrimaryPath)]);
        // The formats as HubJson.ParseSearch reads them from the official repository's file list (Sharplight.Hub.Tests.PyTorchSelectionTests).
        services.Hub.Results.Add(new HubModelSummary("ByteDance-Seed/SeedVR2-3B", 10, 1, false, "seedvr", "video-to-video", null, [ModelFormat.PyTorch], [], License: "apache-2.0"));
        var downloads = new DownloadsViewModel(services.Services);
        var search = new HubSearchViewModel(services.Services, downloads, () => new HashSet<string>());
        await search.Verified.LoadAsync();
        search.Query = "seedvr2";
        await search.SearchCommand.ExecuteAsync(null);
        await search.FitEstimate;

        var result = Assert.Single(search.Results);
        Assert.Equal("PYTORCH", result.Formats);
        Assert.Equal("PyTorch", search.DetailFormat);
        Assert.NotEmpty(search.VerifiedNotice);
        var variant = Assert.Single(search.Variants);
        Assert.Equal(4, variant.Info.FileCount);
        Assert.Equal(13_566_090_228 + 1_002_691_902 + 656_540 + 595_100, variant.Info.Bytes);

        await search.DownloadCommand.ExecuteAsync(null);
        var item = Assert.Single(downloads.Items);
        Assert.Equal("video-enhancement", item.Kind);
        await services.Hub.Started.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        services.Hub.Release();
    }

    [Fact]
    public async Task Source_TakesThePublishedCopy_OnTheSameDate()
    {
        var embedded = Catalog("2026-11-20") with { Entries = [.. Catalog().Entries.Where(entry => entry.Kind != "video-enhancement")] };
        var source = new VerifiedCatalogSource(_ => Task.FromResult(VerifiedCatalogSerializer.Serialize(Catalog("2026-11-20"))), null,
            () => VerifiedCatalogSerializer.Serialize(embedded));

        var catalog = await source.GetAsync(TestContext.Current.CancellationToken);

        Assert.Contains(catalog.Entries, entry => entry.Id == "seedvr2-7b");
    }
}
