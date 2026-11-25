using Sharplight.CatalogTool;
using Sharplight.Hub.Catalog;
using Sharplight.Package;

namespace Sharplight.Hub.Tests;

public sealed class CatalogInstallTests : IDisposable
{
    private const string VaeCommit = "fedcba9876543210fedcba9876543210fedcba98";

    private readonly FakeHub _hub = new();
    private readonly TempDirectory _directory = new();
    private readonly HttpClient _http;
    private readonly ModelStore _store;
    private readonly List<(string Primary, IReadOnlyList<KeyValuePair<string, string>> Auxiliary)> _conversions = [];

    public CatalogInstallTests()
    {
        _hub.Add("owner/Model-GGUF").WithSize("Model-Q4_K_M.gguf", 20_000).WithSize("Model-Q8_0.gguf", 40_000).With("README.md", "readme");
        _hub.Add("owner/Pipeline", commit: VaeCommit).WithSize("vae/diffusion_pytorch_model.safetensors", 3_000).With("vae/config.json", "{}")
            .WithSize("unet/diffusion_pytorch_model.safetensors", 9_000);
        _http = _hub.CreateClient();
        _store = new ModelStore(_directory.Path);
    }

    public void Dispose()
    {
        _http.Dispose();
        _hub.Dispose();
        _directory.Dispose();
    }

    [Fact]
    public async Task InstallDownloadsThePinnedFilesAndBuildsTheCompanionsIn()
    {
        var entry = await EntryAsync();
        var plan = CatalogInstallPlan.For(entry, entry.Variants[0]);

        var result = await CreatePuller().InstallAsync(plan, cancellationToken: TestContext.Current.CancellationToken);

        var manifest = result.Manifest;
        Assert.Equal("model/q8-0-gguf", manifest.Catalog);
        Assert.Equal("Q8_0", manifest.Tag);
        Assert.Equal(CatalogSamples.Commit, manifest.Commit);
        Assert.Equal(40_000 + 3_000 + 2, result.DownloadedBytes);
        Assert.True(File.Exists(manifest.PackagePath));
        Assert.Contains(manifest.SourceFiles, file => file.Path == "companions/vae/Pipeline/vae/diffusion_pytorch_model.safetensors");
        var (primary, auxiliary) = Assert.Single(_conversions);
        Assert.EndsWith(Path.Combine("Model-GGUF", "Model-Q8_0.gguf"), primary, StringComparison.Ordinal);
        var (role, path) = Assert.Single(auxiliary);
        Assert.Equal("vae", role);
        Assert.EndsWith(Path.Combine("vae", "Pipeline", "vae", "diffusion_pytorch_model.safetensors"), path, StringComparison.Ordinal);
        Assert.DoesNotContain(_hub.Requests, request => request.RequestUri!.AbsolutePath.StartsWith("/api/", StringComparison.Ordinal));
        Assert.True(plan.IsInstalledAs(manifest));
    }

    [Fact]
    public async Task SecondInstallOfTheSameVariantIsUpToDate()
    {
        var entry = await EntryAsync();
        var plan = CatalogInstallPlan.For(entry, entry.Variants[0]);
        var puller = CreatePuller();
        await puller.InstallAsync(plan, cancellationToken: TestContext.Current.CancellationToken);
        _hub.Requests.Clear();

        var again = await puller.InstallAsync(plan, cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(again.UpToDate);
        Assert.Empty(_hub.Requests);
        Assert.Single(_conversions);
    }

    [Fact]
    public async Task ChangedFileIsRejectedByItsChecksum()
    {
        var entry = await EntryAsync();
        var variant = entry.Variants[0] with { Files = [entry.Variants[0].Files[0] with { Sha256 = CatalogSamples.Sha }] };
        entry = entry with { Variants = [variant] };

        await Assert.ThrowsAsync<HubException>(() => CreatePuller().InstallAsync(CatalogInstallPlan.For(entry, variant), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Empty(_store.List());
        Assert.Empty(_conversions);
    }

    [Fact]
    public async Task GatedEntryWithoutTokenSaysWhatToDo()
    {
        var entry = await EntryAsync() with { Gated = true };

        var error = await Assert.ThrowsAsync<HubException>(() => CreatePuller().InstallAsync(CatalogInstallPlan.For(entry, entry.Variants[0]), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Contains("Accept its terms", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CompanionWithSeveralRolesIsDownloadedOnceAndPassedInEach()
    {
        var entry = await EntryAsync();
        entry = entry with { Companions = [entry.Companions[0] with { Role = "text-encoder", AdditionalRoles = ["vae"] }] };

        var result = await CreatePuller().InstallAsync(CatalogInstallPlan.For(entry, entry.Variants[0]), cancellationToken: TestContext.Current.CancellationToken);

        var (_, auxiliary) = Assert.Single(_conversions);
        Assert.Equal(["text-encoder", "vae"], auxiliary.Select(pair => pair.Key));
        Assert.Single(auxiliary.Select(pair => pair.Value).Distinct());
        Assert.Equal(2, result.Manifest.SourceFiles.Count(file => file.Path.StartsWith("companions/", StringComparison.Ordinal)));
        Assert.Empty(CatalogValidator.Validate(CatalogSamples.Catalog(entry)));
    }

    /// <summary>
    /// A PyTorch folder variant (SeedVR2 7B: the DiT and VAE of one repository, the text embeddings of another) is downloaded into one
    /// folder that the importer opens, recorded as a PyTorch model and converted without auxiliary roles.
    /// </summary>
    [Fact]
    public async Task PyTorchFolderVariant_WithFilesOfAnotherRepository_IsConvertedAsOneFolder()
    {
        _hub.Add("owner/SeedVR2-7B").WithSize("seedvr2_ema_7b.pth", 30_000).WithSize("seedvr2_ema_7b_sharp.pth", 30_000).WithSize("ema_vae.pth", 3_000).With("README.md", "readme");
        _hub.Add("owner/SeedVR2-3B", commit: VaeCommit).WithSize("seedvr2_ema_3b.pth", 12_000).WithSize("pos_emb.pt", 1_100).WithSize("neg_emb.pt", 1_200)
            .WithSize("apex-0.1-cp39-cp39-linux_x86_64.whl", 2_000);
        var variant = new CatalogVariant
        {
            Id = "official", Label = "Official checkpoint (F32)", Format = CatalogFormats.PyTorch, Repository = "owner/SeedVR2-7B",
            Include = ["seedvr2_ema_7b.pth", "ema_vae.pth"], Companions = ["text-embeddings"],
        };
        var companion = new CatalogCompanion
        {
            Id = "text-embeddings", Role = "text-encoder", Label = "Text embeddings", Repository = "owner/SeedVR2-3B", Include = ["pos_emb.pt", "neg_emb.pt"],
            IntoVariantFolder = true,
        };
        var catalog = CatalogSamples.Catalog(CatalogSamples.Entry("seedvr2-7b", variant) with { Kind = "video-enhancement", Companions = [companion] });
        var entry = (await new CatalogResolver(new HubClient(_http, _hub.Options()), _ => { }).ResolveAsync(catalog, ResolveMode.Missing, null, TestContext.Current.CancellationToken)).Entries[0];
        Assert.Empty(CatalogValidator.Validate(CatalogSamples.Catalog(entry)));
        var plan = CatalogInstallPlan.For(entry, entry.Variants[0]);
        List<string>? converted = null;
        IReadOnlyList<KeyValuePair<string, string>>? roles = null;
        var options = _hub.Options();
        var puller = new ModelPuller(new HubClient(_http, options), new FileDownloader(_http, options), _store, TestHost.Converter)
        {
            FreeSpace = _ => 1L << 40,
            CompositeConverter = (primary, auxiliary, destination, _) =>
            {
                converted = [.. Directory.GetFiles(primary).Select(path => Path.GetFileName(path)).Order(StringComparer.Ordinal)];
                roles = auxiliary;
                Assert.True(new Sharplight.Import.SeedVr.SeedVrImporter().CanOpen(primary));
                File.WriteAllText(destination, "package");
                return new ConversionResult(destination, "SeedVR2-7B", "seedvr", "seedvr", 4, 1, 7, TimeSpan.Zero, TimeSpan.Zero, 0, 0, "seedvr");
            },
        };

        var result = await puller.InstallAsync(plan, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(ModelFormat.PyTorch, result.Manifest.Format);
        Assert.Empty(roles!);
        Assert.Equal(["ema_vae.pth", "neg_emb.pt", "pos_emb.pt", "seedvr2_ema_7b.pth"], converted);
        Assert.Equal(["SeedVR2-7B/ema_vae.pth", "SeedVR2-7B/neg_emb.pt", "SeedVR2-7B/pos_emb.pt", "SeedVR2-7B/seedvr2_ema_7b.pth"],
            result.Manifest.SourceFiles.Select(file => file.Path).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void StoreTagFollowsThePullNaming()
    {
        var gguf = CatalogSamples.Variant() with { Quantization = "Q8_0" };
        var folder = CatalogSamples.Variant("bf16-safetensors") with { Format = CatalogFormats.Safetensors, Repository = "owner/Model" };
        var single = CatalogSamples.Variant("single-file") with { Format = CatalogFormats.SingleFile, Repository = "owner/Model" };
        var entry = CatalogSamples.Entry("model", gguf, folder, single);

        Assert.Equal("owner/Model-GGUF:Q8_0", CatalogInstallPlan.For(entry, gguf).Reference.ToString());
        Assert.Equal("owner/Model:bf16-safetensors", CatalogInstallPlan.For(entry, folder).Reference.ToString());
        Assert.Equal("owner/Model", CatalogInstallPlan.For(CatalogSamples.Entry("other", folder), folder).Reference.ToString());
        Assert.Throws<ArgumentException>(() => CatalogInstallPlan.For(CatalogSamples.Entry("x"), folder));
    }

    /// <summary>A Q8_0 entry with a VAE companion, resolved against the fake Hub (so sizes and checksums are the served ones).</summary>
    private async Task<CatalogEntry> EntryAsync()
    {
        var variant = new CatalogVariant
        {
            Id = "q8-0-gguf", Label = "Q8_0 GGUF", Format = "gguf", Quantization = "Q8_0", Repository = "owner/Model-GGUF", Tag = "Q8_0", Companions = ["vae"],
        };
        var companion = new CatalogCompanion
        {
            Id = "vae", Role = "vae", Label = "VAE", Repository = "owner/Pipeline", Include = ["vae/*"], Primary = "vae/diffusion_pytorch_model.safetensors",
        };
        var catalog = CatalogSamples.Catalog(CatalogSamples.Entry("model", variant) with { Companions = [companion] });
        var resolved = await new CatalogResolver(new HubClient(_http, _hub.Options()), _ => { }).ResolveAsync(catalog, ResolveMode.Missing, null, TestContext.Current.CancellationToken);
        _hub.Requests.Clear();
        return resolved.Entries[0];
    }

    private ModelPuller CreatePuller()
    {
        var options = _hub.Options();
        return new ModelPuller(new HubClient(_http, options), new FileDownloader(_http, options), _store, TestHost.Converter)
        {
            FreeSpace = _ => 1L << 40,
            CompositeConverter = FakeConvert,
        };
    }

    private ConversionResult FakeConvert(string primary, IReadOnlyList<KeyValuePair<string, string>> auxiliary, string destination, Action<string>? log)
    {
        Assert.True(File.Exists(primary));
        Assert.All(auxiliary, pair => Assert.True(File.Exists(pair.Value)));
        _conversions.Add((primary, auxiliary));
        File.WriteAllText(destination, "package");
        return new ConversionResult(destination, "Model", "decoder", "gguf", 1, 1, 7, TimeSpan.Zero, TimeSpan.Zero, 0, 0, "flux");
    }
}
