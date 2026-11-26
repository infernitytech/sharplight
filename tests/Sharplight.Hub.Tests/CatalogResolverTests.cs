using Sharplight.CatalogTool;
using Sharplight.Hub.Catalog;

namespace Sharplight.Hub.Tests;

public sealed class CatalogResolverTests : IDisposable
{
    private readonly FakeHub _hub = new();
    private readonly HttpClient _http;

    public CatalogResolverTests()
    {
        _hub.Add("owner/Model-GGUF").WithSize("Model-Q4_K_M.gguf", 20_000).WithSize("Model-Q8_0.gguf", 40_000).With("README.md", "readme");
        _hub.Add("owner/Vae", commit: "fedcba9876543210fedcba9876543210fedcba98").WithSize("vae/diffusion_pytorch_model.safetensors", 3_000).With("vae/config.json", "{}")
            .WithSize("unet/diffusion_pytorch_model.safetensors", 9_000);
        _http = _hub.CreateClient();
    }

    public void Dispose()
    {
        _http.Dispose();
        _hub.Dispose();
    }

    [Fact]
    public async Task ResolvesTagsAndPatternsToPinnedFiles()
    {
        var variant = new CatalogVariant { Id = "q8-0-gguf", Label = "Q8_0 GGUF", Format = "gguf", Repository = "owner/Model-GGUF", Tag = "Q8_0", Companions = ["vae"] };
        var companion = new CatalogCompanion { Id = "vae", Role = "vae", Label = "VAE", Repository = "owner/Vae", Include = ["vae/*"], Primary = "vae/diffusion_pytorch_model.safetensors" };
        var catalog = CatalogSamples.Catalog(CatalogSamples.Entry("model", variant) with { Companions = [companion] });

        var resolver = new CatalogResolver(new HubClient(_http, _hub.Options()), _ => { });
        var resolved = await resolver.ResolveAsync(catalog, ResolveMode.Missing, null, TestContext.Current.CancellationToken);

        var entry = Assert.Single(resolved.Entries);
        var pinned = Assert.Single(entry.Variants);
        Assert.Empty(resolver.Failures);
        Assert.Equal("0123456789abcdef0123456789abcdef01234567", pinned.Revision);
        Assert.Equal("Model-Q8_0.gguf", pinned.Primary);
        Assert.Equal(40_000, Assert.Single(pinned.Files).Size);
        Assert.Matches("^[0-9a-f]{64}$", pinned.Files[0].Sha256!);
        Assert.Equal(["vae/config.json", "vae/diffusion_pytorch_model.safetensors"], entry.Companions[0].Files.Select(file => file.Path).Order(StringComparer.Ordinal));
        Assert.Equal("fedcba9876543210fedcba9876543210fedcba98", entry.Companions[0].Revision);
        Assert.True(pinned.MemoryBytes > 43_000);
        Assert.Empty(CatalogValidator.Validate(resolved));
    }

    [Fact]
    public async Task RepackagingTakesTheLicenseOfTheOriginalModel()
    {
        _hub.Add("owner/Base", commit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").With("config.json", "{}").License = "apache-2.0";
        var repack = _hub.Add("repackager/Model-GGUF", commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").WithSize("Model-Q8_0.gguf", 40_000);
        repack.License = "creativeml-openrail-m";
        repack.BaseModel = "owner/Base";

        var resolved = await ResolveAsync(CatalogSamples.Entry("model", Variant("repackager/Model-GGUF")));

        Assert.Equal("apache-2.0", Assert.Single(resolved.Entries).License);
    }

    [Fact]
    public async Task RepackagingFallsBackToItsOwnLicenseWhenTheOriginalIsUnreadable()
    {
        var repack = _hub.Add("repackager/Model-GGUF", commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").WithSize("Model-Q8_0.gguf", 40_000);
        repack.License = "mit";
        repack.BaseModel = "owner/Gone";

        var resolved = await ResolveAsync(CatalogSamples.Entry("model", Variant("repackager/Model-GGUF")));

        Assert.Equal("mit", Assert.Single(resolved.Entries).License);
    }

    [Fact]
    public async Task ARepositoryThatIsNotARepackagingKeepsItsOwnLicense()
    {
        _hub.Add("owner/Own", commit: "cccccccccccccccccccccccccccccccccccccccc").WithSize("Model-Q8_0.gguf", 40_000).License = "mit";

        var resolved = await ResolveAsync(CatalogSamples.Entry("model", Variant("owner/Own")));

        Assert.Equal("mit", Assert.Single(resolved.Entries).License);
    }

    [Theory]
    [InlineData("Missing")]
    [InlineData("Refresh")]
    [InlineData("Update")]
    public async Task ALicenseSetByHandIsNeverOverwritten(string modeName)
    {
        _hub.Add("owner/Base", commit: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").With("config.json", "{}").License = "other";
        var repack = _hub.Add("repackager/Model-GGUF", commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").WithSize("Model-Q8_0.gguf", 40_000);
        repack.License = "creativeml-openrail-m";
        repack.BaseModel = "owner/Base";

        var resolved = await ResolveAsync(CatalogSamples.Entry("model", Variant("repackager/Model-GGUF")) with { License = "apache-2.0" }, Enum.Parse<ResolveMode>(modeName));

        Assert.Equal("apache-2.0", Assert.Single(resolved.Entries).License);
    }

    [Fact]
    public async Task AGatedRepositoryWithMaskedLfsOidsIsPinnedWithTheChecksumsOfItsBlobListing()
    {
        var gated = _hub.Add("publisher/Gated", commit: "dddddddddddddddddddddddddddddddddddddddd", gated: true).WithSize("Model-Q8_0.gguf", 40_000).With("config.json", "{}");
        gated.MaskLfsOids = true;
        var variant = Variant("publisher/Gated") with { Include = ["*.gguf", "config.json"] };

        var resolved = await ResolveAsync(CatalogSamples.Entry("model", variant));

        var entry = Assert.Single(resolved.Entries);
        var files = entry.Variants[0].Files.ToDictionary(file => file.Path);
        Assert.Equal(Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(gated.Files["Model-Q8_0.gguf"])), files["Model-Q8_0.gguf"].Sha256);
        Assert.Null(files["Model-Q8_0.gguf"].GitOid);
        Assert.NotNull(files["config.json"].GitOid);
        Assert.True(entry.Gated);
        Assert.Empty(CatalogValidator.Validate(resolved));
    }

    private static CatalogVariant Variant(string repository)
        => new() { Id = "q8-0-gguf", Label = "Q8_0 GGUF", Format = "gguf", Repository = repository, Include = ["*.gguf"] };

    private async Task<VerifiedCatalog> ResolveAsync(CatalogEntry entry, ResolveMode mode = ResolveMode.Missing)
    {
        var resolver = new CatalogResolver(new HubClient(_http, _hub.Options()), _ => { });
        return await resolver.ResolveAsync(CatalogSamples.Catalog(entry), mode, null, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task UnmatchedPatternIsReportedAndLeavesTheSetUnresolved()
    {
        var variant = new CatalogVariant { Id = "x", Label = "X", Format = "gguf", Repository = "owner/Model-GGUF", Include = ["*.missing"] };

        var resolver = new CatalogResolver(new HubClient(_http, _hub.Options()), _ => { });
        var resolved = await resolver.ResolveAsync(CatalogSamples.Catalog(CatalogSamples.Entry("model", variant)), ResolveMode.Missing, null, TestContext.Current.CancellationToken);

        Assert.Contains("no file matches '*.missing'", Assert.Single(resolver.Failures), StringComparison.Ordinal);
        Assert.Empty(Assert.Single(resolved.Entries).Variants[0].Files);
    }

    [Theory]
    [InlineData("*.gguf", "a.gguf", true)]
    [InlineData("*.gguf", "dir/a.gguf", false)]
    [InlineData("**/*.gguf", "dir/sub/a.gguf", true)]
    [InlineData("**/*.gguf", "a.gguf", true)]
    [InlineData("vae/*", "vae/config.json", true)]
    [InlineData("model-0000?-of-00002.safetensors", "model-00001-of-00002.safetensors", true)]
    [InlineData("model.safetensors", "model.fp16.safetensors", false)]
    public void GlobPatternsMatchRepositoryPaths(string pattern, string path, bool expected)
    {
        Assert.Equal(expected, GlobPattern.IsMatch(pattern, path));
    }
}
