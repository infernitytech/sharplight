using Sharplight.ModelFeed.Publishing;

namespace Sharplight.ModelFeed.Tests;

public sealed class ModelGroupTests
{
    private static Func<PackageManifest, PackageManifest> Variant(string group, string variant, double bits, bool recommended = false, string? quality = null, string level = "verified") =>
        manifest => manifest with
        {
            Group = new PackageGroup { Id = group, Name = "Model " + group, Variant = variant, Label = variant.ToUpperInvariant(), Format = "gguf", Quantization = variant.ToUpperInvariant(), BitsPerWeight = bits, Recommended = recommended, Quality = quality, Reference = "BF16", Size = manifest.PrimaryFile!.Size },
            Verification = new PackageVerification { Level = level, Reference = "llama.cpp", PackageCheck = new PackageCheck { Loaded = true } },
            Requirements = manifest.Requirements! with
            {
                Memory = [new MemoryRequirement { Device = "gpu", Weights = 1000, Overhead = 100, KvBytesPerToken = 2, StateBytesPerSlot = 50, PerSlotContext = new SlotContextMemory { Tokens = 10, Bytes = 70 } }],
                Kernels = [new PackageKernel { WeightType = "q4_k", Bytes = 1000, NativeFeatures = ["matrix_f16"] }],
            },
        };

    [Fact]
    public async Task ModelListing_GroupsVariants_AndIsChainedIntoTheSignedIndex()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(
            await feed.StageAsync("big.q8", "1.0.0", customize: Variant("big", "q8", 8.5, quality: PackageQualityTiers.NearLossless)),
            await feed.StageAsync("big.q4", "1.0.0", customize: Variant("big", "q4", 4.8, recommended: true, quality: PackageQualityTiers.Good)),
            await feed.StageAsync("big.iq2", "1.0.0", customize: Variant("big", "iq2", 2.4, quality: PackageQualityTiers.Low, level: "partial")),
            await feed.StageAsync("solo", "1.0.0"));

        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        Assert.Equal(2, snapshot.Index.Catalog.Models);
        Assert.NotNull(snapshot.Index.Catalog.ModelsDigest);
        Assert.Equal("catalog/models.json", snapshot.Index.Resources.Models);
        Assert.Equal("big", snapshot.Find("big.q4")!.Group!.Id);
        Assert.Equal("partial", snapshot.Find("big.iq2")!.Verification!.Level);
        Assert.Null(snapshot.Find("big.iq2")!.Verification!.PackageCheck);

        var listing = await client.GetModelListingAsync(snapshot, TestContext.Current.CancellationToken);
        Assert.Equal(["big", "solo"], listing.Models.Select(model => model.Id));
        var big = listing.Models[0];
        Assert.Equal("Model big", big.Name);
        Assert.Equal("big.q4", big.Recommended);
        Assert.Equal(["big.iq2", "big.q4", "big.q8"], big.Variants.Select(variant => variant.Package));
        Assert.Equal([PackageQualityTiers.Low, PackageQualityTiers.Good, PackageQualityTiers.NearLossless], big.Variants.Select(variant => variant.Quality));
        Assert.Equal("partial", big.Variants[0].Verification);
        Assert.Equal(2, big.Variants[1].Requirements!.Memory[0].KvBytesPerToken);
        Assert.Equal("q4_k", big.Variants[1].Requirements!.Kernels[0].WeightType);
        Assert.Equal(["solo"], listing.Models[1].Variants.Select(variant => variant.Package));
        Assert.Null(listing.Models[1].Recommended);
        Assert.Equal(snapshot.Models, listing, ListingComparer.Instance);

        Assert.Empty(await FeedVerifier.VerifyAsync(feed.Client(), packageFiles: false, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ModelListing_IsVerifiedAgainstItsDigest()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("big.q4", "1.0.0", customize: Variant("big", "q4", 4.8)));
        var client = feed.Client();
        var snapshot = await client.GetSnapshotAsync(TestContext.Current.CancellationToken);
        var path = feed.PathOf("catalog/models.json");
        await File.WriteAllTextAsync(path, (await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken)).Replace("big.q4", "big.q5", StringComparison.Ordinal), TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<FeedVerificationException>(() => client.GetModelListingAsync(snapshot, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Publish_RejectsAGroupThatDoesNotNameThePackageId()
    {
        using var feed = new TestFeed();
        var wrong = await feed.StageAsync("big.q4", "1.0.0", customize: Variant("big", "q5", 4.8));
        await Assert.ThrowsAsync<FeedFormatException>(() => feed.PublishAsync(wrong));
        var tier = await feed.StageAsync("big.q8", "1.0.0", customize: Variant("big", "q8", 8.5, quality: "excellent"));
        await Assert.ThrowsAsync<FeedFormatException>(() => feed.PublishAsync(tier));
    }

    [Fact]
    public async Task Verifier_ReportsTwoRecommendedVariants()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(
            await feed.StageAsync("big.q4", "1.0.0", customize: Variant("big", "q4", 4.8, recommended: true)),
            await feed.StageAsync("big.q8", "1.0.0", customize: Variant("big", "q8", 8.5, recommended: true)));
        var problem = Assert.Single(await FeedVerifier.VerifyAsync(feed.Client(), packageFiles: false, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("recommended", problem, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ManifestWithoutGroup_IsGroupedByItsCatalogProvenance()
    {
        using var feed = new TestFeed();
        await feed.PublishAsync(await feed.StageAsync("qwen3-0.6b.q8-0-gguf", "1.0.0", customize: manifest => manifest with
        {
            Display = new PackageDisplay { Name = "Qwen3 0.6B", Variant = "Q8_0 GGUF" },
            Provenance = new PackageProvenance { CatalogEntry = "qwen3-0.6b", CatalogVariant = "q8-0-gguf" },
        }));
        var snapshot = await feed.Client().GetSnapshotAsync(TestContext.Current.CancellationToken);
        var group = snapshot.Find("qwen3-0.6b.q8-0-gguf")!.Group!;
        Assert.Equal("qwen3-0.6b", group.Id);
        Assert.Equal("q8-0-gguf", group.Variant);
        Assert.Equal("Q8_0 GGUF", group.Label);
        Assert.Equal("qwen3-0.6b", Assert.Single(snapshot.Models.Models).Id);
    }

    [Fact]
    public void MemoryRequirement_SizesSlotsAtAnyContext()
    {
        var exact = new MemoryRequirement { Device = "gpu", Weights = 1000, Overhead = 100, KvBytesPerToken = 4, StateBytesPerSlot = 10, PerSlotContext = new SlotContextMemory { Tokens = 100, Bytes = 410 } };
        Assert.Equal(1000 + 100 + (2 * (10 + (4 * 1000))), exact.TotalFor(2, 1000));
        Assert.Equal(1000 + 100 + 410, exact.TotalFor(1));
        var scaled = new MemoryRequirement { Device = "gpu", Weights = 1000, PerSlotContext = new SlotContextMemory { Tokens = 100, Bytes = 400 } };
        Assert.Equal(1000 + 800, scaled.TotalFor(1, 200));
        Assert.Equal(1000, new MemoryRequirement { Device = "cpu", Weights = 1000 }.TotalFor(4, 8192));
    }
}
