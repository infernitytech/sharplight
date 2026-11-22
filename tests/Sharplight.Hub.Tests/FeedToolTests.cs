using System.Text.Json;
using Sharplight.CatalogTool;
using Sharplight.FeedTool;
using Sharplight.Graph;
using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Signing;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Hub.Tests;

public sealed class FeedToolTests
{
    private static readonly string s_root = CatalogPaths.FindRepositoryRoot(AppContext.BaseDirectory);

    private static VerifiedCatalog Catalog() => VerifiedCatalogSerializer.Parse(File.ReadAllText(Path.Combine(s_root, CatalogPaths.Catalog)));

    private static FeedOverlay Overlay() =>
        JsonSerializer.Deserialize(File.ReadAllBytes(Path.Combine(s_root, FeedToolPaths.Overlay)), FeedToolJsonContext.Default.FeedOverlay)!;

    [Fact]
    public void Overlay_NamesOnlyCatalogPackages()
    {
        var catalog = Catalog();
        var ids = catalog.Entries.Where(entry => entry.IsInstallable)
            .SelectMany(entry => entry.Variants.Select(variant => ManifestFactory.PackageId(entry, variant))).ToHashSet(StringComparer.Ordinal);
        var overlay = Overlay();
        Assert.All(overlay.Packages.Keys, id => Assert.Contains(id, ids));
        Assert.All(overlay.Packages.Values.SelectMany(package => package.Dependencies), dependency => Assert.Contains(dependency.Id, ids));
        var entries = catalog.Entries.Where(entry => entry.IsInstallable).Select(entry => entry.Id).ToHashSet(StringComparer.Ordinal);
        Assert.All(overlay.Models.Keys, id => Assert.Contains(id, entries));
    }

    [Fact]
    public void Overlay_PackageEntriesOverrideTheirModelDefaults()
    {
        var overlay = new FeedOverlay
        {
            Models = new Dictionary<string, FeedOverlayPackage>
            {
                ["m"] = new() { Summary = "Model.", Languages = ["en", "de"], Dependencies = [new PackageDependency { Id = "vad", Range = "1.0.0" }] },
            },
            Packages = new Dictionary<string, FeedOverlayPackage>
            {
                ["m.q8"] = new() { Summary = "Q8.", Listed = false, Dependencies = [new PackageDependency { Id = "draft", Range = "1.0.0" }] },
            },
        };

        var q8 = overlay.For("m", "m.q8")!;
        Assert.Equal("Q8.", q8.Summary);
        Assert.Equal(["en", "de"], q8.Languages);
        Assert.Equal(["vad", "draft"], q8.Dependencies.Select(dependency => dependency.Id));
        Assert.False(q8.Listed);
        Assert.Equal("Model.", overlay.For("m", "m.q4")!.Summary);
        Assert.Null(overlay.For("other", "other.q4"));
    }

    [Fact]
    public void Manifest_ComesFromCatalogEntryAndOverlay()
    {
        var entry = Catalog().Entries.Single(candidate => candidate.Id == "whisper-tiny");
        var variant = entry.Variants.Single(candidate => candidate.Id == "q5-1-whispercpp");
        var plan = CatalogInstallPlan.For(entry, variant);
        var file = new PackageFile { Name = "whisper-tiny.q5-1-whispercpp.1.0.0.slm", Role = PackageFileRoles.Package, Size = 1000, Sha256 = new string('a', 64) };
        var inspection = new PackageInspection("Whisper tiny", "speech-recognition", "whisper", 448, 39_000_000, 30_000_000, 3, false);
        var manifest = ManifestFactory.Create(plan, inspection, file, "1.0.0", Overlay().Packages["whisper-tiny.q5-1-whispercpp"], "0.4.2-alpha", null, DateTimeOffset.UnixEpoch);

        Assert.Equal("whisper-tiny.q5-1-whispercpp", manifest.Id);
        Assert.Equal([PackageTasks.Transcription], manifest.Model!.Tasks);
        Assert.Equal("OpenAI", manifest.Publisher?.Name);
        Assert.Equal("US", manifest.Publisher?.CountryOfOrigin);
        Assert.Equal("mit", manifest.License?.Spdx);
        Assert.True(manifest.License?.CommercialUse);
        Assert.Equal(">=0.4.0", manifest.Requirements!.Engine);
        Assert.Equal(variant.Revision, manifest.Provenance!.Sources[0].Revision);
        Assert.Equal(variant.Files[0].Sha256, manifest.Provenance.Sources[0].Files[0].Sha256);
        var dependency = Assert.Single(manifest.Dependencies);
        Assert.Equal("silero-vad-v5.onnx", dependency.Id);
        Assert.True(dependency.IsRequired);
        Assert.Equal("verified", manifest.Verification!.Level);

        // The manifest round-trips through the feed reader.
        var parsed = FeedJson.ReadManifest(FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest));
        Assert.Equal(manifest.Id, parsed.Id);
    }

    [Fact]
    public void Manifest_PlacesTheVariantInItsModelGroup()
    {
        var entry = Catalog().Entries.Single(candidate => candidate.Id == "qwen3.8-27b");
        var variant = entry.Variants.Single(candidate => candidate.Id == "ud-iq3-xxs-gguf");
        var plan = CatalogInstallPlan.For(entry, variant);
        var file = new PackageFile { Name = "qwen3.8-27b.ud-iq3-xxs-gguf.1.0.0.slm", Role = PackageFileRoles.Package, Size = 17_000_000_000, Sha256 = new string('a', 64) };
        var context = new ContextMemory(65_536, 150_000_000, 4096, 418_435_456, 262_144, 17_329_717_248, 900_000_000);
        var inspection = new PackageInspection("Qwen3.8 27B", "text-generation", "qwen3_5", 262_144, 27_000_000_000, 16_500_000_000, 3, true)
        {
            WeightTypes = [KeyValuePair.Create("q4_k", 12_000_000_000L), KeyValuePair.Create("q6_k", 4_400_000_000L), KeyValuePair.Create("f32", 100_000_000L), KeyValuePair.Create("i32", 10L)],
            Context = context,
        };
        var manifest = ManifestFactory.Create(plan, inspection, file, "1.0.0", null, "0.4.2-alpha", null, DateTimeOffset.UnixEpoch);

        var group = manifest.Group!;
        Assert.Equal("qwen3.8-27b", group.Id);
        Assert.Equal("ud-iq3-xxs-gguf", group.Variant);
        Assert.Equal($"{group.Id}.{group.Variant}", manifest.Id);
        Assert.Equal("UD-IQ3_XXS", group.Quantization);
        Assert.Equal("gguf", group.Format);
        Assert.Equal("reduced", group.Quality);
        Assert.Equal("BF16", group.Reference);
        Assert.Equal(4.89, group.BitsPerWeight);
        Assert.False(group.Recommended);
        Assert.Equal(file.Size, group.Size);
        Assert.Equal(variant.TotalBytes, group.SourceSize);
        Assert.Equal("partial", manifest.Verification!.Level);
        Assert.StartsWith(variant.Verification!, manifest.Verification.Summary, StringComparison.Ordinal);

        Assert.Equal(["q4_k", "q6_k", "f32"], manifest.Requirements!.Kernels.Select(kernel => kernel.WeightType));
        Assert.All(manifest.Requirements.Kernels, kernel => Assert.True(kernel.Fallback));
        var gpu = manifest.Requirements.Memory.Single(memory => memory.Device == "gpu");
        Assert.Equal(16_500_000_000, gpu.Weights);
        Assert.Equal(65_536, gpu.KvBytesPerToken);
        Assert.Equal(150_000_000, gpu.StateBytesPerSlot);
        Assert.Equal(4096, gpu.PerSlotContext!.Tokens);
        Assert.Equal(262_144, gpu.PerSlotContextMax!.Tokens);
        Assert.True(gpu.Overhead >= context.ActivationBytes);
        Assert.Equal(gpu.Weights + gpu.Overhead + (2 * (150_000_000 + (65_536L * 32_768))), gpu.TotalFor(2, 32_768));
    }

    [Fact]
    public void Manifest_ListsTheRoutedExpertsOfAMixtureOfExpertsModel()
    {
        var entry = Catalog().Entries.Single(candidate => candidate.Id == "qwen3.8-27b");
        var variant = entry.Variants.Single(candidate => candidate.Id == "ud-iq3-xxs-gguf");
        var plan = CatalogInstallPlan.For(entry, variant);
        var file = new PackageFile { Name = "moe.slm", Role = PackageFileRoles.Package, Size = 17_000_000_000, Sha256 = new string('a', 64) };
        var inspection = new PackageInspection("MoE", "text-generation", "qwen3moe", 32_768, 30_000_000_000, 16_500_000_000, 3, true)
        {
            Context = new ContextMemory(65_536, 0, 4096, 268_435_456, 32_768, 2_147_483_648, 900_000_000),
            Experts = new ExpertMemory(15_000_000_000, 450_000_000),
        };

        var manifest = ManifestFactory.Create(plan, inspection, file, "1.0.0", null, "0.4.2-alpha", null, DateTimeOffset.UnixEpoch);

        Assert.All(manifest.Requirements!.Memory, memory =>
        {
            Assert.Equal(15_000_000_000, memory.ExpertWeights);
            Assert.Equal(450_000_000, memory.ExpertBytesPerToken);
        });
        var bytes = FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest);
        Assert.Contains("\"expertWeights\":15000000000", System.Text.Encoding.UTF8.GetString(bytes).Replace(" ", "", StringComparison.Ordinal), StringComparison.Ordinal);
        var read = FeedJson.ReadManifest(bytes);
        Assert.Equal(450_000_000, read.Requirements!.Memory[0].ExpertBytesPerToken);
    }

    [Fact]
    public void Inspector_CountsOnlyRoutedExpertBanks()
    {
        // Two MoE layers: a routed bank of 8 experts (2 selected per token) for gate and down, and a shared expert (1 of 1, every row
        // runs through it) and a dense projection that do not count.
        const int hidden = 8;
        const int inner = 16;
        const int experts = 8;
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        Value Weight(GraphBuilder builder, string name, params long[] dims)
        {
            var shape = new Shape(dims.Select(dim => (Dim)dim));
            weights[name] = new WeightInfo { Name = name, Type = DataType.F32, Shape = shape, Source = new MemoryWeightSource(new byte[shape.Elements() * sizeof(float)]) };
            return builder.Weight(name);
        }

        var builder = new GraphBuilder("decode", name => weights.GetValueOrDefault(name));
        var tokens = Dim.Sym("T");
        var h = builder.Input("x", InputRole.Other, DataType.F32, new Shape([tokens, hidden]));
        var ids = builder.Input("ids", InputRole.Other, DataType.I32, new Shape([tokens, 2]));
        var shared = builder.Input("shared_ids", InputRole.Other, DataType.I32, new Shape([tokens, 1]));
        for (var layer = 0; layer < 2; layer++)
        {
            h = builder.MatMul(h, Weight(builder, $"dense{layer}", hidden, hidden));
            var up = builder.Emit1("MatMulId", [h, Weight(builder, $"up{layer}", experts, inner, hidden), ids]);
            var down = builder.Emit1("MatMulId", [up, Weight(builder, $"down{layer}", experts, hidden, inner), ids]);
            var sharedUp = builder.Emit1("MatMulId", [h, Weight(builder, $"shared{layer}", 1, inner, hidden), shared]);
            builder.Output($"down{layer}", down);
            builder.Output($"shared{layer}", sharedUp);
        }

        builder.Output("y", h);
        var measured = PackageInspector.MeasureExperts(builder.Build());

        long bank = (long)experts * inner * hidden * sizeof(float);
        Assert.NotNull(measured);
        Assert.Equal(4 * bank, measured.Bytes);
        Assert.Equal(4 * (bank / experts * 2), measured.BytesPerToken);
    }

    [Fact]
    public void Inspector_SizesTheDecoderStateOfARealPackage()
    {
        // A built package from the local store when present (feedtool build --entry qwen3-0.6b); skipped otherwise.
        var path = Environment.GetEnvironmentVariable("SHARPLIGHT_FEED_TEST_PACKAGE");
        if (path is null || !File.Exists(path))
        {
            Assert.Skip("SHARPLIGHT_FEED_TEST_PACKAGE (a built qwen3-0.6b .slm) is not set.");
        }

        var inspection = PackageInspector.Inspect(path);
        var context = inspection.Context!;
        // Qwen3 0.6B: 28 layers x 8 KV heads x 128 x (K + V) x 2 bytes.
        Assert.Equal(28L * 8 * 128 * 2 * 2, context.KvBytesPerToken);
        Assert.Equal(0, context.StateBytesPerSlot);
        Assert.Equal(context.KvBytesPerToken * context.DefaultTokens, context.DefaultBytes);
        Assert.Equal("q8_0", inspection.WeightTypes[0].Key);
    }

    [Fact]
    public void QualityTiersAreTheSameInCatalogAndFeed() => Assert.Equal(CatalogQualityTiers.All, PackageQualityTiers.All);

    [Theory]
    [InlineData("qwen3-0.6b", "chat")]
    [InlineData("all-minilm-l6-v2", "embedding")]
    [InlineData("ms-marco-minilm-l6-v2", "rerank")]
    [InlineData("silero-vad-v5", "voice-activity")]
    public void Tasks_FollowKindAndTags(string entryId, string task)
    {
        var entry = Catalog().Entries.Single(candidate => candidate.Id == entryId);
        Assert.Contains(task, ManifestFactory.Describe(entry).Tasks);
    }

    private static ProvenanceSource[] Pins(string revision = "1") => [new() { Repository = "org/model", Revision = new string(revision[0], 40), Role = "primary" }];

    private static PackageManifest Draft(string kind, string sha256, ProvenanceSource[] pins) => new()
    {
        Id = "model.q8",
        Version = "1.0.0",
        Files = [new PackageFile { Name = "pending.slm", Role = PackageFileRoles.Package, Size = 3, Sha256 = sha256 }],
        Model = new PackageModel { Kind = kind, Tasks = [kind] },
        Provenance = new PackageProvenance { Sources = pins, Built = DateTimeOffset.UtcNow, Converter = new ConverterInfo { Engine = "9.9.9", Tool = "feedtool 9.9.9" } },
        Requirements = new PackageRequirements { Engine = ">=9.9.0" },
    };

    private static async Task<(string Feed, FeedSigningKey Key, PackageManifest Published)> PublishedFeedAsync(TempDirectory directory, string kind)
    {
        var feed = directory.Combine("feed");
        Directory.CreateDirectory(feed);
        using var root = FeedSigningKey.Generate();
        var key = FeedSigningKey.Generate();
        FeedPublisher.WriteKeySet(feed, new FeedKeySet { Version = 1, Issued = DateTimeOffset.UtcNow.AddDays(-1), Keys = [new FeedKeySetEntry { KeyId = key.KeyId, PublicKey = key.PublicKey.Base64 }] }, root);
        var staged = directory.Combine("staged");
        Directory.CreateDirectory(staged);
        await File.WriteAllBytesAsync(Path.Combine(staged, "model.q8.1.0.0.slm"), [1, 2, 3], TestContext.Current.CancellationToken);
        var file = await PackageFileHasher.DescribeAsync(Path.Combine(staged, "model.q8.1.0.0.slm"), "model.q8.1.0.0.slm", cancellationToken: TestContext.Current.CancellationToken);
        var manifest = Draft(kind, file.Sha256, Pins()) with { Files = [file] };
        await File.WriteAllBytesAsync(Path.Combine(staged, FeedLayout.ManifestFileName), FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest), TestContext.Current.CancellationToken);
        await new FeedPublisher().PublishAsync(feed, [staged], key, new FeedPublishOptions { Channel = FeedChannels.Unstable }, TestContext.Current.CancellationToken);
        return (feed, key, manifest);
    }

    [Fact]
    public async Task Versions_FollowPinsAndContent()
    {
        using var directory = new TempDirectory();
        Assert.Equal("1.0.0", VersionPlanner.Plan(null, Draft("embeddings", new string('a', 64), Pins()))!.Version);
        var (feed, key, published) = await PublishedFeedAsync(directory, "embeddings");
        using var _ = key;
        var sha = published.PrimaryFile!.Sha256;

        // Identical file and metadata (a rebuild by a newer engine changes only build time, converter and minimum engine): nothing.
        Assert.Null(VersionPlanner.Plan(feed, Draft("embeddings", sha, Pins())));
        // A changed file is a new version as before.
        Assert.Equal("1.1.0", VersionPlanner.Plan(feed, Draft("embeddings", new string('b', 64), Pins()))!.Version);
        Assert.Null(VersionPlanner.Plan(feed, Draft("embeddings", new string('b', 64), Pins()))!.ReusesPayloadOf);
        Assert.Equal("2.0.0", VersionPlanner.Plan(feed, Draft("embeddings", new string('b', 64), Pins("2")))!.Version);
    }

    [Fact]
    public async Task ChangedMetadataWithSamePayload_PublishesPatchVersionSharingThePayload()
    {
        using var directory = new TempDirectory();
        var (feed, key, published) = await PublishedFeedAsync(directory, "embeddings");
        using var _ = key;
        var file = published.PrimaryFile!;
        var plan = VersionPlanner.Plan(feed, Draft("reranker", file.Sha256, Pins()))!;
        Assert.Equal("1.0.1", plan.Version);
        Assert.Equal("1.0.0", plan.ReusesPayloadOf!.Version);

        var staged = directory.Combine("staged-1.0.1");
        Directory.CreateDirectory(staged);
        var manifest = Draft("reranker", file.Sha256, Pins()) with { Version = "1.0.1", Files = [file with { Name = "model.q8.1.0.1.slm" }] };
        await File.WriteAllBytesAsync(Path.Combine(staged, FeedLayout.ManifestFileName), FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest), TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(Path.Combine(staged, FeedLayout.PayloadReuseFileName), FeedJson.Write(new PayloadReuse { Version = "1.0.0", File = file.Name }, FeedJsonContext.Default.PayloadReuse), TestContext.Current.CancellationToken);
        var result = await new FeedPublisher().PublishAsync(feed, [staged], key, new FeedPublishOptions { Channel = FeedChannels.Unstable }, TestContext.Current.CancellationToken);
        Assert.Equal(["model.q8@1.0.1"], result.Added);

        // The new version has its own manifest (new kind), and the same bytes as the first version's file under its own name.
        var stored = FeedJson.ReadManifest(File.ReadAllBytes(Path.Combine(feed, FeedLayout.Manifest("model.q8", "1.0.1"))));
        Assert.Equal("reranker", stored.Model!.Kind);
        Assert.Equal(File.ReadAllBytes(Path.Combine(feed, FeedLayout.VersionFile("model.q8", "1.0.0", file.Name))), File.ReadAllBytes(Path.Combine(feed, FeedLayout.VersionFile("model.q8", "1.0.1", "model.q8.1.0.1.slm"))));
        Assert.Equal("embeddings", FeedJson.ReadManifest(File.ReadAllBytes(Path.Combine(feed, FeedLayout.Manifest("model.q8", "1.0.0")))).Model!.Kind);

        // Promoting the id takes the latest unstable version.
        await new FeedPublisher().PromoteAsync(feed, FeedChannels.Stable, ["model.q8"], FeedChannels.Unstable, key, cancellationToken: TestContext.Current.CancellationToken);
        var stable = FeedJson.Read(File.ReadAllBytes(Path.Combine(feed, FeedLayout.InChannel(FeedChannels.Stable, FeedLayout.PackageIndex("model.q8")))), FeedJsonContext.Default.FeedPackageIndex, "index");
        Assert.Equal(["1.0.1"], stable.Versions.Select(version => version.Version));

        // The new latest version now matches: nothing more to build.
        Assert.Null(VersionPlanner.Plan(feed, Draft("reranker", file.Sha256, Pins())));
    }

    [Fact]
    public async Task ReusedPayload_MustMatchTheEarlierFile()
    {
        using var directory = new TempDirectory();
        var (feed, key, published) = await PublishedFeedAsync(directory, "embeddings");
        using var _ = key;
        var staged = directory.Combine("staged-bad");
        Directory.CreateDirectory(staged);
        var manifest = Draft("reranker", new string('c', 64), Pins()) with { Version = "1.0.1", Files = [published.PrimaryFile! with { Name = "model.q8.1.0.1.slm", Sha256 = new string('c', 64) }] };
        await File.WriteAllBytesAsync(Path.Combine(staged, FeedLayout.ManifestFileName), FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest), TestContext.Current.CancellationToken);
        await File.WriteAllBytesAsync(Path.Combine(staged, FeedLayout.PayloadReuseFileName), FeedJson.Write(new PayloadReuse { Version = "1.0.0", File = published.PrimaryFile!.Name }, FeedJsonContext.Default.PayloadReuse), TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<FeedFormatException>(() => new FeedPublisher().PublishAsync(feed, [staged], key, new FeedPublishOptions { Channel = FeedChannels.Unstable }, TestContext.Current.CancellationToken));
    }
}
