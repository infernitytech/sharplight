using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Resolution;
using Sharplight.Studio.Services.FirstRun;

namespace Sharplight.Studio.Tests.FirstRun;

/// <summary>The starter recommendation: a chat model sized to the GPU (or the CPU), optional speech and embeddings, from feed metadata only.</summary>
public sealed class StarterRecommenderTests
{
    private const long Gib = StarterCatalog.Gib;

    private static readonly CompatibilityEnvironment s_environment = new();

    private static StarterRecommender Recommender() => new(StarterPreferencesLoader.Load(null));

    private static MachineProfile Gpu(long vramGib, long ramGib = 64, long? diskGib = 500)
        => new("Test GPU", vramGib * Gib, "vulkan:0", 1, ramGib * Gib, diskGib is { } disk ? disk * Gib : null);

    private static MachineProfile CpuOnly(long ramGib, long? diskGib = 500) => new(null, 0, null, 0, ramGib * Gib, diskGib is { } disk ? disk * Gib : null);

    private static string ChatId(StarterRecommendation recommendation) => recommendation.Items.Single(item => item.Role == StarterRole.Chat).Package.Id;

    [Theory]
    [InlineData(8, "qwen3-8b.q4-k-m-gguf")]
    [InlineData(24, "qwen3-14b.q4-k-m-gguf")]
    [InlineData(32, "qwen3-32b.q4-k-m-gguf")]
    public void Chat_IsSizedToVram(long vramGib, string expected)
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), Gpu(vramGib), s_environment);

        Assert.Equal(StarterOutcome.Recommended, recommendation.Outcome);
        Assert.Equal(expected, ChatId(recommendation));
        Assert.False(recommendation.CpuOnly);
        Assert.True(recommendation.Items.Single(item => item.Role == StarterRole.Chat).OnGpu);
        Assert.True(recommendation.Items[0].MemoryBytes <= vramGib * Gib);
    }

    [Fact]
    public void SmallGpu_FallsBackToTheCpuSizedModel()
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), Gpu(1, ramGib: 16), s_environment);

        Assert.Equal("qwen3-1.7b.q8-0-gguf", ChatId(recommendation));
        Assert.False(recommendation.Items[0].OnGpu);
        Assert.True(recommendation.CpuOnly);
    }

    [Fact]
    public void WithoutGpu_RecommendsACpuSizedModel()
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), CpuOnly(16), s_environment);

        Assert.Equal(StarterOutcome.Recommended, recommendation.Outcome);
        Assert.True(recommendation.CpuOnly);
        Assert.Equal("qwen3-1.7b.q8-0-gguf", ChatId(recommendation));
        Assert.All(recommendation.Items, item => Assert.False(item.OnGpu));
        Assert.True(recommendation.MemoryBudgetBytes <= 3 * Gib);
    }

    [Fact]
    public void WithoutGpuAndLittleMemory_RecommendsTheSmallestModel()
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), CpuOnly(4), s_environment);

        Assert.Equal("qwen3-0.6b.q8-0-gguf", ChatId(recommendation));
    }

    [Fact]
    public void Offline_WithoutAListing_SaysThatNothingIsListed()
    {
        var recommendation = Recommender().Recommend([], Gpu(24), s_environment);

        Assert.Equal(StarterOutcome.NoPackages, recommendation.Outcome);
        Assert.Empty(recommendation.Items);
        Assert.NotEmpty(recommendation.Message);
    }

    [Fact]
    public void SpeechAndEmbeddings_AreOfferedAfterChat_WithTheirDependencyCounted()
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), Gpu(24), s_environment);

        Assert.Equal([StarterRole.Chat, StarterRole.Speech, StarterRole.Embedding], recommendation.Items.Select(item => item.Role));
        var speech = recommendation.Items[1];
        Assert.Equal("whisper-tiny.q5-1-whispercpp", speech.Package.Id);
        Assert.Equal((long)(0.04 * Gib) + (long)(0.002 * Gib), speech.DownloadBytes);
        Assert.True(speech.Preselected);
        Assert.False(recommendation.Items[2].Preselected);
        Assert.All(recommendation.Items, item => Assert.True(item.DownloadTime > TimeSpan.Zero));
        Assert.True(recommendation.Items[0].DownloadTime > recommendation.Items[1].DownloadTime);
    }

    [Fact]
    public void DownloadTime_FollowsTheAssumedSpeed()
    {
        var preferences = StarterPreferencesLoader.Load(null) with { AssumedDownloadMbps = 80 };

        var time = new StarterRecommender(preferences).EstimateTime(1_000_000_000);

        Assert.Equal(100, time.TotalSeconds, 0.5);
    }

    [Fact]
    public void LittleDisk_PicksASmallerModel_AndNoDisk_PicksNone()
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), Gpu(32, diskGib: 4), s_environment);
        var none = Recommender().Recommend(StarterCatalog.Feed(), Gpu(32, diskGib: 1), s_environment);

        Assert.Equal("qwen3-1.7b.q8-0-gguf", ChatId(recommendation));
        Assert.Equal(StarterOutcome.NothingFits, none.Outcome);
        Assert.Contains("disk space", none.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingFits_WhenEveryChatModelIsTooBig()
    {
        var recommendation = Recommender().Recommend([StarterCatalog.Chat("big", "Qwen3", 40, 38)], Gpu(8), s_environment);

        Assert.Equal(StarterOutcome.NothingFits, recommendation.Outcome);
        Assert.Contains("memory", recommendation.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WithdrawnAndDeprecatedPackages_AreNotRecommended()
    {
        var packages = new[]
        {
            StarterCatalog.Chat("qwen3-8b", "Qwen3", 5.8, 5, state: new PackageState { Yanked = new PackageYank { Reason = "bad" } }),
            StarterCatalog.Chat("qwen3-4b", "Qwen3", 3.0, 2.5, state: new PackageState { Deprecated = new PackageDeprecation { Reason = "old" } }),
            StarterCatalog.Chat("qwen3-1.7b", "Qwen3", 2.2, 1.8),
        };

        Assert.Equal("qwen3-1.7b", ChatId(Recommender().Recommend(packages, Gpu(24), s_environment)));
    }

    [Fact]
    public void ARequiredDependencyMissingFromTheListing_DisqualifiesThePackage()
    {
        var packages = new[] { StarterCatalog.Chat("qwen3-1.7b", "Qwen3", 2.2, 1.8), StarterCatalog.Speech("whisper-tiny", 0.45, 0.04, "silero-missing") };

        var recommendation = Recommender().Recommend(packages, Gpu(24), s_environment);

        Assert.Equal([StarterRole.Chat], recommendation.Items.Select(item => item.Role));
    }

    [Fact]
    public void Preferences_RankFamilies_AndCanBeOverriddenByAFile()
    {
        using var dir = new TempDir();
        string file = dir.File("starter-models.json");
        File.WriteAllText(file, """
            {
              // Prefer Gemma.
              "roles": [ { "role": "Chat", "kind": "text-generation", "task": "chat", "preselected": true, "prefer": ["gemma", "qwen3"] } ]
            }
            """);

        var preferences = StarterPreferencesLoader.Load(file);
        var recommendation = new StarterRecommender(preferences).Recommend(StarterCatalog.Feed(), Gpu(32), s_environment);

        Assert.Equal("gemma-3-27b.q4-k-m-gguf", ChatId(recommendation));
        Assert.Equal([StarterRole.Chat], recommendation.Items.Select(item => item.Role));
    }

    [Fact]
    public void AnInvalidOverrideFile_FallsBackToTheBuiltInPreferences()
    {
        using var dir = new TempDir();
        string file = dir.File("starter-models.json");
        File.WriteAllText(file, "{ not json");

        var preferences = StarterPreferencesLoader.Load(file);

        Assert.Equal(3, preferences.Roles.Count);
        Assert.NotEmpty(preferences.ExamplePrompts);
    }

    [Fact]
    public void TheBuiltInPreferences_NameNoPackageId()
    {
        var preferences = StarterPreferencesLoader.Load(null);

        Assert.Equal([StarterRole.Chat, StarterRole.Speech, StarterRole.Embedding], preferences.Roles.Select(role => role.Role));
        Assert.All(preferences.Roles.SelectMany(role => role.Prefer), fragment => Assert.DoesNotContain('.', fragment));
    }

    [Fact]
    public void FreeGpuMemory_NotTotalMemory_SizesTheModel()
    {
        // A GPU of 24 GB that other applications leave 7 GB of: the 14B model would not load.
        var machine = new MachineProfile("Test GPU", 24 * Gib, "vulkan:0", 1, 64 * Gib, 500 * Gib, GpuUsableBytes: 24 * Gib, GpuFreeBytes: (long)(7.5 * Gib));

        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), machine, s_environment);

        Assert.Equal("qwen3-8b.q4-k-m-gguf", ChatId(recommendation));
        Assert.True(recommendation.Items[0].OnGpu);
        Assert.Contains("7.5 GB free of 24.0 GB", recommendation.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void ACappedDevice_UsesTheCapacity_NotTheTotal()
    {
        var machine = new MachineProfile("Test GPU", 24 * Gib, "vulkan:0", 1, 64 * Gib, 500 * Gib, GpuUsableBytes: 8 * Gib, GpuFreeBytes: 8 * Gib);

        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), machine, s_environment);

        Assert.Equal("qwen3-8b.q4-k-m-gguf", ChatId(recommendation));
    }

    [Fact]
    public void TheModelLeavesHeadroomForTheContext()
    {
        // 6.3 GB free x 0.9 = 5.7 GB budget: the 5.8 GB figure of the 8B model is out, and so is it with the KV cache of a 4096-token context.
        var machine = new MachineProfile("Test GPU", 8 * Gib, "vulkan:0", 1, 16 * Gib, 500 * Gib, GpuUsableBytes: 8 * Gib, GpuFreeBytes: (long)(6.3 * Gib));

        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), machine, s_environment);

        Assert.NotEqual("qwen3-8b.q4-k-m-gguf", ChatId(recommendation));
        Assert.All(recommendation.Items.Where(item => item.OnGpu), item => Assert.True(item.MemoryBytes <= machine.GpuAvailableBytes));
        Assert.Contains("4096-token context", recommendation.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void AFeedContextFigure_ReplacesTheHeadroomEstimate()
    {
        var package = StarterCatalog.Chat("qwen3-8b", "Qwen3", 5.0, 4.0) with
        {
            Requirements = new PackageRequirements
            {
                Memory = [new MemoryRequirement { Device = "gpu", Weights = 5 * Gib, PerSlotContext = new SlotContextMemory { Bytes = Gib / 2, Tokens = 4096 } }],
            },
        };

        var recommendation = Recommender().Recommend([package], Gpu(8), s_environment);

        Assert.Equal(5 * Gib + Gib / 2, recommendation.Items[0].MemoryBytes);
    }

    [Fact]
    public void ALowVramGpu_RunsTheModelOnTheCpu_AndSaysSo()
    {
        var machine = new MachineProfile("Small GPU", 2 * Gib, "vulkan:0", 1, 16 * Gib, 500 * Gib, GpuUsableBytes: 2 * Gib, GpuFreeBytes: Gib);

        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), machine, s_environment);

        Assert.True(recommendation.CpuOnly);
        Assert.False(recommendation.Items[0].OnGpu);
        Assert.Contains("only 1.0 GB free", recommendation.Reason, StringComparison.Ordinal);
        Assert.Contains("CPU", recommendation.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void WithoutAGpu_TheFreeSystemMemory_SizesTheModel()
    {
        var machine = new MachineProfile(null, 0, null, 0, 16 * Gib, 500 * Gib, SystemFreeBytes: 4 * Gib);

        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), machine, s_environment);

        // 4 GB free x 0.4 = 1.6 GB: only the smallest model with its context headroom fits.
        Assert.Equal("qwen3-0.6b.q8-0-gguf", ChatId(recommendation));
        Assert.Contains("No GPU found", recommendation.Reason, StringComparison.Ordinal);
        Assert.Contains("16.0 GB system memory", recommendation.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void AUnifiedGpu_SharesItsMemoryWithTheOtherModels()
    {
        // The chat model takes 5.6 of the 5.85 GB budget: the speech and embedding models must not be placed on a second, imaginary pool.
        var machine = new MachineProfile("Integrated", 8 * Gib, "vulkan:0", 1, 8 * Gib, 500 * Gib, GpuUsableBytes: 8 * Gib, GpuFreeBytes: (long)(6.5 * Gib), GpuIsUnified: true);
        var packages = new[]
        {
            StarterCatalog.Chat("qwen3-8b", "Qwen3", 5.0, 4.0),
            StarterCatalog.Speech("whisper-tiny", 0.5, 0.04, "silero-vad-v5.onnx"),
            StarterCatalog.Detector("silero-vad-v5.onnx"),
        };

        var recommendation = Recommender().Recommend(packages, machine, s_environment);

        Assert.Equal("qwen3-8b", ChatId(recommendation));
        Assert.Contains("shared with the system", recommendation.Reason, StringComparison.Ordinal);
        Assert.DoesNotContain(recommendation.Items, item => item.Role == StarterRole.Speech);
    }

    [Fact]
    public void AMachineWithoutLiveFigures_UsesItsTotals()
    {
        var recommendation = Recommender().Recommend(StarterCatalog.Feed(), Gpu(8), s_environment);

        Assert.Contains("Fits in the 8.0 GB of GPU memory", recommendation.Reason, StringComparison.Ordinal);
    }
}
