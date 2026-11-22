using Sharplight.Hosting.Fit;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>Before downloading: an approximate footprint from the metadata, a fit badge per quantization and a warning that does not block.</summary>
public sealed class DownloadFitTests
{
    private const long GiB = 1L << 30;

    private static readonly ModelShape s_qwen8b = new(36, 32, 8, 4096, 128, 128, 40960, 151936);

    [Fact]
    public void Estimate_SizesTheKvCacheFromTheDimensions()
    {
        var settings = new LoadSettings { ContextLength = 8192, Slots = 2, KvCacheType = "f16", BatchTokens = 512 };
        var footprint = DownloadEstimate.Estimate(5 * GiB, s_qwen8b, settings);
        Assert.Equal(5 * GiB, footprint.WeightBytes);
        // 36 layers · 8 KV heads · (128 + 128) · 8192 tokens · 2 slots · 2 bytes.
        Assert.Equal(36L * 8 * 256 * 8192 * 2 * 2, footprint.StateBytes);
        Assert.InRange(footprint.ActivationBytes, 256L << 20, 1L << 30);

        var q8 = DownloadEstimate.Estimate(5 * GiB, s_qwen8b, settings with { KvCacheType = "q8_0" });
        Assert.Equal(footprint.StateBytes * 17 / 32, q8.StateBytes);

        // The context is capped at what the model was trained for.
        var shortModel = s_qwen8b with { ContextLength = 4096 };
        Assert.Equal(4096, DownloadEstimate.Context(shortModel, settings));
        Assert.Equal(footprint.StateBytes / 2, DownloadEstimate.Estimate(5 * GiB, shortModel, settings).StateBytes);
    }

    [Fact]
    public void Estimate_LatentAttention_CachesTheLatentPerLayer()
    {
        var mla = new ModelShape(60, 128, 128, 7168, 192, 128, 163840, 129280, LatentRank: 512, RopeDimension: 64);
        Assert.Equal(576, mla.KvElementsPerTokenLayer);
    }

    [Fact]
    public void Estimate_UnknownDimensions_IsConservative()
    {
        var settings = new LoadSettings { ContextLength = 8192, Slots = 2 };
        var footprint = DownloadEstimate.Estimate(10 * GiB, null, settings);
        Assert.Equal(2 * GiB, footprint.StateBytes);
        Assert.True(footprint.Total > 12 * GiB);
    }

    [Fact]
    public void ShapeReader_ReadsCheckpointConfigs()
    {
        var plain = ModelShapeReader.FromConfig("""
            {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "hidden_size": 2048, "head_dim": 128,
             "max_position_embeddings": 40960, "vocab_size": 151936}
            """);
        Assert.Equal(new ModelShape(28, 16, 8, 2048, 128, null, 40960, 151936), plain);
        Assert.Equal(8 * 256, plain!.KvElementsPerTokenLayer);

        var nested = ModelShapeReader.FromConfig("""{"architectures": ["Gemma3ForConditionalGeneration"], "text_config": {"num_hidden_layers": 34, "num_attention_heads": 8, "num_key_value_heads": 4, "hidden_size": 2560}}""");
        Assert.Equal(34, nested!.Layers);
        Assert.Equal(4L * (320 + 320), nested.KvElementsPerTokenLayer);

        Assert.Null(ModelShapeReader.FromConfig("""{"model_type": "unknown"}"""));
        Assert.Null(ModelShapeReader.FromConfig("not json"));
    }

    private static async Task<HubSearchViewModel> SearchAsync(TestServices services)
    {
        services.Hub.Results.Add(new HubModelSummary("Qwen/Qwen3-8B-GGUF", 1, 1, false, "gguf", "text-generation", 8_190_000_000, [ModelFormat.Gguf], ["Q4_K_M", "Q5_K_M", "Q8_0"]));
        var search = new HubSearchViewModel(services.Services, new DownloadsViewModel(services.Services), () => new HashSet<string>());
        search.Query = "qwen3";
        await search.SearchCommand.ExecuteAsync(null);
        for (int i = 0; i < 200 && (search.Variants.Count < 3 || search.Variants.Any(variant => variant.Fit is null)); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
            await search.FitEstimate;
        }

        return search;
    }

    [Fact]
    public async Task Variants_GetAFitBadge_AndTheSelectedOneWarnsWithoutBlockingTheDownload()
    {
        using var services = new TestServices();
        services.Runtime.Memory = () =>
        [
            new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 12 * GiB, 10 * GiB, true),
            new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 32 * GiB, 24 * GiB, true),
        ];
        var search = await SearchAsync(services);

        var badges = search.Variants.ToDictionary(variant => variant.Tag, variant => variant.Fit!);
        Assert.Equal("Fits", badges["Q4_K_M"].Text);
        Assert.True(badges["Q4_K_M"].IsGood);
        Assert.StartsWith("CPU offload ~", badges["Q8_0"].Text, StringComparison.Ordinal);
        Assert.True(badges["Q8_0"].IsWarning);
        Assert.Contains("approximate", badges["Q8_0"].ToolTip, StringComparison.Ordinal);
        Assert.Contains("the model's dimensions", badges["Q8_0"].Detail, StringComparison.Ordinal);
        Assert.Equal("Fits with CPU offload (approximate)", badges["Q8_0"].Title);

        Assert.True(search.HasFitOk);
        search.SelectedVariant = search.Variants.Single(variant => variant.Tag == "Q8_0");
        Assert.True(search.HasFitWarning);
        Assert.Same(badges["Q8_0"], search.SelectedFit);
        Assert.True(search.DownloadCommand.CanExecute(null));
    }

    [Fact]
    public async Task Variants_TooLargeForGpuAndSystemMemory_AreMarkedTooLarge()
    {
        using var services = new TestServices();
        services.Runtime.Memory = () =>
        [
            new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 4 * GiB, 4 * GiB, true),
            new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 8 * GiB, 4 * GiB, true),
        ];
        services.Hub.Shape = null;
        var search = await SearchAsync(services);
        var q8 = search.Variants.Single(variant => variant.Tag == "Q8_0").Fit!;
        Assert.Equal("Too large", q8.Text);
        Assert.True(q8.IsBad);
        Assert.Contains("the download size only", q8.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Variants_WithSeveralGpus_FitSplit()
    {
        using var services = new TestServices();
        var second = FakeRuntime.Gpu with { Id = "vulkan:1", Name = "Second GPU" };
        services.Runtime.Devices = [FakeRuntime.Gpu, second, ComputeDeviceOption.AllGpus([FakeRuntime.Gpu, second]), FakeRuntime.Cpu];
        services.Runtime.Memory = () =>
        [
            new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 10 * GiB, 9 * GiB, true),
            new DeviceMemory(second.Id, second.Name, true, 8 * GiB, 7 * GiB, true),
            new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 32 * GiB, 24 * GiB, true),
        ];
        var search = await SearchAsync(services);
        var q8 = search.Variants.Single(variant => variant.Tag == "Q8_0").Fit!;
        Assert.Equal("Fits on 2 GPUs", q8.Text);
        Assert.True(q8.IsSplit);
        Assert.Equal("Fits", search.Variants.Single(variant => variant.Tag == "Q4_K_M").Fit!.Text);
    }
}
