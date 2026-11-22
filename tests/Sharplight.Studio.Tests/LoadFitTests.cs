using Sharplight.Hosting.Fit;
using Sharplight.Hub;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The load panel's memory estimate follows the choices, explains the verdict and suggests changes that make the model fit.</summary>
public sealed class LoadFitTests
{
    private const long GiB = 1L << 30;

    static LoadFitTests() => LoadOptionsViewModel.EstimateDelay = TimeSpan.FromMilliseconds(1);

    /// <summary>10 GiB of weights, a KV cache of 128 KiB per token and slot per KV byte, 0.5 GiB of activations.</summary>
    private static ModelFootprint Footprint(ModelLoadRequest request)
    {
        var settings = request.Settings;
        double kv = settings.ContextLength * (128.0 * 1024) * DownloadEstimate.KvBytesPerElement(settings.KvCacheType) * settings.Slots / 2;
        return new ModelFootprint(10 * GiB, (long)kv, GiB / 2);
    }

    private static LibraryEntry Entry(string id, long size, string? quantization = null) => new()
    {
        Id = id,
        DisplayName = $"Qwen/Qwen3-8B-GGUF:{quantization}",
        Path = $"/models/{id}/model.slm",
        Quantization = quantization,
        SizeBytes = size,
        Manifest = new ModelManifest { Repository = "Qwen/Qwen3-8B-GGUF", Tag = quantization ?? "", Alias = "qwen3-8b", Commit = "abc" },
    };

    private static async Task Settle(LoadOptionsViewModel panel)
    {
        // An estimate can schedule another (the runtime changed); wait until the latest is done.
        for (int i = 0; i < 20; i++)
        {
            var estimate = panel.Estimate;
            await estimate.WaitAsync(TestContext.Current.CancellationToken);
            if (ReferenceEquals(estimate, panel.Estimate))
            {
                return;
            }
        }
    }

    [Fact]
    public async Task Estimate_Fits_ShowsTheBreakdownAgainstTheDevice()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = Footprint;
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry("qwen3-8b", 6 * GiB, "Q8_0")), services.Services, _ => { }, () => { });
        await Settle(panel);

        var fit = Assert.IsType<FitEstimateViewModel>(panel.Fit);
        Assert.Equal(FitVerdict.Fits, fit.Verdict);
        Assert.True(fit.IsGood);
        Assert.Equal("Fits on the GPU", fit.Title);
        Assert.Equal("Weights 10.0 GB · KV cache 1.0 GB · activations 512 MB", fit.Breakdown);
        var device = Assert.Single(fit.Devices);
        Assert.Equal(FakeRuntime.Gpu.Name, device.Name);
        Assert.Contains("free of 16.0 GB · 1.0 GB reserve", device.MemoryText, StringComparison.Ordinal);
        Assert.False(fit.HasSuggestions);
        Assert.Equal(8192, services.Runtime.Estimates[^1].Settings.ContextLength);
    }

    [Fact]
    public async Task Estimate_FollowsTheChoices_AndAppliesTheLongestContextThatFits()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = Footprint;
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry("qwen3-8b", 6 * GiB, "Q8_0")), services.Services, _ => { }, () => { });
        await Settle(panel);

        // 64k tokens: 8 GiB of KV cache, 18.5 GiB in all against 15 GiB available (16 GiB free, 1 GiB reserve).
        panel.ContextLength = 65536;
        await Settle(panel);
        var fit = panel.Fit!;
        Assert.Equal(FitVerdict.NeedsOffload, fit.Verdict);
        Assert.Equal("Needs CPU offload", fit.Title);
        Assert.True(fit.IsWarning);
        Assert.Contains("Turn on CPU offload, or the load fails.", fit.Message, StringComparison.Ordinal);
        Assert.Contains(fit.Devices, device => device.IsOffload && device.Name == "System memory");

        var texts = fit.Suggestions.Select(suggestion => suggestion.Text).ToList();
        Assert.Contains(texts, text => text.StartsWith("Turn on CPU offload · about ", StringComparison.Ordinal));
        Assert.Contains(texts, text => text.StartsWith("Q8_0 KV cache", StringComparison.Ordinal));
        Assert.DoesNotContain(texts, text => text.StartsWith("Context of", StringComparison.Ordinal));

        // The longest context on the GPU (32k: 14.5 GiB) and with offload (the trained-for limit, 128k when unknown) are offered instead.
        var sizing = fit.Sizing.Select(row => row.Text).ToList();
        Assert.Contains($"Up to 32,768 tokens on {FakeRuntime.Gpu.Name} · needs about 14.5 GB", sizing);
        Assert.Contains(sizing, text => text.StartsWith("Up to 131,072 tokens with CPU offload", StringComparison.Ordinal));
        var context = fit.Sizing.First(row => row.Text.StartsWith("Up to 32,768", StringComparison.Ordinal));
        Assert.True(context.Fits);
        context.ApplyCommand.Execute(null);
        Assert.Equal(32768, panel.ContextLength);
        await Settle(panel);
        Assert.Equal(FitVerdict.Fits, panel.Fit!.Verdict);
        Assert.Equal(32768, services.Runtime.Estimates.Last(request => request.Settings.ContextLength == 32768).Settings.ContextLength);
        Assert.Equal($"32,768 tokens is the longest context that fits on the GPU.", panel.Fit.ContextSummary);
        Assert.DoesNotContain(panel.Fit.Sizing, row => row.Text.StartsWith("Up to 32,768", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Estimate_NamesTheModelsToUnload()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = Footprint;
        services.Runtime.Add(new FakeModel("idle-model") { ResidentBytes = 6 * GiB, IsPinned = false, LastUsed = DateTimeOffset.UtcNow.AddHours(-1) });
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry("qwen3-8b", 6 * GiB, "Q8_0")), services.Services, _ => { }, () => { });
        await Settle(panel);
        var fit = panel.Fit!;
        Assert.Equal(FitVerdict.FitsAfterUnloading, fit.Verdict);
        Assert.Equal("Fits after unloading other models", fit.Title);
        Assert.Contains("Fits once idle-model is unloaded; idle models unload automatically", fit.Message, StringComparison.Ordinal);
        Assert.True(fit.IsGood);

        // Pinned, it must be unloaded by hand.
        await services.Runtime.UnloadAsync("idle-model");
        services.Runtime.Add(new FakeModel("pinned-model") { ResidentBytes = 6 * GiB, IsPinned = true });
        await Settle(panel);
        Assert.Contains("Unload pinned-model first", panel.Fit!.Message, StringComparison.Ordinal);
        Assert.True(panel.Fit.IsWarning);
    }

    [Fact]
    public async Task Estimate_OnTheCpu_ComparesAgainstSystemMemory_AndSuggestsASmallerQuantization()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = request => new ModelFootprint(30 * GiB, GiB, GiB);
        var big = Entry("qwen3-8b-q8", 30 * GiB, "Q8_0");
        var small = Entry("qwen3-8b-q4", 15 * GiB, "Q4_K_M");
        LibraryEntry? opened = null;
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(big), services.Services, _ => { }, () => { }, () => [big, small], entry => opened = entry);
        panel.SelectedDevice = FakeRuntime.Cpu;
        await Settle(panel);

        var fit = panel.Fit!;
        Assert.Equal(FitVerdict.DoesNotFit, fit.Verdict);
        Assert.True(fit.IsBad);
        Assert.Equal("System memory", Assert.Single(fit.Devices).Name);
        Assert.Contains("system memory has 23.0 GB available", fit.Message, StringComparison.Ordinal);
        var smaller = Assert.Single(fit.Suggestions, suggestion => suggestion.Suggestion.Variant is not null);
        Assert.StartsWith("Load Q4_K_M instead (15.0 GB)", smaller.Text, StringComparison.Ordinal);
        smaller.ApplyCommand.Execute(null);
        Assert.Same(small, opened);
    }

    [Fact]
    public async Task Estimate_WithSeveralGpus_SuggestsSplitting()
    {
        using var services = new TestServices();
        var second = FakeRuntime.Gpu with { Id = "vulkan:1", Name = "Second GPU" };
        services.Runtime.Devices = [FakeRuntime.Gpu, second, ComputeDeviceOption.AllGpus([FakeRuntime.Gpu, second]), FakeRuntime.Cpu];
        services.Runtime.Memory = () =>
        [
            new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 16 * GiB, 12 * GiB, true),
            new DeviceMemory(second.Id, second.Name, true, 16 * GiB, 12 * GiB, true),
            new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 32 * GiB, 24 * GiB, true),
        ];
        services.Runtime.Footprint = new ModelFootprint(16 * GiB, 2 * GiB, GiB);
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry("big", 16 * GiB)), services.Services, _ => { }, () => { });
        await Settle(panel);
        var split = Assert.Single(panel.Fit!.Suggestions, suggestion => suggestion.Text == "Split over all 2 GPUs");
        Assert.True(split.Fits);

        split.ApplyCommand.Execute(null);
        Assert.True(panel.SelectedDevice!.IsSplit);
        await Settle(panel);
        Assert.Equal(FitVerdict.Fits, panel.Fit!.Verdict);
        Assert.Equal(2, panel.Fit.Devices.Count);
        Assert.Contains("on 2 GPUs", panel.Fit.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Estimate_ModelThatCannotBeRead_ShowsWhy()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = _ => throw new InvalidDataException("Not a model file.");
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry("broken", GiB)), services.Services, _ => { }, () => { });
        await Settle(panel);
        Assert.Null(panel.Fit);
        Assert.Equal("No memory estimate: Not a model file.", panel.FitError);
        Assert.False(panel.IsEstimating);
    }
}
