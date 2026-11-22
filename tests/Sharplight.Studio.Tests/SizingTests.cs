using Sharplight.Hosting.Fit;
using Sharplight.Hub;
using Sharplight.Runtime;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Sizing from the engine's footprint estimate with mocked device figures: the longest context that fits on the GPU and with (expert) offload,
/// the KV cache type's effect, and memory figures that follow loads, unloads and a periodic read.
/// </summary>
public sealed class SizingTests
{
    private const long GiB = 1L << 30;

    static SizingTests() => LoadOptionsViewModel.EstimateDelay = TimeSpan.FromMilliseconds(1);

    /// <summary>
    /// A sliding-window model like Gemma 4 31B Q4_K_M on the session path: 17 GiB of weights, 1.25 GiB of window rings and 75 KiB per token of
    /// the full-attention layers' cache (f16; q8_0 at 34/64 of both), 0.5 GiB of activations: 32K about 21.1 GiB, 128K about 28.1 GiB.
    /// </summary>
    private static ModelFootprint Gemma(ModelLoadRequest request)
    {
        var settings = request.Settings;
        double scale = DownloadEstimate.KvBytesPerElement(settings.KvCacheType) / 2;
        long state = (long)(((1.25 * GiB) + (settings.ContextLength * 75.0 * 1024)) * scale) * settings.Slots;
        return new ModelFootprint(17 * GiB, state, GiB / 2);
    }

    /// <summary>A 32 GB card with 30.5 GiB free, 48 GiB of system memory.</summary>
    private static IReadOnlyList<DeviceMemory> Card32() =>
    [
        new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 32 * GiB, (long)(30.5 * GiB), true),
        new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 62 * GiB, 48 * GiB, true),
    ];

    private static LibraryEntry Entry(int? trained = null) => new()
    {
        Id = "gemma-4-31b",
        DisplayName = "google/gemma-4-31B-it-GGUF:Q4_K_M",
        Path = "/models/gemma-4-31b/model.slm",
        Quantization = "Q4_K_M",
        SizeBytes = 17 * GiB,
        TrainedContextLength = trained,
        Manifest = new ModelManifest { Repository = "google/gemma-4-31B-it-GGUF", Tag = "Q4_K_M", Alias = "gemma-4-31b", Commit = "abc" },
    };

    private static async Task Settle(LoadOptionsViewModel panel)
    {
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
    public void Candidates_ArePowersOfTwoUpToTheLimit_WithTheConfiguredContext()
    {
        Assert.Equal([2048, 4096, 8192, 16384, 32768, 40960], ContextSizer.Candidates(40960));
        Assert.Equal([2048, 4096, 8192, 10000, 16384, 32768], ContextSizer.Candidates(32768, 10000));
        Assert.Equal([1024], ContextSizer.Candidates(1024));
        Assert.Equal(ContextSizer.DefaultMaxContext, ContextSizer.Limit(null));
        Assert.Equal(262144, ContextSizer.Limit(262144));
        Assert.Equal(ContextSizer.MaxContext, ContextSizer.Limit(10_000_000));
    }

    [Fact]
    public async Task Longest_FindsTheLastFittingCandidate_WithFewProbes()
    {
        var candidates = ContextSizer.Candidates(1 << 20);
        var probed = new List<int>();
        int? longest = await ContextSizer.LongestAsync(candidates, context =>
        {
            probed.Add(context);
            return Task.FromResult(context <= 100_000);
        });
        Assert.Equal(65536, longest);
        Assert.True(probed.Count <= 4, string.Join(", ", probed));
        Assert.Null(await ContextSizer.LongestAsync(candidates, _ => Task.FromResult(false)));
        Assert.Equal(1 << 20, await ContextSizer.LongestAsync(candidates, _ => Task.FromResult(true)));
    }

    [Fact]
    public async Task SlidingWindowModel_On32GbCard_ShowsItsFootprints_AndTheLongestContext()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = Gemma;
        services.Runtime.Memory = Card32;
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry(trained: 262144)), services.Services, _ => { }, () => { });
        panel.ContextLength = 32768;
        await Settle(panel);

        var fit = panel.Fit!;
        Assert.Equal(FitVerdict.Fits, fit.Verdict);
        Assert.Equal("Fits on the GPU", fit.Title);
        Assert.Equal("≈ 21.1 GB", fit.TotalText);
        // 128K is the longest power of two that fits 29.5 GiB (30.5 free, 1 reserve).
        Assert.Contains(fit.Sizing, row => row.Text.StartsWith($"Up to 131,072 tokens on {FakeRuntime.Gpu.Name} · needs about 28.1 GB", StringComparison.Ordinal));
        // With offload the model's full 256K context fits.
        Assert.Contains(fit.Sizing, row => row.Text.StartsWith("Up to 262,144 tokens with CPU offload", StringComparison.Ordinal) && !row.Fits);
        // The other KV cache type and what it saves.
        var kv = Assert.Single(fit.Sizing, row => row.Text.StartsWith("Q8_0 KV cache", StringComparison.Ordinal));
        Assert.Contains("1.7 GB less", kv.Text, StringComparison.Ordinal);
        Assert.Equal(1, services.Runtime.Estimates[^1].Settings.Slots);

        kv.ApplyCommand.Execute(null);
        Assert.Equal("q8_0", panel.KvCache.Value);
        await Settle(panel);
        Assert.Equal("≈ 19.4 GB", panel.Fit!.TotalText);
        Assert.Contains(panel.Fit.Sizing, row => row.Text.StartsWith("F16 KV cache", StringComparison.Ordinal) && row.Text.Contains("more", StringComparison.Ordinal));
        // With a q8_0 cache the model's whole 256K context fits the card.
        Assert.Contains(panel.Fit.Sizing, row => row.Text.StartsWith($"Up to 262,144 tokens on {FakeRuntime.Gpu.Name}", StringComparison.Ordinal) && row.Fits);
    }

    [Fact]
    public async Task MixtureOfExperts_ThatOffloadsOnlyExperts_SaysSo()
    {
        using var services = new TestServices();
        // 33 GiB against 15 GiB available: 18 GiB go to the CPU, all of them experts.
        services.Runtime.Footprint = new ModelFootprint(30 * GiB, 2 * GiB, GiB);
        services.Runtime.ExpertBytes = 26 * GiB;
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry()), services.Services, _ => { }, () => { });
        panel.CpuOffload = true;
        await Settle(panel);

        var fit = panel.Fit!;
        Assert.Equal(FitVerdict.NeedsOffload, fit.Verdict);
        Assert.True(fit.ExpertOffload);
        Assert.Equal("Fits with CPU expert offload", fit.Title);
        Assert.Contains("The experts of some layers", fit.Message, StringComparison.Ordinal);
        Assert.Contains("attention and the shared layers stay on the GPU", fit.Message, StringComparison.Ordinal);
        Assert.Equal("Not even 2,048 tokens of context fit on the GPU.", fit.ContextSummary);

        // A dense model of the same size offloads layers.
        services.Runtime.ExpertBytes = 0;
        panel.ContextLength = 4096;
        await Settle(panel);
        Assert.False(panel.Fit!.ExpertOffload);
        Assert.Equal("Fits with CPU offload", panel.Fit.Title);
    }

    [Fact]
    public async Task OnTheCpu_SizesAgainstSystemMemory()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = Gemma;
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry(trained: 32768)), services.Services, _ => { }, () => { });
        panel.SelectedDevice = FakeRuntime.Cpu;
        panel.ContextLength = 32768;
        await Settle(panel);

        Assert.Equal("Fits in system memory", panel.Fit!.Title);
        Assert.Equal("The model's full context (32,768 tokens) fits in system memory.", panel.Fit.ContextSummary);
        Assert.DoesNotContain(panel.Fit.Sizing, row => row.Text.Contains("offload", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Watch_ReadsNowAndPeriodically_AndReportsOnlyRealChanges()
    {
        using var services = new TestServices();
        var time = new ManualTime();
        long free = 16 * GiB;
        services.Runtime.Memory = () =>
        [
            new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 16 * GiB, Interlocked.Read(ref free), true),
            new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 32 * GiB, 24 * GiB, true),
        ];
        using var watch = new DeviceMemoryWatch(services.Runtime, services.Settings, time, TimeSpan.FromSeconds(5));
        var changes = new List<long>();
        watch.Changed += (_, memory) =>
        {
            lock (changes)
            {
                changes.Add(memory[0].FreeBytes);
            }
        };

        using (watch.Watch())
        {
            Assert.Equal(1, watch.Watchers);
            await WaitFor(() => watch.Snapshot is not null);
            Assert.Equal(16 * GiB, watch.Snapshot![0].FreeBytes);

            // Small drift is not a change; another application taking 4 GiB is.
            Interlocked.Exchange(ref free, (16 * GiB) - (8L << 20));
            time.Advance(TimeSpan.FromSeconds(5));
            await Task.Delay(50, TestContext.Current.CancellationToken);
            Interlocked.Exchange(ref free, 12 * GiB);
            time.Advance(TimeSpan.FromSeconds(5));
            await WaitFor(() => watch.Snapshot![0].FreeBytes == 12 * GiB);
        }

        Assert.Equal(0, watch.Watchers);
        lock (changes)
        {
            Assert.Equal([16 * GiB, 12 * GiB], changes);
        }

        // Not watched: no periodic reads, but a load or unload reads again.
        Interlocked.Exchange(ref free, 8 * GiB);
        time.Advance(TimeSpan.FromSeconds(30));
        await Task.Delay(50, TestContext.Current.CancellationToken);
        Assert.Equal(12 * GiB, watch.Snapshot![0].FreeBytes);
        services.Runtime.Add(new FakeModel("other") { ResidentBytes = 4 * GiB });
        await WaitFor(() => watch.Snapshot![0].FreeBytes == 8 * GiB);
    }

    [Fact]
    public async Task LoadPanel_PlansAgain_WhenTheFiguresChange()
    {
        using var services = new TestServices();
        services.Runtime.Estimator = Gemma;
        long free = (long)(30.5 * GiB);
        services.Runtime.Memory = () =>
        [
            new DeviceMemory(FakeRuntime.Gpu.Id, FakeRuntime.Gpu.Name, true, 32 * GiB, Interlocked.Read(ref free), true),
            new DeviceMemory(FakeRuntime.Cpu.Id, FakeRuntime.Cpu.Name, false, 62 * GiB, 48 * GiB, true),
        ];
        var panel = new LoadOptionsViewModel(new LibraryEntryViewModel(Entry(trained: 131072)), services.Services, _ => { }, () => { });
        panel.ContextLength = 131072;
        await Settle(panel);
        Assert.Equal(FitVerdict.Fits, panel.Fit!.Verdict);
        Assert.True(services.Memory.Watchers > 0);

        // Another application takes 8 GiB: the next read moves the panel to offload.
        Interlocked.Exchange(ref free, (long)(22.5 * GiB));
        await services.Memory.RefreshAsync(TestContext.Current.CancellationToken);
        await WaitFor(() => panel.Fit?.Verdict == FitVerdict.NeedsOffload);
        await Settle(panel);
        Assert.Contains(panel.Fit!.Sizing, row => row.Text.StartsWith($"Up to 32,768 tokens on {FakeRuntime.Gpu.Name}", StringComparison.Ordinal));

        panel.CloseCommand.Execute(null);
        Assert.Equal(0, services.Memory.Watchers);
    }

    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 200 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }
}
