using Sharplight.Runtime;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests;

/// <summary><see cref="FitPlanner"/> decides like the engine's model host admits a load.</summary>
public sealed class FitPlannerTests
{
    private const long GiB = 1L << 30;

    private static readonly DeviceMemory s_gpu = new("gpu:0", "GPU A", true, 16 * GiB, 12 * GiB, true);
    private static readonly DeviceMemory s_gpu2 = new("gpu:1", "GPU B", true, 8 * GiB, 8 * GiB, true);
    private static readonly DeviceMemory s_system = new("cpu:0", "CPU", false, 32 * GiB, 20 * GiB, true);

    private static FitQuery Query(ModelFootprint footprint, IReadOnlyList<DeviceMemory>? targets = null, params ResidentUsage[] residents) => new()
    {
        Footprint = footprint,
        Targets = targets ?? [s_gpu],
        SystemMemory = s_system,
        ReserveBytes = 1 * GiB,
        Residents = residents,
        CpuOffload = true,
    };

    [Fact]
    public void FitsWithinFreeMemoryMinusTheReserve()
    {
        var report = FitPlanner.Plan(Query(new ModelFootprint(8 * GiB, 2 * GiB, GiB)));
        Assert.Equal(FitVerdict.Fits, report.Verdict);
        var device = Assert.Single(report.Devices);
        Assert.Equal(11 * GiB, device.NeededBytes);
        Assert.Equal(11 * GiB, device.AvailableBytes);
        Assert.True(report.LoadsAsConfigured);
    }

    [Fact]
    public void FitsAfterUnloading_NamesTheLeastRecentlyUsedIdleModelsFirst()
    {
        var now = DateTimeOffset.UtcNow;
        var report = FitPlanner.Plan(Query(new ModelFootprint(11 * GiB, GiB, GiB), null,
            new ResidentUsage("recent", ["gpu:0"], 2 * GiB, true, now),
            new ResidentUsage("old", ["gpu:0"], 2 * GiB, true, now.AddHours(-1)),
            new ResidentUsage("pinned", ["gpu:0"], 4 * GiB, false, now.AddDays(-1)),
            new ResidentUsage("elsewhere", ["gpu:1"], 6 * GiB, true, now.AddDays(-2))));
        Assert.Equal(FitVerdict.FitsAfterUnloading, report.Verdict);
        Assert.Equal(["old"], report.Unload);
        Assert.True(report.UnloadIsAutomatic);
        Assert.Equal(2 * GiB, report.Devices[0].FreedBytes);
    }

    [Fact]
    public void FitsAfterUnloading_PinnedModels_MustBeUnloadedByHand()
    {
        var report = FitPlanner.Plan(Query(new ModelFootprint(14 * GiB, GiB, GiB), null,
            new ResidentUsage("idle", ["gpu:0"], 2 * GiB, true, null),
            new ResidentUsage("pinned", ["gpu:0"], 3 * GiB, false, null)));
        Assert.Equal(FitVerdict.FitsAfterUnloading, report.Verdict);
        Assert.Equal(["idle", "pinned"], report.Unload);
        Assert.False(report.UnloadIsAutomatic);
        Assert.False(report.LoadsAsConfigured);
    }

    [Fact]
    public void NeedsOffload_ReportsTheShareOnTheCpu()
    {
        // 11 GiB available, 1 GiB of it for activations: 10 of the 20 GiB of weights and state run on the GPU.
        var report = FitPlanner.Plan(Query(new ModelFootprint(18 * GiB, 2 * GiB, GiB)) with { CpuOffload = false });
        Assert.Equal(FitVerdict.NeedsOffload, report.Verdict);
        Assert.Equal(0.5, report.CpuShare, 3);
        Assert.Equal(10 * GiB, report.SystemMemory!.NeededBytes);
        Assert.True(report.SystemMemory.Fits);
        Assert.False(report.LoadsAsConfigured);
    }

    [Fact]
    public void DoesNotFit_WhenTheOffloadedPartExceedsSystemMemory()
    {
        var report = FitPlanner.Plan(Query(new ModelFootprint(60 * GiB, 4 * GiB, GiB)));
        Assert.Equal(FitVerdict.DoesNotFit, report.Verdict);
        Assert.False(report.SystemMemory!.Fits);
    }

    [Fact]
    public void CpuOnly_ComparesAgainstSystemMemory()
    {
        Assert.Equal(FitVerdict.Fits, FitPlanner.Plan(Query(new ModelFootprint(16 * GiB, 2 * GiB, GiB), [s_system])).Verdict);
        var tooLarge = FitPlanner.Plan(Query(new ModelFootprint(24 * GiB, 2 * GiB, GiB), [s_system]));
        Assert.Equal(FitVerdict.DoesNotFit, tooLarge.Verdict);
        Assert.Null(tooLarge.SystemMemory);
    }

    [Fact]
    public void Split_SharesTheModelInProportionToFreeMemory()
    {
        // 11 + 7 GiB available: 18 GiB fits split, not on either GPU alone.
        var footprint = new ModelFootprint(15 * GiB, 2 * GiB, GiB);
        Assert.NotEqual(FitVerdict.Fits, FitPlanner.Plan(Query(footprint)).Verdict);
        var report = FitPlanner.Plan(Query(footprint, [s_gpu, s_gpu2]));
        Assert.Equal(FitVerdict.Fits, report.Verdict);
        Assert.Equal(footprint.Total, report.Devices.Sum(device => device.NeededBytes));
        Assert.Equal(11.0 / 7, (double)report.Devices[0].NeededBytes / report.Devices[1].NeededBytes, 3);
    }

    [Fact]
    public void Split_ModelResidentOnBothGpus_FreesMemoryOnEach()
    {
        var footprint = new ModelFootprint(17 * GiB, 2 * GiB, GiB);
        var report = FitPlanner.Plan(Query(footprint, [s_gpu, s_gpu2], new ResidentUsage("split", ["gpu:0", "gpu:1"], 4 * GiB, true, null)));
        Assert.Equal(FitVerdict.FitsAfterUnloading, report.Verdict);
        Assert.All(report.Devices, device => Assert.Equal(2 * GiB, device.FreedBytes));
    }
}
