using Sharplight.Hosting.Fit;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.FirstRun;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests.FirstRun;

/// <summary>The machine description: live free and usable memory, unified GPUs, the container's memory limit and devices that cannot be queried.</summary>
public sealed class MachineProfilerTests
{
    private const long Gib = 1L << 30;

    private static readonly ComputeDeviceOption s_small = new("vulkan:1", "Small GPU", "Vulkan", true, 8 * Gib, [], "gpu-1");

    [Fact]
    public void LiveMemory_GivesTheUsableAndFreeFigures()
    {
        using var runtime = new FakeRuntime { Devices = [s_small, FakeRuntime.Cpu] };
        IReadOnlyList<DeviceMemory> memory =
        [
            new DeviceMemory(s_small.Id, s_small.Name, true, 8 * Gib, (long)(7.2 * Gib), true, 8 * Gib),
            new DeviceMemory(FakeRuntime.Cpu.Id, "CPU", false, 32 * Gib, 20 * Gib, true),
        ];

        var machine = new MachineProfiler(runtime, _ => null).Describe("/store", memory);

        Assert.Equal(8 * Gib, machine.GpuMemoryBytes);
        Assert.Equal((long)(7.2 * Gib), machine.GpuFreeBytes);
        Assert.Equal((long)(7.2 * Gib), machine.GpuAvailableBytes);
        Assert.Equal(20 * Gib, machine.SystemFreeBytes);
    }

    [Fact]
    public void TheGpuWithTheMostFreeMemory_IsChosen()
    {
        var big = new ComputeDeviceOption("vulkan:0", "Busy GPU", "Vulkan", true, 24 * Gib, [], "gpu-0");
        using var runtime = new FakeRuntime { Devices = [big, s_small, FakeRuntime.Cpu] };
        IReadOnlyList<DeviceMemory> memory =
        [
            new DeviceMemory(big.Id, big.Name, true, 24 * Gib, 2 * Gib, true, 24 * Gib),
            new DeviceMemory(s_small.Id, s_small.Name, true, 8 * Gib, 7 * Gib, true, 8 * Gib),
        ];

        var machine = new MachineProfiler(runtime, _ => null).Describe("/store", memory);

        Assert.Equal(s_small.Id, machine.GpuDeviceId);
    }

    [Fact]
    public void AnIntegratedGpu_IsMarkedUnified()
    {
        var integrated = s_small with { IsUnified = true };
        using var runtime = new FakeRuntime { Devices = [integrated, FakeRuntime.Cpu] };

        var machine = new MachineProfiler(runtime, _ => null).Describe("/store");

        Assert.True(machine.GpuIsUnified);
    }

    [Fact]
    public void WithoutLiveMemory_TheTotalsAreUsed()
    {
        using var runtime = new FakeRuntime { Devices = [s_small, FakeRuntime.Cpu] };

        var machine = new MachineProfiler(runtime, _ => null).Describe("/store");

        Assert.Null(machine.GpuFreeBytes);
        Assert.Equal(8 * Gib, machine.GpuAvailableBytes);
    }

    [Fact]
    public void SystemMemory_NeverExceedsTheProcessLimit()
    {
        using var runtime = new FakeRuntime { Devices = [FakeRuntime.Cpu with { TotalMemory = long.MaxValue / 2 }] };

        var machine = new MachineProfiler(runtime, _ => null).Describe("/store");

        Assert.InRange(machine.SystemMemoryBytes, 1, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes);
    }

    [Fact]
    public async Task ADeviceQueryThatFails_FallsBackToTheTotals()
    {
        using var runtime = new FakeRuntime { Devices = [s_small, FakeRuntime.Cpu], Memory = () => throw new InvalidOperationException("no driver") };

        var machine = await new MachineProfiler(runtime, _ => null).DescribeAsync("/store", [], CancellationToken.None);

        Assert.Equal(8 * Gib, machine.GpuAvailableBytes);
        Assert.Null(machine.GpuFreeBytes);
    }

    [Fact]
    public async Task TheRealRuntime_ReportsAGpuWhoseFreeMemoryIsWithinItsCapacity()
    {
        using var runtime = new Sharplight.Studio.Services.Runtime.EngineModelRuntime(Sharplight.Studio.Services.Runtime.EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(!runtime.Devices.Any(device => device.IsGpu), "no GPU");

        var machine = await new MachineProfiler(runtime).DescribeAsync(Path.GetTempPath(), [], TestContext.Current.CancellationToken);

        // SHARPLIGHT_LIMIT_DEVICE_MEMORY lowers the usable and free figures below the total (docs/hardware.md).
        Assert.True(machine.HasGpu);
        Assert.InRange(machine.GpuUsableBytes, 1, machine.GpuMemoryBytes);
        Assert.InRange(machine.GpuFreeBytes ?? 0, 1, machine.GpuUsableBytes);
        Console.WriteLine($"GPU total {machine.GpuMemoryBytes >> 20} MiB, usable {machine.GpuUsableBytes >> 20} MiB, free {machine.GpuFreeBytes >> 20} MiB, unified {machine.GpuIsUnified}");
    }
}
