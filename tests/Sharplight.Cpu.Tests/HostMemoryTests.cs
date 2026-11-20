namespace Sharplight.Cpu.Tests;

/// <summary>
/// The CPU device's memory figures respect the container's memory limit (cgroup), which /proc/meminfo does not show: placement must
/// not offload more to host memory than the OOM killer allows.
/// </summary>
public class HostMemoryTests
{
    [Fact]
    public void MemoryFiguresStayWithinTheProcessLimit()
    {
        var provider = new CpuExecutionProvider();
        using var device = provider.OpenDevice(provider.GetDevices()[0]);
        // The process's cgroup v2 limit (containers), when there is one at the root of its hierarchy.
        const string limitFile = "/sys/fs/cgroup/memory.max";
        if (File.Exists(limitFile) && long.TryParse(File.ReadAllText(limitFile).Trim(), out long limit))
        {
            Assert.True(device.Info.TotalMemory <= limit, $"total {device.Info.TotalMemory} exceeds the cgroup limit {limit}");
            Assert.True(device.AvailableMemory <= limit, $"available {device.AvailableMemory} exceeds the cgroup limit {limit}");
        }

        Assert.InRange(device.AvailableMemory, 0, device.Info.TotalMemory);
    }
}
