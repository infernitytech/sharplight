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

    [Fact]
    public void ActiveAndInactiveFileCacheBothCountAsReclaimable()
    {
        // A pod whose usage is mostly the cache of model files just read: 44 GB on the active list, 9 GB inactive, 1 GB dirty.
        var v2 = new Dictionary<string, long> { ["active_file"] = 44L << 30, ["inactive_file"] = 9L << 30, ["file_dirty"] = 1L << 30, ["file_writeback"] = 0 };
        Assert.Equal(52L << 30, HostCpuInfo.ReclaimableFileCache(key => v2.TryGetValue(key, out long value) ? value : null, cgroupV1: false));

        var v1 = new Dictionary<string, long> { ["total_active_file"] = 3L << 30, ["total_inactive_file"] = 2L << 30, ["total_writeback"] = 1L << 30 };
        Assert.Equal(4L << 30, HostCpuInfo.ReclaimableFileCache(key => v1.TryGetValue(key, out long value) ? value : null, cgroupV1: true));

        // Never negative, and nothing when the counters are missing.
        Assert.Equal(0, HostCpuInfo.ReclaimableFileCache(key => key == "file_dirty" ? 5 : null, cgroupV1: false));
    }
}
