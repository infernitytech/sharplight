using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.FirstRun;

/// <summary>Describes the machine from the engine's devices and the disk the models are stored on.</summary>
internal sealed class MachineProfiler
{
    private readonly IModelRuntime _runtime;
    private readonly Func<string, long?> _freeDisk;

    /// <param name="runtime">The runtime whose devices are described.</param>
    /// <param name="freeDisk">Free bytes of the disk holding a directory; null reads the real disk.</param>
    public MachineProfiler(IModelRuntime runtime, Func<string, long?>? freeDisk = null)
    {
        _runtime = runtime;
        _freeDisk = freeDisk ?? FreeSpaceOf;
    }

    /// <summary>Describes the machine.</summary>
    /// <param name="storeRoot">The model store, whose disk is measured.</param>
    public MachineProfile Describe(string storeRoot)
    {
        var devices = _runtime.Devices;
        var gpus = devices.Where(device => device.IsGpu && !device.IsSplit).ToList();
        var best = gpus.OrderByDescending(device => device.TotalMemory).FirstOrDefault();
        long system = devices.Where(device => !device.IsGpu).Select(device => device.TotalMemory).DefaultIfEmpty(0).Max();
        if (system <= 0)
        {
            system = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        }

        return new MachineProfile(best?.Name, best?.TotalMemory ?? 0, best?.Id, gpus.Count, system, _freeDisk(storeRoot));
    }

    private static long? FreeSpaceOf(string directory)
    {
        try
        {
            string? existing = Path.GetFullPath(directory);
            while (existing is not null && !Directory.Exists(existing))
            {
                existing = Path.GetDirectoryName(existing);
            }

            return existing is null ? null : new DriveInfo(existing).AvailableFreeSpace;
        }
        catch (Exception error) when (error is IOException or ArgumentException or UnauthorizedAccessException or NotSupportedException)
        {
            return null;
        }
    }
}
