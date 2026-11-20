using Sharplight.Hosting.Fit;
using Sharplight.Studio.Models;
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

    /// <summary>Describes the machine from the devices' totals and, with <paramref name="memory"/>, their usable and free memory.</summary>
    /// <param name="storeRoot">The model store, whose disk is measured.</param>
    /// <param name="memory">Live memory of the devices (<see cref="IModelRuntime.GetDeviceMemoryAsync"/>); null uses the totals only.</param>
    public MachineProfile Describe(string storeRoot, IReadOnlyList<DeviceMemory>? memory = null)
    {
        var devices = _runtime.Devices;
        var gpus = devices.Where(device => device.IsGpu && !device.IsSplit).ToList();
        DeviceMemory? Live(ComputeDeviceOption device) => memory?.FirstOrDefault(entry => entry.Id == device.Id);
        long Usable(ComputeDeviceOption device) => Live(device) is { } live ? live.UsableBytes : device.TotalMemory;
        long Free(ComputeDeviceOption device) => Live(device) is { } live ? Math.Min(live.UsableBytes, live.FreeBytes) : device.TotalMemory;
        var best = gpus.OrderByDescending(Free).ThenByDescending(Usable).ThenByDescending(device => device.TotalMemory).FirstOrDefault();
        long system = devices.Where(device => !device.IsGpu).Select(device => device.TotalMemory).DefaultIfEmpty(0).Max();

        // The container's limit (cgroup) is what the process can really get, whatever the host has.
        long limit = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (limit > 0)
        {
            system = system > 0 ? Math.Min(system, limit) : limit;
        }

        long? systemFree = memory?.Where(entry => !entry.IsGpu).Select(entry => (long?)entry.FreeBytes).FirstOrDefault();
        var live = best is null ? null : Live(best);
        return new MachineProfile(best?.Name, best?.TotalMemory ?? 0, best?.Id, gpus.Count, system, _freeDisk(storeRoot),
            live is null ? 0 : live.UsableBytes, live?.FreeBytes, best?.IsUnified == true, systemFree);
    }

    /// <summary>Describes the machine with the live memory of its devices; devices that cannot be queried fall back to their totals.</summary>
    /// <param name="storeRoot">The model store, whose disk is measured.</param>
    /// <param name="disabledFeatures">Device features the user switched off, for devices opened by the query.</param>
    /// <param name="cancellationToken">Cancels the query.</param>
    public async Task<MachineProfile> DescribeAsync(string storeRoot, IReadOnlyList<string> disabledFeatures, CancellationToken cancellationToken)
    {
        IReadOnlyList<DeviceMemory>? memory = null;
        try
        {
            memory = await _runtime.GetDeviceMemoryAsync(disabledFeatures, cancellationToken);
        }
        catch (Exception error) when (error is not OperationCanceledException and not OutOfMemoryException)
        {
            // The totals still size a recommendation.
        }

        return Describe(storeRoot, memory);
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
