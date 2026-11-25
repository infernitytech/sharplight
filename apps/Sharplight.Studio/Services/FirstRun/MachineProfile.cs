namespace Sharplight.Studio.Services.FirstRun;

/// <summary>What the machine offers a model: the best GPU, system memory and the free disk space of the model store.</summary>
/// <param name="GpuName">The name of the GPU models would run on, or null without one.</param>
/// <param name="GpuMemoryBytes">Memory of that GPU (0 without one).</param>
/// <param name="GpuDeviceId">The runtime's id of that GPU, or null without one.</param>
/// <param name="GpuCount">How many GPUs the runtime offers.</param>
/// <param name="SystemMemoryBytes">System memory the process may use (the container limit when there is one; 0 when unknown).</param>
/// <param name="FreeDiskBytes">Free space where models are stored, or null when it could not be read.</param>
/// <param name="GpuUsableBytes">What a model could occupy on the GPU once Studio's own models are unloaded (a configured cap taken off); 0 when unknown.</param>
/// <param name="GpuFreeBytes">What can be allocated on the GPU now (other applications' use taken off), or null when the driver does not say.</param>
/// <param name="GpuIsUnified">The GPU shares system memory, so GPU and CPU models draw on one pool.</param>
/// <param name="SystemFreeBytes">System memory that can be allocated now, or null when unknown.</param>
internal sealed record MachineProfile(string? GpuName, long GpuMemoryBytes, string? GpuDeviceId, int GpuCount, long SystemMemoryBytes, long? FreeDiskBytes,
    long GpuUsableBytes = 0, long? GpuFreeBytes = null, bool GpuIsUnified = false, long? SystemFreeBytes = null)
{
    /// <summary>A GPU with memory is available.</summary>
    public bool HasGpu => GpuMemoryBytes > 0;

    /// <summary>The GPU memory a model can really get: the usable capacity, less what other applications hold now.</summary>
    public long GpuAvailableBytes
    {
        get
        {
            if (!HasGpu)
            {
                return 0;
            }

            long usable = GpuUsableBytes > 0 ? Math.Min(GpuUsableBytes, GpuMemoryBytes) : GpuMemoryBytes;
            return GpuFreeBytes is { } free ? Math.Min(usable, Math.Max(0, free)) : usable;
        }
    }

    /// <summary>The system memory a model can really get on the CPU: the limit, less what is in use now.</summary>
    public long SystemAvailableBytes => SystemFreeBytes is { } free && SystemMemoryBytes > 0 ? Math.Min(SystemMemoryBytes, Math.Max(0, free))
        : SystemFreeBytes is { } onlyFree ? Math.Max(0, onlyFree) : SystemMemoryBytes;
}
