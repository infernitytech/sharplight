namespace Sharplight.Studio.Services.FirstRun;

/// <summary>What the machine offers a model: the best GPU, system memory and the free disk space of the model store.</summary>
/// <param name="GpuName">The name of the GPU models would run on, or null without one.</param>
/// <param name="GpuMemoryBytes">Memory of that GPU (0 without one).</param>
/// <param name="GpuDeviceId">The runtime's id of that GPU, or null without one.</param>
/// <param name="GpuCount">How many GPUs the runtime offers.</param>
/// <param name="SystemMemoryBytes">System memory (0 when unknown).</param>
/// <param name="FreeDiskBytes">Free space where models are stored, or null when it could not be read.</param>
internal sealed record MachineProfile(string? GpuName, long GpuMemoryBytes, string? GpuDeviceId, int GpuCount, long SystemMemoryBytes, long? FreeDiskBytes)
{
    /// <summary>A GPU with memory is available.</summary>
    public bool HasGpu => GpuMemoryBytes > 0;
}
