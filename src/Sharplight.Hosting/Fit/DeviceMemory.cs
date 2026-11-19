namespace Sharplight.Hosting.Fit;

/// <summary>Free and total memory of one physical device, as a load decision sees it.</summary>
/// <param name="Id">Device id (<c>provider:index</c>).</param>
/// <param name="Name">Device name.</param>
/// <param name="IsGpu">A GPU (else the CPU, whose memory is the system memory).</param>
/// <param name="TotalBytes">Memory the device has.</param>
/// <param name="FreeBytes">Memory this app can still allocate (other applications' and the resident models' use already taken off).</param>
/// <param name="IsExact">The driver reports the free memory (else it is this app's own tally).</param>
/// <param name="CapacityBytes">
/// Memory this app could hold on the device: its own use plus <paramref name="FreeBytes"/> (see <see cref="Sharplight.Runtime.DeviceMemoryInfo.Capacity"/>);
/// 0 when unknown, then <see cref="UsableBytes"/> is <paramref name="TotalBytes"/>.
/// </param>
public sealed record DeviceMemory(string Id, string Name, bool IsGpu, long TotalBytes, long FreeBytes, bool IsExact, long CapacityBytes = 0)
{
    /// <summary>
    /// What a model can use on this device once the app's own models are unloaded: <see cref="CapacityBytes"/> when known, else
    /// <see cref="TotalBytes"/>. "Fits this machine" decisions use it; memory other applications hold is not counted.
    /// </summary>
    public long UsableBytes => CapacityBytes > 0 ? Math.Min(CapacityBytes, TotalBytes) : TotalBytes;
}
