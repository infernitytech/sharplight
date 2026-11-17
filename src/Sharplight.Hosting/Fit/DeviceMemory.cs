namespace Sharplight.Hosting.Fit;

/// <summary>Free and total memory of one physical device, as a load decision sees it.</summary>
/// <param name="Id">Device id (<c>provider:index</c>).</param>
/// <param name="Name">Device name.</param>
/// <param name="IsGpu">A GPU (else the CPU, whose memory is the system memory).</param>
/// <param name="TotalBytes">Memory the device has.</param>
/// <param name="FreeBytes">Memory this app can still allocate (other applications' and the resident models' use already taken off).</param>
/// <param name="IsExact">The driver reports the free memory (else it is this app's own tally).</param>
public sealed record DeviceMemory(string Id, string Name, bool IsGpu, long TotalBytes, long FreeBytes, bool IsExact);
