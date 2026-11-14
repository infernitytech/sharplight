namespace Sharplight.Studio.Services.Runtime;

/// <summary>Memory use of one opened device.</summary>
/// <param name="Id">Device id (<c>provider:index</c>).</param>
/// <param name="Name">Device name.</param>
/// <param name="UsedBytes">Memory in use (by every process when <paramref name="IsExact"/>, else by this app).</param>
/// <param name="TotalBytes">Memory the device offers.</param>
/// <param name="IsExact">The driver reports system-wide usage.</param>
internal sealed record DeviceUsage(string Id, string Name, long UsedBytes, long TotalBytes, bool IsExact);
