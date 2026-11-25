using Sharplight.Runtime;

namespace Sharplight.Distributed;

/// <summary>A device a segment worker compiles its segments on.</summary>
/// <param name="Provider">The worker's execution provider (<c>cuda</c>, <c>vulkan</c>, <c>cpu</c>, ...).</param>
/// <param name="Name">Device name as the driver reports it.</param>
/// <param name="Kind">Device class.</param>
/// <param name="TotalMemory">Device memory in bytes (system memory for the CPU).</param>
/// <param name="AvailableMemory">What the worker can still allocate on it, in bytes, when the figures were taken.</param>
/// <param name="IsFallback">Whether it is the worker's fallback device (CPU offload), not a pipeline device.</param>
public sealed record SegmentWorkerDevice(string Provider, string Name, DeviceKind Kind, long TotalMemory, long AvailableMemory, bool IsFallback);
