using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Monitoring;

/// <summary>Resource use at one point in time.</summary>
/// <param name="Timestamp">When it was taken.</param>
/// <param name="CpuPercent">CPU time of this process since the previous sample, in percent of all cores.</param>
/// <param name="ProcessMemoryBytes">Working set of this process.</param>
/// <param name="SystemMemoryBytes">Physical memory of the machine.</param>
/// <param name="Devices">Memory of the opened devices.</param>
/// <param name="EngineBusyPercent">Share of wall time the loaded models' schedulers spent in forward passes (summed over models).</param>
/// <param name="ActiveSequences">Sequences generating right now.</param>
/// <param name="QueueDepth">Requests waiting for a slot.</param>
/// <param name="TokensPerSecond">Tokens generated per second over the last seconds (all models).</param>
internal sealed record SystemSample(
    DateTimeOffset Timestamp,
    double CpuPercent,
    long ProcessMemoryBytes,
    long SystemMemoryBytes,
    IReadOnlyList<DeviceUsage> Devices,
    double EngineBusyPercent,
    int ActiveSequences,
    int QueueDepth,
    double TokensPerSecond);
