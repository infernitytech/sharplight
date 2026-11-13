namespace Sharplight.Runtime;

/// <summary>Memory figures of an opened device, in bytes.</summary>
/// <param name="Total">Physical capacity (device-local heap; system RAM for CPU).</param>
/// <param name="Budget">Most this process may use in total: the smaller of the configured cap and what the driver reports
/// as obtainable (which shrinks when other applications use the device).</param>
/// <param name="Usage">What this process uses now, as reported by the driver when <paramref name="IsExact"/>, else our own tally.</param>
/// <param name="UsedByUs">Bytes Sharplight allocated on this device instance (weights, state, activations, staging).</param>
/// <param name="IsExact">True when <paramref name="Budget"/>/<paramref name="Usage"/> come from the driver.</param>
public readonly record struct DeviceMemoryInfo(long Total, long Budget, long Usage, long UsedByUs, bool IsExact)
{
    /// <summary>Bytes this process can still allocate.</summary>
    public long Available => Math.Max(0, Budget - Usage);
    public override string ToString() => $"total {Total / 1048576} MiB, budget {Budget / 1048576} MiB, used {Usage / 1048576} MiB (ours {UsedByUs / 1048576} MiB), available {Available / 1048576} MiB{(IsExact ? "" : " (estimate)")}";
}
