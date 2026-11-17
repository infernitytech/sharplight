namespace Sharplight.Hosting;

/// <summary>Memory of one device (or memory pool) of a host.</summary>
public sealed record DeviceStatus
{
    /// <summary>Device id (<c>provider:index</c>).</summary>
    public required string Id { get; init; }

    /// <summary>Device name.</summary>
    public required string Name { get; init; }

    /// <summary>A GPU (else the CPU, whose memory is the system memory).</summary>
    public required bool IsGpu { get; init; }

    /// <summary>The device is open; closed devices report their total memory as free (<see cref="IsExact"/> false).</summary>
    public required bool IsOpen { get; init; }

    /// <summary>Memory the device has.</summary>
    public required long TotalBytes { get; init; }

    /// <summary>Memory in use (by every process when <see cref="IsExact"/>, else by this one).</summary>
    public required long UsedBytes { get; init; }

    /// <summary>Memory this process can still allocate.</summary>
    public required long FreeBytes { get; init; }

    /// <summary>What a new model may use: <see cref="FreeBytes"/> minus the reserve (never negative).</summary>
    public required long AvailableBytes { get; init; }

    /// <summary>Memory held by this host's loaded models (their resident bytes, spread evenly over their devices).</summary>
    public required long ModelBytes { get; init; }

    /// <summary>The driver reports system-wide usage (else the figures are this process's own tally).</summary>
    public required bool IsExact { get; init; }

    /// <summary>Names of the loaded models with memory on this device.</summary>
    public required IReadOnlyList<string> Models { get; init; }
}
