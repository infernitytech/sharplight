namespace Sharplight.Runtime.Experts;

/// <summary>
/// Where a read's bytes go: host memory (<see cref="HostAddress"/>), or for readers that support it (<see cref="IExpertReader.SupportsDeviceDestinations"/>)
/// a device buffer of that reader's provider at <see cref="DeviceOffset"/>.
/// </summary>
/// <param name="HostAddress">The first host byte, or 0 for a device destination.</param>
/// <param name="DeviceBuffer">The provider's device buffer, or null for a host destination.</param>
/// <param name="DeviceOffset">Offset in <paramref name="DeviceBuffer"/>.</param>
public readonly record struct ExpertReadDestination(nint HostAddress, object? DeviceBuffer = null, long DeviceOffset = 0)
{
    /// <summary>A destination in host memory.</summary>
    /// <param name="address">The first byte.</param>
    public static ExpertReadDestination Host(nint address) => new(address);
}
