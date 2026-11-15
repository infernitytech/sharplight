namespace Sharplight.Runtime;

/// <summary>
/// Stage two of precompilation (<see cref="IExecutionProvider.CompileDeviceSpecificAsync"/>): the driver's compiled form of a
/// <see cref="PortableCodeSet"/> for exactly one device model and driver build (Vulkan: pipeline binaries or a pipeline cache, later
/// Metal binary archives). Stored as the <see cref="ArtifactTiers.Binary"/> tier; its descriptor carries the device identity, driver
/// version and kernel ABI, and a loader only consults it for a device that matches all of them. Even then the provider validates
/// every entry before handing it to the driver and falls back to the portable code when the driver rejects it.
/// </summary>
public sealed class DeviceCodeSet : CodeSet
{
    /// <summary>Creates a device code set.</summary>
    /// <param name="provider">The provider that produced the code.</param>
    /// <param name="device">Display name of the device it was produced on.</param>
    /// <param name="descriptor">Binary-tier descriptor (provider, tier <see cref="ArtifactTiers.Binary"/>, kind, device identity, driver); may
    /// be empty when <paramref name="entries"/> is.</param>
    /// <param name="entries">The driver binaries by key.</param>
    /// <exception cref="ArgumentException">The set has entries but the descriptor is not a binary-tier descriptor of the provider.</exception>
    public DeviceCodeSet(string provider, string device, IReadOnlyDictionary<string, string> descriptor, IReadOnlyDictionary<string, ReadOnlyMemory<byte>> entries)
        : base(provider, ArtifactTiers.Binary, descriptor, entries, 0)
    {
        Device = device ?? "";
    }

    /// <summary>Display name of the device the code was produced on (every device with the same descriptor can use it).</summary>
    public string Device { get; }

    /// <summary>An empty set: the provider has no device-specific stage (CPU) or there was no portable code to compile.</summary>
    /// <param name="provider">The provider name.</param>
    /// <param name="device">Display name of the device.</param>
    public static DeviceCodeSet Empty(string provider, string device)
        => new(provider, device, new Dictionary<string, string>(), new Dictionary<string, ReadOnlyMemory<byte>>());
}
