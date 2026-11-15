namespace Sharplight.Runtime;

/// <summary>Stands in for a provider whose factory threw (see <see cref="ProviderRegistry.Discover"/>): lists no devices and compiles nothing.</summary>
/// <param name="name">Name to report.</param>
/// <param name="reason">Why the provider is unavailable.</param>
internal sealed class UnavailableProvider(string name, string reason) : IExecutionProvider
{
    public string Name => name;

    public string? PortableCodeFormat => null;

    public ProviderAvailability CheckAvailability() => ProviderAvailability.Unavailable(reason);

    public IReadOnlyList<DeviceInfo> GetDevices() => [];

    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null) => throw new NotSupportedException(reason);

    public bool CanCompileFor(DeviceProfile profile) => false;

    public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default)
        => Task.FromResult(PortableCodeSet.Empty(name));

    public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default)
        => Task.FromResult(DeviceCodeSet.Empty(name, device.Info.Name));

    public void Dispose()
    {
    }
}
