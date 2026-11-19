using Sharplight.Runtime;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>An execution provider with listed devices that only exists to be discovered and, when told to, to fail opening them.</summary>
internal sealed class FakeGpuProvider(string name, IReadOnlyList<DeviceInfo> devices, ProviderAvailability? availability = null, Func<DeviceInfo, IComputeDevice>? open = null)
    : IExecutionProvider
{
    public string Name => name;

    /// <summary>The devices that were asked to open.</summary>
    public List<DeviceInfo> Opened { get; } = [];

    public ProviderAvailability CheckAvailability() => availability ?? ProviderAvailability.Available("fake");

    public IReadOnlyList<DeviceInfo> GetDevices() => devices;

    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        Opened.Add(device);
        return open is null ? throw new NotSupportedException($"{name} fake devices cannot be opened") : open(device);
    }

    public string? PortableCodeFormat => null;

    public bool CanCompileFor(DeviceProfile profile) => false;

    public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default)
        => throw new NotSupportedException();

    public void Dispose()
    {
    }

    /// <summary>A GPU as <paramref name="provider"/> lists it.</summary>
    public static DeviceInfo Gpu(string provider, string uuid, string vendor = "NVIDIA", string name = "NVIDIA RTX Test", string? cudaDriver = null, int index = 0) => new()
    {
        Provider = provider,
        Index = index,
        Name = name,
        Kind = DeviceKind.DiscreteGpu,
        TotalMemory = 24L << 30,
        Vendor = vendor,
        Uuid = uuid,
        Features = new HashSet<string> { "fp16" },
        Capabilities = cudaDriver is null ? new Dictionary<string, string>() : new Dictionary<string, string> { ["cuda_driver"] = cudaDriver },
    };
}
