using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Execution provider running graphs on the host CPU (multithreaded, SIMD, dequantize-on-the-fly quantized matmul).
/// It exposes exactly one device describing the host. The provider is stateless and safe to use from any thread.
/// </summary>
public sealed class CpuExecutionProvider : IExecutionProvider
{
    /// <summary>The provider name used in <see cref="DeviceInfo.Provider"/> and for registry lookups.</summary>
    public const string ProviderName = "cpu";

    /// <inheritdoc />
    public string Name => ProviderName;

    /// <inheritdoc />
    /// <remarks>The CPU provider is always available.</remarks>
    public ProviderAvailability CheckAvailability() => ProviderAvailability.Available("host CPU");

    /// <inheritdoc />
    public IReadOnlyList<DeviceInfo> GetDevices() => [CpuDevice.DescribeHost()];

    /// <inheritdoc />
    /// <remarks>
    /// <see cref="DeviceOptions.Settings"/> key <c>threads</c> sets the degree of parallelism used by all CPU kernels.
    /// The value is process-wide, not per device. <see cref="DeviceOptions.MemoryBudget"/> caps the reported available memory.
    /// </remarks>
    /// <exception cref="ArgumentException"><paramref name="device"/> was not produced by this provider.</exception>
    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        if (device.Provider != Name)
        {
            throw new ArgumentException($"Device {device} does not belong to the cpu provider.");
        }

        return new CpuDevice(device, options);
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }

    /// <summary>Registers the CPU provider with <see cref="ProviderRegistry"/>.</summary>
    public static void Register() => ProviderRegistry.Register(() => new CpuExecutionProvider());
}
