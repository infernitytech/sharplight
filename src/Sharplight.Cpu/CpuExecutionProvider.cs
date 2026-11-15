using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Execution provider running graphs on the host CPU (multithreaded, SIMD, dequantize-on-the-fly quantized matmul).
/// It exposes exactly one device describing the host. The provider is stateless and safe to use from any thread.
/// </summary>
public sealed class CpuExecutionProvider : IExecutionProvider
{
    /// <summary>The provider name used in <see cref="DeviceInfo.Provider"/> and for <see cref="ExecutionProviderCatalog.TryFind"/>.</summary>
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
    /// <remarks>Always <see langword="null"/>: CPU kernels are compiled ahead of time with Sharplight itself.</remarks>
    public string? PortableCodeFormat => null;

    /// <inheritdoc />
    /// <remarks>Always false: the CPU has no portable code stage.</remarks>
    public bool CanCompileFor(DeviceProfile profile) => false;

    /// <inheritdoc />
    /// <remarks>Always an empty set: there is nothing to precompile for the CPU.</remarks>
    public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Task.FromResult(PortableCodeSet.Empty(Name));
    }

    /// <inheritdoc />
    /// <remarks>Always an empty set: there is nothing to precompile for the CPU.</remarks>
    public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(portable);
        if (device.Info.Provider != Name)
        {
            throw new ArgumentException($"Device {device.Info} does not belong to the cpu provider.", nameof(device));
        }

        return Task.FromResult(DeviceCodeSet.Empty(Name, device.Info.Name));
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }

    /// <summary>
    /// The CPU provider has kernels for weights of this element type (float, the supported ggml block quantizations, FP8). Every graph op that
    /// a GPU provider lacks falls back to the CPU, so a weight type without CPU kernels cannot be loaded at all (e.g. <c>q8_k</c>, which
    /// ggml uses for activations only). Integer weights are index tables (e.g. DeepSeek-V4's hash routing table), read by gather and compare
    /// ops, which every integer type has.
    /// </summary>
    /// <param name="type">The stored element type of a weight.</param>
    public static bool SupportsWeightType(DataType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        return Quant.FormatOf(type) is not null || type.Kind is DataTypeKind.Int or DataTypeKind.Bool;
    }
}
