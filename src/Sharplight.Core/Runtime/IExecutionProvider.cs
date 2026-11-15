namespace Sharplight.Runtime;

/// <summary>
/// A backend (Vulkan, CUDA, CPU, Metal, ...). Providers are cheap to construct; they only touch the
/// driver when <see cref="CheckAvailability"/> / <see cref="GetDevices"/> are called.
/// </summary>
/// <remarks>
/// <para>
/// Every provider implements the two-stage precompilation contract, so hosts precompile a model for any provider the same way:
/// </para>
/// <list type="number">
/// <item><see cref="CompilePortableAsync"/> generates the model's device code in the provider's portable format
/// (<see cref="PortableCodeFormat"/>) for a <see cref="DeviceProfile"/>, without a device or driver
/// (<see cref="PortableCodeSet"/>, stored as the <see cref="ArtifactTiers.Code"/> tier).</item>
/// <item><see cref="CompileDeviceSpecificAsync"/> has the driver of an opened device compile that portable code into its final,
/// hardware-specific form (<see cref="DeviceCodeSet"/>, stored as the <see cref="ArtifactTiers.Binary"/> tier).</item>
/// </list>
/// <para>
/// Loading then uses the best stage stored for a device on its own: device code whose descriptor matches the device and driver,
/// else portable code for its profile, else the kernels are compiled from source. Providers whose kernels need no precompilation
/// (the CPU) return empty sets from both stages. Vulkan: SPIR-V, then pipeline binaries (VK_KHR_pipeline_binary) or a pipeline
/// cache; Metal: AIR / a metallib, then an MTLBinaryArchive.
/// </para>
/// </remarks>
public interface IExecutionProvider : IDisposable
{
    /// <summary>Provider name (for example "vulkan" or "cpu").</summary>
    public string Name { get; }
    /// <summary>Detects whether the provider can run on this machine (driver/library present, ...).</summary>
    public ProviderAvailability CheckAvailability();
    /// <summary>Lists the devices; empty when the provider is unavailable.</summary>
    public IReadOnlyList<DeviceInfo> GetDevices();
    /// <summary>
    /// Opens a device for compiling and running segments; the caller disposes it.
    /// </summary>
    /// <param name="device">A device from <see cref="GetDevices"/>.</param>
    /// <param name="options">Device options, or <see langword="null"/> for the defaults.</param>
    /// <exception cref="NotSupportedException">The provider is unavailable.</exception>
    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null);

    /// <summary>
    /// The portable code format the provider generates in <see cref="CompilePortableAsync"/> (the <see cref="DeviceProfile.CodeTarget"/>
    /// of its profiles, e.g. <c>spirv_1_6</c>), or <see langword="null"/> when it has no portable stage (the CPU).
    /// </summary>
    public string? PortableCodeFormat { get; }

    /// <summary>Whether <see cref="CompilePortableAsync"/> generates code for <paramref name="profile"/> (a device of this provider in its format).</summary>
    /// <param name="profile">The device profile.</param>
    public bool CanCompileFor(DeviceProfile profile);

    /// <summary>
    /// Stage one: generates the device code of a model for <see cref="PortableCompilationRequest.Target"/> in
    /// <see cref="PortableCodeFormat"/> by running the request's workload on a compile-only device of that profile. Needs no device or
    /// driver, so it also works for devices that are not present. Code already stored with the model
    /// (<see cref="PortableCompilationRequest.Stored"/>) is reused.
    /// </summary>
    /// <param name="request">The profile and the model's compilation.</param>
    /// <param name="cancellationToken">Cancels the compilation between kernels.</param>
    /// <returns>The code; empty when the provider has no portable stage or the request has no target.</returns>
    /// <exception cref="NotSupportedException">The provider cannot compile for the request's profile (<see cref="CanCompileFor"/>).</exception>
    public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Stage two: has the driver of <paramref name="device"/> compile <paramref name="portable"/> into its hardware-specific form for
    /// exactly that device model and driver build. Needs only the portable code, not the model. Every device with the same device
    /// identity and driver can use the result.
    /// </summary>
    /// <param name="device">An opened device of this provider.</param>
    /// <param name="portable">Portable code generated for the device's own profile (<see cref="IComputeDevice.CreateProfile"/>).</param>
    /// <param name="cancellationToken">Cancels between kernels.</param>
    /// <returns>The device code; empty when the provider has no device-specific stage or <paramref name="portable"/> is empty.</returns>
    /// <exception cref="ArgumentException">The device belongs to another provider, or the portable code was generated for another profile.</exception>
    public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default);
}
