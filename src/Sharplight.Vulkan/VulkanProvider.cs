using Sharplight.Runtime;

namespace Sharplight.Vulkan;

/// <summary>
/// Vulkan compute execution provider. Requires Vulkan 1.2+ with buffer device addresses,
/// 8/16-bit storage and 64-bit integers (kernels address memory through raw pointers).
/// </summary>
/// <remarks>
/// The Vulkan instance (loader, layers, physical-device enumeration) is process-wide state: it is created on the first provider call
/// and lives until the process exits, because devices opened by any provider instance (and the pipeline caches and memory allocators
/// behind them) are shared across models and must outlive the provider objects, and because creating and destroying instances
/// repeatedly is slow and leaks driver memory on several drivers. Provider objects themselves are cheap and stateless.
/// </remarks>
/// <param name="options">Provider options, or <see langword="null"/> for the defaults.</param>
public sealed unsafe class VulkanProvider(VulkanProviderOptions? options = null) : IExecutionProvider, IOfflineCompilationProvider
{
    private static readonly Lock s_lock = new();
    private static VulkanInstance? s_instance;
    private static string? s_initError;
    private readonly bool _enableValidation = (options ?? new VulkanProviderOptions()).EnableValidation;

    /// <inheritdoc />
    public string Name => "vulkan";

    internal VulkanInstance Instance => GetOrCreateInstance(_enableValidation);

    private static VulkanInstance GetOrCreateInstance(bool enableValidation)
    {
        lock (s_lock)
        {
            if (s_instance is not null)
            {
                return s_instance;
            }

            if (s_initError is not null)
            {
                throw new InvalidOperationException(s_initError);
            }

            try
            {
                return s_instance = new VulkanInstance(enableValidation);
            }
            catch (Exception e)
            {
                s_initError = e.Message;
                throw;
            }
        }
    }

    /// <inheritdoc />
    public ProviderAvailability CheckAvailability()
    {
        try
        {
            var inst = Instance;
            var visible = DiagnosticSwitches.VisibleDevices;
            var usable = inst.PhysicalDevices.Where((p, i) => p.MissingFeatures.Count == 0 && (visible is null || visible.Contains(i))).ToList();
            if (usable.Count == 0)
            {
                return ProviderAvailability.Unavailable("No Vulkan device with the required features: " +
                    string.Join("; ", inst.PhysicalDevices.Select(p => $"{p.Name}: missing {string.Join(", ", p.MissingFeatures)}")));
            }

            return ProviderAvailability.Available($"Vulkan {inst.ApiVersion}, {usable.Count} usable device(s)");
        }
        catch (Exception e)
        {
            return ProviderAvailability.Unavailable("Vulkan loader/driver not available: " + e.Message);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Devices lacking a required feature are not listed; their index in <see cref="DeviceInfo.Index"/> still refers to the
    /// Vulkan enumeration order. <c>SHARPLIGHT_VK_DEVICES=1,2</c> lists only the devices with those enumeration indices,
    /// which pins a process to some GPUs of a multi-GPU machine.
    /// </remarks>
    public IReadOnlyList<DeviceInfo> GetDevices()
    {
        var inst = Instance;
        var visible = DiagnosticSwitches.VisibleDevices;
        var list = new List<DeviceInfo>();
        for (int i = 0; i < inst.PhysicalDevices.Count; i++)
        {
            var p = inst.PhysicalDevices[i];
            if (p.MissingFeatures.Count > 0 || (visible is not null && !visible.Contains(i)))
            {
                continue;
            }

            list.Add(p.ToDeviceInfo(i));
        }
        return list;
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException"><paramref name="device"/> was listed by another provider.</exception>
    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        if (device.Provider != Name)
        {
            throw new ArgumentException($"Device belongs to provider '{device.Provider}'.");
        }

        return new VulkanDevice(Instance, Instance.PhysicalDevices[device.Index], device, options ?? new DeviceOptions());
    }

    /// <inheritdoc />
    /// <remarks>Vulkan profiles target SPIR-V (<see cref="DeviceProfile.CodeTarget"/> <c>spirv_1_6</c>).</remarks>
    public bool CanCompileFor(DeviceProfile profile)
        => profile.Provider == Name && profile.CodeTarget == SpirvCodeGenerator.CodeTarget;

    /// <inheritdoc />
    /// <remarks>Does not load the Vulkan driver.</remarks>
    public IComputeDevice OpenProfile(DeviceProfile profile, DeviceOptions? options = null)
    {
        if (!CanCompileFor(profile))
        {
            throw new NotSupportedException($"The Vulkan provider generates {SpirvCodeGenerator.CodeTarget} code; cannot compile for {profile}.");
        }

        return new VulkanProfileDevice(profile, options ?? new DeviceOptions());
    }

    /// <inheritdoc />
    /// <remarks>Does nothing: the Vulkan instance is process-wide and lives until exit.</remarks>
    public void Dispose()
    {
    }
}
