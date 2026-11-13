using Sharplight.Runtime;

namespace Sharplight.Vulkan;

/// <summary>
/// Vulkan compute execution provider. Requires Vulkan 1.2+ with buffer device addresses,
/// 8/16-bit storage and 64-bit integers (kernels address memory through raw pointers).
/// </summary>
public sealed unsafe class VulkanProvider : IExecutionProvider
{
    private static readonly Lock s_lock = new();
    private static VulkanInstance? s_instance;
    private static string? s_initError;

    /// <inheritdoc />
    public string Name => "vulkan";

    /// <summary>
    /// Enables the Khronos validation layer for the Vulkan instance, which is created on first use: set it before the
    /// first provider call. Process-wide; defaults to on when <c>SHARPLIGHT_VK_VALIDATION=1</c>.
    /// </summary>
    public static bool EnableValidation { get; set; } = DiagnosticSwitches.Validation;

    internal static VulkanInstance Instance
    {
        get
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
                    return s_instance = new VulkanInstance(EnableValidation);
                }
                catch (Exception e)
                {
                    s_initError = e.Message;
                    throw;
                }
            }
        }
    }

    /// <inheritdoc />
    public ProviderAvailability CheckAvailability()
    {
        try
        {
            var inst = Instance;
            var usable = inst.PhysicalDevices.Where(p => p.MissingFeatures.Count == 0).ToList();
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
    /// <remarks>Devices lacking a required feature are not listed; their index in <see cref="DeviceInfo.Index"/> still refers to the Vulkan enumeration order.</remarks>
    public IReadOnlyList<DeviceInfo> GetDevices()
    {
        var inst = Instance;
        var list = new List<DeviceInfo>();
        for (int i = 0; i < inst.PhysicalDevices.Count; i++)
        {
            var p = inst.PhysicalDevices[i];
            if (p.MissingFeatures.Count > 0)
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
    /// <remarks>Does nothing: the Vulkan instance is process-wide and lives until exit.</remarks>
    public void Dispose()
    {
    }
}
