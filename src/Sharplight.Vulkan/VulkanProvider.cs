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

    public string Name => "vulkan";

    /// <summary>Enable the Khronos validation layer for instances created after this is set.</summary>
    public static bool EnableValidation { get; set; } = Environment.GetEnvironmentVariable("SHARPLIGHT_VK_VALIDATION") == "1";

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

    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        if (device.Provider != Name)
        {
            throw new ArgumentException($"Device belongs to provider '{device.Provider}'.");
        }

        return new VulkanDevice(Instance, Instance.PhysicalDevices[device.Index], device, options ?? new DeviceOptions());
    }

    public void Dispose() { }
}
