using System.Runtime.InteropServices;
using System.Text;
using Vortice.Vulkan;
using static Vortice.Vulkan.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>Process-wide Vulkan instance and physical device inventory.</summary>
internal sealed unsafe class VulkanInstance
{
    public VkInstance Handle { get; }
    public VkInstanceApi Api { get; }
    public IReadOnlyList<PhysicalDeviceInfo> PhysicalDevices { get; }
    public VkVersion ApiVersion { get; }

    public VulkanInstance(bool validation)
    {
        // null everywhere but macOS, where the bundled MoltenVK (or SHARPLIGHT_VULKAN_LIBRARY) is tried first.
        vkInitialize(VulkanLibrary.Resolve()).CheckResult();
        uint ver = 0;
        vkEnumerateInstanceVersion(&ver);
        ApiVersion = new VkVersion(ver);
        if (ApiVersion < VkVersion.Version_1_2)
        {
            throw new NotSupportedException($"Vulkan {ApiVersion} < 1.2");
        }

        VkUtf8ReadOnlyString appName = "Sharplight"u8;
        var app = new VkApplicationInfo { pApplicationName = appName, apiVersion = VkVersion.Version_1_3, engineVersion = new VkVersion(0, 1, 0), pEngineName = appName };
        var layers = new List<nint>();
        if (validation)
        {
            layers.Add(Marshal.StringToHGlobalAnsi("VK_LAYER_KHRONOS_validation"));
        }
        // Allow portability drivers (MoltenVK) to be enumerated.
        var exts = new List<nint>();
        uint extCount = 0;
        vkEnumerateInstanceExtensionProperties(&extCount, null);
        var extProps = new VkExtensionProperties[extCount];
        fixed (VkExtensionProperties* ep = extProps)
        {
            vkEnumerateInstanceExtensionProperties(&extCount, ep);
        }

        bool portability = extProps.Any(e => Name(e) == "VK_KHR_portability_enumeration");
        if (portability)
        {
            exts.Add(Marshal.StringToHGlobalAnsi("VK_KHR_portability_enumeration"));
        }

        try
        {
            fixed (nint* pl = layers.ToArray())
            fixed (nint* pe = exts.ToArray())
            {
                var ci = new VkInstanceCreateInfo
                {
                    pApplicationInfo = &app,
                    enabledLayerCount = (uint)layers.Count,
                    ppEnabledLayerNames = (byte**)pl,
                    enabledExtensionCount = (uint)exts.Count,
                    ppEnabledExtensionNames = (byte**)pe,
                    flags = portability ? VkInstanceCreateFlags.EnumeratePortabilityKHR : 0,
                };
                VkInstance inst;
                vkCreateInstance(&ci, null, &inst).CheckResult();
                Handle = inst;
            }
        }
        finally
        {
            foreach (var p in layers.Concat(exts))
            {
                Marshal.FreeHGlobal(p);
            }
        }
        Api = new VkInstanceApi(Handle);

        uint n = 0;
        Api.vkEnumeratePhysicalDevices(&n, null);
        var pds = new VkPhysicalDevice[n];
        fixed (VkPhysicalDevice* pp = pds)
        {
            Api.vkEnumeratePhysicalDevices(&n, pp);
        }

        PhysicalDevices = [.. pds.Select(pd => new PhysicalDeviceInfo(this, pd))];
    }

    internal static string Name(VkExtensionProperties e) => Encoding.UTF8.GetString(MemoryMarshal.CreateReadOnlySpanFromNullTerminated(e.extensionName));
}
