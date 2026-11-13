using Sharplight.Cpu;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>Device access; tests skip when no Vulkan device is available.</summary>
internal static class Gpu
{
    private static readonly Lazy<(IComputeDevice? Dev, string Reason)> s_vk = new(() =>
    {
        try
        {
            var p = new VulkanProvider();
            var av = p.CheckAvailability();
            if (!av.IsAvailable)
            {
                return (null, av.Reason);
            }
            var devs = p.GetDevices();
            var dev = TestSwitches.DeviceIndex is { } index && index >= 0 && index < devs.Count ? devs[index]
                : devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
            if (dev is null)
            {
                return (null, "no Vulkan device");
            }

            return (p.OpenDevice(dev), "");
        }
        catch (Exception e)
        {
            return (null, e.Message);
        }
    });

    private static readonly Lazy<IComputeDevice> s_cpu = new(() =>
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    public static IComputeDevice Cpu => s_cpu.Value;
    public static IComputeDevice? Vulkan => s_vk.Value.Dev;
    public static void Require() => Assert.SkipUnless(Vulkan is not null, "No usable Vulkan device: " + s_vk.Value.Reason);
}
