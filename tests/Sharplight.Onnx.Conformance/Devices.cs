using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Onnx.Conformance;

/// <summary>Devices shared by the conformance tests (opened once, lazily).</summary>
internal static class Devices
{
    private static readonly Lazy<IComputeDevice> s_cpu = new(() =>
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    private static readonly Lazy<(IComputeDevice? Device, string Reason)> s_vulkan = new(OpenVulkan);

    public static IComputeDevice Cpu => s_cpu.Value;

    /// <summary>The Vulkan device (<c>SHARPLIGHT_TEST_VK_DEVICE</c>, default 0) or null with the reason it is unavailable.</summary>
    public static (IComputeDevice? Device, string Reason) Vulkan => s_vulkan.Value;

    private static (IComputeDevice?, string) OpenVulkan()
    {
        try
        {
            var provider = new VulkanProvider();
            var availability = provider.CheckAvailability();
            if (!availability.IsAvailable)
            {
                return (null, availability.Reason);
            }

            var devices = provider.GetDevices();
            int index = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE"), out var configured) ? configured : 0;
            return devices.Count <= index ? (null, $"no Vulkan device {index}") : (provider.OpenDevice(devices[index]), "");
        }
        catch (Exception e)
        {
            return (null, e.Message);
        }
    }
}
