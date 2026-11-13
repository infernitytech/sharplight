using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Vulkan;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>Providers shared by the end-to-end tests (opened once, lazily).</summary>
internal static class Providers
{
    private static readonly Lazy<IComputeDevice> s_cpu = new(OpenCpuDevice);
    private static readonly Lazy<(IComputeDevice? Device, string Reason)> s_vk = new(() =>
    {
        try
        {
            var p = new VulkanProvider();
            var a = p.CheckAvailability();
            if (!a.IsAvailable)
            {
                return (null, a.Reason);
            }

            var d = p.GetDevices();
            int idx = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE"), out var di) ? di : 0;
            return d.Count <= idx ? (null, "no Vulkan device " + idx) : (p.OpenDevice(d[idx]), "");
        }
        catch (Exception e)
        {
            return (null, e.Message);
        }
    });

    public static IComputeDevice Cpu => s_cpu.Value;

    private static IComputeDevice OpenCpuDevice()
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    }

    public static IComputeDevice VulkanOrSkip()
    {
        var (d, reason) = s_vk.Value;
        if (d is null)
        {
            Assert.Skip("Vulkan unavailable: " + reason);
        }

        return d;
    }

    public static IComputeDevice Get(string provider) => provider == "cpu" ? Cpu : VulkanOrSkip();
}
