using Sharplight.Cpu;
using Sharplight.Cuda;
using Sharplight.Metal;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Device access: the GPU under test is a Vulkan device, or a Metal or CUDA device with <c>SHARPLIGHT_TEST_PROVIDER=metal|cuda</c>; tests skip when it
/// is not available.
/// </summary>
internal static class Gpu
{
    private static readonly Lazy<(IComputeDevice? Dev, string Reason)> s_gpu = new(() =>
    {
        try
        {
            IExecutionProvider p = TestSwitches.Provider switch
            {
                MetalProvider.ProviderName => new MetalProvider(),
                CudaProvider.ProviderName => new CudaProvider(),
                _ => new VulkanProvider(),
            };
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
                return (null, $"no {p.Name} device");
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

    /// <summary>The GPU under test, or null when unavailable.</summary>
    public static IComputeDevice? Device => s_gpu.Value.Dev;

    /// <summary>
    /// The compute units (SMs) the GPU under test reports (capability <c>compute_units</c>), or 0 when unknown: the lowering sizes
    /// waves and stream-K launches by them.
    /// </summary>
    public static int ComputeUnits
        => Device?.Info.Capabilities.TryGetValue("compute_units", out var text) is true && int.TryParse(text, System.Globalization.CultureInfo.InvariantCulture, out int units)
            ? units : 0;

    public static void Require() => Assert.SkipUnless(Device is not null, $"No usable {TestSwitches.Provider} device: " + s_gpu.Value.Reason);

    /// <summary>Requires the Vulkan provider under test (tests of Vulkan internals: pipeline caches, profile devices, segment statistics).</summary>
    public static void RequireVulkan()
    {
        Assert.SkipUnless(TestSwitches.Provider == "vulkan", "Tests Vulkan internals; the provider under test is " + TestSwitches.Provider);
        Require();
    }

    /// <summary>Requires the CUDA provider under test (tests of properties of its hand-written kernels, such as stream-K's summation order).</summary>
    public static void RequireCuda()
    {
        Assert.SkipUnless(TestSwitches.Provider == "cuda", "Tests CUDA kernels; the provider under test is " + TestSwitches.Provider);
        Require();
    }
}
