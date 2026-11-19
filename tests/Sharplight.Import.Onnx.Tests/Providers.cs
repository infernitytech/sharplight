using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Tests;
using Xunit;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>
/// Devices shared by the end-to-end tests (opened once, lazily): the CPU and the GPU under test (<see cref="TestGpu"/>:
/// <c>SHARPLIGHT_TEST_PROVIDER=vulkan|cuda|metal</c>, default Vulkan). Tests name them "cpu" and "gpu".
/// </summary>
internal static class Providers
{
    private static readonly Lazy<IComputeDevice> s_cpu = new(OpenCpuDevice);
    private static readonly Lazy<(IComputeDevice? Device, string Reason)> s_gpu = new(TestGpu.Open);

    public static IComputeDevice Cpu => s_cpu.Value;

    private static IComputeDevice OpenCpuDevice()
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    }

    /// <summary>The GPU under test; skips the test when it is not available.</summary>
    public static IComputeDevice GpuOrSkip()
    {
        var (d, reason) = s_gpu.Value;
        if (d is null)
        {
            Assert.Skip($"{TestGpu.ProviderName} unavailable: {reason}");
        }

        return d;
    }

    /// <summary>The device a test names: "cpu" or "gpu" (the GPU under test).</summary>
    public static IComputeDevice Get(string provider) => provider == "cpu" ? Cpu : GpuOrSkip();
}
