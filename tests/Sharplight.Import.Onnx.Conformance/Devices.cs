using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Tests;

namespace Sharplight.Import.Onnx.Conformance;

/// <summary>Devices shared by the conformance tests (opened once, lazily).</summary>
internal static class Devices
{
    private static readonly Lazy<IComputeDevice> s_cpu = new(() =>
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    private static readonly Lazy<(IComputeDevice? Device, string Reason)> s_gpu = new(TestGpu.Open);

    public static IComputeDevice Cpu => s_cpu.Value;

    /// <summary>
    /// The GPU under test (<see cref="TestGpu"/>: <c>SHARPLIGHT_TEST_PROVIDER=vulkan|cuda|metal</c>, default Vulkan;
    /// <c>SHARPLIGHT_TEST_DEVICE</c> selects the index) or null with the reason it is unavailable.
    /// </summary>
    public static (IComputeDevice? Device, string Reason) Gpu => s_gpu.Value;

    /// <summary>The name of the GPU provider under test, as the skip list and the report name it.</summary>
    public static string GpuProvider => TestGpu.ProviderName;
}
