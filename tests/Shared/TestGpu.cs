using Sharplight.Cuda;
using Sharplight.Metal;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Tests;

/// <summary>
/// The GPU provider under test, shared (as a linked source file) by the test projects that run models on a GPU device.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_TEST_PROVIDER=vulkan|cuda|metal</c>: the provider (default <c>vulkan</c>), the same switch as the kernel suite
/// (<c>tests/Sharplight.Vulkan.Tests/TestSwitches.cs</c>). Tests of Vulkan internals skip on the other providers.</item>
/// <item><c>SHARPLIGHT_TEST_DEVICE=&lt;index&gt;</c> (or the older <c>SHARPLIGHT_TEST_VK_DEVICE</c>): the device index within the
/// provider; the default is the first discrete GPU, else the first device. <c>SHARPLIGHT_VK_DEVICES</c> and
/// <c>CUDA_VISIBLE_DEVICES</c> restrict what the providers list.</item>
/// </list>
/// </summary>
internal static class TestGpu
{
    /// <summary>The name of the provider under test (<c>vulkan</c>, <c>cuda</c> or <c>metal</c>).</summary>
    public static string ProviderName { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_PROVIDER") is { Length: > 0 } name
        ? name.ToLowerInvariant()
        : "vulkan";

    /// <summary>True when the provider under test is Vulkan (tests of Vulkan internals run only then).</summary>
    public static bool IsVulkan => ProviderName == "vulkan";

    /// <summary>A new instance of the provider under test.</summary>
    public static IExecutionProvider CreateProvider() => ProviderName switch
    {
        CudaProvider.ProviderName => new CudaProvider(),
        MetalProvider.ProviderName => new MetalProvider(),
        _ => new VulkanProvider(),
    };

    /// <summary>
    /// Opens the device under test, or returns null with the reason it is not available. Callers open it once per process and share
    /// it: drivers refuse new contexts after a few dozen live ones.
    /// </summary>
    public static (IComputeDevice? Device, string Reason) Open()
    {
        try
        {
            var provider = CreateProvider();
            var availability = provider.CheckAvailability();
            if (!availability.IsAvailable)
            {
                return (null, $"{provider.Name}: {availability.Reason}");
            }

            var devices = provider.GetDevices();
            var device = DeviceIndex is { } index
                ? (index >= 0 && index < devices.Count ? devices[index] : null)
                : devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices.FirstOrDefault();
            if (device is null)
            {
                return (null, $"no {provider.Name} device{(DeviceIndex is { } i ? " " + i : "")}");
            }

            try
            {
                return (provider.OpenDevice(device), "");
            }
            catch (Exception error)
            {
                return (null, $"{OpenFailedPrefix}{device} ({error.Message}). The suite opens one device per process; if this fails, the driver "
                    + "has no free context for this process (another component opened devices without disposing them) or the GPU is unusable "
                    + "(check nvidia-smi, SHARPLIGHT_VK_DEVICES and CUDA_VISIBLE_DEVICES).");
            }
        }
        catch (Exception error)
        {
            return (null, $"{ProviderName}: {error.Message}");
        }
    }

    /// <summary>The start of <see cref="Open"/>'s reason when a listed device failed to open (an error, not a missing GPU).</summary>
    public const string OpenFailedPrefix = "opening the test device failed: ";

    private static int? DeviceIndex
        => int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_DEVICE") ?? Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE"), out int index)
            ? index
            : null;
}
