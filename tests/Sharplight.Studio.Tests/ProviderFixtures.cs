using Sharplight.Runtime;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>An NVIDIA GPU that Vulkan and CUDA both list (fake providers), wired into the fake runtime the way the engine runtime exposes it.</summary>
internal static class ProviderFixtures
{
    // Vulkan reports the device UUID as hex, CUDA as a dashed GUID: the same GPU.
    public const string VulkanUuid = "0123456789ABCDEF0123456789ABCDEF";
    public const string CudaUuid = "01234567-89ab-cdef-0123-456789abcdef";

    /// <summary>The devices of fake Vulkan, CUDA and CPU providers for one NVIDIA GPU.</summary>
    public static DeviceDiscovery Discover(string? cudaDriver = "13.2", ProviderAvailability? cudaAvailability = null, bool cudaListed = true, string vendor = "NVIDIA")
    {
        var vulkan = new FakeGpuProvider("vulkan", [FakeGpuProvider.Gpu("vulkan", VulkanUuid, vendor)]);
        var cuda = new FakeGpuProvider("cuda", cudaListed ? [FakeGpuProvider.Gpu("cuda", CudaUuid, "NVIDIA", cudaDriver: cudaDriver)] : [], cudaAvailability);
        var cpu = new FakeGpuProvider("cpu", [new DeviceInfo { Provider = "cpu", Index = 0, Name = "CPU", Kind = DeviceKind.Cpu, TotalMemory = 32L << 30 }]);
        return DeviceCatalog.Discover(new ExecutionProviderCatalog([vulkan, cuda, cpu]));
    }

    /// <summary>Makes the fake runtime of <paramref name="services"/> list <paramref name="discovery"/> through a selector that follows the settings.</summary>
    public static void Wire(TestServices services, DeviceDiscovery discovery, IProviderPerformanceHints? hints = null)
    {
        var selector = new DeviceProviderSelector(discovery, () => services.Settings.Current.DeviceProviders, hints);
        services.Runtime.DevicesSource = () => selector.Devices;
        services.Runtime.GpuSource = () => selector.Gpus;
    }
}
