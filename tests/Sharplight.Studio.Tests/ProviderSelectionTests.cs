using Microsoft.Extensions.Logging.Abstractions;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Settings;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The per-GPU provider choice (Automatic, Vulkan, CUDA): which entry a GPU runs on, the hook for per-device data, the notes when CUDA cannot
/// be used, the fallback to Vulkan, and the Engine settings view model that edits it.
/// </summary>
public sealed class ProviderSelectionTests
{
    private static DeviceDiscovery Discover(string? cudaDriver = "13.2", ProviderAvailability? cudaAvailability = null, bool cudaListed = true, string vendor = "NVIDIA")
        => ProviderFixtures.Discover(cudaDriver, cudaAvailability, cudaListed, vendor);

    private sealed class FasterCuda : IProviderPerformanceHints
    {
        public string? FasterProvider(IReadOnlyList<ComputeDeviceOption> gpu) => gpu.Any(entry => entry.Provider == "cuda") ? "cuda" : null;
    }

    [Fact]
    public void TheSameGpuThroughTwoProviders_IsOnePhysicalGpuWithTwoLabelledEntries()
    {
        var selector = new DeviceProviderSelector(Discover());

        var gpu = Assert.Single(selector.Gpus);
        Assert.Equal(["vulkan", "cuda"], gpu.Entries.Select(entry => entry.Provider));
        Assert.Equal(["NVIDIA RTX Test (Vulkan) · 24.0 GB", "NVIDIA RTX Test (CUDA) · 24.0 GB"], gpu.Entries.Select(entry => entry.Display));
        Assert.Equal(["vulkan:0", "cuda:0", "cpu:0"], selector.Devices.Select(device => device.Id));
        Assert.True(gpu.HasChoice);
        Assert.Null(gpu.Note);
    }

    [Fact]
    public void Automatic_IsVulkan_UntilTheHookKnowsBetter()
    {
        Assert.Equal("vulkan:0", Assert.Single(new DeviceProviderSelector(Discover()).Gpus).Effective.Id);

        var selector = new DeviceProviderSelector(Discover(), hints: new FasterCuda());
        Assert.Equal("cuda:0", Assert.Single(selector.Gpus).Effective.Id);
        Assert.Equal("cuda:0", selector.Devices[0].Id);
    }

    [Fact]
    public void AnExplicitChoice_BeatsTheHook_AndOrdersTheDeviceList()
    {
        var key = new DeviceProviderSelector(Discover()).Gpus[0].Key;
        var choices = new Dictionary<string, ProviderChoice> { [key] = ProviderChoice.Cuda };
        var selector = new DeviceProviderSelector(Discover(), () => choices);
        Assert.Equal(["cuda:0", "vulkan:0", "cpu:0"], selector.Devices.Select(device => device.Id));

        choices[key] = ProviderChoice.Vulkan;
        var fastCuda = new DeviceProviderSelector(Discover(), () => choices, new FasterCuda());
        Assert.Equal("vulkan:0", fastCuda.Gpus[0].Effective.Id);
    }

    [Fact]
    public void ADriverOlderThanTheShippedCompiler_FallsBackToVulkanWithANote()
    {
        var discovery = Discover(cudaDriver: "12.8");
        var key = new DeviceProviderSelector(discovery).Gpus[0].Key;
        var selector = new DeviceProviderSelector(discovery, () => new Dictionary<string, ProviderChoice> { [key] = ProviderChoice.Cuda });

        var gpu = selector.Gpus[0];
        Assert.Equal("vulkan:0", gpu.Effective.Id);
        Assert.Contains("12.8", gpu.Note);
        Assert.Contains("R580", gpu.Note);
        Assert.Contains("Vulkan is used", gpu.Note);
        Assert.Single(selector.Notes);
        Assert.False(selector.IsUsable(gpu.Entries.Single(entry => entry.Provider == "cuda")));

        // A load that names the CUDA device runs on Vulkan.
        Assert.Equal("vulkan:0", selector.Usable(selector.Devices.Single(device => device.Id == "cuda:0")).Id);
    }

    [Fact]
    public void CudaUnavailable_OnAnNvidiaGpu_IsNoted_ButNotOnOtherVendors()
    {
        var unavailable = ProviderAvailability.Unavailable("libcuda.so.1 was not found");
        var nvidia = new DeviceProviderSelector(Discover(cudaAvailability: unavailable, cudaListed: false));
        var gpu = Assert.Single(nvidia.Gpus);
        Assert.False(gpu.HasChoice);
        Assert.Equal("vulkan:0", gpu.Effective.Id);
        Assert.Contains("libcuda.so.1 was not found", gpu.Note);

        var amd = new DeviceProviderSelector(Discover(cudaAvailability: unavailable, cudaListed: false, vendor: "AMD"));
        Assert.Null(Assert.Single(amd.Gpus).Note);
    }

    [Fact]
    public void AFailedProvider_HandsTheGpuToVulkan_AndStaysSkipped()
    {
        var discovery = Discover();
        var key = new DeviceProviderSelector(discovery).Gpus[0].Key;
        var selector = new DeviceProviderSelector(discovery, () => new Dictionary<string, ProviderChoice> { [key] = ProviderChoice.Cuda }, logger: NullLogger.Instance);
        var cuda = selector.Devices[0];
        Assert.Equal("cuda:0", cuda.Id);

        var fallback = selector.Failover(cuda, new InvalidOperationException("CUDA_ERROR_UNSUPPORTED_PTX_VERSION"));

        Assert.Equal("vulkan:0", fallback?.Id);
        Assert.Equal("vulkan:0", selector.Gpus[0].Effective.Id);
        Assert.Contains("CUDA_ERROR_UNSUPPORTED_PTX_VERSION", selector.Gpus[0].Note);
        // Nothing to fall back to from Vulkan, the CPU, a cancelled load or a full GPU.
        Assert.Null(selector.Failover(fallback!, new InvalidOperationException("boom")));
        Assert.Null(selector.Failover(selector.Devices.Single(device => !device.IsGpu), new InvalidOperationException("boom")));
        Assert.Null(new DeviceProviderSelector(discovery).Failover(cuda, new OperationCanceledException()));
        Assert.Null(new DeviceProviderSelector(discovery).Failover(cuda, new OutOfMemoryException()));
    }

    private static TestServices Settings(out FakeRuntime runtime, DeviceDiscovery discovery, IProviderPerformanceHints? hints = null)
    {
        var services = new TestServices();
        ProviderFixtures.Wire(services, discovery, hints);
        runtime = services.Runtime;
        return services;
    }

    [Fact]
    public void PickingCuda_IsSavedPerGpu_AndTheDefaultDeviceFollows()
    {
        using var services = Settings(out var runtime, Discover());
        var page = new SettingsPageViewModel(services.Services);
        Assert.True(page.HasGpus);
        var row = Assert.Single(page.GpuProviders);
        Assert.Equal([ProviderChoice.Automatic, ProviderChoice.Vulkan, ProviderChoice.Cuda], row.Choices.Select(item => item.Value));
        Assert.Equal(ProviderChoice.Automatic, row.Selected?.Value);
        Assert.Equal("Runs on Vulkan (automatic)", row.InUse);
        Assert.Equal("vulkan:0", page.DefaultDevice?.Id);

        row.Selected = ProviderChoiceItem.Cuda;

        Assert.Equal(ProviderChoice.Cuda, Assert.Single(services.Settings.Current.DeviceProviders).Value);
        Assert.Equal(row.Key, Assert.Single(services.Settings.Current.DeviceProviders).Key);
        Assert.Equal("Runs on CUDA", row.InUse);
        Assert.Equal("cuda:0", page.DefaultDevice?.Id);
        Assert.Equal("cuda:0", services.Settings.Current.DefaultLoad.DeviceId);
        Assert.Equal("cuda:0", runtime.Devices[0].Id);

        row.Selected = ProviderChoiceItem.Automatic;

        Assert.Empty(services.Settings.Current.DeviceProviders);
        Assert.Equal("vulkan:0", services.Settings.Current.DefaultLoad.DeviceId);
        Assert.Equal("Runs on Vulkan (automatic)", row.InUse);
    }

    [Fact]
    public void ADefaultDeviceOnAnotherGpu_IsLeftAlone_WhenAProviderIsPicked()
    {
        using var services = Settings(out _, Discover());
        var page = new SettingsPageViewModel(services.Services);
        page.DefaultDevice = page.Devices.Single(device => !device.IsGpu);

        page.GpuProviders[0].Selected = ProviderChoiceItem.Cuda;

        Assert.Equal("cpu:0", services.Settings.Current.DefaultLoad.DeviceId);
    }

    [Fact]
    public void ACudaNote_ShowsOnTheGpuRow_AndHidesThePicker_WhenThereIsNothingToPick()
    {
        using var services = Settings(out _, Discover(cudaAvailability: ProviderAvailability.Unavailable("no NVIDIA driver"), cudaListed: false));
        var page = new SettingsPageViewModel(services.Services);

        var row = Assert.Single(page.GpuProviders);
        Assert.False(row.HasChoice);
        Assert.True(row.HasNote);
        Assert.Contains("no NVIDIA driver", row.Note);
        Assert.Equal("Runs on Vulkan (automatic)", row.InUse);
    }

    [Fact]
    public void WithoutGpus_ThereIsNoComputeDevicesCard()
    {
        using var services = new TestServices();
        Assert.False(new SettingsPageViewModel(services.Services).HasGpus);
    }
}
