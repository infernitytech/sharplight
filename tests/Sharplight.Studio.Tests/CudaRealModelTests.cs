using Microsoft.Extensions.DependencyInjection;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The CUDA provider in Studio's headless runtime on a real small model (skipped without the model or an NVIDIA GPU with a current driver): the device
/// list offers CUDA after Vulkan, Vulkan stays the default, a load on the CUDA device chats correctly, and a CUDA provider that fails to open its
/// device hands the load to the provider behind it. <c>CUDA_VISIBLE_DEVICES</c> and <c>SHARPLIGHT_VK_DEVICES</c> pick the GPU.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class CudaRealModelTests
{
    private static readonly string s_qwen = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");

    private static ChatInput Ask(string text) => new([ChatInputMessage.FromText(ChatRole.User, text)], new SamplingSettings { Temperature = 0, MaxTokens = 24 }, false);

    private static async Task<string> ChatAsync(ILoadedModel model, string text)
    {
        var answer = "";
        await foreach (var update in model.StreamChatAsync(Ask(text), TestContext.Current.CancellationToken))
        {
            answer += update.Kind == ChatStreamKind.Content ? update.Text : "";
        }

        return answer;
    }

    private static readonly LoadSettings s_load = new() { ContextLength = 1024, BatchTokens = 128, Slots = 1 };

    [Fact]
    public async Task Qwen3_LoadsAndChatsOnCuda_WhileVulkanStaysTheDefault()
    {
        Assert.SkipUnless(File.Exists(s_qwen), "Qwen3-0.6B is not on disk");
        using var container = new ServiceCollection().AddSharplight().AddStudioProviders().BuildServiceProvider();
        using var runtime = new EngineModelRuntime(EngineServices.From(container), () => new ResidencySettings());
        var cuda = runtime.Devices.FirstOrDefault(device => device.Provider == "cuda");
        Assert.SkipWhen(cuda is null, "no usable CUDA device (no NVIDIA driver, or this package has no CUDA provider)");

        var vulkan = runtime.Devices.FirstOrDefault(device => device.Provider == "vulkan");
        if (vulkan is not null)
        {
            Assert.Equal(vulkan.Id, runtime.Devices[0].Id);
            Assert.Equal(vulkan.GpuKey, cuda.GpuKey);
            Assert.Equal("CUDA", cuda.Variant);
        }

        var model = await runtime.LoadAsync(new ModelLoadRequest("qwen-cuda", s_qwen, s_load with { DeviceId = cuda.Id }, []), null, TestContext.Current.CancellationToken);

        Assert.Equal("cuda", model.Device.Provider);
        Assert.Contains("Paris", await ChatAsync(model, "What is the capital of France?"), StringComparison.OrdinalIgnoreCase);
        Assert.Contains(runtime.GetDeviceUsage(), usage => usage.Id == cuda.Id && usage.UsedBytes > 0);
    }

    [Fact]
    public async Task ACudaProviderThatCannotOpenItsDevice_FallsBackToTheProviderBehindIt_AndChats()
    {
        Assert.SkipUnless(File.Exists(s_qwen), "Qwen3-0.6B is not on disk");
        Assert.True(TestHost.Providers.TryFind("cpu", out var cpu));
        var cpuInfo = cpu.GetDevices()[0];
        // "Vulkan" here is the CPU's device behind a GPU-shaped listing, so the test needs no GPU; "CUDA" lists the same GPU and fails on open, as with a driver that
        // initializes but cannot create a context.
        var vulkan = new FakeGpuProvider("vulkan", [FakeGpuProvider.Gpu("vulkan", ProviderFixtures.VulkanUuid)], open: _ => cpu.OpenDevice(cpuInfo));
        var cuda = new FakeGpuProvider("cuda", [FakeGpuProvider.Gpu("cuda", ProviderFixtures.CudaUuid, cudaDriver: "13.2")],
            open: _ => throw new InvalidOperationException("CUDA_ERROR_UNKNOWN: cuDevicePrimaryCtxRetain failed"));
        var engine = new EngineServices(TestHost.Services, TestHost.Loader, new ExecutionProviderCatalog([vulkan, cuda]), TestHost.Converter);
        var logs = new CapturingLogger();
        var key = DeviceCatalog.Discover(engine.Providers).Options[0].GpuKey!;
        using var runtime = new EngineModelRuntime(engine, () => new ResidencySettings(), logger: logs,
            providerChoices: () => new Dictionary<string, ProviderChoice> { [key] = ProviderChoice.Cuda });
        Assert.Equal("cuda:0", runtime.Devices[0].Id);

        var model = await runtime.LoadAsync(new ModelLoadRequest("qwen-fallback", s_qwen, s_load with { DeviceId = "cuda:0" }, []), null, TestContext.Current.CancellationToken);

        Assert.Single(cuda.Opened);
        Assert.Equal("vulkan", model.Device.Provider);
        Assert.Contains("Paris", await ChatAsync(model, "What is the capital of France?"), StringComparison.OrdinalIgnoreCase);
        var gpu = Assert.Single(runtime.Gpus);
        Assert.Equal("vulkan:0", gpu.Effective.Id);
        Assert.Contains("cuDevicePrimaryCtxRetain", gpu.Note);
        Assert.Contains(logs.Lines, line => line.Contains("falling back to Vulkan", StringComparison.Ordinal) && line.Contains("CUDA failed", StringComparison.Ordinal));

        // The next load does not try CUDA again.
        var second = await runtime.LoadAsync(new ModelLoadRequest("qwen-again", s_qwen, s_load with { DeviceId = "cuda:0" }, []), null, TestContext.Current.CancellationToken);
        Assert.Equal("vulkan", second.Device.Provider);
        Assert.Single(cuda.Opened);
    }
}
