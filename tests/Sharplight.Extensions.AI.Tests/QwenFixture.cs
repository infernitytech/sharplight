using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>Loads Qwen3.5-0.8B once (Vulkan if available, else CPU). Tests skip when the file is missing.</summary>
public sealed class QwenFixture : IDisposable
{
    public static readonly string Path = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf");
    public SharplightChatClient? Client { get; }
    private readonly DeviceSlot? _slot;

    public QwenFixture()
    {
        if (!File.Exists(Path))
        {
            return;
        }

        GgufImporter.Register();
        var vk = new VulkanProvider();
        if (vk.CheckAvailability().IsAvailable && vk.GetDevices().Count > 0)
        {
            var devs = vk.GetDevices();
            var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs[0];
            _slot = new DeviceSlot { Device = vk.OpenDevice(dev) };
        }
        else
        {
            var cpu = new CpuExecutionProvider();
            _slot = new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
        }
        Client = SharplightChatClient.Load(Path, new LlmLoadOptions { Devices = [_slot], ContextLength = 1024, MaxBatchTokens = 256 },
            new SharplightChatClientOptions { Sampling = SamplingOptionsGreedy });
    }

    private static SamplingOptions SamplingOptionsGreedy => SamplingOptions.Greedy;

    public void Dispose()
    {
        Client?.Dispose();
        _slot?.Device.Dispose();
    }
}
