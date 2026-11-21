using Sharplight.Cpu;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Tearing down a mixture-of-experts model loaded partly offloaded to the CPU (split decode passes, streamed experts, host-mapped weights)
/// on each GPU provider, and a small model on the CPU provider, the way a server stops: after decoding, while an unload is still freeing the
/// model, and while a request is decoding. Every case must return the device memory and leave the process able to exit (skipped when the
/// model file or the provider is missing).
/// </summary>
[Collection("RealModel")]
public sealed class OffloadTeardownRealTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    /// <summary>Offloaded on the GPUs.</summary>
    private static readonly string MoePath = Path.Combine(Home, "models/gguf/OLMoE-1B-7B-0924-Instruct/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
    /// <summary>On the CPU provider (small, so that the host admits it beside the other processes of a shared machine).</summary>
    private static readonly string CpuPath = Path.Combine(Home, "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
    private const string Name = "m";
    private const long MiB = 1L << 20;
    /// <summary>A third of the model's 4 GiB: most experts run on the CPU.</summary>
    private const long GpuBudget = 1400 * MiB;
    /// <summary>Upper bound of the staging buffers a device keeps for streamed weights after the streamed model is gone.</summary>
    private const long DeviceStreamingStaging = 128 * MiB;

    /// <summary>How the host is torn down after the model decoded.</summary>
    public enum Teardown
    {
        /// <summary>Unload, then dispose the host.</summary>
        UnloadThenDispose,
        /// <summary>Start an unload and dispose the host at once (the process stops while the model is still unloading).</summary>
        DisposeDuringUnload,
        /// <summary>Dispose the host while a request is decoding.</summary>
        DisposeDuringDecode,
    }

    [Theory]
    [InlineData("cpu", Teardown.UnloadThenDispose)]
    [InlineData("cpu", Teardown.DisposeDuringUnload)]
    [InlineData("cpu", Teardown.DisposeDuringDecode)]
    [InlineData("vulkan", Teardown.UnloadThenDispose)]
    [InlineData("vulkan", Teardown.DisposeDuringUnload)]
    [InlineData("vulkan", Teardown.DisposeDuringDecode)]
    [InlineData("cuda", Teardown.UnloadThenDispose)]
    [InlineData("cuda", Teardown.DisposeDuringUnload)]
    [InlineData("cuda", Teardown.DisposeDuringDecode)]
    public async Task Model_TearsDownCleanly(string providerName, Teardown teardown)
    {
        // On a GPU the model is offloaded to the CPU; on the CPU provider all its layers (and their native memory) are there.
        bool offloaded = providerName != "cpu";
        string path = offloaded ? MoePath : CpuPath;
        Assert.SkipUnless(File.Exists(path), "model not found");
        var cpuProvider = new CpuExecutionProvider();
        using var cpu = cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]);
        // Disposed last: the provider after the devices, the devices after the models (the order of a stopping process).
        using var cuda = providerName == CudaProvider.ProviderName ? new CudaProvider() : null;
        using var pool = new DevicePool();
        var device = offloaded ? OpenGpu(pool, cuda ?? TestHost.Providers.Discover().FirstOrDefault(x => x.Provider.Name == providerName)?.Provider, providerName) : cpu;
        var options = new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Fallback = offloaded ? new DeviceSlot { Device = cpu } : null,
            ContextLength = 1024,
            MaxBatchTokens = 128,
            Log = Console.Error.WriteLine,
        };
        var host = new ModelHost(new ModelHostOptions { ReserveBytes = 0, Log = Console.Error.WriteLine });
        try
        {
            host.Register(new ModelRegistration { Name = Name, Loader = new LlmChatModelLoader(TestHost.Loader, Name, path, options, new ChatModelOptions { Slots = 2 }) });
            var reply = await GenerateAsync(host, 8);
            Assert.NotEmpty(reply);
            if (offloaded)
            {
                Assert.InRange(device.GetMemoryInfo().UsedByUs, 1, GpuBudget);
            }

            switch (teardown)
            {
                case Teardown.UnloadThenDispose:
                    Assert.True(await host.UnloadAsync(Name));
                    host.Dispose();
                    break;
                case Teardown.DisposeDuringUnload:
                    var unload = host.UnloadAsync(Name);
                    host.Dispose();
                    await unload.WaitAsync(TimeSpan.FromMinutes(1));
                    break;
                case Teardown.DisposeDuringDecode:
                    var running = GenerateAsync(host, 200);
                    await WaitForDecodeAsync(host);
                    host.Dispose();
                    // The request is cut off: it ends early (cancelled) or fails, but it ends.
                    await Task.WhenAny(running).WaitAsync(TimeSpan.FromMinutes(1));
                    break;
            }

            Assert.Equal(ModelState.Unloaded, host.Get(Name)!.State);
            if (offloaded)
            {
                // What stays is the device's streaming staging, which it keeps for the next streamed model until it is disposed.
                Assert.InRange(device.GetMemoryInfo().UsedByUs, 0, DeviceStreamingStaging);
            }
        }
        finally
        {
            host.Dispose();
        }
    }

    /// <summary>Opens the first device of <paramref name="provider"/> (the default catalog has no CUDA provider: the test creates it).</summary>
    private static IComputeDevice OpenGpu(DevicePool pool, IExecutionProvider? provider, string providerName)
    {
        Assert.SkipUnless(provider is not null && provider.CheckAvailability().IsAvailable && provider.GetDevices().Count > 0, $"no {providerName} device");
        return pool.Open(provider, provider.GetDevices()[0], new DeviceOptions { MemoryBudget = GpuBudget });
    }

    private static async Task<string> GenerateAsync(ModelHost host, int tokens)
    {
        using var lease = await host.AcquireAsync(Name);
        var service = ((IChatModel)lease.Model).Service;
        var request = new ChatRequest
        {
            Messages = [new ChatTurn("user", "Count from one to one hundred in words.")],
            Sampling = SamplingParams.GreedyDefault,
            MaxTokens = tokens,
            IgnoreEos = true,
            EnableThinking = false,
        };
        var result = await service.StartChat(request).CompleteAsync();
        return result.Choices[0].Content;
    }

    /// <summary>Waits until the host's model is decoding a request (bounded).</summary>
    private static async Task WaitForDecodeAsync(ModelHost host)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromMinutes(1);
        while (host.Get(Name)!.ActiveLeases == 0 || ((IChatModel)host.LoadedInstances()[0].Instance).Engine.GetMetrics().ActiveSequences == 0)
        {
            Assert.True(DateTime.UtcNow < deadline, "The request did not start decoding.");
            await Task.Delay(10);
        }

        await Task.Delay(200);   // a few decode steps (split passes and streamed copies) are in flight
    }
}
