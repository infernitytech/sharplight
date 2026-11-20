using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;

namespace Sharplight.Serving.Tests;

/// <summary>
/// A real small model on a GPU constrained with <see cref="DeviceMemoryLimits"/> (skipped when the file or the provider is missing):
/// without a BAR window, with a tiny one, with a small largest allocation and without page-locked memory it generates the same
/// greedy tokens as unconstrained; on a device too small for it, loading fails with a clear error, or offloads to the CPU when a
/// CPU slot is given. Runs on the Vulkan provider (any device: <c>SHARPLIGHT_VK_DEVICES</c>, e.g. llvmpipe) and on CUDA.
/// </summary>
[Collection("RealModel")]
public class HardwareLimitsRealTests
{
    private static readonly string ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        "models/gguf/SmolLM2-135M/SmolLM2-135M.Q8_0.gguf");
    private const string Prompt = "The capital of France is";
    private const int Tokens = 12;
    private static readonly Dictionary<string, int[]> s_reference = [];

    public static TheoryData<string, string> Cases => new()
    {
        { "vulkan", "no-bar" },
        { "vulkan", "tiny-bar" },
        { "vulkan", "max-allocation" },
        { "cuda", "no-pinned" },
        { "cuda", "tiny-pinned" },
        { "cuda", "max-allocation" },
    };

    private static IExecutionProvider Provider(string name)
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        // CUDA is not among the default providers (services.AddCudaProvider() opts in): the test opens it itself.
        IExecutionProvider? provider = name == Cuda.CudaProvider.ProviderName
            ? new Cuda.CudaProvider()
            : TestHost.Providers.Discover().FirstOrDefault(x => x.Provider.Name == name && x.Availability.IsAvailable)?.Provider;
        Assert.SkipUnless(provider is not null && provider.CheckAvailability().IsAvailable, $"no {name} provider");
        Assert.SkipUnless(provider is not null && provider.GetDevices().Count > 0, $"no {name} device");
        return provider!;
    }

    private static void Apply(string limit, DeviceMemoryLimits limits)
    {
        switch (limit)
        {
            case "no-bar":
                limits.HostVisibleDeviceMemory = 0;
                break;
            case "tiny-bar":
                limits.HostVisibleDeviceMemory = 6L << 20;
                break;
            case "max-allocation":
                // The 30 MiB token embedding no longer fits one allocation: bound to several (Vulkan sparse binding); staging shrinks.
                limits.MaxAllocation = 8L << 20;
                break;
            case "no-pinned":
                limits.PinnedHostMemory = 0;
                break;
            case "tiny-pinned":
                limits.PinnedHostMemory = 3L << 20;
                break;
            default:
                // "none": the unconstrained reference.
                break;
        }
    }

    private static int[] Generate(IExecutionProvider provider, string limit)
    {
        var options = new DeviceOptions();
        Apply(limit, options.MemoryLimits);
        using var device = provider.OpenDevice(provider.GetDevices()[0], options);
        using var model = LlmModel.Load(TestHost.Loader, ModelPath, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 32 });
        using var decoder = model.CreateBatchDecoder(1, 256);
        return Greedy(decoder, model);
    }

    private static int[] Greedy(IBatchDecoder decoder, LlmModel model)
    {
        decoder.Reset(0);
        var input = model.Tokenizer.Encode(Prompt, addSpecial: true, parseSpecial: false);
        var tokens = new List<int>();
        for (int i = 0; i < Tokens; i++)
        {
            var logits = decoder.Step([new SequenceInput(0, input)])[0].Span;
            int best = 0;
            for (int k = 1; k < logits.Length; k++)
            {
                if (logits[k] > logits[best])
                {
                    best = k;
                }
            }

            tokens.Add(best);
            input = [best];
        }

        return [.. tokens];
    }

    private static int[] Reference(IExecutionProvider provider)
    {
        lock (s_reference)
        {
            if (!s_reference.TryGetValue(provider.Name, out var tokens))
            {
                tokens = Generate(provider, "none");
                s_reference[provider.Name] = tokens;
            }

            return tokens;
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ConstrainedDeviceGeneratesTheSameTokens(string providerName, string limit)
    {
        var provider = Provider(providerName);
        Assert.Equal(Reference(provider), Generate(provider, limit));
    }

    [Theory]
    [InlineData("vulkan")]
    [InlineData("cuda")]
    public void TooSmallDeviceFailsClearlyOrOffloads(string providerName)
    {
        var provider = Provider(providerName);
        var options = new DeviceOptions();
        options.MemoryLimits.DeviceMemory = 48L << 20;   // the model's weights alone take 140 MiB
        using (var device = provider.OpenDevice(provider.GetDevices()[0], options))
        {
            Assert.Equal(48L << 20, device.GetMemoryInfo().Total);
            var error = Assert.ThrowsAny<Exception>(() =>
            {
                using var model = LlmModel.Load(TestHost.Loader, ModelPath,
                    new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 32 });
                using var decoder = model.CreateBatchDecoder(1, 256);
            });
            Assert.True(error is OutOfMemoryException or InsufficientMemoryException, $"unexpected {error.GetType().Name}: {error.Message}");
            Console.Error.WriteLine($"[{providerName}] too small: {error.GetType().Name}: {error.Message}");
            Assert.Equal(0, device.GetMemoryInfo().UsedByUs);
        }

        // With a CPU slot the layers that do not fit run on the CPU.
        Assert.Equal(Tokens, GenerateOffloaded(provider).Length);
    }

    private static int[] GenerateOffloaded(IExecutionProvider provider)
    {
        var options = new DeviceOptions();
        options.MemoryLimits.DeviceMemory = 96L << 20;
        using var device = provider.OpenDevice(provider.GetDevices()[0], options);
        var cpuProvider = TestHost.Providers.Discover().First(x => x.Provider.Name == "cpu").Provider;
        using var cpu = cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]);
        using var model = LlmModel.Load(TestHost.Loader, ModelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }, new DeviceSlot { Device = cpu }],
            ContextLength = 256,
            MaxBatchTokens = 32,
        });
        using var decoder = model.CreateBatchDecoder(1, 256);
        Assert.True(device.GetMemoryInfo().UsedByUs <= 96L << 20);
        return Greedy(decoder, model);
    }
}
