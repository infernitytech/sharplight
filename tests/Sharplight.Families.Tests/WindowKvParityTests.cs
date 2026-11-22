using Sharplight.Cpu;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Window-sized KV caches (<see cref="DeviceFeatures.WindowKv"/>) on real model graphs with sliding-window layers (tiny random Cohere2 MoE
/// and Gemma 4 MoE, windows of 8 keys): sequences decoding far past the window give the same tokens as with full-length caches for those
/// layers, also when prompts attach cached prefixes (whose window blocks stayed cached), run concurrently, and fork the context of a
/// sample of the same prompt. The GPU cases run on Vulkan, or on CUDA with <c>SHARPLIGHT_TEST_PROVIDER=cuda</c>.
/// </summary>
public class WindowKvParityTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    public static TheoryData<string, bool> Cases()
    {
        var data = new TheoryData<string, bool>();
        foreach (string path in new[] { "gguf/tiny-random-cohere2-moe-f32.gguf", "gguf/tiny-random-gemma4-moe-f32.gguf", "hf/tiny-random-cohere2-moe" })
        {
            data.Add(path, false);
            data.Add(path, true);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task WindowSizedCaches_DecodeLikeFullCaches(string name, bool gpu)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        var ct = TestContext.Current.CancellationToken;
        var rng = new Random(7);
        int[][] prompts;
        int[][] expected;
        using (var device = OpenDevice(gpu))
        using (var full = Load(path, device, window: false))
        {
            int vocab = Math.Min(1000, full.Tokenizer.VocabSize);
            int[] prefix = [.. Enumerable.Range(0, 40).Select(_ => rng.Next(10, vocab))];
            prompts = [.. new[] { 5, 9, 30, 7, 11 }.Select(n => prefix.Concat(Enumerable.Range(0, n).Select(_ => rng.Next(10, vocab))).ToArray())];
            var decoder = full.CreateBatchDecoder(slots: 2);
            Assert.Equal(0, ((MultiSequenceDecoder)decoder).KvWindow);
            using var engine = new InferenceEngine(decoder, full.Tokenizer, new EngineOptions { PrefixCache = false });
            expected = new int[prompts.Length][];
            for (int i = 0; i < prompts.Length; i++)
            {
                expected[i] = (await Generate(engine, prompts[i], ct)).Tokens;
            }
        }

        using var device2 = OpenDevice(gpu);
        using var model = Load(path, device2, window: true);
        var windowed = model.CreateBatchDecoder(slots: 2);
        Assert.Equal(8, ((MultiSequenceDecoder)windowed).KvWindow);
        using var cached = new InferenceEngine(windowed, model.Tokenizer, new EngineOptions { StepCost = new StepCostModel(fixedMs: 1, perRowMs: 1, copyMs: 0, frozen: true) });
        var hits = new int[3];
        for (int i = 0; i < 3; i++)
        {
            var (tokens, hit) = await Generate(cached, prompts[i], ct);
            Assert.Equal(expected[i], tokens);
            hits[i] = hit;
        }

        Assert.True(hits[1] >= 32 && hits[2] >= 32, $"cached tokens per prompt: {string.Join(", ", hits)}");
        var both = await Task.WhenAll(Generate(cached, prompts[3], ct), Generate(cached, prompts[4], ct));
        Assert.Equal(expected[3], both[0].Tokens);
        Assert.Equal(expected[4], both[1].Tokens);

        // Three samples of one prompt at once: the second and third fork the first's context, window blocks included.
        long forks = cached.GetMetrics().ForkedSequences;
        var samples = await Task.WhenAll(Enumerable.Range(0, 3).Select(_ => Generate(cached, prompts[2], ct)));
        Assert.All(samples, x => Assert.Equal(expected[2], x.Tokens));
        Assert.True(cached.GetMetrics().ForkedSequences - forks >= 1 || samples.All(x => x.Cached > 0), "the samples neither forked nor attached the prompt");
    }

    private static LlmModel Load(string path, IComputeDevice device, bool window)
    {
        var options = new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 16 };
        if (!window)
        {
            options.DisabledFeatures.Add(DeviceFeatures.WindowKv);
        }

        return LlmModel.Load(TestHost.Loader, path, options);
    }

    private static async Task<(int[] Tokens, int Cached)> Generate(InferenceEngine engine, int[] prompt, CancellationToken ct)
    {
        var r = await engine.Submit(new GenerationRequest
        {
            PromptTokens = prompt, MaxTokens = 40, StopAtEog = false, Logprobs = true, Sampling = SamplingParams.GreedyDefault,
        }, ct).CompleteAsync(ct);
        Assert.Null(r.Error);
        return ([.. r.Logprobs.Select(l => l.TokenId)], r.Usage.CachedTokens);
    }

    /// <summary>The CPU, or the test GPU (Vulkan, or CUDA with <c>SHARPLIGHT_TEST_PROVIDER=cuda</c>; skips without one).</summary>
    internal static IComputeDevice OpenDevice(bool gpu)
    {
        if (!gpu)
        {
            var cpu = new CpuExecutionProvider();
            return cpu.OpenDevice(cpu.GetDevices()[0]);
        }

        IExecutionProvider p = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_PROVIDER")?.ToLowerInvariant() == CudaProvider.ProviderName
            ? new CudaProvider()
            : new VulkanProvider();
        Assert.SkipUnless(p.CheckAvailability().IsAvailable, $"{p.Name} is not available");
        var devs = p.GetDevices();
        var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
        Assert.SkipWhen(dev is null, $"no {p.Name} device");
        return p.OpenDevice(dev!);
    }
}
