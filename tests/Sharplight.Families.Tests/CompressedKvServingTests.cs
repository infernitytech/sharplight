using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Compressed-KV attention (DeepSeek-V4) behind the continuous-batching engine: the window pools, compressed caches and indexer selection of the
/// tiny V4 model go through their multi-sequence form (rows, runs and pages). Greedy output of prompts of different lengths, one of them long
/// enough for the 128-position entries, decoded together and prefilled in small chunks, is token-identical to decoding each prompt alone. The
/// tiny V4.1 GGUF adds layers sharing an earlier layer's caches and selection, candidate blocks and the Engram window state; its indexer has
/// near-ties, so its caches stay in f32.
/// </summary>
public class CompressedKvServingTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    [Theory]
    [InlineData("hf/tiny-random-deepseek-v4", false, false)]
    [InlineData("hf/tiny-random-deepseek-v4", true, false)]
    [InlineData("hf/tiny-random-deepseek-v4", false, true)]
    [InlineData("gguf/tiny-random-deepseek-v41.gguf", false, true)]
    [InlineData("gguf/tiny-random-deepseek-v41.gguf", true, true)]
    public async Task Batched_GreedyOutput_Matches_SingleSequence(string name, bool vulkan, bool f32Kv)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }], ContextLength = 256, MaxBatchTokens = 16, KvCacheType = f32Kv ? Sharplight.Tensors.DataType.F32 : null,
        });
        var rng = new Random(5);
        int[][] prompts = [.. new[] { 6, 23, 45, 141 }.Select(n => Enumerable.Range(0, n).Select(_ => rng.Next(10, 1000)).ToArray())];
        var ct = TestContext.Current.CancellationToken;

        var expected = new int[prompts.Length][];
        for (int i = 0; i < prompts.Length; i++)
        {
            expected[i] = (await Generate(model, model.CreateBatchDecoder(slots: 1), [prompts[i]], ct))[0];
        }

        var actual = await Generate(model, model.CreateBatchDecoder(slots: prompts.Length), prompts, ct);
        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    private static async Task<int[][]> Generate(LlmModel model, IBatchDecoder decoder, int[][] prompts, CancellationToken ct)
    {
        using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions());
        var handles = prompts.Select(p => engine.Submit(new GenerationRequest
        {
            PromptTokens = p, MaxTokens = 24, StopAtEog = false, Logprobs = true, Sampling = SamplingParams.GreedyDefault,
        }, ct)).ToList();
        var results = await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct)));
        foreach (var r in results)
        {
            Assert.Null(r.Error);
        }

        return [.. results.Select(r => r.Logprobs.Select(l => l.TokenId).ToArray())];
    }

    private static IComputeDevice OpenDevice(bool vulkan)
    {
        if (!vulkan)
        {
            var cpu = new CpuExecutionProvider();
            return cpu.OpenDevice(cpu.GetDevices()[0]);
        }

        var p = new VulkanProvider();
        Assert.SkipUnless(p.CheckAvailability().IsAvailable, "Vulkan is not available");
        var devs = p.GetDevices();
        var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
        Assert.SkipWhen(dev is null, "no Vulkan device");
        return p.OpenDevice(dev!);
    }
}
