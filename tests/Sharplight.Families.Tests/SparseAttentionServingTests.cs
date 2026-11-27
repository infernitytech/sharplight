using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// DeepSeek sparse attention behind the continuous-batching engine: tiny DSA models whose indexer budget (8) is far below the context, so the
/// indexer key cache and the per-row selection go through the multi-sequence form (rows, runs and pages). Greedy output of prompts of different
/// lengths decoded together, prefilled in small chunks, is token-identical to decoding each prompt alone. Hunyuan v4 Flash adds the pooled
/// indexer (pools of 4, the query's own pool always attended) next to sliding-window layers and Engram modules.
/// </summary>
public class SparseAttentionServingTests
{
    private static readonly string Models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    [Theory]
    [InlineData("hf/tiny-random-deepseek-v32-sparse", false)]
    [InlineData("hf/tiny-random-deepseek-v32-sparse", true)]
    [InlineData("hf/tiny-random-hy-v4-sparse", false)]
    [InlineData("hf/tiny-random-hy-v4-sparse", true)]
    [InlineData("hf/tiny-random-hy-v4-flash", false)]
    [InlineData("hf/tiny-random-hy-v4-flash", true)]
    [InlineData("gguf/tiny-random-hy-v4-flash-f32.gguf", false)]
    [InlineData("gguf/tiny-random-hy-v4-flash-f32.gguf", true)]
    public async Task Batched_GreedyOutput_Matches_SingleSequence(string name, bool vulkan)
    {
        string path = Path.Combine(Models, name);
        Assert.SkipUnless(Directory.Exists(path) || File.Exists(path), $"{path} not present");
        using var device = OpenDevice(vulkan);
        using var model = LlmModel.Load(TestHost.Loader, path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 128, MaxBatchTokens = 16 });
        var rng = new Random(3);
        int[][] prompts = [.. new[] { 5, 13, 22, 37 }.Select(n => Enumerable.Range(0, n).Select(_ => rng.Next(10, 1000)).ToArray())];
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
